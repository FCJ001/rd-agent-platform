"""统一对话入口。

B2 重构后没有"分诊 bypass"：分诊多轮追问由 call_triage_agent 工具内的
langgraph interrupt() 驱动，回合控制权在 Supervisor checkpointer。
本路由看快照里有没有挂起的 interrupt：
  - 有 → 用户这条消息是追问的回答，Command(resume) 恢复图继续跑
  - 有且是退出指令 → resume 进图，工具清 store 退出诊断
  - 有且判定为离题请求 → **不碰图**：旁路助手就地回答 + 回放追问
    （docs/design-side-assistant-bypass.md，SIDE_ASSISTANT_ENABLED 开关）
  - 无 → 正常作为新消息进 Supervisor

★ 身份只来自服务端认证（get_current_user），请求体里的 user_id/role
仅做一致性校验，绝不作为身份来源 —— 否则任何人可冒充任意用户/角色。
"""

import json
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.supervisor_agent import get_supervisor_agent
from src.agents.triage.gate import triage_gate
from src.agents.triage.lock import CHAT_LOCK_PREFIX, session_lock
from src.agents.triage.offtopic import INTENT_OTHER, classify_offtopic
from src.agents.sub_agents import (
    SWITCH_EXIT_TOKEN, entry_tool_for, is_task_exit, is_task_switch,
)
from src.agents.workers.side_assistant import answer_offtopic
from src.core.deps import UserContext, get_current_user
from src.core.base_schema import ResponseSchema
from src.core.config import get_settings
from src.core.exceptions import (
    ERR_CONVERSATION_BUSY, ERR_PERMISSION_DENIED, ERR_SYSTEM_BUSY,
    BizException, ConversationBusyError, TriageSystemBusyError,
)
from src.core.logger import logger
from src.core.rate_limit import rate_limit
from src.utils.mask import mask_free_text

router = APIRouter(prefix="/api/v1/chat", tags=["智能对话"])

settings = get_settings()

# 同步与流式共用一个桶：预算按用户算，不按端点算
_chat_rate_limit = rate_limit("chat", limit=30, window_seconds=60)


class ChatRequest(BaseModel):
    session_id: str = Field(..., max_length=100, description="会话ID（同会话多轮使用相同ID）")
    # 长度上限：进 LLM 的文本无上限等于把成本 DoS 的面直接敞开
    message: str = Field(..., max_length=8000, description="用户消息")
    # 兼容旧前端的冗余字段：服务端不作为身份来源，只校验一致性
    user_id: str | None = Field(None, description="已废弃：身份以认证为准，不匹配时拒绝")
    role: str | None = Field(None, description="已废弃：角色以认证为准")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    # 本回合结束时是否挂起分诊追问（前端据此显示「诊断进行中」横幅/提示语）
    triage_pending: bool = False


def _authed_ctx(user: UserContext, req: ChatRequest) -> UserContext:
    """认证身份 + 请求体会话号 → Agent 上下文。身份维度全部来自认证。"""
    if req.user_id and req.user_id != user.user_id:
        raise BizException(
            f"请求体 user_id={req.user_id} 与认证身份 {user.user_id} 不一致",
            ERR_PERMISSION_DENIED,
        )
    return UserContext(
        user_id=user.user_id,
        session_id=req.session_id,
        role=user.role,
        business_line=user.business_line,
        owner_domain_id=user.owner_domain_id,
        real_name=user.real_name,
    )


async def _get_chat_agent():
    """chat 入口的 agent：P1 开关闭合时用编排图（单图化），否则旧 supervisor。

    两者对外接口一致（ainvoke/astream/aget_state + context），chat 层的
    路由/旁路/忙回放逻辑零感知。"""
    if settings.SUBGRAPH_TRIAGE_ENABLED:
        from src.agents.orchestrator import get_orchestrator_agent
        return await get_orchestrator_agent()
    return await get_supervisor_agent()


def _needs_turn_gate() -> bool:
    """编排图模式下，分诊的模型容量闸门由 chat 层承担。

    旧路径的闸门在 call_triage_agent 工具内（每诊断轮一次）；编排图下
    追问轮 = resume 回合，工具层不存在了，闸门挪到这里。
    只 gate resume 回合（确定是分诊轮）；新消息回合是否进诊断要跑完
    supervisor 路由才知道，ungated —— 代价是每场诊断的第一轮不占闸门，
    上限由回合锁 + 用户级限流兜住（容量口径收紧而非放开）。
    """
    return settings.SUBGRAPH_TRIAGE_ENABLED


def _pending_payload_from_snapshot(snapshot) -> dict:
    """从快照对象提取挂起任务的 interrupt 载荷（没有则空 dict）。

    载荷形状契约（子智能体注册表 sub_agents.py 锚定）：
    {"type": "<agent>_followup", "round": n, "question": str}
    type 用于查退出词表/旁路摘除，question 用于回放与渲染。
    """
    for task in snapshot.tasks:
        for intr in task.interrupts:
            payload = intr.value or {}
            if payload.get("question"):
                return payload
    return {}


def _question_from_snapshot(snapshot) -> str:
    """从快照对象提取挂起的追问文本（没有则空串）。"""
    return _pending_payload_from_snapshot(snapshot).get("question", "")


def _compose_bypass_reply(answer: str, question: str) -> str:
    """旁路回复编排：回答 + 分隔线 + 回放挂起追问（bypass 设计 §3.4）。"""
    return (
        f"{answer}\n\n---\n\n"
        f"我们接着刚才的诊断 —— {question}\n\n"
        f"（想结束诊断去办别的事，回复「切换」，将自动办理你刚才的问题；"
        f"仅退出不办事，回复「退出诊断」）"
    )


def _compose_bypass_failure(question: str) -> str:
    """旁路助手失败的兜底：★ 不回落 resume（离题文本进分诊会白烧一轮且
    污染现象集），回放追问 + 失败提示，状态零变动。"""
    return (
        "刚才那条消息没能处理成功，可以先回答下面的追问，或重发一次。\n\n"
        f"---\n\n我们接着刚才的诊断 —— {question}\n\n"
        f"（想结束诊断去办别的事，回复「切换」，将自动办理你刚才的问题；"
        f"仅退出不办事，回复「退出诊断」）"
    )


# ── 任务切换暂存（确认式换台）──────────────────────────────────────────
# 旁路回合把离题消息暂存（脱敏后文本），用户确认「切换」后自动作为新输入
# 续发 —— 离题消息没进主图历史（旁路不碰图），不暂存切换后系统不知道
# 用户要办什么。短命 key：TTL 5 分钟 + 消费即删，不做任何诊断快照。

SWITCH_STASH_TTL_SECONDS = 300


async def _stash_switch_msg(thread_id: str, message: str) -> None:
    """暂存脱敏后的离题消息。失败静默（丢暂存只是切换退化为提示重发）。"""
    try:
        from src.infra.redis_cache import get_redis_client
        client = await get_redis_client()
        await client.set(f"switch_stash:{thread_id}", message, ex=SWITCH_STASH_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"[CHAT] 切换暂存失败（切换将退化为提示重发）thread={thread_id}: {e}")


async def _pop_switch_msg(thread_id: str) -> str | None:
    """取出并删除暂存。任何异常按无暂存处理（走兜底文案）。"""
    try:
        from src.infra.redis_cache import get_redis_client
        client = await get_redis_client()
        key = f"switch_stash:{thread_id}"
        msg = await client.get(key)
        if msg is not None:
            await client.delete(key)
        return msg if isinstance(msg, str) else None
    except Exception:
        return None


async def _route_turn(agent, config: dict, message: str, ctx: UserContext):
    """回合路由（同步/SSE 共用）：快照只取一次，判定发生在任何图调用之前。

    返回 (action, bypass_reply)：
      ("new", None)    无挂起 → 作为新消息进 Supervisor
      ("resume", None) 挂起且是追问回答/退出指令 → Command(resume) 进图
      ("bypass", text) 挂起且判定离题且开关开 → 旁路回答 + 回放追问，不碰图

    顺序即语义（bypass 设计 §3.1 分流表）：
      快照 → 退出词（精确匹配，零成本）→ 离题判定（正则门 + 小模型）→
      开关（影子模式判定照跑只记日志，不改路由）。
    ★ 判定必须在图调用之前（langgraph 契约 [E]：工具消费 resume 值之后再
      抛异常会导致值错位），所以本函数只读快照、绝不 ainvoke。
    """
    snapshot = await agent.aget_state(config)
    payload = _pending_payload_from_snapshot(snapshot)
    question = payload.get("question", "")
    if not question:
        return "new", None
    # 退出判定按挂起任务的 type 查注册表（未登记类型一律不是退出，fail-safe）
    if is_task_exit(payload.get("type", ""), message):
        return "resume", None
    # ★ 切换拦截必须在离题判定之前：「切换」不命中离题正则，漏拦截会被
    #   当成追问回答喂进分诊。精确匹配、只由用户显式关键词触发。
    if is_task_switch(message):
        return "switch", None

    # 影子模式：classify 照跑（[SIDE] 日志是误判率统计的数据源），
    # 路由是否改变由 SIDE_ASSISTANT_ENABLED 决定
    try:
        verdict = await classify_offtopic(question, message)
    except Exception as e:
        # classify 自身已兜底到 answer；这里再防一层 —— 判定器崩了
        # 最坏回到旁路上线前的行为，绝不能挡正常对话
        logger.warning(f"[CHAT] 离题判定异常，按回答处理: {type(e).__name__} {e}")
        verdict = "answer"
    if verdict != INTENT_OTHER or not settings.SIDE_ASSISTANT_ENABLED:
        return "resume", None

    try:
        # 摘除当前活跃子智能体的入口工具：旁路线程不得另起同种流程
        # （与挂起任务抢状态的老问题，对每个子智能体都成立）
        exclude = entry_tool_for(payload.get("type", ""))
        answer = await answer_offtopic(message, question, ctx, exclude_tool=exclude)
        # 暂存脱敏后的离题消息（入口已 mask）：确认「切换」后自动续发
        await _stash_switch_msg(config["configurable"]["thread_id"], message)
        return "bypass", _compose_bypass_reply(answer, question)
    except Exception as e:
        # 旁路助手失败：不回落 resume，回放追问 + 失败提示（§3.4）
        logger.warning(
            f"[CHAT] 旁路助手失败，回放追问 session={ctx.session_id}: "
            f"{type(e).__name__} {e}"
        )
        return "bypass", _compose_bypass_failure(question)


def _reply_from_result(result: dict) -> str:
    """从图运行结果提取给用户的回复。

    ★ 分诊追问轮图会暂停在工具内：最后一条消息是模型的 tool_call（content
    为空），追问文本只在 interrupt payload 里，必须从这里取。
    """
    interrupts = result.get("__interrupt__")
    if interrupts:
        question = (interrupts[0].value or {}).get("question", "")
        return question or "请继续描述故障细节。"
    return result["messages"][-1].content


async def _pending_question(agent, config: dict) -> str:
    """取快照里挂起的追问文本（没有则空串）。"""
    try:
        snapshot = await agent.aget_state(config)
    except Exception:
        return ""
    return _question_from_snapshot(snapshot)


async def _on_conversation_busy(agent, config: dict, session_id: str, exc: Exception):
    """分诊被拒（会话忙/系统容量满）的兜底：尽力把挂起的追问还给用户，而不是甩一个错误。

    ★ 为什么要捞追问而不是直接报错：call_triage_agent 在追问轮被拒时会把
      异常原样抛出图外（否则工具正常返回会消费掉挂起的 interrupt，会话
      状态机错位 —— 见 worker_tools._triage_busy_fallback）。checkpoint
      因此还停在追问点，这里把问题文本从快照里捞出来回放，用户重答即可。
      同一会话并发是**正常用户行为**（连点两次发送、多标签页），报一句
      「请稍后再试」等于把问题弄丢，用户不知道该答什么。
    """
    question = await _pending_question(agent, config)
    if question:
        logger.info(
            f"[CHAT] 忙被拒，回退为回放挂起追问 session={session_id} exc={type(exc).__name__}"
        )
        return ResponseSchema(data=ChatResponse(
            reply=question, session_id=session_id, triage_pending=True,
        ))

    if isinstance(exc, TriageSystemBusyError):
        logger.warning(f"[CHAT] 系统容量满且无挂起追问，按系统忙拒绝 session={session_id}")
        raise BizException("当前诊断请求较多，请稍后再试。", ERR_SYSTEM_BUSY)

    logger.warning(f"[CHAT] 会话忙且无挂起追问，拒绝并发 session={session_id}")
    raise BizException(
        "上一次请求还在处理中，请稍等几秒再发送。", ERR_CONVERSATION_BUSY,
    )


def _chat_turn_lock(thread_id: str):
    """chat 回合锁：保护本会话在 Supervisor checkpointer 里的消息历史。

    ★ 这是控制面（谁有权推进状态机）的互斥，与分诊数据锁（triage_state）
      覆盖的是不同资源，所以是两把前缀不同的独立锁，而不是同一把：
      本回合内会调用 call_triage_agent，工具自己会去拿 triage_lock ——
      合并成一把就是同一请求对自己加锁，不可重入会直接自锁。

    为什么必须有：两条并发请求（用户连点发送、多标签页、前端重试）会各自
    读历史、各自写 checkpoint，后写覆盖先写导致丢消息；而且两边各自判断
    「有没有挂起追问」形成 TOCTOU，可能同时对同一 thread 发 Command(resume)。

    ★ 这不是「理论上可能」：langgraph-checkpoint-redis 的 aput 里，更新
      「最新 checkpoint 指针」是一个裸 SET（aio.py 里
      `await self._redis.set(latest_pointer_key, checkpoint_key)`），
      checkpoint blob 的写入走 `pipeline(transaction=False)` —— 全链路没有
      CAS / WATCH / 版本比对。两条并发 run 各写各的 blob，最后写指针的那个赢，
      另一条的状态直接成为不可达的孤儿（表现就是丢消息）。所以这把锁是承重的，
      不是"再保险一道"。

    TTL 取 CHAT_LOCK_TIMEOUT_SECONDS（比会话锁小得多）：持有时长是本回合
    全部工具调用的累加（可以好几分钟），靠续租兜正常慢任务，TTL 只负责
    「进程崩溃后多久自动解锁」。
    """
    return session_lock(
        thread_id,
        prefix=CHAT_LOCK_PREFIX,
        ttl_seconds=settings.CHAT_LOCK_TIMEOUT_SECONDS,
    )


# ── 回合级结构化 trace（v1）──────────────────────────────────────────────
# production-plan §6.4 的最小落地：每回合一条 [TRACE] JSON，
# 覆盖「路由决策 + 延迟 + token 记账」。token 是 thread 累计口径
# （messages 带全量历史，逐条 usage_metadata 求和会跨回合重复计），
# 相邻两条 trace 相减即得单回合成本；后续接 Langfuse/OTel 时换 sink 即可。

def _thread_token_usage(result: dict | None) -> dict:
    """按 AIMessage.usage_metadata 求和（thread 累计口径）。"""
    if not result:
        return {}
    prompt = completion = 0
    for m in result.get("messages", []):
        um = getattr(m, "usage_metadata", None)
        if um:
            prompt += um.get("input_tokens") or 0
            completion += um.get("output_tokens") or 0
    if prompt or completion:
        return {"thread_prompt_tokens": prompt, "thread_completion_tokens": completion}
    return {}


def _log_turn_trace(
    trace_id: str, ctx: UserContext, action: str, started: float,
    reply: str = "", result: dict | None = None, pending: bool | None = None,
) -> None:
    """回合结束打一条结构化 trace。失败静默（trace 不能挡正常回复）。

    pending：本回合结束时是否挂起分诊追问（同步路径从 result 推导，
    SSE 路径由调用方显式传）。result 只用于 token 记账，可为 None。
    """
    try:
        if pending is None:
            pending = bool(result and result.get("__interrupt__"))
        trace = {
            "trace_id": trace_id,
            "user_id": ctx.user_id,
            "session_id": ctx.session_id,
            "role": ctx.role,
            "route": action,   # new | resume | bypass
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "reply_chars": len(reply or ""),
            "pending_followup": pending,
            **_thread_token_usage(result),
        }
        logger.info("[TRACE] " + json.dumps(trace, ensure_ascii=False))
    except Exception:
        pass


@router.post("", response_model=ResponseSchema[ChatResponse])
async def chat(
    req: ChatRequest,
    user: UserContext = Depends(get_current_user),
    _rl: None = Depends(_chat_rate_limit),
):
    """
    统一对话接口。

    路由依据（单一控制面 = Supervisor checkpointer）：
    - 分诊工具挂起等待追问回答（interrupt）→ Command(resume) 恢复
    - 挂起且判定离题且开关开 → 旁路回答 + 回放追问（不碰图）
    - 否则 → 作为新消息进 Supervisor 决策路由

    异常统一走全局处理器：500 只回兜底文案，str(e) 的内部细节绝不外泄。
    """
    import time

    trace_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    agent = await _get_chat_agent()
    ctx = _authed_ctx(user, req)
    # thread_id 以认证身份开头：会话按用户物理隔离，猜 ID 劫持他人会话无效
    thread_id = f"{ctx.user_id}:{req.session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # 原始敏感数据不进 LLM：自由文本先打码（VIN/手机号）
    message = mask_free_text(req.message)

    try:
        async with _chat_turn_lock(thread_id):
            action, bypass_reply = await _route_turn(agent, config, message, ctx)
            if action == "bypass":
                # 离题请求：旁路回答 + 回放追问，图一次都没碰（状态零变动）
                logger.info(
                    f"[CHAT] side bypass user={ctx.user_id} session={req.session_id}"
                )
                _log_turn_trace(trace_id, ctx, "bypass", started, reply=bypass_reply, pending=True)
                return ResponseSchema(data=ChatResponse(
                    reply=bypass_reply, session_id=req.session_id, triage_pending=True,
                ))
            if action == "switch":
                # 确认式换台：① 规范化退出令牌终结当前流程（复用既有退出分支，
                # 旧/新路径都认）② 暂存的离题消息自动续发（无暂存则提示重发）
                logger.info(f"[CHAT] task switch user={ctx.user_id} session={req.session_id}")
                if _needs_turn_gate():
                    async with triage_gate():
                        result = await agent.ainvoke(
                            Command(resume=SWITCH_EXIT_TOKEN), config=config, context=ctx,
                        )
                else:
                    result = await agent.ainvoke(
                        Command(resume=SWITCH_EXIT_TOKEN), config=config, context=ctx,
                    )
                stashed = await _pop_switch_msg(thread_id)
                if stashed:
                    result2 = await agent.ainvoke(
                        {"messages": [{"role": "user", "content": stashed}]},
                        config=config, context=ctx,
                    )
                    reply = _reply_from_result(result) + "\n\n---\n\n" + _reply_from_result(result2)
                    pending = bool(result2.get("__interrupt__"))
                else:
                    reply = _reply_from_result(result) + "\n\n请把你的新需求直接发送给我。"
                    pending = False
                _log_turn_trace(trace_id, ctx, "switch", started, reply=reply, pending=pending)
                return ResponseSchema(data=ChatResponse(
                    reply=reply, session_id=req.session_id, triage_pending=pending,
                ))
            if action == "resume":
                logger.info(f"[CHAT] triage resume user={ctx.user_id} session={req.session_id}")
                payload = Command(resume=message)
            else:
                payload = {"messages": [{"role": "user", "content": message}]}
            if action == "resume" and _needs_turn_gate():
                # 编排图模式：闸门在图外（拒绝发生在任何图执行之前，
                # 契约 [E] —— checkpoint 原地，忙回放兜底）
                async with triage_gate():
                    result = await agent.ainvoke(payload, config=config, context=ctx)
            else:
                result = await agent.ainvoke(payload, config=config, context=ctx)
    except (ConversationBusyError, TriageSystemBusyError) as e:
        return await _on_conversation_busy(agent, config, req.session_id, e)

    reply = _reply_from_result(result)
    _log_turn_trace(trace_id, ctx, action, started, reply=reply, result=result)
    return ResponseSchema(data=ChatResponse(
        reply=reply, session_id=req.session_id,
        triage_pending=bool(result.get("__interrupt__")),
    ))


# ── SSE 流式接口 ────────────────────────────────────────────────────────────

def _sse_frame(data: dict) -> str:
    """构建 SSE 帧：data: {json}\n\n"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    user: UserContext = Depends(get_current_user),
    _rl: None = Depends(_chat_rate_limit),
):
    """
    流式对话接口（Server-Sent Events）。

    路由逻辑与同步接口一致（checkpointer 快照决定 resume 或新消息）。

    SSE 帧格式：
        data: {"type":"token","content":"..."}
        data: {"type":"done","session_id":"..."}
        data: {"type":"error","message":"...","trace_id":"..."}
    """
    ctx = _authed_ctx(user, req)

    async def event_generator():
        trace_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        triage_pending = False   # 本回合结束时是否挂起追问（done 帧带给前端）
        try:
            agent = await _get_chat_agent()
            thread_id = f"{ctx.user_id}:{req.session_id}"
            config = {"configurable": {"thread_id": thread_id}}
            message = mask_free_text(req.message)

            # 回合锁与同步接口同一命名空间：流式/同步两种调用在同一个会话上
            # 也互相排斥，而不只是「同步 vs 同步」。
            # ★ 锁覆盖到 SSE 帧推送结束为止：生成器把帧 yield 给客户端时回合
            #   并未结束，checkpoint 仍在被写。客户端断开时 Starlette 取消
            #   该任务，CancelledError 落在 yield 处 → 这里展开释放锁。
            async with _chat_turn_lock(thread_id):
                action, bypass_reply = await _route_turn(agent, config, message, ctx)
                if action == "bypass":
                    # 离题请求：跳过图 token 流，整段文本作为一帧下发（前端零协议改动）
                    logger.info(
                        f"[CHAT/STREAM] side bypass user={ctx.user_id} session={req.session_id}"
                    )
                    _log_turn_trace(trace_id, ctx, "bypass", started, reply=bypass_reply, pending=True)
                    triage_pending = True   # 旁路不碰图，挂起原样保持
                    yield _sse_frame({"type": "token", "content": bypass_reply})
                elif action == "switch":
                    # 换台：退出段 + 续发段串行流式，分隔线拼接
                    logger.info(f"[CHAT/STREAM] task switch user={ctx.user_id} session={req.session_id}")
                    if _needs_turn_gate():
                        async with triage_gate():
                            async for chunk in agent.astream(
                                Command(resume=SWITCH_EXIT_TOKEN), config=config,
                                context=ctx, stream_mode="messages",
                            ):
                                if isinstance(chunk, tuple):
                                    msg_chunk, _ = chunk
                                    if hasattr(msg_chunk, "content") and msg_chunk.content:
                                        yield _sse_frame({"type": "token", "content": msg_chunk.content})
                    else:
                        async for chunk in agent.astream(
                            Command(resume=SWITCH_EXIT_TOKEN), config=config,
                            context=ctx, stream_mode="messages",
                        ):
                            if isinstance(chunk, tuple):
                                msg_chunk, _ = chunk
                                if hasattr(msg_chunk, "content") and msg_chunk.content:
                                    yield _sse_frame({"type": "token", "content": msg_chunk.content})
                    yield _sse_frame({"type": "token", "content": "\n\n---\n\n"})
                    stashed = await _pop_switch_msg(thread_id)
                    if stashed:
                        async for chunk in agent.astream(
                            {"messages": [{"role": "user", "content": stashed}]},
                            config=config, context=ctx, stream_mode="messages",
                        ):
                            if isinstance(chunk, tuple):
                                msg_chunk, _ = chunk
                                if hasattr(msg_chunk, "content") and msg_chunk.content:
                                    yield _sse_frame({"type": "token", "content": msg_chunk.content})
                        snapshot = await agent.aget_state(config)
                        triage_pending = any(t.interrupts for t in snapshot.tasks)
                        for task in snapshot.tasks:
                            for intr in task.interrupts:
                                question = (intr.value or {}).get("question", "")
                                if question:
                                    yield _sse_frame({"type": "token", "content": question})
                    else:
                        yield _sse_frame({"type": "token", "content": "请把你的新需求直接发送给我。"})
                else:
                    if action == "resume":
                        logger.info(f"[CHAT/STREAM] triage resume user={ctx.user_id} session={req.session_id}")
                        payload = Command(resume=message)
                    else:
                        payload = {"messages": [{"role": "user", "content": message}]}

                    if action == "resume" and _needs_turn_gate():
                        async with triage_gate():
                            async for chunk in agent.astream(
                                payload, config=config, context=ctx, stream_mode="messages",
                            ):
                                if isinstance(chunk, tuple):
                                    msg_chunk, _ = chunk
                                    if hasattr(msg_chunk, "content") and msg_chunk.content:
                                        yield _sse_frame({"type": "token", "content": msg_chunk.content})
                    else:
                        async for chunk in agent.astream(
                            payload, config=config, context=ctx, stream_mode="messages",
                        ):
                            if isinstance(chunk, tuple):
                                msg_chunk, _ = chunk
                                if hasattr(msg_chunk, "content") and msg_chunk.content:
                                    yield _sse_frame({"type": "token", "content": msg_chunk.content})

                    # 分诊追问轮图暂停在工具内，问题文本不经过消息流，从快照补推
                    snapshot = await agent.aget_state(config)
                    has_pending = any(t.interrupts for t in snapshot.tasks)
                    for task in snapshot.tasks:
                        for intr in task.interrupts:
                            question = (intr.value or {}).get("question", "")
                            if question:
                                yield _sse_frame({"type": "token", "content": question})
                    _log_turn_trace(
                        trace_id, ctx, action, started,
                        result={"messages": snapshot.values.get("messages", [])},
                        pending=has_pending,
                    )
                    triage_pending = has_pending

            yield _sse_frame({"type": "done", "session_id": req.session_id, "triage_pending": triage_pending})

        except (ConversationBusyError, TriageSystemBusyError) as e:
            # 忙不是错：回放挂起追问（checkpoint 停在追问点，用户重答即可）；
            # 没有挂起追问则回一句可重试的业务提示，绝不能落进下面的
            # 「服务内部异常」文案 —— 那会让用户以为系统挂了。
            # ★ code 字段区分忙类业务错与真故障：前端据此渲染「可重试」
            #   的排队提示而不是错误样式（见 src/static/index.html）。
            logger.warning(
                f"[CHAT/STREAM] 忙被拒 exc={type(e).__name__} trace_id={trace_id}"
            )
            question = await _pending_question(agent, config)
            if question:
                triage_pending = True
                yield _sse_frame({"type": "token", "content": question})
            elif isinstance(e, TriageSystemBusyError):
                yield _sse_frame({
                    "type": "error",
                    "code": ERR_SYSTEM_BUSY,
                    "message": "当前诊断请求较多，请稍后再试。",
                    "trace_id": trace_id,
                })
            else:
                yield _sse_frame({
                    "type": "error",
                    "code": ERR_CONVERSATION_BUSY,
                    "message": "上一次请求还在处理中，请稍等几秒再发送。",
                    "trace_id": trace_id,
                })
            yield _sse_frame({"type": "done", "session_id": req.session_id, "triage_pending": triage_pending})

        except Exception:
            logger.exception(f"[CHAT/STREAM] error trace_id={trace_id}")
            yield _sse_frame({
                "type": "error",
                "code": ERR_INTERNAL,
                "message": "服务内部异常，请稍后重试",
                "trace_id": trace_id,
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
