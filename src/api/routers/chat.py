"""统一对话入口。

B2 重构后没有"分诊 bypass"：分诊多轮追问由 call_triage_agent 工具内的
langgraph interrupt() 驱动，回合控制权在 Supervisor checkpointer。
本路由只做一件事 —— 看快照里有没有挂起的 interrupt：
  - 有 → 用户这条消息是追问的回答，Command(resume) 恢复图继续跑
  - 无 → 正常作为新消息进 Supervisor

★ 身份只来自服务端认证（get_current_user），请求体里的 user_id/role
  仅做一致性校验，绝不作为身份来源 —— 否则任何人可冒充任意用户/角色。
"""

import json
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.supervisor_agent import get_supervisor_agent
from src.agents.triage.lock import CHAT_LOCK_PREFIX, session_lock
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


async def _has_pending_triage(agent, config: dict) -> bool:
    """快照里有挂起的 interrupt = 分诊工具正在等用户的追问回答。"""
    snapshot = await agent.aget_state(config)
    return any(task.interrupts for task in snapshot.tasks)


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
    for task in snapshot.tasks:
        for intr in task.interrupts:
            question = (intr.value or {}).get("question", "")
            if question:
                return question
    return ""


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
        return ResponseSchema(data=ChatResponse(reply=question, session_id=session_id))

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
    - 否则 → 作为新消息进 Supervisor 决策路由

    异常统一走全局处理器：500 只回兜底文案，str(e) 的内部细节绝不外泄。
    """
    agent = await get_supervisor_agent()
    ctx = _authed_ctx(user, req)
    # thread_id 以认证身份开头：会话按用户物理隔离，猜 ID 劫持他人会话无效
    thread_id = f"{ctx.user_id}:{req.session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # 原始敏感数据不进 LLM：自由文本先打码（VIN/手机号）
    message = mask_free_text(req.message)

    try:
        async with _chat_turn_lock(thread_id):
            if await _has_pending_triage(agent, config):
                logger.info(f"[CHAT] triage resume user={ctx.user_id} session={req.session_id}")
                result = await agent.ainvoke(Command(resume=message), config=config, context=ctx)
            else:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config=config, context=ctx,
                )
    except (ConversationBusyError, TriageSystemBusyError) as e:
        return await _on_conversation_busy(agent, config, req.session_id, e)

    reply = _reply_from_result(result)
    return ResponseSchema(data=ChatResponse(reply=reply, session_id=req.session_id))


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
        trace_id = str(uuid.uuid4())[:12]
        try:
            agent = await get_supervisor_agent()
            thread_id = f"{ctx.user_id}:{req.session_id}"
            config = {"configurable": {"thread_id": thread_id}}
            message = mask_free_text(req.message)

            # 回合锁与同步接口同一命名空间：流式/同步两种调用在同一个会话上
            # 也互相排斥，而不只是「同步 vs 同步」。
            # ★ 锁覆盖到 SSE 帧推送结束为止：生成器把帧 yield 给客户端时回合
            #   并未结束，checkpoint 仍在被写。客户端断开时 Starlette 取消
            #   该任务，CancelledError 落在 yield 处 → 这里展开释放锁。
            async with _chat_turn_lock(thread_id):
                if await _has_pending_triage(agent, config):
                    logger.info(f"[CHAT/STREAM] triage resume user={ctx.user_id} session={req.session_id}")
                    payload = Command(resume=message)
                else:
                    payload = {"messages": [{"role": "user", "content": message}]}

                async for chunk in agent.astream(
                    payload, config=config, context=ctx, stream_mode="messages",
                ):
                    if isinstance(chunk, tuple):
                        msg_chunk, _ = chunk
                        if hasattr(msg_chunk, "content") and msg_chunk.content:
                            yield _sse_frame({"type": "token", "content": msg_chunk.content})

                # 分诊追问轮图暂停在工具内，问题文本不经过消息流，从快照补推
                snapshot = await agent.aget_state(config)
                for task in snapshot.tasks:
                    for intr in task.interrupts:
                        question = (intr.value or {}).get("question", "")
                        if question:
                            yield _sse_frame({"type": "token", "content": question})

            yield _sse_frame({"type": "done", "session_id": req.session_id})

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
            yield _sse_frame({"type": "done", "session_id": req.session_id})

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
