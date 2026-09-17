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
from src.core.deps import UserContext, get_current_user
from src.core.base_schema import ResponseSchema
from src.core.exceptions import (
    ERR_CONVERSATION_BUSY, ERR_PERMISSION_DENIED, BizException, ConversationBusyError,
)
from src.core.logger import logger
from src.core.rate_limit import rate_limit
from src.utils.mask import mask_free_text

router = APIRouter(prefix="/api/v1/chat", tags=["智能对话"])

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


async def _on_conversation_busy(agent, config: dict, session_id: str):
    """并发被拒时的兜底：尽力把挂起的追问还给用户，而不是甩一个错误。

    ★ 为什么要捞追问而不是直接报错：同一会话并发是**正常用户行为**
      （连点两次发送、多标签页）。如果恰好赶上分诊追问轮，用户等的是
      「黑屏时音响还有声音吗」这个问题 —— 报一句「请稍后再试」等于
      把问题弄丢了，用户不知道该答什么。
    """
    question = await _pending_question(agent, config)
    if question:
        logger.info(f"[CHAT] 会话忙，回退为回放挂起追问 session={session_id}")
        return ResponseSchema(data=ChatResponse(reply=question, session_id=session_id))

    logger.warning(f"[CHAT] 会话忙且无挂起追问，拒绝并发 session={session_id}")
    raise BizException(
        "上一次请求还在处理中，请稍等几秒再发送。", ERR_CONVERSATION_BUSY,
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
    config = {"configurable": {"thread_id": f"{ctx.user_id}:{req.session_id}"}}

    # 原始敏感数据不进 LLM：自由文本先打码（VIN/手机号）
    message = mask_free_text(req.message)

    try:
        if await _has_pending_triage(agent, config):
            logger.info(f"[CHAT] triage resume user={ctx.user_id} session={req.session_id}")
            result = await agent.ainvoke(Command(resume=message), config=config, context=ctx)
        else:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config, context=ctx,
            )
    except ConversationBusyError:
        return await _on_conversation_busy(agent, config, req.session_id)

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
            config = {"configurable": {"thread_id": f"{ctx.user_id}:{req.session_id}"}}
            message = mask_free_text(req.message)

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

        except Exception:
            logger.exception(f"[CHAT/STREAM] error trace_id={trace_id}")
            yield _sse_frame({
                "type": "error",
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
