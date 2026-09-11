"""统一对话入口。

B2 重构后没有"分诊 bypass"：分诊多轮追问由 call_triage_agent 工具内的
langgraph interrupt() 驱动，回合控制权在 Supervisor checkpointer。
本路由只做一件事 —— 看快照里有没有挂起的 interrupt：
  - 有 → 用户这条消息是追问的回答，Command(resume) 恢复图继续跑
  - 无 → 正常作为新消息进 Supervisor
"""

import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.supervisor_agent import get_supervisor_agent
from src.core.deps import UserContext
from src.core.base_schema import ResponseSchema
from src.core.logger import logger
from src.utils.mask import mask_free_text

router = APIRouter(prefix="/api/v1/chat", tags=["智能对话"])


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID（同会话多轮使用相同ID）")
    message: str = Field(..., description="用户消息")
    role: str = Field(default="engineer", description="提问者角色：engineer/business/aftersales/customer")


class ChatResponse(BaseModel):
    reply: str
    session_id: str


def _user_ctx(req: ChatRequest) -> UserContext:
    return UserContext(user_id=req.user_id, session_id=req.session_id, role=req.role,
                       business_line=None, owner_domain_id=None)


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


@router.post("", response_model=ResponseSchema[ChatResponse])
async def chat(req: ChatRequest):
    """
    统一对话接口。

    路由依据（单一控制面 = Supervisor checkpointer）：
    - 分诊工具挂起等待追问回答（interrupt）→ Command(resume) 恢复
    - 否则 → 作为新消息进 Supervisor 决策路由
    """
    try:
        agent = await get_supervisor_agent()
        config = {"configurable": {"thread_id": f"{req.user_id}:{req.session_id}"}}
        ctx = _user_ctx(req)

        # 原始敏感数据不进 LLM：自由文本先打码（VIN/手机号）
        message = mask_free_text(req.message)

        if await _has_pending_triage(agent, config):
            logger.info(f"[CHAT] triage resume user={req.user_id} session={req.session_id}")
            result = await agent.ainvoke(Command(resume=message), config=config, context=ctx)
        else:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config, context=ctx,
            )

        reply = _reply_from_result(result)
        return ResponseSchema(data=ChatResponse(reply=reply, session_id=req.session_id))

    except Exception as e:
        logger.exception("chat 接口异常")
        raise HTTPException(status_code=500, detail=str(e))


# ── SSE 流式接口 ────────────────────────────────────────────────────────────

def _sse_frame(data: dict) -> str:
    """构建 SSE 帧：data: {json}\n\n"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口（Server-Sent Events）。

    路由逻辑与同步接口一致（checkpointer 快照决定 resume 或新消息）。

    SSE 帧格式：
        data: {"type":"token","content":"..."}
        data: {"type":"done","session_id":"..."}
        data: {"type":"error","message":"...","trace_id":"..."}
    """
    async def event_generator():
        trace_id = str(uuid.uuid4())[:12]
        try:
            agent = await get_supervisor_agent()
            config = {"configurable": {"thread_id": f"{req.user_id}:{req.session_id}"}}
            ctx = _user_ctx(req)
            message = mask_free_text(req.message)

            if await _has_pending_triage(agent, config):
                logger.info(f"[CHAT/STREAM] triage resume user={req.user_id} session={req.session_id}")
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
