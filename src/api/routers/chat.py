"""统一对话入口。含分诊 bypass：追问轮次绕过 Supervisor 直连 StateGraph。"""

import json
import time
import traceback
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_community.embeddings import DashScopeEmbeddings
from pydantic import BaseModel, Field

from src.agents.supervisor_agent import get_supervisor_agent
from src.agents.triage.graph import run_triage, TriageDeps
from src.core.deps import UserContext, get_current_user
from src.agents.triage.state import TriageState, TriagePhase
from src.core.base_schema import ResponseSchema
from src.core.config import get_settings
from src.core.logger import logger
from src.infra.redis_cache import get_checkpointer_redis
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.milvus_store import MilvusStore
from src.infra.db import AsyncSessionLocal

router = APIRouter(prefix="/api/v1/chat", tags=["智能对话"])


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID（同会话多轮使用相同ID）")
    message: str = Field(..., description="用户消息")
    role: str = Field(default="engineer", description="提问者角色：engineer/business/aftersales/customer")


class ChatResponse(BaseModel):
    reply: str
    session_id: str


_milvus_store = None


def _get_milvus_store() -> MilvusStore:
    global _milvus_store
    if _milvus_store is None:
        s = get_settings()
        alias = get_milvus_client_alias()
        embedding_model = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=s.DASHSCOPE_API_KEY)
        _milvus_store = MilvusStore(alias=alias, embeddings=embedding_model, dims=1024)
    return _milvus_store


async def _save_diagnosis_memory(state: TriageState, thread_id: str) -> None:
    """诊断收敛后将结论写入长期记忆，供 search_memory 检索。"""
    if not state.candidate_causes:
        return

    top = state.candidate_causes[0]
    user_id = thread_id.split(":")[0]

    phenomena_str = "、".join(state.confirmed_phenomena) if state.confirmed_phenomena else "未知"
    dtc_str = "、".join(state.dtc_codes) if state.dtc_codes else "无"

    content = (
        f"诊断结论：{top.name}({top.code}) 置信度{top.confidence:.0%}，"
        f"现象={phenomena_str}，DTC={dtc_str}。{state.diagnostic_summary}"
    )

    store = _get_milvus_store()
    key = f"diagnosis_{int(time.time())}"
    await store.aput(
        namespace=("users", user_id, "memories"),
        key=key,
        value={"content": content, "timestamp": time.time()},
    )
    logger.info(f"[CHAT] saved diagnosis memory for user={user_id}: {content[:120]}...")


async def _build_triage_deps():
    from src.agents.triage.graph import _get_llm_json, _get_llm_chat

    llm_json = _get_llm_json()
    llm_chat = _get_llm_chat()

    async def _db_factory():
        async with AsyncSessionLocal() as session:
            yield session

    return TriageDeps(llm_json=llm_json, llm_chat=llm_chat, db_session_factory=_db_factory)


async def _run_triage_turn(message: str, thread_id: str, redis, role: str = "customer") -> str:
    """执行一轮分诊对话（绕过 Supervisor，直连 StateGraph）。"""
    active_key = f"triage_active:{thread_id}"
    state_key = f"triage_state:{thread_id}"

    # 从 Redis 恢复上一轮状态
    raw = await redis.get(state_key)
    existing_state = TriageState.model_validate_json(raw) if raw else None

    deps = await _build_triage_deps()
    reply, new_state = await run_triage(
        user_message=message,
        thread_id=thread_id,
        deps=deps,
        existing_state=existing_state,
        viewer_role=role,
    )

    # 收敛：保存长期记忆 → 清除 Redis 标记
    if new_state.phase == TriagePhase.CONCLUDE:
        await _save_diagnosis_memory(new_state, thread_id)
        await redis.delete(active_key, state_key)
        return reply

    # 未收敛：更新 Redis 状态，重置 TTL
    await redis.set(state_key, new_state.model_dump_json(), ex=3600)
    await redis.set(active_key, "1", ex=3600)
    return reply


@router.post("", response_model=ResponseSchema[ChatResponse])
async def chat(req: ChatRequest):
    """
    统一对话接口。

    路由逻辑：
    - 分诊进行中（triage_active flag → Redis）→ 绕过 Supervisor，直连 StateGraph
    - 无活跃分诊 → 走 Supervisor 决策路由
    """
    try:
        redis = get_checkpointer_redis()
        thread_id = f"{req.user_id}:{req.session_id}"
        active_key = f"triage_active:{thread_id}"

        # ── 分诊进行中：绕过 Supervisor ──
        if await redis.exists(active_key):
            logger.info(f"[CHAT] triage bypass user={req.user_id} session={req.session_id}")
            reply = await _run_triage_turn(req.message, thread_id, redis, role=req.role)
            return ResponseSchema(data=ChatResponse(reply=reply, session_id=req.session_id))

        # ── 无活跃分诊：走 Supervisor ──
        agent = await get_supervisor_agent()
        config = {"configurable": {"thread_id": thread_id}}

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            context=UserContext(user_id=req.user_id, session_id=req.session_id, role=req.role,
                               business_line=None, owner_domain_id=None),
        )
        reply = result["messages"][-1].content
        return ResponseSchema(data=ChatResponse(reply=reply, session_id=req.session_id))

    except Exception as e:
        logger.exception("chat 接口异常")
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── SSE 流式接口 ────────────────────────────────────────────────────────────

def _sse_frame(data: dict) -> str:
    """构建 SSE 帧：data: {json}\n\n"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口（Server-Sent Events）。

    路由逻辑与同步接口一致：
    - 分诊进行中 → 绕过 Supervisor，直连 StateGraph（整体推送）
    - 无活跃分诊 → Supervisor 流式推送

    SSE 帧格式：
        data: {"type":"token","content":"..."}
        data: {"type":"done","session_id":"..."}
        data: {"type":"error","message":"...","trace_id":"..."}
    """
    async def event_generator():
        trace_id = str(uuid.uuid4())[:12]
        try:
            redis = get_checkpointer_redis()
            thread_id = f"{req.user_id}:{req.session_id}"
            active_key = f"triage_active:{thread_id}"

            # ── 分诊进行中：绕开 Supervisor，整体推送 ──
            if await redis.exists(active_key):
                logger.info(f"[CHAT/STREAM] triage bypass user={req.user_id} session={req.session_id}")
                reply = await _run_triage_turn(req.message, thread_id, redis, role=req.role)
                yield _sse_frame({"type": "token", "content": reply})

            else:
                # ── 无活跃分诊：Supervisor 流式推送 ──
                agent = await get_supervisor_agent()
                config = {"configurable": {"thread_id": thread_id}}
                ctx = UserContext(user_id=req.user_id, session_id=req.session_id, role=req.role,
                                  business_line=None, owner_domain_id=None)

                async for chunk in agent.astream(
                    {"messages": [{"role": "user", "content": req.message}]},
                    config=config,
                    context=ctx,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, tuple):
                        msg_chunk, _ = chunk
                        if hasattr(msg_chunk, "content") and msg_chunk.content:
                            yield _sse_frame({"type": "token", "content": msg_chunk.content})

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
