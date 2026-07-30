"""分诊 API Router。POST /api/v1/triage"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.agents.workers.triage_agent import get_triage_agent
from src.core.base_schema import ResponseSchema
from src.core.logger import logger
from src.infra.redis_cache import get_checkpointer_redis

router = APIRouter(prefix="/api/v1/triage", tags=["分诊诊断"])


# ── 请求体 ──
class TriageRequest(BaseModel):
    raw_input: str = Field(..., description="故障描述（口语），如'车机偶尔黑屏'")
    session_id: str | None = Field(None, description="继续追问时传入上次返回的 session_id")
    issue_id: int | None = Field(None, description="关联问题单 ID")


# ── 响应体 ──
class CandidateCauseOut(BaseModel):
    code: str
    name: str
    domain: str
    confidence: float
    base_confidence: float
    matched_phenomena: list[str]
    all_phenomena: list[str]
    fix_way: str
    fix_duration: str
    verify_items: str
    is_core_match: bool
    dtc_matched: list[str]


class TriageResult(BaseModel):
    session_id: str
    status: str = Field(..., description="converged | asking | max_turns")
    round: int
    normalized_phenomena: list[str]
    candidate_causes: list[CandidateCauseOut]
    confidence: float
    follow_up_questions: list[str]
    diagnostic_summary: str


@router.post("", response_model=ResponseSchema[TriageResult])
async def triage(req: TriageRequest):
    """
    分诊诊断接口。向后兼容：支持多轮追问（通过 Redis 持久化会话状态）。

    输入一句故障描述（如"车机偶尔黑屏"），返回：
    - 规范化后的现象名
    - 候选根因列表（按置信度降序）
    - 如置信度不足，返回追问问题，前端可继续调用本接口传入 session_id 继续对话
    """
    logger.info(f"[TRIAGE] session={req.session_id or 'new'} input={req.raw_input[:80]}")

    # 从 Redis 恢复上一轮状态（如有）
    existing_state = None
    if req.session_id:
        redis = get_checkpointer_redis()
        state_key = f"triage_state:{req.session_id}"
        raw = await redis.get(state_key)
        if raw:
            import json
            existing_state = json.loads(raw)

    agent = get_triage_agent()
    result = await agent.diagnose(
        raw_input=req.raw_input,
        session_id=req.session_id,
        issue_id=req.issue_id,
        existing_state=existing_state,
    )

    # 如果未收敛，将状态写入 Redis 供下一轮使用
    if result["status"] == "asking" and result["_state"]:
        redis = get_checkpointer_redis()
        state_key = f"triage_state:{result['session_id']}"
        import json
        await redis.set(state_key, json.dumps(result["_state"]), ex=3600)

    triage_result = TriageResult(
        session_id=result["session_id"],
        status=result["status"],
        round=result["round"],
        normalized_phenomena=result["normalized_phenomena"],
        candidate_causes=[CandidateCauseOut(**c) for c in result["candidate_causes"]],
        confidence=result["confidence"],
        follow_up_questions=result["follow_up_questions"],
        diagnostic_summary=result["diagnostic_summary"],
    )

    logger.info(
        f"[TRIAGE] session={triage_result.session_id} "
        f"status={triage_result.status} round={triage_result.round} "
        f"confidence={triage_result.confidence:.3f} candidates={len(triage_result.candidate_causes)}"
    )

    return ResponseSchema(data=triage_result)


# ── 反馈回写 API ────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="分诊会话 ID")
    adopted: bool = Field(..., description="是否采纳诊断结论")
    comment: str | None = Field(None, description="反馈备注（如不采纳原因）")


class FeedbackResult(BaseModel):
    session_id: str
    updated: bool


@router.post("/feedback", response_model=ResponseSchema[FeedbackResult])
async def submit_feedback(req: FeedbackRequest):
    """
    分诊反馈回写接口。

    人工确认诊断结论是否准确，回写到 ai_triage_results.adopted 字段，
    用于统计分诊准确率、优化知识库。
    """
    import json
    from sqlalchemy import text as sa_text
    from src.infra.db import AsyncSessionLocal

    logger.info(f"[TRIAGE-FB] session={req.session_id} adopted={req.adopted}")

    async with AsyncSessionLocal() as db:
        # 先查当前记录拿到 confirmed_phenomena 和 primary_cause_code
        select_result = await db.execute(
            sa_text(
                "SELECT confirmed_phenomena, primary_cause_code "
                "FROM ai_triage_results WHERE session_id = :session_id"
            ),
            {"session_id": req.session_id},
        )
        row = select_result.fetchone()

        result = await db.execute(
            sa_text(
                "UPDATE ai_triage_results SET adopted = :adopted, feedback_comment = :comment, "
                "updated_at = NOW() WHERE session_id = :session_id"
            ),
            {
                "adopted": req.adopted,
                "comment": req.comment,
                "session_id": req.session_id,
            },
        )
        await db.commit()
        updated = result.rowcount > 0

    # ── 触发诊断结论回流图谱 ──
    if updated and req.adopted and row:
        try:
            confirmed = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or [])
            cause_code = row[1]
            from src.agents.triage.feedback_loop import reinforce_graph_on_adopted
            await reinforce_graph_on_adopted(
                confirmed_phenomena=confirmed,
                primary_cause_code=cause_code,
                session_id=req.session_id,
            )
            logger.info(f"[TRIAGE-FB] 图谱增强已触发 session={req.session_id}")
        except Exception as e:
            logger.warning(f"[TRIAGE-FB] 图谱增强失败: {e}")
    elif updated and not req.adopted and row:
        try:
            confirmed = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or [])
            cause_code = row[1]
            from src.agents.triage.feedback_loop import weaken_graph_on_rejected
            await weaken_graph_on_rejected(
                confirmed_phenomena=confirmed,
                primary_cause_code=cause_code,
            )
            logger.info(f"[TRIAGE-FB] 图谱弱化已触发 session={req.session_id}")
        except Exception as e:
            logger.warning(f"[TRIAGE-FB] 图谱弱化失败: {e}")

    return ResponseSchema(data=FeedbackResult(session_id=req.session_id, updated=updated))
