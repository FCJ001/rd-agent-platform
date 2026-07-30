"""影响分析 & 报告解读反馈回写 API。"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.base_schema import ResponseSchema
from src.core.logger import logger

router = APIRouter(prefix="/api/v1", tags=["反馈回写"])


# ── 共享请求体 ──

class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    adopted: bool = Field(..., description="是否采纳分析结论")
    comment: str | None = Field(None, description="反馈备注（如不采纳原因）")


class FeedbackResult(BaseModel):
    session_id: str
    updated: bool


# ── 影响分析反馈 ────────────────────────────────────────────────────────────

@router.post("/impact/feedback", response_model=ResponseSchema[FeedbackResult])
async def submit_impact_feedback(req: FeedbackRequest):
    """
    影响分析反馈回写接口。

    人工确认影响分析结论是否准确，回写到 ai_impact_analysis.adopted 字段。
    """
    from sqlalchemy import text as sa_text
    from src.infra.db import AsyncSessionLocal

    logger.info(f"[IMPACT-FB] session={req.session_id} adopted={req.adopted}")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            sa_text(
                "UPDATE ai_impact_analysis SET adopted = :adopted, feedback_comment = :comment, "
                "updated_at = NOW() WHERE session_id = :session_id"
            ),
            {"adopted": req.adopted, "comment": req.comment, "session_id": req.session_id},
        )
        await db.commit()
        updated = result.rowcount > 0

    return ResponseSchema(data=FeedbackResult(session_id=req.session_id, updated=updated))


# ── 报告解读反馈 ────────────────────────────────────────────────────────────

@router.post("/report/feedback", response_model=ResponseSchema[FeedbackResult])
async def submit_report_feedback(req: FeedbackRequest):
    """
    报告解读反馈回写接口。

    人工确认报告解读结论是否准确，回写到 ai_report_interpretations.adopted 字段。
    """
    from sqlalchemy import text as sa_text
    from src.infra.db import AsyncSessionLocal

    logger.info(f"[REPORT-FB] session={req.session_id} adopted={req.adopted}")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            sa_text(
                "UPDATE ai_report_interpretations SET adopted = :adopted, feedback_comment = :comment, "
                "updated_at = NOW() WHERE session_id = :session_id"
            ),
            {"adopted": req.adopted, "comment": req.comment, "session_id": req.session_id},
        )
        await db.commit()
        updated = result.rowcount > 0

    return ResponseSchema(data=FeedbackResult(session_id=req.session_id, updated=updated))
