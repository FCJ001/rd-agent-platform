"""影响分析 & 报告解读反馈回写 API。

★ 所有权校验（IDOR 防护）：UPDATE 强制 WHERE user_id = 当前认证用户。
  以前只按 session_id 过滤，而 session_id 是裸会话号、不含归属 ——
  任何已认证用户拿到别人的 session_id 就能篡改其反馈记录。
  现在 worker 工具写入时记录 user_id（见 worker_tools.call_impact_agent /
  call_report_agent）；user_id 为 NULL 的存量行不匹配任何用户，
  fail-closed —— 谁也更新不了，需要反馈就重新生成一条。
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.core.base_schema import ResponseSchema
from src.core.deps import UserContext, get_current_user
from src.core.logger import logger

router = APIRouter(prefix="/api/v1", tags=["反馈回写"])


# ── 共享请求体 ──

class FeedbackRequest(BaseModel):
    session_id: str = Field(..., max_length=100, description="会话 ID")
    adopted: bool = Field(..., description="是否采纳分析结论")
    comment: str | None = Field(None, max_length=2000, description="反馈备注（如不采纳原因）")


class FeedbackResult(BaseModel):
    session_id: str
    updated: bool


async def _update_owned_row(table: str, req: FeedbackRequest, user: UserContext) -> bool:
    """按 (session_id, user_id) 双条件回写。返回是否有行被更新。

    ★ user_id 必须进 WHERE：这是「谁创建的记录谁才能反馈」的落点。
      匹配不到（别人的会话 / 存量 NULL 行）一律 updated=False，
      不区分「不存在」和「无权限」，避免向探测者确认记录存在。
    """
    from sqlalchemy import text as sa_text
    from src.infra.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            sa_text(
                f"UPDATE {table} SET adopted = :adopted, feedback_comment = :comment, "
                "updated_at = NOW() WHERE session_id = :session_id AND user_id = :user_id"
            ),
            {
                "adopted": req.adopted,
                "comment": req.comment,
                "session_id": req.session_id,
                "user_id": int(user.user_id),
            },
        )
        await db.commit()
        updated = result.rowcount > 0

    if not updated:
        logger.warning(
            f"[FEEDBACK] 无匹配行 table={table} user={user.user_id} session={req.session_id} "
            f"（他人会话或存量行，已拒绝）"
        )
    else:
        logger.info(f"[FEEDBACK] table={table} user={user.user_id} session={req.session_id} adopted={req.adopted}")
    return updated


# ── 影响分析反馈 ────────────────────────────────────────────────────────────

@router.post("/impact/feedback", response_model=ResponseSchema[FeedbackResult])
async def submit_impact_feedback(req: FeedbackRequest, user: UserContext = Depends(get_current_user)):
    """
    影响分析反馈回写接口。

    人工确认影响分析结论是否准确，回写到 ai_impact_analysis.adopted 字段。
    只能反馈本人会话的结论。
    """
    updated = await _update_owned_row("ai_impact_analysis", req, user)
    return ResponseSchema(data=FeedbackResult(session_id=req.session_id, updated=updated))


# ── 报告解读反馈 ────────────────────────────────────────────────────────────

@router.post("/report/feedback", response_model=ResponseSchema[FeedbackResult])
async def submit_report_feedback(req: FeedbackRequest, user: UserContext = Depends(get_current_user)):
    """
    报告解读反馈回写接口。

    人工确认报告解读结论是否准确，回写到 ai_report_interpretations.adopted 字段。
    只能反馈本人会话的结论。
    """
    updated = await _update_owned_row("ai_report_interpretations", req, user)
    return ResponseSchema(data=FeedbackResult(session_id=req.session_id, updated=updated))
