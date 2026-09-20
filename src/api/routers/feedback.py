"""影响分析 & 报告解读反馈回写 API。

★ 所有权校验（IDOR 防护）：UPDATE 强制 WHERE user_id = 当前认证用户。
  以前只按 session_id 过滤，而 session_id 是裸会话号、不含归属 ——
  任何已认证用户拿到别人的 session_id 就能篡改其反馈记录。
  现在 worker 工具写入时记录 user_id（见 worker_tools.call_impact_agent /
  call_report_agent）；user_id 为 NULL 的存量行不匹配任何用户，
  fail-closed —— 谁也更新不了，需要反馈就重新生成一条。

★ 一次反馈只回写一行：同一 session_id 下本来就会有多行（工具每调一次写一行，
  一次会话里可能调多次）。按 session_id 整批更新时，点一次「采纳」会把该会话
  所有历史结论一起标记，准确率统计直接失真。默认取最新一行（用户刚看到的那条），
  调用方也可以带 record_id 精确指定。
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
    record_id: int | None = Field(
        None,
        description="要回写的记录 id（可选）。不传则作用于本会话最新一条 —— "
                    "同一会话有多条结论时用它避免反馈错行",
    )


class FeedbackResult(BaseModel):
    session_id: str
    updated: bool
    # 实际回写的行 id：调用方后续若要再次反馈同一行，带上它可以避免歧义
    record_id: int | None = None


async def _resolve_target_id(db, table: str, req: FeedbackRequest, user_id: int) -> int | None:
    """定位要回写的唯一一行。返回 None = 无匹配（他人会话 / 存量 NULL 行）。

    ★ 必须带 user_id：这是「谁创建的记录谁才能反馈」的落点。
      匹配不到一律 None，不区分「不存在」和「无权限」，避免向探测者确认记录存在。
    """
    from sqlalchemy import text as sa_text

    if req.record_id is not None:
        sql = (
            f"SELECT id FROM {table} WHERE id = :record_id "
            "AND session_id = :session_id AND user_id = :user_id"
        )
        params = {"record_id": req.record_id, "session_id": req.session_id, "user_id": user_id}
    else:
        sql = (
            f"SELECT id FROM {table} WHERE session_id = :session_id AND user_id = :user_id "
            "ORDER BY id DESC LIMIT 1"
        )
        params = {"session_id": req.session_id, "user_id": user_id}
    row = (await db.execute(sa_text(sql), params)).fetchone()
    return int(row[0]) if row else None


async def _update_owned_row(table: str, req: FeedbackRequest, user: UserContext) -> int | None:
    """回写本人会话的**一行**。返回被更新的行 id，None = 未更新。"""
    from sqlalchemy import text as sa_text
    from src.infra.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        target_id = await _resolve_target_id(db, table, req, int(user.user_id))
        if target_id is None:
            logger.warning(
                f"[FEEDBACK] 无匹配行 table={table} user={user.user_id} session={req.session_id} "
                f"record={req.record_id}（他人会话或存量行，已拒绝）"
            )
            return None
        # 更新条件只用主键：目标行已经过 (session_id, user_id) 校验
        await db.execute(
            sa_text(
                f"UPDATE {table} SET adopted = :adopted, feedback_comment = :comment, "
                "updated_at = NOW() WHERE id = :target_id"
            ),
            {"adopted": req.adopted, "comment": req.comment, "target_id": target_id},
        )
        await db.commit()

    logger.info(
        f"[FEEDBACK] table={table} user={user.user_id} session={req.session_id} "
        f"record={target_id} adopted={req.adopted}"
    )
    return target_id


# ── 影响分析反馈 ────────────────────────────────────────────────────────────

@router.post("/impact/feedback", response_model=ResponseSchema[FeedbackResult])
async def submit_impact_feedback(req: FeedbackRequest, user: UserContext = Depends(get_current_user)):
    """
    影响分析反馈回写接口。

    人工确认影响分析结论是否准确，回写到 ai_impact_analysis.adopted 字段。
    只能反馈本人会话的结论；一次只回写一行（见模块头注释）。
    """
    record_id = await _update_owned_row("ai_impact_analysis", req, user)
    return ResponseSchema(
        data=FeedbackResult(
            session_id=req.session_id, updated=record_id is not None, record_id=record_id
        )
    )


# ── 报告解读反馈 ────────────────────────────────────────────────────────────

@router.post("/report/feedback", response_model=ResponseSchema[FeedbackResult])
async def submit_report_feedback(req: FeedbackRequest, user: UserContext = Depends(get_current_user)):
    """
    报告解读反馈回写接口。

    人工确认报告解读结论是否准确，回写到 ai_report_interpretations.adopted 字段。
    只能反馈本人会话的结论；一次只回写一行（见模块头注释）。
    """
    record_id = await _update_owned_row("ai_report_interpretations", req, user)
    return ResponseSchema(
        data=FeedbackResult(
            session_id=req.session_id, updated=record_id is not None, record_id=record_id
        )
    )
