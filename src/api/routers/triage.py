"""分诊 API Router。POST /api/v1/triage"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.agents.workers.triage_agent import get_triage_agent
from src.agents.triage.session_store import TriageSessionStore
from src.agents.triage.state import TriageState
from src.core.base_schema import ResponseSchema
from src.core.deps import UserContext, get_current_user
from src.core.exceptions import ERR_PERMISSION_DENIED, BizException
from src.core.logger import logger
from src.core.rate_limit import rate_limit
from src.infra.redis_cache import get_checkpointer_redis
from src.utils.mask import mask_free_text

router = APIRouter(prefix="/api/v1/triage", tags=["分诊诊断"])

_triage_rate_limit = rate_limit("triage", limit=20, window_seconds=60)


# ── 请求体 ──
class TriageRequest(BaseModel):
    raw_input: str = Field(..., max_length=8000, description="故障描述（口语），如'车机偶尔黑屏'")
    session_id: str | None = Field(None, max_length=100, description="继续追问时传入上次返回的 session_id")
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
async def triage(
    req: TriageRequest,
    user: UserContext = Depends(get_current_user),
    _rl: None = Depends(_triage_rate_limit),
):
    """
    分诊诊断接口。向后兼容：支持多轮追问（通过 Redis 持久化会话状态）。

    输入一句故障描述（如"车机偶尔黑屏"），返回：
    - 规范化后的现象名
    - 候选根因列表（按置信度降序）
    - 如置信度不足，返回追问问题，前端可继续调用本接口传入 session_id 继续对话

    ★ 会话 key 以认证用户开头（user_id:session_id）：REST 路径与 chat 工具的
      thread 规范一致，且别人猜到 session_id 也拿不到他人的会话状态。
    """
    thread_key = f"{user.user_id}:{req.session_id}" if req.session_id else None
    # 进 LLM / 日志 / 落库前统一脱敏（chat 入口同规）；日志不打原文
    masked_input = mask_free_text(req.raw_input)
    logger.info(f"[TRIAGE] user={user.user_id} session={req.session_id or 'new'} input={masked_input[:80]}")

    # 从 store 恢复上一轮状态（如有）
    existing_state = None
    if thread_key:
        store = TriageSessionStore(get_checkpointer_redis())
        progress = await store.load(thread_key)
        if progress:
            existing_state = progress.state

    agent = get_triage_agent()
    result = await agent.diagnose(
        raw_input=masked_input,
        session_id=thread_key,
        issue_id=req.issue_id,
        existing_state=existing_state,
    )

    # 如果未收敛，将状态写入 store 供下一轮使用
    if result["status"] == "asking" and result["_state"]:
        store = TriageSessionStore(get_checkpointer_redis())
        await store.save(
            result["session_id"],
            TriageState(**result["_state"]),
            reply=result.get("follow_up_questions", [""])[0] if result.get("follow_up_questions") else "",
        )

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
    session_id: str = Field(..., max_length=100, description="分诊会话 ID")
    adopted: bool = Field(..., description="是否采纳诊断结论")
    comment: str | None = Field(None, max_length=2000, description="反馈备注（如不采纳原因）")
    correct_cause_code: str | None = Field(
        None,
        description="人工纠正时的正确根因编码（如 RC-EV-0012）；不采纳时传入，回写图谱",
    )


class FeedbackResult(BaseModel):
    session_id: str
    updated: bool


@router.post("/feedback", response_model=ResponseSchema[FeedbackResult])
async def submit_feedback(req: FeedbackRequest, user: UserContext = Depends(get_current_user)):
    """
    分诊反馈回写接口。

    人工确认诊断结论是否准确，回写到 ai_triage_results.adopted 字段，
    用于统计分诊准确率、优化知识库。

    ★ 所有权校验：session_id 是 user_id:session_id 结构（本路由签发），
      只允许操作本人会话 —— 否则任何人可篡改他人诊断结论并污染图谱权重。
    """
    import json

    owner = req.session_id.split(":", 1)[0] if ":" in req.session_id else None
    if owner != user.user_id and user.role != "admin":
        raise BizException("只能反馈本人会话的诊断结论", ERR_PERMISSION_DENIED)
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
                correct_cause_code=req.correct_cause_code,
            )
            logger.info(
                f"[TRIAGE-FB] 图谱弱化已触发 session={req.session_id} "
                f"correct={req.correct_cause_code or '未提供'}"
            )
        except Exception as e:
            logger.warning(f"[TRIAGE-FB] 图谱弱化失败: {e}")

    return ResponseSchema(data=FeedbackResult(session_id=req.session_id, updated=updated))
