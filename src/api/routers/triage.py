"""分诊 API Router。POST /api/v1/triage"""

from contextlib import nullcontext

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.agents.workers.triage_agent import get_triage_agent
from src.agents.triage.gate import triage_gate
from src.agents.triage.lock import ConversationBusyError, session_lock
from src.agents.triage.session_store import TriageSessionStore
from src.agents.triage.state import TriageState
from src.core.base_schema import ResponseSchema
from src.core.deps import UserContext, get_current_user
from src.core.exceptions import ERR_CONVERSATION_BUSY, ERR_PERMISSION_DENIED, BizException
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
    # 本次结论在 ai_triage_results 的行 id（收敛时才有）。反馈接口带上它可以
    # 精确回写这一条 —— 同一会话里重复诊断是正常行为，会有多行结论。
    record_id: int | None = Field(None, description="诊断记录 id，用于 /feedback 精确回写")


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

    agent = get_triage_agent()
    # 全局并发闸门（chat 工具 / webhook 自动分诊共用同一额度）：满员时
    # 有界排队，排队失败抛 TriageSystemBusyError —— 它是 BizException 子类，
    # 全局处理器会回 42902/HTTP 429，无需在这里捕获。
    # ★ 闸门在会话锁外层：排队等空位的几十秒不该计入锁的持有时长。
    #
    # ★ 会话锁与本模块的 key 规范必须和 chat 工具一致（triage_lock:{thread_key}）：
    #   这条路径的 load→diagnose→save 和 chat 的分诊工具作用于同一份
    #   triage_state:{thread_key}。不互斥的话两者会各自读到同一份进度、
    #   各跑一轮、后写覆盖先写，state.round 直接错位（下次追问文本对不上）。
    #   原来这里只有全局闸门 —— 限总量不挡具体谁，挡不住这个。
    #   新建会话（无 session_id）不锁：session_id 由 diagnose 现场生成，
    #   不存在第二个请求能摸到它。
    lock_cm = session_lock(thread_key) if thread_key else nullcontext()
    try:
        async with triage_gate():
            async with lock_cm:
                # 从 store 恢复上一轮状态（如有）
                existing_state = None
                if thread_key:
                    store = TriageSessionStore(get_checkpointer_redis())
                    progress = await store.load(thread_key)
                    if progress:
                        existing_state = progress.state

                result = await agent.diagnose(
                    raw_input=masked_input,
                    session_id=thread_key,
                    issue_id=req.issue_id,
                    existing_state=existing_state,
                )

                # 未收敛 → 落盘进度供下一轮；收敛/超轮 → 清掉进度。
                # ★ 收敛必须清，与 chat 路径的 _finish_triage 保持一致（清进度 =
                #   这段对话结束）。不清的话，同一 session_id 的下一次诊断会读到
                #   上一段的进度、按「第 N+1 轮」续聊 —— 两个不相干的诊断串在一起
                #   （确认现象、否定现象、候选根因全被继承）。
                #   挂起标记也随之清掉：收敛之后没有「在等回答的追问」，
                #   留着它只会让后续判断误判。
                store = TriageSessionStore(get_checkpointer_redis())
                if result["status"] == "asking" and result["_state"]:
                    await store.save(
                        result["session_id"],
                        TriageState(**result["_state"]),
                        reply=(result.get("follow_up_questions") or [""])[0],
                    )
                else:
                    await store.clear(result["session_id"])
    except ConversationBusyError as e:
        # ★ 必须在这里转成 BizException：ConversationBusyError 故意不是
        #   BizException 子类（它的设计前提是「被 chat 路由捕获、图可能停在
        #   追问中途」），漏到全局处理器会变成 500 —— 用户以为系统挂了。
        #   REST 这条路径没有图内状态要保，忙就是干净的业务拒绝（409）。
        logger.info(f"[TRIAGE] 会话忙，拒绝并发处理 thread={thread_key}")
        raise BizException(
            "上一次诊断还在处理中，请稍等几秒再发送。", ERR_CONVERSATION_BUSY
        ) from e

    triage_result = TriageResult(
        session_id=result["session_id"],
        status=result["status"],
        round=result["round"],
        normalized_phenomena=result["normalized_phenomena"],
        candidate_causes=[CandidateCauseOut(**c) for c in result["candidate_causes"]],
        confidence=result["confidence"],
        follow_up_questions=result["follow_up_questions"],
        diagnostic_summary=result["diagnostic_summary"],
        record_id=result.get("record_id"),
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
    record_id: int | None = Field(
        None,
        description="要回写的诊断记录 id（ai_triage_results.id）。不传则作用于本会话"
                    "最新一条 —— 同一会话多次诊断时用它避免反馈错行",
    )


class FeedbackResult(BaseModel):
    session_id: str
    updated: bool
    # 实际回写的行 id：后续对同一行再次反馈时带上它可避免歧义
    record_id: int | None = None


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
        # ★ 先定位唯一一行，再按主键回写。同一 session_id 下会有多行（用户在
        #   同一会话里重复诊断是正常行为），按 session_id 整批更新会把该会话
        #   所有历史结论一起标记为已采纳；而且拿去回流图谱的现象/根因也会取自
        #   其中任一行（原来的 fetchone 没有任何排序，取哪行不确定）。
        if req.record_id is not None:
            target_sql = (
                "SELECT id, confirmed_phenomena, primary_cause_code FROM ai_triage_results "
                "WHERE id = :record_id AND session_id = :session_id"
            )
            target_params = {"record_id": req.record_id, "session_id": req.session_id}
        else:
            # 不带 id 时取最新一行 —— 用户刚看到的那条结论
            target_sql = (
                "SELECT id, confirmed_phenomena, primary_cause_code FROM ai_triage_results "
                "WHERE session_id = :session_id ORDER BY id DESC LIMIT 1"
            )
            target_params = {"session_id": req.session_id}
        row = (await db.execute(sa_text(target_sql), target_params)).fetchone()
        if row is None:
            logger.warning(
                f"[TRIAGE-FB] 无匹配记录 session={req.session_id} record={req.record_id}"
            )
            return ResponseSchema(
                data=FeedbackResult(session_id=req.session_id, updated=False)
            )
        target_id, phenomena_raw, cause_code = row[0], row[1], row[2]
        # 额外带 session_id：即使有人拿别人的 record_id 来试，也落不到那行上
        await db.execute(
            sa_text(
                "UPDATE ai_triage_results SET adopted = :adopted, feedback_comment = :comment, "
                "updated_at = NOW() WHERE id = :target_id AND session_id = :session_id"
            ),
            {
                "adopted": req.adopted,
                "comment": req.comment,
                "target_id": target_id,
                "session_id": req.session_id,
            },
        )
        await db.commit()
        updated = True

    confirmed = json.loads(phenomena_raw) if isinstance(phenomena_raw, str) else (phenomena_raw or [])

    # ── 触发诊断结论回流图谱 ──
    if req.adopted:
        try:
            from src.agents.triage.feedback_loop import reinforce_graph_on_adopted
            await reinforce_graph_on_adopted(
                confirmed_phenomena=confirmed,
                primary_cause_code=cause_code,
                session_id=req.session_id,
            )
            logger.info(f"[TRIAGE-FB] 图谱增强已触发 session={req.session_id} record={target_id}")
        except Exception as e:
            logger.warning(f"[TRIAGE-FB] 图谱增强失败: {e}")
    else:
        try:
            from src.agents.triage.feedback_loop import weaken_graph_on_rejected
            await weaken_graph_on_rejected(
                confirmed_phenomena=confirmed,
                primary_cause_code=cause_code,
                correct_cause_code=req.correct_cause_code,
            )
            logger.info(
                f"[TRIAGE-FB] 图谱弱化已触发 session={req.session_id} record={target_id} "
                f"correct={req.correct_cause_code or '未提供'}"
            )
        except Exception as e:
            logger.warning(f"[TRIAGE-FB] 图谱弱化失败: {e}")

    return ResponseSchema(
        data=FeedbackResult(session_id=req.session_id, updated=updated, record_id=target_id)
    )
