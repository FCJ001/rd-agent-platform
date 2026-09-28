"""Java ALM 平台交互工具。开发期调 Java API 打 logger 占位。

建单是两段式（HITL 责任边界）：call_create_issue 只落草稿，
call_confirm_issue 在用户明确确认后才向平台提交 —— AI 永远不直接建单。
"""

from datetime import datetime, timezone

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.core.config import get_settings
from src.core.deps import UserContext
from src.core.logger import logger

settings = get_settings()


# 开发期占位域名。Java 平台接入后把 PLATFORM_ALM_API_URL 配成真实地址即可
_PLACEHOLDER_HOST = "alm.internal"
_SEVERITIES = {"blocker", "critical", "normal", "minor"}
_DRAFT_PENDING = "pending"
_DRAFT_SUBMITTED = "submitted"
_DRAFT_EXPIRED = "expired"
# 业务线取值来自 settings.BUSINESS_LINES（见 config.py 注释），不再写死在代码里


def _platform_ready() -> bool:
    """Java 平台 API 是否已真实配置（不是占位符）。"""
    return bool(settings.PLATFORM_ALM_API_URL) and _PLACEHOLDER_HOST not in settings.PLATFORM_ALM_API_URL


# ── 草稿存取（薄封装，单测通过 monkeypatch 这几个函数隔离 DB）──────────

async def _save_draft(payload: dict) -> int:
    """落一张 pending 草稿，返回草稿编号。失败抛异常由工具层兜底。"""
    from src.infra.db import AsyncSessionLocal
    from src.modules.alm.model import AiIssueDraft

    async with AsyncSessionLocal() as db:
        draft = AiIssueDraft(**payload, status=_DRAFT_PENDING)
        db.add(draft)
        await db.commit()
        return draft.id


async def _load_draft(draft_id: int):
    """按编号取草稿，不存在返回 None。"""
    from src.infra.db import AsyncSessionLocal
    from src.modules.alm.model import AiIssueDraft

    async with AsyncSessionLocal() as db:
        return await db.get(AiIssueDraft, draft_id)


async def _mark_draft(draft_id: int, status: str, issue_no: str | None = None) -> None:
    """草稿状态迁移（pending → submitted/expired，单向）。"""
    from sqlalchemy import update
    from src.infra.db import AsyncSessionLocal
    from src.modules.alm.model import AiIssueDraft

    async with AsyncSessionLocal() as db:
        stmt = (
            update(AiIssueDraft)
            .where(
                AiIssueDraft.id == draft_id,
                # 条件更新 = 幂等：并发的第二次确认/过期判定都改不动行
                AiIssueDraft.status == _DRAFT_PENDING,
            )
            .values(status=status, issue_no=issue_no)
        )
        await db.execute(stmt)
        await db.commit()


def _draft_age_seconds(draft) -> float:
    """草稿距今秒数。created_at 是 DB 服务器时间（naive 视为 UTC）。"""
    created = draft.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created).total_seconds()


@tool
async def call_create_issue(
    title: str,
    description: str,
    severity: str,
    business_line: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """生成问题单草稿（**不建单**，等待用户确认）。
    适用场景：诊断收敛后需要正式跟踪、或用户要求建单时。
    调用时机：Agent 判断需要走正式流程。返回草稿编号并展示给用户；
    用户明确同意后必须调用 call_confirm_issue 正式提交，绝不能跳过确认。

    Args:
        title: 问题标题（简短描述）
        description: 问题描述（故障现象、DTC码等）
        severity: 严重度（blocker/critical/normal/minor）
        business_line: 业务线。★ 仅作兜底：请求上下文中有用户所属业务线时
            以它为准，LLM 传的值只在上下文取不到时使用（作用域不能让模型决定）
    """
    # 参数先校验：LLM 传参不可信，脏数据不能进草稿
    if severity not in _SEVERITIES:
        return f"草稿生成失败：severity 取值不合法（{severity!r}），只允许 {'/'.join(sorted(_SEVERITIES))}。"

    ctx = runtime.context
    # 作用域优先级：请求上下文（用户所属业务线）> LLM 入参。
    # 允许的取值来自配置，新项目上线改 BUSINESS_LINES 即可
    allowed = settings.business_lines
    scope = ctx.business_line or business_line
    if scope not in allowed:
        return (
            f"草稿生成失败：business_line 取值不合法（{scope!r}），"
            f"只允许 {'/'.join(sorted(allowed))}。"
        )

    try:
        user_id = int(ctx.user_id) if ctx.user_id else None
        draft_id = await _save_draft({
            "user_id": user_id,
            "session_id": ctx.session_id or "unknown",
            "title": title[:200],
            "description": description,
            "severity": severity,
            "business_line": scope,
            "source": ctx.role,
            "owner_domain_id": ctx.owner_domain_id,
        })
    except Exception as e:
        # 草稿落不下去就绝不能往下走（确认时无据可依）
        logger.warning(f"[JAVA-API] 草稿保存失败 user={ctx.user_id} title={title!r}: {e}")
        return "草稿保存失败（数据库异常），请稍后重试。"

    ttl_min = settings.ISSUE_DRAFT_TTL_SECONDS // 60
    return (
        f"已生成问题单草稿（编号 {draft_id}）：\n"
        f"- 标题：{title}\n"
        f"- 严重度：{severity} | 业务线：{scope}\n"
        f"- 描述：{description}\n\n"
        f"请把草稿内容展示给用户并请其确认。用户确认后调用 "
        f"call_confirm_issue(draft_id={draft_id}) 正式提交；用户要求修改时，"
        f"按新内容重新生成草稿即可（旧草稿 {ttl_min} 分钟未确认自动过期）。"
    )


@tool
async def call_confirm_issue(
    draft_id: int,
    runtime: ToolRuntime[UserContext],
) -> str:
    """确认提交问题单草稿（用户明确同意创建后才能调用）。
    属主校验 + 幂等：只有本人创建、仍处于 pending 且未过期的草稿可提交。

    Args:
        draft_id: 草稿编号（call_create_issue 返回的编号）
    """
    ctx = runtime.context
    try:
        draft = await _load_draft(draft_id)
    except Exception as e:
        logger.warning(f"[JAVA-API] 草稿读取失败 draft={draft_id}: {e}")
        return "草稿读取失败，请稍后重试。"

    if draft is None:
        return f"草稿 {draft_id} 不存在，请重新生成。"

    # ★ IDOR 防护：属主不符直接拒，不泄露草稿内容
    if draft.user_id is None or int(ctx.user_id or 0) != draft.user_id:
        logger.warning(
            f"[JAVA-API] 草稿属主校验失败 draft={draft_id} "
            f"owner={draft.user_id} caller={ctx.user_id}"
        )
        return f"无权操作草稿 {draft_id}。"

    if draft.status == _DRAFT_SUBMITTED:
        return f"草稿 {draft_id} 已提交过（单号 {draft.issue_no or '待平台回写'}），请勿重复确认。"
    if draft.status == _DRAFT_EXPIRED:
        return f"草稿 {draft_id} 已过期，请重新生成。"

    # 惰性过期：超时未确认的 pending 草稿就地关闭
    if _draft_age_seconds(draft) > settings.ISSUE_DRAFT_TTL_SECONDS:
        await _mark_draft(draft_id, _DRAFT_EXPIRED)
        ttl_min = settings.ISSUE_DRAFT_TTL_SECONDS // 60
        return f"草稿 {draft_id} 已超过 {ttl_min} 分钟未确认，已过期。请重新生成。"

    payload = {
        "draft_id": draft_id,
        "title": draft.title,
        "description": draft.description,
        "severity": draft.severity,
        "business_line": draft.business_line,
        "source": draft.source,
        "reporter_id": ctx.user_id,
        "owner_domain_id": draft.owner_domain_id,
    }

    if not _platform_ready():
        # ★ 诚实失败 + 关闭草稿防重复确认：开发期平台没接入，
        #   绝不能让用户以为建单成功；草稿置 submitted 使其不可再确认
        logger.warning(f"[JAVA-API] 平台未接入，草稿关闭未真正建单 draft={draft_id} user={ctx.user_id}")
        await _mark_draft(draft_id, _DRAFT_SUBMITTED)
        return (
            "ALM 平台接口尚未接入（开发环境），本次**没有**真正创建问题单。\n"
            f"草稿 {draft_id} 已关闭。请到 ALM 平台手动创建。"
        )

    logger.info(
        f"[JAVA-API] POST {settings.PLATFORM_ALM_API_URL}/issues "
        f"user={ctx.user_id} draft={draft_id} payload={payload}"
    )
    # 平台接入后：POST 成功拿回 issue_no 再落库；当前占位阶段单号留空
    await _mark_draft(draft_id, _DRAFT_SUBMITTED)

    return (
        f"已确认并提交问题单（草稿 {draft_id}）。\n"
        f"平台地址：{settings.PLATFORM_ALM_URL}/issues\n"
        f"标题：{draft.title}\n"
        f"严重度：{draft.severity} | 业务线：{draft.business_line}"
    )


@tool
async def call_link_issue(
    issue_no: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """关联已有问题单到当前诊断会话。
    适用场景：用户提到已有问题单号，需要基于此单进行诊断。

    Args:
        issue_no: ALM 平台问题单号，如 ISS-2025-00123
    """
    ctx = runtime.context

    logger.info(
        f"[JAVA-API] GET {settings.PLATFORM_ALM_API_URL}/issues/{issue_no} "
        f"user={ctx.user_id} session={ctx.session_id}"
    )

    return (
        f"已在本会话关联问题单 {issue_no}（仅本地关联，未调用平台接口）。\n"
        f"查看详情：{settings.PLATFORM_ALM_URL}/issues/{issue_no}\n"
        f"接下来可以对此问题进行诊断分析。"
    )


@tool
async def call_close_issue(
    issue_no: str,
    root_cause: str,
    fix_action: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """在 ALM 平台结案。
    适用场景：诊断出明确根因且修复方案已实施时。
    注意：Agent 只建议结案，最终由平台侧审批。

    Args:
        issue_no: ALM 平台问题单号
        root_cause: 根因描述
        fix_action: 修复措施
    """
    ctx = runtime.context
    payload = {
        "issue_no": issue_no,
        "root_cause": root_cause,
        "fix_action": fix_action,
        "status": "verified",
        "operator_id": ctx.user_id,
    }

    if not _platform_ready():
        logger.warning(f"[JAVA-API] 平台未接入，未真正提交结案 issue_no={issue_no}")
        return (
            "ALM 平台接口尚未接入（开发环境），本次**没有**真正提交结案建议。\n"
            f"待提交内容：{issue_no} 结案，根因 {root_cause}\n"
            "请到 ALM 平台手动走结案流程。"
        )

    logger.info(
        f"[JAVA-API] POST {settings.PLATFORM_ALM_API_URL}/issues/{issue_no}/close "
        f"user={ctx.user_id} payload={payload}"
    )

    return (
        f"已提交结案建议。\n"
        f"问题单号：{issue_no}\n"
        f"根因：{root_cause}\n"
        f"修复措施：{fix_action}\n"
        f"平台审批链接：{settings.PLATFORM_ALM_URL}/issues/{issue_no}"
    )


PLATFORM_TOOLS = [call_create_issue, call_confirm_issue, call_link_issue, call_close_issue]
