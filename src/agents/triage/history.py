# ============================================================
# 历史诊断复用：查「以前诊断过的同类问题」
#
# 定位：分诊结论沉淀下来的**项目知识**，按两个维度检索
#   · mine —— 我诊断过的（个人维度，跨业务线可见）
#   · line —— 这条业务线上别人诊断过的（项目维度，跨用户共享）
#
# ★ 为什么不再走长期记忆：记忆集合的 namespace 是 users/{uid}/memories，
#   结论写进去只有本人能召回 —— 同一个故障两个人各诊断一次互相看不到。
#   知识复用需要的是「项目内共享」，那是记忆通道给不了的语义。
#
# ★ 角色门禁：项目维度对工程师是知识共享，对客户是隐私泄漏。客户只能查
#   自己那条线以上的东西 —— 规则写成代码常量，未知角色 fail-closed 到个人维度，
#   与 src/core/permissions.py 同一套思路（不建 RBAC 表，权限主数据在平台侧）。
#
# ★ 两级可信度：人工采纳过的结论优先；一条都没有时降级到未复核记录，但
#   必须在返回文本里带「未复核」标记 —— 让 LLM 转述时保留这个不确定性。
#   不这样做的话，反馈闭环没人点采纳时这功能就永远空转。
# ============================================================

import json

import psycopg2

from src.core.config import get_settings
from src.core.logger import logger
from src.infra.pg_scope import apply_scope

# 可查项目维度（跨用户）的角色。未列出的角色一律降级为个人维度
PROJECT_SCOPE_ROLES = frozenset({"engineer", "business", "admin"})

SCOPE_MINE = "mine"
SCOPE_LINE = "line"

MAX_KEYWORDS = 5
MAX_LIMIT = 10

_LEVEL_ADOPTED = "adopted"
_LEVEL_UNREVIEWED = "unreviewed"


def _connect():
    s = get_settings()
    conn = psycopg2.connect(
        host=s.DB_HOST, port=s.DB_PORT,
        user=s.DB_USER, password=s.DB_PASSWORD, dbname=s.DB_NAME,
    )
    # ★ 裸连接必须先带上作用域：RLS 策略读 current_setting('app.business_line')，
    #   不设置就是"没作用域"→ 查回空集，且不报错
    apply_scope(conn)
    return conn


def _keywords(raw: str) -> list[str]:
    """把关键词串拆成检索词。逗号 / 顿号 / 空白都当分隔符。

    要求调用方传**现象名或关键词**而不是整句提问：中文没有词间空格，
    整句 ILIKE 必然匹配不到（"%上次那个黑屏的问题怎么解决的%"）。
    """
    parts: list[str] = []
    for chunk in raw.replace("，", ",").replace("、", ",").split(","):
        for token in chunk.split():
            token = token.strip()
            if len(token) >= 2:
                parts.append(token)
    # 去重保序
    seen: set[str] = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:MAX_KEYWORDS]


def _query_sync(
    keywords: list[str], user_id: str, business_line: str, scope: str,
    level: str, limit: int,
) -> list[dict]:
    conn = _connect()
    try:
        cur = conn.cursor()

        where = []
        params: list = []

        if scope == SCOPE_MINE:
            where.append("tir.user_id = %s")
            params.append(int(user_id))
        else:
            where.append("tir.business_line = %s")
            params.append(business_line)

        if level == _LEVEL_ADOPTED:
            where.append("tir.adopted = true")
        else:
            # 未复核：既没被采纳也没被否决
            where.append("tir.adopted IS NULL")

        # 关键词来自 LLM 抽取，按不可信参数处理：模式占位符化，通配符拼在值里
        likes = [f"%{k}%" for k in keywords]
        or_clause = " OR ".join(
            ["tir.confirmed_phenomena::text ILIKE %s OR tir.raw_input ILIKE %s OR ai.title ILIKE %s"] * len(likes)
        )
        for like in likes:
            params.extend([like, like, like])
        where.append(f"({or_clause})")

        where.append("tir.primary_cause_code IS NOT NULL")

        cur.execute(
            f"""SELECT tir.id, tir.primary_cause_code, tir.primary_confidence,
                       tir.confirmed_phenomena, tir.total_rounds, tir.created_at,
                       tir.adopted, ai.issue_no, ai.title, ai.status
                FROM ai_triage_results tir
                LEFT JOIN alm_issues ai ON ai.id = tir.source_issue_id
                WHERE {' AND '.join(where)}
                ORDER BY tir.primary_confidence DESC, tir.created_at DESC
                LIMIT %s""",
            (*params, limit),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    results = []
    for r in rows:
        try:
            phenom = json.loads(r[3]) if isinstance(r[3], str) else r[3]
        except Exception:
            phenom = r[3]
        results.append({
            "cause_code": r[1],
            "confidence": r[2] or 0.0,
            "confirmed_phenomena": phenom,
            "total_rounds": r[4],
            "created_at": str(r[5])[:10] if r[5] else "",
            "adopted": r[6],
            "issue_no": r[7],
            "issue_title": r[8],
            "issue_status": r[9],
        })
    return results


async def query_past_diagnoses(
    keywords: str,
    role: str = "customer",
    user_id: str = "",
    business_line: str = "",
    scope: str = SCOPE_MINE,
    limit: int = 5,
) -> dict:
    """检索历史诊断结论。返回 {items, scope, level, note}。

    scope 请求项目维度但角色不允许、或业务线未知时会降级为个人维度，
    降级原因写在 note 里（调用方应转述给用户，而不是静默给个空结果）。
    """
    import asyncio

    kws = _keywords(keywords)
    if not kws:
        return {"items": [], "scope": scope, "level": "", "note": "没有可用的检索关键词"}

    requested_scope = scope if scope in (SCOPE_MINE, SCOPE_LINE) else SCOPE_MINE
    note = ""
    effective_scope = requested_scope

    if requested_scope == SCOPE_LINE:
        if role not in PROJECT_SCOPE_ROLES:
            effective_scope = SCOPE_MINE
            note = "该项目维度的历史仅对内部角色开放，已按你本人的诊断记录检索。"
        elif not business_line:
            effective_scope = SCOPE_MINE
            note = "当前会话没有业务线归属，已按你本人的诊断记录检索。"

    if effective_scope == SCOPE_MINE and not user_id:
        return {"items": [], "scope": effective_scope, "level": "", "note": "无法确定用户身份，未检索。"}

    limit = max(1, min(limit, MAX_LIMIT))

    def _fetch(level: str) -> list[dict]:
        return _query_sync(kws, user_id, business_line, effective_scope, level, limit)

    try:
        items = await asyncio.to_thread(_fetch, _LEVEL_ADOPTED)
        level = _LEVEL_ADOPTED
        if not items:
            # 一级为空 → 降级到未复核，但必须让上层知道这批结论没经人确认
            items = await asyncio.to_thread(_fetch, _LEVEL_UNREVIEWED)
            level = _LEVEL_UNREVIEWED
    except Exception as e:
        logger.warning(f"[HISTORY] 历史诊断检索失败: {e}")
        return {"items": [], "scope": effective_scope, "level": "", "note": "检索历史记录时出错。"}

    logger.info(
        f"[HISTORY] scope={effective_scope} level={level or '-'} "
        f"keywords={kws} hits={len(items)}"
    )
    return {"items": items, "scope": effective_scope, "level": level, "note": note}


def format_past_diagnoses(result: dict) -> str:
    """把检索结果格式化成给 LLM 的文本。

    ★ 未复核那批必须带标记与提醒：口径不同（人工确认过 vs 系统当时的输出），
      混在一起说会让用户以为都是经过验证的结论。
    """
    items = result.get("items") or []
    note = result.get("note") or ""
    if not items:
        return ("未找到相关的历史诊断记录。" + (f"\n{note}" if note else ""))

    adopted = result.get("level") == "adopted"
    scope_label = "你本人的" if result.get("scope") == "mine" else "本业务线"
    lines = [f"找到 {len(items)} 条{scope_label}历史诊断："]
    if not adopted:
        lines.append("（以下记录**尚未经人工复核**，仅供参照，不要当作已验证结论）")
    if note:
        lines.append(f"（{note}）")
    lines.append("")

    for i, it in enumerate(items, 1):
        phenom = it.get("confirmed_phenomena")
        if isinstance(phenom, list):
            phenom = "、".join(str(p) for p in phenom)
        head = f"{i}. 根因 {it['cause_code']}（置信度 {it['confidence']:.0%}）"
        if it.get("issue_no"):
            head += f"，关联问题单 {it['issue_no']}"
            if it.get("issue_status"):
                head += f"（{it['issue_status']}）"
        lines.append(head)
        if phenom:
            lines.append(f"   现象：{phenom}")
        if it.get("created_at"):
            lines.append(f"   时间：{it['created_at']}")
    return "\n".join(lines)
