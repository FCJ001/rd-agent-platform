"""PostgreSQL 异步查询函数。参考天宫医疗版 db_queries.py。"""

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.alm.model import Phenomenon, RootCause, CausePhenomenon, OwnerDomain, DtcCode, AlmIssue
from src.utils.mask import mask_free_text


async def load_issue_context(db: AsyncSession, issue_id: int) -> dict | None:
    """加载问题单上下文：标题、描述、DTC 快照。

    标题/描述是自由文本，可能嵌着完整 VIN/手机号 —— 进 LLM 前兜底打码，
    兜住存量数据未脱敏的情况（新数据在 sync/webhook 写入路径已脱敏）。
    """
    issue = await db.get(AlmIssue, issue_id)
    if issue is None:
        return None
    return {
        "issue_id": issue.id,
        "issue_title": mask_free_text(issue.title) or "",
        "issue_desc": mask_free_text(issue.description) or "",
        "issue_dtc_snapshot": issue.dtc_snapshot or "",
        "source": issue.source or "customer",
        # 问题单上的业务线是这件事的权威归属 —— 分诊带上它才能把现象词表、
        # 候选根因都限定在同一条线内
        "business_line": issue.business_line or "",
    }


async def get_all_phenomena(db: AsyncSession, business_line: str = "") -> list[dict]:
    """获取现象码（供 LLM prompt 注入）。

    ★ business_line 非空时只给该线的词表。现象名跨线共享，混着给会让
      LLM 把故障归一化到另一条线的现象上。
    """
    stmt = select(
        Phenomenon.id, Phenomenon.name, Phenomenon.code,
        Phenomenon.colloquial, Phenomenon.business_line,
    )
    if business_line:
        stmt = stmt.where(Phenomenon.business_line == business_line)
    result = await db.execute(stmt.order_by(Phenomenon.id))
    rows = result.all()
    return [
        {"id": r.id, "name": r.name, "code": r.code, "colloquial": r.colloquial, "business_line": r.business_line}
        for r in rows
    ]


async def match_phenomena_by_names(
    db: AsyncSession, names: list[str], business_line: str = "",
) -> list[dict]:
    """精确匹配现象名 → 返回现象 id/code/name。

    ★ 现象名在 PG 里是 (business_line, name) 复合唯一 —— 同名跨线各一行。
      business_line 非空时只取本线的行，否则同一个名字会返回两条（分属两线）。
    """
    if not names:
        return []
    stmt = select(
        Phenomenon.id, Phenomenon.name, Phenomenon.code, Phenomenon.business_line,
    ).where(Phenomenon.name.in_(names))
    if business_line:
        stmt = stmt.where(Phenomenon.business_line == business_line)
    result = await db.execute(stmt)
    rows = result.all()
    return [{"id": r.id, "name": r.name, "code": r.code, "business_line": r.business_line} for r in rows]


async def get_causes_by_phenomenon_ids(db: AsyncSession, phenom_ids: list[int]) -> list[dict]:
    """通过 cause_phenomena 表查询关联的根因（含 weight、is_core）。"""
    if not phenom_ids:
        return []
    result = await db.execute(
        select(
            CausePhenomenon.cause_id,
            CausePhenomenon.phenomenon_id,
            CausePhenomenon.weight,
            CausePhenomenon.is_core,
            RootCause.code,
            RootCause.name,
            RootCause.domain_id,
            RootCause.fix_way,
            RootCause.fix_duration,
            RootCause.verify_items,
        )
        .join(RootCause, RootCause.id == CausePhenomenon.cause_id)
        .where(CausePhenomenon.phenomenon_id.in_(phenom_ids))
    )
    rows = result.all()
    return [
        {
            "cause_id": r.cause_id, "phenomenon_id": r.phenomenon_id,
            "weight": r.weight, "is_core": r.is_core,
            "code": r.code, "name": r.name, "domain_id": r.domain_id,
            "fix_way": r.fix_way, "fix_duration": r.fix_duration,
            "verify_items": r.verify_items,
        }
        for r in rows
    ]


async def lookup_dtc_codes(db: AsyncSession, dtc_list: list[str]) -> list[dict]:
    """查询 DTC 故障码信息。"""
    if not dtc_list:
        return []
    result = await db.execute(
        select(DtcCode.code, DtcCode.system, DtcCode.description_zh, DtcCode.business_line)
        .where(DtcCode.code.in_(dtc_list))
    )
    rows = result.all()
    return [{"code": r.code, "system": r.system, "description": r.description_zh, "business_line": r.business_line} for r in rows]


async def save_triage_result(db: AsyncSession, result: dict) -> int:
    """写入 ai_triage_results 表，返回记录 ID。"""
    import json
    from sqlalchemy import text as sa_text

    candidate_causes_json = json.dumps([
        c.model_dump() if hasattr(c, 'model_dump') else c
        for c in result.get("candidate_causes", [])
    ], ensure_ascii=False)

    params = {
        "source_issue_id": result.get("issue_id"),
        "session_id": result.get("session_id", ""),
        # 归属与作用域：空值一律落 NULL（不是空串）—— 检索侧对 NULL 的
        # fail-closed 语义就是靠这个区分的
        "user_id": int(result["user_id"]) if result.get("user_id") else None,
        "business_line": result.get("business_line") or None,
        "raw_input": result.get("raw_input", ""),
        "confirmed_phenomena": json.dumps(result.get("confirmed_phenomena", []), ensure_ascii=False),
        "denied_phenomena": json.dumps(result.get("denied_phenomena", []), ensure_ascii=False),
        "candidate_causes": candidate_causes_json,
        "primary_cause_code": result.get("primary_cause_code"),
        "primary_confidence": result.get("primary_confidence", 0.0),
        "suggest_domain_id": result.get("suggest_domain_id"),
        "total_rounds": result.get("total_rounds", 1),
        "force_conclude": result.get("force_conclude", False),
    }

    r = await db.execute(sa_text("SELECT nextval('ai_triage_results_id_seq') AS id"))
    new_id = r.scalar_one()
    params["id_val"] = new_id

    await db.execute(
        sa_text("""
            INSERT INTO ai_triage_results
                (id, source_issue_id, user_id, business_line,
                 session_id, raw_input,
                 confirmed_phenomena, denied_phenomena,
                 candidate_causes, primary_cause_code, primary_confidence,
                 suggest_domain_id, total_rounds, force_conclude)
            VALUES
                (:id_val, :source_issue_id, :user_id, :business_line,
                 :session_id, :raw_input,
                 :confirmed_phenomena, :denied_phenomena,
                 :candidate_causes, :primary_cause_code, :primary_confidence,
                 :suggest_domain_id, :total_rounds, :force_conclude)
        """),
        params,
    )
    return new_id
