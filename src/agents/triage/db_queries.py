"""PostgreSQL 异步查询函数。参考天宫医疗版 db_queries.py。"""

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.alm.model import Phenomenon, RootCause, CausePhenomenon, OwnerDomain, DtcCode, AlmIssue


async def load_issue_context(db: AsyncSession, issue_id: int) -> dict | None:
    """加载问题单上下文：标题、描述、DTC 快照。"""
    issue = await db.get(AlmIssue, issue_id)
    if issue is None:
        return None
    return {
        "issue_id": issue.id,
        "issue_title": issue.title or "",
        "issue_desc": issue.description or "",
        "issue_dtc_snapshot": issue.dtc_snapshot or "",
        "source": issue.source or "customer",
    }


async def get_all_phenomena(db: AsyncSession) -> list[dict]:
    """获取全部现象码（供 LLM prompt 注入）。"""
    result = await db.execute(
        select(Phenomenon.id, Phenomenon.name, Phenomenon.code, Phenomenon.colloquial, Phenomenon.business_line)
        .order_by(Phenomenon.id)
    )
    rows = result.all()
    return [
        {"id": r.id, "name": r.name, "code": r.code, "colloquial": r.colloquial, "business_line": r.business_line}
        for r in rows
    ]


async def match_phenomena_by_names(db: AsyncSession, names: list[str]) -> list[dict]:
    """精确匹配现象名 → 返回现象 id/code/name。"""
    if not names:
        return []
    result = await db.execute(
        select(Phenomenon.id, Phenomenon.name, Phenomenon.code, Phenomenon.business_line)
        .where(Phenomenon.name.in_(names))
    )
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

    sql = sa_text("""
        INSERT INTO ai_triage_results
            (source_issue_id, session_id, raw_input,
             confirmed_phenomena, denied_phenomena,
             candidate_causes, primary_cause_code, primary_confidence,
             suggest_domain_id, total_rounds, force_conclude)
        VALUES
            (:source_issue_id, :session_id, :raw_input,
             :confirmed_phenomena, :denied_phenomena,
             :candidate_causes, :primary_cause_code, :primary_confidence,
             :suggest_domain_id, :total_rounds, :force_conclude)
        RETURNING id
    """)

    candidate_causes_json = json.dumps([
        c.model_dump() if hasattr(c, 'model_dump') else c
        for c in result.get("candidate_causes", [])
    ], ensure_ascii=False)

    params = {
        "source_issue_id": result.get("issue_id"),
        "session_id": result.get("session_id", ""),
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
                (id, source_issue_id, session_id, raw_input,
                 confirmed_phenomena, denied_phenomena,
                 candidate_causes, primary_cause_code, primary_confidence,
                 suggest_domain_id, total_rounds, force_conclude)
            VALUES
                (:id_val, :source_issue_id, :session_id, :raw_input,
                 :confirmed_phenomena, :denied_phenomena,
                 :candidate_causes, :primary_cause_code, :primary_confidence,
                 :suggest_domain_id, :total_rounds, :force_conclude)
        """),
        params,
    )
    return new_id
