"""Neo4j Cypher 查询函数。参考天宫医疗版 neo4j_queries.py。

★ 本模块全是同步调用（neo4j sync driver / psycopg2）。
  从 async 节点调用时必须 `asyncio.to_thread(...)` 包装（见 graph.py 节点③）。
"""

import asyncio

from src.core.logger import logger
from src.agents.triage.state import CandidateCause
from src.infra.neo4j_client import get_neo4j_driver


def query_causes_by_phenomena(
    phenomenon_names: list[str],
    business_line: str = "",
    top_k: int = 10,
) -> list[CandidateCause]:
    """
    根据已确认现象名，从 Neo4j 查询候选根因。
    按基础置信度（命中现象数 / 该根因总现象数）降序，取 Top K。

    ★ business_line 非空时按业务线过滤候选根因。现象节点在 Neo4j 里是
      (business_line, name) 复合键 —— 同一个现象名在两条线各有节点，
      不过滤会把另一条线的根因一起捞出来（且窄根因因分母小反而排更前）。
      留空 = 不做过滤（调用方拿不到业务线时的兼容路径，会打日志）。
    """
    if not phenomenon_names:
        return []

    driver = get_neo4j_driver()

    # 谓词按需拼接（值始终走参数绑定，不拼进语句文本）
    line_predicate = "AND rc.business_line = $business_line" if business_line else ""
    cypher = f"""
    MATCH (rc:RootCause)-[r:INDICATES]->(ph:Phenomenon)
    WHERE ph.name IN $phenom_names {line_predicate}
    WITH rc, collect(ph.name) AS matched_phenomena, count(ph) AS matched_count
    MATCH (rc)-[:INDICATES]->(all_ph:Phenomenon)
    WITH rc, matched_phenomena, matched_count, count(all_ph) AS total_count
    ORDER BY toFloat(matched_count) / total_count DESC
    LIMIT $top_k
    RETURN
        rc.code AS code,
        rc.name AS name,
        rc.domain AS domain,
        rc.business_line AS business_line,
        rc.fix_way AS fix_way,
        rc.fix_duration AS fix_duration,
        rc.description AS description,
        matched_phenomena,
        matched_count,
        total_count,
        toFloat(matched_count) / total_count AS base_confidence
    """

    with driver.session() as session:
        result = session.run(
            cypher,
            phenom_names=phenomenon_names,
            business_line=business_line,
            top_k=top_k,
        )
        records = [record.data() for record in result]

    candidates = []
    for r in records:
        candidates.append(CandidateCause(
            code=r["code"],
            name=r["name"],
            domain=r.get("domain", ""),
            business_line=r.get("business_line", ""),
            base_confidence=round(r["base_confidence"], 4),
            confidence=round(r["base_confidence"], 4),
            matched_phenomena=r["matched_phenomena"],
            all_phenomena=[],  # 由 enrich_cause_details 补充
            fix_way=r.get("fix_way", ""),
            fix_duration=r.get("fix_duration", ""),
            verify_items="",
        ))

    return candidates


def _enrich_part_phenom_domain(candidates: list[CandidateCause]) -> dict:
    """富化①：批量查全部现象（is_core/weight）与责任域。"""
    cause_codes = [c.code for c in candidates]
    driver = get_neo4j_driver()
    phenom_cypher = """
    MATCH (rc:RootCause)-[r:INDICATES]->(ph:Phenomenon)
    WHERE rc.code IN $codes
    RETURN rc.code AS code, collect({name: ph.name, is_core: r.is_core, weight: r.weight}) AS phenomena
    """
    domain_cypher = """
    MATCH (rc:RootCause)-[:BELONGS_TO]->(od:OwnerDomain)
    WHERE rc.code IN $codes
    RETURN rc.code AS code, od.name AS domain
    """
    with driver.session() as session:
        phenom_map = {r["code"]: r["phenomena"] for r in session.run(phenom_cypher, codes=cause_codes)}
        domain_map = {r["code"]: r["domain"] for r in session.run(domain_cypher, codes=cause_codes)}
    return {"phenom": phenom_map, "domain": domain_map}


def _enrich_part_dtc(candidates: list[CandidateCause]) -> dict[str, list[str]]:
    """富化②：根因的 DTC 列表（rc.dtc 属性）。失败返回空（可选项）。"""
    cause_codes = [c.code for c in candidates]
    driver = get_neo4j_driver()
    cypher = """
    MATCH (rc:RootCause)
    WHERE rc.code IN $codes AND rc.dtc IS NOT NULL
    RETURN rc.code AS code, rc.dtc AS dtc
    """
    try:
        with driver.session() as session:
            dtc_map = {}
            for r in session.run(cypher, codes=cause_codes):
                dtc_val = r["dtc"]
                if isinstance(dtc_val, list):
                    dtc_map[r["code"]] = dtc_val
                elif isinstance(dtc_val, str):
                    dtc_map[r["code"]] = [x.strip() for x in dtc_val.split(",") if x.strip()]
            return dtc_map
    except Exception:
        return {}


def _enrich_part_verify_items(candidates: list[CandidateCause]) -> dict[str, str]:
    """富化③：验证项（PG，root_causes 受 RLS 管控须带作用域）。失败返回空。"""
    cause_codes = [c.code for c in candidates]
    try:
        import psycopg2
        from src.core.config import get_settings
        from src.infra.pg_scope import apply_scope
        s_ = get_settings()
        conn = psycopg2.connect(
            host=s_.DB_HOST, port=s_.DB_PORT,
            user=s_.DB_USER, password=s_.DB_PASSWORD, dbname=s_.DB_NAME,
        )
        apply_scope(conn)
        try:
            cur = conn.cursor()
            placeholders = ",".join(["%s"] * len(cause_codes))
            cur.execute(
                f"SELECT code, verify_items FROM root_causes WHERE code IN ({placeholders})",
                cause_codes,
            )
            rows = {r[0]: r[1] for r in cur.fetchall()}
            cur.close()
        finally:
            conn.close()
        return rows
    except Exception as e:
        logger.warning(f"[GRAPH-QUERIES] verify_items 富化失败（可选项）: {e}")
        return {}


def _enrich_part_located_in(cause_codes: list[str]) -> dict[str, list[dict]]:
    return query_located_in(cause_codes)


def _enrich_part_co_occurs(cause_codes: list[str]) -> dict[str, list[dict]]:
    return query_co_occurs_with(cause_codes)


def _apply_enrich_parts(
    candidates: list[CandidateCause], pd: dict, dtc_map: dict,
    verify_map: dict, located_in: dict, co_occurs: dict,
) -> list[CandidateCause]:
    """把五个富化部分的结果合并回候选（纯函数，sync/async 版共用）。"""
    phenom_map, domain_map = pd["phenom"], pd["domain"]
    for c in candidates:
        all_ph = phenom_map.get(c.code, [])
        c.all_phenomena = [p["name"] for p in all_ph]
        # 现象名 → INDICATES.weight（反馈回写实时值），置信度评分用作证据调制（C1）
        c.phenomena_weight = {
            p["name"]: float(p.get("weight") or 0.0) for p in all_ph if p.get("weight") is not None
        }
        c.is_core_match = any(
            p["is_core"] and p["name"] in c.matched_phenomena for p in all_ph
        )
        c.domain = domain_map.get(c.code, c.domain)
        c.dtc_matched = dtc_map.get(c.code, [])
        if c.code in verify_map:
            c.verify_items = verify_map[c.code] or ""
        if c.code in located_in:
            c.related_config_items = located_in[c.code]
        if c.code in co_occurs:
            c.related_causes = [f"{r['name']}({r['domain']})" for r in co_occurs[c.code]]
    return candidates


def enrich_cause_details(candidates: list[CandidateCause]) -> list[CandidateCause]:
    """补充候选根因详情（串行版，兼容旧调用方）。"""
    if not candidates:
        return candidates
    codes = [c.code for c in candidates]
    return _apply_enrich_parts(
        candidates,
        _enrich_part_phenom_domain(candidates),
        _enrich_part_dtc(candidates),
        _enrich_part_verify_items(candidates),
        _enrich_part_located_in(codes),
        _enrich_part_co_occurs(codes),
    )


async def enrich_cause_details_async(candidates: list[CandidateCause]) -> list[CandidateCause]:
    """并行版：五个富化查询相互独立，gather 并行。

    原串行实现 = 5 次顺序往返（Neo4j×3 + PG×1 + 关系查询×1）；
    并行后耗时 ≈ 最长一路。分诊节点③的固定开销从"5 跳"压到"1 跳"。
    """
    if not candidates:
        return candidates
    codes = [c.code for c in candidates]
    pd, dtc_map, verify_map, located_in, co_occurs = await asyncio.gather(
        asyncio.to_thread(_enrich_part_phenom_domain, candidates),
        asyncio.to_thread(_enrich_part_dtc, candidates),
        asyncio.to_thread(_enrich_part_verify_items, candidates),
        asyncio.to_thread(_enrich_part_located_in, codes),
        asyncio.to_thread(_enrich_part_co_occurs, codes),
    )
    return _apply_enrich_parts(candidates, pd, dtc_map, verify_map, located_in, co_occurs)


# ════════════════════════════════════════════════════════════════════════
# Neo4j 关系查询（补充 DTC/LOCATED_IN/CO_OCCURS_WITH 关系模型）
# ════════════════════════════════════════════════════════════════════════

def query_dtc_by_relationship(cause_codes: list[str]) -> dict[str, list[str]]:
    """
    查询 DTC 码（通过 (:DTC)-[:POINTS_TO]->(:RootCause) 关系）。
    返回 {cause_code: [dtc_code, ...]} 映射。
    """
    if not cause_codes:
        return {}

    driver = get_neo4j_driver()
    cypher = """
    MATCH (dtc:DTC)-[:POINTS_TO]->(rc:RootCause)
    WHERE rc.code IN $codes
    RETURN rc.code AS cause_code, collect(dtc.code) AS dtc_codes
    """
    try:
        with driver.session() as session:
            result = session.run(cypher, codes=cause_codes)
            return {r["cause_code"]: r["dtc_codes"] for r in result}
    except Exception:
        return {}  # DTC nodes may not exist yet, fall back to property-based


def query_located_in(cause_codes: list[str]) -> dict[str, list[dict]]:
    """
    查询根因关联的配置项：(RootCause)-[:LOCATED_IN]->(ConfigItem)。
    返回 {cause_code: [{name, ci_no, module, supplier}, ...]} 映射。
    """
    if not cause_codes:
        return {}

    driver = get_neo4j_driver()
    cypher = """
    MATCH (rc:RootCause)-[:LOCATED_IN]->(ci:ConfigItem)
    WHERE rc.code IN $codes
    RETURN rc.code AS cause_code,
           collect({name: ci.name, ci_no: ci.ci_no, module: ci.module, supplier: ci.supplier}) AS config_items
    """
    try:
        with driver.session() as session:
            result = session.run(cypher, codes=cause_codes)
            return {r["cause_code"]: r["config_items"] for r in result}
    except Exception:
        return {}


def query_co_occurs_with(cause_codes: list[str]) -> dict[str, list[dict]]:
    """
    查询伴随根因：(RootCause)-[:CO_OCCURS_WITH]->(RootCause)。
    返回 {cause_code: [{code, name, domain}, ...]} 映射。
    """
    if not cause_codes:
        return {}

    driver = get_neo4j_driver()
    cypher = """
    MATCH (rc:RootCause)-[:CO_OCCURS_WITH]->(related:RootCause)
    WHERE rc.code IN $codes
    RETURN rc.code AS cause_code,
           collect({code: related.code, name: related.name, domain: related.domain}) AS related_causes
    """
    try:
        with driver.session() as session:
            result = session.run(cypher, codes=cause_codes)
            return {r["cause_code"]: r["related_causes"] for r in result}
    except Exception:
        return {}
