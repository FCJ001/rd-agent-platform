"""Neo4j Cypher 查询函数。参考天宫医疗版 neo4j_queries.py。"""

from src.agents.triage.state import CandidateCause
from src.infra.neo4j_client import get_neo4j_driver


def query_causes_by_phenomena(
    phenomenon_names: list[str],
    top_k: int = 10,
) -> list[CandidateCause]:
    """
    根据已确认现象名，从 Neo4j 查询候选根因。
    按基础置信度（命中现象数 / 该根因总现象数）降序，取 Top K。
    """
    if not phenomenon_names:
        return []

    driver = get_neo4j_driver()

    cypher = """
    MATCH (rc:RootCause)-[r:INDICATES]->(ph:Phenomenon)
    WHERE ph.name IN $phenom_names
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
        result = session.run(cypher, phenom_names=phenomenon_names, top_k=top_k)
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


def enrich_cause_details(candidates: list[CandidateCause]) -> list[CandidateCause]:
    """补充候选根因的全部现象、验证项、is_core 信息（批量查询）。"""
    if not candidates:
        return candidates

    cause_codes = [c.code for c in candidates]
    driver = get_neo4j_driver()

    # 批量查全部现象 + is_core/weight
    phenom_cypher = """
    MATCH (rc:RootCause)-[r:INDICATES]->(ph:Phenomenon)
    WHERE rc.code IN $codes
    RETURN rc.code AS code, collect({name: ph.name, is_core: r.is_core, weight: r.weight}) AS phenomena
    """
    # 批量查责任域
    domain_cypher = """
    MATCH (rc:RootCause)-[:BELONGS_TO]->(od:OwnerDomain)
    WHERE rc.code IN $codes
    RETURN rc.code AS code, od.name AS domain
    """

    with driver.session() as session:
        phenom_result = session.run(phenom_cypher, codes=cause_codes)
        phenom_map = {r["code"]: r["phenomena"] for r in phenom_result}

        domain_result = session.run(domain_cypher, codes=cause_codes)
        domain_map = {r["code"]: r["domain"] for r in domain_result}

    for c in candidates:
        all_ph = phenom_map.get(c.code, [])
        c.all_phenomena = [p["name"] for p in all_ph]
        c.is_core_match = any(
            p["is_core"] and p["name"] in c.matched_phenomena
            for p in all_ph
        )
        c.domain = domain_map.get(c.code, c.domain)

    # Load dtc codes from Neo4j (stored as rc.dtc list on RootCause nodes)
    dtc_cypher = """
    MATCH (rc:RootCause)
    WHERE rc.code IN $codes AND rc.dtc IS NOT NULL
    RETURN rc.code AS code, rc.dtc AS dtc
    """
    try:
        with driver.session() as session:
            dtc_result = session.run(dtc_cypher, codes=cause_codes)
            dtc_map = {}
            for r in dtc_result:
                dtc_val = r["dtc"]
                if isinstance(dtc_val, list):
                    dtc_map[r["code"]] = dtc_val
                elif isinstance(dtc_val, str):
                    dtc_map[r["code"]] = [x.strip() for x in dtc_val.split(",") if x.strip()]
            for c in candidates:
                c.dtc_matched = dtc_map.get(c.code, [])
    except Exception:
        pass  # dtc matching is optional

    # Load verify_items from PostgreSQL (batch query via psycopg2)
    try:
        import psycopg2
        from src.core.config import get_settings
        s = get_settings()
        conn = psycopg2.connect(
            host=s.DB_HOST, port=s.DB_PORT,
            user=s.DB_USER, password=s.DB_PASSWORD, dbname=s.DB_NAME,
        )
        cur = conn.cursor()
        placeholders = ",".join(["%s"] * len(cause_codes))
        cur.execute(
            f"SELECT code, verify_items FROM root_causes WHERE code IN ({placeholders})",
            cause_codes,
        )
        rows = {r[0]: r[1] for r in cur.fetchall()}
        cur.close()
        conn.close()
        for c in candidates:
            if c.code in rows:
                c.verify_items = rows[c.code] or ""
    except Exception:
        pass  # verify_items is optional enrichment

    return candidates
