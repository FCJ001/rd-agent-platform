# ============================================================
# Neo4j 图谱数据导入脚本
# 将 alm_kg.json + alm_mirror.json 导入 Neo4j
#
# 节点类型（7 种）：
#   OwnerDomain / RootCause / Phenomenon / DTC / ConfigItem / Baseline / Requirement
#
# 关系类型（10 种）：
#   BELONGS_TO      RootCause → OwnerDomain
#   INDICATES       RootCause → Phenomenon (weight, is_core)
#   POINTS_TO       DTC → RootCause
#   LOCATED_IN      RootCause → ConfigItem
#   CO_OCCURS_WITH  RootCause → RootCause
#   DEPENDS_ON      ConfigItem → ConfigItem
#   ASSIGNED_TO     Requirement → Baseline
#   AFFECTS         Requirement → ConfigItem
#   TARGETS         ChangeRequest → Baseline
#   TRIGGERED_BY    ChangeRequest → Issue
#
# ★ 全部使用 MERGE，脚本天然幂等
#
# 前置：gen_alm_kg.py + gen_alm_mirror.py + init_postgres.py 已跑
# 用法: cd rd-agent-platform && python scripts/init_neo4j.py
# ============================================================

import json
import logging
import sys
from pathlib import Path

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _clear(driver):
    """清空图谱（开发期用，生产环境不要）"""
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    logger.info("图谱已清空")


def import_kg(driver):
    """导入诊断图谱：OwnerDomain / RootCause / Phenomenon + BELONGS_TO / INDICATES"""
    kg_path = DATA_DIR / "alm_kg.json"
    if not kg_path.exists():
        logger.warning(f"诊断图谱文件不存在: {kg_path}")
        return
    causes = [json.loads(line) for line in kg_path.read_text(encoding="utf-8").strip().split("\n") if line]

    with driver.session() as s:
        # OwnerDomain 节点
        from scripts.seed_domains import OWNER_DOMAINS
        for d in OWNER_DOMAINS:
            s.run(
                """MERGE (od:OwnerDomain {name: $name})
                   SET od.business_line = $line, od.description = $desc""",
                name=d["name"], line=d["business_line"], desc=d.get("description", ""),
            )

        cause_count = phenom_count = cp_count = dtc_count = dtc_rel_count = 0
        for c in causes:
            # RootCause 节点
            s.run(
                """MERGE (rc:RootCause {code: $code})
                   SET rc.name = $name, rc.domain = $domain, rc.business_line = $line,
                       rc.description = $desc, rc.fix_way = $fix_way, rc.fix_duration = $fix_duration,
                       rc.dtc = $dtc""",
                code=c["code"], name=c["name"], domain=c.get("domain"), line=c.get("business_line"),
                desc=c.get("description", ""), fix_way=c.get("fix_way", ""),
                fix_duration=c.get("fix_duration", ""),
                dtc=c.get("dtc", []),
            )
            cause_count += 1

            # DTC nodes + POINTS_TO relationship（从 rc.dtc 属性同步）
            for dtc_code in c.get("dtc", []):
                s.run(
                    """MERGE (d:DTC {code: $code})
                       SET d.system = $system, d.business_line = $line""",
                    code=dtc_code, system="", line=c.get("business_line", "ev"),
                )
                s.run(
                    """MATCH (d:DTC {code: $dtc_code})
                       MATCH (rc:RootCause {code: $cause_code})
                       MERGE (d)-[:POINTS_TO]->(rc)""",
                    dtc_code=dtc_code, cause_code=c["code"],
                )
                dtc_count += 1
                dtc_rel_count += 1

            # BELONGS_TO: RootCause → OwnerDomain
            if c.get("domain"):
                s.run(
                    """MATCH (rc:RootCause {code: $code})
                       MATCH (od:OwnerDomain {name: $domain})
                       MERGE (rc)-[:BELONGS_TO]->(od)""",
                    code=c["code"], domain=c["domain"],
                )

            # Phenomenon + INDICATES
            for pm in c.get("phenomena_meta", []):
                s.run(
                    """MERGE (ph:Phenomenon {name: $name})
                       SET ph.business_line = $line""",
                    name=pm["name"], line=c.get("business_line", "ev"),
                )
                s.run(
                    """MATCH (rc:RootCause {code: $code})
                       MATCH (ph:Phenomenon {name: $name})
                       MERGE (rc)-[r:INDICATES]->(ph)
                       SET r.weight = $weight, r.is_core = $is_core""",
                    code=c["code"], name=pm["name"],
                    weight=pm.get("weight", 1.0), is_core=pm.get("is_core", False),
                )
                phenom_count += 1
                cp_count += 1

    logger.info(f"Neo4j KG: {cause_count} RootCause / {phenom_count} Phenomenon / {dtc_count} DTC / {cp_count} INDICATES / {dtc_rel_count} POINTS_TO")


def import_mirror(driver):
    """导入 ALM 镜像：ConfigItem / Baseline / Requirement + 依赖关系"""
    mirror_path = DATA_DIR / "alm_mirror.json"
    if not mirror_path.exists():
        logger.warning(f"ALM 镜像文件不存在: {mirror_path}")
        return
    mirror = json.loads(mirror_path.read_text(encoding="utf-8"))

    with driver.session() as s:
        # ConfigItem 节点 + DEPENDS_ON 关系
        ci_count = dep_count = 0
        for ci in mirror.get("config_items", []):
            s.run(
                """MERGE (ci:ConfigItem {ci_no: $ci_no})
                   SET ci.name = $name, ci.module = $module, ci.category = $category,
                       ci.business_line = $line, ci.is_safety_related = $safety,
                       ci.lifecycle_status = $status""",
                ci_no=ci["ci_no"], name=ci["name"], module=ci.get("module"),
                category=ci.get("category"), line=ci.get("business_line"),
                safety=ci.get("is_safety_related", False),
                status=ci.get("lifecycle_status", "active"),
            )
            ci_count += 1
            for dep_no in ci.get("depends_on", []):
                s.run(
                    """MATCH (a:ConfigItem {ci_no: $ci_no})
                       MATCH (b:ConfigItem {ci_no: $dep_no})
                       MERGE (a)-[:DEPENDS_ON]->(b)""",
                    ci_no=ci["ci_no"], dep_no=dep_no,
                )
                dep_count += 1

        logger.info(f"Neo4j CI: {ci_count} 节点 / {dep_count} DEPENDS_ON")

        # Baseline 节点
        bl_count = 0
        for bl in mirror.get("baselines", []):
            s.run(
                """MERGE (bl:Baseline {baseline_no: $no})
                   SET bl.name = $name, bl.business_line = $line,
                       bl.is_frozen = $frozen, bl.freeze_date = $freeze_date""",
                no=bl["baseline_no"], name=bl["name"], line=bl.get("business_line"),
                frozen=bl.get("is_frozen", False), freeze_date=bl.get("freeze_date"),
            )
            bl_count += 1
        logger.info(f"Neo4j Baseline: {bl_count} 节点")

        # Requirement 节点 + ASSIGNED_TO + AFFECTS 关系
        req_count = assigned = affects = 0
        for req in mirror.get("requirements", []):
            s.run(
                """MERGE (r:Requirement {req_no: $no})
                   SET r.title = $title, r.business_line = $line,
                       r.priority = $priority, r.status = $status""",
                no=req["req_no"], title=req.get("title", ""), line=req.get("business_line"),
                priority=req.get("priority"), status=req.get("status"),
            )
            req_count += 1

            if req.get("baseline_no"):
                s.run(
                    """MATCH (r:Requirement {req_no: $req_no})
                       MATCH (bl:Baseline {baseline_no: $bl_no})
                       MERGE (r)-[:ASSIGNED_TO]->(bl)""",
                    req_no=req["req_no"], bl_no=req["baseline_no"],
                )
                assigned += 1

            for ci_no in req.get("affected_ci_nos", []):
                s.run(
                    """MATCH (r:Requirement {req_no: $req_no})
                       MATCH (ci:ConfigItem {ci_no: $ci_no})
                       MERGE (r)-[:AFFECTS]->(ci)""",
                    req_no=req["req_no"], ci_no=ci_no,
                )
                affects += 1

        logger.info(f"Neo4j Req: {req_count} 节点 / {assigned} ASSIGNED_TO / {affects} AFFECTS")

        # ChangeRequest 节点 + TARGETS + TRIGGERED_BY 关系
        cr_count = targets = triggered = 0
        for cr in mirror.get("change_requests", []):
            s.run(
                """MERGE (cr:ChangeRequest {cr_no: $no})
                   SET cr.title = $title, cr.business_line = $line, cr.status = $status""",
                no=cr["cr_no"], title=cr.get("title", ""), line=cr.get("business_line"),
                status=cr.get("status"),
            )
            cr_count += 1

            if cr.get("target_baseline_no"):
                s.run(
                    """MATCH (cr:ChangeRequest {cr_no: $cr_no})
                       MATCH (bl:Baseline {baseline_no: $bl_no})
                       MERGE (cr)-[:TARGETS]->(bl)""",
                    cr_no=cr["cr_no"], bl_no=cr["target_baseline_no"],
                )
                targets += 1

            if cr.get("source_issue_no"):
                s.run(
                    """MATCH (cr:ChangeRequest {cr_no: $cr_no})
                       MATCH (iss:Issue {issue_no: $iss_no})
                       MERGE (cr)-[:TRIGGERED_BY]->(iss)""",
                    cr_no=cr["cr_no"], iss_no=cr["source_issue_no"],
                )
                triggered += 1

        logger.info(f"Neo4j CR: {cr_count} 节点 / {targets} TARGETS / {triggered} TRIGGERED_BY")


def import_issues(driver):
    """导入已有问题单节点（步 1 种子里的 24 条）"""
    import psycopg2
    settings = get_settings()
    conn = psycopg2.connect(
        host=settings.DB_HOST, port=settings.DB_PORT,
        user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
    )
    cur = conn.cursor()
    cur.execute("SELECT issue_no, title, source, business_line, severity, status FROM alm_issues")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    with driver.session() as s:
        for issue_no, title, source, line, severity, status in rows:
            s.run(
                """MERGE (iss:Issue {issue_no: $no})
                   SET iss.title = $title, iss.source = $source,
                       iss.business_line = $line, iss.severity = $severity, iss.status = $status""",
                no=issue_no, title=title, source=source, line=line,
                severity=severity, status=status,
            )
    logger.info(f"Neo4j Issue: {len(rows)} 个节点")


def import_graph_extras(driver):
    """
    补充 LOCATED_IN（RootCause → ConfigItem）和 CO_OCCURS_WITH（RootCause → RootCause）关系。

    LOCATED_IN: 按 business_line 将根因关联到同线配置项（前 3 个）。
    CO_OCCURS_WITH: 共享 ≥ 1 个现象的根因对，权重 = 共享现象数 / 总现象数。
    """
    located_count = 0
    co_occur_count = 0

    with driver.session() as s:
        # ── LOCATED_IN: 同 business_line 的 RootCause ↔ ConfigItem ──
        result = s.run("""
            MATCH (rc:RootCause)
            MATCH (ci:ConfigItem)
            WHERE rc.business_line = ci.business_line
            WITH rc, ci
            ORDER BY rc.code, ci.ci_no
            WITH rc, collect(ci)[0..3] AS top_cis
            UNWIND top_cis AS ci
            MERGE (rc)-[:LOCATED_IN]->(ci)
            RETURN count(*) AS cnt
        """)
        located_count = result.single().get("cnt", 0)

        # ── CO_OCCURS_WITH: 共享 ≥ 1 个现象的根因对 ──
        result = s.run("""
            MATCH (rc1:RootCause)-[:INDICATES]->(ph:Phenomenon)<-[:INDICATES]-(rc2:RootCause)
            WHERE rc1.code < rc2.code
            WITH rc1, rc2, count(ph) AS shared_phenomena
            MATCH (rc1)-[:INDICATES]->(ph1:Phenomenon)
            WITH rc1, rc2, shared_phenomena, count(ph1) AS total1
            MATCH (rc2)-[:INDICATES]->(ph2:Phenomenon)
            WITH rc1, rc2, shared_phenomena, total1, count(ph2) AS total2
            WITH rc1, rc2, shared_phenomena,
                 toFloat(shared_phenomena) / (total1 + total2 - shared_phenomena) AS weight
            MERGE (rc1)-[:CO_OCCURS_WITH {weight: weight}]->(rc2)
            RETURN count(*) AS cnt
        """)
        co_occur_count = result.single().get("cnt", 0)

    logger.info(f"Neo4j Extras: {located_count} LOCATED_IN / {co_occur_count} CO_OCCURS_WITH")


def main():
    settings = get_settings()
    # Neo4j 的 verify=True 需要 .local TLD hostname，本地测试关闭
    import ssl
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
    except Exception:
        logger.warning("Neo4j 连接验证失败，尝试继续...")

    logger.info("清空旧图谱...")
    _clear(driver)

    logger.info("1/3 导入诊断图谱...")
    import_kg(driver)

    logger.info("2/3 导入 ALM 镜像...")
    import_mirror(driver)

    logger.info("3/4 导入问题单...")
    import_issues(driver)

    logger.info("4/4 补充 LOCATED_IN / CO_OCCURS_WITH 关系...")
    import_graph_extras(driver)

    driver.close()
    logger.info("Neo4j 导入完成")


if __name__ == "__main__":
    main()
