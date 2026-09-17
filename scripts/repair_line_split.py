#!/usr/bin/env python
# ============================================================
# 存量修复：把跨业务线塌缩的现象 / DTC 拆回各自业务线
#
# 病根：现象名和 DTC 码是跨业务线共享的词汇表（实测 26 个现象名有 13 个跨线、
#   U0155 在 ev/ia 都出现），而旧代码按 name / code 单键 MERGE / 去重，导致：
#     · Neo4j：两条线共用同一个节点，business_line 属性被后写入者覆盖
#     · PG   ：只能存一行，business_line / code 被后写入者覆盖
#
# 修复依据（这是为什么本脚本能还原真相，而不是猜）：
#   · Neo4j：RootCause.code 带线前缀且全局唯一 → 沿 INDICATES 反查边两端的线
#   · PG   ：root_causes.business_line 是权威的 → 沿 cause_phenomena 反查
#
# 幂等：拆分后每条边/关联都落在正确的线上，重复执行不再命中歧义行。
#
# 用法：
#   python scripts/repair_line_split.py            # 干跑，只报告不写
#   python scripts/repair_line_split.py --apply    # 实际执行
#   python scripts/repair_line_split.py --apply --only pg|neo4j
# ============================================================

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Neo4j：Phenomenon / DTC 节点拆分
# ============================================================

_AMBIGUOUS_PHENOMENA = """
MATCH (ph:Phenomenon)<-[:INDICATES]-(rc:RootCause)
WITH ph, collect(DISTINCT rc.business_line) AS lines
WHERE size(lines) > 1
RETURN elementId(ph) AS eid, ph.name AS name, ph.business_line AS current, lines
"""

_AMBIGUOUS_DTC = """
MATCH (d:DTC)<-[:POINTS_TO]-(rc:RootCause)
WITH d, collect(DISTINCT rc.business_line) AS lines
WHERE size(lines) > 1
RETURN elementId(d) AS eid, d.code AS key, d.business_line AS current, lines
"""

_MOVE_PHENOMENON_EDGES = """
MATCH (ph:Phenomenon) WHERE elementId(ph) = $eid
MATCH (ph)<-[r:INDICATES]-(rc:RootCause {business_line: $line})
MERGE (target:Phenomenon {name: $key, business_line: $line})
CREATE (rc)-[:INDICATES {weight: r.weight, is_core: r.is_core}]->(target)
DELETE r
RETURN count(*) AS moved
"""

_MOVE_DTC_EDGES = """
MATCH (d:DTC) WHERE elementId(d) = $eid
MATCH (d)<-[r:POINTS_TO]-(rc:RootCause {business_line: $line})
MERGE (target:DTC {code: $key, business_line: $line})
CREATE (rc)-[:POINTS_TO]->(target)
DELETE r
RETURN count(*) AS moved
"""

_ORPHAN_EDGES = "MATCH (n) WHERE elementId(n) = $eid RETURN size([(n)--() | 1]) AS deg"

_DELETE_NODE = "MATCH (n) WHERE elementId(n) = $eid DELETE n"


def _repair_neo4j(driver, apply: bool) -> int:
    """返回命中的塌缩节点数（干跑与执行都计数，便于比对）。"""
    hits = 0
    with driver.session() as s:
        for label, find_query, move_query, key_field in (
            ("Phenomenon", _AMBIGUOUS_PHENOMENA, _MOVE_PHENOMENON_EDGES, "name"),
            ("DTC", _AMBIGUOUS_DTC, _MOVE_DTC_EDGES, "key"),
        ):
            rows = s.run(find_query).data()
            logger.info(f"[NEO4J] {label}: 发现 {len(rows)} 个跨线塌缩节点")
            for row in rows:
                eid, key, current, lines = row["eid"], row[key_field], row["current"], row["lines"]
                logger.info(f"[NEO4J]   {label} {key!r} 当前线={current} 实际线={lines}")
                hits += 1
                if not apply:
                    continue
                for line in lines:
                    if line == current:
                        continue  # 当前节点就是这条线的，留用
                    moved = s.run(move_query, eid=eid, key=key, line=line).data()
                    logger.info(f"[NEO4J]     迁往 {line}: {moved[0]['moved'] if moved else 0} 条边")
                # 原节点若已无任何边（归属线不在实际线集合里），删掉
                deg = s.run(_ORPHAN_EDGES, eid=eid).data()
                if deg and deg[0]["deg"] == 0:
                    s.run(_DELETE_NODE, eid=eid)
                    logger.info("[NEO4J]     原节点已无关联，删除")
    return hits


# ============================================================
# PG：phenomena / dtc_codes 行拆分
# ============================================================

_AMBIGUOUS_PHENOMENA_PG = """
SELECT p.id, p.name, p.business_line,
       array_agg(DISTINCT rc.business_line) AS lines
FROM phenomena p
JOIN cause_phenomena cp ON cp.phenomenon_id = p.id
JOIN root_causes rc ON rc.id = cp.cause_id
GROUP BY p.id, p.name, p.business_line
HAVING COUNT(DISTINCT rc.business_line) > 1
"""

_AMBIGUOUS_DTC_PG = """
SELECT d.id, d.code, d.business_line, d.system,
       array_agg(DISTINCT rc.business_line) AS lines
FROM dtc_codes d
JOIN cause_dtc cd ON cd.dtc_id = d.id
JOIN root_causes rc ON rc.id = cd.cause_id
GROUP BY d.id, d.code, d.business_line, d.system
HAVING COUNT(DISTINCT rc.business_line) > 1
"""


def _unique_code(cur, table: str, line: str, base_id: int) -> str:
    """生成一个不冲突的 code（code 仍是全局唯一列）。"""
    prefix = "PH" if table == "phenomena" else "DTC"
    candidate = f"{prefix}-{line.upper()}-{base_id}"
    n = 0
    while True:
        cur.execute(f"SELECT 1 FROM {table} WHERE code = %s", (candidate,))
        if not cur.fetchone():
            return candidate
        n += 1
        candidate = f"{prefix}-{line.upper()}-{base_id}-{n}"


def _repair_pg(conn, apply: bool) -> int:
    """返回命中的塌缩行数（干跑与执行都计数，便于比对）。"""
    hits = 0
    cur = conn.cursor()

    # ── 现象 ──
    cur.execute(_AMBIGUOUS_PHENOMENA_PG)
    rows = cur.fetchall()
    logger.info(f"[PG] phenomena: 发现 {len(rows)} 行跨线塌缩")
    for pid, name, current, lines in rows:
        lines = list(lines)
        logger.info(f"[PG]   {name!r} 当前线={current} 实际线={lines}")
        hits += 1
        if not apply:
            continue
        for line in lines:
            if line == current:
                continue
            new_code = _unique_code(cur, "phenomena", line, pid)
            cur.execute(
                """INSERT INTO phenomena (code, name, business_line, colloquial)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (new_code, name, line, name),
            )
            new_id = cur.fetchone()[0]
            # 把该线的根因关联迁到新行
            cur.execute(
                """UPDATE cause_phenomena SET phenomenon_id = %s
                   WHERE phenomenon_id = %s
                     AND cause_id IN (SELECT id FROM root_causes WHERE business_line = %s)""",
                (new_id, pid, line),
            )
            logger.info(f"[PG]     + 新建 id={new_id} code={new_code}，迁移 {cur.rowcount} 条关联")

    # ── DTC ──
    cur.execute(_AMBIGUOUS_DTC_PG)
    rows = cur.fetchall()
    logger.info(f"[PG] dtc_codes: 发现 {len(rows)} 行跨线塌缩")
    for did, code, current, system, lines in rows:
        lines = list(lines)
        logger.info(f"[PG]   {code!r} 当前线={current} 实际线={lines}")
        hits += 1
        if not apply:
            continue
        for line in lines:
            if line == current:
                continue
            new_code = _unique_code(cur, "dtc_codes", line, did)
            cur.execute(
                """INSERT INTO dtc_codes (code, system, business_line)
                   VALUES (%s, %s, %s) RETURNING id""",
                (new_code, system, line),
            )
            new_id = cur.fetchone()[0]
            cur.execute(
                """UPDATE cause_dtc SET dtc_id = %s
                   WHERE dtc_id = %s
                     AND cause_id IN (SELECT id FROM root_causes WHERE business_line = %s)""",
                (new_id, did, line),
            )
            logger.info(f"[PG]     + 新建 id={new_id} code={new_code}，迁移 {cur.rowcount} 条关联")

    if apply:
        conn.commit()
    cur.close()
    return hits


# ============================================================

def main():
    ap = argparse.ArgumentParser(description="拆分跨业务线塌缩的现象 / DTC")
    ap.add_argument("--apply", action="store_true", help="实际写库（默认只报告）")
    ap.add_argument("--only", choices=["pg", "neo4j"], help="只修一端")
    args = ap.parse_args()

    if not args.apply:
        logger.info("干跑模式（只报告）。确认无误后加 --apply 执行。")

    total = 0

    if args.only != "neo4j":
        import psycopg2
        s = get_settings()
        conn = psycopg2.connect(
            host=s.DB_HOST, port=s.DB_PORT,
            user=s.DB_USER, password=s.DB_PASSWORD, dbname=s.DB_NAME,
        )
        try:
            total += _repair_pg(conn, args.apply)
        finally:
            conn.close()

    if args.only != "pg":
        from neo4j import GraphDatabase
        s = get_settings()
        driver = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))
        try:
            total += _repair_neo4j(driver, args.apply)
        finally:
            driver.close()

    logger.info(f"完成：{total} 个塌缩项{'已修复' if args.apply else '待修复（干跑）'}")


if __name__ == "__main__":
    main()
