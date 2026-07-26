# ============================================================
# PostgreSQL 数据导入脚本
# 将 gen_alm_kg.py 和 gen_alm_mirror.py 的产出导入 PG
#
# ★ 责任域从 seed_domains.py 导入（单一来源），不重复插入
# ★ 全部使用 ON CONFLICT DO NOTHING/UPDATE，脚本天然幂等
#
# 前置：gen_alm_kg.py + gen_alm_mirror.py 已跑
# 用法: cd rd-agent-platform && python scripts/init_postgres.py
# ============================================================

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _upsert(cur, table: str, key_col: str, key_val, data: dict) -> bool:
    """返回 True 表示有插入/更新，False 表示跳过"""
    columns = list(data.keys())
    if key_col not in columns:
        columns.append(key_col)
        data[key_col] = key_val
    placeholders = [f"%({c})s" for c in columns]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != key_col)
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(placeholders)}) "
        f"ON CONFLICT ({key_col}) DO UPDATE SET {set_clause}"
    )
    cur.execute(sql, data)
    return cur.rowcount > 0


def seed():
    import psycopg2

    settings = get_settings()
    conn = psycopg2.connect(
        host=settings.DB_HOST, port=settings.DB_PORT,
        user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
    )
    cur = conn.cursor()

    # ============ 0. 责任域：从唯一来源 scripts/seed_domains.py import ============
    from scripts.seed_domains import OWNER_DOMAINS

    domain_to_id: dict[str, int] = {}
    for d in OWNER_DOMAINS:
        _upsert(cur, "owner_domains", "name", d["name"], {
            "business_line": d["business_line"],
            "description": d["description"],
            "owner_name": d.get("owner_name"),
        })
        cur.execute("SELECT id FROM owner_domains WHERE name = %s", (d["name"],))
        row = cur.fetchone()
        if row:
            domain_to_id[d["name"]] = row[0]
    conn.commit()
    logger.info(f"责任域: {len(domain_to_id)} 个")

    # ============ 1. 诊断图谱 (alm_kg.json) ============
    kg_path = DATA_DIR / "alm_kg.json"
    phenom_to_id: dict[str, int] = {}
    cause_to_id: dict[str, int] = {}

    if kg_path.exists():
        causes = [json.loads(line) for line in kg_path.read_text(encoding="utf-8").strip().split("\n") if line]

        # 1a. RootCauses
        for c in causes:
            _upsert(cur, "root_causes", "code", c["code"], {
                "name": c["name"],
                "domain_id": domain_to_id.get(c.get("domain")),
                "business_line": c.get("business_line", "ev"),
                "description": c.get("description"),
                "cause": c.get("cause"),
                "prevent": c.get("prevent"),
                "fix_way": c.get("fix_way"),
                "fix_duration": c.get("fix_duration"),
                "fix_success_rate": c.get("fix_success_rate"),
                "easy_hit": c.get("easy_hit"),
                "cost_money": c.get("cost_money"),
                "verify_items": c.get("verify_items"),
            })
            cur.execute("SELECT id FROM root_causes WHERE code = %s", (c["code"],))
            row = cur.fetchone()
            if row:
                cause_to_id[c["code"]] = row[0]
        conn.commit()

        # 1b. Phenomena
        seen_phenomena: set[str] = set()
        for c in causes:
            for p in c.get("phenomena", []):
                if p in seen_phenomena:
                    continue
                seen_phenomena.add(p)
                _upsert(cur, "phenomena", "name", p, {
                    "code": f"PH-{c.get('business_line', 'EV').upper()}-{len(phenom_to_id)+1:03d}",
                    "business_line": c.get("business_line", "ev"),
                    "colloquial": p,
                })
                cur.execute("SELECT id FROM phenomena WHERE name = %s", (p,))
                row = cur.fetchone()
                if row:
                    phenom_to_id[p] = row[0]
        conn.commit()

        # 1c. CausePhenomenon (多对多)
        for c in causes:
            cause_id = cause_to_id.get(c["code"])
            if not cause_id:
                continue
            for pm in c.get("phenomena_meta", []):
                phenom_id = phenom_to_id.get(pm["name"])
                if not phenom_id:
                    continue
                cur.execute(
                    """INSERT INTO cause_phenomena (cause_id, phenomenon_id, weight, is_core)
                       VALUES (%s,%s,%s,%s)
                       ON CONFLICT (cause_id, phenomenon_id) DO UPDATE
                       SET weight = EXCLUDED.weight, is_core = EXCLUDED.is_core""",
                    (cause_id, phenom_id, pm.get("weight", 1.0), pm.get("is_core", False)),
                )
        conn.commit()
        logger.info(f"根因: {len(cause_to_id)} | 现象: {len(phenom_to_id)} | 关联: {sum(1 for c in causes for _ in c.get('phenomena_meta', []))}")

    # ============ 2. ALM 镜像 (alm_mirror.json) ============
    mirror_path = DATA_DIR / "alm_mirror.json"
    if mirror_path.exists():
        mirror = json.loads(mirror_path.read_text(encoding="utf-8"))

        ci_map: dict[str, int] = {}
        baseline_map: dict[str, int] = {}

        # 2a. Config Items
        for ci in mirror.get("config_items", []):
            _upsert(cur, "alm_config_items", "ci_no", ci["ci_no"], {
                "name": ci["name"],
                "alias": ci.get("alias"),
                "category": ci.get("category", "software"),
                "module": ci.get("module"),
                "supplier": ci.get("supplier"),
                "part_number": ci.get("part_number"),
                "sw_version": ci.get("sw_version"),
                "is_safety_related": ci.get("is_safety_related", False),
                "lifecycle_status": ci.get("lifecycle_status", "active"),
                "business_line": ci.get("business_line", "ev"),
            })
            cur.execute("SELECT id FROM alm_config_items WHERE ci_no = %s", (ci["ci_no"],))
            row = cur.fetchone()
            if row:
                ci_map[ci["ci_no"]] = row[0]
        conn.commit()
        logger.info(f"配置项: {len(ci_map)}")

        # 2b. Baselines
        for bl in mirror.get("baselines", []):
            _upsert(cur, "alm_baselines", "baseline_no", bl["baseline_no"], {
                "name": bl["name"],
                "business_line": bl.get("business_line", "ev"),
                "is_frozen": bl.get("is_frozen", False),
                "freeze_date": bl.get("freeze_date"),
                "release_date": bl.get("release_date"),
            })
            cur.execute("SELECT id FROM alm_baselines WHERE baseline_no = %s", (bl["baseline_no"],))
            row = cur.fetchone()
            if row:
                baseline_map[bl["baseline_no"]] = row[0]
        conn.commit()
        logger.info(f"基线: {len(baseline_map)}")

        # 2c. Requirements
        req_count = 0
        for req in mirror.get("requirements", []):
            _upsert(cur, "alm_requirements", "req_no", req["req_no"], {
                "title": req["title"],
                "description": req.get("description"),
                "business_line": req.get("business_line", "ev"),
                "priority": req.get("priority", "P2"),
                "status": req.get("status", "open"),
                "baseline_id": baseline_map.get(req.get("baseline_no")) if req.get("baseline_no") else None,
            })
            req_count += 1
        conn.commit()
        logger.info(f"需求: {req_count}")

        # 2d. Change Requests
        cr_count = 0
        for cr in mirror.get("change_requests", []):
            _upsert(cur, "alm_change_requests", "cr_no", cr["cr_no"], {
                "title": cr["title"],
                "reason": cr.get("reason"),
                "scope_desc": cr.get("scope_desc"),
                "business_line": cr.get("business_line", "ev"),
                "status": cr.get("status", "submitted"),
                "target_baseline_id": baseline_map.get(cr.get("target_baseline_no")) if cr.get("target_baseline_no") else None,
                "source_issue_id": None,   # 变更单 → 问题单的关联在 Neo4j 层建立，PG 这里跳过
            })
            cr_count += 1
        conn.commit()
        logger.info(f"变更单: {cr_count}")

    # ============ 3. DTC 故障码（简版，从 KG 里提取）============
    if kg_path.exists():
        causes = [json.loads(line) for line in kg_path.read_text(encoding="utf-8").strip().split("\n") if line]
        seen_dtc: set[str] = set()
        for c in causes:
            for dtc in c.get("dtc", []):
                if dtc in seen_dtc:
                    continue
                seen_dtc.add(dtc)
                line = c.get("business_line", "ev")
                _upsert(cur, "dtc_codes", "code", dtc, {
                    "system": "powertrain" if line == "ev" else "network",
                    "description_zh": dtc,
                    "business_line": line,
                })
        conn.commit()
        if seen_dtc:
            logger.info(f"DTC 故障码: {len(seen_dtc)}")

    cur.close()
    conn.close()
    logger.info("PostgreSQL 导入完成")


if __name__ == "__main__":
    seed()
