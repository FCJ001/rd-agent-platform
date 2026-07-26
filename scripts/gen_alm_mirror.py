# ============================================================
# ALM 流程数据生成脚本（规则生成，不调 LLM）
# 产出：data/raw/alm_mirror.json
#   配置项 300 / 需求 200 / 变更单 150 / 基线 20
#
# 不是随机造数，而是刻意造出【可验证的拓扑结构】：
#   1. 配置项依赖链深度 ≥ 3，让多跳 Cypher 有意义
#   2. 有已冻结基线，且确实阻塞了在途需求（杀手级查询有结果）
#   3. 有跨模块依赖，让基线冲突二级 module_match 能命中
#
# 用法: cd rd-agent-platform && python scripts/gen_alm_mirror.py
# ============================================================

import json
import logging
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RAW_DIR / "alm_mirror.json"

random.seed(42)

# ---- 模块定义 ----
MODULES = [
    ("BMS电池管理", "ev", True, "BMS"),
    ("MCU电机控制", "ev", True, "MCU"),
    ("VCU整车控制", "ev", True, "VCU"),
    ("OBC车载充电", "ev", False, "OBC"),
    ("TMS热管理", "ev", False, "TMS"),
    ("ADAS智驾", "ia", True, "ADS"),
    ("HMI座舱", "ia", False, "HMI"),
    ("TBOX车联网", "ia", False, "TBX"),
    ("OTA升级", "ia", False, "OTA"),
    ("GW网关", "ia", True, "GWY"),
]

CI_CATEGORIES = ["software", "hardware", "calibration", "doc"]
SUPPLIERS = ["博世", "大陆", "德赛西威", "均胜电子", "宁德时代", "华为", "地平线", "自研"]


def gen_config_items(n: int = 300) -> list[dict]:
    items = []
    per_module = n // len(MODULES)

    for mod_name, line, safety, prefix in MODULES:
        for i in range(1, per_module + 1):
            cat = CI_CATEGORIES[i % len(CI_CATEGORIES)]
            items.append({
                "ci_no": f"CI-{prefix}-{i:03d}",
                "name": f"{mod_name}-{cat}-{i:03d}",
                "alias": f"{prefix}{i}",
                "category": cat,
                "module": mod_name,
                "supplier": random.choice(SUPPLIERS),
                "part_number": f"{prefix}-PN-{random.randint(10000, 99999)}",
                "sw_version": f"{random.randint(1, 4)}.{random.randint(0, 20)}.{random.randint(0, 9)}",
                "is_safety_related": safety and cat in ("software", "calibration"),
                "lifecycle_status": random.choices(
                    ["dev", "active", "frozen", "obsolete"], weights=[2, 6, 1, 1]
                )[0],
                "business_line": line,
                "depends_on": [],
            })

    by_module = {mod_name: [i for i in items if i["module"] == mod_name] for mod_name, *_ in MODULES}
    gateway = [i for i in items if i["module"] == "GW网关" and i["category"] == "software"]

    for it in items:
        deps = []
        # 软件 → 同模块标定件
        if it["category"] == "software":
            same = [x for x in by_module[it["module"]] if x["category"] == "calibration"]
            if same:
                deps.append(random.choice(same)["ci_no"])
        # 标定件 → 同模块硬件
        if it["category"] == "calibration":
            hw = [x for x in by_module[it["module"]] if x["category"] == "hardware"]
            if hw:
                deps.append(random.choice(hw)["ci_no"])
        # 30% 跨模块依赖网关
        if gateway and it["module"] != "GW网关" and random.random() < 0.3:
            deps.append(random.choice(gateway)["ci_no"])
        it["depends_on"] = list(dict.fromkeys(deps))

    return items


def gen_baselines(n: int = 20) -> list[dict]:
    """ev/ia 各有 2 条已冻结基线"""
    baselines = []
    today = date.today()
    for line in ("ev", "ia"):
        for q in range(1, n // 2 + 1):
            frozen = q <= 2
            baselines.append({
                "baseline_no": f"BL-{line.upper()}-2025-{q:02d}",
                "name": f"{'电动化' if line == 'ev' else '智能化'}平台基线 2025-{q:02d}",
                "business_line": line,
                "is_frozen": frozen,
                "freeze_date": str(today - timedelta(days=30 * q)) if frozen else None,
                "release_date": str(today + timedelta(days=30 * q)),
            })
    return baselines


def gen_requirements(
    n: int, cis: list[dict], baselines: list[dict]
) -> list[dict]:
    """前 25% 需求挂已冻结基线且状态为在途 → 制造阻塞"""
    frozen = [b for b in baselines if b["is_frozen"]]
    unfrozen = [b for b in baselines if not b["is_frozen"]]
    reqs = []

    for i in range(1, n + 1):
        line = "ev" if i % 2 == 0 else "ia"
        line_cis = [c for c in cis if c["business_line"] == line]
        # 前 25% 挂已冻结基线且状态为在途
        if i <= n // 4 and frozen:
            bl = random.choice([b for b in frozen if b["business_line"] == line] or frozen)
            status = random.choice(["open", "developing"])
        else:
            pool = [b for b in unfrozen if b["business_line"] == line] or unfrozen
            bl = random.choice(pool) if pool else None
            status = random.choice(["draft", "open", "developing", "verified", "closed"])

        target_cis = random.sample(line_cis, min(random.randint(1, 3), len(line_cis)))
        reqs.append({
            "req_no": f"REQ-{line.upper()}-{i:04d}",
            "title": f"需求-{line.upper()}-{i:04d}",
            "description": f"需求描述 REQ-{line.upper()}-{i:04d}",
            "business_line": line,
            "priority": random.choice(["P0", "P1", "P2", "P3"]),
            "status": status,
            "baseline_no": bl["baseline_no"] if bl else None,
            "affected_ci_nos": [c["ci_no"] for c in target_cis],
        })

    return reqs


def gen_change_requests(
    n: int, cis: list[dict], baselines: list[dict], issues: list[dict] | None = None
) -> list[dict]:
    """变更单关联基线 + 可选关联触发问题单"""
    crs = []
    for i in range(1, n + 1):
        line = "ev" if i % 2 == 0 else "ia"
        line_baselines = [b for b in baselines if b["business_line"] == line]
        line_cis = [c for c in cis if c["business_line"] == line]
        target_ci = random.choice(line_cis) if line_cis else None
        bl = random.choice(line_baselines) if line_baselines else None
        source_issue_no = (
            random.choice([iss["issue_no"] for iss in issues]) if issues and random.random() < 0.3 else None
        )
        crs.append({
            "cr_no": f"CR-{line.upper()}-{i:04d}",
            "title": f"变更-{line.upper()}-{i:04d}",
            "reason": f"变更原因 CR-{line.upper()}-{i:04d}",
            "scope_desc": f"变更范围描述，涉及 {target_ci['name'] if target_ci else 'N/A'}",
            "business_line": line,
            "status": random.choice(["submitted", "reviewing", "approved", "rejected", "done"]),
            "target_baseline_no": bl["baseline_no"] if bl else None,
            "source_issue_no": source_issue_no,
            "affected_ci_nos": [target_ci["ci_no"]] if target_ci else [],
        })
    return crs


def main():
    logger.info("生成配置项 300 ...")
    cis = gen_config_items(300)

    logger.info("生成基线 20 ...")
    baselines = gen_baselines(20)
    frozen_count = sum(1 for b in baselines if b["is_frozen"])
    logger.info(f"  其中已冻结: {frozen_count} 条")

    logger.info("生成需求 200 ...")
    reqs = gen_requirements(200, cis, baselines)
    blocked = sum(1 for r in reqs if r["status"] in ("open", "developing") and r["baseline_no"] and any(
        b["is_frozen"] for b in baselines if b["baseline_no"] == r["baseline_no"]
    ))
    logger.info(f"  在途且挂冻结基线: {blocked} 条（Cypher 查询的期望结果）")

    logger.info("生成变更单 150 ...")
    crs = gen_change_requests(150, cis, baselines)

    output = {
        "config_items": cis,
        "baselines": baselines,
        "requirements": reqs,
        "change_requests": crs,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"输出: {OUT_PATH}")
    logger.info(f"  配置项: {len(cis)} | 基线: {len(baselines)} | 需求: {len(reqs)} | 变更单: {len(crs)}")


if __name__ == "__main__":
    main()
