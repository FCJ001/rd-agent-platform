# ============================================================
# 拟合数据驱动的分诊收敛阈值
#
# 数据源：ai_triage_results 里带人工反馈的记录（adopted 非空）。
# 特征：top1 置信度 / top1-top2 分差 / 已进行轮数（归一化）。
# 标签：adopted = 1（采纳）/ 0（拒绝）。
# 产物：eval/convergence_policy.json（check_convergence 自动加载）。
#
# 默认剔除 force_conclude=true 的记录 —— 轮次耗尽被强推的结论，
# 采纳与否反映的是"被逼着给结论"，不是置信度校准信号。
#
# 用法：
#   python scripts/fit_convergence.py                    # 拟合并写策略文件
#   python scripts/fit_convergence.py --threshold 0.85   # 收敛概率门限
#   python scripts/fit_convergence.py --dry-run          # 只看指标不写文件
# ============================================================

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.agents.triage.thresholds import (  # noqa: E402
    DEFAULT_PROB_THRESHOLD,
    MIN_FIT_SAMPLES,
    features,
    fit_logistic,
    policy_path,
    sigmoid,
)
from src.core.logger import logger  # noqa: E402


async def _load_feedback_rows() -> list[dict]:
    from src.infra.db import AsyncSessionLocal

    sql = """
        SELECT primary_confidence, candidate_causes, total_rounds, adopted, force_conclude
        FROM ai_triage_results
        WHERE adopted IS NOT NULL
          AND primary_confidence IS NOT NULL
        ORDER BY id
    """
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text(sql))).mappings().all()
    return [dict(r) for r in rows]


def _margin_of(candidate_causes) -> float:
    """从落库的 candidate_causes JSON 取 top1-top2 分差；缺失时回退 0。"""
    if isinstance(candidate_causes, str):
        try:
            candidate_causes = json.loads(candidate_causes)
        except Exception:
            return 0.0
    if not isinstance(candidate_causes, list) or not candidate_causes:
        return 0.0
    confs = sorted(
        (float(c.get("confidence", 0.0)) for c in candidate_causes
         if isinstance(c, dict) and c.get("confidence") is not None),
        reverse=True,
    )
    if not confs:
        return 0.0
    return confs[0] - (confs[1] if len(confs) > 1 else 0.0)


def evaluate(weights: list[float], threshold: float, samples) -> tuple[float, float, float]:
    """返回 (accuracy, adopted_recall, rejected_recall)。"""
    ok = adopted_hit = adopted_total = rejected_hit = rejected_total = 0
    for x, y in samples:
        p = 1 if sigmoid(sum(w * xi for w, xi in zip(weights, x))) >= threshold else 0
        ok += p == y
        adopted_total += y == 1
        rejected_total += y == 0
        adopted_hit += p == y == 1
        rejected_hit += p == y == 0
    acc = ok / len(samples)
    rec_ad = adopted_hit / adopted_total if adopted_total else 0.0
    rec_re = rejected_hit / rejected_total if rejected_total else 0.0
    return acc, rec_ad, rec_re


async def run(threshold: float, include_forced: bool, dry_run: bool) -> int:
    rows = await _load_feedback_rows()
    samples: list[tuple[list[float], int]] = []
    skipped_forced = 0
    for r in rows:
        if r["force_conclude"] and not include_forced:
            skipped_forced += 1
            continue
        top1 = float(r["primary_confidence"])
        margin = _margin_of(r["candidate_causes"])
        x = features(top1, margin, int(r["total_rounds"] or 1), max_rounds=5)
        samples.append((x, 1 if r["adopted"] else 0))

    print(f"样本：{len(samples)} 条（剔除强推结论 {skipped_forced} 条）")
    print(f"  采纳 {sum(y for _, y in samples)} / 拒绝 {sum(1 - y for _, y in samples)}")

    if len(samples) < MIN_FIT_SAMPLES:
        print(f"样本不足 {MIN_FIT_SAMPLES} 条，拒绝拟合 —— 权重少于此就是噪声。")
        return 1

    weights = fit_logistic(samples)
    acc, rec_ad, rec_re = evaluate(weights, threshold, samples)
    print(f"拟合权重：{[round(w, 4) for w in weights]}  （w0偏置, w1 top1, w2 分差, w3 轮数）")
    print(f"门限 {threshold:.2f} 下：accuracy={acc:.1%}  采纳召回={rec_ad:.1%}  拒绝召回={rec_re:.1%}")

    if dry_run:
        print("dry-run，不写策略文件。")
        return 0

    policy = {
        "weights": [round(w, 6) for w in weights],
        "prob_threshold": threshold,
        "n_samples": len(samples),
        "accuracy": round(acc, 4),
        "fit_at": datetime.now(timezone.utc).isoformat(),
    }
    out = policy_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"策略已写入 {out}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="拟合数据驱动收敛阈值")
    parser.add_argument("--threshold", type=float, default=DEFAULT_PROB_THRESHOLD,
                        help=f"P(采纳) 收敛门限（默认 {DEFAULT_PROB_THRESHOLD}）")
    parser.add_argument("--include-forced", action="store_true",
                        help="把轮次耗尽的强推结论也计入训练样本")
    parser.add_argument("--dry-run", action="store_true", help="只打印指标，不写文件")
    args = parser.parse_args()

    start = time.time()
    code = asyncio.run(run(args.threshold, args.include_forced, args.dry_run))
    logger.info(f"[FIT] 耗时 {time.time() - start:.1f}s")
    sys.exit(code)


if __name__ == "__main__":
    main()
