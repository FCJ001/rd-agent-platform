#!/usr/bin/env python
# ============================================================
# 清理长期记忆里被误存的诊断结论
#
# 背景：分诊收敛时 save_diagnosis_memory 会把结论写进个人记忆
#   （namespace = users/{uid}/memories，key = diagnosis_{ts}）。
#   那批数据只有本人能召回 —— 同一故障两个人各诊断一次，互相看不到。
#   现已停写，结论改落 ai_triage_results 并由 search_past_diagnoses 复用。
#
# 识别依据（不需要额外字段）：Milvus 主键是 "{namespace}::{key}"，
#   所以结论行必然带 "::diagnosis_" 片段。同时 key 前缀也是 diagnosis_，
#   两条都查一遍，避免只依赖其中一种形态。
#
# 用法：
#   python scripts/cleanup_memory_conclusions.py            # 干跑，只统计
#   python scripts/cleanup_memory_conclusions.py --apply    # 实际删除
# ============================================================

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logger import logger  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

COLLECTION = "agent_long_term_memory"
# 两种形态都覆盖：主键里的 "::diagnosis_" 与 key 字段前缀 "diagnosis_"
EXPRS = [
    'id like "%::diagnosis_%"',
    'key like "diagnosis_%"',
]


def main():
    ap = argparse.ArgumentParser(description="清理记忆集合里被误存的诊断结论")
    ap.add_argument("--apply", action="store_true", help="实际删除（默认只统计）")
    args = ap.parse_args()

    from pymilvus import Collection, utility

    from src.infra.milvus_client import get_milvus_client_alias

    alias = get_milvus_client_alias()
    if not utility.has_collection(COLLECTION, using=alias):
        logger.info(f"集合 {COLLECTION} 不存在，无需清理")
        return

    collection = Collection(COLLECTION, using=alias)
    collection.load()

    if not args.apply:
        logger.info("干跑模式（只统计，不删除）。确认条数后再加 --apply。")

    total = 0
    for expr in EXPRS:
        rows = collection.query(expr=expr, output_fields=["id", "namespace", "created_at"])
        logger.info(f"[{expr}] 命中 {len(rows)} 条")
        for r in rows[:3]:
            logger.info(f"    样例 id={r.get('id')}")
        total += len(rows)

        if args.apply and rows:
            collection.delete(expr=expr)
            logger.info(f"    已删除 {len(rows)} 条")

    if args.apply:
        collection.flush()
        # 删除后复查：应归零
        leftover = sum(
            len(collection.query(expr=e, output_fields=["id"])) for e in EXPRS
        )
        logger.info(f"清理完成，剩余命中 {leftover} 条" + ("（应为 0）" if leftover else ""))
    else:
        logger.info(
            f"共命中 {total} 条（两种表达式可能有重叠，实际去重后不超过此数）。"
            "注意：结论的正式副本在 ai_triage_results，删除记忆行不丢诊断结果"
        )


if __name__ == "__main__":
    main()
