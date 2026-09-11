# ============================================================
# 回填问题单去重向量索引（Milvus：alm_issue_dedup）
#
# 场景：
#   - 首次启用 Milvus 去重召回前的存量数据建索引；
#   - 写入钩子（alm/sync.py）失败导致索引缺口时的兜底。
#
# 用法：
#   python scripts/backfill_dedup_index.py                # 全量（90 天内）
#   python scripts/backfill_dedup_index.py --days 30      # 最近 30 天
#   python scripts/backfill_dedup_index.py --business-line ev
#
# 分批 embed + upsert，可重复执行（按主键 id 覆盖写，幂等）。
# ============================================================

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.core.logger import logger  # noqa: E402

BATCH_SIZE = 50


async def _load_issues(days: int, business_line: str | None) -> list[dict]:
    from src.infra.db import AsyncSessionLocal

    sql = """
        SELECT id, issue_no, business_line, title, description
        FROM alm_issues
        WHERE updated_at > NOW() - (:days || ' days')::interval
    """
    params: dict = {"days": days}
    if business_line:
        sql += " AND business_line = :bl"
        params["bl"] = business_line
    sql += " ORDER BY id"

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text(sql), params)).mappings().all()
    return [dict(r) for r in rows]


async def main_async(days: int, business_line: str | None) -> None:
    from src.agents.dedup import vector_index

    issues = await _load_issues(days, business_line)
    logger.info(f"[BACKFILL] 待索引问题单 {len(issues)} 条")

    indexed = 0
    for i in range(0, len(issues), BATCH_SIZE):
        batch = issues[i : i + BATCH_SIZE]
        count = await asyncio.to_thread(vector_index.upsert_issues, batch)
        indexed += count
        logger.info(f"[BACKFILL] 进度 {min(i + BATCH_SIZE, len(issues))}/{len(issues)}")

    logger.info(f"[BACKFILL] 完成，共写入 {indexed} 条向量")


def main():
    parser = argparse.ArgumentParser(description="回填问题单去重向量索引")
    parser.add_argument("--days", type=int, default=90, help="回填最近 N 天的问题单（默认 90）")
    parser.add_argument("--business-line", default=None, help="只回填指定业务线（ev/ia）")
    args = parser.parse_args()
    asyncio.run(main_async(args.days, args.business_line))


if __name__ == "__main__":
    main()
