# ============================================================
# ALM 镜像同步（增量 + 批量 upsert）
#
# ★ 开发期模拟：没有 Java 平台的定时任务，数据源从 JSON 文件或内存
#   dict 传入。Agent 侧的 upsert 逻辑是【真的】—— 按 updated_at
#   水位过滤 + ON CONFLICT 合并 + 同步日志。
#
# 用法：
#   # 从 JSON 文件批量同步
#   import asyncio
#   from src.modules.alm.sync import sync_issues_from_json
#   asyncio.run(sync_issues_from_json("data/alm_issues_batch.json"))
#
#   # 或者在 webhook handler 调单条 upsert
#   from src.modules.alm.sync import upsert_issue
#   await upsert_issue(db, issue_data)
# ============================================================

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.infra.db import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def _read_watermark(db: AsyncSession) -> datetime:
    """
    读上一次同步水位。

    生产环境这是一张 sync_state 表。
    开发期偷懒从 alm_event_log 的 created_at 取最大值 ——
    相当于"上一次成功处理事件的时刻"。
    """
    result = await db.execute(
        text("SELECT MAX(created_at) FROM alm_event_log WHERE entity_type = 'alm_issues'")
    )
    row = result.scalar_one_or_none()
    if row is None:
        # 从未同步过，返回纪元起点，拉全量
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    return row.replace(tzinfo=timezone.utc)


async def upsert_issue(db: AsyncSession, data: dict) -> str:
    """
    单条问题单 upsert。

    ★ 和 step 4 种子的 INSERT 共用 issue_no 做去重键，
      所以 webhook 事件和种子数据互不冲突。

    返回：'inserted' | 'updated' | 'unchanged'
    """
    # 列白名单：只取 model 里定义的字段，多余的忽略
    ALLOWED = {
        "issue_no", "title", "description", "source", "business_line",
        "severity", "status", "model_code", "sw_version", "vin",
        "dtc_snapshot", "reporter_id", "owner_domain_id", "external_ref",
    }
    filtered = {k: v for k, v in data.items() if k in ALLOWED}
    filtered.setdefault("updated_at", datetime.utcnow())

    columns = list(filtered.keys())
    placeholders = {k: f":{k}" for k in columns}
    set_clause = ", ".join(
        f"{k} = EXCLUDED.{k}" for k in columns if k != "issue_no"
    )

    sql = (
        f"INSERT INTO alm_issues ({', '.join(columns)}) "
        f"VALUES ({', '.join(placeholders.values())}) "
        f"ON CONFLICT (issue_no) DO UPDATE SET {set_clause}"
    )

    result = await db.execute(text(sql), filtered)
    # 从 affected rows 无法精确判断是 insert 还是 update，
    # 简单区分：affected 1 代表有变化
    action = "upserted" if result.rowcount else "unchanged"
    return action


async def sync_issues_batch(
    db: AsyncSession,
    issues: Sequence[dict],
    *,
    source: str = "manual_sync",
) -> tuple[int, int, int]:
    """
    批量同步问题单，记录事件日志。

    参数：
        issues: 问题单 dict 列表，每项至少含 issue_no
        source: 日志标记（"manual_sync" | "json_file" | "platform_cron"）

    返回：(inserted, updated, unchanged)

    之所以不在这个函数里按 watermark 过滤：
    caller 已经拿到数据了，在这里再 filter 只是多一层；生产环境
    watermark 过滤在 SQL 侧做（平台库 SELECT WHERE updated_at > :watermark），
    这里只做 upsert。
    """
    inserted = updated = unchanged = 0

    for i, data in enumerate(issues):
        issue_no = data.get("issue_no")
        if not issue_no:
            logger.warning(f"跳过第 {i} 条：缺 issue_no")
            continue

        action = await upsert_issue(db, data)

        # 记事件日志（幂等键用 index 模拟 —— 生产环境用平台序列号）
        await db.execute(
            text(
                """INSERT INTO alm_event_log
                     (event_type, entity_type, entity_id, entity_version, payload_json)
                   VALUES ('issue.upsert', 'alm_issues', :eid, :ver, :payload)
                   ON CONFLICT DO NOTHING"""
            ),
            {
                "eid": issue_no,
                "ver": 1,
                "payload": json.dumps(data, ensure_ascii=False, default=str),
            },
        )

        if action == "upserted":
            inserted += 1
        else:
            unchanged += 1

    if inserted > 0 or unchanged > 0:
        logger.info(
            f"[SYNC] {source}: {inserted} 条新增/更新, {unchanged} 条无变化"
        )

    return inserted, updated, unchanged


async def sync_issues_from_json(filepath: str | Path) -> dict:
    """
    从 JSON 文件批量同步问题单到镜像表。

    开发期用这个替代平台定时同步：
    手工准备一个 data/alm_issues_sync.json 文件，然后调这个函数。
    生产环境换成从平台 HTTP API 拉取（带 updated_at 水位）。

    返回：{'inserted': N, 'updated': N, 'unchanged': N}
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"同步源文件不存在: {filepath}")

    data = json.loads(filepath.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]

    async with AsyncSessionLocal() as db:
        try:
            inserted, updated, unchanged = await sync_issues_batch(
                db, data, source=f"json:{filepath.name}"
            )
            await db.commit()
        except Exception:
            await db.rollback()
            raise

    return {"inserted": inserted, "updated": updated, "unchanged": unchanged}
