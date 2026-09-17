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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.infra.db import AsyncSessionLocal
from src.utils.mask import apply_mask

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """tz-aware UTC now。naive 的 utcnow() 在 timestamptz 列上会按本地时区解释。"""
    return datetime.now(timezone.utc)


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


async def upsert_issue(
    db: AsyncSession,
    data: dict,
    entity_version: int | None = None,
) -> str:
    """
    单条问题单 upsert。

    ★ 和 step 4 种子的 INSERT 共用 issue_no 做去重键，
      所以 webhook 事件和种子数据互不冲突。

    Args:
        entity_version: 平台侧版本号。传入时启用乱序守卫
            （旧版本不覆盖新数据），不传则无条件覆盖（种子/手工同步）。

    返回：'inserted' | 'updated' | 'unchanged'
          unchanged = 事件版本落后于镜像，被乱序守卫挡下
    """
    # 列白名单：只取 model 里定义的字段，多余的忽略（updated_at 允许平台时间透传）
    ALLOWED = {
        "issue_no", "title", "description", "source", "business_line",
        "severity", "status", "model_code", "sw_version", "vin",
        "dtc_snapshot", "reporter_id", "owner_domain_id", "external_ref",
        "updated_at",
    }
    filtered = {k: v for k, v in data.items() if k in ALLOWED}
    # 入库前字段脱敏：vin 只留后 6 位（见 AlmIssue.vin 注释），原始 VIN 不落库
    filtered = apply_mask(filtered, "aftersales")
    filtered.setdefault("updated_at", _utcnow())

    columns = list(filtered.keys())
    columns.append("entity_version")
    filtered["entity_version"] = entity_version if entity_version is not None else 0

    placeholders = {k: f":{k}" for k in columns}
    set_clause = ", ".join(
        f"{k} = EXCLUDED.{k}" for k in columns if k != "issue_no"
    )

    sql = (
        f"INSERT INTO alm_issues ({', '.join(columns)}) "
        f"VALUES ({', '.join(placeholders.values())}) "
        f"ON CONFLICT (issue_no) DO UPDATE SET {set_clause}"
    )
    if entity_version is not None:
        # 乱序守卫：只有更新的事件版本才允许覆盖既有行
        sql += " WHERE alm_issues.entity_version < EXCLUDED.entity_version"
    # xmax = 0 → 本次是 INSERT（新行），否则是 UPDATE；
    # WHERE 守卫挡下旧版本时无返回行 → unchanged。
    # 之前用 rowcount 判断，但 ON CONFLICT DO UPDATE 无论是否真变化 rowcount 都是 1
    sql += " RETURNING (xmax = 0) AS inserted"

    result = await db.execute(text(sql), filtered)
    row = result.first()
    if row is None:
        action = "unchanged"
    else:
        action = "inserted" if row[0] else "updated"

    return action


def _schedule_dedup_index(issue_no: str) -> None:
    """把问题单向量写入 Milvus 去重索引（后台任务）。

    ★ 必须在调用方 commit 之后调用 —— 任务用独立短会话读库，
      commit 前调度会读到空行/幻影行（rollback 时 Milvus 留下脏索引）。
      索引滞后由 scripts/backfill_dedup_index.py 兜底。"""
    import asyncio

    async def _index():
        try:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(
                        text(
                            "SELECT id, issue_no, business_line, title, description "
                            "FROM alm_issues WHERE issue_no = :issue_no"
                        ),
                        {"issue_no": issue_no},
                    )
                ).mappings().first()
            if not row:
                return
            from src.agents.dedup import vector_index

            await asyncio.to_thread(vector_index.upsert_issues, [dict(row)])
        except Exception as e:
            logger.warning(f"[SYNC] 去重索引更新失败 issue_no={issue_no}: {e}")

    try:
        asyncio.get_running_loop().create_task(_index())
    except RuntimeError:
        # 无事件循环（脚本/同步上下文）→ 同步执行
        asyncio.run(_index())


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

    ★ 去重向量索引在 commit 之后统一调度（见 _schedule_dedup_index 注释），
      所以本函数不做索引 —— 由 sync_issues_from_json 等调用方负责。
    """
    inserted = updated = unchanged = 0

    for i, data in enumerate(issues):
        issue_no = data.get("issue_no")
        if not issue_no:
            logger.warning(f"跳过第 {i} 条：缺 issue_no")
            continue

        ver = data.get("entity_version")
        action = await upsert_issue(db, data, entity_version=ver)

        # 记事件日志。幂等键 = (event_type, entity_type, entity_id, entity_version)：
        # ver 用平台版本号；批量文件没有版本号时用纳秒时间戳，
        # 保证每行每次同步都留痕（以前硬编码 1，第二条起全被 ON CONFLICT 吞掉）
        event_ver = int(ver) if ver is not None else time.time_ns()
        await db.execute(
            text(
                """INSERT INTO alm_event_log
                     (event_type, entity_type, entity_id, entity_version, payload_json)
                   VALUES ('issue.upsert', 'alm_issues', :eid, :ver, :payload)
                   ON CONFLICT DO NOTHING"""
            ),
            {
                "eid": issue_no,
                "ver": event_ver,
                "payload": json.dumps(data, ensure_ascii=False, default=str),
            },
        )

        if action == "inserted":
            inserted += 1
        elif action == "updated":
            updated += 1
        else:
            unchanged += 1

    if inserted or updated or unchanged:
        logger.info(
            f"[SYNC] {source}: {inserted} 新增, {updated} 更新, {unchanged} 无变化/被乱序守卫挡下"
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

    upserted_nos = []
    async with AsyncSessionLocal() as db:
        try:
            inserted, updated, unchanged = await sync_issues_batch(
                db, data, source=f"json:{filepath.name}"
            )
            await db.commit()
            # ★ commit 成功后再调度向量索引：索引任务用独立会话读库，
            #   提前调度会在 rollback 时留下幻影索引
            for d in data:
                if d.get("issue_no"):
                    upserted_nos.append(d["issue_no"])
        except Exception:
            await db.rollback()
            raise

    for issue_no in upserted_nos:
        _schedule_dedup_index(issue_no)

    return {"inserted": inserted, "updated": updated, "unchanged": unchanged}
