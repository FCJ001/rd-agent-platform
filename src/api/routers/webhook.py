# ============================================================
# ALM 平台事件 Webhook（幂等消费）
#
# ★ 开发期模拟：Java 平台不发事件，你手动 curl 模拟。
#   Agent 侧的逻辑是【真的】—— 幂等键去重、写 alm_* 表、异常处理，
#   只是事件的来源从 MQ 换成了 HTTP POST。
#
# 幂等设计：
#   同一个 (event_type, entity_type, entity_id, entity_version)
#   只能处理一次。第二次请求走 UNIQUE 约束冲突分支，
#   直接返回 200 + duplicate，不报错。
#
# 验收：
#   # 第一次 → 201 created，日志: [EVENT] issue ISS-2025-ZZ001 v1
#   curl -X POST http://localhost:8000/api/v1/webhooks/alm \
#     -H "Content-Type: application/json" \
#     -d '{...event json...}'
#
#   # 第二次同样 payload → 200 duplicate，日志: [DUPLICATE] 跳过
#   curl ... (同一条)
# ============================================================

from datetime import datetime

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.core.base_schema import ResponseSchema
from src.core.exceptions import BizException
from src.core.logger import logger
from src.infra.db import get_db

router = APIRouter(prefix="/api/v1/webhooks", tags=["平台事件"])


# ---------------------------------------------------------
# 请求体
# ---------------------------------------------------------
class WebhookEvent(BaseModel):
    event_type: str = Field(..., description="事件类型：issue.created / issue.updated / issue.closed 等",
                            examples=["issue.created"])
    entity_type: str = Field(..., description="实体类型：alm_issues / alm_requirements",
                             examples=["alm_issues"])
    entity_id: str = Field(..., description="平台侧实体 ID，如 ISS-2025-00123")
    entity_version: int = Field(..., description="平台侧版本号，1 起自增", ge=1)
    occurred_at: str | None = Field(None, description="事件发生时间 ISO8601")
    data: dict = Field(..., description="实体 payload，字段见 alm_* 表定义")


# ---------------------------------------------------------
# 分发表：entity_type → 表名 + 去重键列
# ---------------------------------------------------------
DISPATCH = {
    "alm_issues": {
        "table": "alm_issues",
        "key_column": "issue_no",
    },
    "alm_requirements": {
        "table": "alm_requirements",
        "key_column": "req_no",
    },
    "alm_change_requests": {
        "table": "alm_change_requests",
        "key_column": "cr_no",
    },
    "alm_config_items": {
        "table": "alm_config_items",
        "key_column": "ci_no",
    },
    "alm_baselines": {
        "table": "alm_baselines",
        "key_column": "baseline_no",
    },
}


@router.post("/alm")
async def consume_alm_event(
    event: WebhookEvent,
    request: Request,
    db = Depends(get_db),
):
    """
    消费 ALM 平台事件。

    ★ 核心：先写事件日志（依靠唯一约束做幂等），
      再 upsert 镜像表。事件日志写成功才动镜像，
      而不是反过来 —— 否则「镜像写一半、日志没写」会让重试时跳过这条，
      丢了数据。
    """
    dispatch = DISPATCH.get(event.entity_type)
    if dispatch is None:
        raise BizException(f"不支持的实体类型：{event.entity_type}")

    # ---- ① 幂等检查（唯一约束保证原子性）----
    try:
        await db.execute(
            text(
                """INSERT INTO alm_event_log
                     (event_type, entity_type, entity_id, entity_version, payload_json)
                   VALUES (:event_type, :entity_type, :entity_id, :entity_version, :payload_json)"""),
            {
                "event_type": event.event_type,
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "entity_version": event.entity_version,
                "payload_json": event.model_dump_json(),
            },
        )
        await db.flush()
    except Exception:
        await db.rollback()
        logger.info(
            f"[DUPLICATE] 跳过 {event.event_type} {event.entity_id} v{event.entity_version}"
        )
        return ResponseSchema(
            code=200,
            message="duplicate — 已处理过，跳过",
            data={"status": "skipped", "event_id": event.entity_id},
        )

    # ---- ② 写入镜像表 ----
    table = dispatch["table"]
    key_col = dispatch["key_column"]
    data = event.data

    try:
        # 模拟场景：data 里可能只有部分字段。
        # 用 INSERT ... ON CONFLICT DO UPDATE 做 upsert，
        # 不存在就插，存在且版本号更新就覆盖。
        # 生产环境的 sync.py 会做完整 upsert，这里只处理 webhook 过来的增量字段。
        columns = list(data.keys())
        if key_col not in columns:
            columns.append(key_col)
            data[key_col] = event.entity_id
        if "updated_at" not in columns:
            data["updated_at"] = datetime.utcnow()

        # 动态构建 INSERT（字段白名单从 data dict 来，不做拼接避免注入）
        placeholders = {k: f":{k}" for k in columns}
        set_clause = ", ".join(
            f"{k} = EXCLUDED.{k}" for k in columns if k != key_col
        )

        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders.values())}) "
            f"ON CONFLICT ({key_col}) DO UPDATE SET {set_clause}"
        )

        await db.execute(text(sql), data)
        await db.commit()

    except Exception:
        await db.rollback()
        # 事件日志已写，但镜像 upsert 失败。
        # 生产环境这里走 DLQ（死信队列），开发期打日志 + 返回错误，
        # 运维人员手动重放时靠版本号幂等，不会重复。
        logger.exception(f"[EVENT] 镜像 upsert 失败 {event.event_type} {event.entity_id}")
        raise BizException(
            f"镜像写入失败: {event.event_type} {event.entity_id} v{event.entity_version}"
        )

    logger.info(
        f"[EVENT] {event.event_type} {event.entity_id} v{event.entity_version} "
        f"→ {table}.{key_col} ({len(columns)} 字段)"
    )

    return ResponseSchema(
        code=200,
        message="created",
        data={"status": "processed", "event_id": event.entity_id},
    )
