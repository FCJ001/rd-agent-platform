# ============================================================
# ALM 平台事件 Webhook（幂等消费）
#
# 安全（生产必需）：
#   HMAC-SHA256 验签。Java 平台对 raw body 计算
#     hex(hmac_sha256(WEBHOOK_SECRET, body))
#   放在 X-Webhook-Signature 头，并附 X-Webhook-Timestamp（±5 分钟内有效）。
#   APP_ENV=prod 且 WEBHOOK_SECRET 未配置 → 启动期直接拒绝（config.validate_production）；
#   dev 未配置则跳过验签并打警告（本地 curl 模拟方便）。
#
# 幂等设计：
#   同一个 (event_type, entity_type, entity_id, entity_version)
#   只能处理一次。第二次请求触发 UNIQUE 约束冲突（IntegrityError），
#   直接返回 200 + duplicate，不报错。
#   ★ 只捕 IntegrityError —— 以前裸 except Exception，DB 宕机也会被
#     误判成 "duplicate" 返回 200，事件被静默丢弃且永不可重放。
#
# 乱序防护：
#   镜像表带 entity_version，upsert 带
#   WHERE entity_version < EXCLUDED.entity_version 守卫 ——
#   后到的旧事件不会覆盖新数据。
#
# 验收：
#   curl -X POST http://localhost:8000/api/v1/webhooks/alm \
#     -H "Content-Type: application/json" -d '{...event json...}'
#   # 第一次 → 200 processed；第二次同样 payload → 200 duplicate
# ============================================================

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.core.base_schema import ResponseSchema
from src.core.config import get_settings
from src.core.exceptions import BizException
from src.core.logger import logger
from src.infra.db import session_scope
from src.utils.mask import apply_mask, mask_free_text

settings = get_settings()

router = APIRouter(prefix="/api/v1/webhooks", tags=["平台事件"])

# 验签时间窗：超过 ±5 分钟的请求视为重放，拒绝
SIGNATURE_MAX_SKEW_SECONDS = 300


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
# 分发表：entity_type → 表名 + 去重键列 + Model（列白名单来源）
# ---------------------------------------------------------
from src.modules.alm.model import (  # noqa: E402
    AlmBaseline, AlmChangeRequest, AlmConfigItem, AlmIssue, AlmRequirement,
)

DISPATCH = {
    "alm_issues": {
        "table": "alm_issues",
        "key_column": "issue_no",
        "model": AlmIssue,
    },
    "alm_requirements": {
        "table": "alm_requirements",
        "key_column": "req_no",
        "model": AlmRequirement,
    },
    "alm_change_requests": {
        "table": "alm_change_requests",
        "key_column": "cr_no",
        "model": AlmChangeRequest,
    },
    "alm_config_items": {
        "table": "alm_config_items",
        "key_column": "ci_no",
        "model": AlmConfigItem,
    },
    "alm_baselines": {
        "table": "alm_baselines",
        "key_column": "baseline_no",
        "model": AlmBaseline,
    },
}

# 不可写列：自增主键和行创建时间不接受平台 payload 覆盖
_NON_WRITABLE_COLUMNS = {"id", "created_at"}


def _writable_columns(model) -> set[str]:
    """从 Model 元数据导出可写列白名单。表名/列名绝不来自请求体。"""
    return {c.key for c in model.__table__.columns} - _NON_WRITABLE_COLUMNS


# ---------------------------------------------------------
# HMAC 验签
# ---------------------------------------------------------
def _verify_signature(request: Request, raw_body: bytes) -> None:
    secret = settings.WEBHOOK_SECRET
    if not secret:
        if not settings.is_dev:
            # ★ is_dev 白名单之外（staging/uat/未知取值）一律拒绝：
            # 没有验签的 webhook 等于任何人可伪造平台事件写入镜像表
            # validate_production 会在启动期拦住，这里是运行时兜底
            raise HTTPException(status_code=503, detail="webhook 验签未配置，拒绝处理")
        logger.warning("[WEBHOOK] WEBHOOK_SECRET 未配置，跳过验签（仅限开发环境）")
        return

    signature = request.headers.get("X-Webhook-Signature", "")
    ts_raw = request.headers.get("X-Webhook-Timestamp", "")
    if not signature or not ts_raw:
        raise BizException("缺少 X-Webhook-Signature / X-Webhook-Timestamp 头", 40001)

    try:
        ts = int(ts_raw)
    except ValueError:
        raise BizException("X-Webhook-Timestamp 必须是 Unix 秒级时间戳", 40001)

    if abs(time.time() - ts) > SIGNATURE_MAX_SKEW_SECONDS:
        raise BizException("事件时间戳超出允许窗口（疑似重放）", 40001)

    expect = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature.lower(), expect):
        logger.warning(f"[WEBHOOK] 签名校验失败 entity 校验拒绝")
        raise BizException("webhook 签名校验失败", 40001)


# ---------------------------------------------------------
# 后台分诊任务：强引用持有，防止 create_task 的任务被 GC 掉
# ---------------------------------------------------------
_background_tasks: set[asyncio.Task] = set()


@router.post("/alm")
async def consume_alm_event(
    event: WebhookEvent,
    request: Request,
):
    """
    消费 ALM 平台事件。

    ★ 顺序不可换：先写事件日志（唯一约束做幂等），再 upsert 镜像表。
      事件日志写成功才动镜像，而不是反过来 —— 否则「镜像写一半、日志没写」
      会让重试时跳过这条，丢了数据。
    """
    _verify_signature(request, await request.body())

    dispatch = DISPATCH.get(event.entity_type)
    if dispatch is None:
        raise BizException(f"不支持的实体类型：{event.entity_type}", 40001)

    # ---- ① 幂等检查（唯一约束保证原子性）----
    # 入库前脱敏：事件日志里的 payload 也必须是脱敏后的 ——
    # 「原始 VIN 不落库」是全库不变量，审计表不豁免
    data = apply_mask(dict(event.data), "aftersales")
    # 自由文本（description/title 等）里嵌的 VIN/手机号也要打码
    for k, v in data.items():
        if isinstance(v, str):
            data[k] = mask_free_text(v)

    try:
        async with session_scope() as db:
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
                    "payload_json": json.dumps(
                        {**event.model_dump(exclude={"data"}), "data": data},
                        ensure_ascii=False, default=str,
                    ),
                },
            )
    except IntegrityError:
        # 唯一约束冲突 = 重复投递，正常分支
        logger.info(
            f"[DUPLICATE] 跳过 {event.event_type} {event.entity_id} v{event.entity_version}"
        )
        return ResponseSchema(
            code=200,
            message="duplicate — 已处理过，跳过",
            data={"status": "skipped", "event_id": event.entity_id},
        )
    except Exception:
        # 非约束冲突（连接断、语句错等）是服务端故障：
        # 返回 5xx 让平台侧重试，绝不能吞成 duplicate 静默丢事件
        logger.exception(f"[EVENT] 事件日志写入失败 {event.event_type} {event.entity_id}")
        raise HTTPException(status_code=503, detail="事件日志写入失败，请稍后重试")

    # ---- ② 写入镜像表 ----
    table = dispatch["table"]
    key_col = dispatch["key_column"]
    writable = _writable_columns(dispatch["model"])

    dropped = [k for k in data.keys() if k not in writable]
    if dropped:
        # payload 里的未知字段丢弃并留痕 —— 列名绝不能未校验地拼进 SQL
        logger.warning(f"[EVENT] 丢弃未知字段 entity={event.entity_id} fields={dropped}")
    data = {k: v for k, v in data.items() if k in writable}

    columns = list(data.keys())
    if key_col not in columns:
        columns.append(key_col)
        data[key_col] = event.entity_id
    if "entity_version" not in columns:
        columns.append("entity_version")
    data["entity_version"] = event.entity_version
    if "updated_at" not in columns:
        columns.append("updated_at")
        data["updated_at"] = datetime.now(timezone.utc)

    placeholders = {k: f":{k}" for k in columns}
    set_clause = ", ".join(
        f"{k} = EXCLUDED.{k}" for k in columns if k != key_col
    )

    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(placeholders.values())}) "
        f"ON CONFLICT ({key_col}) DO UPDATE SET {set_clause} "
        # 乱序防护：只有更新的事件版本才允许覆盖
        f"WHERE {table}.entity_version < EXCLUDED.entity_version"
    )

    try:
        async with session_scope() as db:
            result = await db.execute(text(sql), data)
        stale = result.rowcount == 0
    except Exception:
        # 事件日志已提交、镜像失败：删除刚写的事件日志行作为补偿，
        # 否则平台重试会被幂等键挡住（duplicate），这条变更就永远丢了
        logger.exception(f"[EVENT] 镜像 upsert 失败 {event.event_type} {event.entity_id}")
        try:
            async with session_scope() as db:
                await db.execute(
                    text(
                        "DELETE FROM alm_event_log WHERE event_type = :et AND entity_type = :ty "
                        "AND entity_id = :eid AND entity_version = :v"
                    ),
                    {
                        "et": event.event_type,
                        "ty": event.entity_type,
                        "eid": event.entity_id,
                        "v": event.entity_version,
                    },
                )
        except Exception:
            logger.exception(
                f"[EVENT] 事件日志补偿删除失败，该事件需人工重放 {event.entity_id} v{event.entity_version}"
            )
        raise HTTPException(
            status_code=503,
            detail=f"镜像写入失败: {event.event_type} {event.entity_id} v{event.entity_version}",
        )

    if stale:
        logger.info(
            f"[STALE] 忽略旧版本事件 {event.entity_id} v{event.entity_version}（镜像已有更新版本）"
        )
        return ResponseSchema(
            code=200,
            message="stale — 事件版本落后于镜像，忽略",
            data={"status": "stale", "event_id": event.entity_id},
        )

    logger.info(
        f"[EVENT] {event.event_type} {event.entity_id} v{event.entity_version} "
        f"→ {table}.{key_col} ({len(columns)} 字段)"
    )

    # ---- ③ issue.created → 后台触发分诊 ----
    if event.entity_type == "alm_issues" and event.event_type == "issue.created":
        logger.info(f"[EVENT] 触发分诊 issue={event.entity_id}")
        # 不阻塞 webhook 响应；持强引用防任务被 GC
        task = asyncio.create_task(_auto_triage_new_issue(event.entity_id, data))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    return ResponseSchema(
        code=200,
        message="created",
        data={"status": "processed", "event_id": event.entity_id},
    )


async def _auto_triage_new_issue(issue_no: str, data: dict) -> None:
    """
    新问题单创建后自动执行分诊。
    开发期直接在 webhook 里调用，生产环境可改为 Celery/MQ 异步任务。
    """
    import uuid

    try:
        from src.agents.triage.graph import run_triage, TriageDeps, _get_llm_json, _get_llm_chat

        llm_json = _get_llm_json()
        llm_chat = _get_llm_chat()
        deps = TriageDeps(llm_json=llm_json, llm_chat=llm_chat, db_session_factory=session_scope)

        thread_id = f"auto:{issue_no}:{uuid.uuid4().hex[:6]}"
        # ★ 进 LLM 前打码：description 可能带完整 VIN/手机号，也是注入面
        message = mask_free_text(data.get("description", "") or data.get("title", ""))
        if data.get("dtc_snapshot"):
            message += f"\nDTC: {data['dtc_snapshot']}"

        logger.info(f"[AUTO-TRIAGE] 开始分诊 issue={issue_no} thread={thread_id}")
        reply, new_state = await run_triage(
            user_message=message,
            thread_id=thread_id,
            deps=deps,
            existing_state=None,
            viewer_role=data.get("source", "customer"),
            # 业务线取平台 payload（alm_issues.business_line 非空），
            # 否则现象词表和候选根因不过滤、会跨线串味
            business_line=data.get("business_line") or "",
        )

        logger.info(
            f"[AUTO-TRIAGE] 分诊完成 issue={issue_no} "
            f"phase={new_state.phase.value} "
            f"candidates={len(new_state.candidate_causes)} "
            f"confidence={new_state.confidence:.3f}"
        )

    except Exception as e:
        logger.warning(f"[AUTO-TRIAGE] 分诊失败 issue={issue_no}: {e}")
