# ============================================================
# 日志（loguru）
#
# 两个职责：
#   ① 统一格式 + 控制台彩色 + 按天切文件
#   ② trace_id 全链路透传 —— 用 ContextVar 存，loguru patcher 自动塞进每条日志
#
# 为什么用 ContextVar 而不是把 trace_id 一路当参数传：
#   业务函数嵌套很深（router → service → agent → tool → repository），
#   一路传参会污染所有函数签名。ContextVar 在 asyncio 里天然按任务隔离，
#   中间件在请求入口 set 一次，这条请求链上所有日志都自动带上。
#
# 用法：
#   from src.core.logger import logger, setup_logger, trace_id_var
#   setup_logger()                      # main.py 启动时调一次
#   logger.info("xxx")                  # 输出自动带 trace_id
# ============================================================

import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

from loguru import logger

from src.core.config import get_settings

# ------------------------------------------------------------
# trace_id 上下文变量
# 默认 "-" 表示不在请求上下文里（比如启动日志、脚本日志）
# ------------------------------------------------------------
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    """生成 trace_id。取 uuid4 前 12 位，够用且日志不至于太长"""
    return uuid.uuid4().hex[:12]


def _patcher(record) -> None:
    """每条日志落盘前，把当前上下文的 trace_id 补进 extra"""
    record["extra"].setdefault("trace_id", trace_id_var.get())


_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>{extra[trace_id]}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logger() -> None:
    settings = get_settings()

    # 去掉 loguru 默认的 stderr sink，否则日志会打两遍
    logger.remove()
    logger.configure(patcher=_patcher)

    # 控制台
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=_LOG_FORMAT,
        colorize=True,
    )

    # 文件：按天切，留 30 天，旧文件压缩
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "{time:YYYY-MM-DD}.log",
        level=settings.LOG_LEVEL,
        format=_LOG_FORMAT,
        rotation="00:00",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
    )


__all__ = ["logger", "setup_logger", "trace_id_var", "new_trace_id"]
