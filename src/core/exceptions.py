# ============================================================
# 业务异常 + 全局异常处理
#
# 约定：
#   BizException          → HTTP 200 + code != 200，前端按 code 展示
#   未捕获的 Exception     → HTTP 500 + 兜底文案，细节只进日志不给前端
#
# 为什么业务异常走 200：见 base_schema.py 顶部注释。
# ============================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.logger import logger, trace_id_var

# ------------------------------------------------------------
# 错误码
#   4 开头 = 调用方的问题，5 开头 = 服务端的问题
#   第二段两位标资源域：01 通用 / 02 用户与权限 / 03 问题单 / 04 Agent
# ------------------------------------------------------------
ERR_BAD_REQUEST = 40001        # 参数不合法
ERR_NOT_FOUND = 40004          # 资源不存在
ERR_USER_NOT_FOUND = 40201     # X-User-Id 在 users 表里查不到
ERR_USER_INACTIVE = 40202      # 用户已停用
ERR_PERMISSION_DENIED = 40203  # 角色无权访问该资源（行过滤之外的显式拒绝）
ERR_ISSUE_NOT_FOUND = 40301    # 问题单不存在或不在当前角色可见范围内
ERR_AGENT_FAILED = 40401       # Agent 执行失败（模型超时、工具调用异常等）
ERR_INTERNAL = 50000           # 服务端未知错误


class BizException(Exception):
    """业务异常。凡是"能预期的失败"都抛这个，不要抛裸 Exception"""

    def __init__(self, message: str, code: int = ERR_BAD_REQUEST):
        self.code = code
        self.message = message
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BizException)
    async def biz_exception_handler(request: Request, exc: BizException):
        logger.warning(f"业务异常 code={exc.code} path={request.url.path} msg={exc.message}")
        return JSONResponse(
            status_code=200,
            content={
                "code": exc.code,
                "message": exc.message,
                "data": None,
                "trace_id": trace_id_var.get(),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # logger.exception 会把 traceback 一起写进日志
        logger.exception(f"未捕获异常 path={request.url.path}")
        return JSONResponse(
            status_code=500,
            content={
                "code": ERR_INTERNAL,
                # ★ 不把 str(exc) 返给前端：可能带库表名、SQL、连接串
                "message": "服务内部错误，请联系管理员",
                "data": None,
                "trace_id": trace_id_var.get(),
            },
        )
