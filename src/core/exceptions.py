# ============================================================
# 全局业务异常 + 异常处理器
#
# 约定：
#   BizException          → HTTP 200 + code != 200，前端按 code 展示
#   ── 例外：认证/授权类错误必须映射真实 HTTP 状态码（401/403），
#      否则网关、监控、告警在 HTTP 层看不到任何鉴权失败，
#      撞库/越权扫描和正常业务失败在访问日志里同形。
#   未捕获的 Exception     → HTTP 500 + 兜底文案，细节只进日志不给前端
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
ERR_TOKEN_INVALID = 40100      # 身份令牌缺失/无效/过期
ERR_USER_NOT_FOUND = 40201     # 令牌对应的用户在 users 表里查不到
ERR_USER_INACTIVE = 40202      # 用户已停用
ERR_PERMISSION_DENIED = 40203  # 角色无权访问该资源（行过滤之外的显式拒绝）
ERR_ISSUE_NOT_FOUND = 40301    # 问题单不存在或不在当前角色可见范围内
ERR_AGENT_FAILED = 40401       # Agent 执行失败（模型超时、工具调用异常等）
ERR_CONVERSATION_BUSY = 40409  # 同一会话已有请求在处理中（并发被拒）
ERR_RATE_LIMITED = 42901       # 请求过于频繁（限流），LMM 端点成本保护
ERR_INTERNAL = 50000           # 服务端未知错误

# 业务码 → HTTP 状态码。不在表里的业务码一律 HTTP 200（前端按 code 展示）
BIZ_HTTP_STATUS = {
    ERR_TOKEN_INVALID: 401,
    ERR_USER_NOT_FOUND: 401,
    ERR_USER_INACTIVE: 401,
    ERR_PERMISSION_DENIED: 403,
    # 409 Conflict：并发写坏会话状态是资源状态冲突，不是服务端故障 ——
    # 映射成 5xx 会让监控告警、重试中间件把它当成服务不可用
    ERR_CONVERSATION_BUSY: 409,
    ERR_RATE_LIMITED: 429,
}


class BizException(Exception):
    """业务异常。凡是"能预期的失败"都抛这个，不要抛裸 Exception"""

    def __init__(self, message: str, code: int = ERR_BAD_REQUEST):
        self.code = code
        self.message = message
        super().__init__(message)


class ConversationBusyError(Exception):
    """同一会话已有请求在处理中。

    ★ 与 BizException 分开，是因为它必须在 chat 路由内被捕获 ——
      它抛出时 Agent 的图状态可能停在中途，直接把异常交给全局处理器
      会让「暂停在分诊追问」的会话对外表现为 500，用户以为系统挂了。
      路由抓到它后应据快照判断：分诊在等回答 → 回提问，否则回「稍后再试」。
    """

    def __init__(self, thread_id: str = ""):
        self.thread_id = thread_id
        super().__init__(f"会话正在处理中: {thread_id}")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BizException)
    async def biz_exception_handler(request: Request, exc: BizException):
        logger.warning(f"业务异常 code={exc.code} path={request.url.path} msg={exc.message}")
        return JSONResponse(
            status_code=BIZ_HTTP_STATUS.get(exc.code, 200),
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
