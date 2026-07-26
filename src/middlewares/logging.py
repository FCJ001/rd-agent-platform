# ============================================================
# 请求日志中间件 + trace_id 全链路透传
#
# trace_id 的来源有优先级：
#   ① 请求头 X-Trace-Id  —— Java 网关生成的，跨服务要沿用同一个
#   ② 本地生成           —— 直连调试时没有网关
#
# 为什么必须沿用而不是每跳都新生成：
#   一次用户请求会穿过 Java 网关 → 本服务 → 项目二知识服务。
#   每跳各生成一个 id，出问题时三个服务的日志就串不起来了。
#   trace_id 的价值全在"同一个值出现在多个服务的日志里"。
#
# 响应头也回带 X-Trace-Id：前端报错时截个图就能定位到日志。
# ============================================================

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.core.logger import logger, new_trace_id, trace_id_var


class TraceLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or new_trace_id()
        token = trace_id_var.set(trace_id)

        start = time.perf_counter()
        method, path = request.method, request.url.path
        # 开发期把身份头也打出来，验行过滤时不用再猜当前是哪个角色
        user_hint = request.headers.get("X-User-Id", "-")
        logger.info(f"--> {method} {path} user={user_hint}")

        try:
            response = await call_next(request)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"<-- {method} {path} {response.status_code} {elapsed:.1f}ms")
            response.headers["X-Trace-Id"] = trace_id
            return response
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception(f"<-- {method} {path} 异常 {elapsed:.1f}ms")
            raise
        finally:
            # ★ ContextVar 用完要 reset，且必须在【打完出参日志之后】：
            #   reset 放在返回日志前面的话，那条日志的 trace_id 就变回 "-" 了。
            #   不 reset 则更糟 —— ASGI 的协程是复用的，
            #   下一个请求会继承上一个请求的 trace_id。
            trace_id_var.reset(token)
