# ============================================================
# BI 查询的 MCP 传输路径（BI_TRANSPORT=mcp 时启用）
#
# 与 HTTP 路径（remote_knowledge._post_bi）的返回形状严格对齐：
#   成功 → {"data": <BIQueryResponse 字段>}（交给既有的格式化代码）
#   失败 → {"error": <中文文案>}（给 LLM 看的，不含内网细节）
# 失败文案对齐 _post_bi 的三档（不可用/超时/异常）；业务级失败（如数据源
# 未注册）MCP 侧会带上 REST 的 message，比 HTTP 侧更完整。
#
# 传输形态：按次无状态 MCP 连接（initialize → tools/call → 关闭）。
# 不维持长连接会话：ToolRuntime 的用户身份是按请求的，常驻会话没法按次
# 换身份头；而 BI 查询本身是几十秒级 LLM 流水线，握手三次往返的开销可忽略。
# rd-chatBI 侧（/mcp 门面）按请求头做行级权限与限流，与 HTTP 路径等价。
#
# ★ 前置条件：身份只带 X-User-* 头（与 HTTP 路径一致），因此 rd-chatBI 需跑
#   AUTH_MODE=header（网关剥离外部身份头）。jwt 模式下两条路径都缺 token，
#   会 fail-closed 报"缺少认证凭证"——需要时再给 UserContext 补 token 透传。
# ============================================================

import asyncio
import json

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.core.config import get_settings
from src.core.deps import UserContext
from src.core.logger import logger

# 单次调用的总超时：rd-chatBI 的 9 阶段流水线是几十秒级，须大于其 P99。
# 握手（initialize）单独收紧：连不通就快速失败，别让用户等满整个流水线超时。
BI_MCP_HANDSHAKE_TIMEOUT_SECONDS = 15
BI_MCP_CALL_TIMEOUT_SECONDS = 120


def _mcp_headers(ctx: UserContext) -> dict[str, str]:
    """身份透传：与 remote_knowledge._auth_headers 同一套 X-User-* 头。
    rd-chatBI 的 /mcp 门面按这些头走路级权限与限流，两个传输路径语义一致。"""
    settings = get_settings()
    headers = {
        "X-User-Id": ctx.user_id,
        "X-Session-Id": ctx.session_id,
        "X-User-Role": ctx.role,
        "X-Project-Id": settings.BI_PROJECT_ID,
    }
    if ctx.business_line:
        headers["X-Business-Line"] = ctx.business_line
    if ctx.owner_domain_id:
        headers["X-Owner-Domain-Id"] = str(ctx.owner_domain_id)
    return headers


def _internal_http_client(headers: dict | None = None, timeout=None, auth=None):
    """内网服务间调用的 httpx 客户端：禁用环境/系统代理。

    ★ httpx 默认 trust_env=True，在 macOS 上会读到系统代理却不认它的
      例外清单（127.0.0.1 在清单里也会被代理劫持，返回 502）。服务间
      直连必须绕开代理，这也是生产内网调用的正确姿势。"""
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout or httpx.Timeout(30.0, read=300.0),
        auth=auth,
        trust_env=False,
    )


async def _call_tool_once(url: str, headers: dict, tool: str, arguments: dict) -> dict:
    """一次完整的 MCP Streamable HTTP 调用：连接 → 握手 → tools/call → 关闭。"""
    # 用 mcp 1.30 的新入口 streamable_http_client（旧名 streamablehttp_client 已废弃）；
    # 新版没有 headers 入参，身份头挂在自建的 httpx 客户端上
    async with streamable_http_client(
        url, http_client=_internal_http_client(headers=headers)
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(
                session.initialize(), BI_MCP_HANDSHAKE_TIMEOUT_SECONDS
            )
            result = await asyncio.wait_for(
                session.call_tool(tool, arguments), BI_MCP_CALL_TIMEOUT_SECONDS
            )
    if result.isError:
        text = result.content[0].text if result.content else "工具执行失败"
        raise RuntimeError(text)
    if getattr(result, "structuredContent", None) is not None:
        return dict(result.structuredContent)
    if result.content and result.content[0].text:
        return json.loads(result.content[0].text)
    return {}


def _degrade(e: BaseException) -> str:
    """把传输/协议异常映射成给 LLM 的降级文案（与 _post_bi 一致）。

    网络错误常包在 ExceptionGroup 里（anyio task group），要递归拆开看根因；
    细节只进日志，文案回给 LLM。"""
    stack = [e]
    while stack:
        cur = stack.pop()
        # 取消与进程级信号不归类成业务失败：调用方（LangGraph/用户 Ctrl-C）
        # 需要看到真实的取消，吞掉会把中断伪装成"服务异常"
        if isinstance(cur, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise cur
        # ★ 超时判定必须在 OSError 之前：Python≥3.11 起 TimeoutError 是
        #   OSError 的子类，顺序写反会让所有超时误报成"服务不可用"
        if isinstance(cur, (asyncio.TimeoutError, httpx.TimeoutException)):
            return "BI 服务响应超时，请稍后重试"
        if isinstance(cur, (httpx.ConnectError, ConnectionError, OSError)):
            return "BI 服务暂不可用，请稍后重试"
        stack.extend(getattr(cur, "exceptions", ()) or ())
    return "BI 服务异常，请稍后重试"


async def bi_query_via_mcp(question: str, session_id: str, ctx: UserContext) -> dict:
    """经 MCP 调 rd-chatBI 的 bi_query 工具。返回形状与 _post_bi 一致。"""
    settings = get_settings()
    # ★ 尾斜杠不是笔误：rd-chatBI 把 MCP 应用挂在 /mcp，访问无尾斜杠的 /mcp
    #   会 307 重定向，而 mcp SDK 只有 1.30+ 才自行跟随同源重定向 —— 默认值
    #   直接用 /mcp/ 免掉这一跳，也让旧版客户端不会静默失败
    url = settings.BI_MCP_URL or f"{settings.BI_SVC_URL.rstrip('/')}/mcp/"
    arguments = {"question": question, "session_id": session_id, "with_chart": False}
    try:
        result = await _call_tool_once(url, _mcp_headers(ctx), "bi_query", arguments)
    except BaseException as e:
        logger.warning(f"[BI-MCP] 调用失败: {url} err={e!r}")
        return {"error": _degrade(e)}
    return {"data": result}
