# ============================================================
# BI MCP 桥接测试
#
# 用官方 mcp SDK 起一个最小「假 rd-chatBI」MCP server（uvicorn 随机端口）：
# bi_query 工具把收到的请求头回显进 summary —— 直接验证「按次身份头透传」。
# 另覆盖连接失败降级、BI_TRANSPORT 分支接线。
# 不需要真实 rd-chatBI / PG / Redis。
# ============================================================

from types import SimpleNamespace

import pytest

from src.core.config import get_settings
from src.core.deps import UserContext


@pytest.fixture
def fake_chatbi_mcp():
    """最小 MCP server：bi_query 回显请求头（验证身份透传）。

    ★ server 跑在独立线程自己的 event loop 里（uvicorn.Server.run 自建 loop）：
      pytest-asyncio 的 fixture 默认 loop_scope 挂 session loop，而测试用
      function loop，session loop 无人驱动 —— 客户端请求永远得不到处理，
      挂到超时。线程隔离彻底绕开 loop 归属问题。"""
    import threading
    import time

    import uvicorn
    from mcp.server.fastmcp import Context, FastMCP

    mcp = FastMCP("fake-chatbi")

    @mcp.tool()
    async def bi_query(
        question: str,
        session_id: str = "default",
        with_chart: bool = False,
        ctx: Context = None,
    ) -> dict:
        request = ctx.request_context.request
        return {
            "success": True,
            "question": question,
            "sql": "SELECT count(*) FROM t",
            "data": [{"cnt": 1}],
            "columns": ["cnt"],
            "row_count": 1,
            "summary": (
                f"user={request.headers.get('x-user-id')},"
                f"role={request.headers.get('x-user-role')},"
                f"project={request.headers.get('x-project-id')},"
                f"domain={request.headers.get('x-owner-domain-id')}"
            ),
            "error": "",
        }

    config = uvicorn.Config(mcp.streamable_http_app(), host="127.0.0.1", port=0,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "假 MCP server 启动失败"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=5)


def _ctx() -> UserContext:
    return UserContext(user_id="u-bridge", session_id="s-1", role="engineer", owner_domain_id=3)


async def test_bridge_forwards_identity_and_returns_data(fake_chatbi_mcp, monkeypatch):
    from src.agents.tools import bi_mcp_bridge

    monkeypatch.setattr(get_settings(), "BI_MCP_URL", fake_chatbi_mcp)
    resp = await bi_mcp_bridge.bi_query_via_mcp("近7天问题单", "u-bridge:s-1", _ctx())

    assert "error" not in resp
    data = resp["data"]
    assert data["success"] is True
    assert data["row_count"] == 1
    # rd-chatBI 侧收到的就是本次调用的用户身份 —— 行级权限/限流按用户生效
    assert "user=u-bridge" in data["summary"]
    assert "role=engineer" in data["summary"]
    assert "project=rd_agent" in data["summary"]
    assert "domain=3" in data["summary"]


async def test_bridge_connection_error_degrades(monkeypatch):
    from src.agents.tools import bi_mcp_bridge

    monkeypatch.setattr(get_settings(), "BI_MCP_URL", "http://127.0.0.1:9/mcp/")
    resp = await bi_mcp_bridge.bi_query_via_mcp("q", "s", _ctx())
    # 文案与 HTTP 路径（_post_bi）一致，LLM 拿到的是可向用户转述的降级话术
    assert resp == {"error": "BI 服务暂不可用，请稍后重试"}


def test_degrade_branch_order_and_cancellation():
    """超时/不可用/取消的分类顺序（回归：TimeoutError 是 OSError 子类，
    顺序写反会让所有超时误报成"服务不可用"）"""
    import asyncio

    from src.agents.tools.bi_mcp_bridge import _degrade

    assert _degrade(asyncio.TimeoutError()) == "BI 服务响应超时，请稍后重试"
    # 真实调用里超时常被 anyio 包进 ExceptionGroup，要能拆到根因
    grouped = ExceptionGroup("g", [ExceptionGroup("inner", [TimeoutError()])])
    assert _degrade(grouped) == "BI 服务响应超时，请稍后重试"
    assert _degrade(ConnectionRefusedError()) == "BI 服务暂不可用，请稍后重试"
    assert _degrade(RuntimeError("协议错误")) == "BI 服务异常，请稍后重试"
    # 取消必须原样抛出：吞掉会把中断伪装成业务失败
    with pytest.raises(asyncio.CancelledError):
        _degrade(asyncio.CancelledError())


async def test_default_mcp_url_has_trailing_slash(monkeypatch):
    """/mcp（无尾斜杠）会 307，只有 mcp 1.30+ 会跟随 —— 默认值直接给 /mcp/"""
    from src.agents.tools import bi_mcp_bridge

    settings = get_settings()
    monkeypatch.setattr(settings, "BI_MCP_URL", "")
    monkeypatch.setattr(settings, "BI_SVC_URL", "http://svc:8004")

    captured = {}

    async def _fake_call(url, headers, tool, arguments):
        captured["url"] = url
        return {"success": True}

    monkeypatch.setattr(bi_mcp_bridge, "_call_tool_once", _fake_call)
    await bi_mcp_bridge.bi_query_via_mcp("q", "s", _ctx())
    assert captured["url"] == "http://svc:8004/mcp/"


async def test_transport_switch_wiring(fake_chatbi_mcp, monkeypatch):
    """BI_TRANSPORT=mcp 时 call_operation_agent 走桥接；默认 http 不碰 MCP"""
    from src.agents.tools import remote_knowledge

    settings = get_settings()
    calls = {}

    async def _fake_mcp(question, session_id, ctx):
        calls["mcp"] = (question, session_id, ctx)
        return {"data": {"success": True, "summary": "经MCP", "sql": "", "data": [],
                         "row_count": 0, "error": ""}}

    async def _fail_http(*a, **k):
        raise AssertionError("BI_TRANSPORT=mcp 下不应再走 HTTP 直连")

    monkeypatch.setattr(settings, "BI_TRANSPORT", "mcp")
    monkeypatch.setattr(settings, "BI_MCP_URL", fake_chatbi_mcp)
    monkeypatch.setattr(remote_knowledge, "bi_query_via_mcp", _fake_mcp)
    monkeypatch.setattr(remote_knowledge, "_post_bi", _fail_http)

    answer = await remote_knowledge.call_operation_agent.coroutine(
        message="Q3关闭率趋势", runtime=SimpleNamespace(context=_ctx())
    )
    assert "经MCP" in answer
    assert calls["mcp"][1] == "u-bridge:s-1"  # session_id 组装与 HTTP 路径一致

    # 默认配置（BI_TRANSPORT=http）不碰 MCP 桥接
    async def _fail_mcp(*a, **k):
        raise AssertionError("默认 http 传输不应走 MCP 桥接")

    monkeypatch.setattr(settings, "BI_TRANSPORT", "http")
    monkeypatch.setattr(remote_knowledge, "bi_query_via_mcp", _fail_mcp)

    async def _fake_http(endpoint, body, ctx, timeout=60):
        return {"data": {"success": True, "summary": "经HTTP", "sql": "", "data": [],
                         "row_count": 0, "error": ""}}

    monkeypatch.setattr(remote_knowledge, "_post_bi", _fake_http)
    answer = await remote_knowledge.call_operation_agent.coroutine(
        message="Q3关闭率趋势", runtime=SimpleNamespace(context=_ctx())
    )
    assert "经HTTP" in answer
