# ============================================================
# triage_pending 信号测试（前端「诊断进行中」横幅的协议契约）
#
# 锁住的行为：
#   1. ChatResponse 默认带 triage_pending=False（向后兼容）
#   2. 同步端点：旁路回合（图没碰、挂起保持）必须回报 True
#   3. SSE done 帧带上 triage_pending 字段（前端 onDone 读取的契约）
#
# 端点级测试：monkeypatch agent/路由/锁/限流，不碰真实 Redis。
# ============================================================

import types
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

import src.api.routers.chat as chat_mod
from src.core.deps import UserContext, get_current_user


class _StubAgent:
    async def aget_state(self, config):
        return types.SimpleNamespace(tasks=[])


def _ctx() -> UserContext:
    return UserContext(user_id="u1", session_id="s1", role="engineer", business_line="ev")


@pytest.fixture()
def client(monkeypatch):
    """旁路回合的假装配：图不碰、锁/限流直通。"""
    from src.main import app

    async def fake_agent():
        return _StubAgent()

    @asynccontextmanager
    async def null_lock(thread_id):
        yield

    async def fake_route(agent, config, message, ctx):
        return "bypass", "旁路回答\n\n---\n\n我们接着刚才的诊断 —— 追问Q"

    monkeypatch.setattr(chat_mod, "get_supervisor_agent", fake_agent)
    monkeypatch.setattr(chat_mod, "_chat_turn_lock", null_lock)
    monkeypatch.setattr(chat_mod, "_route_turn", fake_route)

    app.dependency_overrides[get_current_user] = lambda: _ctx()
    app.dependency_overrides[chat_mod._chat_rate_limit] = lambda: None
    yield app
    app.dependency_overrides.clear()


def test_chat_response_default():
    from src.api.routers.chat import ChatResponse

    resp = ChatResponse(reply="r", session_id="s1")
    assert resp.triage_pending is False   # 默认值：老调用方不受影响


async def test_sync_bypass_turn_reports_pending(client):
    """旁路回合图一次没碰，挂起的追问原样保持 → triage_pending=True。"""
    transport = ASGITransport(app=client)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/api/v1/chat", json={"session_id": "s1", "message": "查统计"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["reply"].startswith("旁路回答")
    assert data["triage_pending"] is True


async def test_sse_done_frame_carries_triage_pending(client):
    """SSE：done 帧必须带 triage_pending 字段（前端横幅的数据源）。"""
    import json as _json

    transport = ASGITransport(app=client)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        async with ac.stream(
            "POST", "/api/v1/chat/stream", json={"session_id": "s1", "message": "查统计"},
        ) as resp:
            frames = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    frames.append(_json.loads(line[len("data: "):]))

    done = [f for f in frames if f.get("type") == "done"]
    assert len(done) == 1
    assert done[0]["triage_pending"] is True
    # token 帧先于 done 到达（旁路整段文本作为一帧下发）
    assert any(f.get("type") == "token" and "旁路回答" in f.get("content", "") for f in frames)
