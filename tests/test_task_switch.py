# ============================================================
# 确认式任务切换（换台）测试
# 最终方案：docs「确认式任务切换」—— 提示回合零状态变动，
# 终止只由用户精确关键词触发，暂存脱敏文本、消费即删。
#
# 铁律对应：
#   1. 「切换」在离题判定器之前拦截（漏拦截会被当追问回答）
#   2. 切换回合 resume 的是规范化退出令牌「退出诊断」
#   3. 暂存消费后自动作为新输入续发；无暂存走兜底文案
#   4. 旁路关（shadow 模式）不影响切换退出能力
# ============================================================

import types
from contextlib import asynccontextmanager

import pytest
from langchain_core.messages import AIMessage

import src.api.routers.chat as chat_mod
from src.agents.sub_agents import SWITCH_EXIT_TOKEN, is_task_switch
from src.core.deps import UserContext


# ── 1. 关键词判定 ────────────────────────────────────────────────────────

def test_switch_commands_exact_match():
    assert is_task_switch("切换")
    assert is_task_switch(" 切换。")
    assert is_task_switch("切换流程")
    assert not is_task_switch("我们切换一下思路看看")     # 长句不误伤
    assert not is_task_switch("充电时跳闸")


def test_switch_exit_token_is_registered_exit():
    """发给图的令牌必须是全局退出词表成员（旧/新路径的退出分支都认它）。"""
    from src.agents.triage.session_store import is_triage_exit
    assert is_triage_exit(SWITCH_EXIT_TOKEN)


# ── 2. 路由拦截（判定器之前）────────────────────────────────────────────

async def test_switch_intercepted_before_classifier(monkeypatch):
    """「切换」→ ("switch", None)，离题判定器根本不该被调到。"""
    agent, config, store = None, None, None
    from tests.test_offtopic_interruption import _agent_with_pending, _ctx

    agent, config, store = await _agent_with_pending()
    monkeypatch.setattr(chat_mod.settings, "SIDE_ASSISTANT_ENABLED", True)

    calls = {"classify": 0}

    async def classify_spy(q, m):
        calls["classify"] += 1
        return "other"

    monkeypatch.setattr(chat_mod, "classify_offtopic", classify_spy)
    action, reply = await chat_mod._route_turn(agent, config, "切换", _ctx())
    assert action == "switch"
    assert reply is None
    assert calls["classify"] == 0            # ★ 拦截发生在判定器之前


async def test_bypass_turn_stashes_masked_message(monkeypatch):
    """真实旁路发生时暂存离题消息（入口已脱敏的文本）。"""
    from src.agents.triage.offtopic import INTENT_OTHER
    from tests.test_offtopic_interruption import _agent_with_pending, _ctx

    agent, config, store = await _agent_with_pending()
    monkeypatch.setattr(chat_mod.settings, "SIDE_ASSISTANT_ENABLED", True)

    async def fake_classify(q, m):
        return INTENT_OTHER

    async def fake_answer(m, q, ctx, exclude_tool=None):
        return "答"

    stashed = []

    async def fake_stash(tid, msg):
        stashed.append((tid, msg))

    monkeypatch.setattr(chat_mod, "classify_offtopic", fake_classify)
    monkeypatch.setattr(chat_mod, "answer_offtopic", fake_answer)
    monkeypatch.setattr(chat_mod, "_stash_switch_msg", fake_stash)

    action, reply = await chat_mod._route_turn(
        agent, config, "帮我查下这个月EV线故障统计", _ctx(),
    )
    assert action == "bypass"
    assert stashed and stashed[0][0] == "u9:s9"          # thread_id 正确
    assert stashed[0][1] == "帮我查下这个月EV线故障统计"  # 消息原文（脱敏后）


# ── 3. 端点级：双段调用 ──────────────────────────────────────────────────

class _RecordingAgent:
    """记录每次 ainvoke 的载荷，按脚本返回结果。"""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def aget_state(self, config):
        return types.SimpleNamespace(tasks=[])

    async def ainvoke(self, payload, config=None, context=None):
        self.calls.append(payload)
        return self.results.pop(0)


def _result(text: str, interrupt: bool = False) -> dict:
    d = {"messages": [AIMessage(content=text)]}
    if interrupt:
        d["__interrupt__"] = (types.SimpleNamespace(
            value={"type": "triage_followup", "question": "新追问"}),)
    return d


@pytest.fixture()
def client(monkeypatch):
    from src.main import app

    @asynccontextmanager
    async def null_lock(tid):
        yield

    monkeypatch.setattr(chat_mod, "_chat_turn_lock", null_lock)
    app.dependency_overrides[chat_mod.get_current_user] = lambda: UserContext(
        user_id="u1", session_id="s1", role="engineer", business_line="ev",
    )
    app.dependency_overrides[chat_mod._chat_rate_limit] = lambda: None
    yield app
    app.dependency_overrides.clear()


async def test_switch_exit_then_resend_stashed(client, monkeypatch):
    """有暂存：第一次 resume 退出令牌，第二次暂存消息作为新输入。"""
    agent = _RecordingAgent([
        _result("分诊已按你的要求退出，本轮诊断作废。"),
        _result("新的诊断开始了：空调不响", interrupt=True),
    ])
    monkeypatch.setattr(chat_mod, "get_supervisor_agent", lambda: _async(agent))
    monkeypatch.setattr(chat_mod.settings, "SUBGRAPH_TRIAGE_ENABLED", False)

    async def fake_route(a, c, m, ctx):
        return "switch", None

    monkeypatch.setattr(chat_mod, "_route_turn", fake_route)

    async def fake_pop(tid):
        return "空调不制冷，帮我看看"

    monkeypatch.setattr(chat_mod, "_pop_switch_msg", fake_pop)

    from httpx import ASGITransport, AsyncClient
    transport = ASGITransport(app=client)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/api/v1/chat", json={"session_id": "s1", "message": "切换"})

    assert r.status_code == 200
    data = r.json()["data"]
    # 两段回复拼接；续发段挂起新追问 → 回复尾部是新追问（与 _reply_from_result
    # 的既有契约一致：挂起轮取 interrupt 载荷的追问），triage_pending=True
    assert "作废" in data["reply"] and "新追问" in data["reply"]
    assert data["triage_pending"] is True
    # ★ 第一次调用的 resume 值是规范化退出令牌
    first = agent.calls[0]
    assert getattr(first, "resume", None) == SWITCH_EXIT_TOKEN
    # ★ 第二次调用以暂存消息为新输入
    second = agent.calls[1]
    assert second["messages"][0]["content"] == "空调不制冷，帮我看看"


async def test_switch_without_stash_falls_back(client, monkeypatch):
    """无暂存：只退出 + 提示重发，不发起第二次图调用。"""
    agent = _RecordingAgent([_result("分诊已按你的要求退出，本轮诊断作废。")])
    monkeypatch.setattr(chat_mod, "get_supervisor_agent", lambda: _async(agent))
    monkeypatch.setattr(chat_mod.settings, "SUBGRAPH_TRIAGE_ENABLED", False)

    async def fake_route(a, c, m, ctx):
        return "switch", None

    monkeypatch.setattr(chat_mod, "_route_turn", fake_route)
    monkeypatch.setattr(chat_mod, "_pop_switch_msg", lambda tid: _none())

    from httpx import ASGITransport, AsyncClient
    transport = ASGITransport(app=client)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/api/v1/chat", json={"session_id": "s1", "message": "切换"})

    data = r.json()["data"]
    assert "请把你的新需求直接发送给我" in data["reply"]
    assert data["triage_pending"] is False
    assert len(agent.calls) == 1                    # 没有第二次调用


async def _async(val):
    return val


async def _none():
    return None
