# ============================================================
# 追问期间离题分流（旁路回答）测试
# 设计：docs/design-side-assistant-bypass.md §7
#
# 锁住的行为：
#   1. 典型追问回答不触发判定（正则门零成本）
#   2. 旁路主路径：不碰图、快照仍挂起、store 进度不变、回复含答案+回放追问
#   3. 判定器异常 → 回落 answer（= 旁路上线前的行为）
#   4. 旁路助手异常 → 回放追问 + 失败提示（绝不回落 resume）
#   5. 回复编排格式（同步/SSE 共用同一 compose）
#   6. 影子模式：判定照跑只记日志，路由不变
#   7. 退出指令优先于旁路判定
#
# 骨架沿用 test_triage_hitl.py（StubRedis + fake model + MemorySaver），
# 全部纯内存，CI 可跑。
# ============================================================

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

import src.api.routers.chat as chat_mod
import src.agents.triage.offtopic as offtopic_mod
from src.agents.triage.offtopic import (
    INTENT_ANSWER, INTENT_OTHER, classify_offtopic, hits_offtopic_regex, _parse_verdict,
)
from src.agents.triage.session_store import TriageSessionStore
from src.agents.triage.state import TriageState
from src.core.deps import UserContext
from tests.test_triage_hitl import StubRedis, _make_fake_model

QUESTION = "是否伴随死机？"


# ── 测试基建 ─────────────────────────────────────────────────────────────

async def _agent_with_pending(thread: str = "u9:s9"):
    """构造一个挂在分诊追问上的 agent（快照有 pending interrupt）。"""
    redis = StubRedis()
    store = TriageSessionStore(redis)

    @tool
    async def stub_triage(message: str) -> str:
        """模拟分诊工具：保存进度后挂起追问。"""
        saved = await store.load(thread)
        if saved is None:
            await store.save(thread, TriageState(session_id=thread), QUESTION)
        answer = interrupt({"type": "triage_followup", "round": 1, "question": QUESTION})
        await store.clear(thread)
        return f"结论（基于回答：{answer}）"

    model = _make_fake_model([
        AIMessage(content="", tool_calls=[
            {"name": "stub_triage", "args": {"message": "车机黑屏"}, "id": "c1"}
        ]),
        AIMessage(content="诊断完成。"),
    ])
    agent = create_agent(
        model=model, tools=[stub_triage], system_prompt="test", checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": thread}}
    await agent.ainvoke({"messages": [{"role": "user", "content": "车机黑屏"}]}, config)
    return agent, config, store


class _NoInvokeAgent:
    """路由期间图绝不能被调用的哨兵：aget_state 放行，ainvoke 即失败。"""

    def __init__(self, inner):
        self._inner = inner

    async def aget_state(self, config):
        return await self._inner.aget_state(config)

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError(f"旁路路径调用了图: {args} {kwargs}")


def _ctx() -> UserContext:
    return UserContext(user_id="u9", session_id="s9", role="engineer", business_line="ev")


async def _patched_route(monkeypatch, *, enabled, verdict, answer=None, error=None):
    """装配好 monkeypatch 的 _route_turn 运行器。"""
    agent, config, store = await _agent_with_pending()
    monkeypatch.setattr(chat_mod.settings, "SIDE_ASSISTANT_ENABLED", enabled)

    calls = {"classify": 0, "answer": 0}

    async def fake_classify(q, m):
        calls["classify"] += 1
        return verdict

    async def fake_answer(m, q, ctx, exclude_tool=None):
        calls["answer"] += 1
        calls.setdefault("exclude", []).append(exclude_tool)
        if error:
            raise error
        return answer or "stub-answer"

    monkeypatch.setattr(chat_mod, "classify_offtopic", fake_classify)
    monkeypatch.setattr(chat_mod, "answer_offtopic", fake_answer)
    return _NoInvokeAgent(agent), config, store, calls


# ── 1. 正则门与判定器 ────────────────────────────────────────────────────

def test_regex_gate_typical_answers_do_not_hit():
    """典型追问回答不命中正则门 → 不产生任何模型调用。"""
    for msg in ["没有其他现象", "是的每次都这样", "充电时跳闸", "好像是，不太确定", "重启过两次"]:
        assert not hits_offtopic_regex(msg), msg


def test_regex_gate_offtopic_requests_hit():
    for msg in [
        "帮我查下这个月 EV 线故障统计",
        "800V 绝缘设计标准是什么",
        "解读一下这份 DTC 报告",
        "这个变更单有影响吗",
        "知识库里有没有 BMS 的资料",
    ]:
        assert hits_offtopic_regex(msg), msg


def test_parse_verdict_defensive():
    assert _parse_verdict('{"intent":"other"}') == INTENT_OTHER
    assert _parse_verdict('{"intent":"answer"}') == INTENT_ANSWER
    assert _parse_verdict('```json\n{"intent":"other"}\n```') == INTENT_OTHER
    assert _parse_verdict("我觉得是回答") == INTENT_ANSWER      # 非 JSON
    assert _parse_verdict('{"intent":"乱写"}') == INTENT_ANSWER  # 非法取值
    assert _parse_verdict("") == INTENT_ANSWER


async def test_typical_answer_skips_judge_and_routes_resume(monkeypatch):
    """正则门未命中 → 小模型零调用；路由按追问回答处理（resume）。"""
    calls = {"judge": 0}

    async def fake_judge(q, m):
        calls["judge"] += 1
        return INTENT_OTHER  # 即使裁决器存在也不该被调到

    monkeypatch.setattr(offtopic_mod, "_judge_with_llm", fake_judge)
    verdict = await classify_offtopic(QUESTION, "没有其他现象，充电时会出现")
    assert verdict == INTENT_ANSWER
    assert calls["judge"] == 0

    agent, config, store, route_calls = await _patched_route(
        monkeypatch, enabled=True, verdict=INTENT_ANSWER,
    )
    action, reply = await chat_mod._route_turn(
        agent, config, "没有其他现象，充电时会出现", _ctx(),
    )
    assert action == "resume"
    assert route_calls["answer"] == 0


# ── 2. 旁路主路径 ────────────────────────────────────────────────────────

async def test_bypass_main_path_state_untouched(monkeypatch):
    """命中且裁决 other → 图未被调用、快照仍挂起、store 不变、回复编排完整。"""
    agent, config, store, calls = await _patched_route(
        monkeypatch, enabled=True, verdict=INTENT_OTHER,
        answer="本月 EV 线共 12 起同类故障。",
    )
    before = await store.load("u9:s9")

    action, reply = await chat_mod._route_turn(
        agent, config, "帮我查下这个月 EV 线故障统计", _ctx(),
    )

    assert action == "bypass"
    assert calls["classify"] == 1 and calls["answer"] == 1
    # 回复 = 旁路答案 + 分隔线 + 回放追问 + 退出提示
    assert "本月 EV 线共 12 起同类故障。" in reply
    assert "---" in reply
    assert QUESTION in reply
    assert "退出诊断" in reply
    # 状态零变动：store 进度原样
    after = await store.load("u9:s9")
    assert after is not None
    assert after.state.round == before.state.round
    assert after.reply == before.reply


async def test_bypass_keeps_interrupt_pending(monkeypatch):
    """旁路后快照仍挂起 interrupt（下一轮回答照常 resume）。"""
    agent, config, store, _ = await _patched_route(
        monkeypatch, enabled=True, verdict=INTENT_OTHER, answer="a",
    )
    await chat_mod._route_turn(agent, config, "查下统计", _ctx())
    snapshot = await agent.aget_state(config)
    assert any(t.interrupts for t in snapshot.tasks)


# ── 3. 判定器异常回落 ────────────────────────────────────────────────────

async def test_judge_exception_falls_back_to_answer(monkeypatch):
    """裁决器抛异常 → classify 自身兜底为 answer（内部防线）。"""
    async def boom(q, m):
        raise TimeoutError("judge timeout")

    monkeypatch.setattr(offtopic_mod, "_judge_with_llm", boom)
    assert await classify_offtopic(QUESTION, "帮我查下这个月统计数据") == INTENT_ANSWER


async def test_route_survives_classify_exception(monkeypatch):
    """classify 整个挂掉 → 路由层再防一道，按 resume（旁路上线前行为）。"""
    agent, config, store = await _agent_with_pending()
    monkeypatch.setattr(chat_mod.settings, "SIDE_ASSISTANT_ENABLED", True)

    async def boom(q, m):
        raise RuntimeError("classify broken")

    monkeypatch.setattr(chat_mod, "classify_offtopic", boom)
    action, _ = await chat_mod._route_turn(agent, config, "帮我查下统计", _ctx())
    assert action == "resume"


# ── 4. 旁路助手异常 ──────────────────────────────────────────────────────

async def test_side_assistant_failure_replays_question(monkeypatch):
    """旁路助手失败：不回落 resume，回放追问 + 失败提示，状态零变动。"""
    agent, config, store, calls = await _patched_route(
        monkeypatch, enabled=True, verdict=INTENT_OTHER,
        error=TimeoutError("side assistant timeout"),
    )
    before = await store.load("u9:s9")

    action, reply = await chat_mod._route_turn(
        agent, config, "帮我查下这个月EV线故障统计", _ctx(),
    )

    assert action == "bypass"          # ★ 不是 resume：离题文本绝不能进分诊
    assert "没能处理成功" in reply
    assert QUESTION in reply
    after = await store.load("u9:s9")
    assert after.state.round == before.state.round
    snapshot = await agent.aget_state(config)
    assert any(t.interrupts for t in snapshot.tasks)


# ── 5. 回复编排格式 ──────────────────────────────────────────────────────

def test_compose_bypass_reply_format():
    reply = chat_mod._compose_bypass_reply("答案A", "追问Q")
    assert reply.startswith("答案A")
    assert "\n\n---\n\n" in reply
    assert "我们接着刚才的诊断 —— 追问Q" in reply
    assert "「切换」" in reply and "「退出诊断」" in reply   # 换台提示 + 退出提示


def test_compose_bypass_failure_format():
    reply = chat_mod._compose_bypass_failure("追问Q")
    assert "没能处理成功" in reply
    assert "追问Q" in reply
    assert "「切换」" in reply and "「退出诊断」" in reply


# ── 6. 影子模式 ──────────────────────────────────────────────────────────

async def test_shadow_mode_judges_but_does_not_reroute(monkeypatch):
    """ENABLED=false：判定照常运行（留痕），路由不变（resume）。"""
    agent, config, store, calls = await _patched_route(
        monkeypatch, enabled=False, verdict=INTENT_OTHER, answer="不该被用到",
    )
    action, reply = await chat_mod._route_turn(
        agent, config, "帮我查下这个月EV线故障统计", _ctx(),
    )
    assert action == "resume"
    assert calls["classify"] == 1      # 判定跑了（影子日志的数据源）
    assert calls["answer"] == 0        # 旁路助手没跑，路由没变


# ── 7. 退出指令优先 ──────────────────────────────────────────────────────

async def test_exit_command_takes_priority_over_bypass(monkeypatch):
    """退出指令（精确匹配、零成本）优先于离题判定：判定器不该被调到。"""
    agent, config, store, calls = await _patched_route(
        monkeypatch, enabled=True, verdict=INTENT_OTHER, answer="不该被用到",
    )
    action, _ = await chat_mod._route_turn(agent, config, "退出诊断", _ctx())
    assert action == "resume"
    assert calls["classify"] == 0


# ── 8. 回合级 trace v1 ───────────────────────────────────────────────────

def test_thread_token_usage_sums_metadata():
    from langchain_core.messages import AIMessage

    m1 = AIMessage(content="a", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    m2 = AIMessage(content="b", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
    assert chat_mod._thread_token_usage({"messages": [m1, m2]}) == {
        "thread_prompt_tokens": 17, "thread_completion_tokens": 8,
    }
    assert chat_mod._thread_token_usage(None) == {}
    # 无 usage_metadata 的消息（Human/Tool）不计
    assert chat_mod._thread_token_usage({"messages": [AIMessage(content="c")]}) == {}


def test_log_turn_trace_emits_structured_json(monkeypatch):
    import json as _json
    import time as _time
    import types

    captured = []
    monkeypatch.setattr(
        chat_mod, "logger",
        types.SimpleNamespace(info=captured.append, warning=lambda *a, **k: None),
    )
    chat_mod._log_turn_trace(
        "trace1", _ctx(), "bypass", _time.perf_counter() - 0.05, reply="x" * 10,
    )
    assert len(captured) == 1
    line = captured[0]
    assert line.startswith("[TRACE] ")
    data = _json.loads(line[len("[TRACE] "):])
    assert data["trace_id"] == "trace1"
    assert data["route"] == "bypass"
    assert data["user_id"] == "u9" and data["session_id"] == "s9"
    assert data["reply_chars"] == 10
    assert data["pending_followup"] is False
    assert 0 <= data["latency_ms"] < 10_000


# ── 9. 子智能体注册表接线 ────────────────────────────────────────────────

async def test_bypass_excludes_active_task_entry_tool(monkeypatch):
    """分诊挂起时，旁路助手必须摘除 call_triage_agent（注册表驱动）。

    这是红线「旁路线程不得另起同种流程」的泛化：以后每个子智能体
    挂起时摘各自的入口工具（sub_agents.entry_tool_for）。
    """
    agent, config, store, calls = await _patched_route(
        monkeypatch, enabled=True, verdict=INTENT_OTHER, answer="答",
    )
    await chat_mod._route_turn(agent, config, "帮我查下这个月EV线故障统计", _ctx())
    assert calls["exclude"] == ["call_triage_agent"]
