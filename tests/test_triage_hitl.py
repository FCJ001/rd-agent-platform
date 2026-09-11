# ============================================================
# B2 重构回归测试：interrupt 驱动的分诊多轮追问
#
# 锁住四件事：
#   1. langgraph 语义：同一节点内顺序 interrupt() 按调用序匹配 resume 值
#   2. call_triage_agent 的快进结构：外部 store 落盘 + 空转消耗，
#      resume 重放时不重跑已完成轮次的工作
#   3. 逃生口：resume 退出指令 → 清 store、正常返回
#   4. 真实栈集成：interrupt 能穿透 create_agent/ToolNode，
#      aget_state 快照能作为 chat 路由判据
#
# 全部纯内存（stub redis + fake chat model），CI 可跑。
# ============================================================

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing import TypedDict

from src.agents.triage.session_store import TriageSessionStore, is_triage_exit
from src.agents.triage.state import TriageState


def _make_fake_model(responses: list):
    """依次返回 responses 的 fake chat model，支持 bind_tools（create_agent 必需）。"""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    state = {"i": 0}

    class _Fake(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake-tool-model"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            i = min(state["i"], len(responses) - 1)
            state["i"] += 1
            return ChatResult(generations=[ChatGeneration(message=responses[i])])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    return _Fake()


class StubRedis:
    """异步接口的最小 redis stub（set/get/delete）。"""

    def __init__(self):
        self.data = {}

    async def set(self, key, value, ex=None):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, *keys):
        for k in keys:
            self.data.pop(k, None)


# ════════════════════════════════════════════════════════════════════════
# 1. langgraph 顺序 interrupt 语义回归
# ════════════════════════════════════════════════════════════════════════

class _SemState(TypedDict, total=False):
    log: list


def test_sequential_interrupt_order():
    """第 k 个 interrupt() 必须返回第 k 个 resume 值 —— 快进结构的前提。"""
    calls = {"real_work": 0}

    def node(state):
        log = list(state.get("log", []))
        for i in range(3):
            answer = interrupt({"q": f"round{i}"})
            calls["real_work"] += 1
            log.append(f"round{i}:{answer}")
        return {"log": log}

    g = StateGraph(_SemState)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "sem"}}

    r = app.invoke({}, cfg)
    assert "__interrupt__" in r

    for ans in ("a1", "a2", "a3"):
        r = app.invoke(Command(resume=ans), cfg)

    assert r["log"] == ["round0:a1", "round1:a2", "round2:a3"]
    # 纯回放：每个 resume 都从头重放已完成轮次（这正是工具需要快进的原因）
    assert calls["real_work"] == 3 + 2 + 1


# ════════════════════════════════════════════════════════════════════════
# 2. 工具快进结构：外部 store 落盘 + 空转消耗 → 每轮工作只执行一次
# ════════════════════════════════════════════════════════════════════════

class _FFState(TypedDict, total=False):
    log: list


def test_fast_forward_no_replay():
    """与 call_triage_agent 相同的结构：进度存外部 store，重放时空转对齐。"""
    calls = []
    store = {}
    TOTAL_ROUNDS = 3

    def node(state, config):
        tid = config["configurable"]["thread_id"]
        saved = store.get(tid)
        if saved is None:
            round_no, log = 0, []
        else:
            round_no, log = saved
            # 快进空转：消耗已完成轮次的 interrupt（历史回答，丢弃）
            for _ in range(round_no):
                interrupt({"q": "fast-forward"})
        while round_no < TOTAL_ROUNDS:
            answer = interrupt({"q": f"round{round_no}"})
            calls.append((round_no, answer))
            log = log + [f"round{round_no}:{answer}"]
            round_no += 1
            store[tid] = (round_no, log)  # interrupt 前落盘
        store.pop(tid, None)
        return {"log": log}

    g = StateGraph(_FFState)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "ff"}}

    app.invoke({}, cfg)
    for ans in ("a1", "a2", "a3"):
        app.invoke(Command(resume=ans), cfg)

    # 每轮工作恰好一次，答案严格对齐
    assert calls == [(0, "a1"), (1, "a2"), (2, "a3")]
    assert store.get("ff") is None


def test_fast_forward_exit_clears_store():
    calls = []
    store = {}

    def node(state, config):
        tid = config["configurable"]["thread_id"]
        saved = store.get(tid)
        if saved is None:
            round_no, log = 0, []
        else:
            round_no, log = saved
            for _ in range(round_no):
                interrupt({"q": "fast-forward"})
        while True:
            answer = interrupt({"q": f"round{round_no}"})
            if answer == "EXIT":
                store.pop(tid, None)
                return {"log": log + ["exited"]}
            calls.append((round_no, answer))
            round_no += 1
            store[tid] = (round_no, log)

    g = StateGraph(_FFState)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "ex"}}

    app.invoke({}, cfg)
    r = app.invoke(Command(resume="EXIT"), cfg)

    assert r["log"] == ["exited"]
    assert calls == []
    assert store.get("ex") is None


# ════════════════════════════════════════════════════════════════════════
# 3. TriageSessionStore + 退出指令
# ════════════════════════════════════════════════════════════════════════

async def test_session_store_roundtrip():
    redis = StubRedis()
    store = TriageSessionStore(redis)

    assert await store.load("t:1") is None

    state = TriageState(session_id="t:1", confirmed_phenomena=["中控屏显示异常"])
    await store.save("t:1", state, reply="是否伴随死机？")
    progress = await store.load("t:1")
    assert progress.reply == "是否伴随死机？"
    assert progress.state.confirmed_phenomena == ["中控屏显示异常"]

    await store.clear("t:1")
    assert await store.load("t:1") is None
    assert await store.clear("t:1") is None  # 幂等


def test_is_triage_exit():
    assert is_triage_exit("退出诊断")
    assert is_triage_exit(" 取消。")
    assert is_triage_exit("不诊了")
    assert not is_triage_exit("中控屏黑屏了，想取消之前的误解")
    assert not is_triage_exit("充电时跳闸")


# ════════════════════════════════════════════════════════════════════════
# 4. 真实栈集成：interrupt 穿透 create_agent/ToolNode + 快照路由判据
# ════════════════════════════════════════════════════════════════════════

async def test_create_agent_triage_interrupt_flow():
    """模拟 call_triage_agent 的完整生命周期：
    首轮 → 图暂停（interrupt）→ 快照检测到挂起 → resume 用户回答 →
    工具无重放地完成 → 模型收尾。"""
    work_calls = []
    redis = StubRedis()
    store = TriageSessionStore(redis)

    @tool
    async def stub_triage(message: str) -> str:
        """模拟分诊工具：首轮追问一次，收到回答后给结论。"""
        saved = await store.load("u1:s1")
        if saved is None:
            work_calls.append(("round0", message))
            await store.save("u1:s1", TriageState(session_id="u1:s1"), "是否黑屏？")
        else:
            for _ in range(saved.state.round):
                interrupt({"type": "fast-forward"})

        answer = interrupt({"type": "triage_followup", "round": 1, "question": "是否黑屏？"})
        if is_triage_exit(answer):
            await store.clear("u1:s1")
            return "分诊已退出"
        work_calls.append(("round1", answer))
        await store.clear("u1:s1")
        return f"结论：屏幕排线松动（基于回答：{answer}）"

    model = _make_fake_model([
        AIMessage(content="", tool_calls=[
            {"name": "stub_triage", "args": {"message": "车机黑屏"}, "id": "call_1"}
        ]),
        AIMessage(content="诊断完成：屏幕排线松动，建议到店检修。"),
    ])

    agent = create_agent(model=model, tools=[stub_triage], system_prompt="test", checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "u1:s1"}}

    # ── 首轮：模型调工具 → 工具暂停在 interrupt ──
    r1 = await agent.ainvoke({"messages": [{"role": "user", "content": "车机黑屏"}]}, config)
    assert "__interrupt__" in r1, r1
    payload = r1["__interrupt__"][0].value
    assert payload["type"] == "triage_followup"
    assert payload["question"] == "是否黑屏？"
    # 回归保护：暂停轮最后一条消息是 tool_call（content 为空），
    # chat 层必须从 interrupt payload 取追问，不能退回 messages[-1]
    assert r1["messages"][-1].content == ""

    # chat 层的回复提取：暂停轮取 payload，正常轮取最后一条消息
    from src.api.routers.chat import _reply_from_result
    assert _reply_from_result(r1) == "是否黑屏？"

    # chat.py 的路由判据：快照检测到挂起
    snapshot = await agent.aget_state(config)
    assert any(t.interrupts for t in snapshot.tasks)

    # ── resume：工具完成，无重放 ──
    r2 = await agent.ainvoke(Command(resume="是黑屏，重启也没用"), config)

    assert work_calls == [("round0", "车机黑屏"), ("round1", "是黑屏，重启也没用")]
    tool_msg = [m for m in r2["messages"] if m.type == "tool"][0]
    assert "屏幕排线松动" in tool_msg.content
    assert r2["messages"][-1].content == "诊断完成：屏幕排线松动，建议到店检修。"
    # 恢复轮图已跑完 → 回复取最后一条消息
    from src.api.routers.chat import _reply_from_result
    assert _reply_from_result(r2) == "诊断完成：屏幕排线松动，建议到店检修。"

    # 完成后快照无挂起 → 后续消息走正常路由
    snapshot = await agent.aget_state(config)
    assert not any(t.interrupts for t in snapshot.tasks)
    assert await store.load("u1:s1") is None


async def test_create_agent_exit_during_triage():
    """追问中 resume 退出指令 → 工具清 store 并返回提示，图正常走完。"""
    redis = StubRedis()
    store = TriageSessionStore(redis)

    @tool
    async def stub_triage(message: str) -> str:
        """模拟分诊工具。"""
        await store.save("u2:s2", TriageState(session_id="u2:s2"), "追问")
        answer = interrupt({"type": "triage_followup"})
        if is_triage_exit(answer):
            await store.clear("u2:s2")
            return "分诊已按你的要求退出，本轮诊断作废。"
        return "结论"

    model = _make_fake_model([
        AIMessage(content="", tool_calls=[
            {"name": "stub_triage", "args": {"message": "异响"}, "id": "call_1"}
        ]),
        AIMessage(content="好的，已为你退出诊断。"),
    ])

    agent = create_agent(model=model, tools=[stub_triage], system_prompt="test", checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "u2:s2"}}

    r1 = await agent.ainvoke({"messages": [{"role": "user", "content": "异响"}]}, config)
    assert "__interrupt__" in r1

    r2 = await agent.ainvoke(Command(resume="退出诊断"), config)
    tool_msg = [m for m in r2["messages"] if m.type == "tool"][0]
    assert "退出" in tool_msg.content
    assert await store.load("u2:s2") is None
