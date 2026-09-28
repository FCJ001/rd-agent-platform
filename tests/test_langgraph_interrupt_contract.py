# ============================================================
# langgraph 挂起/恢复语义契约测试（[A]/[D]/[E]/[F]）
#
# 四条语义是 docs/design-side-assistant-bypass.md §2 的形态依据，
# 由 langgraph 1.1.6 + MemorySaver 实测得出。升级 langgraph /
# langgraph-checkpoint-redis 后必须先跑本文件：任何一条红 =
# 挂起恢复语义变了，chat 分流（_route_turn）与 worker_tools 的
# 快进结构都要重新推演，不能直接发版。
#
# 全部纯内存，CI 可跑。
# ============================================================

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing import TypedDict


class _LogState(TypedDict, total=False):
    log: list
    messages: list


def _build(node_fn):
    g = StateGraph(_LogState)
    g.add_node("n", node_fn)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile(checkpointer=MemorySaver())


# ════════════════════════════════════════════════════════════════════════
# [A] 挂起期间用新输入（非 Command(resume)）调用图：消息并进 state，
#     同一个 interrupt 原样重抛，图不前进。
#     → 「绕过 interrupt 把消息交给 Supervisor」不可行：消息被静默吞掉，
#       还留在历史里。这是旁路必须发生在 chat 入口层（图外）的根因。
# ════════════════════════════════════════════════════════════════════════

def test_A_new_input_during_suspend_does_not_advance():
    def node(state):
        answer = interrupt({"q": "q1"})
        return {"log": [f"a:{answer}"]}

    app = _build(node)
    cfg = {"configurable": {"thread_id": "a"}}

    r1 = app.invoke({"messages": ["u1"]}, cfg)
    assert r1["__interrupt__"][0].value["q"] == "q1"

    # 挂起期间塞新输入：不 resume，直接喂新消息
    r2 = app.invoke({"messages": ["u2"]}, cfg)
    # 同一个 interrupt 原样重抛（载荷不变），图没前进
    assert r2["__interrupt__"][0].value["q"] == "q1"
    # 新消息并进了 state（留在历史里），但没被任何节点消费
    assert "u2" in [m for m in r2["messages"]]

    # resume 后图只认 resume 值；u2 从未到达节点逻辑
    r3 = app.invoke(Command(resume="ans1"), cfg)
    assert r3["log"] == ["a:ans1"]


# ════════════════════════════════════════════════════════════════════════
# [D] 工具消费 resume 值之后再抛异常：值已提交（写入 checkpoint 的
#     resume 映射）；用户重答的新值会挂到下一个 interrupt 上。
#     → resume 轮被拒必须「原样上抛」且不能 return 文本（否则消费掉
#       interrupt 造成状态机错位）；也决定了离题判定绝不能发生在
#       工具消费 resume 值之后 —— 必须推到 chat 入口层。
# ════════════════════════════════════════════════════════════════════════

def test_D_raise_after_consume_commits_value_and_misaligns():
    flags = {"raise_once": True}

    def node(state):
        a1 = interrupt({"q": "q1"})
        if flags["raise_once"]:
            flags["raise_once"] = False
            raise RuntimeError("busy-after-consume")
        a2 = interrupt({"q": "q2"})
        return {"log": [f"r0:{a1}", f"r1:{a2}"]}

    app = _build(node)
    cfg = {"configurable": {"thread_id": "d"}}

    app.invoke({"messages": ["u1"]}, cfg)

    # 第一次 resume：值被 interrupt 消费后抛异常 → 异常穿出
    with pytest.raises(RuntimeError):
        app.invoke(Command(resume="ans1"), cfg)
    # checkpoint 未越过追问点
    snap = app.get_state(cfg)
    assert any(t.interrupts for t in snap.tasks)

    # 用户重答了不一样的新值 "ans2"：已提交的 "ans1" 不被替换，
    # 新值挂到下一个 interrupt 上 —— 两条回答与两个问题错位。
    r = app.invoke(Command(resume="ans2"), cfg)
    assert r["log"] == ["r0:ans1", "r1:ans2"]


# ════════════════════════════════════════════════════════════════════════
# [E] 工具在 interrupt() 之前抛异常（= 生产里闸门/会话锁的真实顺序）：
#     值不提交，重试时使用新值。
#     → 现有 busy 路径语义正确；这也是「判定分流必须在图调用之前」
#       对应的安全侧 —— 拒绝发生在消费之前，回答不会错位。
# ════════════════════════════════════════════════════════════════════════

def test_E_raise_before_interrupt_keeps_value_uncommitted():
    store = {}
    flags = {"reject_once": True}

    def node(state, config):
        tid = config["configurable"]["thread_id"]
        saved = store.get(tid)
        if saved is None:
            round_no, log = 0, []
        else:
            round_no, log = saved
            # ★ 拒绝发生在任何 interrupt() 之前（gate/lock 的真实顺序）
            if flags["reject_once"]:
                flags["reject_once"] = False
                raise RuntimeError("busy-before-interrupt")
            for _ in range(round_no):
                interrupt({"q": "ff"})
        answer = interrupt({"q": "q1"})
        return {"log": log + [f"r0:{answer}"]}

    app = _build(node)
    cfg = {"configurable": {"thread_id": "e"}}

    # 首轮：round-0 工作发生在 interrupt 之前（与真实 worker 一致），
    # 挂起时 store 落盘 round=0（已解决 interrupt 数 = 0）
    app.invoke({"messages": ["u1"]}, cfg)
    store["e"] = (0, [])

    # resume 被拒（interrupt 之前抛）：异常穿出，值未提交
    with pytest.raises(RuntimeError):
        app.invoke(Command(resume="ans1"), cfg)
    assert any(t.interrupts for t in app.get_state(cfg).tasks)

    # 重试用新值：answer 拿到的是新值，旧值没有残留
    r = app.invoke(Command(resume="ans2"), cfg)
    assert r["log"] == ["r0:ans2"]


# ════════════════════════════════════════════════════════════════════════
# [F] park（工具返回文本消费 interrupt、进度留外部 store、之后新 task
#     重问）朴素沿用 state.round 快进的缺陷：park 后下一次调用会停在
#     快进空载荷上 → 用户看到兜底文案，真实消息没有进入诊断。
#     → park 被否决、改为入口层旁路的实测依据（bypass 设计 §8）。
# ════════════════════════════════════════════════════════════════════════

def test_F_park_naive_fastforward_stops_on_empty_payload():
    store = {}

    def node(state, config):
        tid = config["configurable"]["thread_id"]
        saved = store.get(tid)
        if saved is None:
            round_no, log = 0, []
        else:
            round_no, log = saved
            for _ in range(round_no):
                interrupt({"q": ""})   # 朴素快进：空载荷
        answer = interrupt({"q": f"real-q{round_no}"})
        store[tid] = (round_no + 1, log + [f"r{round_no}:{answer}"])
        if round_no == 0:
            # park：第一轮拿到回答后不开下一个 interrupt，直接返回文本
            # → 节点完成。快进结构赖以工作的「节点一直处于未完成态」被打破
            return {"log": log + ["parked"]}
        return {"log": log + [f"done-r{round_no}"]}

    app = _build(node)
    cfg = {"configurable": {"thread_id": "f"}}

    app.invoke({"messages": ["故障描述"]}, cfg)          # 挂在 real-q0
    r1 = app.invoke(Command(resume="ans"), cfg)          # 消费回答 → park 返回
    assert r1["log"] == ["parked"]

    # 用户带着新消息回来（走 supervisor → 工具再次被调，全新 task）
    r2 = app.invoke({"messages": ["继续，追加现象 X"]}, cfg)
    # 朴素快进 range(1) 里的 interrupt({"q": ""}) 被原样抛出挂起：
    # 图停在了空载荷上 —— 用户看到的是兜底文案「请继续描述故障细节。」，
    # 真实消息没进诊断。park 被否决（bypass 设计 §8）的实测依据。
    intr = r2["__interrupt__"][0].value
    assert intr["q"] == "", f"期望停在快进空载荷，实际: {intr}"
    # store 仍停在 park 点，轮次没有推进
    assert store["f"] == (1, ["r0:ans"])
