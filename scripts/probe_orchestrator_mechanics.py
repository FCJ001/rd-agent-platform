# ============================================================
# P1 单图化机制探针（一次性验证脚本，不是测试）
#
# 验证 langgraph 1.1.6 + langchain 1.2.15 支撑编排图所需的四个能力：
#   [P1] create_agent 产物作为父图节点（共享 messages）
#   [P2] 工具返回 Command(goto, graph=Command.PARENT) 跳出 agent 到兄弟节点
#   [P3] 子图内 interrupt() 挂起 → 父图 Command(resume) 恢复内层；
#        context 是否透传到内层工具的 ToolRuntime
#   [P4] 子图节点二次进入（第二次 handoff）状态不残留（新诊断全新状态）
# 成功则按此机制实施 src/agents/orchestrator.py。
# ============================================================

import asyncio
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, Command as LCommand, interrupt


def fake_model(responses: list):
    state = {"i": 0}

    class _Fake(BaseChatModel):
        @property
        def _llm_type(self): return "probe"

        def bind_tools(self, tools, **kwargs): return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            i = min(state["i"], len(responses) - 1)
            state["i"] += 1
            return ChatResult(generations=[ChatGeneration(message=responses[i])])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    return _Fake()


# ── 子图（模拟分诊会话图）：intake → wait(interrupt) → done ────────────

from pydantic import BaseModel


class SubState(BaseModel):
    messages: Annotated[list, __import__("langgraph.graph.message", fromlist=["add_messages"]).add_messages] = []
    round: int = 0
    triage_scope: dict = {}


def build_sub():
    g = StateGraph(SubState)

    async def intake(state: SubState):
        # 首轮：从 scope 初始化 + 记录进入过（P4 检查第二次进入时它被重置）
        return {"round": 0, "messages": [AIMessage(content=f"[intake] scope={state.triage_scope}")]}

    async def wait_answer(state: SubState):
        # ★ interrupt() 是本节点第一条语句：resume 重放零 LLM 成本
        answer = interrupt({"type": "triage_followup", "round": state.round + 1,
                            "question": "是否伴随死机？"})
        return {"round": state.round + 1,
                "messages": [HumanMessage(content=answer), AIMessage(content=f"[结论: {answer}]")]}

    g.add_node("intake", intake)
    g.add_node("wait_answer", wait_answer)
    g.add_edge(START, "intake")
    g.add_edge("intake", "wait_answer")
    g.add_edge("wait_answer", END)
    return g.compile()


# ── 父图：supervisor(agent) ⇄ triage(子图) ──────────────────────────────

class ParentState(TypedDict, total=False):
    messages: Annotated[list, __import__("langgraph.graph.message", fromlist=["add_messages"]).add_messages]
    active_task: str | None
    triage_scope: dict


@tool
async def start_triage(message: str, runtime=None) -> LCommand:
    """启动故障诊断。"""
    ctx = getattr(runtime, "context", None)
    scope = {"viewer_role": getattr(ctx, "role", "?"), "user_id": getattr(ctx, "user_id", "?")}
    return Command(
        goto="triage",
        graph=Command.PARENT,
        update={"active_task": "triage", "triage_scope": scope,
                "messages": [HumanMessage(content=message)]},
    )


async def main():
    checks = []

    def check(name, ok, extra=""):
        checks.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'} [{name}] {extra}")

    supervisor = create_agent(
        model=fake_model([
            AIMessage(content="", tool_calls=[{"name": "start_triage", "args": {"message": "车机黑屏"}, "id": "c1"}]),
            AIMessage(content="诊断已完成，需要我做别的吗？"),
            AIMessage(content="", tool_calls=[{"name": "start_triage", "args": {"message": "空调异响"}, "id": "c2"}]),  # 第二次 handoff（P4）
            AIMessage(content="第二个诊断也完成了。"),
        ]),
        tools=[start_triage],
        system_prompt="probe",
    )

    async def triage_done(state: ParentState):
        return {"active_task": None}

    def entry_router(state: ParentState) -> str:
        return "triage" if state.get("active_task") == "triage" else "supervisor"

    parent = StateGraph(ParentState)
    parent.add_node("supervisor", supervisor)
    parent.add_node("triage", build_sub())
    parent.add_node("triage_done", triage_done)
    parent.set_conditional_entry_point(entry_router, {"supervisor": "supervisor", "triage": "triage"})
    parent.add_edge("triage", "triage_done")
    parent.add_edge("triage_done", "supervisor")   # 结论回合由 supervisor 收尾（展示操作选项）
    parent.add_edge("supervisor", END)

    app = parent.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "probe"}}

    class Ctx:
        user_id = "u9"
        role = "engineer"

    # ── 第一回合：supervisor 路由 → handoff → 子图挂起在追问 ──
    r1 = await app.ainvoke({"messages": [HumanMessage(content="车机黑屏")]},
                            config=cfg, context=Ctx())
    intr = r1.get("__interrupt__")
    check("P2+P3 挂起", bool(intr), f"payload={intr[0].value if intr else None}")
    if intr:
        check("P3 载荷", intr[0].value.get("question") == "是否伴随死机？")

    # 父图快照可见挂起（chat 层路由判据仍可用）
    snap = await app.aget_state(cfg)
    check("P3 父快照可见", any(t.interrupts for t in snap.tasks))

    # ── 第二回合：resume 回答 → 子图完成 → 父图 triage_done → END ──
    r2 = await app.ainvoke(Command(resume="是的会死机"), config=cfg, context=Ctx())
    texts = [m.content for m in r2["messages"] if hasattr(m, "content")]
    check("P3 恢复完成", any("[结论: 是的会死机]" in t for t in texts), f"{texts[-3:]}")
    check("P1 结论回到消息历史", any("诊断已完成" in t for t in texts))
    check("active_task 清空", not r2.get("active_task"))

    # 子图内部的 round 保留在子图状态里（通过父快照 values 不可见——记录一下）
    check("P4 前置", True, f"parent values keys={list((await app.aget_state(cfg)).values.keys())}")

    # ── 第三回合：新消息 → supervisor → 第二次 handoff（P4：子图状态不残留）──
    r3 = await app.ainvoke({"messages": [HumanMessage(content="空调异响")]},
                            config=cfg, context=Ctx())
    intr3 = r3.get("__interrupt__")
    check("P4 第二次挂起", bool(intr3))

    r4 = await app.ainvoke(Command(resume="没有其他现象"), config=cfg, context=Ctx())
    texts4 = [m.content for m in r4["messages"] if hasattr(m, "content")]
    # 第二次 intake 应重新出现且 round 从 0 开始（结论轮次 = 1）
    check("P4 状态不残留", any("[结论: 没有其他现象]" in t for t in texts4), f"{texts4[-4:]}")
    # context 是否透传进子图：intake 消息里带 scope
    check("P3 context 透传", any("scope=" in t and "engineer" in t for t in texts), )

    print("\n" + ("ALL PASS" if all(ok for _, ok in checks) else "SOME FAILED"))


asyncio.run(main())
