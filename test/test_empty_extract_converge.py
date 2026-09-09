"""回归测试：L1 全程抽不出有效现象时，分诊必须在 MAX_ROUNDS 处强制收敛，不能无限追问。

背景：多轮追问由调用方循环实现（chat.py / triage.py 每轮调 run_triage）。
MAX_ROUNDS=5 原本只在 query_candidates 节点判收敛 —— L1 一直抽空时根本到不了
query，会无限追问。修复后在 ask_details 加了同款守卫。

本测试用假 LLM 驱动真实 StateGraph 多轮，验证：
  - 抽空路径每轮 status == "asking"（继续循环）
  - 轮次耗尽后 status == "converged"，diagnostic_summary 提示重新描述
  - 收敛后不再追问（follow_up_questions 为空）
"""

import asyncio

from src.agents.triage.confidence import MAX_ROUNDS
from src.agents.triage.graph import build_triage_graph, TriageDeps
from src.agents.triage.state import TriageState, TriagePhase


class _FakeJSONLLM:
    """假 LLM（llm_json）：永远返回空 phenomena —— 模拟 L1 抽不出。"""

    async def ainvoke(self, messages):
        class _Resp:
            content = '{"phenomena": [], "dtc_codes": []}'
        return _Resp()


class _FakeChatLLM:
    """假 LLM（llm_chat）：返回固定追问文案。"""

    async def ainvoke(self, messages):
        class _Resp:
            content = "请再描述一下具体的故障现象？"
        return _Resp()


def _fake_db():
    yield None


def build_deps():
    return TriageDeps(
        llm_json=_FakeJSONLLM(),
        llm_chat=_FakeChatLLM(),
        db_session_factory=lambda: _fake_db(),
    )


async def _run_one_round(deps, state: TriageState) -> tuple[TriageState, str, list[str]]:
    """跑一轮分诊，返回 (新状态, 助手回复, 追问列表)。"""
    graph = build_triage_graph(deps)
    result = TriageState(**await graph.ainvoke(state))
    reply = result.messages[-1].content if result.messages else ""
    return result, reply, result.follow_up_questions


async def run():
    deps = build_deps()
    state = TriageState(session_id="empty-extract-loop")

    turns = []
    for i in range(MAX_ROUNDS + 2):  # 故意多跑 2 轮，证明守卫触发后不再追问
        state, reply, follow_ups = await _run_one_round(deps, state)
        state.round += 1
        state.phase = TriagePhase.EXTRACT
        state.messages = []  # 模拟新一轮用户输入（循环内不追加，避免历史累积）
        turns.append((state.force_conclude, state.diagnostic_summary, follow_ups))

    ok = True

    # 前 MAX_ROUNDS 轮（round 0..4）：在守卫触发前应该是追问状态
    for i in range(MAX_ROUNDS):
        force, summary, follow_ups = turns[i]
        if force or summary or not follow_ups:
            ok = False
            print(f"[FAIL] 第{i+1}轮（round={i}）应仍在追问: force={force} summary={summary!r} follow_ups={follow_ups}")

    # 守卫触发轮（round == MAX_ROUNDS）必须强制收敛、不再追问
    for i in (MAX_ROUNDS, MAX_ROUNDS + 1):
        force, summary, follow_ups = turns[i]
        if not force:
            ok = False
            print(f"[FAIL] 第{i+1}轮（round={i}）应强制收敛，实际未收敛")
        if not summary:
            ok = False
            print(f"[FAIL] 第{i+1}轮强制收敛后 diagnostic_summary 不应为空")
        if follow_ups:
            ok = False
            print(f"[FAIL] 第{i+1}轮强制收敛后不应再追问，实际 follow_ups={follow_ups}")

    print(f"\nMAX_ROUNDS={MAX_ROUNDS}，共模拟 {MAX_ROUNDS + 2} 轮")
    print("结果:", "PASS — 空提取路径在轮次耗尽处强制收敛，不再无限追问" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ok = asyncio.run(run())
    raise SystemExit(0 if ok else 1)
