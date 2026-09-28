# ============================================================
# L0 路由评测 runner：消息 → 工具选择
#
# 与 L1-L6（分诊内核）互补：这层测的是 Supervisor 的路由质量 ——
# 改 system prompt / 加删工具后跑它，路由退化在这里现形。
#
# 用法（需要 DASHSCOPE_API_KEY，发版前 / 改 prompt 后手动跑）：
#   python eval/run_routing_eval.py [--threshold 0.9]
#
# 机制：
#   - 用真实 supervisor 的 system_prompt + 工具【签名与描述】构建探针
#     agent，但工具体全部替换为无副作用桩（不查库、不调远端、不建单）；
#   - 温度 0，跑完整回合，收集全部被调用工具名；
#   - 业务工具集合 = 全部工具 - {search_memory, save_memory}（原则 1 的
#     记忆前奏不参与判定；允许合法链式调用：历史→判重→分诊）；
#   - 断言：expect_tools 任一 ∈ 被调集合；expect_not 任一被调即失败；
#     expect_none_business=true 断言业务工具零调用。
# ============================================================

import argparse
import asyncio
import json
import sys
from pathlib import Path

CASES_FILE = Path(__file__).parent / "cases" / "routing_cases.json"

# 记忆前奏（工作原则 1/2 的固定动作，不参与路由质量判定）
PREAMBLE_TOOLS = {"search_memory", "save_memory"}

# 工具实现 import 放函数内：直接跑脚本也能找到 src/
TOOLSET = None


def _load_cases() -> list[dict]:
    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    return data["cases"]


def judge(case: dict, called_tools: list[str]) -> list[str]:
    """纯函数判定：case + 被调用工具名序列 → 错误列表（空 = 通过）。

    ★ 单测锚点：判定语义锁在 tests/test_routing_eval.py。
    """
    business = [t for t in called_tools if t not in PREAMBLE_TOOLS]
    errors = []
    if case.get("expect_none_business"):
        if business:
            errors.append(f"期望零业务工具调用，实际调用了 {business}")
        return errors
    expect = case.get("expect_tools") or []
    if expect and not any(t in business for t in expect):
        errors.append(f"期望命中 {expect} 之一，业务工具实际调用 {business or '（无）'}")
    for forbidden in case.get("expect_not") or []:
        if forbidden in business:
            errors.append(f"禁止调用的 {forbidden} 被调用了")
    return errors


def _build_stub_tools():
    """把真实工具替换为同名同描述同 schema 的无副作用桩。"""
    from langchain_core.tools import StructuredTool

    from src.agents.supervisor_agent import get_supervisor_toolset

    stubs = []
    for tool in get_supervisor_toolset():
        async def _stub(**kwargs):
            return "（评测桩：调用已记录，无真实数据返回）"

        stubs.append(StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            coroutine=_stub,
            handle_tool_error=True,
        ))
    return stubs


async def run_case(agent, ctx, case: dict) -> list[str]:
    """跑一个 case，返回被调用工具名序列。"""
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": case["message"]}]},
        context=ctx,
    )
    called = []
    for m in result.get("messages", []):
        for tc in getattr(m, "tool_calls", None) or []:
            called.append(tc["name"])
    return called


async def main(threshold: float) -> int:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    from src.agents.supervisor_agent import SUPERVISOR_SYSTEM_PROMPT
    from src.core.config import get_settings
    from src.core.deps import UserContext

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0,
        timeout=60,
    )
    agent = create_agent(
        model=llm,
        tools=_build_stub_tools(),
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        context_schema=UserContext,
    )
    # 工程师上下文（带业务线）：覆盖最常用的路由场景
    ctx = UserContext(user_id="eval", session_id="routing-eval",
                      role="engineer", business_line="ev")

    cases = _load_cases()
    passed, failed = 0, []
    for case in cases:
        called = await run_case(agent, ctx, case)
        errors = judge(case, called)
        mark = "✓" if not errors else "✗"
        print(f"{mark} {case['id']}  tools={called}")
        for e in errors:
            print(f"    {e}")
        if errors:
            failed.append(case["id"])
        else:
            passed += 1

    accuracy = passed / len(cases)
    print(f"\nL0 路由准确率：{passed}/{len(cases)} = {accuracy:.0%}（门限 {threshold:.0%}）")
    if failed:
        print(f"失败用例：{', '.join(failed)}")
    return 0 if accuracy >= threshold else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.threshold)))
