"""测试角色分层输出：同一个故障，4 种角色看到不同结论。"""

import asyncio

from src.agents.triage.graph import build_triage_graph, TriageDeps
from src.agents.triage.state import TriageState, TriagePhase, CandidateCause
from langchain_core.messages import HumanMessage


async def test():
    deps = TriageDeps(
        llm_json=None,   # 不调用 LLM，只测 prompt 选择逻辑
        llm_chat=None,
        db_session_factory=lambda: _fake_db(),
    )

    # 构造一个已收敛的诊断状态（跳过 LLM 调用，直接测 node_conclude）
    for role in ["engineer", "business", "aftersales", "customer"]:
        state = TriageState(
            session_id="test",
            phase=TriagePhase.CONCLUDE,
            viewer_role=role,
            confirmed_phenomena=["电池可用容量下降"],
            denied_phenomena=["充电异常"],
            dtc_codes=["P0A7F"],
            candidate_causes=[
                CandidateCause(
                    code="RC-EV-0012",
                    name="BMS SOC估算算法漂移",
                    domain="电池系统域",
                    confidence=0.87,
                    fix_way="升级BMS固件到v3.2.1,重新标定SOC",
                    fix_duration="3-5个工作日",
                    verify_items="读取BMS日志,测量单体电压,检查SOC跳变点",
                )
            ],
            diagnostic_summary="",
            round=2,
        )

        # 直接看选中的 prompt 模板
        from src.agents.triage.prompts import ROLE_PROMPTS

        prompt = ROLE_PROMPTS[role].format(
            confirmed_phenomena="、".join(state.confirmed_phenomena),
            denied_phenomena="、".join(state.denied_phenomena),
            dtc_codes="、".join(state.dtc_codes),
            primary_cause=state.candidate_causes[0].name,
            confidence=state.candidate_causes[0].confidence,
            suspected_causes="无",
            domain=state.candidate_causes[0].domain,
            force_conclude=False,
        )

        # 提取 prompt 模板的特征关键词（即角色关注点）
        indicator = {
            "engineer":    "技术机理",
            "business":    "项目时间线",
            "aftersales":  "客户沟通",
            "customer":    "通俗易懂",
        }
        key = indicator.get(role, "")

        print(f"--- {role.upper()} (关键特征: {key}) ---")
        # 打印 prompt 中「要求」那部分
        lines = prompt.split("\n")
        in_requirements = False
        for line in lines:
            if "关注点" in line or "要求" in line.lower():
                in_requirements = True
                continue
            if in_requirements and line.strip().startswith(("要求", "直接", "请", "{")):
                continue
            if in_requirements and line.strip():
                print(f"  {line.strip()}")
        print()


def _fake_db():
    """空生成器，供 TriageDeps 初始化用。"""
    yield None


if __name__ == "__main__":
    asyncio.run(test())
