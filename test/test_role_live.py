"""端到端测试角色分层输出：调用真实 LLM 生成 4 种角色结论。"""

import asyncio

from src.agents.triage.graph import node_conclude, TriageDeps, _get_llm_chat
from src.agents.triage.state import TriageState, TriagePhase, CandidateCause
from src.infra.db import AsyncSessionLocal


async def test():
    llm = _get_llm_chat()

    async def _db_factory():
        async with AsyncSessionLocal() as session:
            yield session

    deps = TriageDeps(llm_json=None, llm_chat=llm, db_session_factory=_db_factory)

    roles = ["engineer", "business", "aftersales", "customer"]

    for role in roles:
        state = TriageState(
            session_id=f"role_test_{role}",
            phase=TriagePhase.CONCLUDE,
            viewer_role=role,
            confirmed_phenomena=["电池可用容量下降", "SOC跳变"],
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

        print(f"\n{'='*60}")
        print(f"角色: {role.upper()}")
        print(f"{'='*60}")

        result = await node_conclude(state, deps)
        msg = result.get("messages", [])
        if msg:
            print(msg[0].content[:600])
        print()


if __name__ == "__main__":
    asyncio.run(test())
