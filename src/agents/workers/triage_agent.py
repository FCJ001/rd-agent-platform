"""分诊 Agent 封装。提供 diagnose() 接口，会话管理由调用方负责。"""

import uuid
from functools import lru_cache

from src.agents.triage.state import TriageState, TriagePhase
from src.agents.triage.graph import (
    TriageDeps, run_triage, _get_llm_json, _get_llm_chat,
)
from src.core.config import get_settings
from src.infra.db import AsyncSessionLocal

settings = get_settings()


class TriageAgent:
    """分诊 Agent。封装 StateGraph 的构建和调用。"""

    def __init__(self):
        self.llm_json = _get_llm_json()
        self.llm_chat = _get_llm_chat()

    def _build_deps(self) -> TriageDeps:
        async def _db_factory():
            async with AsyncSessionLocal() as session:
                yield session
        return TriageDeps(llm_json=self.llm_json, llm_chat=self.llm_chat, db_session_factory=_db_factory)

    async def diagnose(
        self,
        raw_input: str,
        session_id: str | None = None,
        issue_id: int | None = None,
        existing_state: dict | None = None,
    ) -> dict:
        """
        执行一轮分诊诊断。

        Args:
            raw_input: 故障描述
            session_id: 会话 ID（新会话自动生成）
            issue_id: 关联问题单 ID
            existing_state: 上一轮 TriageState 的序列化 dict（多轮时由调用方传入）

        Returns:
            dict with keys: session_id, status, round, normalized_phenomena,
            candidate_causes, confidence, follow_up_questions, diagnostic_summary
        """
        sid = session_id or str(uuid.uuid4())
        deps = self._build_deps()

        # Deserialize existing state if provided
        prev_state = None
        if existing_state:
            prev_state = TriageState(**existing_state)

        reply, result = await run_triage(
            user_message=raw_input,
            thread_id=sid,
            deps=deps,
            existing_state=prev_state,
        )

        # Marshal candidates to dicts
        candidates_out = []
        for c in result.candidate_causes:
            candidates_out.append({
                "code": c.code,
                "name": c.name,
                "domain": c.domain,
                "confidence": c.confidence,
                "base_confidence": c.base_confidence,
                "matched_phenomena": c.matched_phenomena,
                "all_phenomena": c.all_phenomena,
                "fix_way": c.fix_way or "",
                "fix_duration": c.fix_duration or "",
                "verify_items": c.verify_items or "",
                "is_core_match": c.is_core_match,
                "dtc_matched": c.dtc_matched,
            })

        status_map = {
            TriagePhase.CONCLUDE: "converged",
            TriagePhase.ASK: "asking",
        }
        status = status_map.get(result.phase, "converged")

        return {
            "session_id": sid,
            "status": "max_turns" if result.force_conclude and status == "converged" else status,
            "round": result.round,
            "normalized_phenomena": result.confirmed_phenomena,
            "candidate_causes": candidates_out,
            "confidence": result.confidence,
            "follow_up_questions": result.follow_up_questions if status == "asking" else [],
            "diagnostic_summary": result.diagnostic_summary if status == "converged" else "",
            "_state": result.model_dump(),  # 调用方可缓存用于下一轮
        }


@lru_cache(maxsize=1)
def get_triage_agent() -> TriageAgent:
    return TriageAgent()
