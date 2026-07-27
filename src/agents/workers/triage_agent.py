"""分诊 Agent 封装。装配 graph + vocabulary，提供 diagnose() 接口。"""

import uuid
from functools import lru_cache

from langchain_core.messages import HumanMessage

from src.agents.triage.state import TriageState, TriagePhase
from src.agents.triage.graph import (
    TriageDeps, build_triage_graph, load_phenomenon_vocabulary, _get_llm_json, _get_llm_chat,
)
from src.core.config import get_settings
from src.infra.db import AsyncSessionLocal

settings = get_settings()

# ── 简易内存态存储（生产换 Redis） ──
_session_states: dict[str, TriageState] = {}


class TriageAgent:
    """分诊 Agent。封装 StateGraph 的构建和调用。"""

    def __init__(self):
        self.llm_json = _get_llm_json()
        self.llm_chat = _get_llm_chat()
        self._vocabulary: str | None = None

    async def _get_vocabulary(self) -> str:
        if self._vocabulary is None:
            self._vocabulary = await load_phenomenon_vocabulary()
        return self._vocabulary

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
    ) -> dict:
        """
        执行一轮分诊诊断。

        Returns:
            dict with keys: session_id, status, round, normalized_phenomena,
            candidate_causes, confidence, follow_up_questions, diagnostic_summary
        """
        sid = session_id or str(uuid.uuid4())
        deps = self._build_deps()
        graph = build_triage_graph(deps)

        vocabulary = await self._get_vocabulary()

        # Load or create session state
        previous = _session_states.get(sid)
        if previous is not None:
            state = previous.model_copy()
            state.round = previous.round + 1
            state.messages.append(HumanMessage(content=raw_input))
            state.phase = TriagePhase.EXTRACT  # reset for this pass
        else:
            state = TriageState(
                session_id=sid,
                issue_id=issue_id,
                phenomenon_vocabulary=vocabulary,
            )
            state.messages.append(HumanMessage(content=raw_input))

        config = {"configurable": {"thread_id": sid}}
        result_dict = await graph.ainvoke(state, config=config)

        # Reconstruct TriageState from result dict and persist for next round
        result = TriageState(**result_dict)
        _session_states[sid] = result

        # Extract last AI message
        reply = ""
        for msg in reversed(result.messages):
            if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
                reply = msg.content
                break

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
        }


@lru_cache(maxsize=1)
def get_triage_agent() -> TriageAgent:
    return TriageAgent()
