from enum import Enum
from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class TriagePhase(str, Enum):
    EXTRACT = "EXTRACT"
    QUERY = "QUERY"
    ASK = "ASK"
    CONCLUDE = "CONCLUDE"
    END = "__end__"


class CandidateCause(BaseModel):
    code: str
    name: str
    domain: str = ""
    business_line: str = ""
    confidence: float = 0.0
    base_confidence: float = 0.0
    matched_phenomena: list[str] = []
    all_phenomena: list[str] = []
    # 现象名 → 图谱 INDICATES.weight（反馈回写更新过），评分时调制证据强度
    phenomena_weight: dict[str, float] = {}
    # 可解释性：逐条证据的得分推导，跟结论一起输出
    evidence_trace: list[str] = []
    fix_way: str = ""
    fix_duration: str = ""
    verify_items: str = ""
    is_core_match: bool = False
    dtc_matched: list[str] = []
    related_config_items: list[dict] = []
    related_causes: list[str] = []


class TriageState(BaseModel):
    messages: Annotated[list, add_messages] = []
    phase: TriagePhase = TriagePhase.EXTRACT
    round: int = 0
    session_id: str = ""
    # 诊断发起人（users.id 的字符串形式）。落进 ai_triage_results.user_id，
    # 是「我上次的诊断」这个检索维度的依据，也是反馈接口的所有权来源
    user_id: str = ""
    issue_id: int | None = None
    # issue context (loaded from DB when issue_id is provided)
    issue_title: str = ""
    issue_desc: str = ""
    issue_dtc_snapshot: str = ""
    # symptom/phenomenon tracking
    confirmed_phenomena: list[str] = []
    denied_phenomena: list[str] = []
    # 现象/DTC → 证据来源（instrument/reported/confirmed/hedged），
    # 供置信度评分区分"诊断仪测过"和"好像是"（C1）
    phenomena_evidence: dict[str, str] = {}
    dtc_evidence: dict[str, str] = {}
    # candidates
    candidate_causes: list[CandidateCause] = []
    # DTC codes extracted from user input
    dtc_codes: list[str] = []
    # overall confidence
    confidence: float = 0.0
    # follow-up
    follow_up_questions: list[str] = []
    # conclusion
    diagnostic_summary: str = ""
    force_conclude: bool = False
    # phenomenon vocabulary cache (populated at graph build time)
    phenomenon_vocabulary: str = Field(default="", description="已知现象名+别名列表，用于 LLM prompt")
    # 当前查看结论的人的角色，影响输出格式（engineer/business/aftersales/customer）
    viewer_role: str = Field(default="customer", description="查看者角色")
