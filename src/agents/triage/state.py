from enum import Enum
from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class TriagePhase(str, Enum):
    EXTRACT = "EXTRACT"
    QUERY = "QUERY"
    ASK = "ASK"
    CONCLUDE = "CONCLUDE"


class CandidateCause(BaseModel):
    code: str
    name: str
    domain: str = ""
    business_line: str = ""
    confidence: float = 0.0
    base_confidence: float = 0.0
    matched_phenomena: list[str] = []
    all_phenomena: list[str] = []
    fix_way: str = ""
    fix_duration: str = ""
    verify_items: str = ""
    is_core_match: bool = False
    dtc_matched: list[str] = []


class TriageState(BaseModel):
    messages: Annotated[list, add_messages] = []
    phase: TriagePhase = TriagePhase.EXTRACT
    round: int = 0
    session_id: str = ""
    issue_id: int | None = None
    # issue context (loaded from DB when issue_id is provided)
    issue_title: str = ""
    issue_desc: str = ""
    issue_dtc_snapshot: str = ""
    # symptom/phenomenon tracking
    confirmed_phenomena: list[str] = []
    denied_phenomena: list[str] = []
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
