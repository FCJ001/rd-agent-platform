# ============================================================
# ALM 数据 SQLAlchemy Model 定义
#
# 通用设计（沿用 src/core/base_model.py）：
# - 继承 BaseModel，自带 id / created_at / updated_at
# - 字段命名用英文，comment 写中文业务含义
# - 长文本字段用 Text，结构化字段用 String/Integer
# - 关联关系只通过外键维护，不在 ORM 层定义 relationship（查询走 Repository）
#
# 分组：
#   ① alm_*        平台镜像表，Agent 只读，由 sync.py 增量同步
#   ② 诊断知识     自建主数据，同时导入 Neo4j
#   ③ ai_*         AI 影子产物，与平台原始数据物理隔离
#
# ★ 新增 model 后必须在 alembic/env.py 补 import，否则 autogenerate 静默少表
# ============================================================

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.core.base_model import BaseModel

# ==========================================================
# ① ALM 平台镜像表
# 来源：定时增量同步（按 updated_at 水位）+ Webhook 事件
# 原则：Agent 只读。所有 AI 产出写到 ai_* 表，绝不污染镜像
# ==========================================================


class AlmIssue(BaseModel):
    """问题单镜像（分诊的输入源）"""

    __tablename__ = "alm_issues"

    issue_no: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="平台问题单号，如 ISS-2025-00123")
    title: Mapped[str] = mapped_column(String(500), nullable=False, comment="问题标题")
    description: Mapped[str | None] = mapped_column(Text, comment="问题详述（客户投诉原文 / 工程师描述）")
    # ★ 四类反馈来源，决定三层标准化第一层的抽取策略和最终输出话术
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="customer",
                                        comment="来源：engineer/business/aftersales/customer")
    business_line: Mapped[str] = mapped_column(String(10), nullable=False, default="ev", comment="业务线：ev=电动化 / ia=智能化")
    severity: Mapped[str] = mapped_column(String(20), default="normal", comment="严重度：blocker/critical/normal/minor")
    status: Mapped[str] = mapped_column(String(20), default="open", comment="状态：open/analyzing/fixing/verified/closed")
    model_code: Mapped[str | None] = mapped_column(String(50), comment="车型代号（脱敏后），如 EV-A01")
    sw_version: Mapped[str | None] = mapped_column(String(50), comment="软件版本，如 2024.32.5")
    vin: Mapped[str | None] = mapped_column(String(32), comment="车架号（脱敏，仅留后 6 位）")
    dtc_snapshot: Mapped[str | None] = mapped_column(String(500), comment="上报时的 DTC 快照，逗号分隔")
    reporter_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), comment="上报人")
    owner_domain_id: Mapped[int | None] = mapped_column(ForeignKey("owner_domains.id", ondelete="SET NULL"), comment="当前责任域")
    external_ref: Mapped[str | None] = mapped_column(String(100), comment="外部溯源，如 nhtsa:11512345")

    __table_args__ = (
        Index("ix_alm_issues_status", "status"),
        Index("ix_alm_issues_line_model", "business_line", "model_code"),
        Index("ix_alm_issues_owner", "owner_domain_id"),
    )


class AlmRequirement(BaseModel):
    """需求单镜像"""

    __tablename__ = "alm_requirements"

    req_no: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="需求编号，如 REQ-EV-0042")
    title: Mapped[str] = mapped_column(String(500), nullable=False, comment="需求标题")
    description: Mapped[str | None] = mapped_column(Text, comment="需求描述")
    business_line: Mapped[str] = mapped_column(String(10), nullable=False, default="ev", comment="业务线")
    priority: Mapped[str] = mapped_column(String(20), default="P2", comment="优先级：P0/P1/P2/P3")
    status: Mapped[str] = mapped_column(String(20), default="open", comment="状态：draft/open/developing/verified/closed")
    baseline_id: Mapped[int | None] = mapped_column(ForeignKey("alm_baselines.id", ondelete="SET NULL"), comment="目标基线")

    __table_args__ = (Index("ix_alm_req_status", "status"),)


class AlmChangeRequest(BaseModel):
    """变更单镜像 —— 变更影响分析 Worker 的输入"""

    __tablename__ = "alm_change_requests"

    cr_no: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="变更单号，如 CR-2025-0088")
    title: Mapped[str] = mapped_column(String(500), nullable=False, comment="变更标题")
    reason: Mapped[str | None] = mapped_column(Text, comment="变更原因")
    scope_desc: Mapped[str | None] = mapped_column(Text, comment="变更范围自然语言描述（LLM 解析入口）")
    business_line: Mapped[str] = mapped_column(String(10), nullable=False, default="ev", comment="业务线")
    status: Mapped[str] = mapped_column(String(20), default="submitted", comment="状态：submitted/reviewing/approved/rejected/done")
    target_baseline_id: Mapped[int | None] = mapped_column(ForeignKey("alm_baselines.id", ondelete="SET NULL"), comment="目标基线")
    source_issue_id: Mapped[int | None] = mapped_column(ForeignKey("alm_issues.id", ondelete="SET NULL"), comment="触发该变更的问题单")

    __table_args__ = (Index("ix_alm_cr_status", "status"),)


class AlmConfigItem(BaseModel):
    """配置项镜像 —— 软硬件模块，依赖链的节点"""

    __tablename__ = "alm_config_items"

    ci_no: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="配置项编号，如 CI-BMS-001")
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="配置项名称，如 电池管理系统BMS")
    alias: Mapped[str | None] = mapped_column(String(500), comment="别名，逗号分隔（三层标准化第一层用）")
    category: Mapped[str] = mapped_column(String(30), default="software", comment="类别：software/hardware/calibration/doc")
    module: Mapped[str | None] = mapped_column(String(100), comment="所属模块，用于基线冲突二级 module_match")
    supplier: Mapped[str | None] = mapped_column(String(200), comment="供应商")
    part_number: Mapped[str | None] = mapped_column(String(100), comment="零件号/软件件号")
    sw_version: Mapped[str | None] = mapped_column(String(50), comment="当前软件版本")
    # ★ 功能安全标记，命中则 impact 分析强制升级 risk_level
    is_safety_related: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否功能安全相关（ISO 26262）")
    lifecycle_status: Mapped[str] = mapped_column(String(20), default="active", comment="生命周期：dev/active/frozen/obsolete")
    business_line: Mapped[str] = mapped_column(String(10), nullable=False, default="ev", comment="业务线")

    __table_args__ = (
        Index("ix_alm_ci_module", "module"),
        Index("ix_alm_ci_safety", "is_safety_related"),
    )


class AlmBaseline(BaseModel):
    """基线镜像 —— 冻结后阻塞在途需求，杀手级 Cypher 的起点"""

    __tablename__ = "alm_baselines"

    baseline_no: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="基线编号，如 BL-EV-2025Q1")
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="基线名称")
    business_line: Mapped[str] = mapped_column(String(10), nullable=False, default="ev", comment="业务线")
    is_frozen: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否已冻结")
    freeze_date: Mapped[str | None] = mapped_column(String(20), comment="冻结日期 YYYY-MM-DD")
    release_date: Mapped[str | None] = mapped_column(String(20), comment="计划发布日期 YYYY-MM-DD")

    __table_args__ = (Index("ix_alm_baseline_frozen", "is_frozen"),)


# ==========================================================
# ② 诊断知识主数据（自建，同时导入 Neo4j）
# ==========================================================


class OwnerDomain(BaseModel):
    """责任域（分诊路径的终点）"""

    __tablename__ = "owner_domains"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, comment="责任域名称，如：电池系统域")
    business_line: Mapped[str] = mapped_column(String(10), nullable=False, default="ev", comment="业务线")
    description: Mapped[str | None] = mapped_column(Text, comment="职责范围")
    owner_name: Mapped[str | None] = mapped_column(String(100), comment="域负责人")


class RootCause(BaseModel):
    """根因库（分诊的候选实体）"""

    __tablename__ = "root_causes"

    code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False, comment="根因编码，如 RC-EV-0012")
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, comment="根因名称")
    domain_id: Mapped[int | None] = mapped_column(ForeignKey("owner_domains.id", ondelete="SET NULL"), comment="责任域 ID")
    business_line: Mapped[str] = mapped_column(String(10), nullable=False, default="ev", comment="业务线")
    description: Mapped[str | None] = mapped_column(Text, comment="根因描述")
    cause: Mapped[str | None] = mapped_column(Text, comment="成因机理")
    prevent: Mapped[str | None] = mapped_column(Text, comment="预防措施")
    fix_way: Mapped[str | None] = mapped_column(Text, comment="处置方式，逗号分隔")
    fix_duration: Mapped[str | None] = mapped_column(String(100), comment="处置周期，如：2-5个工作日")
    fix_success_rate: Mapped[str | None] = mapped_column(String(50), comment="一次修复成功率，如：85%")
    easy_hit: Mapped[str | None] = mapped_column(Text, comment="易发场景/车型/工况")
    cost_money: Mapped[str | None] = mapped_column(String(100), comment="单车处置成本区间")
    # ★ 分诊结论里给工程师的「下一步做什么」
    verify_items: Mapped[str | None] = mapped_column(Text, comment="验证项，逗号分隔，如：读取BMS日志,测单体压差")

    __table_args__ = (
        Index("ix_root_causes_name", "name"),
        Index("ix_root_causes_domain", "domain_id"),
    )


class Phenomenon(BaseModel):
    """现象码（三层标准化的目标词表）"""

    __tablename__ = "phenomena"

    code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False, comment="现象码，如 PH-IA-001")
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, comment="标准现象名称，如：中控屏显示异常")
    business_line: Mapped[str] = mapped_column(String(10), nullable=False, default="ev", comment="业务线")
    # ★ 口语别名，是三层标准化第一层 LLM 归一规则的数据来源
    colloquial: Mapped[str | None] = mapped_column(Text, comment="客户口语说法，逗号分隔，如：黑屏,白屏,花屏,死机")
    category: Mapped[str | None] = mapped_column(String(50), comment="现象分类，如：显示类/动力类/充电类")

    __table_args__ = (Index("ix_phenomena_name", "name"),)


class CausePhenomenon(BaseModel):
    """根因-现象关联（多对多）"""

    __tablename__ = "cause_phenomena"

    cause_id: Mapped[int] = mapped_column(ForeignKey("root_causes.id", ondelete="CASCADE"), nullable=False, comment="根因 ID")
    phenomenon_id: Mapped[int] = mapped_column(ForeignKey("phenomena.id", ondelete="CASCADE"), nullable=False, comment="现象 ID")
    weight: Mapped[float] = mapped_column(Float, default=1.0, comment="关联强度 0~1，写入 Neo4j INDICATES.weight")
    # ★ is_core 是 confidence.py -0.20 权重的判定依据
    # 注意：运行时的「核心现象」判定走 Neo4j 实时统计（phenom_cause_count == 1），
    # 这里的字段是知识维护时的人工标注，用于 verify_data.py 交叉校验两者是否一致
    is_core: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否该根因的核心（独有）现象")

    __table_args__ = (
        UniqueConstraint("cause_id", "phenomenon_id", name="uq_cause_phenomenon"),
        Index("ix_cp_cause", "cause_id"),
        Index("ix_cp_phenomenon", "phenomenon_id"),
    )


class DtcCode(BaseModel):
    """OBD-II 故障码 —— 置信度加权 +0.35 的证据来源"""

    __tablename__ = "dtc_codes"

    code: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, comment="故障码，如 P0A0F / U0155")
    system: Mapped[str] = mapped_column(String(30), nullable=False, comment="系统：powertrain/chassis/body/network")
    description: Mapped[str | None] = mapped_column(Text, comment="英文原始描述")
    description_zh: Mapped[str | None] = mapped_column(Text, comment="中文描述（LLM 批量翻译）")
    business_line: Mapped[str] = mapped_column(String(10), nullable=False, default="ev", comment="业务线（码段规则映射，不过 LLM）")

    __table_args__ = (Index("ix_dtc_line", "business_line"),)


# ==========================================================
# ③ AI 影子产物
# 与平台镜像物理隔离：AI 只写这里，回写平台走 API 且只写「建议」
# ==========================================================


class AiTriageResult(BaseModel):
    """分诊结论"""

    __tablename__ = "ai_triage_results"

    source_issue_id: Mapped[int | None] = mapped_column(ForeignKey("alm_issues.id", ondelete="CASCADE"), comment="来源问题单")
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, comment="会话 ID（对应 Redis triage_state key）")
    raw_input: Mapped[str | None] = mapped_column(Text, comment="用户原始描述")
    confirmed_phenomena: Mapped[str | None] = mapped_column(Text, comment="已确认现象 JSON 数组")
    denied_phenomena: Mapped[str | None] = mapped_column(Text, comment="已否认现象 JSON 数组")
    candidate_causes: Mapped[str | None] = mapped_column(Text, comment="候选根因 Top-N JSON（含 confidence / 证据链）")
    primary_cause_code: Mapped[str | None] = mapped_column(String(30), comment="首要根因编码")
    primary_confidence: Mapped[float] = mapped_column(Float, default=0.0, comment="首要根因置信度")
    suggest_domain_id: Mapped[int | None] = mapped_column(ForeignKey("owner_domains.id", ondelete="SET NULL"), comment="建议责任域")
    total_rounds: Mapped[int] = mapped_column(Integer, default=0, comment="追问轮次")
    # ★ 复盘用：轮次耗尽强制收敛的比例，是评估图谱质量的核心指标
    force_conclude: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否轮次耗尽强制收敛")
    adopted: Mapped[bool | None] = mapped_column(Boolean, comment="人工是否采纳（None=未反馈），准确率统计口径")
    feedback_comment: Mapped[str | None] = mapped_column(Text, comment="人工反馈备注")

    __table_args__ = (
        Index("ix_triage_session", "session_id"),
        Index("ix_triage_issue", "source_issue_id"),
    )


class AiDedupLink(BaseModel):
    """去重链接 —— 记录问题单之间的重复判定"""

    __tablename__ = "ai_dedup_links"

    source_issue_id: Mapped[int] = mapped_column(ForeignKey("alm_issues.id", ondelete="CASCADE"), nullable=False, comment="源问题单 ID")
    matched_issue_id: Mapped[int] = mapped_column(ForeignKey("alm_issues.id", ondelete="CASCADE"), nullable=False, comment="被匹配的重复问题单 ID")
    similarity: Mapped[float] = mapped_column(Float, default=0.0, comment="向量余弦相似度")
    evidence: Mapped[str] = mapped_column(String(50), nullable=False, comment="匹配证据：model_and_sw / dtc / model_and_sw+dtc")
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=True, comment="是否判定为重复")
    reviewed: Mapped[bool | None] = mapped_column(Boolean, comment="人工复核结果（None=未复核）")

    __table_args__ = (
        UniqueConstraint("source_issue_id", "matched_issue_id", name="uq_dedup_link"),
        Index("ix_dedup_source", "source_issue_id"),
        Index("ix_dedup_matched", "matched_issue_id"),
    )


class CauseDtc(BaseModel):
    """根因-DTC 关联表（关系型备份，与 Neo4j POINTS_TO 同步）"""

    __tablename__ = "cause_dtc"

    cause_id: Mapped[int] = mapped_column(ForeignKey("root_causes.id", ondelete="CASCADE"), nullable=False, comment="根因 ID")
    dtc_id: Mapped[int] = mapped_column(ForeignKey("dtc_codes.id", ondelete="CASCADE"), nullable=False, comment="DTC 故障码 ID")
    relation_type: Mapped[str] = mapped_column(String(20), default="direct", comment="关联类型：direct=直接指向 / indirect=间接关联")

    __table_args__ = (
        UniqueConstraint("cause_id", "dtc_id", name="uq_cause_dtc"),
        Index("ix_cd_cause", "cause_id"),
        Index("ix_cd_dtc", "dtc_id"),
    )


class AiImpactAnalysis(BaseModel):
    """变更影响分析影子表"""

    __tablename__ = "ai_impact_analysis"

    change_request_id: Mapped[int | None] = mapped_column(ForeignKey("alm_change_requests.id", ondelete="CASCADE"), comment="来源变更单 ID")
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, comment="会话 ID")
    raw_input: Mapped[str | None] = mapped_column(Text, comment="用户原始输入")
    scope_result: Mapped[str | None] = mapped_column(Text, comment="影响范围 JSON")
    dependency_result: Mapped[str | None] = mapped_column(Text, comment="依赖冲突 JSON")
    baseline_conflicts: Mapped[str | None] = mapped_column(Text, comment="基线冲突 JSON")
    duplicate_result: Mapped[str | None] = mapped_column(Text, comment="重复变更 JSON")
    risk_level: Mapped[str] = mapped_column(String(20), default="medium", comment="风险等级：low/medium/high/critical")
    report_md: Mapped[str | None] = mapped_column(Text, comment="Markdown 分析报告")

    __table_args__ = (
        Index("ix_impact_cr", "change_request_id"),
        Index("ix_impact_session", "session_id"),
    )


class AiReportInterpretation(BaseModel):
    """报告/日志解读影子表"""

    __tablename__ = "ai_report_interpretations"

    source_issue_id: Mapped[int | None] = mapped_column(ForeignKey("alm_issues.id", ondelete="CASCADE"), comment="来源问题单 ID")
    report_type: Mapped[str] = mapped_column(String(30), nullable=False, comment="报告类型：DTC扫描/台架测试/OTA回归")
    minio_key: Mapped[str | None] = mapped_column(String(200), comment="MinIO 附件路径")
    raw_text: Mapped[str | None] = mapped_column(Text, comment="原始报告文本")
    parsed_metrics: Mapped[str | None] = mapped_column(Text, comment="解析指标 JSON [{metric_key, value, unit, normal, severity}]")
    abnormal_count: Mapped[int] = mapped_column(Integer, default=0, comment="异常指标数")
    interpretation: Mapped[str | None] = mapped_column(Text, comment="LLM 生成的结构化解读")
    reviewed: Mapped[bool | None] = mapped_column(Boolean, comment="人工复核结果")

    __table_args__ = (
        Index("ix_report_issue", "source_issue_id"),
        Index("ix_report_type", "report_type"),
    )


# ==========================================================
# ④ 幂等事件日志
# Webhook / 定时同步的去重依据：同一个 (event_type, entity_type, entity_id, entity_version)
# 只能处理一次。第二次来直接跳过，日志里打 [DUPLICATE]。
# ==========================================================


class AlmEventLog(BaseModel):
    """幂等事件日志"""

    __tablename__ = "alm_event_log"

    event_type: Mapped[str] = mapped_column(String(50), nullable=False, comment="事件类型：issue.created / issue.updated / issue.closed")
    entity_type: Mapped[str] = mapped_column(String(30), nullable=False, comment="实体类型：alm_issues / alm_requirements")
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False, comment="平台侧实体 ID，如 ISS-2025-00123")
    entity_version: Mapped[int] = mapped_column(Integer, nullable=False, comment="平台侧版本号，防重复投递")
    payload_json: Mapped[str | None] = mapped_column(Text, comment="原始事件 JSON（审计/回放用）")

    __table_args__ = (
        UniqueConstraint("event_type", "entity_type", "entity_id", "entity_version", name="uq_event_idempotent"),
        Index("ix_event_entity", "entity_type", "entity_id"),
    )
