"""置信度计算与收敛判断（C1 重写版）。

旧版是硬编码常数加减（+0.15/+0.15/-0.20），两个问题：
  1. 常数无来源，"好像是"和"诊断仪测过"给了同样的权重；
  2. 加减法不满足概率语义，多证据叠加会溢出/为负，靠 clamp 兜底。

新版：证据分级 + Noisy-OR 合成 + 先验。
  - 每条证据独立给出"该证据暗示此根因"的概率 p_i = 证据强度 × 图谱权重调制；
  - Noisy-OR 合成：P = 1 - Π(1 - p_i)，多证据单调趋近 1 但永不越过；
  - 先验 + 泄漏项：无证据时起评 PRIOR，防 0 置信度把排序信息丢光；
  - 覆盖率阻尼：根因的典型现象只命中一部分时，说明证据不全，压低得分；
  - 否认证据：核心现象被明确否认时按软否定乘法降权。

每个候选根因的完整推导写入 evidence_trace，结论页可展开"这个置信度怎么来的"。
"""

from src.agents.triage.state import CandidateCause


MAX_ROUNDS = 5

# ── 证据分级 ──────────────────────────────────────────────────────────────
# 来源可靠性递减：仪器实测 > 用户主动描述 > 追问后确认 > 模糊表述
EVIDENCE_GRADE_INSTRUMENT = "instrument"   # 诊断仪/DTC 快照（车端上报）
EVIDENCE_GRADE_REPORTED = "reported"       # 用户主动描述（自发说出）
EVIDENCE_GRADE_CONFIRMED = "confirmed"     # 追问后确认（用户对"是否黑屏"答"是"）
EVIDENCE_GRADE_HEDGED = "hedged"           # 模糊表述（"好像""可能"）

# 各级证据的暗示强度 p_i 上限。取值依据：
#   instrument 0.90 —— DTC 是车端 ECU 实报，与根因强关联，但传感器误报存在，不满配；
#   reported   0.70 —— 用户亲眼所见，但口语描述可能偏离标准现象；
#   confirmed  0.55 —— 追问确认有引导性（用户倾向顺着问），比自发描述弱；
#   hedged     0.30 —— "好像是黑屏"，连现象本身都不确定。
EVIDENCE_FACTORS = {
    EVIDENCE_GRADE_INSTRUMENT: 0.90,
    EVIDENCE_GRADE_REPORTED: 0.70,
    EVIDENCE_GRADE_CONFIRMED: 0.55,
    EVIDENCE_GRADE_HEDGED: 0.30,
}

# DTC 证据单独定级：用户口述的 DTC（可能记错）比快照里的低
DTC_FACTORS = {
    EVIDENCE_GRADE_INSTRUMENT: 0.90,
    EVIDENCE_GRADE_REPORTED: 0.60,
}

# 无任何证据时的基础患病率先验：任一根因"本来就坏"的起点概率
PRIOR = 0.02
# Noisy-OR 泄漏项：知识库不完备（未收录的关联）的兜底概率
LEAK = 0.01

# 覆盖率阻尼：coverage = 命中现象数/该根因全部典型现象数。
# 只命中 1/3 典型现象 → 证据片面，乘 0.6+0.4×(1/3)；全命中不阻尼。
COVERAGE_FLOOR = 0.6

# 否认证据（软否定）：用户明确否认某现象时，
# 该现象是根因核心标志 → 权重乘 0.4；只是相关现象 → 乘 0.7。
DENY_CORE_FACTOR = 0.4
DENY_NON_CORE_FACTOR = 0.7

# 单个根因最多计入的 DTC 证据条数：防 DTC 列表长时把置信度堆满
MAX_DTC_EVIDENCE = 3

# 旧状态（Redis 里反序列化）没有 evidence 字段时的兜底口径
DEFAULT_GRADE = EVIDENCE_GRADE_REPORTED

# 证据等级强弱序，merge_evidence 用它保证只升不降
GRADE_RANK = {
    EVIDENCE_GRADE_HEDGED: 0,
    EVIDENCE_GRADE_CONFIRMED: 1,
    EVIDENCE_GRADE_REPORTED: 2,
    EVIDENCE_GRADE_INSTRUMENT: 3,
}


def merge_evidence(existing: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """合并证据标记，同一证据的等级只升不降（hedged 后被明确确认会升级）。"""
    merged = dict(existing)
    for k, v in updates.items():
        if GRADE_RANK.get(v, 0) >= GRADE_RANK.get(merged.get(k, ""), 0):
            merged[k] = v
    return merged


def _factor(grade: str | None) -> float:
    return EVIDENCE_FACTORS.get(grade or DEFAULT_GRADE, EVIDENCE_FACTORS[DEFAULT_GRADE])


def apply_context_weights(
    candidates: list[CandidateCause],
    dtc_codes: list[str],
    denied_phenomena: list[str],
    phenomena_evidence: dict[str, str] | None = None,
    dtc_evidence: dict[str, str] | None = None,
) -> list[CandidateCause]:
    """
    证据合成评分：按 Noisy-OR 重算每个候选根因的置信度，返回降序列表。

    Args:
        candidates: 图谱查询返回的候选根因（base_confidence = 命中/全部现象数）
        dtc_codes: 会话中收集到的 DTC 码
        denied_phenomena: 用户明确否认的现象
        phenomena_evidence: 现象名 → 证据等级（C1）
        dtc_evidence: DTC 码 → 证据等级（C1）
    """
    phenomena_evidence = phenomena_evidence or {}
    dtc_evidence = dtc_evidence or {}

    # 计算每个现象出现在几个候选根因中 → 唯一命中 = 该根因的标志性现象
    phenom_cause_count: dict[str, int] = {}
    for c in candidates:
        for p in c.all_phenomena:
            phenom_cause_count[p] = phenom_cause_count.get(p, 0) + 1

    for c in candidates:
        trace: list[str] = []
        odds = 1.0  # 连乘 (1 - p_i)，最后 1 - odds

        # ── 现象证据（Noisy-OR 主项）──
        for p in c.matched_phenomena:
            grade = phenomena_evidence.get(p, DEFAULT_GRADE)
            p_i = _factor(grade)
            # 图谱权重调制：反馈回写实时的 weight ∈ (0,1]，
            # 被否决衰减过的关联，证据强度打折；缺权重按 1.0（不调制）
            w = c.phenomena_weight.get(p, 1.0)
            weight_mod = 0.5 + 0.5 * max(0.0, min(1.0, w))
            p_i *= weight_mod
            odds *= (1.0 - p_i)
            trace.append(
                f"现象「{p}」[{grade}] 强度{p_i:.2f}"
                f"{'（图谱权重' + f'{w:.2f}调制）' if w < 1.0 else ''}"
            )

        p_phen = 1.0 - odds

        # ── DTC 证据 ──
        matched_dtc = [d for d in dtc_codes if d in c.dtc_matched][:MAX_DTC_EVIDENCE]
        if matched_dtc:
            dtc_odds = 1.0
            for d in matched_dtc:
                grade = dtc_evidence.get(d, DEFAULT_GRADE)
                factor = DTC_FACTORS.get(grade, DTC_FACTORS[DEFAULT_GRADE])
                dtc_odds *= (1.0 - factor)
                trace.append(f"DTC「{d}」[{grade}] 强度{factor:.2f}")
            p_dtc = 1.0 - dtc_odds
        else:
            p_dtc = 0.0

        # ── 合成：先验 + 现象 + DTC + 泄漏 ──
        conf = 1.0 - (1.0 - PRIOR) * (1.0 - p_phen) * (1.0 - p_dtc) * (1.0 - LEAK)

        # ── 覆盖率阻尼：典型现象只命中一部分 → 证据片面 ──
        coverage = c.base_confidence  # 图谱查询层已算好 命中数/全部数
        damp = COVERAGE_FLOOR + (1.0 - COVERAGE_FLOOR) * coverage
        conf *= damp
        if damp < 0.999:
            trace.append(f"覆盖率阻尼 ×{damp:.2f}（命中 {len(c.matched_phenomena)}/{len(c.all_phenomena)} 典型现象）")

        # ── 否认证据（软否定）──
        for p in c.all_phenomena:
            if p in denied_phenomena:
                if phenom_cause_count.get(p, 0) == 1:
                    conf *= DENY_CORE_FACTOR
                    trace.append(f"核心现象「{p}」被否认 ×{DENY_CORE_FACTOR}")
                    c.is_core_match = False
                else:
                    conf *= DENY_NON_CORE_FACTOR
                    trace.append(f"相关现象「{p}」被否认 ×{DENY_NON_CORE_FACTOR}")

        c.confidence = round(max(0.0, min(1.0, conf)), 4)
        c.evidence_trace = trace

    candidates.sort(key=lambda x: x.confidence, reverse=True)
    return candidates


def check_convergence(
    candidates: list[CandidateCause],
    current_round: int,
    max_rounds: int = MAX_ROUNDS,
) -> tuple[bool, bool]:
    """
    判断诊断是否可以收敛（输出结论）。

    优先走数据驱动策略（thresholds.load_policy，用历史采纳/拒绝反馈拟合的
    逻辑回归 P(采纳)≥门限）；没有有效策略文件时回退下面的常数规则。

    Returns:
        (should_conclude, force_conclude)
    """
    if current_round >= max_rounds:
        return True, True

    if not candidates:
        return False, False

    from src.agents.triage import thresholds

    top1 = candidates[0].confidence
    top2 = candidates[1].confidence if len(candidates) >= 2 else 0.0
    margin = top1 - top2

    policy = thresholds.load_policy()
    if policy:
        should = thresholds.check_convergence_data_driven(
            top1, margin, current_round, max_rounds, policy,
        )
        return should, False

    # ── 常数规则（无拟合数据时的回退口径）──
    # 条件1：Top1 置信度 ≥ 65%
    if top1 >= 0.65:
        return True, False

    # 条件2：Top1 与 Top2 差值 ≥ 30%
    if len(candidates) >= 2:
        if margin >= 0.30:
            return True, False

    # 条件3：Round ≥ 3 且 top1 ≥ 60% → 尽早给出结论
    if current_round >= 3 and top1 >= 0.60:
        return True, True

    return False, False
