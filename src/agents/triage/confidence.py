"""置信度计算与收敛判断。参考天宫医疗版 confidence.py。"""

from src.agents.triage.state import CandidateCause


MAX_ROUNDS = 5


def apply_context_weights(
    candidates: list[CandidateCause],
    dtc_codes: list[str],
    denied_phenomena: list[str],
) -> list[CandidateCause]:
    """
    在基础置信度上叠加上下文权重，返回重新排序后的候选根因列表。

    权重规则：
      +0.15  命中 is_core 现象（该现象是该根因的独有标志性现象）
      +0.15  用户提供的 DTC 码与该根因的 dtc 列表匹配
      -0.20  用户明确否认了该根因的 is_core 现象
    """
    # 计算每个现象出现在几个候选根因中（用于判断"核心现象"）
    phenom_cause_count: dict[str, int] = {}
    for c in candidates:
        for p in c.all_phenomena:
            phenom_cause_count[p] = phenom_cause_count.get(p, 0) + 1

    for c in candidates:
        score = c.base_confidence

        # +0.15 is_core 现象命中
        for p in c.matched_phenomena:
            for all_p in c.all_phenomena:
                if p == all_p and phenom_cause_count.get(p, 0) == 1:
                    score += 0.15
                    c.is_core_match = True
                    break
            if c.is_core_match:
                break

        # +0.15 DTC 码匹配
        for dtc in dtc_codes:
            if dtc in c.dtc_matched:
                score += 0.15
                break

        # -0.20 核心现象被否认
        for p in c.all_phenomena:
            if p in denied_phenomena and phenom_cause_count.get(p, 0) == 1:
                score -= 0.20
                break

        c.confidence = round(max(0.0, min(1.0, score)), 4)

    candidates.sort(key=lambda x: x.confidence, reverse=True)
    return candidates


def check_convergence(
    candidates: list[CandidateCause],
    current_round: int,
    max_rounds: int = MAX_ROUNDS,
) -> tuple[bool, bool]:
    """
    判断诊断是否可以收敛（输出结论）。

    Returns:
        (should_conclude, force_conclude)
    """
    if current_round >= max_rounds:
        return True, True

    if not candidates:
        return False, False

    top1 = candidates[0].confidence

    # 条件1：Top1 置信度 ≥ 70%
    if top1 >= 0.70:
        return True, False

    # 条件2：Top1 与 Top2 差值 ≥ 30%
    if len(candidates) >= 2:
        top2 = candidates[1].confidence
        if top1 - top2 >= 0.30:
            return True, False

    return False, False
