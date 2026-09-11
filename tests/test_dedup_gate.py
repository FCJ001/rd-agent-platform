# ============================================================
# Dedup 结构化门槛（双门槛之二）单元测试
#
# 向量召回路径（Milvus / 暴力降级）都依赖同一个 _structured_gate，
# 这里锁住行为契约：相似度过阈值但车型+SW/DTC 不匹配 → 不判重。
# ============================================================

from src.agents.dedup.matcher import DedupMatcher, _split_dtc


def _match(cand: dict, source_dtc: set[str], model: str = "EV-A01", sw: str = "2024.32.5"):
    return DedupMatcher._structured_gate(
        source_dtc=source_dtc, source_model=model, source_sw=sw,
        cand=cand, sim_score=0.93,
    )


def test_split_dtc_normalizes_separator():
    assert _split_dtc("U0155，P0A7F") == {"U0155", "P0A7F"}
    assert _split_dtc(None) == set()


def test_gate_model_sw_match():
    m = _match({"id": 2, "model_code": "EV-A01", "sw_version": "2024.32.5",
                "dtc_snapshot": "B1234", "issue_no": "ISS-2", "title": "t"}, set())
    assert m is not None and m.evidence == "model_and_sw"


def test_gate_dtc_overlap_only():
    m = _match({"id": 3, "model_code": "EV-B02", "sw_version": "9.9",
                "dtc_snapshot": "U0155", "issue_no": "ISS-3", "title": "t"}, {"U0155"})
    assert m is not None and m.evidence == "dtc"


def test_gate_both_evidence():
    m = _match({"id": 4, "model_code": "EV-A01", "sw_version": "2024.32.5",
                "dtc_snapshot": "U0155", "issue_no": "ISS-4", "title": "t"}, {"U0155"})
    assert m is not None and m.evidence == "model_and_sw+dtc"


def test_gate_no_structural_match_rejected():
    # 向量相似 0.93 过阈值，但车型/SW/DTC 全不同 → 不判重
    m = _match({"id": 5, "model_code": "EV-C03", "sw_version": "1.0",
                "dtc_snapshot": "", "issue_no": "ISS-5", "title": "t"}, set())
    assert m is None


def test_gate_missing_model_side_rejected():
    # 一方车型信息缺失 → same_model_and_sw 不成立，不允许误判
    m = _match({"id": 6, "model_code": "", "sw_version": "2024.32.5",
                "dtc_snapshot": "", "issue_no": "ISS-6", "title": "t"}, set())
    assert m is None
