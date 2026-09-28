# ============================================================
# 反馈闭环「改答案」测试：误诊时把正确根因写回图谱
#
# 背景洞（production-plan §4.3）：否决只降错误根因权重时，系统只会
# 「更不确定」，不会「更准」——修正路径必须同时强化正确根因。
# 实现已接线（weaken_graph_on_rejected + correct_cause_code），
# 本文件锁语义防回归。
# ★ 不碰 Neo4j：_weaken_sync / reinforce_graph_on_adopted 被替换为记录器。
# ============================================================

import pytest

import src.agents.triage.feedback_loop as fl


@pytest.fixture()
def recorder(monkeypatch):
    rec = {"weaken": [], "reinforce": []}

    def fake_weaken_sync(phenomena, cause_code):
        rec["weaken"].append((tuple(phenomena), cause_code))
        return []

    async def fake_reinforce(confirmed_phenomena, primary_cause_code, session_id=""):
        rec["reinforce"].append((tuple(confirmed_phenomena), primary_cause_code))
        return True

    monkeypatch.setattr(fl, "_weaken_sync", fake_weaken_sync)
    monkeypatch.setattr(fl, "reinforce_graph_on_adopted", fake_reinforce)
    return rec


PHENOMENA = ["中控屏黑屏", "伴随死机"]
WRONG = "RC-IA-0001"
RIGHT = "RC-IA-0099"


async def test_rejection_with_correction_reinforces_right_cause(recorder):
    """带 correct_cause_code：弱化错误根因 + 用采纳逻辑强化正确根因。"""
    ok = await fl.weaken_graph_on_rejected(
        confirmed_phenomena=PHENOMENA, primary_cause_code=WRONG,
        correct_cause_code=RIGHT,
    )
    assert ok is True
    assert recorder["weaken"] == [(tuple(PHENOMENA), WRONG)]      # 错误根因被弱化
    assert recorder["reinforce"] == [(tuple(PHENOMENA), RIGHT)]    # ★ 正确答案回写


async def test_rejection_without_correction_only_weakens(recorder):
    """无纠正：只弱化（旧行为），不触发任何强化。"""
    ok = await fl.weaken_graph_on_rejected(
        confirmed_phenomena=PHENOMENA, primary_cause_code=WRONG,
    )
    assert ok is True
    assert recorder["reinforce"] == []


async def test_correction_same_as_wrong_cause_is_ignored(recorder):
    """纠正值与错误根因相同（脏数据）：不强化，防自我提权。"""
    ok = await fl.weaken_graph_on_rejected(
        confirmed_phenomena=PHENOMENA, primary_cause_code=WRONG,
        correct_cause_code=WRONG,
    )
    assert ok is True
    assert recorder["reinforce"] == []


async def test_empty_inputs_rejected(recorder):
    """缺现象或缺根因编码：直接 False，不碰图谱。"""
    assert await fl.weaken_graph_on_rejected([], WRONG) is False
    assert await fl.weaken_graph_on_rejected(PHENOMENA, "") is False
    assert recorder["weaken"] == [] and recorder["reinforce"] == []


def test_feedback_api_accepts_correct_cause_code():
    """API 契约：FeedbackRequest 带.correct_cause_code 字段（前端/测试平台依赖）。"""
    from src.api.routers.triage import FeedbackRequest

    req = FeedbackRequest(session_id="s1", adopted=False,
                          correct_cause_code="RC-EV-0012")
    assert req.correct_cause_code == "RC-EV-0012"
