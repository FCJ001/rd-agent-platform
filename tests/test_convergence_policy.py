# ============================================================
# 收敛策略单元测试
# 覆盖：逻辑回归拟合 / 特征构造 / 数据驱动判定 /
#       check_convergence 无策略文件时回退常数规则
# ============================================================

import pytest

from src.agents.triage import thresholds as th
from src.agents.triage.confidence import check_convergence
from src.agents.triage.state import CandidateCause


def _candidate(conf: float, code: str = "RC-X") -> CandidateCause:
    return CandidateCause(code=code, name="测试根因", confidence=conf, base_confidence=conf)


def test_sigmoid_boundary():
    assert th.sigmoid(0.0) == 0.5
    assert th.sigmoid(100.0) > 0.999999
    assert th.sigmoid(-100.0) < 0.000001


def test_features_round_normalized():
    x = th.features(top1=0.7, margin=0.2, current_round=3, max_rounds=5)
    assert x[0] == 1.0 and x[1] == 0.7 and x[2] == 0.2
    assert x[3] == 0.6
    # 超过 max_rounds 封顶为 1
    assert th.features(0.7, 0.2, 99, 5)[3] == 1.0


def test_fit_logistic_separable():
    samples = []
    for t, m, r, y in [(0.9, 0.5, 0, 1), (0.2, 0.05, 0, 0), (0.8, 0.4, 1, 1), (0.3, 0.02, 2, 0)]:
        samples.extend([(th.features(t, m, r, 5), y)] * 20)
    w = th.fit_logistic(samples)
    pol = {"weights": w, "prob_threshold": 0.8}
    assert th.check_convergence_data_driven(0.9, 0.5, 0, 5, pol) is True
    assert th.check_convergence_data_driven(0.2, 0.05, 0, 5, pol) is False


def test_check_convergence_falls_back_without_policy(monkeypatch):
    # 无策略文件 → 常数规则（top1 ≥ 0.65 即收敛）
    monkeypatch.setattr(th, "load_policy", lambda force=False: None)
    should, force = check_convergence([_candidate(0.7), _candidate(0.2)], 0)
    assert should and not force

    should, _ = check_convergence([_candidate(0.5), _candidate(0.45)], 0)
    assert not should


def test_check_convergence_uses_policy(monkeypatch):
    policy = {"weights": [0.0, 10.0, 0.0, 0.0], "prob_threshold": 0.9}
    monkeypatch.setattr(th, "load_policy", lambda force=False: policy)
    # sigmoid(7.0) ≈ 0.999 → 收敛；sigmoid(5.0) ≈ 0.993 也 ≥ 0.9 → 收敛
    should, force = check_convergence([_candidate(0.7)], 0)
    assert should and not force
    # max_rounds 强制收敛不受策略影响
    should, force = check_convergence([_candidate(0.1)], 99)
    assert should and force


def test_load_policy_invalid_file(tmp_path, monkeypatch):
    bad = tmp_path / "policy.json"
    bad.write_text("{'weights': 'not-a-list'}", encoding="utf-8")
    monkeypatch.setattr(th, "policy_path", lambda: bad)
    monkeypatch.setattr(th, "_policy_cache", None)
    assert th.load_policy() is None
