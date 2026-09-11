"""数据驱动的收敛阈值（C4）。

旧版 check_convergence 的 0.65 / 0.30 / round≥3+0.60 是拍脑袋常数，
调它没有依据。这里把"该不该收敛"转成可从历史数据学的问题：

    给定本轮证据（top1 置信度、top1-top2 分差、已进行轮数），
    P(这版结论会被工程师采纳) 是否足够高？

用 ai_triage_results 里的人工采纳/拒绝反馈拟合一个逻辑回归
（纯 Python 梯度下降，样本量小，不值得引 numpy/sklearn 依赖）：

    p(adopt) = sigmoid(w0 + w1*top1 + w2*margin + w3*round_norm)

拟合产物写到策略 JSON（默认 eval/convergence_policy.json），
check_convergence 每次调用前读文件（mtime 缓存）：
    - 有策略文件 → 数据驱动判定；
    - 没有文件 / 样本不足 / 解析失败 → 回退 confidence.py 的常数规则
      （拟合是优化项，缺失时系统行为与历史版本完全一致）。

拟合入口：scripts/fit_convergence.py（离线跑，产出策略文件）。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from src.core.config import get_settings
from src.core.logger import logger

# p(adopt) 达到该值才自动收敛：宁可多问一轮，不给低把握结论
DEFAULT_PROB_THRESHOLD = 0.8

# 样本量下限：低于它拟合出来的权重就是噪声，拒绝产出策略文件
MIN_FIT_SAMPLES = 30

# 策略文件缓存（mtime 变化才重新解析，避免每轮诊断都读盘）
_policy_cache: tuple[float, dict] | None = None


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def features(top1: float, margin: float, current_round: int, max_rounds: int) -> list[float]:
    """收敛判定的特征向量（含偏置 1）。

    round 归一化到 [0,1]（round/max_rounds 封顶 1）：
    轮数越深，"再问一轮"的边际价值越低，与 top1 置信度同量纲加权。"""
    round_norm = min(1.0, max(0.0, current_round / max(1, max_rounds)))
    return [1.0, top1, margin, round_norm]


def fit_logistic(
    samples: list[tuple[list[float], int]],
    *,
    epochs: int = 3000,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> list[float]:
    """批量梯度下降拟合逻辑回归，返回 [w0, w1, w2, w3]。

    L2 轻微正则防共线（top1 与 margin 高相关时权重不炸）。"""
    if not samples:
        raise ValueError("无训练样本")
    dim = len(samples[0][0])
    w = [0.0] * dim
    n = len(samples)
    for _ in range(epochs):
        grad = [0.0] * dim
        for x, y in samples:
            p = sigmoid(sum(wi * xi for wi, xi in zip(w, x)))
            err = p - y
            for j in range(dim):
                grad[j] += err * x[j]
        for j in range(dim):
            w[j] -= lr * (grad[j] / n + l2 * w[j])
    return w


def policy_path() -> Path:
    """策略文件位置：CONVERGENCE_POLICY_PATH 覆盖，默认仓库 eval/ 下。"""
    s = get_settings()
    if s.CONVERGENCE_POLICY_PATH:
        return Path(s.CONVERGENCE_POLICY_PATH)
    return Path(__file__).resolve().parents[3] / "eval" / "convergence_policy.json"


def load_policy(force: bool = False) -> dict | None:
    """读拟合产物。返回 None 表示无有效策略（调用方走常数规则回退）。"""
    global _policy_cache
    path = policy_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _policy_cache = None
        return None
    if not force and _policy_cache and _policy_cache[0] == mtime:
        return _policy_cache[1]
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
        weights = policy["weights"]
        assert len(weights) == 4 and all(isinstance(w, (int, float)) for w in weights)
        policy.setdefault("prob_threshold", DEFAULT_PROB_THRESHOLD)
        _policy_cache = (mtime, policy)
        return policy
    except Exception as e:
        logger.warning(f"[CONVERGENCE] 策略文件无效，回退常数规则: {e}")
        _policy_cache = None
        return None


def check_convergence_data_driven(
    top1: float,
    margin: float,
    current_round: int,
    max_rounds: int,
    policy: dict,
) -> bool:
    """数据驱动判定：P(采纳) ≥ prob_threshold 才收敛。"""
    x = features(top1, margin, current_round, max_rounds)
    p = sigmoid(sum(w * xi for w, xi in zip(policy["weights"], x)))
    return p >= policy["prob_threshold"]
