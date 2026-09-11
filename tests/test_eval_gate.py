# ============================================================
# C3 CI 回归门禁：评分级评测案例
#
# 案例在 eval/cases/scoring_cases.json，纯函数驱动
# confidence.apply_context_weights + check_convergence，
# 不依赖 DB / Neo4j / LLM —— CI 里直接跑 pytest 即可。
#
# 评测失败 = 证据分级 / Noisy-OR / 否认与覆盖率阻尼的行为契约被改动破坏。
# ============================================================

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(_EVAL_DIR))

from run_eval import run_scoring_cases  # noqa: E402


def test_scoring_gate():
    passed, failed, failures = run_scoring_cases()
    assert failed == 0, (
        f"评分评测失败 {failed}/{passed + failed} 条：\n" + "\n".join(failures)
    )
