# ============================================================
# C3 评测运行器
#
# 两种模式：
#   python eval/run_eval.py                 # 评分级门禁（默认）—— 纯函数，
#                                           # 不依赖 DB/Neo4j/LLM，CI 可跑
#   python eval/run_eval.py --live          # 端到端实况 —— 需要真实服务，
#                                           # 发版前/每日跑，输出分层准确率
#   pytest tests/test_eval_gate.py          # CI 门禁的 pytest 入口（同一套案例）
#
# 任一模式失败都返回非零退出码，作为回归门禁。
# ============================================================

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent / "cases"
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ════════════════════════════════════════════════════════════════════════
# 模式一：评分级门禁（纯函数）
# ════════════════════════════════════════════════════════════════════════

def _check_expectations(case: dict, candidates: list) -> list[str]:
    """对评分结果核对断言，返回错误列表（空 = 通过）。"""
    errors = []
    expect = case.get("expect", {})
    by_code = {c.code: c for c in candidates}
    cid = case["id"]

    if "top_code" in expect and candidates and candidates[0].code != expect["top_code"]:
        actual = candidates[0].code if candidates else "（空）"
        errors.append(f"{cid}: 期望 top={expect['top_code']} 实际 top={actual}")

    for code, mn in expect.get("confidence_min", {}).items():
        c = by_code.get(code)
        if c is None:
            errors.append(f"{cid}: 断言引用了不存在的候选 {code}")
        elif c.confidence < mn:
            errors.append(f"{cid}: {code} 置信度 {c.confidence} < 下限 {mn}")

    for code, mx in expect.get("confidence_max", {}).items():
        c = by_code.get(code)
        if c is None:
            errors.append(f"{cid}: 断言引用了不存在的候选 {code}")
        elif c.confidence > mx:
            errors.append(f"{cid}: {code} 置信度 {c.confidence} > 上限 {mx}")

    if "order" in expect:
        actual_order = [c.code for c in candidates]
        if actual_order != expect["order"]:
            errors.append(f"{cid}: 期望排序 {expect['order']} 实际 {actual_order}")

    conv = expect.get("convergence")
    if conv:
        from src.agents.triage.confidence import check_convergence
        should, _force = check_convergence(candidates, conv["round"])
        if should != conv["should_conclude"]:
            errors.append(
                f"{cid}: round={conv['round']} 期望 should_conclude="
                f"{conv['should_conclude']} 实际 {should}"
            )

    return errors


def run_scoring_cases(path: Path | None = None) -> tuple[int, int, list[str]]:
    """
    跑评分级案例，返回 (通过数, 失败数, 失败详情)。

    每个案例用合成候选根因 + 证据输入驱动 confidence.apply_context_weights，
    锁住证据分级 / Noisy-OR / 否认与覆盖率阻尼的行为契约。
    """
    from src.agents.triage.confidence import apply_context_weights
    from src.agents.triage.state import CandidateCause

    data = _load(path or CASES_DIR / "scoring_cases.json")
    passed = failed = 0
    failures: list[str] = []

    for case in data["cases"]:
        candidates = [
            CandidateCause(**c) for c in case["candidates"]
        ]
        candidates = apply_context_weights(
            candidates,
            case.get("dtc_codes", []),
            case.get("denied_phenomena", []),
            phenomena_evidence=case.get("phenomena_evidence", {}),
            dtc_evidence=case.get("dtc_evidence", {}),
        )
        errors = _check_expectations(case, candidates)
        if errors:
            failed += 1
            failures.extend(errors)
        else:
            passed += 1
            print(f"  ✓ {case['id']} [{case['layer']}] {case['title']}")

    return passed, failed, failures


# ════════════════════════════════════════════════════════════════════════
# 模式二：端到端实况（需要 DB / Neo4j / LLM）
# ════════════════════════════════════════════════════════════════════════

async def _load_alias_map() -> dict[str, str]:
    """现象名 → 口语别名字符串，供评测关键词匹配（黑屏 ↔ 中控屏显示异常）。"""
    from src.agents.triage.db_queries import get_all_phenomena
    from src.infra.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        rows = await get_all_phenomena(db)
    return {r["name"]: (r.get("colloquial") or "") for r in rows}


def _phenomena_hit(confirmed: list[str], keywords: list[str], alias_map: dict) -> bool:
    """每个关键词在已确认现象集合（含别名）中任意命中即算。
    旧实现要求所有关键词命中同一条现象，多现象断言永远不可能通过。"""
    if not confirmed:
        return False
    hay = [p + " " + alias_map.get(p, "") for p in confirmed]
    return all(any(k in h for h in hay) for k in keywords)


async def run_live_cases(threshold: float = 0.8, path: Path | None = None) -> bool:
    """跑端到端实况场景，输出分层指标，返回是否达标。

    指标（分 layer 与总体）：
      - 通过率：案例级断言全部满足
      - Top-1 命中率：首选根因 ∈ expect_cause（只统计配置了 expect_cause 的案例）
        —— 没有命中率就没有优化方向，置信度/收敛阈值的每次调整都要看它
      - 平均收敛轮数：到达 CONCLUDE 消耗的对话轮数（全案例统计）
    """
    from src.agents.triage.graph import _get_llm_chat, _get_llm_json, run_triage, TriageDeps
    from src.agents.triage.state import TriagePhase
    from src.infra.db import AsyncSessionLocal

    data = _load(path or CASES_DIR / "live_cases.json")
    alias_map = await _load_alias_map()

    from src.infra.db import session_scope  # 提交-回滚-关闭一体，与 webhook.py 同契约

    deps = TriageDeps(
        llm_json=_get_llm_json(),
        llm_chat=_get_llm_chat(),
        db_session_factory=session_scope,
    )

    layer_stats: dict[str, list[bool]] = {}
    layer_top1: dict[str, list[bool]] = {}   # 只收 expect_cause 非空的案例
    layer_top3: dict[str, list[bool]] = {}   # Top-3：前三候选任一命中（排序质量的兜底指标）
    layer_rounds: dict[str, list[int]] = {}
    all_ok = True

    for case in data["cases"]:
        cid = case["id"]
        thread_id = f"eval:{cid}:{uuid.uuid4().hex[:6]}"
        state = None
        reply = ""
        rounds_used = 0
        concluded = False

        for i, turn in enumerate(case["turns"]):
            reply, state = await run_triage(
                user_message=turn,
                thread_id=thread_id,
                deps=deps,
                existing_state=state,
                viewer_role="engineer",
                # 场景可标注所属业务线；不标则不过滤（与加 scope 前的行为一致）。
                # 要复现线上「按线检索」的效果，给 case 补 business_line 字段即可
                business_line=case.get("business_line", ""),
            )
            rounds_used = i + 1
            # ★ 收敛路径 conclude→save_record 后 phase 被置为 END，
            #   只判 CONCLUDE 会永远漏判收敛（断言空转时代没暴露）
            if state.phase in (TriagePhase.CONCLUDE, TriagePhase.END):
                concluded = True
                break

        errors = []
        # ★ 兼容两种格式：嵌套 expect{...} 或顶层平铺键（旧 cases 用顶层，
        #   之前这里只读嵌套导致现象/根因断言全部空转，7/7 是"跑通即通过"）
        expect = case.get("expect") or case
        if expect.get("expect_completed", False) and not (reply or "").strip():
            errors.append("未产生任何回复")
        if expect.get("expect_safety_stop"):
            if "安全警示" not in (reply or ""):
                errors.append(f"期望安全终止，实际回复: {reply[:80]}")
        if expect.get("expect_denied_nonempty") and not state.denied_phenomena:
            errors.append("期望有否认现象，实际为空")
        if "expect_converged" in expect and concluded != expect["expect_converged"]:
            errors.append(f"期望{'收敛' if expect['expect_converged'] else '不收敛'}，实际{'收敛' if concluded else '未收敛'}")
        kws = expect.get("expect_phenomena")
        if kws and not _phenomena_hit(state.confirmed_phenomena, kws, alias_map):
            errors.append(
                f"现象关键词 {kws} 未命中，confirmed={state.confirmed_phenomena}"
            )
        expect_causes = expect.get("expect_cause") or expect.get("expect_causes")
        top1_hit = None
        top3_hit = None
        if expect_causes:
            codes = [c.code for c in state.candidate_causes]
            top = codes[0] if codes else None
            top1_hit = top in expect_causes
            # Top-3：前三候选任一命中（回答"排序质量"——top1 之外的候选有没有兜住）
            top3_hit = any(code in expect_causes for code in codes[:3])
            if not top1_hit:
                errors.append(f"期望根因 ∈ {expect_causes}，实际 top={top}")

        # 收敛轮数：没收敛的案例按"轮数耗尽"计（拉高均值，暴露收敛过慢的案例）
        if not concluded:
            rounds_used = len(case["turns"])

        ok = not errors
        all_ok = all_ok and ok
        layer = case["layer"]
        layer_stats.setdefault(layer, []).append(ok)
        layer_rounds.setdefault(layer, []).append(rounds_used)
        layer_rounds.setdefault("_overall", []).append(rounds_used)
        if top1_hit is not None:
            layer_top1.setdefault(layer, []).append(top1_hit)
            layer_top1.setdefault("_overall", []).append(top1_hit)
            layer_top3.setdefault(layer, []).append(top3_hit)
            layer_top3.setdefault("_overall", []).append(top3_hit)
        mark = "✓" if ok else "✗"
        hit_mark = "" if top1_hit is None else ("  top1✓" if top1_hit else "  top1✗")
        if top1_hit is not None and not top1_hit and top3_hit:
            hit_mark += "（top3✓）"
        print(f"  {mark} {cid} [{layer}] {case['title']}{hit_mark}  rounds={rounds_used}")
        for e in errors:
            print(f"      - {e}")

    def _rate(results: list[bool]) -> str:
        return f"{sum(results)}/{len(results)} = {sum(results) / len(results):.0%}" if results else "n/a"

    def _avg_rounds(results: list[int]) -> str:
        return f"{sum(results) / len(results):.2f} 轮" if results else "n/a"

    print("\n分层指标：")
    for layer in sorted(k for k in layer_stats if k != "_overall"):
        print(f"  {layer}: 通过 {_rate(layer_stats[layer])}"
              f"  Top-1 {_rate(layer_top1[layer]) if layer in layer_top1 else 'n/a'}"
              f"  Top-3 {_rate(layer_top3[layer]) if layer in layer_top3 else 'n/a'}"
              f"  平均收敛 {_avg_rounds(layer_rounds[layer])}")
    print("总体：")
    print(f"  通过率: {_rate(layer_stats.get('_overall', []))}")
    print(f"  Top-1 命中率: {_rate(layer_top1.get('_overall', []))}")
    print(f"  Top-3 命中率: {_rate(layer_top3.get('_overall', []))}")
    print(f"  平均收敛轮数: {_avg_rounds(layer_rounds.get('_overall', []))}"
          f"（门限 {threshold:.0%}）")

    overall_pass = layer_stats.get("_overall", [])
    overall = (sum(overall_pass) / len(overall_pass)) if overall_pass else 0.0
    top1_all = layer_top1.get("_overall", [])
    top1_acc = (sum(top1_all) / len(top1_all)) if top1_all else 1.0

    # 门禁：通过率与 Top-1 命中率都要达标（命中率无案例考核时视为通过）
    return overall >= threshold and top1_acc >= threshold and all_ok


# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="分诊评测运行器（C3）")
    parser.add_argument("--live", action="store_true", help="跑端到端实况（需真实服务）")
    parser.add_argument("--threshold", type=float, default=0.8, help="live 模式准确率门限")
    args = parser.parse_args()

    if not args.live:
        print("评分级门禁（纯函数，无外部依赖）：")
        passed, failed, failures = run_scoring_cases()
        print(f"\n通过 {passed} / {passed + failed}")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1 if failed else 0)

    ok = asyncio.run(run_live_cases(threshold=args.threshold))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
