# ============================================================
# 从知识图谱批量生成实况评测场景
#
# 思路：场景不手写——遍历图谱真实数据（根因×现象），对每个候选场景
# 用「真实图谱查询 + 真实评分代码」仿真，按仿真结果机械标定断言：
#   - 首轮收敛且 top1 置信 ≥0.65 → L2 场景（expect_cause=[top1]）
#   - 首轮不收敛 → L1 场景（expect_converged=False，验证不硬收敛）
#   - 两现象场景取 top1 的竞争根因做否认 → L4 场景
#   - L5 安全/边界沿用手工场景
# 已有 20 个手工场景保留，新增挂在其后。
# ============================================================
import sys, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
random.seed(42)  # 确定性：措辞模板可复现

from src.agents.triage.graph_queries import query_causes_by_phenomena, enrich_cause_details
from src.agents.triage.confidence import apply_context_weights, check_convergence


def _line_of(code: str) -> str:
    """RC-EV-0012 → ev。根因编码带业务线前缀，是场景作用域的唯一来源。"""
    parts = (code or "").split("-")
    return parts[1].lower() if len(parts) >= 2 else ""


def simulate(phenomena, denied=None, dtc=None, business_line=""):
    """模拟一轮候选检索 + 置信度计算。

    ★ business_line 必须与线上一致：现象节点在 Neo4j 里是
      (business_line, name) 复合键，不过滤会把另一条线的根因一起召回，
      生成的期望值就与真实分诊行为不符（批量场景会集体假失败）。
    """
    cands = query_causes_by_phenomena(phenomena, business_line)
    if not cands:
        return None
    cands = enrich_cause_details(cands)
    cands = apply_context_weights(cands, dtc or [], denied or [])
    top = cands[0]
    should, _ = check_convergence(cands, 0)
    return {
        "top": top.code, "conf": top.confidence,
        "should": should, "n": len(cands),
        "all": cands,
    }


def main():
    cases_path = Path(__file__).resolve().parent / "cases" / "live_cases.json"
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    existing = data["cases"]
    have_ids = {c["id"] for c in existing}

    # 图谱全量：根因 → 典型现象
    from src.infra.neo4j_client import get_neo4j_driver
    driver = get_neo4j_driver()
    with driver.session() as sess:
        rows = sess.run("""
            MATCH (rc:RootCause)-[:INDICATES]->(ph:Phenomenon)
            RETURN rc.code AS code, rc.name AS name,
                   collect(DISTINCT ph.name) AS phenoms
            ORDER BY rc.code""").data()

    new_cases = []
    n = 0

    def add(layer, title, turns, **expect):
        nonlocal n
        n += 1
        cid = f"GEN-{n:03d}"
        if cid in have_ids:
            return
        c = {"id": cid, "layer": layer, "title": title, "turns": turns}
        c.update(expect)
        new_cases.append(c)

    # ── 遍历每个根因 → L2 双现象场景 ──
    for rc in rows:
        code, name, phenoms = rc["code"], rc["name"], rc["phenoms"]
        if len(phenoms) < 2:
            continue
        p1, p2 = phenoms[0], phenoms[1]
        r = simulate([p1, p2], business_line=_line_of(code))
        if r is None or r["top"] != code:
            continue  # 双现象下它不是第一候选 → 跳过（避免断言依赖排序细节）
        if r["conf"] >= 0.65 and r["should"]:
            add("L2", f"批量：{name}——典型双现象组合，首轮收敛",
                [f"{p1}，而且{p2}"],
                expect_phenomena=[p1[:2]], expect_cause=[code], expect_converged=True)

    # ── 单现象 → L1 不硬收敛（验证追问而非幻觉结论）──
    all_phenoms = []
    for rc in rows:
        all_phenoms.extend(rc["phenoms"])
    for ph in sorted(set(all_phenoms)):
        r = simulate([ph])
        if r is None:
            continue
        if not r["should"]:
            add("L1", f"批量：单现象「{ph}」区分度不足，应追问不硬收敛",
                [f"车有点{ph}的问题"],
                expect_phenomena=[ph[:2]], expect_converged=False)

    # ── L4：否认竞争根因的核心现象 ──
    for rc in rows:
        code, phenoms = rc["code"], rc["phenoms"]
        if len(phenoms) < 3:
            continue
        main2 = phenoms[:2]
        r_all = simulate(main2, business_line=_line_of(code))
        if r_all is None or r_all["top"] != code:
            continue
        # ★ 结构约束：主流程（轮1）必须不收敛，否认轮（轮2）才可达。
        #   否则轮 1 就结束，否认断言必然失败（此前 7 个假失败全是这个原因）。
        if r_all["should"]:
            continue
        # 找一个共享但非核心的现象做否认目标
        for other in r_all["all"][1:3]:
            shared = set(other.matched_phenomena) & set(phenoms)
            if not shared:
                continue
            deny_ph = sorted(shared)[0]
            r4 = simulate(main2, denied=[deny_ph], business_line=_line_of(code))
            if r4 is None:
                continue
            kept_top = r4["top"]
            # 否认后 top1 不应是「被否认现象唯一指向」的根因
            add("L4", f"批量：{name}——否认竞争现象「{deny_ph}」收窄候选",
                [f"{main2[0]}，{main2[1]}",
                 f"另外{deny_ph}是不存在的，没这个问题"],
                expect_phenomena=[main2[0][:2]],
                expect_denied_nonempty=True,
                expect_cause=[kept_top] if kept_top in [code] else None,
                expect_converged=r4["should"])
            break

    # 合并：保留手工 20，追加批量
    merged = existing + [c for c in new_cases if c not in existing]
    out = {"description": f"L1-L5 端到端实况场景：手工 20 + 图谱批量生成 {len(new_cases)}（断言由真实图谱+评分代码仿真标定）",
           "cases": merged}
    cases_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import Counter
    print(f"新增 {len(new_cases)} 场景，总计 {len(merged)}")
    print("新增分层:", dict(Counter(c["layer"] for c in new_cases)))
    print("总分层:", dict(Counter(c["layer"] for c in merged)))


if __name__ == "__main__":
    main()
