# ============================================================
# 跨业务线泄漏门禁（集成测试）
#
# ★ 这套测试的存在意义：区分「设计上隔离了」和「真的隔离了」。
#   单测只能证明「发出去的查询长什么样」（谓词拼了、参数绑了），
#   证明不了「查回来的东西里没有别人家的数据」——那需要真数据。
#
# 覆盖三条真实隔离边界（都是本次改造动过的地方）：
#   1. Neo4j 候选根因：现象节点是 (business_line, name) 复合键，
#      按线查不得捞出另一条线的根因
#   2. PG 历史结论：按项目维度查不得超过 business_line
#   3. PG 行级安全：启用 RLS 且以非超级用户连接时，未设作用域必须查不到数据
#
# 前置：docker compose 起来 + alembic upgrade head + 种子/图谱导入
#   docker compose up -d
#   alembic upgrade head
#   python scripts/init_postgres.py && python scripts/init_neo4j.py
#   pytest -m integration tests/test_scope_isolation_integration.py -v
#
# ★ 断言写法要点：不要断言「返回了 N 条」，要断言「返回的每一条都属于
#   当前线」。前者在数据变化时会脆断，也分辨不出「过滤生效」和「恰好只有一边有数据」。
# ============================================================

import os

import pytest
import psycopg2

from src.core.config import get_settings
from src.core.logger import logger

pytestmark = [pytest.mark.integration]

settings = get_settings()

# 跨线共享的现象名（实测 26 个现象名里 13 个跨线）—— 用它才能试出泄漏：
# 若现象节点/查询任一环节漏了线过滤，两边数据会混在一起
SHARED_PHENOMENON = "系统响应迟滞"


def _pg_connect():
    """测试自用的 PG 连接 —— 与应用同源：建连后立刻带上当前作用域。

    ★ 不带上作用域的话，RLS 启用时这里任何查询都读不到行，
      会把「读不到」误判成「归属不符」，测出假阳性。
    """
    from src.infra.pg_scope import apply_scope

    conn = psycopg2.connect(
        host=settings.DB_HOST, port=settings.DB_PORT,
        user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
    )
    apply_scope(conn)
    return conn


def _lines_in_graph() -> list[str]:
    from src.infra.neo4j_client import get_neo4j_driver

    driver = get_neo4j_driver()
    with driver.session() as s:
        rows = s.run(
            "MATCH (rc:RootCause) RETURN DISTINCT rc.business_line AS line ORDER BY line"
        ).data()
    return [r["line"] for r in rows if r["line"]]


# ── 1. Neo4j：候选根因不得跨线 ──────────────────────────────────

def test_candidate_causes_never_cross_line():
    """按线查候选根因，返回的每一个根因都必须属于该线。

    ★ 同时断言每条线都查到了东西 —— 否则"没查到"会让跨线断言空转通过。
    """
    from src.agents.triage.graph_queries import query_causes_by_phenomena

    lines = _lines_in_graph()
    if len(lines) < 2:
        pytest.fail(
            f"图谱只有 {lines} 条业务线，无法验证跨线隔离。"
            "请先跑 scripts/init_neo4j.py 导入完整图谱"
        )

    hit_lines = []
    for line in lines:
        # top_k 放大：默认 10 条会被排序切掉一部分线，看不到全貌
        candidates = query_causes_by_phenomena(
            [SHARED_PHENOMENON], business_line=line, top_k=100
        )
        if candidates:
            hit_lines.append(line)
        wrong = [c.code for c in candidates if c.business_line != line]
        assert not wrong, (
            f"作用域={line} 却查出其他线的根因 {wrong}（现象「{SHARED_PHENOMENON}」跨线共享，"
            "正是最容易漏过滤的场景）"
        )

    assert len(hit_lines) >= 2, (
        f"只有 {hit_lines} 查到了候选 —— 该现象在当前图谱里不够跨线，"
        "跨线隔离断言会空转通过。请换一个真正跨线共享的现象名"
    )


def test_unscoped_query_does_show_multiple_lines():
    """反向断言：不带作用域时确实会捞出多条线。

    ★ 没有这条，上面那条测试可能是空的 —— 如果两条线根本没共用该现象，
    「按线过滤」和「不过滤」返回的结果一样，测试通不出任何东西。

    ★ top_k 必须放大：真实泄漏形态不是整齐的"两线各半"，而是按
      matched_count/total_count 排出来的一个切片 —— 现象越窄的根因越靠前。
      默认 top_k=10 时这个现象的头部可能全落在同一条线，看不出跨线。
    """
    from src.agents.triage.graph_queries import query_causes_by_phenomena

    candidates = query_causes_by_phenomena([SHARED_PHENOMENON], top_k=100)
    found = {c.business_line for c in candidates}
    assert len(found) > 1, (
        f"不带作用域只查到 {found} 一条线 —— 该现象可能已不再跨线共享，"
        "请换一个跨线现象名，否则跨线隔离测试失去意义"
    )


# ── 2. PG：历史结论不得跨线 ─────────────────────────────────────

async def test_past_diagnoses_never_cross_line():
    """项目维度的历史检索，返回的每条记录都必须属于该线。

    ★ 业务线从**配置**取而不是从库里查：RLS 启用后不设作用域一行都读不到，
      想"先查有哪些线"会陷入鸡生蛋问题（这也正是 RLS 该有的样子）。
    ★ 校验用直连查询，且连接必须经 apply_scope 带上同一作用域 ——
      否则在 RLS 下校验查询自己也读不到行，会误判成"归属不符"。
    """
    from src.agents.triage.history import SCOPE_LINE, query_past_diagnoses
    from src.core.scope import business_line_scope
    from src.infra.pg_scope import apply_scope

    lines = sorted(settings.business_lines)
    total = 0
    per_line: dict[str, int] = {}

    for line in lines:
        with business_line_scope(line):
            result = await query_past_diagnoses(
                keywords=SHARED_PHENOMENON,
                role="engineer", user_id="1", business_line=line, scope=SCOPE_LINE,
            )
            per_line[line] = len(result["items"])
            for item in result["items"]:
                conn = _pg_connect()  # 已在作用域内，连接自带 scope
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """SELECT business_line FROM ai_triage_results
                           WHERE primary_cause_code = %s""",
                        (item["cause_code"],),
                    )
                    rows = [r[0] for r in cur.fetchall()]
                    cur.close()
                finally:
                    conn.close()
                assert line in rows, (
                    f"作用域={line} 查出的结论 {item['cause_code']} 在该作用域下读不到 "
                    f"（实际读到 {rows}）—— 说明检索绕过了作用域过滤"
                )
                total += 1

    if total == 0:
        pytest.skip(
            f"两条线都查不到 {SHARED_PHENOMENON} 的历史结论"
            "（新建库未回填 ai_triage_results 的 business_line，属预期）"
        )


# ── 3. PG：RLS 未设作用域必须 fail-closed ──────────────────────

def test_rls_blocks_unscoped_access_when_enabled():
    """RLS 已启用时：不设 app.business_line，受控表必须查不到任何行。

    未启用 RLS 时跳过（启用是运维动作，见 scripts/enable_rls.py）。
    """
    conn = _pg_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT relname, relrowsecurity FROM pg_class
               WHERE relname = 'phenomena' AND relkind = 'r'"""
        )
        row = cur.fetchone()
        rls_on = bool(row and row[1])
        # 未启用时默认跳过（启用是运维动作，需要先做角色拆分）；
        # 但 CI 里可以设 REQUIRE_RLS=1 把"未启用"当失败 ——
        # 否则一旦有人把 RLS 关掉，这条测试会静默跳过，隔离失效无人察觉
        require_rls = os.getenv("REQUIRE_RLS", "").lower() in ("1", "true", "yes")
        if not rls_on:
            if require_rls:
                pytest.fail("REQUIRE_RLS=1 但 phenomena 未启用 RLS —— 隔离已失效")
            pytest.skip("phenomena 未启用 RLS —— 见 scripts/enable_rls.py")

        # 当前角色是否受 RLS 约束：超级用户与 BYPASSRLS 角色无条件绕过
        cur.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
        is_super, bypass = cur.fetchone()
        if is_super or bypass:
            pytest.fail(
                f"当前角色 {settings.DB_USER} 是超级用户或带 BYPASSRLS —— "
                "它绕过 RLS，本测试无法验证隔离效果（这也正是 RLS 部署要先做角色拆分的原因）"
            )

        cur.execute("BEGIN")
        cur.execute("SELECT count(*) FROM phenomena")
        unscoped = cur.fetchone()[0]
        cur.execute("ROLLBACK")

        assert unscoped == 0, (
            f"未设 app.business_line 却看到 {unscoped} 行 —— RLS 未生效。"
            "排查：策略是否建了、表是否 ENABLE ROW LEVEL SECURITY、连接角色是否受约束"
        )
        cur.close()
    finally:
        conn.close()
