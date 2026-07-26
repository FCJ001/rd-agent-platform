# ============================================================
# 行级过滤验收测试（步 1 的验收闸门）
#
# 前置：中间件已起 + 三个 seed 脚本已跑
#   docker compose up -d
#   alembic upgrade head
#   python scripts/seed_domains.py && python scripts/seed_users.py && python scripts/seed_issues.py
#
# 运行：pytest tests/test_row_filter.py -v
#
# ★ 这套测试的核心不是"每个角色返回了 N 条"，而是"N 两两不同"。
#   只断言行数对不对，是无法区分「过滤生效」和「两个分支恰好返回同样多」的。
# ============================================================

import pytest
from httpx import AsyncClient

# 走已运行的 uvicorn，不用 ASGITransport。
# ASGITransport 在 pytest-asyncio 下会创建独立 event loop，
# 而 SQLAlchemy engine 是模块导入时在主进程 event loop 上建的 —— 必炸。
BASE = "http://localhost:8000"

# (user_id, username, role, 期望行数)
ROLE_CASES = [
    (1, "eng01", "engineer", 10),   # 电池系统域
    (10, "biz_ev", "business", 12),  # ev 线
    (12, "service01", "aftersales", 5),   # closed + verified
    (13, "cust01", "customer", 3),   # 只有自己上报的
    (14, "admin", "admin", 24),  # 全部
]

# 反例：这两个域没有问题单，工程师应该什么都看不到
EMPTY_CASES = [(4, "eng04 热管理域"), (5, "eng05 整车控制域")]


@pytest.fixture
async def client():
    async with AsyncClient() as c:
        yield c


async def _total(client, user_id: int, **params) -> int:
    resp = await client.get(f"{BASE}/api/v1/issues",
                            headers={"X-User-Id": str(user_id)},
                            params={"page_size": 100, **params})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 200, body
    return body["data"]["total"]


@pytest.mark.parametrize("user_id,username,role,expect", ROLE_CASES)
async def test_row_filter_by_role(client, user_id, username, role, expect):
    assert await _total(client, user_id) == expect, f"{username}({role}) 行数不符"


@pytest.mark.parametrize("user_id,label", EMPTY_CASES)
async def test_engineer_without_issues_sees_nothing(client, user_id, label):
    """反例：过滤不仅要"能看到该看的"，还要"看不到不该看的\""""
    assert await _total(client, user_id) == 0, f"{label} 不该看到任何单"


async def test_role_totals_are_pairwise_distinct(client):
    """
    ★ 全套测试里最关键的一条。

    如果 engineer 和 business 都返回 12 条，上面的断言照样能过，
    但那证明不了过滤生效 —— 有可能两个分支走到了同一个 WHERE，
    甚至两个都没加条件。行数互不相同，才能从结果反推出走的是不同的过滤路径。
    """
    totals = [await _total(client, uid) for uid, *_ in ROLE_CASES]
    assert len(set(totals)) == len(totals), f"行数出现重复 {totals}，夹具失去区分能力"


async def test_customer_filter_is_reporter_not_source(client):
    """
    customer 的过滤必须是 reporter_id，不能是 source == 'customer'。

    夹具里有 6 条 source='customer'，其中 3 条 reporter_id 为空
    （模拟 NHTSA 外部导入的投诉，平台里没有对应账号）。
    所以 cust01 应该看到 3 条 —— 如果有人把条件误写成 source，这里会变成 6。
    """
    assert await _total(client, 13) == 3


async def test_pagination_total_does_not_leak(client):
    """
    分页时 total 必须是【过滤后】的总数。

    「列表只给 3 条、total 却报 24」是很隐蔽的一类越权：
    行是挡住了，但库里到底有多少条已经泄露出去了。
    """
    resp = await client.get(f"{BASE}/api/v1/issues",
                            headers={"X-User-Id": "13"},
                            params={"page_size": 2})
    data = resp.json()["data"]
    assert data["total"] == 3
    assert len(data["items"]) == 2


async def test_detail_uses_same_filter_as_list(client):
    """
    「列表过滤了、详情没过滤」是越权最常见的入口 ——
    列表里看不到别人的单，但把 URL 里的 id 改一改照样能拿到。
    id=2 是 eng01 报的单，cust01 必须拿不到。
    """
    ok = await client.get(f"{BASE}/api/v1/issues/2", headers={"X-User-Id": "1"})
    assert ok.json()["code"] == 200

    denied = await client.get(f"{BASE}/api/v1/issues/2", headers={"X-User-Id": "13"})
    body = denied.json()
    assert body["code"] == 40301
    # 报"不存在"而不是"无权限"——后者等于确认了这个 id 存在
    assert "不存在" in body["message"]


async def test_unknown_user_is_rejected(client):
    resp = await client.get(f"{BASE}/api/v1/issues", headers={"X-User-Id": "999"})
    assert resp.json()["code"] == 40201


async def test_trace_id_propagates_from_upstream(client):
    """
    Java 网关生成的 trace_id 必须沿用，不能每跳重新生成 ——
    否则一次请求穿过三个服务后，日志串不起来。
    """
    resp = await client.get(f"{BASE}/health", headers={"X-Trace-Id": "java-gw-test-001"})
    assert resp.headers["X-Trace-Id"] == "java-gw-test-001"

    # 没有上游 trace_id 时本地生成一个，不能为空
    resp2 = await client.get(f"{BASE}/health")
    assert resp2.headers.get("X-Trace-Id")
