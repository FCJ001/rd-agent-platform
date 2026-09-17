# ============================================================
# 限流单元测试 —— FakeRedis，无外部服务，CI 可跑
#
# 锁住 src/core/rate_limit.py 的行为底线：
#   1. 窗口内未超限 → 放行；第 limit+1 次 → BizException(42901)
#   2. 不同用户/不同 scope 的计数互不挤兑
#   3. Redis 故障 fail-open（可用性优先，不放大故障）
# ============================================================

import pytest

from src.core import rate_limit as rl
from src.core.exceptions import ERR_RATE_LIMITED, BizException


class FakeRedis:
    """够 incr/expire 语义的最小替身。"""

    def __init__(self):
        self.counts: dict[str, int] = {}
        self.broken = False

    async def incr(self, key: str) -> int:
        if self.broken:
            raise ConnectionError("redis down")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, ttl: int) -> None:
        pass


@pytest.fixture
def fake_redis(monkeypatch):
    fr = FakeRedis()
    monkeypatch.setattr(rl, "_redis_client", fr)
    return fr


async def test_allows_up_to_limit(fake_redis):
    for i in range(3):
        await rl.enforce_rate_limit(fake_redis, "chat", "42", limit=3, window_seconds=60)
    assert sum(fake_redis.counts.values()) == 3


async def test_blocks_over_limit(fake_redis):
    with pytest.raises(BizException) as ei:
        for _ in range(4):
            await rl.enforce_rate_limit(fake_redis, "chat", "42", limit=3, window_seconds=60)
    assert ei.value.code == ERR_RATE_LIMITED


async def test_users_isolated(fake_redis):
    await rl.enforce_rate_limit(fake_redis, "chat", "1", limit=1, window_seconds=60)
    # 用户 1 已打满，用户 2 不受影响
    await rl.enforce_rate_limit(fake_redis, "chat", "2", limit=1, window_seconds=60)
    with pytest.raises(BizException):
        await rl.enforce_rate_limit(fake_redis, "chat", "1", limit=1, window_seconds=60)


async def test_scopes_isolated(fake_redis):
    await rl.enforce_rate_limit(fake_redis, "chat", "42", limit=1, window_seconds=60)
    await rl.enforce_rate_limit(fake_redis, "triage", "42", limit=1, window_seconds=60)


async def test_redis_failure_fail_open(fake_redis):
    fake_redis.broken = True
    # Redis 挂了 → 放行（fail-open），绝不抛 5xx/429
    await rl.enforce_rate_limit(fake_redis, "chat", "42", limit=1, window_seconds=60)
    await rl.enforce_rate_limit(fake_redis, "chat", "42", limit=1, window_seconds=60)
