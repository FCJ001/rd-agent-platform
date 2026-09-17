# ============================================================
# 环境判定 + 生产配置校验的单元测试 —— 纯函数，CI 可跑
#
# 锁住 config.py 的安全底线：
#   1. is_prod / is_dev 对大小写、空白、"production" 等写法归一化 ——
#      防止 "Production" 让某处判 prod、另一处判非 prod 的错位绕过
#   2. validate_production：默认弱口令 / 短密钥 / CORS 全开 → 拒绝启动
#
# 全部字段显式传 kwargs（优先级高于 .env），测试结果不随本机环境漂移
# ============================================================

import pytest

from src.core.config import Settings


def _prod_settings(**overrides) -> Settings:
    """一套能通过生产校验的基线配置，逐项覆盖出反例。"""
    base = dict(
        APP_ENV="prod",
        APP_DEBUG=False,
        AUTH_SECRET="a" * 32,
        WEBHOOK_SECRET="b" * 32,
        CORS_ALLOW_ORIGINS="https://alm.example.com",
        DB_PASSWORD="strong-pg-pass-9f3a",
        NEO4J_PASSWORD="strong-neo-pass-8c2b",
        MINIO_ACCESS_KEY="strong-minio-key",
        MINIO_SECRET_KEY="strong-minio-secret-7d1c",
        REDIS_PASSWORD="strong-redis-pass-6e4d",
    )
    base.update(overrides)
    return Settings(**base)


# ── 环境判定归一化 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("env", ["prod", "PROD", " Production ", "production"])
def test_is_prod_normalization(env):
    assert Settings(APP_ENV=env).is_prod


@pytest.mark.parametrize("env", ["dev", "DEV", " local ", "test", "staging", "", "unknown"])
def test_is_dev_whitelist(env):
    s = Settings(APP_ENV=env)
    # 只有显式开发/测试环境才放行不安全通道；staging/未知值 fail-closed
    assert s.is_dev == (env.strip().lower() in ("dev", "development", "local", "test"))


def test_staging_is_not_prod_but_not_dev():
    """staging：不算生产（不跑 validate_production），但也不许 X-User-Id 直通。"""
    s = Settings(APP_ENV="staging")
    assert not s.is_prod
    assert not s.is_dev


# ── 生产启动校验 ────────────────────────────────────────────────────────────

def test_validate_production_passes_strong_config():
    _prod_settings().validate_production()  # 不抛即通过


@pytest.mark.parametrize("field,weak", [
    ("DB_PASSWORD", "rdagent123"),
    ("DB_PASSWORD", "change-me-postgres"),
    ("DB_PASSWORD", ""),
    ("NEO4J_PASSWORD", "rdagent123"),
    ("NEO4J_PASSWORD", "change-me-neo4j"),
    ("MINIO_ACCESS_KEY", "minioadmin"),
    ("MINIO_ACCESS_KEY", "change-me-minio"),
    ("MINIO_SECRET_KEY", "minioadmin"),
    ("MINIO_SECRET_KEY", "change-me-minio-secret"),
])
def test_validate_production_rejects_default_passwords(field, weak):
    with pytest.raises(RuntimeError, match=field):
        _prod_settings(**{field: weak}).validate_production()


@pytest.mark.parametrize("field", ["AUTH_SECRET", "WEBHOOK_SECRET"])
def test_validate_production_rejects_short_secrets(field):
    with pytest.raises(RuntimeError, match=field):
        _prod_settings(**{field: "short"}).validate_production()


def test_validate_production_rejects_debug():
    with pytest.raises(RuntimeError, match="APP_DEBUG"):
        _prod_settings(APP_DEBUG=True).validate_production()


def test_validate_production_rejects_open_cors():
    with pytest.raises(RuntimeError, match="CORS"):
        _prod_settings(CORS_ALLOW_ORIGINS="").validate_production()


def test_validate_production_rejects_weak_redis_password():
    with pytest.raises(RuntimeError, match="REDIS_PASSWORD"):
        _prod_settings(REDIS_PASSWORD="123456").validate_production()


def test_validate_production_collects_multiple_problems():
    """多个问题一起报，运维不用改一轮启动一轮。"""
    with pytest.raises(RuntimeError) as ei:
        _prod_settings(APP_DEBUG=True, AUTH_SECRET="").validate_production()
    msg = str(ei.value)
    assert "APP_DEBUG" in msg and "AUTH_SECRET" in msg
