# ============================================================
# 身份令牌单元测试 —— 纯函数，CI 可跑
#
# 锁住 deps.py 鉴权改造的安全底线：
#   1. 签发 → 校验往返成功
#   2. 篡改签名 / 过期 / 乱格式 → 一律拒绝
# ============================================================

import pytest

from src.core.auth import sign_user_token, verify_user_token
from src.core.exceptions import ERR_TOKEN_INVALID, BizException


def test_roundtrip():
    token = sign_user_token(42)
    assert verify_user_token(token) == 42


def test_tampered_signature_rejected():
    token = sign_user_token(42)
    bad = token[:-4] + ("0000" if not token.endswith("0000") else "1111")
    with pytest.raises(BizException) as ei:
        verify_user_token(bad)
    assert ei.value.code == ERR_TOKEN_INVALID


def test_wrong_secret_rejected(monkeypatch):
    token = sign_user_token(42, ttl_seconds=60)
    monkeypatch.setattr("src.core.auth._secret", lambda: b"another-secret")
    with pytest.raises(BizException):
        verify_user_token(token)


def test_expired_token_rejected():
    token = sign_user_token(42, ttl_seconds=-10)  # 已过期
    with pytest.raises(BizException):
        verify_user_token(token)


@pytest.mark.parametrize("bad", ["", "garbage", "v1.abc.123.signature", "v2.42.9999999999.abcdef"])
def test_malformed_tokens_rejected(bad):
    with pytest.raises(BizException):
        verify_user_token(bad)
