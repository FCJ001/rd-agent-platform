# ============================================================
# 服务自签身份令牌（HMAC-SHA256）
#
# 生产形态是 Java ALM 平台签发的 JWT（RS256），本服务只验签。
# 平台接入前，用同一个 deps.get_current_user() 换token入口的方式，
# 先用服务自签的 HMAC 令牌顶上 —— 签发入口收敛在 scripts/mint_token.py，
# 将来切 JWT 只改本文件 + deps.py 的解析分支，调用方不动。
#
# 令牌格式：v1.{user_id}.{expires_ts}.{sig}
#   sig = HMAC-SHA256(AUTH_SECRET, "{user_id}.{expires_ts}") 截断 32 位 hex
# ============================================================

import hashlib
import hmac
import time

from src.core.config import get_settings
from src.core.exceptions import ERR_TOKEN_INVALID, BizException

# 令牌前缀，将来兼容 "jwt." 前缀的多方案分发
_PREFIX = "v1"
_SIG_LEN = 32
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600


def _secret() -> bytes:
    secret = get_settings().AUTH_SECRET
    if not secret:
        # dev 缺省：用固定弱密钥并显著告警，保证本地能跑通；
        # 生产由 validate_production() 拒绝启动，走不到这里
        from src.core.logger import logger
        logger.warning("AUTH_SECRET 未配置，回退开发期弱密钥 —— 仅限本地开发！")
        secret = "insecure-dev-secret-do-not-use-in-prod"
    return secret.encode()


def sign_user_token(user_id: int | str, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> str:
    """为指定用户签发令牌。"""
    exp = int(time.time()) + ttl_seconds
    msg = f"{user_id}.{exp}"
    sig = hmac.new(_secret(), msg.encode(), hashlib.sha256).hexdigest()[:_SIG_LEN]
    return f"{_PREFIX}.{user_id}.{exp}.{sig}"


def verify_user_token(token: str) -> int:
    """校验令牌，返回 user_id。无效/过期统一抛 BizException(40100)。"""
    parts = token.strip().split(".")
    if len(parts) != 4 or parts[0] != _PREFIX:
        raise BizException("身份令牌格式不合法", ERR_TOKEN_INVALID)
    _, user_id_raw, exp_raw, sig = parts
    if not user_id_raw.isdigit() or not exp_raw.isdigit():
        raise BizException("身份令牌格式不合法", ERR_TOKEN_INVALID)
    msg = f"{user_id_raw}.{exp_raw}"
    expect = hmac.new(_secret(), msg.encode(), hashlib.sha256).hexdigest()[:_SIG_LEN]
    # compare_digest 防时序侧信道
    if not hmac.compare_digest(sig, expect):
        raise BizException("身份令牌签名校验失败", ERR_TOKEN_INVALID)
    if int(exp_raw) < int(time.time()):
        raise BizException("身份令牌已过期，请重新获取", ERR_TOKEN_INVALID)
    return int(user_id_raw)
