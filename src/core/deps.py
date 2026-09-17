# ============================================================
# FastAPI 依赖：身份注入
#
# ★ 全项目【唯一】一处把「请求」翻译成「身份」的地方。
#   认证链路（三选一，按优先级）：
#     1. X-Auth-Token  服务自签 HMAC 令牌（scripts/mint_token.py 签发）—— 任何环境可用
#     2. X-User-Id     明文用户 ID 直通 —— 仅 config.is_dev 白名单（dev/local/test）
#        可用，staging 与未知环境一律按生产拒绝。保留它是因为本地联调/
#        集成测试换身份最快
#   两个都没有 → 401。★ 绝不允许「缺省放行成某个固定用户」——
#   那等于把整套路由/行过滤/脱敏建在受骗的身份上。
#
#   真实环境 token 由 ALM 平台（Java, RS256）签发，届时在 verify 分支
#   加一个 JWT 分支即可，所有调用方一行不改。
# ============================================================

from collections.abc import AsyncGenerator
from dataclasses import dataclass

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import verify_user_token
from src.core.config import get_settings
from src.core.exceptions import (
    ERR_TOKEN_INVALID,
    ERR_USER_INACTIVE,
    ERR_USER_NOT_FOUND,
    BizException,
)
from src.core.logger import logger
from src.core.scope import business_line_scope
from src.infra.db import get_db
from src.modules.user.model import User

settings = get_settings()


@dataclass
class UserContext:
    """
    贯穿全链路的身份上下文。

    也是 LangGraph Agent 的 context_schema —— 同一个对象既喂给
    Repository 做行过滤，也喂给 Agent 做分层输出，不做第二套。

    ★ 后三个字段是行级过滤、字段脱敏、分层输出的【唯一输入】。
      认证可以 mock，但这三个维度必须一路贯穿：
      砍掉它们等于把「多源异构反馈归一化」这个技术难题一起砍了。
    """

    user_id: str
    session_id: str = ""
    role: str = "customer"              # engineer | business | aftersales | customer | admin
    business_line: str | None = None    # 业务角色按此过滤
    owner_domain_id: int | None = None  # 工程师按此过滤
    real_name: str | None = None


async def get_current_user(
    x_auth_token: str = Header("", alias="X-Auth-Token"),
    x_user_id: str = Header("", alias="X-User-Id"),
    x_session_id: str = Header("", alias="X-Session-Id"),
    db: AsyncSession = Depends(get_db),
) -> AsyncGenerator[UserContext, None]:
    """解析请求身份 → 查 users 表 → 构造 UserContext。见文件头注释。

    ★ 用 yield 形式（生成器依赖）而不是直接 return：请求处理期间把该用户的
      业务线写进 ContextVar，FastAPI 会在响应结束后执行 finally 恢复原值。
      不恢复的话，ASGI 协程复用会让下一个请求继承上一个人的作用域 ——
      表现是「A 业务线的用户偶尔看到 B 业务线的数据」，且难以复现。
    """
    if x_auth_token:
        user_id = verify_user_token(x_auth_token)
    elif x_user_id and settings.is_dev:
        # ★ 用 is_dev 白名单而不是 APP_ENV != "prod"：staging/uat/拼错的
        #   环境名一律按生产处理，明文直通绝不放开（fail-closed）
        if not x_user_id.isdigit():
            raise BizException("X-User-Id 必须是数字用户 ID", ERR_TOKEN_INVALID)
        user_id = int(x_user_id)
    else:
        raise BizException(
            "缺少身份凭证：请携带 X-Auth-Token（生产）或 X-User-Id（仅开发环境）",
            ERR_TOKEN_INVALID,
        )

    # session_id 会拼进 Redis key / checkpointer thread_id / 影子表：
    # 不设上限的请求头能让单条记录撑爆存储，超长直接拒绝
    if len(x_session_id) > 128:
        raise BizException("X-Session-Id 过长（上限 128 字符）", ERR_TOKEN_INVALID)

    user = await db.get(User, user_id)
    if user is None:
        raise BizException(f"用户不存在：user_id={user_id}", ERR_USER_NOT_FOUND)
    if not user.is_active:
        raise BizException(f"用户已停用：{user.username}", ERR_USER_INACTIVE)

    logger.debug(f"身份注入 user={user.username} role={user.role_type} "
                 f"line={user.business_line} domain={user.owner_domain_id}")

    ctx = UserContext(
        user_id=str(user.id),
        session_id=x_session_id,
        role=user.role_type,
        business_line=user.business_line,
        owner_domain_id=user.owner_domain_id,
        real_name=user.real_name,
    )

    # 作用域在整个请求生命周期内有效（含 LLM 工具调用、图节点里的 DB 访问），
    # 响应结束后由 FastAPI 执行清理，恢复进入前的值
    with business_line_scope(user.business_line):
        yield ctx