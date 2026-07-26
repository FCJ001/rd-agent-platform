# ============================================================
# FastAPI 依赖：身份注入
#
# ★ 全项目【唯一】一处把「请求」翻译成「身份」的地方。
#   当前是开发期实现：不验签，直接按 X-User-Id 查 users 表。
#   真实环境 token 由 ALM 平台（Java, RS256）签发，届时把函数体换成
#   verify_jwt(token) 解 claims 即可，所有调用方一行不改。
#   —— 收敛在一个函数里，正是为了让这次替换只有一个改动点。
#
# 调试：curl -H "X-User-Id: 3"，换个数字就换角色，比走登录快得多。
# ============================================================

from dataclasses import dataclass

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import ERR_USER_INACTIVE, ERR_USER_NOT_FOUND, BizException
from src.core.logger import logger
from src.infra.db import get_db
from src.modules.user.model import User


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
    x_user_id: int = Header(1, alias="X-User-Id"),
    x_session_id: str = Header("", alias="X-Session-Id"),
    db: AsyncSession = Depends(get_db),
) -> UserContext:
    """开发期实现：按 header 里的 user_id 查表，不验签"""
    user = await db.get(User, x_user_id)
    if user is None:
        raise BizException(f"用户不存在：user_id={x_user_id}", ERR_USER_NOT_FOUND)
    if not user.is_active:
        raise BizException(f"用户已停用：{user.username}", ERR_USER_INACTIVE)

    logger.debug(f"身份注入 user={user.username} role={user.role_type} "
                 f"line={user.business_line} domain={user.owner_domain_id}")

    return UserContext(
        user_id=str(user.id),
        session_id=x_session_id,
        role=user.role_type,
        business_line=user.business_line,
        owner_domain_id=user.owner_domain_id,
        real_name=user.real_name,
    )
