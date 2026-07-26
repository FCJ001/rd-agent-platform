# ============================================================
# 平台用户
#
# ★ 只提供「授权三维」，不承担认证职责：
#   没有 password_hash、没有 last_login、没有 token 相关字段。
#   登录归 Java ALM 平台，本服务的 get_current_user() 开发期按 X-User-Id 查这张表，
#   将来换成 verify_jwt(token) 也只改那一个函数，这张表不动。
#
# 也没有 roles / permissions / user_roles / role_permissions 四张表：
#   4 个角色 × 一把资源，行过滤规则写成代码常量（src/core/permissions.py）就够，
#   权限主数据在平台侧，这里再存一份就是第二份事实。
# ============================================================

from sqlalchemy import Boolean, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.base_model import BaseModel


class User(BaseModel):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="平台登录名，与 ALM 平台一致")
    real_name: Mapped[str | None] = mapped_column(String(50), comment="真实姓名")
    # ★ 授权三维 —— 行级过滤与字段脱敏的唯一输入
    role_type: Mapped[str] = mapped_column(String(20), nullable=False, default="customer",
                                           comment="engineer/business/aftersales/customer/admin")
    business_line: Mapped[str | None] = mapped_column(String(10), comment="所属业务线（业务角色按此过滤）")
    owner_domain_id: Mapped[int | None] = mapped_column(ForeignKey("owner_domains.id", ondelete="SET NULL"),
                                                        comment="所属责任域（工程师按此过滤）")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, comment="是否启用")

    __table_args__ = (Index("ix_users_role", "role_type"),)
