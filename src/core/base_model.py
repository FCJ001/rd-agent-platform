# ============================================================
# ORM 基类
#
# Base          —— DeclarativeBase，Alembic autogenerate 靠 Base.metadata 找表
# TimestampMixin —— created_at / updated_at，服务端时间，不信客户端传的
# BaseModel     —— 业务表统一继承：自增 BigInteger 主键 + 两个时间戳
#
# ★ __abstract__ = True 一定要写，否则 SQLAlchemy 会把 BaseModel 本身也当成一张表
# ============================================================

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 Model 的根。Alembic env.py 里 target_metadata = Base.metadata"""
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间"
    )


class BaseModel(Base, TimestampMixin):
    __abstract__ = True

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
