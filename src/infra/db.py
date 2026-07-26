# ============================================================
# PostgreSQL 异步连接
#
# 业务代码全走这里的 get_db()；种子脚本（scripts/seed_*.py）另用 psycopg2 同步连接，
# 不共用引擎 —— 脚本是一次性的，没必要拖进 asyncio。
#
# ★ pool_pre_ping=True 必须开：容器重启后连接池里的旧连接是死的，
#   不 ping 会在第一次业务查询时抛 ConnectionDoesNotExistError。
# ============================================================

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG,   # 开发期打印 SQL，看行过滤有没有真拼进 WHERE
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=60 * 5,       # 5 分钟回收，躲开 PG 侧的空闲连接超时
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,    # commit 后对象属性仍可读，否则序列化时会触发一次懒加载
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：一个请求一个 session，正常结束提交，异常回滚"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
