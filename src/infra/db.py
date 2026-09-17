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
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.core.scope import get_business_line

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


class ScopedSession(Session):
    """本应用专用的 Session 子类 —— 业务线作用域写入钩子挂在它上面。

    ★ 不挂在全局 Session 类上：那样进程内所有 Session（含测试/脚本）
      都会被套上钩子。挂自有子类，作用面就是本应用的会话工厂。
    """


# ★ 注册在 Session 类而不是 Engine 上：after_begin 是「会话事件」，
#   Engine 上根本没有这个事件名（注册直接 AttributeError）。
#   而它「每个事务开始」都触发 —— 这点关键：请求中途一次 commit 就会
#   开启新事务，只在建会话时设一次的话，commit 之后作用域就丢了。
@event.listens_for(ScopedSession, "after_begin")
def _apply_business_line_scope(session, transaction, connection) -> None:
    """事务开始时把当前作用域写进连接，供 RLS 策略读取。

    ★ 必须用 SET LOCAL（第三参 is_local=true，即 set_config(..., true)），
      不能用 SET：连接池会把连接还给下一个请求，会话级变量会被下一个
      业务线继承 —— 这是 RLS 部署最常见的事故形态。SET LOCAL 钉在事务里，
      事务一结束自动失效。

    ★ 作用域为空时什么都不做：维护脚本/种子/后台任务没有业务线概念，
      写入空值会让 RLS 策略的 current_setting 拿到 '' 而不是 NULL，
      反而改变策略判定。
    """
    business_line = get_business_line()
    if business_line:
        # ★ 用 text() + 具名绑定，不用 exec_driver_sql：后者绕过 SQLAlchemy 编译，
        #   占位符得用驱动原生 paramstyle（asyncpg 不认 %s，直接当语法错误）。
        #   text() 走方言编译，占位符转换由 SQLAlchemy 负责。
        connection.execute(
            text("SELECT set_config('app.business_line', :line, true)"),
            {"line": business_line},
        )


AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,    # commit 后对象属性仍可读，否则序列化时会触发一次懒加载
    sync_session_class=ScopedSession,
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


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Agent 节点 / 后台任务用的事务边界：提交-回滚-关闭一体。

    ★ 之前分诊节点自己 `async with AsyncSessionLocal() as session: yield`，
      退出只 close 不 commit —— 写入全部被静默回滚（分诊结果从来没落过库）。
      所有非请求上下文的 DB 使用统一走这里。
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
