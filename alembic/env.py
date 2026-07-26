from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.core.base_model import Base
from src.core.config import get_settings

# ============================================================
# ★★★ 关键：必须显式导入【所有】model 模块
#
# Alembic 靠 Base.metadata 收集表定义，而 metadata 是在 model 模块
# 被 import 时才注册的。漏一个模块，autogenerate 就少建对应的表，
# 而且【不会报错】—— 只是静默少表。
#
# 项目一有两个 model 模块：
#   src/modules/alm/model.py   11 张业务表
#   src/modules/user/model.py  users 1 张（seed_users.py 依赖）
# 新增模块时【必须】在这里追加一行
#
# import * 加 noqa 的理由：这是「有副作用的导入」，目的是执行模块顶层
# 代码让 model 注册，不是为了用里面的名字。
# ============================================================
from src.modules.alm.model import *   # noqa: F401,F403,E402
from src.modules.user.model import *  # noqa: F401,F403,E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ★ 从 Settings 注入连接串，覆盖 alembic.ini 里留空的 sqlalchemy.url
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 不连库（alembic upgrade head --sql）"""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,            # ★ 检测字段类型变更
        compare_server_default=True,  # ★ 检测默认值变更（默认关闭）
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """在线模式：异步引擎 + run_sync 桥接到同步的 Alembic API"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,   # ★ 迁移是一次性操作，不要连接池
    )
    async with connectable.connect() as connection:
        # ★ Alembic 的 API 是同步的，run_sync 把它跑在 greenlet 里
        #   直接调会报 MissingGreenlet
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
