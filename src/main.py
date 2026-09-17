# ============================================================
# 应用入口
#
# 启动（开发）：uvicorn src.main:app --reload --port 8000
# 启动（生产）：见 Dockerfile CMD —— 多 worker + no reload
# 文档：http://localhost:8000/docs（prod 自动关闭）
# ============================================================

import asyncio
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.api.routers import chat, feedback, issues, triage, webhook
from src.core.base_schema import ResponseSchema
from src.core.config import get_settings
from src.core.exceptions import register_exception_handlers
from src.core.logger import logger, setup_logger
from src.middlewares.logging import TraceLoggingMiddleware

settings = get_settings()

IS_PROD = settings.is_prod


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    logger.info(f"{settings.APP_NAME} 启动 env={settings.APP_ENV}")

    # 生产配置 fail-fast：密钥缺失 / debug 开着 / CORS 全开 → 拒绝启动
    if IS_PROD:
        settings.validate_production()

    # 预热连接（失败不挡启动 —— 健康检查会如实报告 degraded）
    try:
        from src.infra.neo4j_client import get_neo4j_driver
        await asyncio.to_thread(get_neo4j_driver)
    except Exception as e:
        logger.warning(f"Neo4j 预热失败（degraded 启动）: {e}")

    yield

    # shutdown：显式归还外部连接（之前什么都不关，靠进程退出兜底）
    try:
        from src.infra.neo4j_client import close_neo4j_driver
        from src.infra.milvus_client import close_milvus_client
        from src.infra.redis_cache import _redis_client, _checkpointer_client

        await asyncio.to_thread(close_neo4j_driver)
        await asyncio.to_thread(close_milvus_client)
        await _redis_client.aclose()
        await _checkpointer_client.aclose()
        from src.infra.db import engine
        await engine.dispose()
    except Exception as e:
        logger.warning(f"shutdown 清理异常（忽略）: {e}")

    logger.info(f"{settings.APP_NAME} 关闭")


app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.APP_DEBUG,
    lifespan=lifespan,
    # 生产不暴露交互式 API 文档（侦察面）
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
)

# CORS：白名单来自配置。prod 必须显式配置（validate_production 把关）；
# 通配时禁止携带凭证（浏览器规范也禁止，显式关掉避免误导）
_allow_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins if _allow_origins else ["*"],
    allow_credentials=bool(_allow_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TraceLoggingMiddleware)
register_exception_handlers(app)

app.include_router(chat.router)
app.include_router(issues.router)
app.include_router(triage.router)
app.include_router(feedback.router)
app.include_router(webhook.router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    """依赖探测的健康检查（给 K8s/负载均衡用，不是之前的假检查）。

    critical（PG/Redis）挂 → 503；dependency（Neo4j/Milvus）挂 → 200 + degraded，
    因为它们只影响部分功能，是否摘流量交给编排层按 degraded 字段决策。
    """
    async def _check(probe) -> str:
        try:
            await asyncio.wait_for(probe(), timeout=3)
            return "ok"
        except Exception as e:
            logger.warning(f"[HEALTH] 依赖检查失败: {e}")
            return "fail"

    from sqlalchemy import text as sa_text

    from src.infra.db import AsyncSessionLocal
    from src.infra.redis_cache import get_redis_client

    async def _db():
        async with AsyncSessionLocal() as db:
            await db.execute(sa_text("SELECT 1"))

    async def _redis():
        client = await get_redis_client()
        await client.ping()

    async def _neo4j():
        from src.infra.neo4j_client import get_neo4j_driver
        driver = await asyncio.to_thread(get_neo4j_driver)
        await asyncio.to_thread(driver.verify_connectivity)

    async def _milvus():
        from src.infra.milvus_client import check_milvus_health
        await asyncio.to_thread(check_milvus_health)

    db_status, redis_status, neo4j_status, milvus_status = await asyncio.gather(
        _check(_db),
        _check(_redis),
        _check(_neo4j),
        _check(_milvus),
    )

    critical_ok = db_status == "ok" and redis_status == "ok"
    body = {
        "status": "ok" if critical_ok else "unavailable",
        "dependencies": {
            "postgres": db_status,
            "redis": redis_status,
            "neo4j": neo4j_status,
            "milvus": milvus_status,
        },
    }
    if not critical_ok:
        return JSONResponse(status_code=503, content=body)
    return JSONResponse(status_code=200, content=body)
