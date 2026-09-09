# ============================================================
# 应用入口
#
# 启动：uvicorn src.main:app --reload --port 8000
# 文档：http://localhost:8000/docs
# ============================================================

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.routers import chat, feedback, issues, triage, webhook
from src.core.base_schema import ResponseSchema
from src.core.config import get_settings
from src.core.exceptions import register_exception_handlers
from src.core.logger import logger, setup_logger
from src.middlewares.logging import TraceLoggingMiddleware

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    logger.info(f"{settings.APP_NAME} 启动 env={settings.APP_ENV}")
    yield
    logger.info(f"{settings.APP_NAME} 关闭")


app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.APP_DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


@app.get("/health", response_model=ResponseSchema[dict])
async def health():
    return ResponseSchema(data={"app": settings.APP_NAME, "env": settings.APP_ENV})
