# ============================================================
# rd-agent-platform 应用镜像
#
# 构建：docker build -t rd-agent-platform .
# 运行：docker run --env-file .env -p 127.0.0.1:8000:8000 rd-agent-platform
# ============================================================

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# psycopg2-binary 无需编译依赖；curl 保留给镜像内健康检查
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖层独立缓存：requirements 不变时跳过安装
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/

# 非 root 运行
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 多 worker 由 UVICORN_WORKERS 控制（默认 2）；prometheus 张量暴露待接入
# ★ 生产绝不带 --reload
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
