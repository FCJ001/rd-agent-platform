# ============================================================
# 全局配置
#
# 所有外部依赖的连接信息、模型密钥统一从这里读，来源是 .env。
# ★ 绝不在业务代码里硬编码密钥 —— 医疗版 scripts/init_public_datasets.py:46 犯过这个错
#
# 用法：
#   from src.core.config import get_settings
#   settings = get_settings()        # lru_cache，全进程只解析一次 .env
# ============================================================

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---------------- 应用 ----------------
    APP_NAME: str = "rd-agent-platform"
    APP_ENV: str = "dev"
    APP_DEBUG: bool = True

    # ---------------- PostgreSQL ----------------
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "rdagent"
    DB_PASSWORD: str = "rdagent123"
    DB_NAME: str = "rd_agent"

    # ---------------- Redis Stack ----------------
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # ---------------- MinIO ----------------
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "alm-reports"
    MINIO_SECURE: bool = False

    # ---------------- Milvus ----------------
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # ---------------- Neo4j ----------------
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "rdagent123"

    # ---------------- 模型 ----------------
    DASHSCOPE_API_KEY: str = ""
    BASE_URL_CHAT: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    CHAT_MODEL: str = "qwen-max"
    EMBEDDING_MODEL: str = "text-embedding-v3"

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # ---------------- 项目二（知识服务，步 7 才用到）----------------
    KNOWLEDGE_SVC_URL: str = "http://localhost:8001"

    # ---------------- Java ALM 平台（开发期 logger 占位）----------------
    PLATFORM_ALM_URL: str = "https://alm.internal"
    PLATFORM_ALM_API_URL: str = "https://alm.internal/api"

    # ---------------- 日志 ----------------
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"

    @property
    def DATABASE_URL(self) -> str:
        """业务代码用异步驱动 asyncpg；种子脚本另走 psycopg2 同步连接"""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # .env 里可能有本文件没声明的键（比如只给 docker 用的），忽略而不是报错
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
