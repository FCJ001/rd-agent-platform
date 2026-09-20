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
    # ★ 默认必须关：debug=True 会把完整 traceback 回给客户端，
    #   且 SQLAlchemy echo 会把含业务数据的 SQL 打进日志。
    #   开发要开就在 .env 里显式 APP_DEBUG=true。
    APP_DEBUG: bool = False

    # ---------------- PostgreSQL ----------------
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "rdagent"
    DB_PASSWORD: str = "rdagent123"
    DB_NAME: str = "rd_agent"

    # ---------------- 数据作用域（业务线）----------------
    # ★ 逗号分隔。业务线是数据隔离的作用域键（现象/DTC/工单/知识都按它切分），
    #   项目会越来越多，不能再像以前那样把 ev/ia 写死在代码里 —— 新项目上线
    #   只需要改这个配置，不改代码、不发版。
    BUSINESS_LINES: str = "ev,ia"

    @property
    def business_lines(self) -> frozenset[str]:
        """解析后的业务线集合（去空白、去空项）。"""
        return frozenset(x.strip() for x in self.BUSINESS_LINES.split(",") if x.strip())

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

    # ---------------- 分诊收敛策略 ----------------
    # 数据驱动收敛阈值（scripts/fit_convergence.py 的拟合产物）；
    # 空串 = 用默认位置 eval/convergence_policy.json，文件不存在回退常数规则
    CONVERGENCE_POLICY_PATH: str = ""

    # ---------------- 安全 ----------------
    # 服务自签身份令牌的 HMAC 密钥（scripts/mint_token.py 签发，deps.py 校验）。
    # 生产必填，见 validate_production()
    AUTH_SECRET: str = ""
    # ALM 平台事件 webhook 的 HMAC 共享密钥（X-Webhook-Signature 验签）。
    # 生产必填；dev 不填则跳过验签并打警告
    WEBHOOK_SECRET: str = ""
    # CORS 白名单，逗号分隔；空串 + dev = 全放开（不带凭证），prod 必须显式配置
    CORS_ALLOW_ORIGINS: str = ""

    # ---------------- LangGraph checkpointer ----------------
    # checkpoint 键 TTL（分钟）。不配则 Redis 里 checkpoint 永久累积。
    # 默认 7 天 + 读时续期，覆盖「用户隔很久才回答追问」的场景
    CHECKPOINTER_TTL_MINUTES: int = 7 * 24 * 60

    # ---------------- 分诊进度存储 ----------------
    # triage_state:{thread_id} 的 TTL（秒）。
    # ★ 必须严格大于 CHECKPOINTER_TTL_MINUTES：两者是「同一段追问历史」的两个
    #   副本，进度先过期而挂起的 interrupt 还活着时，resume 重放会让工具读不到
    #   进度、误判成首轮重跑 —— 既白烧一整轮模型调用，又把 checkpointer 里
    #   记录的历史回答当成本轮新输入重新处理一遍，追问序列与 round 从此错位
    #   （实测轨迹见 tests/test_session_store_ttl.py）。
    #   两边都在读时续期（checkpointer 是 refresh_on_read；store 见 load()），
    #   所以只要 TTL 更大就能保证「interrupt 活着时进度一定还在」。
    #   代价只是多留一个 Redis key，取 checkpointer TTL + 1 天余量。
    TRIAGE_STATE_TTL_SECONDS: int = 8 * 24 * 3600

    # ---------------- 分诊会话锁 ----------------
    # Redis 分布式锁 TTL（秒）。★ 不是「用户思考时长」——interrupt() 抛异常时
    # 锁就已释放，真实持有时长 = 两次 interrupt 之间的 LLM 调用（秒~十几秒）。
    # 给 3 分钟是给 LLM 抖动留余量：过期会让并发重新进来（状态写坏），
    # 给太大则进程被 SIGKILL 后会话被锁死。取的是两侧都还能接受的中间值。
    TRIAGE_LOCK_TIMEOUT_SECONDS: int = 180

    # 抢不到锁时等待重试的次数与间隔（次 × 秒）。0 = 立即返回「忙」。
    # 同进程内两条消息（用户连点/多标签页）属于正常行为，稍等即可；
    # 跨进程的长任务则靠这个上限快速失败，不会无限等。
    TRIAGE_LOCK_RETRY_TIMES: int = 3
    TRIAGE_LOCK_RETRY_INTERVAL_SECONDS: float = 0.5

    # ---------------- chat 回合锁 ----------------
    # 保护 Supervisor checkpointer 里该会话的消息历史（chat_lock:{thread_id}）。
    # ★ 与会话锁是两个命名空间而非重复：两者覆盖的资源不同（控制面 vs 数据面），
    #   合并成一把会在「chat 回合内调用分诊工具」时自锁（见 lock.py 头注释）。
    # TTL 取得比会话锁小得多，因为持有时长 = 本回合全部工具调用（可能好几分钟）
    # —— 靠续租兜正常慢任务，TTL 只负责「进程崩溃后多久自动解锁」。
    # 取 90s 与闸门租约同量级，续租周期 30s 给 Redis 抖动留两次余量。
    CHAT_LOCK_TIMEOUT_SECONDS: int = 90

    # ---------------- 分诊全局并发闸门 ----------------
    # 会话锁只保证同一会话不并发，不限总量：1000 个会话同时发起诊断
    # 就是 1000 路并发的模型调用，把 LLM 端点和进程内存一起打挂。
    # 闸门限制全系统同时运行的分诊轮次（跨进程，Redis ZSET 信号量）；
    # 超出的先排队（有界、限时），排不下/等不到则拒绝。
    # ≤0 = 关闭闸门（本地开发与单测直通）。
    #
    # 定标 100 的算法（自建 8B~14B、单卡 H100、上下文 2K~3K）：
    #   ① KV 预算：(80G×0.9 − 16G 权重 − 2G 激活) = 54G
    #      ÷ 128KB/token ≈ 42 万 token ÷ 3K 上下文 ≈ 180 路
    #   ② decode 吞吐：约 4k token/s ÷ 22 token/s/用户 ≈ 180 路
    #      22 token/s/用户 是大厂实测口径（DeepSeek 公开数据 20~22）
    #   两条约束取小者 ≈ 180 路模型请求；闸门单位是「诊断轮次」，一轮
    #   1~2 次模型请求（extract/ask/parse/conclude），180 ÷ 1.5 ≈ 120，
    #   取 100 留约 40% 余量（上下文变长、并发抖动）。
    #   ★ 单副本并发做不大是 KV cache 的物理约束：自建引擎超额请求是
    #     排队不是报错，设高了只会把延迟堆到超时线上，不增加吞吐。
    # ★ 模型走公司共享端点时按「配额 ÷ 1.5」设，不要按本地机器能力
    #   推算——配额是别人给的，本地算得再准也超不过它。
    TRIAGE_GLOBAL_CONCURRENCY: int = 100

    # 单进程最多同时在排队的请求数。排队只是把「稍后重试」推迟一会儿，
    # 队列本身没有产出 —— 与其让第 500 个请求白等 60 秒，不如现在就
    # 告诉它重试，还省一个挂起的协程占着 FastAPI worker。
    TRIAGE_GATE_QUEUE_MAX: int = 200

    # 排队等空位的最长秒数，超时按「系统忙」拒绝，不无限等。
    TRIAGE_GATE_WAIT_TIMEOUT_SECONDS: float = 60.0

    # 空位轮询间隔（秒）。加 ±20% 抖动防多个等待者同拍惊群；
    # 非严格 FIFO：多进程下按轮询先后，统计意义上的先来先得。
    TRIAGE_GATE_POLL_INTERVAL_SECONDS: float = 0.5

    # 空位租约（秒）：持有者进程崩溃（SIGKILL/断电）没走正常释放时，
    # 空位最多这么久自动归还。持有期间每 1/3 租约时长自动续租，
    # 正常的长任务不受影响。
    TRIAGE_GATE_LEASE_SECONDS: float = 90.0

    # ---------------- 项目二（知识服务，步 7 才用到）----------------
    KNOWLEDGE_SVC_URL: str = "http://localhost:8001"

    # ---------------- ChatBI 独立服务（BI 查询，多数据源平台）----------------
    BI_SVC_URL: str = "http://localhost:8004"
    BI_PROJECT_ID: str = "rd_agent"  # 对应 rd-chatBI 的 bi_datasources.code

    # ---------------- Java ALM 平台（开发期 logger 占位）----------------
    PLATFORM_ALM_URL: str = "https://alm.internal"
    PLATFORM_ALM_API_URL: str = "https://alm.internal/api"

    # ---------------- 日志 ----------------
    LOG_LEVEL: str = "INFO"
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

    @property
    def cors_origins(self) -> list[str]:
        """逗号分隔的 CORS 白名单 → list。空串返回空列表。"""
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]

    @property
    def is_prod(self) -> bool:
        """归一化的生产判定。全项目统一用它，绝不要散写 APP_ENV == "prod" ——
        "Production"/"PROD " 这类写法一旦只被某处归一化、另一处没归一化，
        就会出现「X-User-Id 直通开着、启动校验却没跑」的错位绕过。"""
        return self.APP_ENV.strip().lower() in ("prod", "production")

    @property
    def is_dev(self) -> bool:
        """宽松模式白名单：只有显式声明的开发/测试环境才放行
        X-User-Id 直通、webhook 免验签这类不安全通道。
        ★ 未知取值（staging/uat/拼错的）一律按生产处理 —— fail-closed。"""
        return self.APP_ENV.strip().lower() in ("dev", "development", "local", "test")

    def validate_production(self) -> None:
        """APP_ENV=prod 时的启动期检查：不满足直接拒绝启动（fail-fast）。

        在 main.lifespan 里调用。开发期这些只是警告，不挡启动。
        """
        problems: list[str] = []
        if self.APP_DEBUG:
            problems.append("APP_DEBUG 必须为 false（debug 会向客户端回传 traceback）")
        if not self.AUTH_SECRET:
            problems.append("AUTH_SECRET 未配置（身份令牌无法验签）")
        elif len(self.AUTH_SECRET) < 32:
            problems.append("AUTH_SECRET 长度不足 32 字符（HMAC 密钥太短可被暴力破解，建议 openssl rand -hex 32）")
        if not self.WEBHOOK_SECRET:
            problems.append("WEBHOOK_SECRET 未配置（webhook 无法验签，任何人可伪造事件）")
        elif len(self.WEBHOOK_SECRET) < 32:
            problems.append("WEBHOOK_SECRET 长度不足 32 字符（建议 openssl rand -hex 32）")
        if not self.cors_origins:
            problems.append("CORS_ALLOW_ORIGINS 未配置（不能在生产放开任意跨域）")

        # 默认/示例口令检查：部署时忘改 .env 是最常见也最致命的配置事故，
        # 这些值同时也是 .env.example 里的占位符和 config 的 dev 缺省值
        _weak = {
            "DB_PASSWORD": ("rdagent123", "postgres", "change-me-postgres", ""),
            "NEO4J_PASSWORD": ("rdagent123", "neo4j", "change-me-neo4j", ""),
            "MINIO_ACCESS_KEY": ("minioadmin", "change-me-minio", ""),
            "MINIO_SECRET_KEY": ("minioadmin", "minioadmin-secret", "change-me-minio-secret", ""),
        }
        for field, weak_values in _weak.items():
            if getattr(self, field) in weak_values:
                problems.append(f"{field} 仍是默认/弱口令，必须换成强随机值")
        if self.REDIS_PASSWORD and self.REDIS_PASSWORD in ("change-me-redis", "redis", "123456"):
            problems.append("REDIS_PASSWORD 仍是默认/弱口令，必须换成强随机值")

        # 分诊进度必须晚于 checkpointer 过期：反了会让「隔了很久才回答追问」
        # 的会话走成首轮重跑 + 历史回答被当新输入重新处理（见字段注释）。
        # 这类错配不会立刻报错，只在特定时序下悄悄答错，所以放到启动期挡。
        if self.TRIAGE_STATE_TTL_SECONDS <= self.CHECKPOINTER_TTL_MINUTES * 60:
            problems.append(
                f"TRIAGE_STATE_TTL_SECONDS({self.TRIAGE_STATE_TTL_SECONDS}) 必须大于 "
                f"CHECKPOINTER_TTL_MINUTES×60({self.CHECKPOINTER_TTL_MINUTES * 60})"
                "（分诊进度先过期会导致 resume 误判为首轮，追问序列错位）"
            )

        if problems:
            raise RuntimeError(
                "生产环境配置校验失败：\n  - " + "\n  - ".join(problems)
            )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # .env 里可能有本文件没声明的键（比如只给 docker 用的），忽略而不是报错
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
