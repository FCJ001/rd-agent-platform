# ============================================================
# 诊断结论 → Milvus 长期记忆
# 从 chat.py 迁出：B2 重构后分诊收敛发生在 call_triage_agent 工具内，
# 记忆保存跟着搬进 triage 包，chat 路由不再接触分诊内部。
# ============================================================

import time

from src.core.logger import logger

_milvus_store = None


def _get_milvus_store():
    global _milvus_store
    if _milvus_store is None:
        from langchain_community.embeddings import DashScopeEmbeddings

        from src.core.config import get_settings
        from src.infra.milvus_client import get_milvus_client_alias
        from src.infra.milvus_store import MilvusStore

        s = get_settings()
        alias = get_milvus_client_alias()
        embedding_model = DashScopeEmbeddings(
            model="text-embedding-v3", dashscope_api_key=s.DASHSCOPE_API_KEY
        )
        _milvus_store = MilvusStore(alias=alias, embeddings=embedding_model, dims=1024)
    return _milvus_store


async def save_diagnosis_memory(state, thread_id: str) -> None:
    """诊断收敛后将结论写入长期记忆，供 search_memory 检索。"""
    if not state.candidate_causes:
        return

    top = state.candidate_causes[0]
    user_id = thread_id.split(":")[0]

    phenomena_str = "、".join(state.confirmed_phenomena) if state.confirmed_phenomena else "未知"
    dtc_str = "、".join(state.dtc_codes) if state.dtc_codes else "无"

    content = (
        f"诊断结论：{top.name}({top.code}) 置信度{top.confidence:.0%}，"
        f"现象={phenomena_str}，DTC={dtc_str}。{state.diagnostic_summary}"
    )

    store = _get_milvus_store()
    key = f"diagnosis_{int(time.time())}"
    await store.aput(
        namespace=("users", user_id, "memories"),
        key=key,
        value={"content": content, "timestamp": time.time()},
    )
    logger.info(f"[TRIAGE-MEMORY] saved diagnosis memory for user={user_id}: {content[:120]}...")
