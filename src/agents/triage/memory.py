# ============================================================
# ⚠️ 已废弃：诊断结论不再写长期记忆
#
# 废弃原因：记忆集合的 namespace 是 users/{uid}/memories，结论写进去只有
#   本人能召回 —— 同一个故障两个工程师各诊断一次，互相看不到对方的结论。
#   知识复用需要「项目内共享」，那是记忆通道给不了的语义。
#
# 现状：结论落在 ai_triage_results（含 user_id / business_line），由
#   src/agents/triage/history.py 的 query_past_diagnoses 按
#     · mine —— 我本人诊断过的
#     · line —— 本业务线上所有人诊断过的
#   两个维度复用，并通过 search_past_diagnoses 工具暴露给 Supervisor。
#
# ★ 保留本文件的函数只为可追溯，**没有调用点**，不要再接回链路。
#   长期记忆通道（save_memory / search_memory）现在只承载
#   「关于人和车的事实」，语义见 src/agents/tools/store_tools.py。
# ============================================================

import asyncio
import time

from src.core.logger import logger

_milvus_store = None


def _build_milvus_store():
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
    """【已废弃，无调用点】诊断结论写长期记忆的旧实现 —— 见文件头注释。"""
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

    store = await asyncio.to_thread(_build_milvus_store)
    key = f"diagnosis_{int(time.time())}"
    await store.aput(
        namespace=("users", user_id, "memories"),
        key=key,
        value={"content": content, "timestamp": time.time()},
    )
    logger.info(f"[TRIAGE-MEMORY] saved diagnosis memory for user={user_id}: {content[:120]}...")
