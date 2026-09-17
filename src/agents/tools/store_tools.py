"""长期记忆工具：save_memory / search_memory，基于 MilvusStore。

★ 语义边界（别越界，越界会让知识复用失效）：
  这里存**关于人和车的事实** —— 车型、换过什么件、使用环境、用户偏好，
  这类信息「属于这个人」，跨会话记住是有意义的。
  **诊断结论不属于这里** —— 根因、现象、置信度这类知识必须能被同项目的
  其他人复用，走 search_past_diagnoses（按业务线共享）。
  把结论写进记忆 = 把它锁在个人名下，别人永远看不到。
"""

import time

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.core.logger import logger


def _get_user_id(runtime: ToolRuntime) -> str | None:
    """从 thread_id 中提取用户 ID。thread_id 格式: {user_id}:{session_id}"""
    if runtime.config.get("configurable"):
        thread_id = runtime.config.get("configurable").get("thread_id", "")
        if ":" in thread_id:
            return thread_id.split(":")[0]
    return None


@tool
async def save_memory(content: str, runtime: ToolRuntime) -> str:
    """将**关于人和车的事实**保存到长期记忆，跨会话记住。
    适用场景：车辆信息（车型/配置）、维修与更换记录、使用环境、用户偏好。
    ★ 不要用它保存诊断结论（根因/现象/置信度）—— 结论由系统自动沉淀到
      知识库，并可通过 search_past_diagnoses 在同项目内复用；写进记忆
      只会让它变成只有你自己能看到的孤岛。

    Args:
        content: 要记住的内容，用一句话描述
    """
    user_id = _get_user_id(runtime)
    if not user_id:
        return "无法获取用户ID。"

    key = f"memory_{int(time.time())}"
    logger.info(f"[MEM] save_memory user={user_id} content={content[:80]}")
    await runtime.store.aput(
        namespace=("users", user_id, "memories"),
        key=key,
        value={"content": content, "timestamp": time.time()},
    )
    return f"已记住：{content}"


@tool
async def search_memory(query: str, runtime: ToolRuntime) -> str:
    """从长期记忆中检索**关于人和车的事实**。
    适用场景：诊断前检索同一用户/车辆的基本信息与历史处置记录（不是结论）。
    ★ 要查「以前诊断出什么根因」请用 search_past_diagnoses —— 那条路
      按业务线共享，能拿到同项目其他人的结论。

    Args:
        query: 检索关键词或问题
    """
    user_id = _get_user_id(runtime)
    if not user_id:
        return "无法获取用户ID。"

    logger.info(f"[MEM] search_memory user={user_id} query={query[:80]}")
    results = await runtime.store.asearch(
        ("users", user_id, "memories"),
        query=query,
        limit=5,
    )
    logger.info(f"[MEM] search_memory results={len(results)}")
    if not results:
        return "没有找到相关记忆。"

    memories = [f"- {item.value['content']}" for item in results]
    return "相关历史记忆：\n" + "\n".join(memories)
