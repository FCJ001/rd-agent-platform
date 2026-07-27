"""长期记忆工具：save_memory / search_memory，基于 MilvusStore。"""

import time

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime


def _get_user_id(runtime: ToolRuntime) -> str | None:
    """从 thread_id 中提取用户 ID。thread_id 格式: {user_id}:{session_id}"""
    if runtime.config.get("configurable"):
        thread_id = runtime.config.get("configurable").get("thread_id", "")
        if ":" in thread_id:
            return thread_id.split(":")[0]
    return None


@tool
async def save_memory(content: str, runtime: ToolRuntime) -> str:
    """将重要信息保存到长期记忆中。
    适用场景：诊断结论、车辆信息、维修历史等需要跨会话记住的内容。

    Args:
        content: 要记住的内容，用一句话描述
    """
    user_id = _get_user_id(runtime)
    if not user_id:
        return "无法获取用户ID。"

    key = f"memory_{int(time.time())}"
    await runtime.store.aput(
        namespace=("users", user_id, "memories"),
        key=key,
        value={"content": content, "timestamp": time.time()},
    )
    return f"已记住：{content}"


@tool
async def search_memory(query: str, runtime: ToolRuntime) -> str:
    """从长期记忆中检索与问题相关的历史信息。
    适用场景：诊断前检索同用户/车辆的历史故障记录。

    Args:
        query: 检索关键词或问题
    """
    user_id = _get_user_id(runtime)
    if not user_id:
        return "无法获取用户ID。"

    results = await runtime.store.asearch(
        ("users", user_id, "memories"),
        query=query,
        limit=5,
    )
    if not results:
        return "没有找到相关记忆。"

    memories = [f"- {item.value['content']}" for item in results]
    return "相关历史记忆：\n" + "\n".join(memories)
