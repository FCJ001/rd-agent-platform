"""历史诊断复用工具 —— 给 Supervisor 调。

与长期记忆（store_tools 的 save_memory / search_memory）的分工：
  · 记忆：关于「人和车」的事实（车型、换过什么件、偏好）→ 归人，跨会话
  · 本工具：关于「故障怎么定位的」结论 → 归项目，跨用户共享
两者混用会导致结论被锁在个人名下，同一故障各诊断一次、互相看不到。
"""

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.agents.triage.history import (
    SCOPE_LINE,
    SCOPE_MINE,
    format_past_diagnoses,
    query_past_diagnoses,
)
from src.core.deps import UserContext
from src.core.logger import logger


@tool
async def search_past_diagnoses(
    phenomena: str,
    scope: str = SCOPE_MINE,
    runtime: ToolRuntime[UserContext] = None,
) -> str:
    """检索历史诊断结论（以前诊断过的同类问题，含根因与置信度）。
    适用场景：用户问「上次那个问题怎么解决的」「以前有没有遇到过」，
    或需要在诊断前参考同项目已有的结论避免重复劳动。

    Args:
        phenomena: 现象关键词，逗号分隔（如"中控屏黑屏,无法唤醒"）。
            ★ 必须是现象名/关键词，不要把用户整句原话传进来 ——
            中文没有词间空格，整句匹配不到任何记录
        scope: 检索范围。mine=我本人诊断过的（默认）；line=本业务线上
            所有人诊断过的（仅内部角色可用，客户会被自动降级为 mine）
    """
    user_id = runtime.context.user_id if runtime else ""
    role = runtime.context.role if runtime else "customer"
    business_line = (runtime.context.business_line or "") if runtime else ""

    logger.info(
        f"[HISTORY] search_past_diagnoses scope={scope} role={role} "
        f"phenomena={phenomena[:80]}"
    )

    result = await query_past_diagnoses(
        keywords=phenomena,
        role=role,
        user_id=user_id,
        business_line=business_line,
        scope=scope,
    )
    return format_past_diagnoses(result)


HISTORY_TOOLS = [search_past_diagnoses]
