"""变更影响分析 Agent。封装 analyze_impact 为 Worker 工具。"""

from functools import lru_cache

from src.agents.impact.review import analyze_impact


class ImpactAgent:
    """变更影响分析 Agent。4 路并行检查依赖/基线/重复/范围。"""

    async def analyze(self, change_description: str) -> str:
        """执行变更影响分析，返回 Markdown 报告。"""
        return await analyze_impact(change_description)


@lru_cache(maxsize=1)
def get_impact_agent() -> ImpactAgent:
    return ImpactAgent()
