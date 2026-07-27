"""报告解读 Agent。代码 judge() 判阈值，LLM 只做人话翻译。"""

from functools import lru_cache

from src.agents.report.parser import analyze_report


class ReportAgent:
    """报告/日志解读 Agent。judge() 代码判阈值，LLM 翻译成人话。"""

    async def analyze(self, report_text: str, report_type: str = "DTC扫描") -> str:
        """解析并解读一份报告，返回结构化解读。"""
        return await analyze_report(report_text, report_type)


@lru_cache(maxsize=1)
def get_report_agent() -> ReportAgent:
    return ReportAgent()
