"""报告解析器：代码抽取 + judge 判定，LLM 只做人话翻译。"""

import json

from langchain_openai import ChatOpenAI

from src.agents.report.standards import (
    judge, find_standard, Severity,
    METRIC_STANDARDS,
)
from src.agents.report.prompts import REPORT_PARSE_PROMPT, INTERPRET_PROMPT
from src.core.config import get_settings

settings = get_settings()


def _get_llm_json() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.3,
        timeout=30,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _get_llm_chat() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.5,
        timeout=30,
    )


def _parse_llm_json(response) -> dict:
    content = response.content.strip()
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    return json.loads(content)


async def analyze_report(report_text: str, report_type: str = "DTC扫描") -> str:
    """
    报告解读主流程：
    1. LLM 抽取指标三元组
    2. judge() 逐项代码判定
    3. LLM 汇总生成人话解读
    """
    # ── Step 1: LLM 抽取指标 (JSON mode) ──
    llm_json = _get_llm_json()
    metric_keys = [s.metric_name for s in METRIC_STANDARDS]
    parse_prompt = REPORT_PARSE_PROMPT.format(
        report_text=report_text,
        report_type=report_type,
        metric_keys="、".join(metric_keys),
    )
    response = await llm_json.ainvoke(parse_prompt)
    try:
        parsed = _parse_llm_json(response)
    except Exception:
        return "无法解析报告内容，请确认报告格式。"

    metrics = parsed.get("metrics", [])
    report_summary = parsed.get("summary", "未知")

    # ── Step 2: judge() 逐项代码判定 ──
    judge_lines = []
    abnormal_count = 0
    related_causes: set[str] = set()

    for m in metrics:
        key = m.get("key", "")
        value = float(m.get("value", 0))
        unit = m.get("unit", "")

        is_abnormal, severity = judge(key, value)
        standard = find_standard(key)

        if is_abnormal:
            abnormal_count += 1
            if standard and standard.related_cause_codes:
                related_causes.update(standard.related_cause_codes)

        label = "NORMAL" if severity == Severity.NORMAL else severity.value.upper()
        hint = standard.interpretation_hint if standard else ""
        source = standard.standard_source if standard else ""
        judge_lines.append(
            f"- [{label}] {key} = {value}{unit} "
            f"({'正常' if not is_abnormal else '异常'})"
            f"{' | ' + hint if hint and is_abnormal else ''}"
            f"{' | 依据: ' + source if source else ''}"
        )

    judge_results = "\n".join(judge_lines)

    # ── Step 3: LLM 汇总人话解读 (normal mode, no JSON) ──
    llm_chat = _get_llm_chat()
    interpret_prompt = INTERPRET_PROMPT.format(
        judge_results=judge_results,
        report_summary=report_summary,
        related_causes="、".join(related_causes) if related_causes else "无",
    )
    response = await llm_chat.ainvoke(interpret_prompt)
    interpretation = response.content

    # Prepend judge summary
    header = f"## 报告类型：{report_type}\n异常指标：{abnormal_count}/{len(metrics)}\n"
    return header + "\n" + interpretation
