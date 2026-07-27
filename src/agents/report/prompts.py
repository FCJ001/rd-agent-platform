"""报告解读 Prompt 模板。"""

REPORT_PARSE_PROMPT = """你是汽车研发领域的数据分析专家。请从以下报告内容中抽取关键指标三元组。

报告内容：{report_text}
报告类型：{report_type}

可识别的指标关键词：{metric_keys}

请以 JSON 格式输出：
{{
  "metrics": [
    {{"key": "metric_name", "value": 数值, "unit": "单位"}}
  ],
  "summary": "报告一句话摘要"
}}

要求：
- 只抽取数值型指标，忽略文字描述
- key 必须从关键词列表中选择最匹配的
- 数值必须是 float 类型
- 只输出 JSON，不要解释。"""


INTERPRET_PROMPT = """你是汽车研发领域的质量分析专家。请根据以下指标判定结果生成报告解读。

## 指标判定结果
{judge_results}

## 报告摘要
{report_summary}

## 相关根因
{related_causes}

## 要求
1. 异常指标用通俗语言解释（是什么、严重程度、可能后果）
2. 正常指标简要带过
3. 给出处置建议优先级（紧急/建议/观察）
4. 标注判定依据（国标/厂标）
5. 语气专业简洁

直接输出解读报告。"""
