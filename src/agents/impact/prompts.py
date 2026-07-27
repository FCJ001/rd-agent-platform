"""变更影响分析 Prompt 模板。"""

PARSE_CHANGE_PROMPT = """你是汽车研发领域的配置管理专家。用户描述了一个变更请求，请解析出结构化信息。

变更描述：{change_description}

请以 JSON 格式输出：
{{
  "config_items": ["受影响的配置项名称列表"],
  "scope": "变更范围描述",
  "target_baseline": "目标基线名称（如有）",
  "change_type": "软件升级/硬件变更/参数标定/其他",
  "business_line": "ev/ia",
  "risk_signals": ["潜在风险信号"]
}}

只输出 JSON，不要解释。"""


IMPACT_REPORT_PROMPT = """你是汽车研发领域的变更影响分析专家。请根据以下分析结果生成影响评估报告。

## 分析结果
- 变更配置项：{config_items}
- 基线冲突：{baseline_conflicts}
- 依赖冲突：{dependency_conflicts}
- 重复变更：{duplicate_changes}
- 影响范围：{scope_summary}

## 要求
1. 评估风险等级（低/中/高/严重）
2. 列出受影响的需求和配置项
3. 给出建议（是否可合入、需要哪些验证）
4. 语气专业简洁

直接输出评估报告。"""
