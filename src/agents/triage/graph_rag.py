"""GraphRAG — NL2Cypher 图查询（参考天宫医疗版 graph_rag.py）。"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from src.core.logger import logger

MAX_CYPHER_RETRIES = 2

# ════════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════════

ENTITY_EXTRACT_PROMPT = """从用户问题中提取汽车研发领域实体。

用户问题：{question}

以 JSON 格式输出：
{{
  "phenomena": ["现象名1"],
  "root_causes": ["根因名1"],
  "config_items": ["配置项名1"],
  "dtc_codes": ["DTC码1"],
  "domains": ["责任域名1"],
  "baselines": ["基线名1"],
  "requirements": ["需求编号1"]
}}

没有的类别填空列表。只输出 JSON，不要解释。"""


NL2CYPHER_PROMPT = """你是 Neo4j Cypher 查询专家。根据用户问题和图谱 Schema 生成 Cypher 查询。

## 图谱 Schema

节点类型：
- Phenomenon（现象）：属性 name, code, business_line
- RootCause（根因）：属性 code, name, domain, business_line, fix_way, fix_duration, description
- DTC（故障码）：属性 code, system, description, business_line
- OwnerDomain（责任域）：属性 name, business_line
- ConfigItem（配置项）：属性 name, ci_no, module, supplier, part_number, sw_version, is_safety_related
- Requirement（需求）：属性 req_no, title, business_line, status
- ChangeRequest（变更请求）：属性 cr_no, title, reason, status
- Baseline（基线）：属性 name, baseline_no, business_line, is_frozen

关系类型：
- (Phenomenon)-[:INDICATES {{weight, is_core}}]->(RootCause)    现象指示根因
- (RootCause)-[:BELONGS_TO]->(OwnerDomain)                      根因归属责任域
- (DTC)-[:POINTS_TO]->(RootCause)                               DTC 指向根因
- (RootCause)-[:LOCATED_IN]->(ConfigItem)                       根因定位到配置项
- (RootCause)-[:CO_OCCURS_WITH]->(RootCause)                    伴随根因
- (ConfigItem)-[:DEPENDS_ON]->(ConfigItem)                      配置项依赖
- (Requirement)-[:IMPLEMENTED_BY]->(ConfigItem)                 需求由配置项实现
- (ChangeRequest)-[:AFFECTED_BY]->(ConfigItem)                  变更影响配置项
- (ChangeRequest)-[:TARGETS]->(Baseline)                        变更目标基线

## 规则
1. 只使用上述 Schema 中存在的节点和关系类型
2. 查询深度最多 3 跳
3. 返回结果用 LIMIT 限制，最多 20 条
4. 返回有意义的字段（name、属性），不要只返回节点 ID

用户问题：{question}
已提取的实体：{entities}

只输出 Cypher 查询语句，不要解释。"""


# ════════════════════════════════════════════════════════════════════════
# Core functions
# ════════════════════════════════════════════════════════════════════════

async def _extract_entities(question: str, llm: BaseChatModel) -> dict:
    prompt = ENTITY_EXTRACT_PROMPT.format(question=question)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"[GraphRAG] 实体提取失败: {e}")
        return {
            "phenomena": [], "root_causes": [], "config_items": [],
            "dtc_codes": [], "domains": [], "baselines": [], "requirements": [],
        }


async def _generate_cypher(
    question: str, entities: dict, llm: BaseChatModel, error_hint: str = "",
) -> str:
    extra = ""
    if error_hint:
        extra = f"\n\n上一次生成的 Cypher 执行报错：{error_hint}\n请修正后重新生成。"
    prompt = NL2CYPHER_PROMPT.format(
        question=question,
        entities=json.dumps(entities, ensure_ascii=False),
    ) + extra
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    cypher = response.content.strip()
    if "```" in cypher:
        cypher = cypher.split("```")[1].lstrip("cypher").strip()
    return cypher


async def search_graph_raw(
    question: str,
    neo4j_driver,
    llm: BaseChatModel,
) -> list[dict]:
    """GraphRAG 检索，返回原始图谱查询结果（不经过 LLM 生成）。

    NL2Cypher + 错误反馈重试（最多 MAX_CYPHER_RETRIES 次）。
    """
    entities = await _extract_entities(question, llm)
    logger.info(f"[GraphRAG] 实体提取: {entities}")

    error_hint = ""
    for attempt in range(MAX_CYPHER_RETRIES + 1):
        cypher = await _generate_cypher(question, entities, llm, error_hint)
        logger.info(f"[GraphRAG] Cypher (attempt {attempt + 1}): {cypher}")
        try:
            async with neo4j_driver.session() as session:
                result = await session.run(cypher)
                records = await result.data()
                return records[:20]
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"[GraphRAG] Cypher 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_CYPHER_RETRIES:
                return []
    return []
