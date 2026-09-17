"""GraphRAG — NL2Cypher 图查询（参考天宫医疗版 graph_rag.py）。

★ 注入面与执行约束：
  - Cypher 由 LLM 生成、question 可能携带用户注入内容，因此执行前做
    写操作关键字拦截（只读保证），执行走 asyncio.to_thread（同步驱动）。
  - 调用方传入的是同步 neo4j driver —— 之前直接 async with/await 它，
    必然抛异常，导致 GraphRAG 检索静默失效。
"""

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from src.core.logger import logger

MAX_CYPHER_RETRIES = 2

# LLM 生成的 Cypher 只允许读：命中写关键字直接拒绝（防 prompt injection 写删图谱）
_CYPHER_WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|CALL)\b",
    re.IGNORECASE,
)

# 业务线作用域参数名。生成的 Cypher 必须引用它，由调用方绑定实际值 ——
# 不让 LLM 自己决定作用域（question 可携带注入内容，写成字面量就能越线取数）
SCOPE_PARAM = "$business_line"

# 越线的字面量写法：把 business_line 直接和引号里的值比较，例如
#   business_line = 'ia'   /   business_line IN ['ev', 'ia']
# ★ 只在「比较位置」判定，不要泛匹配任意引号短串 —— 否则 Cypher 里的
#   映射键（{"name": ...}）会被误判成业务线字面量，导致合规查询被拒
_LINE_LITERAL_RE = re.compile(
    r"business_line\s*(?:=\s*['\"]|IN\s*\[?\s*['\"])",
    re.IGNORECASE,
)

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
- (RootCause)-[:INDICATES {{weight, is_core}}]->(Phenomenon)    根因指示现象（方向与 init_neo4j.py 建边一致，勿反向）
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
5. ★ 必须限定业务线：凡涉及 Phenomenon / RootCause / DTC / OwnerDomain /
   Requirement / Baseline 的 MATCH，都要在 WHERE 里加
   `节点.business_line = $business_line`。
   只写参数名 $business_line，**绝对不要写具体的业务线字面量**
   （如 'ev' / 'ia'）—— 作用域由系统绑定

用户问题：{question}
已提取的实体：{entities}

只输出 Cypher 查询语句，不要解释。"""


def _scope_violation(cypher: str) -> str | None:
    """检查生成的 Cypher 是否满足业务线约束，返回违规原因（None = 通过）。

    只做静态检查，不做 Cypher 解析：这里防的是「越线取数」这一件事，
    而它只有两种形态 —— 压根没写作用域、或者把作用域写成了字面量。

    先判字面量再判缺失：写死的查询两者都命中，此时更该告诉 LLM
    「不要写死」而不是「你没写过滤条件」—— 提示会作为 error_hint 回喂。
    """
    if _LINE_LITERAL_RE.search(cypher):
        return (
            f"Cypher 把业务线写成了字面量：必须改用 {SCOPE_PARAM} 参数"
            "（作用域由系统绑定，不能让查询自己决定）"
        )
    if SCOPE_PARAM not in cypher:
        return (
            f"Cypher 缺少业务线过滤：必须使用 {SCOPE_PARAM} 参数，"
            "例如 WHERE n.business_line = $business_line"
        )
    return None


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


def _run_cypher_sync(neo4j_driver, cypher: str, params: dict | None = None) -> list[dict]:
    """同步 driver 上执行只读 Cypher（在线程池里跑）。

    ★ execute_read 把事务钉死在 READ 访问模式 —— 写操作关键字黑名单
      是第一道闸（拦截意图明显的注入），这里是驱动层兜底：即使黑名单
      被某种形态绕过，READ 事务里执行写操作会被 Neo4j 直接拒绝。

    params 为绑定参数（含业务线作用域）。作用域值不走 Cypher 文本，
    避免注入内容影响过滤条件本身。
    """
    def _tx(tx) -> list[dict]:
        return [r.data() for r in tx.run(cypher, params or {})]

    with neo4j_driver.session() as session:
        return session.execute_read(_tx)


async def search_graph_raw(
    question: str,
    neo4j_driver,
    llm: BaseChatModel,
    business_line: str = "",
) -> list[dict]:
    """GraphRAG 检索，返回原始图谱查询结果（不经过 LLM 生成）。

    NL2Cypher + 错误反馈重试（最多 MAX_CYPHER_RETRIES 次）。

    business_line 非空时强制作用域：生成的 Cypher 必须引用 $business_line
    参数（值由本函数绑定），否则拒绝执行并重试；重试耗尽返回空，
    绝不退化成"不带作用域照样跑"。留空 = 调用方拿不到作用域（打警告）。
    """
    entities = await _extract_entities(question, llm)
    logger.info(f"[GraphRAG] 实体提取: {entities}")

    if not business_line:
        logger.warning("[GraphRAG] business_line 为空，图谱检索未限定业务线")

    params = {"business_line": business_line} if business_line else {}
    error_hint = ""
    for attempt in range(MAX_CYPHER_RETRIES + 1):
        cypher = await _generate_cypher(question, entities, llm, error_hint)
        logger.info(f"[GraphRAG] Cypher (attempt {attempt + 1}): {cypher}")
        if _CYPHER_WRITE_RE.search(cypher):
            # 写操作一律拒绝：question 可携带用户注入内容，不拦截会被
            # 注入进 DETACH DELETE/SET，污染 feedback_loop 维护的图谱权重
            error_hint = "生成的 Cypher 包含写操作关键字，只允许只读查询"
            logger.warning(f"[GraphRAG] 拒绝执行含写操作的 Cypher: {cypher[:120]}")
            continue
        if business_line:
            violation = _scope_violation(cypher)
            if violation:
                error_hint = violation
                logger.warning(f"[GraphRAG] 拒绝执行越线的 Cypher: {violation} | {cypher[:120]}")
                continue
        try:
            records = await asyncio.to_thread(_run_cypher_sync, neo4j_driver, cypher, params)
            if not records:
                # 空结果必须显式暴露：方向写反/关系缺失导致的静默失效，上游 impact 层
                # 的 try/except 不会报错，只能靠这条告警定位
                logger.warning(
                    f"[GraphRAG] 查询返回空 question={question[:80]} cypher={cypher[:120]}"
                )
            return records[:20]
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"[GraphRAG] Cypher 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_CYPHER_RETRIES:
                return []
    # 重试耗尽仍不合规 → 返回空（fail-closed），且留痕
    logger.warning(
        f"[GraphRAG] 重试 {MAX_CYPHER_RETRIES} 次后仍未生成合规作用域的 Cypher，返回空结果"
    )
    return []
