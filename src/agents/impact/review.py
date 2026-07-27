"""变更影响分析核心逻辑。4 路并行检查 + 风险评级。"""

import asyncio
import json

from langchain_openai import ChatOpenAI

from src.agents.impact.prompts import PARSE_CHANGE_PROMPT, IMPACT_REPORT_PROMPT
from src.core.config import get_settings

settings = get_settings()


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.3,
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


async def _parse_change_request(description: str) -> dict:
    """LLM 解析变更描述 → 结构化信息。"""
    llm = _get_llm()
    prompt = PARSE_CHANGE_PROMPT.format(change_description=description)
    response = await llm.ainvoke(prompt)
    try:
        return _parse_llm_json(response)
    except Exception:
        return {
            "config_items": [],
            "scope": "",
            "target_baseline": "",
            "change_type": "其他",
            "business_line": "ia",
            "risk_signals": [],
        }


async def _check_dependency(config_items: list[str]) -> dict:
    """Neo4j 检查配置项依赖冲突。"""
    if not config_items:
        return {"conflicts": [], "summary": "无依赖冲突"}

    try:
        from src.infra.neo4j_client import get_neo4j_driver
        driver = get_neo4j_driver()

        cypher = """
        MATCH (ci:ConfigItem)-[:DEPENDS_ON*1..2]->(dep:ConfigItem)
        WHERE ci.name IN $items
        RETURN ci.name AS source, collect(dep.name) AS depends_on
        """
        with driver.session() as session:
            result = session.run(cypher, items=config_items)
            deps = [r.data() for r in result]

        if not deps:
            return {"conflicts": [], "summary": "未发现依赖关系"}

        return {
            "conflicts": deps,
            "summary": f"发现 {len(deps)} 条依赖关系，需确认兼容性",
        }
    except Exception as e:
        return {"conflicts": [], "summary": f"依赖检查异常: {str(e)}"}


async def _check_baseline(config_items: list[str], target_baseline: str) -> dict:
    """检查基线冲突：配置项是否与冻结基线冲突。"""
    conflicts = []
    if not target_baseline or not config_items:
        return {"conflicts": [], "summary": "无基线冲突"}

    try:
        from src.infra.neo4j_client import get_neo4j_driver
        driver = get_neo4j_driver()

        cypher = """
        MATCH (b:Baseline {name: $baseline})<-[:TARGETS]-(cr:ChangeRequest)-[:AFFECTED_BY]->(ci:ConfigItem)
        WHERE ci.name IN $items
        RETURN ci.name AS config_item, b.name AS baseline, b.is_frozen AS is_frozen
        """
        with driver.session() as session:
            result = session.run(cypher, baseline=target_baseline, items=config_items)
            for r in result:
                if r.get("is_frozen"):
                    conflicts.append({
                        "config_item": r["config_item"],
                        "baseline": r["baseline"],
                        "level": "exact_match",
                    })

        return {
            "conflicts": conflicts,
            "summary": f"发现 {len(conflicts)} 个冻结基线冲突" if conflicts else "无基线冲突",
        }
    except Exception as e:
        return {"conflicts": [], "summary": f"基线检查异常: {str(e)}"}


async def _check_duplicate(config_items: list[str]) -> dict:
    """检查是否有重复/冲突的变更请求。"""
    if not config_items:
        return {"duplicates": [], "summary": "无重复变更"}

    try:
        # Simplified: check via PostgreSQL alm_change_requests
        import psycopg2
        s = get_settings()
        conn = psycopg2.connect(
            host=s.DB_HOST, port=s.DB_PORT,
            user=s.DB_USER, password=s.DB_PASSWORD, dbname=s.DB_NAME,
        )
        cur = conn.cursor()
        placeholders = ",".join(["%s"] * len(config_items))
        cur.execute(f"""
            SELECT cr.cr_no, cr.title, cr.status
            FROM alm_change_requests cr
            WHERE cr.status NOT IN ('closed', 'cancelled')
              AND cr.scope_desc ILIKE ANY(ARRAY[%s])
            LIMIT 5
        """, (placeholders,))
        # Note: simplified query — production would use proper config item matching
        rows = [{"cr_no": r[0], "title": r[1], "status": r[2]} for r in cur.fetchall()]
        cur.close()
        conn.close()

        return {
            "duplicates": rows,
            "summary": f"发现 {len(rows)} 个可能冲突的在途变更" if rows else "无重复变更",
        }
    except Exception as e:
        return {"duplicates": [], "summary": f"重复检查异常: {str(e)}"}


async def _check_scope(config_items: list[str], business_line: str) -> dict:
    """查询变更影响范围：哪些需求和配置项会受影响。"""
    if not config_items:
        return {"affected_requirements": [], "affected_config_items": [], "summary": "无法确定影响范围"}

    try:
        from src.infra.neo4j_client import get_neo4j_driver
        driver = get_neo4j_driver()

        # Neo4j 查询：配置项 → 实现的需求 → 同一需求下的其他配置项
        cypher = """
        MATCH (ci:ConfigItem)-[:IMPLEMENTED_BY]-(r:Requirement)-[:IMPLEMENTED_BY]-(related:ConfigItem)
        WHERE ci.name IN $items AND related.name <> ci.name
        RETURN ci.name AS source, r.req_no AS req_no, r.title AS req_title,
               collect(DISTINCT related.name) AS affected_config_items
        """
        with driver.session() as session:
            result = session.run(cypher, items=config_items)
            rows = [r.data() for r in result]

        if not rows:
            return {"affected_requirements": [], "affected_config_items": [], "summary": "未发现关联需求和配置项"}

        reqs = list({(r["req_no"], r["req_title"]) for r in rows})
        all_affected_cis = []
        for r in rows:
            all_affected_cis.extend(r["affected_config_items"])
        affected_cis = list(set(all_affected_cis))

        return {
            "affected_requirements": [{"req_no": r[0], "title": r[1]} for r in reqs],
            "affected_config_items": affected_cis[:10],
            "summary": f"影响 {len(reqs)} 条需求、{len(affected_cis)} 个关联配置项",
        }
    except Exception as e:
        return {"affected_requirements": [], "affected_config_items": [], "summary": f"范围检查异常: {str(e)}"}


async def analyze_impact(change_description: str) -> str:
    """
    变更影响分析入口。
    4 路并行检查 → LLM 汇总 → 返回 Markdown 报告。
    """
    # Step 1: 解析变更描述
    parsed = await _parse_change_request(change_description)
    config_items = parsed.get("config_items", [])
    target_baseline = parsed.get("target_baseline", "")
    business_line = parsed.get("business_line", "ia")

    # Step 2: 4 路并行检查
    scope_result, dep_result, baseline_result, dup_result = await asyncio.gather(
        _check_scope(config_items, business_line),
        _check_dependency(config_items),
        _check_baseline(config_items, target_baseline),
        _check_duplicate(config_items),
    )

    # Step 3: LLM 汇总生成报告
    llm = _get_llm()
    prompt = IMPACT_REPORT_PROMPT.format(
        config_items="、".join(config_items) if config_items else "未识别到具体配置项",
        baseline_conflicts=baseline_result["summary"],
        dependency_conflicts=dep_result["summary"],
        duplicate_changes=dup_result["summary"],
        scope_summary=scope_result["summary"],
    )
    response = await llm.ainvoke(prompt)
    report = response.content

    # Append detail sections
    detail_sections = []
    if scope_result["affected_requirements"]:
        detail_sections.append("### 影响范围\n" + "\n".join(
            f"- {r['req_no']}: {r['title']}" for r in scope_result["affected_requirements"]
        ))
    if baseline_result["conflicts"]:
        detail_sections.append("### 基线冲突详情\n" + "\n".join(
            f"- {c['config_item']} 与冻结基线 {c['baseline']} 冲突" for c in baseline_result["conflicts"]
        ))
    if dep_result["conflicts"]:
        detail_sections.append("### 依赖关系\n" + "\n".join(
            f"- {d['source']} → {', '.join(d['depends_on'])}" for d in dep_result["conflicts"]
        ))
    if dup_result["duplicates"]:
        detail_sections.append("### 在途变更\n" + "\n".join(
            f"- {d['cr_no']}: {d['title']} ({d['status']})" for d in dup_result["duplicates"]
        ))

    if detail_sections:
        report += "\n\n" + "\n\n".join(detail_sections)

    return report
