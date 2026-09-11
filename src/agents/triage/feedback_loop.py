"""诊断结论回流 + 历史结论复用。
三段闭环：图谱正向增强 → 历史结论复用 → 误诊修正。
"""

from src.core.config import get_settings
from src.core.logger import logger

settings = get_settings()


async def reinforce_graph_on_adopted(
    confirmed_phenomena: list[str],
    primary_cause_code: str,
    session_id: str = "",
) -> bool:
    """
    一段：诊断被采纳后，强化 Neo4j INDICATES 权重。
    每次验证权重涨 5%，上限 1.0。新关联从 0.3 起步。
    同时清零连续否决计数（采纳即"中断"否决连击）。
    """
    if not confirmed_phenomena or not primary_cause_code:
        return False

    try:
        from src.infra.neo4j_client import get_neo4j_driver
        driver = get_neo4j_driver()

        cypher = """
        MATCH (rc:RootCause {code: $cause_code})
        MATCH (ph:Phenomenon)
        WHERE ph.name IN $phenomena
        MERGE (rc)-[r:INDICATES]->(ph)
        ON CREATE SET r.weight = 0.3, r.is_core = false, r.reject_streak = 0
        ON MATCH  SET r.weight =
            CASE WHEN r.weight * 1.05 > 1.0 THEN 1.0 ELSE r.weight * 1.05 END,
            r.reject_streak = 0
        RETURN rc.code, ph.name, r.weight
        """

        with driver.session() as session:
            result = session.run(
                cypher,
                cause_code=primary_cause_code,
                phenomena=confirmed_phenomena,
            )
            records = [r.data() for r in result]

        for r in records:
            logger.info(
                f"[FEEDBACK-LOOP] INDICATES 权重更新: "
                f"{r['rc.code']} → {r['ph.name']} weight={r['r.weight']:.3f}"
            )

        return len(records) > 0

    except Exception as e:
        logger.warning(f"[FEEDBACK-LOOP] 图谱增强失败: {e}")
        return False


# 否决衰减系数：×0.8 太温和（1.0 跌到删除线 0.1 要连错 11 次），
# 收紧到 ×0.6 后 4 次否决触底，配合连击删除，错误关联不会长期滞留
REJECT_DECAY = 0.6
# 连续否决删除阈值：连续 3 次被否决直接删边，不等权重自然衰减
REJECT_STREAK_LIMIT = 3


async def weaken_graph_on_rejected(
    confirmed_phenomena: list[str],
    primary_cause_code: str,
    correct_cause_code: str | None = None,
) -> bool:
    """
    三段：诊断被拒绝后的误诊修正。

    1. 弱化错误根因：weight ×0.6；连续否决 ≥3 次或 weight < 0.1 → 删除关系
    2. 补写正确根因：人工纠正时反馈接口会带 correct_cause_code，
       用与采纳相同的 MERGE + 提权逻辑把正确答案写回图谱
       —— 否则系统只会"更不确定"，不会"更准"。

    Args:
        confirmed_phenomena: 本次诊断确认的现象
        primary_cause_code: 被否决的错误根因编码
        correct_cause_code: 人工给出的正确根因编码（可选，纠正时传入）
    """
    if not confirmed_phenomena or not primary_cause_code:
        return False

    try:
        from src.infra.neo4j_client import get_neo4j_driver
        driver = get_neo4j_driver()

        cypher_weaken = """
        MATCH (rc:RootCause {code: $cause_code})-[r:INDICATES]->(ph:Phenomenon)
        WHERE ph.name IN $phenomena
        SET r.weight = r.weight * $decay,
            r.reject_streak = coalesce(r.reject_streak, 0) + 1
        RETURN rc.code, ph.name, r.weight, r.reject_streak AS streak
        """

        cypher_remove = """
        MATCH (rc:RootCause {code: $cause_code})-[r:INDICATES]->(ph:Phenomenon)
        WHERE ph.name IN $phenomena
          AND (r.weight < 0.1 OR r.reject_streak >= $streak_limit)
        DELETE r
        RETURN rc.code, ph.name
        """

        with driver.session() as session:
            result = session.run(
                cypher_weaken,
                cause_code=primary_cause_code,
                phenomena=confirmed_phenomena,
                decay=REJECT_DECAY,
            )
            for r in result:
                logger.info(
                    f"[FEEDBACK-LOOP] INDICATES 权重降低: "
                    f"{r['rc.code']} → {r['ph.name']} weight={r['r.weight']:.3f} "
                    f"reject_streak={r['streak']}"
                )

            removed = session.run(
                cypher_remove,
                cause_code=primary_cause_code,
                phenomena=confirmed_phenomena,
                streak_limit=REJECT_STREAK_LIMIT,
            )
            for r in removed:
                logger.info(
                    f"[FEEDBACK-LOOP] INDICATES 关系删除: "
                    f"{r['rc.code']} → {r['ph.name']}"
                )

        # 纠正：把正确根因写回图谱（提权逻辑与采纳一致）
        reinforced = False
        if correct_cause_code and correct_cause_code != primary_cause_code:
            reinforced = await reinforce_graph_on_adopted(
                confirmed_phenomena=confirmed_phenomena,
                primary_cause_code=correct_cause_code,
                session_id="",
            )
            if reinforced:
                logger.info(
                    f"[FEEDBACK-LOOP] 正确根因已回写: {correct_cause_code} "
                    f"← {confirmed_phenomena}"
                )

        return True

    except Exception as e:
        logger.warning(f"[FEEDBACK-LOOP] 图谱弱化失败: {e}")
        return False


async def search_historical_cases(
    phenomena: list[str],
    business_line: str = "ev",
    limit: int = 5,
) -> list[dict]:
    """
    二段：历史结论复用。
    查 ai_triage_results 中已关闭且被采纳的诊断，
    返回 confirmed_phenomena 有交集的记录。
    """
    if not phenomena:
        return []

    try:
        import psycopg2
        conn = psycopg2.connect(
            host=settings.DB_HOST, port=settings.DB_PORT,
            user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
        )
        cur = conn.cursor()

        # 现象名来自用户输入（LLM 抽取），按不可信参数处理：
        # ILIKE 模式占位符化，通配符拼在参数值里，绝不进 SQL 文本
        like_patterns = [f"%{p}%" for p in phenomena[:5]]
        ilike_clauses = " OR ".join(
            ["tir.confirmed_phenomena::text ILIKE %s"] * len(like_patterns)
        )
        cur.execute(
            f"""SELECT tir.id, tir.session_id, tir.primary_cause_code,
                       tir.primary_confidence, tir.confirmed_phenomena,
                       tir.denied_phenomena, tir.total_rounds, tir.created_at,
                       ai.issue_no, ai.title, ai.status
                FROM ai_triage_results tir
                LEFT JOIN alm_issues ai ON ai.id = tir.source_issue_id
                WHERE tir.adopted = true
                  AND ai.business_line = %s
                  AND ({ilike_clauses})
                ORDER BY tir.primary_confidence DESC
                LIMIT %s""",
            (business_line, *like_patterns, limit),
        )

        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = []
        for r in rows:
            import json
            try:
                phenom = json.loads(r[4]) if isinstance(r[4], str) else r[4]
            except Exception:
                phenom = r[4]
            results.append({
                "id": r[0],
                "session_id": r[1],
                "primary_cause_code": r[2],
                "primary_confidence": r[3],
                "confirmed_phenomena": phenom,
                "denied_phenomena": r[5],
                "total_rounds": r[6],
                "created_at": str(r[7]) if r[7] else "",
                "issue_no": r[8],
                "issue_title": r[9],
                "issue_status": r[10],
            })

        logger.info(
            f"[FEEDBACK-LOOP] 历史结论检索: phenomena={phenomena} "
            f"hits={len(results)}"
        )
        return results

    except Exception as e:
        logger.warning(f"[FEEDBACK-LOOP] 历史结论检索失败: {e}")
        return []


def format_historical_cases(cases: list[dict]) -> str:
    """将历史结论格式化为可读文本。"""
    if not cases:
        return ""

    lines = ["\n**历史同类问题（已确诊）：**"]
    for i, c in enumerate(cases[:3], 1):
        cause = c["primary_cause_code"] or "未知"
        conf = c["primary_confidence"] or 0
        lines.append(
            f"{i}. [{c['issue_no']}] {c['issue_title'][:60]}\n"
            f"   根因：{cause}  |  置信度：{conf:.0%}"
        )
    return "\n".join(lines)
