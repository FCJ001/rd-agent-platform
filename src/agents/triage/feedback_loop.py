"""诊断结论回流 + 历史结论复用。
三段闭环：图谱正向增强 → 历史结论复用 → 误诊修正。

★ Neo4j driver / psycopg2 都是同步客户端，async 入口一律 asyncio.to_thread
  包装 —— 否则 FastAPI 事件循环被阻塞，所有并发请求一起卡。
"""

import asyncio

from src.core.config import get_settings
from src.core.logger import logger

settings = get_settings()


def _reinforce_sync(confirmed_phenomena: list[str], primary_cause_code: str) -> list[dict]:
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
        result = session.run(cypher, cause_code=primary_cause_code, phenomena=confirmed_phenomena)
        return [r.data() for r in result]


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
        records = await asyncio.to_thread(_reinforce_sync, confirmed_phenomena, primary_cause_code)

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


def _weaken_sync(confirmed_phenomena: list[str], primary_cause_code: str) -> list[dict]:
    """弱化 + 删边，返回被删除的关联列表。"""
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

    removed_records = []
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
        removed_records = [r.data() for r in removed]

    return removed_records


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
        removed_records = await asyncio.to_thread(
            _weaken_sync, confirmed_phenomena, primary_cause_code
        )
        for r in removed_records:
            logger.info(
                f"[FEEDBACK-LOOP] INDICATES 关系删除: "
                f"{r['rc.code']} → {r['ph.name']}"
            )

        # 纠正：把正确根因写回图谱（提权逻辑与采纳一致）
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


