from neo4j import GraphDatabase

from src.core.config import get_settings

_driver = None  # GraphDatabase.driver 实例（懒初始化）


def get_neo4j_driver() -> GraphDatabase.driver:
    """进程级单例（懒初始化）。应用 shutdown 时用 close_neo4j_driver 归还。"""
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        _driver.verify_connectivity()
    return _driver


def close_neo4j_driver() -> None:
    """应用 shutdown 时归还连接。未初始化时静默跳过。"""
    global _driver
    if _driver is None:
        return
    try:
        _driver.close()
    except Exception:
        pass
    _driver = None
