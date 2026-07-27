from functools import lru_cache

from neo4j import GraphDatabase

from src.core.config import get_settings


@lru_cache(maxsize=1)
def get_neo4j_driver() -> GraphDatabase.driver:
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    return driver
