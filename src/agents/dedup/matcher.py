"""问题去重匹配器。四信号加权打分。"""

from dataclasses import dataclass, field

from src.infra.neo4j_client import get_neo4j_driver


@dataclass
class DedupMatch:
    issue_id: int
    issue_no: str
    title: str
    text_similarity: float = 0.0
    dtc_overlap: float = 0.0
    phenom_overlap: float = 0.0
    root_cause_match: float = 0.0
    combined_score: float = 0.0


@dataclass
class DedupResult:
    source_issue_id: int
    is_duplicate: bool = False
    matches: list[DedupMatch] = field(default_factory=list)


class DedupMatcher:
    """问题去重匹配器。四信号加权：文本相似 / DTC 重叠 / 现象重叠 / 根因匹配。"""

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    async def detect(self, issue_id: int) -> DedupResult:
        """检测指定问题单是否与已有问题重复。"""
        result = DedupResult(source_issue_id=issue_id)

        # 加载源问题单
        source = await self._load_issue(issue_id)
        if not source:
            return result

        # 加载候选问题单（同业务线、open/analyzing、近90天）
        candidates = await self._load_candidates(source)
        if not candidates:
            return result

        # 获取源问题单的 triage 信息
        source_phenomena, source_rc = await self._load_triage_info(issue_id)

        for cand in candidates:
            cand_id = cand["id"]
            if cand_id == issue_id:
                continue

            # 获取候选问题单的 triage 信息
            cand_phenomena, cand_rc = await self._load_triage_info(cand_id)

            # 信号1: 文本相似度（关键词 Jaccard）
            text_sim = self._compute_text_jaccard(
                f"{source.get('title', '')} {source.get('description', '')}",
                f"{cand.get('title', '')} {cand.get('description', '')}",
            )

            # 信号2: DTC 重叠
            dtc_sim = self._compute_dtc_jaccard(
                source.get("dtc_snapshot", ""),
                cand.get("dtc_snapshot", ""),
            )

            # 信号3: 现象重叠（Neo4j: 共享 RootCause 的 phenomena）
            phenom_sim = await self._compute_phenom_overlap(
                source_phenomena, cand_phenomena,
            )

            # 信号4: 根因匹配
            rc_match = 1.0 if source_rc and cand_rc and source_rc == cand_rc else 0.0

            # 加权综合得分
            combined = (
                0.30 * text_sim +
                0.25 * dtc_sim +
                0.25 * phenom_sim +
                0.20 * rc_match
            )

            if combined >= self.threshold:
                result.matches.append(DedupMatch(
                    issue_id=cand_id,
                    issue_no=cand.get("issue_no", ""),
                    title=cand.get("title", ""),
                    text_similarity=text_sim,
                    dtc_overlap=dtc_sim,
                    phenom_overlap=phenom_sim,
                    root_cause_match=rc_match,
                    combined_score=combined,
                ))

        result.matches.sort(key=lambda x: x.combined_score, reverse=True)
        result.is_duplicate = len(result.matches) > 0
        return result

    async def _load_issue(self, issue_id: int) -> dict | None:
        """从 PG 加载问题单数据。"""
        import psycopg2
        from src.core.config import get_settings
        s = get_settings()
        try:
            conn = psycopg2.connect(
                host=s.DB_HOST, port=s.DB_PORT,
                user=s.DB_USER, password=s.DB_PASSWORD, dbname=s.DB_NAME,
            )
            cur = conn.cursor()
            cur.execute(
                "SELECT id, issue_no, title, description, dtc_snapshot, business_line "
                "FROM alm_issues WHERE id = %s",
                (issue_id,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            if not row:
                return None
            return {
                "id": row[0], "issue_no": row[1], "title": row[2],
                "description": row[3] or "", "dtc_snapshot": row[4] or "",
                "business_line": row[5] or "",
            }
        except Exception:
            return None

    async def _load_candidates(self, source: dict) -> list[dict]:
        """加载候选问题单（同业务线、未关闭、近90天）。"""
        import psycopg2
        from src.core.config import get_settings
        s = get_settings()
        try:
            conn = psycopg2.connect(
                host=s.DB_HOST, port=s.DB_PORT,
                user=s.DB_USER, password=s.DB_PASSWORD, dbname=s.DB_NAME,
            )
            cur = conn.cursor()
            cur.execute(
                "SELECT id, issue_no, title, description, dtc_snapshot, business_line "
                "FROM alm_issues "
                "WHERE status IN ('open', 'analyzing') "
                "  AND business_line = %s "
                "  AND id != %s "
                "  AND updated_at > NOW() - INTERVAL '90 days' "
                "ORDER BY updated_at DESC "
                "LIMIT 50",
                (source.get("business_line", ""), source["id"]),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [
                {
                    "id": r[0], "issue_no": r[1], "title": r[2],
                    "description": r[3] or "", "dtc_snapshot": r[4] or "",
                    "business_line": r[5] or "",
                }
                for r in rows
            ]
        except Exception:
            return []

    async def _load_triage_info(self, issue_id: int) -> tuple[list[str], str]:
        """加载某个问题单的最新分诊结果（phenomena + primary_cause_code）。"""
        import psycopg2
        from src.core.config import get_settings
        s = get_settings()
        try:
            conn = psycopg2.connect(
                host=s.DB_HOST, port=s.DB_PORT,
                user=s.DB_USER, password=s.DB_PASSWORD, dbname=s.DB_NAME,
            )
            cur = conn.cursor()
            cur.execute(
                "SELECT confirmed_phenomena, primary_cause_code "
                "FROM ai_triage_results "
                "WHERE source_issue_id = %s "
                "ORDER BY created_at DESC LIMIT 1",
                (issue_id,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            if not row:
                return [], ""
            import json
            phenomena = json.loads(row[0]) if row[0] else []
            return phenomena, row[1] or ""
        except Exception:
            return [], ""

    def _compute_text_jaccard(self, text_a: str, text_b: str) -> float:
        """关键词 Jaccard 相似度。"""
        try:
            import jieba
            tokens_a = set(jieba.lcut(text_a))
            tokens_b = set(jieba.lcut(text_b))
        except ImportError:
            tokens_a = set(text_a)
            tokens_b = set(text_b)

        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _compute_dtc_jaccard(self, dtc_a: str, dtc_b: str) -> float:
        """DTC 码 Jaccard 重叠。"""
        set_a = {x.strip() for x in dtc_a.replace("，", ",").split(",") if x.strip()}
        set_b = {x.strip() for x in dtc_b.replace("，", ",").split(",") if x.strip()}

        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    async def _compute_phenom_overlap(
        self, phenoms_a: list[str], phenoms_b: list[str],
    ) -> float:
        """Neo4j: 计算两个现象集通过共享 RootCause 的重叠度。"""
        if not phenoms_a or not phenoms_b:
            return 0.0

        shared = 0
        total = len(phenoms_a) * len(phenoms_b)
        driver = get_neo4j_driver()

        try:
            with driver.session() as session:
                for pa in phenoms_a:
                    for pb in phenoms_b:
                        result = session.run(
                            """MATCH (p1:Phenomenon {name: $pa})
                               MATCH (p2:Phenomenon {name: $pb})
                               MATCH (p1)<-[:INDICATES]-(rc:RootCause)-[:INDICATES]->(p2)
                               RETURN rc.code LIMIT 1""",
                            pa=pa, pb=pb,
                        )
                        if result.single():
                            shared += 1
        except Exception:
            pass

        return shared / max(total, 1)


_dedup_matcher: DedupMatcher | None = None


def get_dedup_matcher() -> DedupMatcher:
    global _dedup_matcher
    if _dedup_matcher is None:
        _dedup_matcher = DedupMatcher()
    return _dedup_matcher
