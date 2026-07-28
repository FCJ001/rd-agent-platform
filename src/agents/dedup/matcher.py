"""问题去重匹配器。架构设计：双门槛 —— 向量相似 + 结构化精确匹配。

is_dup = sim_score >= 0.88 and (same_model_and_sw or same_dtc)
"""

from dataclasses import dataclass, field

from langchain_community.embeddings import DashScopeEmbeddings

from src.core.config import get_settings

settings = get_settings()

# 向量相似阈值
SIMILARITY_THRESHOLD = 0.88


@dataclass
class DedupMatch:
    issue_id: int
    issue_no: str
    title: str
    similarity: float = 0.0
    evidence: str = ""  # "model_and_sw" | "dtc" | "model_and_sw+dtc"


@dataclass
class DedupResult:
    source_issue_id: int
    is_duplicate: bool = False
    matches: list[DedupMatch] = field(default_factory=list)


def _get_embedding_model():
    return DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = (sum(a * a for a in vec_a)) ** 0.5
    norm_b = (sum(b * b for b in vec_b)) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_dtc(dtc_str: str | None) -> set[str]:
    if not dtc_str:
        return set()
    return {x.strip() for x in dtc_str.replace("\uff0c", ",").split(",") if x.strip()}


class DedupMatcher:
    """问题去重匹配器。双门槛：向量相似 (Milvus embedding) ≥ 0.88 + 结构化精确匹配。"""

    def __init__(self):
        self._embedding_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = _get_embedding_model()
        return self._embedding_model

    async def detect(self, issue_id: int) -> DedupResult:
        """检测指定问题单是否与已有问题重复。"""
        source = await self._load_issue_full(issue_id)
        if not source:
            return DedupResult(source_issue_id=issue_id)

        candidates = await self._load_candidates_full(source["business_line"], exclude_id=issue_id)
        return await self._match(source, candidates, issue_id)

    async def detect_by_text(
        self, description: str, dtc_codes: str = "", business_line: str = "ia",
        model_code: str = "", sw_version: str = "",
    ) -> DedupResult:
        """根据文本描述搜索重复问题（无需 issue_id）。"""
        source = {
            "id": 0,
            "title": description,
            "description": "",
            "dtc_snapshot": dtc_codes,
            "model_code": model_code,
            "sw_version": sw_version,
        }
        candidates = await self._load_candidates_full(business_line)
        return await self._match(source, candidates, 0)

    async def _match(
        self, source: dict, candidates: list[dict], source_id: int,
    ) -> DedupResult:
        """双门槛匹配逻辑。"""
        result = DedupResult(source_issue_id=source_id)
        if not candidates:
            return result

        # ── 门槛一：向量相似 ──
        source_text = f"{source.get('title', '')} {source.get('description', '')}"
        source_dtc = _split_dtc(source.get("dtc_snapshot"))
        source_model = source.get("model_code", "")
        source_sw = source.get("sw_version", "")

        try:
            source_emb = self.embedding_model.embed_query(source_text)
        except Exception:
            return result

        for cand in candidates:
            cand_id = cand["id"]
            if cand_id == source_id:
                continue

            cand_text = f"{cand.get('title', '')} {cand.get('description', '')}"
            try:
                cand_emb = self.embedding_model.embed_query(cand_text)
            except Exception:
                continue

            sim_score = _cosine_similarity(source_emb, cand_emb)
            if sim_score < SIMILARITY_THRESHOLD:
                continue

            # ── 门槛二：结构化精确匹配 ──
            cand_model = cand.get("model_code", "")
            cand_sw = cand.get("sw_version", "")
            cand_dtc = _split_dtc(cand.get("dtc_snapshot"))

            same_model_and_sw = (
                source_model and cand_model and source_model == cand_model
                and source_sw and cand_sw and source_sw == cand_sw
            )
            same_dtc = bool(source_dtc and cand_dtc and (source_dtc & cand_dtc))

            if same_model_and_sw and same_dtc:
                evidence = "model_and_sw+dtc"
            elif same_model_and_sw:
                evidence = "model_and_sw"
            elif same_dtc:
                evidence = "dtc"
            else:
                continue  # 向量过阈值但结构不匹配 → 不判定为重复

            result.matches.append(DedupMatch(
                issue_id=cand_id,
                issue_no=cand.get("issue_no", ""),
                title=cand.get("title", ""),
                similarity=round(sim_score, 4),
                evidence=evidence,
            ))

        result.matches.sort(key=lambda x: x.similarity, reverse=True)
        result.is_duplicate = len(result.matches) > 0

        # ── 持久化到 ai_dedup_links ──
        if result.is_duplicate and source_id > 0:
            await self._save_dedup_links(source_id, result.matches)

        return result

    async def _save_dedup_links(self, source_id: int, matches: list[DedupMatch]) -> None:
        """将去重结果写入 ai_dedup_links 影子表。"""
        import psycopg2
        try:
            conn = psycopg2.connect(
                host=settings.DB_HOST, port=settings.DB_PORT,
                user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
            )
            cur = conn.cursor()
            for m in matches:
                cur.execute(
                    "INSERT INTO ai_dedup_links (source_issue_id, matched_issue_id, similarity, evidence, is_duplicate) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (source_issue_id, matched_issue_id) DO UPDATE SET "
                    "similarity = EXCLUDED.similarity, evidence = EXCLUDED.evidence, "
                    "is_duplicate = EXCLUDED.is_duplicate, updated_at = NOW()",
                    (source_id, m.issue_id, m.similarity, m.evidence, True),
                )
            conn.commit()
            cur.close(); conn.close()
        except Exception:
            pass  # 写回失败不影响主流程

    async def _load_issue_full(self, issue_id: int) -> dict | None:
        import psycopg2
        try:
            conn = psycopg2.connect(
                host=settings.DB_HOST, port=settings.DB_PORT,
                user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
            )
            cur = conn.cursor()
            cur.execute(
                "SELECT id, issue_no, title, description, dtc_snapshot, model_code, "
                "sw_version, business_line FROM alm_issues WHERE id = %s",
                (issue_id,),
            )
            row = cur.fetchone()
            cur.close(); conn.close()
            if not row:
                return None
            return {
                "id": row[0], "issue_no": row[1], "title": row[2],
                "description": row[3] or "", "dtc_snapshot": row[4] or "",
                "model_code": row[5] or "", "sw_version": row[6] or "",
                "business_line": row[7] or "",
            }
        except Exception:
            return None

    async def _load_candidates_full(self, business_line: str, exclude_id: int | None = None) -> list[dict]:
        import psycopg2
        try:
            conn = psycopg2.connect(
                host=settings.DB_HOST, port=settings.DB_PORT,
                user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
            )
            cur = conn.cursor()
            if exclude_id:
                cur.execute(
                    "SELECT id, issue_no, title, description, dtc_snapshot, model_code, "
                    "sw_version, business_line FROM alm_issues "
                    "WHERE status IN ('open', 'analyzing') AND business_line = %s AND id != %s "
                    "AND updated_at > NOW() - INTERVAL '90 days' "
                    "ORDER BY updated_at DESC LIMIT 50",
                    (business_line, exclude_id),
                )
            else:
                cur.execute(
                    "SELECT id, issue_no, title, description, dtc_snapshot, model_code, "
                    "sw_version, business_line FROM alm_issues "
                    "WHERE status IN ('open', 'analyzing') AND business_line = %s "
                    "AND updated_at > NOW() - INTERVAL '90 days' "
                    "ORDER BY updated_at DESC LIMIT 50",
                    (business_line,),
                )
            rows = cur.fetchall()
            cur.close(); conn.close()
            return [
                {"id": r[0], "issue_no": r[1], "title": r[2],
                 "description": r[3] or "", "dtc_snapshot": r[4] or "",
                 "model_code": r[5] or "", "sw_version": r[6] or "",
                 "business_line": r[7] or ""}
                for r in rows
            ]
        except Exception:
            return []


_dedup_matcher: DedupMatcher | None = None


def get_dedup_matcher() -> DedupMatcher:
    global _dedup_matcher
    if _dedup_matcher is None:
        _dedup_matcher = DedupMatcher()
    return _dedup_matcher
