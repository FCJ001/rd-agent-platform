"""问题去重匹配器。架构设计：双门槛 —— 向量相似 + 结构化精确匹配。

is_dup = sim_score >= 0.88 and (same_model_and_sw or same_dtc)

召回层（门槛一）走 Milvus 向量检索（vector_index）：源文本只 embed 一次，
近邻在 Milvus 内检索，不再把候选逐条 embed + 本地算余弦。
Milvus 不可用时自动降级旧的暴力路径（加载近 90 天候选逐条 embed），
降级是性能损失，不影响正确性。门槛二（车型+软件版本 / DTC 重叠）
始终在 PG 侧做精确匹配。
"""

import asyncio
from dataclasses import dataclass, field

from src.core.config import get_settings
from src.core.logger import logger

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


def get_embedding_model():
    from langchain_community.embeddings import DashScopeEmbeddings

    return DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


def _split_dtc(dtc_str: str | None) -> set[str]:
    if not dtc_str:
        return set()
    return {x.strip() for x in dtc_str.replace("\uff0c", ",").split(",") if x.strip()}


class DedupMatcher:
    """问题去重匹配器。双门槛：向量相似 (Milvus ANN) ≥ 0.88 + 结构化精确匹配。"""

    def __init__(self):
        self._embedding_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    async def detect(self, issue_id: int) -> DedupResult:
        """检测指定问题单是否与已有问题重复。"""
        logger.info(f"[DEDUP] detect issue_id={issue_id}")
        source = await self._load_issue_full(issue_id)
        if not source:
            logger.warning(f"[DEDUP] issue_id={issue_id} not found")
            return DedupResult(source_issue_id=issue_id)

        return await self._match(source, issue_id)

    async def detect_by_text(
        self, description: str, dtc_codes: str = "", business_line: str = "",
        model_code: str = "", sw_version: str = "",
    ) -> DedupResult:
        """根据文本描述搜索重复问题（无需 issue_id）。

        ★ business_line 无默认值：曾经默认 "ia" 导致对话入口无论用户在
          哪条线，都只在 ia 的切片里召回。留空 = 跨线检索（调用方拿不到
          scope 时），此时精度由门槛二（车型+版本 / DTC 重叠）兜住。
        """
        source = {
            "id": 0,
            "title": description,
            "description": "",
            "dtc_snapshot": dtc_codes,
            "model_code": model_code,
            "sw_version": sw_version,
            "business_line": business_line,
        }
        return await self._match(source, 0)

    # ── 匹配主流程 ───────────────────────────────────────────────

    async def _match(self, source: dict, source_id: int) -> DedupResult:
        """向量召回（Milvus，失败降级暴力扫描）→ 结构化门槛 → 判重。"""
        logger.info(f"[DEDUP] _match source_id={source_id}")
        result = DedupResult(source_issue_id=source_id)

        source_text = f"{source.get('title', '')} {source.get('description', '')}"
        source_dtc = _split_dtc(source.get("dtc_snapshot"))
        source_model = source.get("model_code", "")
        source_sw = source.get("sw_version", "")

        try:
            # embed_query 是同步 HTTP 调用，丢线程池避免阻塞 event loop
            source_emb = await asyncio.to_thread(self.embedding_model.embed_query, source_text)
        except Exception:
            logger.warning(f"[DEDUP] embedding failed for source_id={source_id}")
            return result

        # ── 门槛一：向量相似召回 ──
        # business_line 留空 = 不过滤（跨线召回），不再兜底成某条固定线
        scope = source.get("business_line", "") or ""
        candidates: list[dict] = []
        try:
            from src.agents.dedup import vector_index

            hits = await asyncio.to_thread(
                vector_index.search_similar,
                source_emb,
                scope,
                exclude_id=source_id or None,
            )
            logger.info(f"[DEDUP] milvus recall: {len(hits)} hits")
            ids = [h["id"] for h in hits]
            sim_by_id = {h["id"]: h["similarity"] for h in hits}
            rows = await self._load_issues_by_ids(ids)
            # 候选行附上召回相似度，交给统一的结构化门槛
            candidates = [dict(row, _similarity=sim_by_id.get(row["id"], 0.0)) for row in rows]
        except Exception as e:
            logger.warning(f"[DEDUP] Milvus 召回失败，降级暴力扫描: {e}")
            candidates = await self._match_bruteforce(source, source_id, source_emb)

        # ── 门槛二：结构化精确匹配 ──
        for cand in candidates:
            cand_id = cand["id"]
            if cand_id == source_id:
                continue
            sim_score = cand.get("_similarity", 0.0)
            if sim_score < SIMILARITY_THRESHOLD:
                continue

            match = self._structured_gate(
                source_dtc=source_dtc,
                source_model=source_model,
                source_sw=source_sw,
                cand=cand,
                sim_score=sim_score,
            )
            if match is None:
                continue  # 向量过阈值但结构不匹配 → 不判定为重复
            result.matches.append(match)

        result.matches.sort(key=lambda x: x.similarity, reverse=True)
        result.is_duplicate = len(result.matches) > 0

        logger.info(f"[DEDUP] _match result is_duplicate={result.is_duplicate} matches={len(result.matches)} "
                    f"top_similarity={result.matches[0].similarity if result.matches else 0}")

        # ── 持久化到 ai_dedup_links ──
        if result.is_duplicate and source_id > 0:
            await self._save_dedup_links(source_id, result.matches)

        return result

    @staticmethod
    def _structured_gate(
        source_dtc: set[str], source_model: str, source_sw: str,
        cand: dict, sim_score: float,
    ) -> DedupMatch | None:
        """门槛二：车型+软件版本一致 / DTC 重叠，二者取其一才算重复。"""
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
            return None

        return DedupMatch(
            issue_id=cand["id"],
            issue_no=cand.get("issue_no", ""),
            title=cand.get("title", ""),
            similarity=round(sim_score, 4),
            evidence=evidence,
        )

    async def _match_bruteforce(
        self, source: dict, source_id: int, source_emb: list[float],
    ) -> list[dict]:
        """降级召回：Milvus 不可用时加载近 90 天候选，逐条 embed + 本地余弦。

        返回的候选带 _similarity 字段，与 Milvus 路径同构。"""
        candidates = await self._load_candidates_full(
            source.get("business_line", "") or "", exclude_id=source_id
        )
        logger.info(f"[DEDUP] bruteforce candidates loaded: {len(candidates)}")

        def _score(cand: dict) -> float | None:
            cand_text = f"{cand.get('title', '')} {cand.get('description', '')}"
            try:
                cand_emb = self.embedding_model.embed_query(cand_text)
            except Exception:
                return None
            return _cosine_similarity(source_emb, cand_emb)

        scored = []
        for cand in candidates:
            sim = await asyncio.to_thread(_score, cand)
            if sim is not None:
                scored.append(dict(cand, _similarity=sim))
        return scored

    # ── PG 访问：psycopg2 是同步驱动，async 入口一律 asyncio.to_thread 包装，
    #    连接在 finally 里关闭，避免异常路径泄漏连接 ──

    @staticmethod
    def _pg_connect():
        import psycopg2

        from src.infra.pg_scope import apply_scope

        conn = psycopg2.connect(
            host=settings.DB_HOST, port=settings.DB_PORT,
            user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
        )
        # ★ 裸连接必须先带上作用域，否则 RLS 开启后这些查询静默返回空
        apply_scope(conn)
        return conn

    async def _save_dedup_links(self, source_id: int, matches: list[DedupMatch]) -> None:
        """将去重结果写入 ai_dedup_links 影子表。"""
        def _sync():
            conn = self._pg_connect()
            try:
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
                cur.close()
            finally:
                conn.close()

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.warning(f"[DEDUP] 写入 ai_dedup_links 失败 source_id={source_id}: {e}")  # 写回失败不影响主流程

    async def _load_issue_full(self, issue_id: int) -> dict | None:
        def _sync():
            conn = self._pg_connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, issue_no, title, description, dtc_snapshot, model_code, "
                    "sw_version, business_line FROM alm_issues WHERE id = %s",
                    (issue_id,),
                )
                row = cur.fetchone()
                cur.close()
                if not row:
                    return None
                return {
                    "id": row[0], "issue_no": row[1], "title": row[2],
                    "description": row[3] or "", "dtc_snapshot": row[4] or "",
                    "model_code": row[5] or "", "sw_version": row[6] or "",
                    "business_line": row[7] or "",
                }
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_sync)
        except Exception as e:
            logger.warning(f"[DEDUP] 加载问题单失败 issue_id={issue_id}: {e}")
            return None

    async def _load_issues_by_ids(self, ids: list[int]) -> list[dict]:
        """按 Milvus 召回的 id 批量加载候选行。"""
        if not ids:
            return []

        def _sync():
            conn = self._pg_connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, issue_no, title, description, dtc_snapshot, model_code, "
                    "sw_version, business_line FROM alm_issues WHERE id = ANY(%s) "
                    "AND status IN ('open', 'analyzing')",
                    (ids,),
                )
                rows = cur.fetchall()
                cur.close()
                return [
                    {"id": r[0], "issue_no": r[1], "title": r[2],
                     "description": r[3] or "", "dtc_snapshot": r[4] or "",
                     "model_code": r[5] or "", "sw_version": r[6] or "",
                     "business_line": r[7] or ""}
                    for r in rows
                ]
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_sync)
        except Exception as e:
            logger.warning(f"[DEDUP] 批量加载候选失败: {e}")
            return []

    async def _load_candidates_full(self, business_line: str = "", exclude_id: int | None = None) -> list[dict]:
        """降级候选池。business_line 留空时不过滤（与 Milvus 路径同构）。"""
        def _sync():
            conn = self._pg_connect()
            try:
                cur = conn.cursor()
                sql = (
                    "SELECT id, issue_no, title, description, dtc_snapshot, model_code, "
                    "sw_version, business_line FROM alm_issues "
                    "WHERE status IN ('open', 'analyzing') "
                    "AND updated_at > NOW() - INTERVAL '90 days' "
                )
                params: list = []
                if business_line:
                    sql += " AND business_line = %s"
                    params.append(business_line)
                if exclude_id:
                    sql += " AND id != %s"
                    params.append(exclude_id)
                sql += " ORDER BY updated_at DESC LIMIT 50"
                cur.execute(sql, params)
                rows = cur.fetchall()
                cur.close()
                return [
                    {"id": r[0], "issue_no": r[1], "title": r[2],
                     "description": r[3] or "", "dtc_snapshot": r[4] or "",
                     "model_code": r[5] or "", "sw_version": r[6] or "",
                     "business_line": r[7] or ""}
                    for r in rows
                ]
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_sync)
        except Exception as e:
            logger.warning(f"[DEDUP] 加载降级候选失败: {e}")
            return []


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = (sum(a * a for a in vec_a)) ** 0.5
    norm_b = (sum(b * b for b in vec_b)) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_dedup_matcher: DedupMatcher | None = None


def get_dedup_matcher() -> DedupMatcher:
    global _dedup_matcher
    if _dedup_matcher is None:
        _dedup_matcher = DedupMatcher()
    return _dedup_matcher
