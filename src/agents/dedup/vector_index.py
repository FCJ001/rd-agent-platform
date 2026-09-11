# ============================================================
# 问题单去重向量索引（Milvus）
#
# DedupMatcher 的召回层：源问题文本只 embed 一次，近邻检索交给 Milvus
# （IVF_FLAT + COSINE）。旧实现把候选逐条拉出来逐条 embed + 本地算余弦，
# 召回成本是 O(N) 次 embedding API 调用 + O(N×D) 本地余弦；新实现是
# O(1) 次 embedding + 一次 ANN 检索。结构化门槛（车型+软件版本 /
# DTC 重叠）仍在 PG 侧做精确匹配 —— 向量只管召回，判定权在规则。
#
# 集合：alm_issue_dedup
#   id            VARCHAR 主键 = str(alm_issues.id)
#   issue_no      VARCHAR
#   business_line VARCHAR（检索时标量过滤）
#   embedding     FLOAT_VECTOR dim=1024（text-embedding-v3）
# ============================================================

from src.core.logger import logger
from src.infra.milvus_client import get_milvus_client_alias

COLLECTION_NAME = "alm_issue_dedup"
EMBEDDING_DIMS = 1024  # text-embedding-v3
SEARCH_TOP_K = 20      # ANN 召回条数，结构化门槛再过滤


def _ensure_collection(alias: str):
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

    if utility.has_collection(COLLECTION_NAME, using=alias):
        return Collection(COLLECTION_NAME, using=alias)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="issue_no", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="business_line", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS),
    ]
    collection = Collection(
        COLLECTION_NAME,
        schema=CollectionSchema(fields, description="ALM issue dedup index"),
        using=alias,
    )
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )
    collection.load()
    logger.info(f"[DEDUP-INDEX] 集合已创建: {COLLECTION_NAME}")
    return collection


def _get_collection():
    return _ensure_collection(get_milvus_client_alias())


def issue_embed_text(issue: dict) -> str:
    """问题单 → 参与向量化的文本（标题 + 描述）。"""
    return f"{issue.get('title', '')} {issue.get('description', '')}".strip()


def upsert_issues(issues: list[dict]) -> int:
    """批量写入/更新问题单向量。每个 issue 需含 id/issue_no/business_line
    （title/description 可选，缺省则跳过该条）。返回写入条数。

    同步阻塞（Milvus ORM），调用方应放到后台任务/脚本里跑。"""
    if not issues:
        return 0

    from src.agents.dedup.matcher import get_embedding_model

    embed_model = get_embedding_model()
    texts, rows = [], []
    for issue in issues:
        text = issue_embed_text(issue)
        if not issue.get("id") or not text:
            continue
        texts.append(text)
        rows.append({
            "id": str(issue["id"]),
            "issue_no": issue.get("issue_no", ""),
            "business_line": issue.get("business_line", ""),
        })
    if not rows:
        return 0

    vectors = embed_model.embed_documents(texts)
    collection = _get_collection()
    # Milvus 布尔表达式的字符串字面量要双引号，不能用 Python list repr（单引号）
    ids_expr = 'id in ["' + '", "'.join(r["id"] for r in rows) + '"]'
    collection.delete(expr=ids_expr)
    collection.insert([{
        "id": r["id"],
        "issue_no": r["issue_no"],
        "business_line": r["business_line"],
        "embedding": vec,
    } for r, vec in zip(rows, vectors)])
    collection.flush()
    logger.info(f"[DEDUP-INDEX] upsert {len(rows)} 条问题单向量")
    return len(rows)


def delete_issue(issue_id: int) -> None:
    """问题单关闭/删除时移除向量（关闭的单不再参与召回）。"""
    collection = _get_collection()
    collection.delete(expr=f'id == "{issue_id}"')
    collection.flush()


def search_similar(
    query_embedding: list[float],
    business_line: str,
    exclude_id: int | None = None,
    top_k: int = SEARCH_TOP_K,
) -> list[dict]:
    """ANN 近邻检索，返回 [{id: int, issue_no: str, similarity: float}, ...]。

    business_line 精确过滤 + 排除自身；相似度即 Milvus COSINE 分数。"""
    collection = _get_collection()
    expr = f'business_line == "{business_line}"'
    if exclude_id:
        expr += f' and id != "{exclude_id}"'

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        expr=expr,
        output_fields=["id", "issue_no"],
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "id": int(hit.entity.get("id")),
            "issue_no": hit.entity.get("issue_no", ""),
            "similarity": float(hit.score),
        })
    return hits
