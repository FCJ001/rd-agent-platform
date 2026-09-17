# ============================================================
# 数据作用域（业务线）隔离的单元测试
#
# 锁住的行为契约：
#   1. 候选根因查询：业务线非空时必须带谓词且走参数绑定；为空时不得过滤
#   2. 去重向量召回：业务线参与 expr；留空不得伪装成某条固定线
#   3. 去重入口：不再有 "ia" 这种硬编码默认值
#
# ★ 这些用例不需要外部服务：driver / collection 都是假的，只验证
#   「发出去的查询长什么样」。真隔离要靠 integration 测试（跨线不可见）。
# ============================================================

import inspect

import pytest

from src.agents.dedup import matcher as dedup_matcher
from src.agents.dedup import vector_index
from src.agents.triage import graph_queries


# ── 1. Neo4j 候选根因查询的谓词 ────────────────────────────────

class _FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def data(self):
        return self._records


class _FakeSession:
    """记录最后一次执行的 Cypher 与参数。"""

    def __init__(self, records):
        self._records = records
        self.cypher = None
        self.params = None

    def run(self, cypher, **params):
        self.cypher = cypher
        self.params = params
        return _FakeResult(self._records)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self, records):
        self.session_obj = _FakeSession(records)

    def session(self):
        return self.session_obj


@pytest.fixture
def fake_neo4j(monkeypatch):
    driver = _FakeDriver([])
    monkeypatch.setattr(graph_queries, "get_neo4j_driver", lambda: driver)
    return driver


def test_candidates_filter_by_business_line(fake_neo4j):
    graph_queries.query_causes_by_phenomena(["中控屏显示异常"], business_line="ia")
    assert "rc.business_line = $business_line" in fake_neo4j.session_obj.cypher
    # 值是参数绑定，不拼进语句文本
    assert fake_neo4j.session_obj.params["business_line"] == "ia"


def test_candidates_without_line_do_not_filter(fake_neo4j):
    graph_queries.query_causes_by_phenomena(["中控屏显示异常"])
    # 只断言谓词：rc.business_line 在 RETURN 子句里本来就有（是输出字段）
    assert "rc.business_line = $business_line" not in fake_neo4j.session_obj.cypher


def test_candidates_empty_input_skips_query(fake_neo4j):
    assert graph_queries.query_causes_by_phenomena([], business_line="ia") == []
    assert fake_neo4j.session_obj.cypher is None


# ── 2. Milvus 去重召回的 expr ──────────────────────────────────

class _FakeEntity:
    def __init__(self, d):
        self._d = d

    def get(self, key, default=None):
        return self._d.get(key, default)


class _FakeHit:
    def __init__(self, d):
        self.entity = _FakeEntity(d)
        self.score = 0.93


class _FakeCollection:
    def __init__(self):
        self.expr = "unset"

    def search(self, **kwargs):
        self.expr = kwargs.get("expr")
        return [[_FakeHit({"id": "7", "issue_no": "ISS-2025-00007"})]]


@pytest.fixture
def fake_milvus(monkeypatch):
    coll = _FakeCollection()
    monkeypatch.setattr(vector_index, "_get_collection", lambda: coll)
    return coll


def test_dedup_expr_scoped_to_line(fake_milvus):
    hits = vector_index.search_similar([0.0], business_line="ia")
    assert fake_milvus.expr == 'business_line == "ia"'
    assert hits == [{"id": 7, "issue_no": "ISS-2025-00007", "similarity": 0.93}]


def test_dedup_expr_line_and_self_exclusion(fake_milvus):
    vector_index.search_similar([0.0], business_line="ev", exclude_id=7)
    assert fake_milvus.expr == 'business_line == "ev" and id != "7"'


def test_dedup_expr_no_line_means_no_filter(fake_milvus):
    """留空 = 跨线召回，而不是悄悄退化成某一条线。"""
    vector_index.search_similar([0.0])
    assert fake_milvus.expr is None


# ── 3. 去重入口不得再有硬编码业务线 ────────────────────────────

def test_detect_by_text_has_no_default_line():
    """曾经默认 "ia"，导致对话入口恒在 ia 切片里召回。"""
    default = inspect.signature(dedup_matcher.DedupMatcher.detect_by_text).parameters["business_line"].default
    assert default == ""


def test_matcher_does_not_fall_back_to_a_fixed_line(monkeypatch):
    """_match 里不得再把空 scope 兜底成某条线。"""
    source = inspect.getsource(dedup_matcher.DedupMatcher._match)
    assert '"ia"' not in source and "'ia'" not in source


# ── 4. GraphRAG（NL2Cypher）作用域强制 ──────────────────────────

from src.agents.triage import graph_rag  # noqa: E402


def test_scope_violation_accepts_parameterized_query():
    cypher = "MATCH (rc:RootCause) WHERE rc.business_line = $business_line RETURN rc.code"
    assert graph_rag._scope_violation(cypher) is None


def test_scope_violation_rejects_missing_scope():
    cypher = "MATCH (rc:RootCause) WHERE rc.name = '座舱卡死' RETURN rc.code"
    reason = graph_rag._scope_violation(cypher)
    assert reason and "$business_line" in reason


def test_scope_violation_rejects_hardcoded_line():
    """写死业务线 = 查询自己决定作用域，注入内容就能越线。"""
    cypher = "MATCH (rc:RootCause) WHERE rc.business_line = 'ia' RETURN rc.code"
    reason = graph_rag._scope_violation(cypher)
    assert reason and "字面量" in reason


def test_scope_violation_rejects_in_list_literal():
    cypher = "MATCH (rc:RootCause) WHERE rc.business_line IN ['ev','ia'] RETURN rc.code"
    assert graph_rag._scope_violation(cypher) is not None


def test_scope_violation_does_not_flag_map_keys():
    """Cypher 映射键（{"name": ...}）不能误判成业务线字面量。"""
    cypher = (
        "MATCH (rc:RootCause) WHERE rc.business_line = $business_line "
        'RETURN collect({"name": rc.name, "code": rc.code}) AS items'
    )
    assert graph_rag._scope_violation(cypher) is None


class _RecordingDriver:
    """记录被执行的 Cypher；search_graph_raw 只应执行合规查询。"""

    def __init__(self):
        self.executed = []
        self.params = []

    def session(self):
        return self

    def execute_read(self, fn):
        return fn(self)

    def run(self, cypher, params=None):
        self.executed.append(cypher)
        self.params.append(params)
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_graph_rag_refuses_unescoped_cypher(monkeypatch):
    """作用域缺失时不得执行，重试耗尽返回空（fail-closed）。"""
    driver = _RecordingDriver()

    async def fake_entities(question, llm):
        return {}

    async def fake_cypher(question, entities, llm, error_hint=""):
        return "MATCH (rc:RootCause) RETURN rc.code"  # 恒不合规

    monkeypatch.setattr(graph_rag, "_extract_entities", fake_entities)
    monkeypatch.setattr(graph_rag, "_generate_cypher", fake_cypher)

    assert await graph_rag.search_graph_raw("q", driver, object(), business_line="ia") == []
    assert driver.executed == []  # 一条都没执行


@pytest.mark.asyncio
async def test_graph_rag_executes_with_bound_scope(monkeypatch):
    driver = _RecordingDriver()
    seen = {}

    async def fake_entities(question, llm):
        return {}

    async def fake_cypher(question, entities, llm, error_hint=""):
        return "MATCH (rc:RootCause) WHERE rc.business_line = $business_line RETURN rc.code"

    def fake_run(drv, cypher, params=None):
        seen["params"] = params
        return [{"code": "RC-IA-0001"}]

    monkeypatch.setattr(graph_rag, "_extract_entities", fake_entities)
    monkeypatch.setattr(graph_rag, "_generate_cypher", fake_cypher)
    monkeypatch.setattr(graph_rag, "_run_cypher_sync", fake_run)

    records = await graph_rag.search_graph_raw("q", driver, object(), business_line="ia")
    assert records == [{"code": "RC-IA-0001"}]
    # 作用域值走绑定参数，不拼进语句文本
    assert seen["params"] == {"business_line": "ia"}


# ── 5. 作用域载体（ContextVar）与请求边界语义 ──────────────────

from src.core.scope import business_line_scope, get_business_line, set_business_line  # noqa: E402


def test_scope_default_is_none():
    assert get_business_line() is None


def test_scope_context_manager_restores():
    set_business_line("ev")
    with business_line_scope("ia"):
        assert get_business_line() == "ia"
    # 退出后必须恢复，否则 ASGI 协程复用会让下一个请求继承上一个人的作用域
    assert get_business_line() == "ev"
    set_business_line(None)


def test_scope_empty_string_normalizes_to_none():
    """空串不是一条真实业务线，归一为 None 以免污染 RLS 判定。"""
    with business_line_scope(""):
        assert get_business_line() is None


def test_scope_manager_resets_on_exception():
    with pytest.raises(RuntimeError):
        with business_line_scope("ia"):
            raise RuntimeError("boom")
    assert get_business_line() is None
