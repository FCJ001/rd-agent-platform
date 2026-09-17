# ============================================================
# 静态守卫：裸 psycopg2 连接必须带上数据作用域
#
# 为什么值得单独立一条测试：这类缺陷出现形态完全一致，而且**不报错** ——
#   RLS 策略读 current_setting('app.business_line')，裸连接不设置它，
#   查询就返回空集。表现是「去重突然什么都发现不了」「verify_items 全空」
#   「重复变更检查永远说没重复」，而不是异常。开发期 RLS 没开启时完全正常，
#   上线开启后才集体发作 —— 靠人 review 抓不住，靠这条守卫抓。
#
# 判据：函数体内出现 psycopg2.connect 的，同一函数体内必须出现 apply_scope。
# （scripts/ 下的维护脚本不在范围内：它们以表属主运行，本就没有业务线概念）
# ============================================================

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"

# 允许豁免的例外：这里放"确实不需要作用域"的连接，必须写明理由
EXEMPT: dict[str, str] = {}


def _iter_source_files():
    yield from SRC_ROOT.rglob("*.py")


def _enclosing_function(tree: ast.AST, lineno: int) -> ast.AST | None:
    """返回覆盖该行号的最内层函数定义。"""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= lineno <= end:
                if best is None or node.lineno > best.lineno:
                    best = node
    return best


def _has_call(node: ast.AST, func_name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            f = child.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name == func_name:
                return True
    return False


def test_raw_psycopg2_connections_apply_scope():
    offenders: list[str] = []

    for path in _iter_source_files():
        rel = str(path.relative_to(SRC_ROOT.parent))
        if rel in EXEMPT:
            continue
        source = path.read_text(encoding="utf-8")
        if "psycopg2.connect" not in source:
            continue

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            is_connect = (
                isinstance(f, ast.Attribute) and f.attr == "connect"
                and isinstance(f.value, ast.Name) and f.value.id == "psycopg2"
            )
            if not is_connect:
                continue

            fn = _enclosing_function(tree, node.lineno)
            if fn is None:
                offenders.append(f"{rel}:{node.lineno} 模块级连接（无法判定作用域）")
                continue
            if not _has_call(fn, "apply_scope"):
                offenders.append(
                    f"{rel}:{node.lineno} 在 {fn.name}() 里 —— 该函数没有调用 apply_scope"
                )

    assert not offenders, (
        "以下裸 psycopg2 连接未带上数据作用域，RLS 启用后会静默返回空：\n  "
        + "\n  ".join(offenders)
        + "\n修法：建连后立刻 apply_scope(conn)（见 src/infra/pg_scope.py）；"
        "确属不需要作用域的，加进本文件 EXEMPT 并写明理由"
    )


def test_scope_hook_is_wired_into_session_factory():
    """SQLAlchemy 会话那条路也必须有钩子（RLS 的另一半）。"""
    from src.infra.db import AsyncSessionLocal

    hook = AsyncSessionLocal.kw.get("sync_session_class")
    assert hook is not None, "会话工厂没绑定带作用域钩子的 Session 子类"
    assert "after_begin" in dir(hook.dispatch), "Session 子类上没有 after_begin 监听"
