# ============================================================
# 数据作用域（业务线）的请求级载体
#
# 为什么用 contextvar 而不是层层传参：作用域必须出现在**每一次**数据访问上，
# 而调用链上有 LLM 工具、图节点、后台任务三层。靠参数一路手传，漏一处就是
# 静默跨线取数（不是报错，是查到别人的数据）。contextvar 的语义是「随任务传播」，
# 在请求边界设一次，下游所有 DB 会话都能读到 —— 与 asyncio 天然契合：
# 每个请求是独立 task，task 创建时复制上下文，互不污染。
#
# 与 permissions.py 的关系：那里是「这个角色能看哪些行」的规则（返回 SQL 条件），
# 这里是「这次访问属于哪条业务线」的事实。两者配合：应用层用规则显式过滤，
# 数据库层用 RLS + 这里的值兜底。
# ============================================================

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

# 当前请求/任务的数据作用域。None = 未设定（维护脚本、种子、后台任务等）
_current_business_line: ContextVar[str | None] = ContextVar(
    "current_business_line", default=None
)


def get_business_line() -> str | None:
    """读取当前作用域；未设定返回 None。"""
    return _current_business_line.get()


def set_business_line(business_line: str | None) -> None:
    """设置当前作用域（空串归一为 None，避免把 '' 当成一条真实业务线）。"""
    _current_business_line.set(business_line or None)


@contextmanager
def business_line_scope(business_line: str | None) -> Iterator[None]:
    """临时设定作用域，退出时恢复原值。

    请求边界（中间件）与后台任务都应该包一层，避免作用域泄漏到相邻任务。
    """
    token = _current_business_line.set(business_line or None)
    try:
        yield
    finally:
        _current_business_line.reset(token)
