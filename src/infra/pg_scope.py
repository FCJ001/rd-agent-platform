# ============================================================
# 给「裸 psycopg2 连接」设置数据作用域
#
# 与 src/infra/db.py 的 after_begin 钩子同源，但管的是另一条路：
#   · db.py 钩子     → SQLAlchemy 会话（请求/Agent 节点走的都是这条）
#   · 本模块 apply_scope → 自己开 psycopg2 的查询（去重索引、历史结论检索）
#
# ★ 为什么必须有：RLS 策略读的是 current_setting('app.business_line')。
#   裸连接不设置这个变量，策略下就等于「没设作用域」→ 一行都查不到。
#   失败形态是**静默返回空**（不是报错）—— 表现成「去重突然什么都发现不了」
#   「历史结论一条都查不到」，极难定位。所以凡是自建 psycopg2 连接的地方，
#   建完连接立刻调用本函数。
#
# 这里用会话级（set_config(..., false)）而不是 SET LOCAL：裸连接的调用方
# 各自建连、用完即关，没有连接池复用，不存在跨请求残留问题；
# 反过来若用 SET LOCAL，调用方不开显式事务时它会随语句结束失效。
# ============================================================

from src.core.logger import logger
from src.core.scope import get_business_line


def apply_scope(conn) -> None:
    """把当前作用域写进这条 psycopg2 连接。无作用域时什么都不做。

    无作用域不动连接是刻意的：维护脚本（种子/回填/修复）本就没有业务线
    概念，且它们应以表属主身份运行（属主在未加 FORCE 时不受策略约束，
    加了 FORCE 也仍可用 —— 见 scripts/enable_rls.py 的前置检查）。
    """
    business_line = get_business_line()
    if not business_line:
        return
    try:
        with conn.cursor() as cur:
            # 值走参数绑定，不拼进 SQL 文本
            cur.execute("SELECT set_config('app.business_line', %s, false)", (business_line,))
    except Exception as e:
        # 设置失败不吞：RLS 下这意味着后面所有查询都会返回空
        logger.warning(f"[PG-SCOPE] 设置业务线作用域失败，后续查询在 RLS 下将为空: {e}")
