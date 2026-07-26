# ============================================================
# 行级过滤规则
#
# ★ 4 个角色 × 一把资源，规则写成代码常量，不建 RBAC 表。
#   权限主数据属于 Java ALM 平台，本服务再存一份就是第二个真相源。
#   平台改了角色定义，这里跟着改一行 lambda，不用做数据迁移。
#
# 设计要点：规则返回的是 SQLAlchemy 条件对象，不是"过滤好的结果"。
#   条件被拼进 WHERE 交给数据库执行 —— 而不是先把 24 条全查出来再在
#   Python 里 if 一遍。后者在数据量大时既慢又危险（分页会算错，
#   而且一旦漏了一处 if，数据就已经离开数据库了）。
#
# 用法：
#   filters = build_filters(ctx, AlmIssue)
#   items, total = await repo.get_page(filters=filters)
# ============================================================

from typing import TYPE_CHECKING, Callable, Sequence

import sqlalchemy as sa
from sqlalchemy import ColumnElement

from src.modules.alm.model import AlmIssue

if TYPE_CHECKING:
    from src.core.deps import UserContext


# ----------------------------------------------------------
# 角色 → 可见行的判定
#
# | 角色       | 能看到什么              | 夹具下的行数 |
# |------------|-------------------------|-------------|
# | engineer   | 自己责任域的单          | 10（eng01） |
# | business   | 自己业务线的单          | 12（biz_ev）|
# | aftersales | 已闭环的单（可复用经验）| 5           |
# | customer   | 只有自己上报的单        | 3           |
# | admin      | 全部                    | 24          |
#
# ★ customer 用 reporter_id 而不是 source == 'customer'：
#   source 说的是"这条单是谁那类人报的"，reporter_id 说的是"是不是你报的"。
#   写成 source 会让张三看到李四的投诉 —— 数据夹具里专门留了
#   source='customer' 但 reporter_id 为空的 3 条来钉死这个区别（见 seed_issues.py）
# ----------------------------------------------------------
ISSUE_ROW_FILTERS: dict[str, Callable[["UserContext"], ColumnElement[bool]]] = {
    "engineer": lambda ctx: AlmIssue.owner_domain_id == ctx.owner_domain_id,
    "business": lambda ctx: AlmIssue.business_line == ctx.business_line,
    "aftersales": lambda ctx: AlmIssue.status.in_(("closed", "verified")),
    "customer": lambda ctx: AlmIssue.reporter_id == int(ctx.user_id),
    "admin": lambda ctx: sa.true(),
}


def build_issue_filters(ctx: "UserContext") -> Sequence[ColumnElement[bool]]:
    """
    按角色生成 alm_issues 的行级过滤条件。

    ★ 未知角色一律返回 false（什么都看不到），不是返回空条件。
      默认拒绝而不是默认放行：将来平台新增一个角色而这里忘了加规则，
      结果应该是"这个角色看不到数据"（有人来报障），
      而不是"这个角色能看到全部"（没人发现，直到出事）。
    """
    rule = ISSUE_ROW_FILTERS.get(ctx.role)
    if rule is None:
        return [sa.false()]

    # engineer / business 的维度可能为空（比如工程师没挂责任域）。
    # 这时 `col == None` 会被翻译成 `IS NULL`，反而把一批无主的单放出去了 —— 同样拒绝。
    if ctx.role == "engineer" and ctx.owner_domain_id is None:
        return [sa.false()]
    if ctx.role == "business" and not ctx.business_line:
        return [sa.false()]

    return [rule(ctx)]
