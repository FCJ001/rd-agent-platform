"""phenomena / dtc_codes 唯一键改复合（business_line + name/code）

Revision ID: c3d4e5f6a7b8
Revises: b8c9d0e1f2a3
Create Date: 2026-09-17

现象名与 DTC 码是**跨业务线共享的词汇表**：实测 26 个现象名里 13 个同时被
ev 和 ia 的根因指向，11 个 DTC 里 U0155 也在两条线都出现。原唯一键是单列
（name / code），导致两条线的同名词表项无法并存 —— PG 侧表现为同一作用域
只能存一行、business_line 被后写入者覆盖；Neo4j 侧同因（已同步改为
(business_line, name/code) 复合 MERGE 键，见 scripts/init_neo4j.py）。

本迁移只动约束、不动数据。存量被覆盖归属的行由
scripts/repair_line_split.py 按 cause_phenomena → root_causes.business_line
反查后拆分。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b8c9d0e1f2a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (表, 原单列唯一键的列, 复合键列序, 复合约束名)
TARGETS = [
    ("phenomena", "name", ["business_line", "name"], "uq_phenomena_line_name"),
    ("dtc_codes", "code", ["business_line", "code"], "uq_dtc_line_code"),
]


def _drop_single_column_unique(table: str, column: str) -> None:
    """按实际名称删除「恰好只覆盖该列」的唯一约束/唯一索引。

    初始迁移用的是未命名 sa.UniqueConstraint，PG 自动命名为 {table}_{col}_key；
    这里按列集合反查而不是硬编码名字，避免命名差异导致漏删。
    """
    bind = op.get_bind()
    insp = sa.inspect(bind)

    for uc in insp.get_unique_constraints(table):
        if list(uc["column_names"]) == [column]:
            op.drop_constraint(uc["name"], table, type_="unique")

    # 兜底：某些环境上唯一性是靠唯一索引实现的
    for ix in insp.get_indexes(table):
        if ix.get("unique") and list(ix["column_names"]) == [column]:
            op.drop_index(ix["name"], table_name=table)


def upgrade() -> None:
    for table, column, composite_cols, constraint_name in TARGETS:
        _drop_single_column_unique(table, column)
        op.create_unique_constraint(constraint_name, table, composite_cols)


def downgrade() -> None:
    """回退会重新施加单列唯一。

    ★ 如果数据已经被 repair_line_split.py 拆成多行（同名跨线各一行），
      重新加单列唯一会失败 —— 这是预期的：回退前必须先合并数据，
      否则等于把「跨线共享词汇表」这个事实重新抹掉。
    """
    for table, column, composite_cols, constraint_name in TARGETS:
        op.drop_constraint(constraint_name, table, type_="unique")
        op.create_unique_constraint(f"{table}_{column}_key", table, [column])
