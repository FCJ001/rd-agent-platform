"""ai_triage_results 加 user_id / business_line

Revision ID: d5e6f7a8b9c0
Revises: c3d4e5f6a7b8
Create Date: 2026-09-17

分诊结论要成为**可复用的项目知识**，就必须能按「谁诊断的」和「属于哪条
业务线」检索 —— 这两列是把结论从个人记忆搬到知识层的前提：

  · user_id      —— 「我上次的诊断」，也是反馈接口做所有权校验的依据
  · business_line —— 「这条线上别人诊断过什么」，项目内共享的检索维度

★ 存量行留 NULL，不回填：`session_id` 是 `{user_id}:{session_id}` 结构，
  理论上能从前缀解析出归属，但那是拿字符串当归属依据 —— 正是要清理掉的
  做法。不可归属就不归属，与 ai_impact_analysis / ai_report_interpretations
  的存量快照语义保持一致（NULL = 迁移前存量，相关功能 fail-closed）。

★ 列可为空而不是 NOT NULL：customer / aftersales / admin 角色的
  UserContext.business_line 本来就是空的，且分诊常发生在建单之前拿不到
  权威归属；此时由 src/utils/business_line.py 从根因编码前缀兜底推导，
  推导不出就留空。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5e6f7a8b9c0'
down_revision: Union[str, Sequence[str], None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ai_triage_results",
        sa.Column(
            "user_id", sa.BigInteger(), nullable=True,
            comment="诊断发起人用户 ID（users.id）；NULL=迁移前存量或自动分诊",
        ),
    )
    op.add_column(
        "ai_triage_results",
        sa.Column(
            "business_line", sa.String(length=10), nullable=True,
            comment="业务线（作用域）；NULL=迁移前存量或推导不出归属",
        ),
    )
    # 按线检索历史结论是高频路径（「这条线上有没有人遇到过」）
    op.create_index("ix_triage_line", "ai_triage_results", ["business_line"])


def downgrade() -> None:
    op.drop_index("ix_triage_line", table_name="ai_triage_results")
    op.drop_column("ai_triage_results", "business_line")
    op.drop_column("ai_triage_results", "user_id")
