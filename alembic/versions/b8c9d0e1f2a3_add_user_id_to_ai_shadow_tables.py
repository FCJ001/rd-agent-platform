"""add user_id to ai_impact_analysis / ai_report_interpretations

Revision ID: b8c9d0e1f2a3
Revises: a1f2c3d4e5f6
Create Date: 2026-09-11

反馈回写接口（/api/v1/impact/feedback、/api/v1/report/feedback）原先只按
session_id 过滤 —— session_id 是裸会话号、不含归属，任何已认证用户拿到
（或猜到）别人的 session_id 就能篡改其反馈记录（IDOR）。

补 user_id 列后：worker 工具写入时记录创建者，UPDATE 强制
WHERE user_id = 当前认证用户。存量行 user_id 为 NULL，不匹配任何
user_id，天然 fail-closed（谁也改不了，需要反馈就重新生成一条）。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8c9d0e1f2a3'
down_revision: Union[str, Sequence[str], None] = 'a1f2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ["ai_impact_analysis", "ai_report_interpretations"]


def upgrade() -> None:
    for t in TABLES:
        op.add_column(
            t,
            sa.Column(
                "user_id",
                sa.BigInteger(),
                nullable=True,
                comment="创建者用户 ID（users.id）；NULL=存量行，反馈接口 fail-closed",
            ),
        )


def downgrade() -> None:
    for t in TABLES:
        op.drop_column(t, "user_id")
