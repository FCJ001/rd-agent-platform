"""add entity_version to alm mirror tables

Revision ID: a1f2c3d4e5f6
Revises: 25ff3a21fe7e
Create Date: 2026-09-11

webhook upsert 需要 WHERE entity_version < EXCLUDED.entity_version 守卫
防乱序/旧事件覆盖新数据。存量行回填 0 —— 首个事件（v1 起）即可覆盖种子数据。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1f2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '25ff3a21fe7e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = [
    "alm_issues",
    "alm_requirements",
    "alm_change_requests",
    "alm_config_items",
    "alm_baselines",
]


def upgrade() -> None:
    for t in TABLES:
        op.add_column(
            t,
            sa.Column(
                "entity_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
                comment="平台侧版本号，乱序防护（0 = 种子/存量数据）",
            ),
        )


def downgrade() -> None:
    for t in TABLES:
        op.drop_column(t, "entity_version")
