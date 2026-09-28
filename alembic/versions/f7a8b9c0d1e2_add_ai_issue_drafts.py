"""add_ai_issue_drafts

建单 HITL 落地：AI 只出草稿（ai_issue_drafts），用户确认后才向 ALM 提交。
状态机 pending → submitted / expired（惰性过期，无定时任务）。

Revision ID: f7a8b9c0d1e2
Revises: d5e6f7a8b9c0
Create Date: 2026-09-26 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7a8b9c0d1e2'
down_revision: Union[str, Sequence[str], None] = 'd5e6f7a8b9c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('ai_issue_drafts',
    sa.Column('user_id', sa.BigInteger(), nullable=True, comment='创建者用户 ID（users.id）'),
    sa.Column('session_id', sa.String(length=100), nullable=False, comment='会话 ID'),
    sa.Column('title', sa.String(length=200), nullable=False, comment='问题标题'),
    sa.Column('description', sa.Text(), nullable=True, comment='问题描述（故障现象、DTC 码等）'),
    sa.Column('severity', sa.String(length=20), nullable=False, comment='严重度：blocker/critical/normal/minor'),
    sa.Column('business_line', sa.String(length=10), nullable=False, comment='业务线（作用域）'),
    sa.Column('source', sa.String(length=30), nullable=True, comment='来源（创建者角色）'),
    sa.Column('owner_domain_id', sa.BigInteger(), nullable=True, comment='责任域 ID'),
    sa.Column('status', sa.String(length=20), nullable=False, comment='pending/submitted/expired'),
    sa.Column('issue_no', sa.String(length=50), nullable=True, comment='提交成功后的平台问题单号'),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False, comment='创建时间'),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False, comment='更新时间'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_draft_user', 'ai_issue_drafts', ['user_id'], unique=False)
    op.create_index('ix_draft_status', 'ai_issue_drafts', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_draft_status', table_name='ai_issue_drafts')
    op.drop_index('ix_draft_user', table_name='ai_issue_drafts')
    op.drop_table('ai_issue_drafts')
