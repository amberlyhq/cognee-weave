"""add_weave_index_attempt_count

Revision ID: e2f4a6b8c0d3
Revises: d1e3f5a7b9c2
Create Date: 2026-08-31 18:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e2f4a6b8c0d3"
down_revision: Union[str, None] = "d1e3f5a7b9c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("weave_index_jobs")}
    if "attempt_count" not in columns:
        op.add_column(
            "weave_index_jobs",
            sa.Column("attempt_count", sa.BigInteger(), server_default="1", nullable=False),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("weave_index_jobs")}
    if "attempt_count" in columns:
        op.drop_column("weave_index_jobs", "attempt_count")
