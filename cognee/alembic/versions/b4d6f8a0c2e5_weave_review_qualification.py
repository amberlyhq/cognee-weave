"""Persist qualified review facts before native memory ingestion."""

import sqlalchemy as sa
from alembic import op

revision = "b4d6f8a0c2e5"
down_revision = "a3c5e7f9b1d2"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("weave_memory_sources", sa.Column("qualification", sa.JSON(), nullable=True))


def downgrade():
    if (
        op.get_bind()
        .execute(
            sa.text("SELECT 1 FROM weave_memory_sources WHERE qualification IS NOT NULL LIMIT 1")
        )
        .first()
    ):
        raise RuntimeError("Forget qualified review sources before dropping their receipts")
    op.drop_column("weave_memory_sources", "qualification")
