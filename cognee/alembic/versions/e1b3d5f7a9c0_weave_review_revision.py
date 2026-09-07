"""Order review source replacements without changing native memory semantics."""

import sqlalchemy as sa
from alembic import op

revision = "e1b3d5f7a9c0"
down_revision = "d9a1c3e5f7b0"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "weave_memory_sources", sa.Column("artifact_revision", sa.BigInteger(), nullable=True)
    )


def downgrade():
    if (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT 1 FROM weave_memory_sources WHERE artifact_revision IS NOT NULL LIMIT 1"
            )
        )
        .first()
    ):
        raise RuntimeError("Forget review sources before dropping their ordering receipts")
    op.drop_column("weave_memory_sources", "artifact_revision")
