"""Keep interrupted native cleanup unavailable and retryable."""

import sqlalchemy as sa
from alembic import op

revision = "f2c4e6a8b0d1"
down_revision = "e1b3d5f7a9c0"
branch_labels = None
depends_on = None


def upgrade():
    for table in ("weave_organization_bindings", "weave_repository_lifecycles"):
        op.add_column(
            table,
            sa.Column("deletion_pending", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade():
    for table in ("weave_repository_lifecycles", "weave_organization_bindings"):
        if (
            op.get_bind()
            .execute(sa.text(f"SELECT 1 FROM {table} WHERE deletion_pending LIMIT 1"))
            .first()
        ):
            raise RuntimeError("Finish pending native cleanup before downgrading")
        op.drop_column(table, "deletion_pending")
