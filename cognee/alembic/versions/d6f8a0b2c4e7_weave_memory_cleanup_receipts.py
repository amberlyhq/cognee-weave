"""Native memory cleanup receipts, separate from accepted note batches."""

import sqlalchemy as sa
from alembic import op

revision = "d6f8a0b2c4e7"
down_revision = "c5e7f9a1b3d6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "weave_memory_cleanups",
        sa.Column(
            "organization_id",
            sa.UUID(),
            sa.ForeignKey("weave_organization_bindings.organization_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("job_id", sa.UUID(), primary_key=True),
        sa.Column("mode", sa.String(16), primary_key=True),
        sa.Column("github_repository_id", sa.BigInteger(), nullable=False),
        sa.Column("lifecycle_generation", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("receipt", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    if op.get_bind().dialect.name == "postgresql":
        predicate = (
            "organization_id = NULLIF(current_setting('app.weave_organization_id', true), '')::uuid"
        )
        op.execute("ALTER TABLE weave_memory_cleanups ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE weave_memory_cleanups FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY weave_memory_cleanups_organization_isolation ON weave_memory_cleanups USING ({predicate}) WITH CHECK ({predicate})"
        )
        op.execute(
            "DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='cognee') THEN GRANT SELECT, INSERT, UPDATE, DELETE ON weave_memory_cleanups TO cognee; END IF; END $$"
        )


def downgrade():
    if op.get_bind().execute(sa.text("SELECT 1 FROM weave_memory_cleanups LIMIT 1")).first():
        raise RuntimeError("Preserve memory cleanup history before downgrading")
    op.drop_table("weave_memory_cleanups")
