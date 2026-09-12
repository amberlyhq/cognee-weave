"""Host-selected native notes and immutable operation receipts."""

import sqlalchemy as sa
from alembic import op

revision = "c5e7f9a1b3d6"
down_revision = "b4d6f8a0c2e5"
branch_labels = None
depends_on = None


def upgrade():
    def org():
        return sa.Column(
            "organization_id",
            sa.UUID(),
            sa.ForeignKey("weave_organization_bindings.organization_id", ondelete="CASCADE"),
            primary_key=True,
        )

    op.create_table(
        "weave_memory_notes",
        org(),
        sa.Column("github_repository_id", sa.BigInteger(), primary_key=True),
        sa.Column("note_id", sa.UUID(), primary_key=True),
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("source_sha", sa.String(40), nullable=False),
        sa.Column("source_paths", sa.JSON(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "weave_memory_jobs",
        org(),
        sa.Column("job_id", sa.UUID(), primary_key=True),
        sa.Column("github_repository_id", sa.BigInteger(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "weave_memory_operations",
        org(),
        sa.Column("operation_id", sa.UUID(), primary_key=True),
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column("github_repository_id", sa.BigInteger(), nullable=False),
        sa.Column("note_id", sa.UUID(), nullable=False),
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("receipt", sa.JSON(), nullable=True),
    )
    if op.get_bind().dialect.name == "postgresql":
        predicate = (
            "organization_id = NULLIF(current_setting('app.weave_organization_id', true), '')::uuid"
        )
        for table in ("weave_memory_notes", "weave_memory_jobs", "weave_memory_operations"):
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            op.execute(
                f"CREATE POLICY {table}_organization_isolation ON {table} USING ({predicate}) WITH CHECK ({predicate})"
            )
            op.execute(
                f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='cognee') THEN GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO cognee; END IF; END $$"
            )


def downgrade():
    if op.get_bind().execute(sa.text("SELECT 1 FROM weave_memory_jobs LIMIT 1")).first():
        raise RuntimeError("Preserve memory job history before downgrading")
    for table in ("weave_memory_operations", "weave_memory_jobs", "weave_memory_notes"):
        op.drop_table(table)
