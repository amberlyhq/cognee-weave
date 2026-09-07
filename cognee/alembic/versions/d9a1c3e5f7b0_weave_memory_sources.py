"""Tenant-scoped native source receipts. No data rewrite or graph changes."""

import sqlalchemy as sa
from alembic import op

revision = "d9a1c3e5f7b0"
down_revision = "c7e9f1a3b5d8"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "weave_memory_sources",
        sa.Column(
            "organization_id",
            sa.UUID(),
            sa.ForeignKey("weave_organization_bindings.organization_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("github_repository_id", sa.BigInteger(), primary_key=True),
        sa.Column("source_key", sa.String(1024), primary_key=True),
        sa.Column("data_id", sa.UUID(), nullable=False, unique=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    if op.get_bind().dialect.name == "postgresql":
        op.execute("ALTER TABLE weave_memory_sources ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE weave_memory_sources FORCE ROW LEVEL SECURITY")
        predicate = (
            "organization_id = NULLIF(current_setting('app.weave_organization_id', true), '')::uuid"
        )
        op.execute(
            f"CREATE POLICY weave_memory_sources_organization_isolation ON weave_memory_sources USING ({predicate}) WITH CHECK ({predicate})"
        )
        op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='cognee') THEN
            GRANT SELECT, INSERT, UPDATE, DELETE ON weave_memory_sources TO cognee;
            END IF; END $$""")


def downgrade():
    # Never erase the only source-ownership receipts while native memory exists.
    if op.get_bind().execute(sa.text("SELECT 1 FROM weave_memory_sources LIMIT 1")).first():
        raise RuntimeError("Forget tracked native sources before downgrading memory receipts")
    op.drop_table("weave_memory_sources")
