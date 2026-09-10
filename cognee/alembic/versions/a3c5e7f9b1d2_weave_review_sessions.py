"""Track incremental review sessions and permit owned code dataset cleanup."""

import importlib

from alembic import op
import sqlalchemy as sa

revision = "a3c5e7f9b1d2"
down_revision = "f2c4e6a8b0d1"
branch_labels = None
depends_on = None


def _native_deletion():
    return importlib.import_module(
        "cognee.alembic.versions.c7e9f1a3b5d8_native_weave_dataset_deletion"
    )


def upgrade():
    op.add_column("weave_memory_sources", sa.Column("dataset_id", sa.UUID(), nullable=True))
    op.add_column("weave_memory_sources", sa.Column("session_id", sa.String(255), nullable=True))
    if op.get_bind().dialect.name == "postgresql":
        original = _native_deletion()
        op.execute(original.DROP_NATIVE_SCHEMA_FUNCTION.replace("-[0-9]+$", "-[0-9]+(-code-v1)?$"))
        # Native SQL session cache tables are created by the migrator, never by
        # granting the application role arbitrary CREATE on public.
        from cognee.infrastructure.databases.cache.sql.tables import cache_metadata

        cache_metadata.create_all(op.get_bind(), checkfirst=True)
        for table in cache_metadata.sorted_tables:
            op.execute(
                sa.text(f"""DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cognee') THEN
                GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE "{table.name}" TO cognee;
                END IF; END $$;""")
            )


def downgrade():
    if (
        op.get_bind()
        .execute(sa.text("SELECT 1 FROM weave_memory_sources WHERE session_id IS NOT NULL LIMIT 1"))
        .first()
    ):
        raise RuntimeError("Remove review-session memory before downgrading")
    if (
        op.get_bind()
        .execute(sa.text("SELECT 1 FROM datasets WHERE name LIKE '%-code-v1' LIMIT 1"))
        .first()
    ):
        raise RuntimeError("Remove code datasets before downgrading")
    if op.get_bind().dialect.name == "postgresql":
        op.execute(_native_deletion().DROP_NATIVE_SCHEMA_FUNCTION)
    op.drop_column("weave_memory_sources", "session_id")
    op.drop_column("weave_memory_sources", "dataset_id")
