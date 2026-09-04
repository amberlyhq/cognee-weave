"""add_weave_organization_control_plane

Revision ID: d1e3f5a7b9c2
Revises: c7e2a9b4d1f3
Create Date: 2026-08-31 18:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d1e3f5a7b9c2"
down_revision: Union[str, None] = "c7e2a9b4d1f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_RLS_TABLES = (
    "weave_organization_bindings",
    "weave_repository_snapshots",
    "weave_index_jobs",
)


def _enable_organization_rls(table_name: str) -> None:
    organization = (
        "organization_id = NULLIF(current_setting('app.weave_organization_id', true), '')::uuid"
    )
    op.execute(f'ALTER TABLE "{table_name}" ENABLE ROW LEVEL SECURITY')
    op.execute(f'ALTER TABLE "{table_name}" FORCE ROW LEVEL SECURITY')
    op.execute(
        f'CREATE POLICY "{table_name}_organization_isolation" ON "{table_name}" '
        f"USING ({organization}) WITH CHECK ({organization})"
    )


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "weave_organization_bindings" not in tables:
        op.create_table(
            "weave_organization_bindings",
            sa.Column("organization_id", sa.UUID(), nullable=False),
            sa.Column("tenant_id", sa.UUID(), nullable=False),
            sa.Column("service_user_id", sa.UUID(), nullable=False),
            sa.Column("primary_dataset_id", sa.UUID(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="RESTRICT"),
            sa.ForeignKeyConstraint(["service_user_id"], ["users.id"], ondelete="RESTRICT"),
            sa.ForeignKeyConstraint(["primary_dataset_id"], ["datasets.id"], ondelete="RESTRICT"),
            sa.PrimaryKeyConstraint("organization_id"),
            sa.UniqueConstraint("tenant_id"),
            sa.UniqueConstraint("service_user_id"),
            sa.UniqueConstraint("primary_dataset_id"),
        )

    if "weave_repository_snapshots" not in tables:
        op.create_table(
            "weave_repository_snapshots",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("organization_id", sa.UUID(), nullable=False),
            sa.Column("github_repository_id", sa.BigInteger(), nullable=False),
            sa.Column("repository_owner", sa.String(length=255), nullable=False),
            sa.Column("repository_name", sa.String(length=255), nullable=False),
            sa.Column("default_branch", sa.String(length=255), nullable=False),
            sa.Column("requested_sha", sa.String(length=40), nullable=True),
            sa.Column("indexed_sha", sa.String(length=40), nullable=True),
            sa.Column("pipeline_version", sa.String(length=64), nullable=True),
            sa.Column("extraction_version", sa.String(length=64), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("error_code", sa.String(length=128), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(
                ["organization_id"],
                ["weave_organization_bindings.organization_id"],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "organization_id",
                "github_repository_id",
                name="uq_weave_repository_org_github_id",
            ),
        )
        op.create_index(
            "ix_weave_repository_org_status",
            "weave_repository_snapshots",
            ["organization_id", "status"],
        )

    if "weave_index_jobs" not in tables:
        op.create_table(
            "weave_index_jobs",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("organization_id", sa.UUID(), nullable=False),
            sa.Column("github_repository_id", sa.BigInteger(), nullable=False),
            sa.Column("requested_sha", sa.String(length=40), nullable=False),
            sa.Column("indexed_sha", sa.String(length=40), nullable=True),
            sa.Column("pipeline_version", sa.String(length=64), nullable=False),
            sa.Column("extraction_version", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("error_code", sa.String(length=128), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(
                ["organization_id"],
                ["weave_organization_bindings.organization_id"],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "organization_id",
                "github_repository_id",
                "requested_sha",
                "pipeline_version",
                "extraction_version",
                name="uq_weave_index_job_identity",
            ),
        )
        op.create_index(
            "ix_weave_index_job_org_status",
            "weave_index_jobs",
            ["organization_id", "status"],
        )

    if conn.dialect.name == "postgresql":
        for table_name in _RLS_TABLES:
            _enable_organization_rls(table_name)


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "weave_index_jobs" in tables:
        op.drop_table("weave_index_jobs")
    if "weave_repository_snapshots" in tables:
        op.drop_table("weave_repository_snapshots")
    if "weave_organization_bindings" in tables:
        op.drop_table("weave_organization_bindings")
