"""add_weave_lifecycle_generations

Revision ID: b6d8f0a2c4e7
Revises: a5c7e9b1d3f6
Create Date: 2026-09-04 22:10:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b6d8f0a2c4e7"
down_revision: Union[str, None] = "a5c7e9b1d3f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = (
    "organization_id = NULLIF(current_setting('app.weave_organization_id', true), '')::uuid"
)


def upgrade() -> None:
    op.add_column(
        "weave_organization_bindings",
        sa.Column("lifecycle_generation", sa.BigInteger(), server_default="0", nullable=False),
    )
    op.create_table(
        "weave_repository_lifecycles",
        sa.Column("organization_id", sa.UUID(), nullable=False),
        sa.Column("github_repository_id", sa.BigInteger(), nullable=False),
        sa.Column("lifecycle_generation", sa.BigInteger(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["weave_organization_bindings.organization_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("organization_id", "github_repository_id"),
    )
    if op.get_bind().dialect.name == "postgresql":
        table = "weave_repository_lifecycles"
        policy = f"{table}_organization_isolation"
        op.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')
        op.execute(f'ALTER TABLE "{table}" FORCE ROW LEVEL SECURITY')
        op.execute(
            f'CREATE POLICY "{policy}" ON "{table}" USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})'
        )
        op.execute(
            "DO $grant$ BEGIN "
            "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cognee') THEN "
            "GRANT SELECT, INSERT, UPDATE, DELETE ON weave_repository_lifecycles TO cognee; "
            "END IF; END $grant$"
        )


def downgrade() -> None:
    op.drop_table("weave_repository_lifecycles")
    op.drop_column("weave_organization_bindings", "lifecycle_generation")
