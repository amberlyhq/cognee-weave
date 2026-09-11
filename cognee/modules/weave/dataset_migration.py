"""Explicit, atomic consolidation of one customer's legacy native datasets.

No inference or native forget is involved. The operation moves stored graph,
vector and relational state together; any unsupported state aborts the entire
transaction. Callers cannot supply a source or destination dataset identity.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import MetaData, Table, delete, inspect, select, text
from sqlalchemy.dialects.postgresql import insert

from cognee.infrastructure.databases.postgres import dataset_schema_name
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.provenance.source_refs import (
    get_data_id_from_source_ref_key,
    get_dataset_id_from_source_ref_key,
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.modules.data.models import Dataset
from cognee.modules.users.models import DatasetDatabase
from cognee.modules.weave.indexing import weave_operation_lock
from cognee.modules.weave.models import WeaveOrganizationBinding, WeaveRepositorySnapshot
from cognee.modules.weave.organizations import set_weave_organization_scope

_PROVENANCE = ("source_ref_keys", "source_dataset_ids", "source_run_ids", "source_run_refs")
_GRAPH_TABLES = {"graph_node", "graph_edge", "graph_metadata"}
_PUBLIC_MOVES = {
    "data",
    "nodes",
    "edges",
    "pipeline_runs",
    "weave_memory_sources",
    "session_records",
    "queries",
    "results",
}
_PUBLIC_RETIRED = {"dataset_database", "dataset_configurations", "acls"}


def _union(left, right):
    return list(dict.fromkeys([*(left or []), *(right or [])]))


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _source_ref(value, source, target):
    if get_dataset_id_from_source_ref_key(value) != source:
        raise ValueError("Foreign dataset in migration provenance")
    return make_source_ref_key(target, get_data_id_from_source_ref_key(value))


def _rewrite_properties(value, source, target):
    """Rewrite only known identity fields, never text or evidence quotations."""
    if isinstance(value, list):
        return [_rewrite_properties(v, source, target) for v in value]
    if not isinstance(value, dict):
        return value
    output = {}
    for key, child in value.items():
        if key == "source_ref_keys":
            output[key] = [_source_ref(v, source, target) for v in child or []]
        elif key == "source_run_refs":
            output[key] = [
                make_source_run_ref(
                    get_pipeline_run_id_from_source_run_ref(v),
                    _source_ref(get_source_ref_key_from_source_run_ref(v), source, target),
                )
                for v in child or []
            ]
        elif key == "source_dataset_ids":
            if any(UUID(v) != source for v in child or []):
                raise ValueError("Foreign dataset in migration provenance")
            output[key] = [str(target)] if child else []
        elif key == "dataset_id" and child == str(source):
            output[key] = str(target)
        else:
            output[key] = _rewrite_properties(child, source, target)
    return output


def _merge_properties(primary, incoming, source):
    """Keep primary semantics and retain differing historical properties."""
    primary, incoming = dict(primary or {}), dict(incoming or {})
    if primary == incoming:
        return primary
    for key in _PROVENANCE + ("belongs_to_set",):
        if key in primary or key in incoming:
            primary[key] = _union(primary.get(key), incoming.get(key))
    differences = {k: v for k, v in incoming.items() if k not in primary or primary[k] != v}
    for key, value in incoming.items():
        primary.setdefault(key, value)
    if differences:
        versions = dict(primary.get("_weave_migration_properties") or {})
        versions[str(source)] = differences
        primary["_weave_migration_properties"] = versions
    return primary


async def _reflect(connection, schema, name):
    # Register pgvector reflection without creating an embedding adapter.
    from pgvector.sqlalchemy import Vector  # noqa: F401

    return await connection.run_sync(
        lambda sync: Table(name, MetaData(), schema=schema, autoload_with=sync)
    )


def _compatible(left, right):
    def signature(table):
        return [(c.name, str(c.type), c.nullable, c.primary_key) for c in table.columns]

    if signature(left) != signature(right):
        raise ValueError(f"Incompatible migration table {left.name}")


async def _merge_table(
    session, source_table, target_table, source_id, target_id, organization_id, repository_id
):
    _compatible(source_table, target_table)
    keys = [c.name for c in source_table.primary_key]
    if not keys:
        raise ValueError(f"Migration table has no primary key: {source_table.name}")
    existing = {
        tuple(r[k] for k in keys): dict(r)
        for r in (await session.execute(select(target_table))).mappings()
    }
    incoming = [dict(r) for r in (await session.execute(select(source_table))).mappings()]
    merged = []
    for row in incoming:
        identity = tuple(row[k] for k in keys)
        row = _rewrite_properties(row, source_id, target_id)
        if source_table.name == "graph_node":
            from cognee.modules.retrieval.code_retriever import CODE_NODE_TYPES

            if row["type"] in {*CODE_NODE_TYPES, "CodeRepository"}:
                properties = dict(row["properties"] or {})
                if properties.get("organization_id") not in (None, str(organization_id)) or (
                    properties.get("github_repository_id") not in (None, repository_id)
                ):
                    raise ValueError("Foreign code ownership during dataset migration")
                row["properties"] = {
                    **properties,
                    "organization_id": str(organization_id),
                    "github_repository_id": repository_id,
                }
        primary = existing.get(identity)
        if primary is not None:
            if source_table.name == "graph_metadata":
                if row != primary:
                    raise ValueError("Incompatible graph metadata during migration")
            elif source_table.name == "graph_node":
                if (row["type"], row["name"]) != (primary["type"], primary["name"]):
                    raise ValueError("Conflicting graph node identity during migration")
                if (
                    row["type"] not in {"Entity", "EntityType"}
                    and row["properties"] != primary["properties"]
                ):
                    raise ValueError("Conflicting non-entity graph node during migration")
            if "properties" in row:
                row["properties"] = _merge_properties(
                    primary["properties"], row["properties"], source_id
                )
            if "payload" in row:
                row["payload"] = _merge_properties(primary["payload"], row["payload"], source_id)
                # Existing vectors describe the retained primary payload. Keep
                # an exact historical copy when the original vector differs.
                old, new = list(primary["vector"]), list(row["vector"])
                if old != new:
                    versions = dict(row["payload"].get("_weave_migration_vectors") or {})
                    versions[str(source_id)] = [float(v) for v in new]
                    row["payload"]["_weave_migration_vectors"] = versions
                row["vector"] = primary["vector"]
            for field in _PROVENANCE:
                if field in row:
                    row[field] = _union(primary[field], row[field])
            for field in ("created_at", "updated_at"):
                if field in row:
                    row[field] = primary[field]
        merged.append(row)
    for start in range(0, len(merged), 200):
        batch = merged[start : start + 200]
        statement = insert(target_table)
        statement = statement.on_conflict_do_update(
            index_elements=keys,
            set_={
                c.name: statement.excluded[c.name]
                for c in target_table.columns
                if c.name not in keys
            },
        )
        await session.execute(statement, batch)
    # Verify exact stored content, not just row counts, before retiring a source.
    landed = {
        tuple(r[k] for k in keys): _plain(dict(r))
        for r in (await session.execute(select(target_table))).mappings()
    }
    if any(landed.get(tuple(row[k] for k in keys)) != _plain(row) for row in merged):
        raise RuntimeError("Migration readback did not preserve stored content")
    return len(merged)


async def _move_relational(session, connection, binding, source_id, target_id):
    tables = await connection.run_sync(lambda sync: inspect(sync).get_table_names(schema="public"))
    for name in tables:
        columns = await connection.run_sync(
            lambda sync: inspect(sync).get_columns(name, schema="public")
        )
        if "dataset_id" not in {column["name"] for column in columns}:
            continue
        table = await _reflect(connection, "public", name)
        rows = [
            dict(r)
            for r in (
                await session.execute(select(table).where(table.c.dataset_id == source_id))
            ).mappings()
        ]
        if not rows:
            continue
        if name not in _PUBLIC_MOVES | _PUBLIC_RETIRED:
            raise ValueError(f"Unsupported dataset reference in {name}")
        for row in rows:
            if name == "data" and (
                row["owner_id"] != binding.service_user_id or row["tenant_id"] != binding.tenant_id
            ):
                raise ValueError("Unverifiable data ownership during migration")
            for owner in ("owner_id", "user_id"):
                if row.get(owner) is not None and row[owner] != binding.service_user_id:
                    raise ValueError(f"Foreign ownership in {name}")
            if row.get("tenant_id") is not None and row["tenant_id"] != binding.tenant_id:
                raise ValueError(f"Foreign tenant ownership in {name}")
            if (
                row.get("organization_id") is not None
                and row["organization_id"] != binding.organization_id
            ):
                raise ValueError(f"Foreign organization ownership in {name}")
        if name in _PUBLIC_MOVES:
            await session.execute(
                table.update().where(table.c.dataset_id == source_id).values(dataset_id=target_id)
            )
        elif name == "dataset_configurations":
            target = (
                (await session.execute(select(table).where(table.c.dataset_id == target_id)))
                .mappings()
                .first()
            )
            for row in rows:
                if any(
                    row[key] != (target[key] if target else None)
                    for key in ("graph_schema", "custom_prompt")
                ):
                    raise ValueError("Conflicting dataset configuration during migration")
        elif name == "acls":
            allowed = {
                (r.principal_id, r.permission_id)
                for r in (
                    await session.execute(select(table).where(table.c.dataset_id == target_id))
                ).all()
            }
            if any((r["principal_id"], r["permission_id"]) not in allowed for r in rows):
                raise ValueError("Migration would change dataset access permissions")


async def _retire_dataset(session, binding, source_id):
    """Drop only a database-verified legacy schema; never native forget shared IDs."""
    await session.execute(
        text("SELECT public.weave_drop_native_dataset_schema(:org, :dataset)"),
        {"org": binding.organization_id, "dataset": source_id},
    )
    await session.execute(
        delete(Dataset).where(
            Dataset.id == source_id,
            Dataset.owner_id == binding.service_user_id,
            Dataset.tenant_id == binding.tenant_id,
        )
    )


async def migrate_customer_datasets(organization_id: UUID, *, dry_run: bool = False) -> dict:
    """Consolidate owned legacy repositories; dry-run validates then rolls back all writes.

    Dry-run also exercises scoped source retirement inside the same transaction,
    so permissions and foreign keys are checked without persistent changes.
    """
    organization_id = UUID(str(organization_id))
    schemas = []
    result = {
        "organization_id": str(organization_id),
        "migrated_datasets": [],
        "rows": {},
        "dry_run": dry_run,
    }
    async with weave_operation_lock(organization_id):
        async with get_relational_engine().get_async_session() as session:
            if session.get_bind().dialect.name != "postgresql":
                raise ValueError("Dataset migration requires Postgres")
            await set_weave_organization_scope(session, organization_id)
            binding = await session.scalar(
                select(WeaveOrganizationBinding)
                .where(
                    WeaveOrganizationBinding.organization_id == organization_id,
                    WeaveOrganizationBinding.deleted_at.is_(None),
                    WeaveOrganizationBinding.deletion_pending.is_(False),
                )
                .with_for_update()
            )
            if binding is None:
                raise ValueError("Active customer binding not found")
            target_id = binding.primary_dataset_id
            target = await session.get(Dataset, target_id)
            if target is None or (target.owner_id, target.tenant_id) != (
                binding.service_user_id,
                binding.tenant_id,
            ):
                raise ValueError("Primary dataset ownership mismatch")
            repositories = list(
                await session.scalars(
                    select(WeaveRepositorySnapshot.github_repository_id).where(
                        WeaveRepositorySnapshot.organization_id == organization_id
                    )
                )
            )
            names = [
                f"weave-memory-{organization_id.hex}-{repo}{suffix}"
                for repo in repositories
                for suffix in ("", "-code-v1")
            ]
            sources = (
                list(
                    await session.scalars(
                        select(Dataset)
                        .where(Dataset.name.in_(names))
                        .order_by(Dataset.name)
                        .with_for_update()
                    )
                )
                if names
                else []
            )
            owned_legacy_names = list(
                await session.scalars(
                    select(Dataset.name).where(
                        Dataset.owner_id == binding.service_user_id,
                        Dataset.tenant_id == binding.tenant_id,
                        Dataset.name.startswith(f"weave-memory-{organization_id.hex}-"),
                    )
                )
            )
            if set(owned_legacy_names) - set(names):
                raise ValueError("Unregistered legacy dataset requires ownership reconciliation")
            connection = await session.connection()
            target_schema = dataset_schema_name(target_id)
            quote = connection.dialect.identifier_preparer.quote
            primary_database = await session.get(DatasetDatabase, target_id)
            for dataset in [target, *sources]:
                if (dataset.owner_id, dataset.tenant_id) != (
                    binding.service_user_id,
                    binding.tenant_id,
                ):
                    raise ValueError("Legacy dataset ownership mismatch")
                database = await session.get(DatasetDatabase, dataset.id)
                expected = dataset_schema_name(dataset.id)
                if (
                    database is None
                    or database.owner_id != binding.service_user_id
                    or (
                        database.graph_dataset_database_handler != "postgres_graph_shared"
                        or database.vector_dataset_database_handler != "pgvector_shared"
                        or database.graph_database_connection_info.get("graph_database_schema")
                        != expected
                        or database.vector_database_connection_info.get("schema") != expected
                    )
                ):
                    raise ValueError("Dataset database ownership or schema mismatch")
                if database.migration_revision != primary_database.migration_revision:
                    raise ValueError("Dataset migration revision mismatch")
            for dataset in sources:
                source_schema = dataset_schema_name(dataset.id)
                names = await connection.run_sync(
                    lambda sync: inspect(sync).get_table_names(schema=source_schema)
                )
                if not _GRAPH_TABLES.issubset(names):
                    raise ValueError("Legacy graph schema is incomplete")
                stats = {}
                # Nodes precede edges to satisfy native graph foreign keys.
                ordered = [
                    "graph_node",
                    "graph_edge",
                    "graph_metadata",
                    *sorted(set(names) - _GRAPH_TABLES),
                ]
                for name in ordered:
                    source_table = await _reflect(connection, source_schema, name)
                    if name not in _GRAPH_TABLES and set(source_table.c.keys()) != {
                        "id",
                        "payload",
                        "vector",
                    }:
                        raise ValueError(f"Unsupported dataset table {name}")
                    exists = await connection.run_sync(
                        lambda sync: inspect(sync).has_table(name, schema=target_schema)
                    )
                    if not exists:
                        await session.execute(
                            text(
                                f"CREATE TABLE {quote(target_schema)}.{quote(name)} (LIKE {quote(source_schema)}.{quote(name)} INCLUDING ALL)"
                            )
                        )
                    target_table = await _reflect(connection, target_schema, name)
                    repository_id = int(
                        dataset.name.removeprefix(
                            f"weave-memory-{organization_id.hex}-"
                        ).removesuffix("-code-v1")
                    )
                    stats[name] = await _merge_table(
                        session,
                        source_table,
                        target_table,
                        dataset.id,
                        target_id,
                        organization_id,
                        repository_id,
                    )
                await _move_relational(session, connection, binding, dataset.id, target_id)
                await _retire_dataset(session, binding, dataset.id)
                result["migrated_datasets"].append(str(dataset.id))
                result["rows"][str(dataset.id)] = stats
                schemas.append(source_schema)
            if dry_run:
                await session.rollback()
            else:
                await session.commit()
        if schemas and not dry_run:
            from cognee.infrastructure.databases.graph.get_graph_engine import graph_engine_cache
            from cognee.infrastructure.databases.vector.create_vector_engine import (
                vector_engine_cache,
            )
            from cognee.modules.retrieval.code_retriever import invalidate_code_graph_snapshot_cache

            for schema in [*schemas, target_schema]:
                graph_engine_cache.evict_matching(graph_database_schema=schema)
                vector_engine_cache.evict_matching(vector_db_schema=schema)
            invalidate_code_graph_snapshot_cache(dataset_id=target_id)
    return result
