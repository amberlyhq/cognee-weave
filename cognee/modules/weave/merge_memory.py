"""Resume bounded, source-verified learning after native default-branch indexing."""

import hashlib
import json
from uuid import uuid5

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods import get_user
from cognee.modules.weave.archive import validated_archive
from cognee.modules.weave.config import get_weave_embedding_config, get_weave_llm_config
from cognee.modules.weave.contracts import MergeMemoryResponse
from cognee.modules.weave.indexing import weave_operation_lock
from cognee.modules.weave.knowledge_lifecycle import fact_digest
from cognee.modules.weave.memory_sources import (
    source_data_id,
    source_records,
    sync_source,
)
from cognee.modules.weave.memory_retrieval import native_source_tag
from cognee.modules.weave.models import WeaveMemorySource, WeaveRepositoryLifecycle
from cognee.modules.weave.native_memory import customer_dataset, customer_snapshots
from cognee.modules.weave.organizations import (
    get_organization_binding,
    set_weave_organization_scope,
)
from cognee.modules.weave.scope import native_organization
from cognee.tasks.ingestion.data_item import DataItem


def merge_readiness(request, lifecycle, snapshot):
    if (
        not lifecycle
        or not lifecycle.active
        or lifecycle.deletion_pending
        or lifecycle.lifecycle_generation != request.lifecycle_generation
    ):
        return "stale_ignored"
    if snapshot is None:
        return "pending"
    if (
        snapshot.repository_owner,
        snapshot.repository_name,
        snapshot.default_branch,
    ) != (request.repository_owner, request.repository_name, request.default_branch):
        raise ValueError("Repository identity mismatch")
    if snapshot.requested_sha != request.requested_sha:
        return "stale_ignored"
    if snapshot.status != "indexed" or snapshot.indexed_sha != request.requested_sha:
        return "pending"
    return None


def build_batches(repository, changed_paths, prior_facts):
    by_path = {}
    for fact in prior_facts:
        by_path.setdefault(fact["code_path"], []).append(fact)
    batches, current, size = [], {"paths": [], "prior_facts": []}, 0
    for path in sorted(set(changed_paths) | set(by_path)):
        target = repository / path
        path_size = min(target.stat().st_size, 28001) if target.is_file() else 200
        facts = by_path.get(path, [])
        for start in range(0, max(1, len(facts)), 12):
            portion = facts[start : start + 12]
            if current["paths"] and (
                size + path_size > 28000
                or len(current["prior_facts"]) + len(portion) > 12
                or len(current["paths"]) >= 12
            ):
                batches.append(current)
                current, size = {"paths": [], "prior_facts": []}, 0
            current["paths"].append(path)
            current["prior_facts"].extend(portion)
            size += path_size
    if current["paths"]:
        batches.append(current)
    return batches


async def save_receipt(
    binding, repository_id, key, qualification, *, status="qualified", content_hash=None
):
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        identity = (binding.organization_id, repository_id, key)
        record = await session.get(WeaveMemorySource, identity)
        if record is None:
            record = WeaveMemorySource(
                organization_id=binding.organization_id,
                github_repository_id=repository_id,
                source_key=key,
                data_id=source_data_id(*identity),
                dataset_id=binding.dataset_id,
                content_hash=content_hash or "0" * 64,
                status=status,
            )
            session.add(record)
        if record.dataset_id != binding.dataset_id or record.data_id != source_data_id(
            *identity
        ):
            raise ValueError("Foreign merge receipt")
        record.qualification = qualification
        record.status = status
        if content_hash is not None:
            record.content_hash = content_hash
        await session.commit()


async def apply_decisions(
    binding, repository_id, decisions, prior_facts, head_sha, replacement_source
):
    by_id = {fact["fact_id"]: fact for fact in prior_facts}
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        for decision in decisions:
            fact = by_id[decision["fact_id"]]
            record = await session.get(
                WeaveMemorySource,
                (binding.organization_id, repository_id, fact["source_key"]),
            )
            if record is None or record.dataset_id != binding.dataset_id:
                raise ValueError("Missing or foreign prior fact receipt")
            receipt = {**record.qualification}
            states = {**receipt.get("fact_states", {})}
            state = {**states.get(fact["digest"], {})}
            # Keep creates an updated, source-quoted copy. Historical originals
            # retain their identity and point to the verified replacement source.
            state.update(
                status=(
                    "unverified" if decision["status"] == "unverified" else "superseded"
                ),
                checked_sha=head_sha,
                reason=decision["reason"],
            )
            if decision["status"] != "unverified":
                state["replacement_source"] = replacement_source
            states[fact["digest"]] = state
            record.qualification = {**receipt, "fact_states": states}
        await session.commit()


async def process_merge(
    binding, user, request, change, repository, snapshot, *, llm, embedding
):
    repository_id = request.github_repository_id
    # A -> B -> A is a new indexing epoch. Redelivery within that epoch reuses
    # the manifest, including the exact qualified payload after a provider error.
    epoch = hashlib.sha256(
        f"{change.base_sha}:{change.head_sha}:{snapshot.updated_at.isoformat()}".encode()
    ).hexdigest()[:24]
    prefix = f"review:merge:{change.head_sha}:{epoch}"
    records = []
    for record in await source_records(binding, repository_id):
        if (
            record.organization_id != binding.organization_id
            or record.github_repository_id != repository_id
            or record.data_id
            != source_data_id(binding.organization_id, repository_id, record.source_key)
        ):
            raise ValueError("Foreign merge receipt identity")
        # Older unqualified review receipts can predate dataset stamping. They
        # supply no facts and must not be adopted, re-homed, or used as evidence.
        # Keep their history untouched while maintaining qualified memory.
        if (
            record.dataset_id is None
            and record.qualification is None
            and record.source_key.startswith("review:")
            and not record.source_key.startswith(
                ("review:qualified:", "review:merge:", "review:code-change:")
            )
        ):
            continue
        if record.dataset_id != binding.dataset_id:
            raise ValueError("Foreign merge receipt identity")
        records.append(record)
    by_key = {r.source_key: r for r in records}
    manifest = by_key.get(prefix + ":plan")
    if manifest and manifest.status == "completed":
        return {**manifest.qualification["result"], "status": "unchanged"}
    if manifest:
        plan = manifest.qualification
    else:
        priors = []
        for record in records:
            if not record.qualification or record.status != "completed":
                continue
            if record.dataset_id != binding.dataset_id:
                raise ValueError("Foreign merge source")
            for fact in record.qualification.get("facts", []):
                digest = fact_digest(fact)
                state = record.qualification.get("fact_states", {}).get(digest, {})
                if state.get("status") == "superseded":
                    continue
                if (
                    fact["code_path"] in change.changed_paths
                    or state.get("status") == "needs_recheck"
                ):
                    fact_id = str(
                        uuid5(
                            binding.organization_id,
                            f"review-fact.v1:{repository_id}:{record.source_key}:{digest}",
                        )
                    )
                    priors.append(
                        {
                            **fact,
                            "fact_id": fact_id,
                            "source_key": record.source_key,
                            "digest": digest,
                        }
                    )
        plan = {
            "facts": [],
            "batches": build_batches(repository, change.changed_paths, priors),
            "review_context": change.review_context[:6000],
            "base_sha": change.base_sha,
            "head_sha": change.head_sha,
        }
        await save_receipt(binding, repository_id, prefix + ":plan", plan)
    processed = 0
    counts = dict(
        facts_saved=0,
        facts_rechecked=0,
        facts_superseded=0,
        facts_unverified=0,
        coverage_complete=True,
    )
    for index, batch in enumerate(plan["batches"]):
        key = f"{prefix}:{index}"
        existing = by_key.get(key)
        if (existing is None or existing.status != "completed") and processed >= 4:
            from cognee.modules.weave.review_code_links import sync_review_code_links

            await sync_review_code_links(binding, repository_id)
            return {"status": "pending", **counts, "coverage_complete": False}
        if existing is None or existing.status != "completed":
            processed += 1
        receipt = existing.qualification if existing else None
        if receipt is None:
            files, deleted, unreadable = {}, [], []
            for path in batch["paths"]:
                target = repository / path
                if not target.exists():
                    deleted.append(path)
                elif target.is_file() and not target.is_symlink():
                    try:
                        with target.open("r", encoding="utf-8") as handle:
                            files[path] = handle.read(28001)
                    except UnicodeError:
                        unreadable.append(path)
                else:
                    unreadable.append(path)
            from cognee.modules.weave.merge_qualification import qualify_merge

            # Binary sources have no text evidence. They remain visibly unverified.
            if unreadable:
                qualified = {
                    "facts": [],
                    "decisions": [
                        {
                            "fact_id": f["fact_id"],
                            "status": "unverified",
                            "reason": "No readable text evidence",
                        }
                        for f in batch["prior_facts"]
                    ],
                    "coverage_complete": False,
                }
            else:
                qualified = await qualify_merge(
                    head_sha=change.head_sha,
                    files=files,
                    prior_facts=batch["prior_facts"],
                    review_context=plan["review_context"],
                    deleted_paths=deleted,
                )
            receipt = {
                **qualified,
                "version": "merge-knowledge.v1",
                "fact_states": {
                    fact_digest(f): {
                        "status": "current",
                        "checked_sha": change.head_sha,
                    }
                    for f in qualified["facts"]
                },
            }
            content = json.dumps(
                {
                    "kind": "Verified merged code knowledge",
                    "head_sha": change.head_sha,
                    "facts": [
                        {
                            **{
                                k: f[k] for k in ("statement", "code_path", "certainty")
                            },
                            "evidence_ids": [e["evidence_id"] for e in f["evidence"]],
                        }
                        for f in receipt["facts"]
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            receipt["native_source_tag"] = native_source_tag(
                source_data_id(binding.organization_id, repository_id, key),
                content_hash,
            )
            await save_receipt(
                binding, repository_id, key, receipt, content_hash=content_hash
            )
        content = json.dumps(
            {
                "kind": "Verified merged code knowledge",
                "head_sha": change.head_sha,
                "facts": [
                    {
                        **{k: f[k] for k in ("statement", "code_path", "certainty")},
                        "evidence_ids": [e["evidence_id"] for e in f["evidence"]],
                    }
                    for f in receipt["facts"]
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if receipt["facts"]:
            item = DataItem(
                data=content,
                label=f"merge-{change.head_sha}-{index}.txt",
                external_metadata={
                    "memory_kind": "qualified_merge",
                    "merge_source": key.removeprefix("review:merge:"),
                    "github_repository_id": repository_id,
                    "head_sha": change.head_sha,
                },
            )
            await sync_source(
                binding,
                user,
                repository_id,
                key,
                item,
                content_hash,
                llm=llm,
                embedding=embedding,
                self_improvement=False,
                node_set=[receipt["native_source_tag"]],
            )
        else:
            await save_receipt(
                binding,
                repository_id,
                key,
                receipt,
                status="completed",
                content_hash=content_hash,
            )
        await apply_decisions(
            binding,
            repository_id,
            receipt["decisions"],
            batch["prior_facts"],
            change.head_sha,
            key,
        )
        counts["facts_saved"] += len(receipt["facts"])
        counts["facts_rechecked"] += len(receipt["decisions"])
        counts["facts_superseded"] += sum(
            d["status"] == "superseded" for d in receipt["decisions"]
        )
        counts["facts_unverified"] += sum(
            d["status"] == "unverified" for d in receipt["decisions"]
        )
        counts["coverage_complete"] &= receipt["coverage_complete"]
    from cognee.modules.weave.review_code_links import sync_review_code_links

    await sync_review_code_links(binding, repository_id)
    result = {"status": "completed", **counts}
    await save_receipt(
        binding,
        repository_id,
        prefix + ":plan",
        {**plan, "result": result},
        status="completed",
    )
    return result


async def verify_indexed_archive(binding, repository_id, repository):
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph = await get_graph_engine()
    nodes, _ = await graph.get_filtered_graph_data(
        [{"type": ["CodeFileReference"]}], max_edges=0
    )
    expected = {
        p["file_path"]: p.get("weave_content_hash")
        for _, p in nodes
        if str(p.get("organization_id")) == str(binding.organization_id)
        and p.get("github_repository_id") == repository_id
    }
    if not expected or any(not digest for digest in expected.values()):
        raise LookupError("Repository needs file-hash indexing before merge learning")
    actual = {
        path.relative_to(repository)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in repository.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise ValueError("Merge archive differs from the exact indexed source")


async def remember_merge(request, change, archive_path):
    if change.head_sha != request.requested_sha:
        raise ValueError("Merge head does not match the archive request")
    base = dict(
        organization_id=request.organization_id,
        github_repository_id=request.github_repository_id,
        head_sha=request.requested_sha,
    )
    async with weave_operation_lock(request.organization_id, wait=False) as acquired:
        if not acquired:
            return MergeMemoryResponse(**base, status="pending")
        binding = await get_organization_binding(request.organization_id)
        if binding is None:
            raise LookupError("Customer is not ready")
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, request.organization_id)
            lifecycle = await session.get(
                WeaveRepositoryLifecycle,
                (request.organization_id, request.github_repository_id),
            )
        snapshots = await customer_snapshots(request.organization_id)
        snapshot = next(
            (
                s
                for s in snapshots
                if s.github_repository_id == request.github_repository_id
            ),
            None,
        )
        readiness = merge_readiness(request, lifecycle, snapshot)
        if readiness:
            return MergeMemoryResponse(**base, status=readiness)
        changes = next(
            (
                r
                for r in await source_records(binding, request.github_repository_id)
                if r.source_key == f"review:code-change:{request.requested_sha}"
            ),
            None,
        )
        if (
            changes
            and changes.dataset_id == binding.dataset_id
            and (changes.qualification or {}).get("complete")
        ):
            change = change.model_copy(
                update={
                    "changed_paths": sorted(
                        set(change.changed_paths)
                        | set(changes.qualification["changed_paths"])
                    ),
                    "complete": True,
                }
            )
        if not change.complete:
            raise ValueError(
                "Incomplete change list requires a file-hash baseline index"
            )
        if change.base_sha == "0" * 40:
            return MergeMemoryResponse(
                **base, status="unchanged", coverage_complete=True
            )
        dataset = await customer_dataset(binding)
        if dataset is None:
            raise LookupError("Customer dataset is not ready")
        user = await get_user(binding.service_user_id)
        llm, embedding = get_weave_llm_config(), get_weave_embedding_config()
        token = native_organization.set(request.organization_id)
        try:
            with validated_archive(archive_path) as repository:
                async with scoped_database_context_variables(
                    dataset.id, user.id, llm_config=llm, embedding_config=embedding
                ):
                    await verify_indexed_archive(
                        binding, request.github_repository_id, repository
                    )
                    result = await process_merge(
                        binding,
                        user,
                        request,
                        change,
                        repository,
                        snapshot,
                        llm=llm,
                        embedding=embedding,
                    )
        finally:
            native_organization.reset(token)
        return MergeMemoryResponse(**base, **result)
