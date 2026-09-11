"""Select current source documents without treating shared semantic nodes as proof."""

import hashlib
import json

from cognee.modules.search.types import SearchType


def native_source_tag(data_id, content_hash: str) -> str:
    return f"weave-current-source:{data_id}:{content_hash}"


def eligible_memory_sources(binding, sources, repository_ids=()):
    """A bundled source is current only when every qualified fact is current.

    Tags are immutable source/content selectors, not mutable status labels.
    Untagged historical documents remain stored but never enter this allowlist.
    """
    selected = set(repository_ids)
    eligible = []
    for source in sources:
        if (
            source.organization_id != binding.organization_id
            or (source.dataset_id or binding.dataset_id) != binding.dataset_id
            or source.status != "completed"
            or source.session_id
            or (selected and source.github_repository_id not in selected)
        ):
            continue
        qualification = source.qualification
        if not isinstance(qualification, dict):
            continue
        facts = qualification.get("facts")
        states = qualification.get("fact_states")
        if not isinstance(facts, list) or not facts or not isinstance(states, dict):
            continue
        if qualification.get("native_source_tag") != native_source_tag(
            source.data_id, source.content_hash
        ):
            continue
        for fact in facts:
            if not isinstance(fact, dict) or not fact:
                break
            digest = hashlib.sha256(json.dumps(fact, sort_keys=True).encode()).hexdigest()
            state = states.get(digest)
            if (
                not isinstance(state, dict)
                or state.get("status") != "current"
                or not isinstance(state.get("checked_sha"), str)
                or not state["checked_sha"]
            ):
                break
        else:
            eligible.append(source)
    return eligible


async def recall_current_memory(
    binding,
    user,
    query,
    sources,
    *,
    repository_ids=(),
    top_k,
    llm,
    embedding,
):
    import cognee

    eligible = eligible_memory_sources(binding, sources, repository_ids)
    # An empty node_name list disables native filtering; never send one.
    if not eligible:
        return []
    tags = sorted({source.qualification["native_source_tag"] for source in eligible})
    data_ids = {str(source.data_id) for source in eligible}
    results = await cognee.recall(
        query,
        query_type=SearchType.CHUNKS,
        scope="graph",
        auto_route=False,
        dataset_ids=[binding.dataset_id],
        node_name=tags,
        node_name_filter_operator="OR",
        user=user,
        top_k=top_k,
        llm_config=llm,
        embedding_config=embedding,
    )
    current = []
    for result in results:
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata")
        if (
            str(payload.get("dataset_id")) == str(binding.dataset_id)
            and isinstance(metadata, dict)
            and str(metadata.get("data_id")) in data_ids
        ):
            current.append(result)
    return current
