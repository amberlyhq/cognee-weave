"""Source identities around Cognee's own directory resolver, not a new loader."""

import hashlib
import json
from pathlib import Path

from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories


async def prepare_repository_sources(repository, request, *, user, dataset_id):
    # This is the disposable validated archive, not a user's checkout. Enola's
    # repository identity uses its basename, which must not change with each SHA.
    stable = (
        repository.parent
        / f"{request.repository_owner}-{request.repository_name}-{request.github_repository_id}"
    )
    if repository != stable:
        repository = repository.rename(stable)
    resolved = await resolve_data_directories(str(repository), user=user, dataset_id=dataset_id)
    sources = []
    for value in resolved:
        if isinstance(value, DataItem):
            manifest = json.loads(value.data)
            key, fingerprint, relative, item = "code", manifest["content_hash"], ".", value
        else:
            path = Path(value)
            relative = path.relative_to(repository).as_posix()
            key = "doc:" + hashlib.sha256(relative.encode()).hexdigest()
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            item = DataItem(data=str(path))
        item.external_metadata = {
            "repository": f"{request.repository_owner}/{request.repository_name}",
            "github_repository_id": request.github_repository_id,
            "source_path": relative,
            "memory_kind": "default_branch_source",
        }
        sources.append((key, item, fingerprint))
    return sources
