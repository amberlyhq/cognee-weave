import os
from collections.abc import Mapping

from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig


_STRICT_RUNTIME_VALUES = {
    "WEAVE_STRICT_MODE": "true",
    "ENABLE_BACKEND_ACCESS_CONTROL": "true",
    "DB_PROVIDER": "postgres",
    "VECTOR_DB_PROVIDER": "pgvector",
    "VECTOR_DATASET_DATABASE_HANDLER": "pgvector_shared",
    "GRAPH_DATABASE_PROVIDER": "postgres_demo",
    "GRAPH_DATASET_DATABASE_HANDLER": "postgres_graph_shared",
}

_POSTGRES_CONNECTION_SUFFIXES = ("HOST", "PORT", "USERNAME", "PASSWORD", "NAME")


def validate_weave_runtime_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    """Fail startup when the dedicated Weave service loses its tenant boundary."""

    values = environment if environment is not None else os.environ
    for name, expected in _STRICT_RUNTIME_VALUES.items():
        if values.get(name) != expected:
            raise ValueError(f"{name} must be {expected} in strict Weave mode")
    for suffix in _POSTGRES_CONNECTION_SUFFIXES:
        relational_name = f"DB_{suffix}"
        relational_value = values.get(relational_name, "")
        if not relational_value:
            raise ValueError(f"{relational_name} is required in strict Weave mode")
        for prefix in ("VECTOR_DB", "GRAPH_DATABASE"):
            provider_name = f"{prefix}_{suffix}"
            if values.get(provider_name) != relational_value:
                raise ValueError(
                    f"{provider_name} must match {relational_name} in strict Weave mode"
                )
    if len(values.get("WEAVE_INTERNAL_TOKEN", "")) < 32:
        raise ValueError("WEAVE_INTERNAL_TOKEN must contain at least 32 characters")


def get_internal_token() -> str:
    """Return the private Amberly-to-Weave bearer token.

    The token is deliberately loaded per request so rotations do not require a
    module reload. Empty tokens are never accepted.
    """

    return os.getenv("WEAVE_INTERNAL_TOKEN", "")


def get_weave_embedding_config() -> EmbeddingConfig:
    """Use a local embedding model by default; no paid API key is required."""

    return EmbeddingConfig(
        embedding_provider=os.getenv("WEAVE_EMBEDDING_PROVIDER", "fastembed"),
        embedding_model=os.getenv("WEAVE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_dimensions=int(os.getenv("WEAVE_EMBEDDING_DIMENSIONS", "384")),
        embedding_api_key=os.getenv("WEAVE_EMBEDDING_API_KEY") or None,
    )
