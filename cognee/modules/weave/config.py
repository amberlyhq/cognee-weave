import os

from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig


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
