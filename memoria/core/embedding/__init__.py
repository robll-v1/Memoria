"""Embedding package — unified embedding client with pluggable providers."""

from memoria.core.embedding.client import EmbeddingClient
from memoria.core.embedding.providers import BaseEmbeddingProvider, LocalProvider, MockProvider, OpenAIProvider

__all__ = [
    "EmbeddingClient",
    "BaseEmbeddingProvider",
    "LocalProvider",
    "MockProvider",
    "OpenAIProvider",
    "get_embedding_client",
    "set_embedding_client",
]

# Process-wide singleton — created once, reused everywhere.
_shared_client: EmbeddingClient | None = None


def set_embedding_client(client: EmbeddingClient) -> None:
    """Pre-inject a shared EmbeddingClient (e.g. from Memoria config).

    Must be called before any get_embedding_client() call.
    """
    global _shared_client
    _shared_client = client


def get_embedding_client() -> EmbeddingClient:
    """Get or create the process-wide EmbeddingClient singleton.

    Configured from application settings. Fails fast if the configured
    provider is unavailable (e.g., missing API key).
    """
    global _shared_client
    if _shared_client is None:
        import logging

        from memoria.config import get_settings
        from memoria.core.embedding.client import KNOWN_DIMENSIONS

        s = get_settings()
        dim = s.embedding_dim
        if dim == 0:
            dim = KNOWN_DIMENSIONS.get(s.embedding_model, 1024)
        _shared_client = EmbeddingClient(
            provider=s.embedding_provider,
            model=s.embedding_model,
            dim=dim,
            api_key=s.embedding_api_key,
            base_url=s.embedding_base_url,
        )
        logging.getLogger(__name__).info(
            "EmbeddingClient: provider=%s, model=%s, dim=%d",
            s.embedding_provider, s.embedding_model, dim,
        )
    return _shared_client
