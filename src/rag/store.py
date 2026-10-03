"""The one place that opens the vector store.

Both the indexer (writes) and the retriever (reads) go through here, so
swapping Chroma for another vector database means changing this file only.
"""

import logging

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from config.settings import settings

logger = logging.getLogger(__name__)


def get_collection(reset: bool = False) -> Collection:
    """Open (or create) the Chroma collection, guarding against model mixing."""
    settings.ensure_dirs()
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        # Chroma sends anonymous usage telemetry by default; keep it local.
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
            logger.info("Deleted collection '%s'", settings.chroma_collection)
        except Exception:  # didn't exist yet
            pass

    collection = client.get_or_create_collection(
        settings.chroma_collection,
        metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
    )
    check_embedding_model(collection)
    return collection


def check_embedding_model(collection: Collection) -> None:
    """Refuse a collection built with a different embedding model."""
    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model != settings.embedding_model:
        raise RuntimeError(
            f"Collection '{collection.name}' was built with {stored_model!r}, "
            f"but settings use {settings.embedding_model!r}. Vectors from different models "
            "can't be compared. Re-run ingestion with --reset."
        )
