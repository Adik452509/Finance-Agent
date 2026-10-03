"""Embed chunks and store them in Chroma - rate-limited and resumable.

Pipeline position:  PDF  ->  loader  ->  chunker  ->  [indexer]  ->  Chroma

Why it is built this way:
  - Gemini's free tier caps embeddings per minute by count (100 texts) AND by
    tokens. One batch goes out per minute, sized to fit both limits
    (`embed_batch_size` texts, `embed_chars_per_minute` characters). A batch
    over the token budget can never succeed, however long you wait.
  - Every batch is written to Chroma immediately. If the run stops (daily
    quota, network, Ctrl+C), nothing already embedded is lost.
  - Chunk IDs are stable, so a re-run skips everything already stored and
    only spends quota on what is missing.
  - Chunks that no longer exist (e.g. after changing chunk_size) are deleted,
    so stale text can't show up in search results.
"""

import logging
import time
from dataclasses import dataclass, field

from chromadb.api.models.Collection import Collection

from config.llm import get_embeddings
from config.settings import settings
from src.ingestion.chunker import Chunk
from src.rag.store import get_collection  # noqa: F401  (re-exported for scripts/ingest.py)

logger = logging.getLogger(__name__)

RATE_WINDOW_SECONDS = 60
MAX_RATE_LIMIT_RETRIES = 3


class DailyQuotaExceeded(RuntimeError):
    """The free tier's daily embedding quota is used up; retry tomorrow."""


@dataclass
class IndexReport:
    total: int = 0
    already_indexed: int = 0
    embedded: int = 0
    removed_stale: int = 0
    stopped_reason: str | None = None
    failed_ids: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.stopped_reason is None


def remove_stale(collection: Collection, chunks: list[Chunk]) -> int:
    """Delete stored chunks of these files whose IDs are no longer produced."""
    current = {c.id for c in chunks}
    removed = 0
    for file in sorted({c.page.meta.file for c in chunks}):
        stored = collection.get(where={"file": file}, include=[])["ids"]
        stale = [i for i in stored if i not in current]
        if stale:
            collection.delete(ids=stale)
            removed += len(stale)
            logger.info("%s: removed %d stale chunks", file, len(stale))
    return removed


def make_batches(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Group chunks into per-minute batches within both free-tier limits."""
    max_texts = settings.embed_batch_size
    max_chars = settings.embed_chars_per_minute

    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_chars = 0
    for chunk in chunks:
        size = len(chunk.embedding_text)
        if current and (len(current) >= max_texts or current_chars + size > max_chars):
            batches.append(current)
            current, current_chars = [], 0
        current.append(chunk)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _is_rate_limit(error: Exception) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def _embed_with_retry(texts: list[str]) -> list[list[float]]:
    """Embed one batch, waiting out per-minute limits; give up on daily ones."""
    embeddings = get_embeddings()
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            if not _is_rate_limit(e):
                raise
            if "PerDay" in str(e):
                raise DailyQuotaExceeded(
                    "Daily embedding quota reached. Progress is saved - run the same command tomorrow."
                ) from e
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            logger.warning("Rate limited (attempt %d/%d); waiting %ds",
                           attempt, MAX_RATE_LIMIT_RETRIES, RATE_WINDOW_SECONDS)
            time.sleep(RATE_WINDOW_SECONDS)
    raise AssertionError("unreachable")


def index_chunks(chunks: list[Chunk], collection: Collection) -> IndexReport:
    """Embed and store every chunk not already in the collection."""
    report = IndexReport(total=len(chunks))
    report.removed_stale = remove_stale(collection, chunks)

    existing = set(collection.get(ids=[c.id for c in chunks], include=[])["ids"]) if chunks else set()
    todo = [c for c in chunks if c.id not in existing]
    report.already_indexed = len(chunks) - len(todo)
    if not todo:
        logger.info("Nothing to embed: all %d chunks already indexed", len(chunks))
        return report

    batches = make_batches(todo)
    logger.info("Embedding %d chunks in %d batches (~%d min, one batch per minute)",
                len(todo), len(batches), len(batches))

    for n, batch in enumerate(batches, start=1):
        started = time.monotonic()
        try:
            vectors = _embed_with_retry([c.embedding_text for c in batch])
        except DailyQuotaExceeded as e:
            report.stopped_reason = str(e)
            break
        except KeyboardInterrupt:
            report.stopped_reason = "Interrupted by user. Progress is saved - re-run to continue."
            break
        except Exception as e:
            report.stopped_reason = f"{type(e).__name__}: {str(e)[:300]}"
            report.failed_ids = [c.id for c in batch]
            break

        collection.upsert(
            ids=[c.id for c in batch],
            embeddings=vectors,
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        report.embedded += len(batch)
        logger.info("Batch %d/%d stored (%d/%d new chunks)", n, len(batches), report.embedded, len(todo))

        # Stay under the per-minute limit: one batch per RATE_WINDOW_SECONDS.
        if n < len(batches):
            wait = RATE_WINDOW_SECONDS - (time.monotonic() - started)
            if wait > 0:
                try:
                    time.sleep(wait)
                except KeyboardInterrupt:
                    report.stopped_reason = "Interrupted by user. Progress is saved - re-run to continue."
                    break

    return report
