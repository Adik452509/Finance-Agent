"""Find the filing passages most relevant to a question, with citations.

Pipeline position:  question -> [retriever] -> top-k chunks -> LLM answers from them

    from src.rag.retriever import retrieve, format_for_prompt

    hits = retrieve("What were Apple's total net sales in fiscal 2025?", ticker="AAPL")
    prompt_block = format_for_prompt(hits)

This module is also what the Filings MCP server (Phase 2) will expose to agents,
so it validates its inputs as if an LLM were calling it - because one will.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

DEFAULT_K = 5
MAX_K = 20
MAX_QUERY_CHARS = 1000


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    score: float  # cosine similarity in [-1, 1]; higher = more relevant
    metadata: dict[str, Any]

    @property
    def ticker(self) -> str:
        return str(self.metadata.get("ticker", ""))

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0))

    def citation(self) -> str:
        """Human-readable source, e.g. 'Apple Inc. 10-K (FY ended 2025-09-27), PDF p. 32 (printed p. 29)'."""
        m = self.metadata
        where = f"PDF p. {m.get('page')}"
        if m.get("page_label"):
            where += f" (printed p. {m['page_label']})"
        return f"{m.get('company')} {m.get('doc_type')} (FY ended {m.get('fiscal_year_end')}), {where}"


class Retriever:
    """Semantic search over the filings collection.

    `collection`, `embeddings` and `valid_tickers` can be injected; the
    defaults are the real Chroma store, Gemini, and config/documents.json.
    Tests pass a temporary store and a fake embedding model instead.
    """

    def __init__(
        self,
        collection: Collection | None = None,
        embeddings: Embeddings | None = None,
        valid_tickers: set[str] | None = None,
    ) -> None:
        if collection is None:
            from src.rag.store import get_collection

            collection = get_collection()
        if embeddings is None:
            from config.llm import get_embeddings

            embeddings = get_embeddings()
        if valid_tickers is None:
            from src.ingestion.loader import load_registry

            valid_tickers = {d.ticker for d in load_registry()}

        self.collection = collection
        self.embeddings = embeddings
        self.valid_tickers = valid_tickers

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        ticker: str | list[str] | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to k chunks most similar to `query`, best first.

        Args:
            query: the question in plain language.
            k: number of chunks to return (1-20).
            ticker: restrict to one company ("AAPL") or several (["AAPL", "7203.T"]).
                The filter is applied BEFORE ranking, so you always get the best
                k chunks of that company - never zero because others ranked higher.
            min_score: drop chunks below this similarity. Returning nothing is
                better than handing the LLM irrelevant text to "answer" from.
        """
        query = self._clean_query(query)
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k must be between 1 and {MAX_K}, got {k}")
        tickers = self._resolve_tickers(ticker)

        if self.collection.count() == 0:
            raise RuntimeError("The filings index is empty. Run: python scripts/ingest.py")

        where = None
        if tickers:
            where = {"ticker": tickers[0]} if len(tickers) == 1 else {"ticker": {"$in": tickers}}

        result = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = [
            # Chroma's cosine "distance" is 1 - similarity.
            RetrievedChunk(id=id_, text=doc, score=round(1.0 - dist, 4), metadata=meta)
            for id_, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
        if min_score is not None:
            hits = [h for h in hits if h.score >= min_score]

        if not hits and tickers:
            missing = [t for t in tickers if not self.collection.get(where={"ticker": t}, limit=1)["ids"]]
            if missing:
                logger.warning("No chunks indexed yet for %s. Run: python scripts/ingest.py", ", ".join(missing))
        return hits

    def _clean_query(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query = " ".join(query.split())  # collapse newlines/tabs from pasted text
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query is too long ({len(query)} chars, max {MAX_QUERY_CHARS})")
        return query

    def _resolve_tickers(self, ticker: str | list[str] | None) -> list[str]:
        """Validate tickers case-insensitively; return them in canonical form."""
        if ticker is None:
            return []
        requested = [ticker] if isinstance(ticker, str) else list(ticker)
        canonical = {t.upper(): t for t in self.valid_tickers}
        resolved, unknown = [], []
        for t in requested:
            match = canonical.get(str(t).strip().upper())
            (resolved if match else unknown).append(match or t)
        if unknown:
            raise ValueError(
                f"Unknown ticker(s): {', '.join(map(str, unknown))}. "
                f"Valid tickers: {', '.join(sorted(self.valid_tickers))}"
            )
        return list(dict.fromkeys(resolved))  # de-duplicate, keep order


def format_for_prompt(hits: list[RetrievedChunk]) -> str:
    """Number the chunks as [S1], [S2]... with source and units, for an LLM prompt.

    Units travel with every source: "9,80,136" means nothing without
    "INR crore", and the model must not have to guess.
    """
    if not hits:
        return "No relevant passages were found in the indexed filings."
    blocks = []
    for i, hit in enumerate(hits, start=1):
        unit = hit.metadata.get("reporting_unit", "unknown units")
        blocks.append(f"[S{i}] {hit.citation()} | figures in {unit}\n{hit.text}")
    return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def _default_retriever() -> Retriever:
    return Retriever()


def retrieve(
    query: str,
    k: int = DEFAULT_K,
    ticker: str | list[str] | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Convenience wrapper using the real index and Gemini embeddings."""
    return _default_retriever().search(query, k=k, ticker=ticker, min_score=min_score)
