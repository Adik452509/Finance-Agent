"""Shared test fixtures. Nothing here calls a paid or rate-limited API."""

import math
import re
import zlib

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

from src.rag.retriever import Retriever

DIM = 256


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors: shared words => similar vectors.

    Good enough to test ranking and filtering logic offline, with zero quota.
    """

    def __init__(self) -> None:
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[zlib.crc32(word.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vector(text)


def _meta(ticker: str, company: str, doc_type: str, fy: str, unit: str, page: int, label: str | None) -> dict:
    meta = {"ticker": ticker, "company": company, "doc_type": doc_type,
            "fiscal_year_end": fy, "reporting_unit": unit, "page": page}
    if label:
        meta["page_label"] = label
    return meta


SAMPLE_DOCS = [
    ("aapl-p32", "Total net sales 416,161 391,035 383,285 Products Services net sales",
     _meta("AAPL", "Apple Inc.", "10-K", "2025-09-27", "USD millions", 32, "29")),
    ("aapl-p10", "Risk factors supply chain disruption competition regulation",
     _meta("AAPL", "Apple Inc.", "10-K", "2025-09-27", "USD millions", 10, "8")),
    ("ril-p99", "Borrowings 1,10,631 1,01,910 Trade Payables as at 31st March 2025",
     _meta("RELIANCE.NS", "Reliance Industries Limited", "Annual Report", "2025-03-31", "INR crore", 99, "194 195")),
    ("toy-p86", "Sales revenues 43,787,709 46,079,610 operating income year ended March 31 2026",
     _meta("7203.T", "Toyota Motor Corporation", "20-F", "2026-03-31", "JPY millions", 86, "81")),
]

# 0700.HK is "registered" but has no chunks: mirrors a company not yet ingested.
VALID_TICKERS = {"AAPL", "RELIANCE.NS", "7203.T", "0700.HK"}


def _client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path), settings=ChromaSettings(anonymized_telemetry=False))


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def sample_collection(tmp_path, fake_embeddings):
    collection = _client(tmp_path).create_collection("test_filings", metadata={"hnsw:space": "cosine"})
    ids, docs, metas = zip(*SAMPLE_DOCS)
    collection.add(ids=list(ids), documents=list(docs), metadatas=list(metas),
                   embeddings=fake_embeddings.embed_documents(list(docs)))
    return collection


@pytest.fixture
def retriever(sample_collection, fake_embeddings) -> Retriever:
    return Retriever(collection=sample_collection, embeddings=fake_embeddings, valid_tickers=VALID_TICKERS)


@pytest.fixture
def empty_retriever(tmp_path, fake_embeddings) -> Retriever:
    collection = _client(tmp_path / "empty").create_collection("empty_filings", metadata={"hnsw:space": "cosine"})
    return Retriever(collection=collection, embeddings=fake_embeddings, valid_tickers=VALID_TICKERS)
