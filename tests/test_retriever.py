"""Retriever behaviour: ranking, filtering, guardrails, citations. Offline."""

import logging

import pytest

from src.rag.retriever import RetrievedChunk, format_for_prompt


def test_most_relevant_chunk_ranks_first(retriever):
    hits = retriever.search("total net sales", k=3)
    assert hits[0].id == "aapl-p32"


def test_scores_are_sorted_best_first(retriever):
    scores = [h.score for h in retriever.search("net sales revenues", k=4)]
    assert scores == sorted(scores, reverse=True)


def test_ticker_filter_is_applied_before_ranking(retriever):
    # Apple's chunk matches "net sales" best overall, but filtering to Toyota
    # must still return Toyota's chunk - not an empty list.
    hits = retriever.search("total net sales", k=1, ticker="7203.T")
    assert [h.ticker for h in hits] == ["7203.T"]


def test_multiple_tickers(retriever):
    hits = retriever.search("sales", k=4, ticker=["AAPL", "7203.T"])
    assert {h.ticker for h in hits} <= {"AAPL", "7203.T"}
    assert {h.ticker for h in hits} == {"AAPL", "7203.T"}


def test_ticker_is_case_insensitive(retriever):
    assert retriever.search("borrowings", k=1, ticker="reliance.ns")[0].ticker == "RELIANCE.NS"


def test_unknown_ticker_lists_valid_ones(retriever):
    # An LLM agent may well pass "RELIANCE" instead of "RELIANCE.NS".
    with pytest.raises(ValueError, match="RELIANCE.NS"):
        retriever.search("borrowings", ticker="RELIANCE")


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t"])
def test_empty_query_rejected(retriever, bad_query):
    with pytest.raises(ValueError, match="non-empty"):
        retriever.search(bad_query)


def test_overlong_query_rejected(retriever):
    with pytest.raises(ValueError, match="too long"):
        retriever.search("revenue " * 300)


@pytest.mark.parametrize("bad_k", [0, -1, 21])
def test_k_out_of_range_rejected(retriever, bad_k):
    with pytest.raises(ValueError, match="k must be"):
        retriever.search("revenue", k=bad_k)


def test_validation_happens_before_spending_an_embedding(retriever, fake_embeddings):
    with pytest.raises(ValueError):
        retriever.search("revenue", ticker="NOPE")
    assert fake_embeddings.query_calls == 0


def test_min_score_returns_nothing_rather_than_junk(retriever):
    assert retriever.search("photosynthesis chlorophyll", k=4, min_score=0.5) == []


def test_unindexed_ticker_returns_empty_and_warns(retriever, caplog):
    with caplog.at_level(logging.WARNING):
        hits = retriever.search("revenue", ticker="0700.HK")
    assert hits == []
    assert "0700.HK" in caplog.text and "ingest" in caplog.text


def test_empty_index_raises_with_instructions(empty_retriever):
    with pytest.raises(RuntimeError, match="ingest"):
        empty_retriever.search("revenue")


def test_citation_shows_pdf_and_printed_page(retriever):
    hit = retriever.search("borrowings trade payables", k=1, ticker="RELIANCE.NS")[0]
    assert hit.citation() == (
        "Reliance Industries Limited Annual Report (FY ended 2025-03-31), PDF p. 99 (printed p. 194 195)"
    )


def test_prompt_block_numbers_sources_and_carries_units(retriever):
    block = format_for_prompt(retriever.search("total net sales", k=2, ticker="AAPL"))
    assert block.startswith("[S1] Apple Inc. 10-K")
    assert "[S2]" in block
    assert "figures in USD millions" in block


def test_prompt_block_for_no_hits_says_so():
    assert "No relevant passages" in format_for_prompt([])


def test_citation_without_page_label():
    hit = RetrievedChunk(id="x", text="t", score=0.9, metadata={
        "company": "Apple Inc.", "doc_type": "10-K", "fiscal_year_end": "2025-09-27", "page": 3})
    assert hit.citation().endswith("PDF p. 3")
