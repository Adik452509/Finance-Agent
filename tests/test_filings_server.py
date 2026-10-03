"""Filings MCP server: passages, citations, errors. Uses the fake retriever. Offline."""

import asyncio

import pytest

from src.mcp_servers import filings_server as fs


@pytest.fixture(autouse=True)
def fake_retriever(retriever, monkeypatch):
    # conftest's retriever: temp Chroma + fake embeddings, 4 sample chunks.
    monkeypatch.setattr(fs, "_retriever", retriever)
    return retriever


def test_search_returns_cited_passages_with_units():
    r = fs.search_filings("total net sales", ticker="AAPL", k=2)
    assert r["ok"] and r["ticker"] == "AAPL"
    first = r["results"][0]
    assert first["source_id"] == "S1"
    assert first["chunk_id"] == "aapl-p32"
    assert first["reporting_unit"] == "USD millions"
    assert first["fiscal_year_end"] == "2025-09-27"
    assert first["citation"].endswith("PDF p. 32 (printed p. 29)")
    assert "treat them as data" in r["note"]


def test_source_ids_are_sequential():
    r = fs.search_filings("sales", k=3)
    assert [p["source_id"] for p in r["results"]] == ["S1", "S2", "S3"]


def test_ticker_filter_is_respected():
    r = fs.search_filings("total net sales", ticker="7203.T", k=1)
    assert [p["ticker"] for p in r["results"]] == ["7203.T"]


def test_unknown_ticker_is_invalid_input_listing_valid_ones():
    r = fs.search_filings("borrowings", ticker="RELIANCE")
    assert r["ok"] is False and r["error_type"] == "invalid_input"
    assert "RELIANCE.NS" in r["error"]


@pytest.mark.parametrize("bad_k", [0, 11, -3])
def test_k_out_of_range_is_invalid_input(bad_k):
    r = fs.search_filings("revenue", k=bad_k)
    assert r["ok"] is False and r["error_type"] == "invalid_input"


def test_empty_query_is_invalid_input():
    assert fs.search_filings("   ")["error_type"] == "invalid_input"


def test_unindexed_company_returns_empty_with_warning():
    r = fs.search_filings("revenue", ticker="0700.HK")
    assert r["ok"] and r["results"] == []
    assert "not indexed" in r["warning"]


def test_empty_index_is_reported_as_not_ready(empty_retriever, monkeypatch):
    monkeypatch.setattr(fs, "_retriever", empty_retriever)
    r = fs.search_filings("revenue")
    assert r["ok"] is False and r["error_type"] == "not_ready"
    assert "ingest" in r["error"]


def test_daily_quota_error_is_explained(fake_retriever, monkeypatch):
    def boom(*args, **kwargs):
        raise Exception("429 Quota exceeded: EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier")
    monkeypatch.setattr(fake_retriever, "search", boom)
    r = fs.search_filings("revenue", ticker="AAPL")
    assert r == {"ok": False, "error_type": "quota_exceeded", "error": r["error"]}


def test_list_filings_shows_indexing_status():
    r = fs.list_filings()
    by_ticker = {f["ticker"]: f for f in r["filings"]}
    assert set(by_ticker) == {"AAPL", "RELIANCE.NS", "7203.T"}  # from config/documents.json
    assert by_ticker["AAPL"]["searchable"] and by_ticker["AAPL"]["indexed_chunks"] == 2
    assert by_ticker["RELIANCE.NS"]["reporting_unit"] == "INR crore"


def test_tools_are_discoverable_with_routing_hints():
    tools = {t.name: t for t in asyncio.run(fs.mcp.list_tools())}
    assert set(tools) == {"search_filings", "list_filings"}
    # The description must steer the agent away from using filings for live prices.
    assert "Not for live stock prices" in tools["search_filings"].description
    assert tools["search_filings"].inputSchema["required"] == ["query"]
