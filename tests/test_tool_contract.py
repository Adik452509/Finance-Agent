"""Every MCP tool on every server follows the same result contract.

success -> {"ok": True, ...}; failure -> {"ok": False, "error_type": <known type>, "error": str}.
Agents rely on this to handle any server the same way. Offline.
"""

import asyncio

import pytest

from src.mcp_servers import calculator_server, filings_server, market_server, news_server, sec_server
from src.mcp_servers.common import ERROR_TYPES

SERVERS = [market_server, filings_server, calculator_server, news_server, sec_server]

# One deliberately bad call per tool. Validation must catch it before any network/API use.
BAD_CALLS = [
    (market_server.get_quote, {"ticker": "!!"}),
    (market_server.get_quarterly_financials, {"ticker": "AAPL", "quarters": 0}),
    (market_server.get_annual_financials, {"ticker": "AAPL", "years": 0}),
    (filings_server.search_filings, {"query": "   "}),
    (calculator_server.percent_change, {"old_value": 0, "new_value": 1}),
    (calculator_server.cagr, {"start_value": -1, "end_value": 1, "years": 1}),
    (calculator_server.ratio, {"numerator": 1, "denominator": 0}),
    (calculator_server.share_of_total, {"part": 1, "total": 0}),
    (calculator_server.convert_units, {"value": 1, "from_unit": "x", "to_unit": "raw"}),
    (calculator_server.series_change, {"points": []}),
    (news_server.search_news, {"query": ""}),
    (sec_server.get_sec_financials, {"ticker": "MSFT", "period": "monthly"}),
    (sec_server.get_sec_filings, {"ticker": "MSFT", "count": 0}),
]


@pytest.fixture(autouse=True)
def offline(retriever, monkeypatch):
    monkeypatch.setattr(filings_server, "_retriever", retriever)  # fake index from conftest


@pytest.mark.parametrize("tool, kwargs", BAD_CALLS, ids=[f.__name__ for f, _ in BAD_CALLS])
def test_bad_input_returns_standard_error(tool, kwargs):
    r = tool(**kwargs)
    assert r["ok"] is False
    assert r["error_type"] in ERROR_TYPES
    assert isinstance(r["error"], str) and r["error"]


def test_every_tool_has_a_bad_input_case():
    covered = {f.__name__ for f, _ in BAD_CALLS}
    no_args = {"list_supported_markets", "list_filings"}  # nothing to validate
    all_tools = {t.name for s in SERVERS for t in asyncio.run(s.mcp.list_tools())}
    assert all_tools - no_args <= covered, f"add a BAD_CALLS case for: {all_tools - no_args - covered}"


def test_tool_names_are_unique_across_servers():
    names = [t.name for s in SERVERS for t in asyncio.run(s.mcp.list_tools())]
    assert len(names) == len(set(names))


def test_every_tool_description_says_when_to_use_it():
    for server in SERVERS:
        for tool in asyncio.run(server.mcp.list_tools()):
            assert len(tool.description) >= 60, f"{tool.name}: description too short to route on"
