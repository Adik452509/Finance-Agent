"""Market Data MCP server: tool behaviour and safety, with yfinance faked. Offline."""

import asyncio
import math

import pandas as pd
import pytest

from src.mcp_servers import market_server as ms


class FakeTicker:
    """Stands in for yf.Ticker; counts calls so caching can be checked."""

    calls = 0
    data: dict[str, dict] = {}

    def __init__(self, symbol: str) -> None:
        FakeTicker.calls += 1
        self._d = FakeTicker.data.get(symbol, {})

    @property
    def info(self) -> dict:
        if self._d.get("raise"):
            raise ConnectionError("Yahoo is down")
        return self._d.get("info", {})

    @property
    def quarterly_income_stmt(self) -> pd.DataFrame:
        return self._d.get("stmt", pd.DataFrame())


def _stmt(dates: list[str], revenue: list[float], net_income: list[float]) -> pd.DataFrame:
    cols = [pd.Timestamp(d) for d in dates]
    return pd.DataFrame([revenue, net_income], index=["Total Revenue", "Net Income"], columns=cols)


@pytest.fixture(autouse=True)
def fake_yahoo(monkeypatch):
    ms._cache.clear()
    FakeTicker.calls = 0
    FakeTicker.data = {
        "AAPL": {
            "info": {"longName": "Apple Inc.", "currentPrice": 333.69, "currency": "USD",
                     "financialCurrency": "USD", "marketCap": 4.9e12, "trailingPE": 41.2},
            "stmt": _stmt(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"],
                          [109.4e9, 111.2e9, 143.8e9, 102.5e9], [23.4e9, 24.8e9, 36.3e9, 27.5e9]),
        },
        # Reliance's real gap: Sep 2025 missing between Jun and Dec.
        "RELIANCE.NS": {
            "info": {"longName": "Reliance Industries Limited", "currentPrice": 1167.7,
                     "currency": "INR", "financialCurrency": "INR"},
            "stmt": _stmt(["2026-06-30", "2026-03-31", "2025-12-31", "2025-06-30"],
                          [3.09e12, 2.94e12, 2.65e12, 2.44e12], [2.7e11, 2.6e11, 2.2e11, 2.0e11]),
        },
        # Tencent's real quirks: trades in HKD, reports in CNY, blank quarters.
        "0700.HK": {
            "info": {"longName": "Tencent Holdings Limited", "currentPrice": 421.2,
                     "currency": "HKD", "financialCurrency": "CNY"},
            "stmt": _stmt(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"],
                          [204.8e9, math.nan, math.nan, math.nan], [5.5e10, math.nan, math.nan, math.nan]),
        },
        "DOWN": {"raise": True},
    }
    monkeypatch.setattr(ms.yf, "Ticker", FakeTicker)


# --- get_quote ------------------------------------------------------------------------


def test_quote_ok_with_currency_and_readable_cap():
    q = ms.get_quote("aapl ")
    assert q["ok"] and q["ticker"] == "AAPL"
    assert q["price"] == 333.69 and q["currency"] == "USD"
    assert q["market_cap_display"] == "4.90 T USD"
    assert q["warnings"] == []


def test_quote_flags_trading_vs_reporting_currency():
    q = ms.get_quote("0700.HK")
    assert q["financial_currency"] == "CNY"
    assert any("HKD" in w and "CNY" in w for w in q["warnings"])


def test_unknown_ticker_is_structured_not_found():
    q = ms.get_quote("ZZZZ")
    assert q == {"ok": False, "error_type": "not_found", "error": q["error"]}
    assert "list_supported_markets" in q["error"]


@pytest.mark.parametrize("bad", ["", "AAPL; rm -rf /", "../etc", "A" * 30, "apple inc"])
def test_invalid_ticker_rejected_before_any_network_call(bad):
    q = ms.get_quote(bad)
    assert not q["ok"] and q["error_type"] == "invalid_input"
    assert FakeTicker.calls == 0


def test_upstream_failure_becomes_structured_error():
    q = ms.get_quote("DOWN")
    assert not q["ok"] and q["error_type"] == "upstream_error"
    assert "Yahoo is down" in q["error"]


def test_successful_results_are_cached():
    ms.get_quote("AAPL")
    ms.get_quote("aapl")
    assert FakeTicker.calls == 1


def test_errors_are_not_cached():
    ms.get_quote("DOWN")
    ms.get_quote("DOWN")
    assert FakeTicker.calls == 2


# --- get_quarterly_financials -------------------------------------------------------------


def test_quarterly_ok_newest_first_with_units():
    r = ms.get_quarterly_financials("AAPL", 4)
    assert r["ok"] and r["consecutive"]
    assert [q["quarter_end"] for q in r["quarters"]] == ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]
    assert r["quarters"][0]["revenue_display"] == "109.40 B USD"
    assert "raw currency units" in r["unit"]
    assert r["warnings"] == []


def test_quarterly_flags_missing_quarter():
    r = ms.get_quarterly_financials("RELIANCE.NS", 4)
    assert r["consecutive"] is False
    assert any("2025-06-30 -> 2025-12-31" in w for w in r["warnings"])


def test_quarterly_blanks_stay_null_never_zero():
    r = ms.get_quarterly_financials("0700.HK", 4)
    assert r["currency"] == "CNY"  # reporting currency, not the HKD trading currency
    assert r["quarters"][1]["revenue"] is None and r["quarters"][1]["revenue_display"] is None
    assert any("never as zero" in w for w in r["warnings"])


def test_quarterly_respects_requested_count():
    assert len(ms.get_quarterly_financials("AAPL", 2)["quarters"]) == 2


@pytest.mark.parametrize("bad", [0, 9, -1])
def test_quarterly_rejects_out_of_range_count(bad):
    r = ms.get_quarterly_financials("AAPL", bad)
    assert not r["ok"] and r["error_type"] == "invalid_input"


# --- registration / discovery ---------------------------------------------------------------


def test_list_supported_markets_includes_suffixes_and_indexed_companies():
    r = ms.list_supported_markets()
    assert {m["suffix"] for m in r["markets"]} >= {".NS", ".T", ".HK"}
    assert {c["ticker"] for c in r["companies_with_filings"]} == {"AAPL", "RELIANCE.NS", "7203.T"}


def test_tools_are_discoverable_with_descriptions_and_schemas():
    tools = {t.name: t for t in asyncio.run(ms.mcp.list_tools())}
    assert set(tools) == {"get_quote", "get_quarterly_financials", "list_supported_markets"}
    assert "exchange suffix" in tools["get_quote"].description
    schema = tools["get_quarterly_financials"].inputSchema
    assert schema["required"] == ["ticker"]
    assert schema["properties"]["quarters"]["default"] == 4
