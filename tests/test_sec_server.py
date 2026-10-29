"""SEC EDGAR MCP server with a fake SEC API (httpx.MockTransport). Offline."""

import asyncio
from datetime import date

import httpx
import pytest

from src.mcp_servers import sec_server as sec

TICKERS = {"0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
           "1": {"cik_str": 1094517, "ticker": "TM", "title": "TOYOTA MOTOR CORP/"},
           "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"}}


def fact(start, end, val, form="10-K", filed="2026-07-29", accn="0001193125-26-323660"):
    f = {"end": end, "val": val, "form": form, "filed": filed, "accn": accn}
    return {**f, "start": start} if start else f


# Shaped like Microsoft's real companyfacts (values from its 10-K filed 2026-07-29 and 10-Q filed 2026-10-28).
MSFT_FACTS = {"facts": {"us-gaap": {
    # Old years under SalesRevenueNet, newer ones under the 2018+ concept: per-period fallback.
    "SalesRevenueNet": {"units": {"USD": [fact("2016-07-01", "2017-06-30", 89950000000, filed="2017-08-02",
                                               accn="0001564590-17-014900")]}},
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        fact("2024-07-01", "2025-06-30", 281724000000, filed="2025-07-30", accn="0000950170-25-100235"),
        fact("2024-07-01", "2025-06-30", 281724000000),                      # repeated as a comparative
        fact("2025-07-01", "2026-06-30", 331839000000),
        fact("2026-07-01", "2026-09-30", 92282000000, "10-Q", "2026-10-28", "0001193125-26-436133"),
        fact("2025-07-01", "2025-09-30", 77673000000, "10-Q", "2026-10-28", "0001193125-26-436133"),
    ]}},
    "NetIncomeLoss": {"units": {"USD": [
        fact("2025-07-01", "2026-06-30", 133750000000),
        fact("2024-07-01", "2025-06-30", 101832000000),
        fact("2025-07-01", "2026-06-30", 133750000000, "DEF 14A", "2026-10-21", "0001193125-26-400000"),  # proxy
        fact("2026-07-01", "2026-09-30", 36630000000, "10-Q", "2026-10-28", "0001193125-26-436133"),
        fact("2026-01-01", "2026-06-30", 70000000000, "10-Q", "2026-07-30", "0001193125-26-300000"),  # 6-month YTD
    ]}},
    "EarningsPerShareDiluted": {"units": {"USD/shares": [fact("2025-07-01", "2026-06-30", 17.95)]}},
    "Assets": {"units": {"USD": [
        fact(None, "2026-06-30", 758376000000), fact(None, "2025-06-30", 619003000000),
        fact(None, "2026-09-30", 779403000000, "10-Q", "2026-10-28", "0001193125-26-436133"),
    ]}},
}}}

# Toyota reports in JPY under IFRS, plus one USD "convenience translation" figure.
TM_FACTS = {"facts": {"ifrs-full": {
    "Revenue": {"units": {"JPY": [fact("2024-04-01", "2025-03-31", 48036704000000, "20-F", "2025-06-18"),
                                  fact("2023-04-01", "2024-03-31", 45095325000000, "20-F", "2025-06-18")],
                          "USD": [fact("2024-04-01", "2025-03-31", 321000000000, "20-F", "2025-06-18")]}},
    "Assets": {"units": {"JPY": [fact(None, "2025-03-31", 93601350000000, "20-F", "2025-06-18"),
                                 fact(None, "2024-03-31", 90114296000000, "20-F", "2025-06-18")]}},
}}}

SUBMISSIONS = {"filings": {"recent": {
    "form": ["PX14A6G", "8-K", "10-Q", "DEF 14A", "10-K"],
    "filingDate": ["2026-10-28", "2026-10-28", "2026-10-28", "2026-10-21", "2026-07-29"],
    "reportDate": ["", "2026-10-28", "2026-09-30", "", "2026-06-30"],
    "items": ["", "2.02,9.01", "", "", ""],
    "accessionNumber": ["0000921895-26-002869", "0001193125-26-436149", "0001193125-26-436133",
                        "0001193125-26-400000", "0001193125-26-323660"],
    "primaryDocument": ["px.htm", "msft-8k.htm", "msft-20260930.htm", "proxy.htm", "msft-20260630.htm"],
}}}


class FakeSEC:
    def __init__(self, status=200):
        self.status, self.requests = status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text="error")
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "companyfacts/CIK0000789019" in url:
            return httpx.Response(200, json=MSFT_FACTS)
        if "companyfacts/CIK0001094517" in url:
            return httpx.Response(200, json=TM_FACTS)
        if "submissions/CIK0000789019" in url:
            return httpx.Response(200, json=SUBMISSIONS)
        return httpx.Response(404, text="not found")


@pytest.fixture
def fake_sec(monkeypatch):
    fake = FakeSEC()
    sec._cache.clear()
    monkeypatch.setattr(sec.settings, "sec_user_agent", "FinanceResearchAgent test@example.com")
    monkeypatch.setattr(sec, "_transport", httpx.MockTransport(fake))
    monkeypatch.setattr(sec, "MIN_REQUEST_GAP", 0)
    return fake


# --- get_sec_financials ----------------------------------------------------------------------


def test_annual_rows_are_exact_newest_first_and_cite_their_filing(fake_sec):
    r = sec.get_sec_financials("msft", "annual", 2)
    assert r["ok"] and r["company"] == "MICROSOFT CORP" and r["currency"] == "USD"
    fy26, fy25 = r["rows"]
    assert (fy26["period_start"], fy26["period_end"]) == ("2025-07-01", "2026-06-30")
    assert fy26["revenue"] == 331839000000 and fy26["revenue_display"] == "331.84 B USD"
    assert fy26["diluted_eps_display"] == "17.95 USD/shares" and fy26["total_assets"] == 758376000000
    assert fy25["net_income"] == 101832000000
    assert fy26["reported_in"] == ["10-K filed 2026-07-29"]
    assert r["filings"]["10-K filed 2026-07-29"] == "https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/"


def test_a_proxy_statement_is_never_the_source(fake_sec):
    # Seen live: DEF 14A repeats net income in its pay table and, filed later, "won".
    r = sec.get_sec_financials("MSFT", "annual", 1)
    assert all("DEF 14A" not in label for label in r["rows"][0]["reported_in"])


def test_the_latest_filing_wins_for_a_repeated_period(fake_sec):
    fy25 = sec.get_sec_financials("MSFT", "annual", 2)["rows"][1]
    assert fy25["reported_in"] == ["10-K filed 2026-07-29"]  # the comparative in the newer 10-K


def test_older_years_fall_back_to_older_concepts(fake_sec):
    rows = sec.get_sec_financials("MSFT", "annual", 5)["rows"]
    assert rows[-1]["period_end"] == "2017-06-30" and rows[-1]["revenue"] == 89950000000


def test_quarterly_uses_only_three_month_periods(fake_sec):
    r = sec.get_sec_financials("MSFT", "quarterly", 4)
    assert [row["period_end"] for row in r["rows"]] == ["2026-09-30", "2025-09-30"]  # the 6-month YTD is skipped
    latest = r["rows"][0]
    assert latest["revenue"] == 92282000000 and latest["net_income"] == 36630000000
    assert latest["total_assets"] == 779403000000 and latest["reported_in"] == ["10-Q filed 2026-10-28"]
    assert any("fourth quarter is not filed separately" in w for w in r["warnings"])


def test_quarterly_always_includes_the_same_quarter_a_year_earlier(fake_sec):
    # Seen live: asked for 3 quarters, the year-ago quarter (row 4, as Q4 is never filed alone) was missing.
    r = sec.get_sec_financials("MSFT", "quarterly", 1)
    assert [row["period_end"] for row in r["rows"]] == ["2026-09-30"]
    assert r["same_quarter_a_year_earlier"]["period_end"] == "2025-09-30"
    assert r["same_quarter_a_year_earlier"]["revenue"] == 77673000000
    assert "same_quarter_a_year_earlier" not in sec.get_sec_financials("MSFT", "annual", 1)


def test_reporting_currency_wins_over_convenience_translations(fake_sec):
    # Seen live: preferring USD labelled Toyota's 48 trillion yen as dollars.
    r = sec.get_sec_financials("TM", "annual", 2, )
    assert r["currency"] == "JPY" and r["rows"][0]["revenue"] == 48036704000000
    assert r["rows"][0]["revenue_display"] == "48.04 T JPY"


def test_known_foreign_tickers_map_to_their_us_listing(fake_sec):
    r = sec.get_sec_financials("7203.T", "annual", 1)
    assert r["ok"] and r["ticker"] == "TM" and "US listing TM" in r["warnings"][0]


def test_foreign_only_ticker_is_not_found_without_any_request(fake_sec):
    r = sec.get_sec_financials("RELIANCE.NS")
    assert r["ok"] is False and r["error_type"] == "not_found" and "market data" in r["error"]
    assert fake_sec.requests == []


def test_class_shares_use_the_secs_spelling(fake_sec):
    assert sec.sec_ticker("BRK.B") == ("BRK-B", None)


def test_unknown_us_ticker_is_not_found(fake_sec):
    r = sec.get_sec_financials("ZZZZ")
    assert r["error_type"] == "not_found" and "company list" in r["error"]


def test_stale_annual_data_is_flagged():
    r = sec.build_financials(1094517, "TM", "TOYOTA", TM_FACTS, "annual", 2, today=date(2026, 10, 29))
    assert any("newest annual figures are for the period ended 2025-03-31" in w for w in r["warnings"])
    fresh = sec.build_financials(1094517, "TM", "TOYOTA", TM_FACTS, "annual", 2, today=date(2025, 7, 1))
    assert not any("newest annual" in w for w in fresh["warnings"])


def test_user_agent_is_sent_and_results_are_cached(fake_sec):
    sec.get_sec_financials("MSFT")
    sec.get_sec_financials("MSFT", "quarterly", 2)
    assert len(fake_sec.requests) == 2  # tickers + companyfacts, both cached afterwards
    assert fake_sec.requests[0].headers["User-Agent"] == "FinanceResearchAgent test@example.com"


def test_missing_user_agent_is_not_configured(monkeypatch, fake_sec):
    monkeypatch.setattr(sec.settings, "sec_user_agent", None)
    r = sec.get_sec_financials("MSFT")
    assert r["error_type"] == "not_configured" and "SEC_USER_AGENT" in r["error"] and fake_sec.requests == []


@pytest.mark.parametrize("status, error_type", [(403, "not_configured"), (429, "rate_limited"), (500, "upstream_error")])
def test_sec_http_errors_are_mapped(monkeypatch, fake_sec, status, error_type):
    monkeypatch.setattr(sec, "_transport", httpx.MockTransport(FakeSEC(status=status)))
    r = sec.get_sec_financials("MSFT")
    assert r["ok"] is False and r["error_type"] == error_type


@pytest.mark.parametrize("kwargs", [{"period": "monthly"}, {"count": 0}, {"period": "annual", "count": 6},
                                    {"period": "quarterly", "count": 9}])
def test_bad_arguments_are_rejected(fake_sec, kwargs):
    r = sec.get_sec_financials("MSFT", **kwargs)
    assert r["error_type"] == "invalid_input" and fake_sec.requests == []


# --- get_sec_filings -------------------------------------------------------------------------


def test_recent_filings_skip_proxy_material_by_default(fake_sec):
    r = sec.get_sec_filings("MSFT")
    assert [f["form"] for f in r["filings"]] == ["8-K", "10-Q", "10-K"]
    assert r["filings"][0]["items"] == "2.02,9.01"
    assert r["filings"][1]["url"].endswith("/789019/000119312526436133/msft-20260930.htm")


def test_filings_can_be_filtered_by_form(fake_sec):
    assert [f["form"] for f in sec.get_sec_filings("MSFT", form="def 14a")["filings"]] == ["DEF 14A"]
    assert sec.get_sec_filings("MSFT", form="20-F")["error_type"] == "not_found"


def test_tools_are_discoverable():
    tools = {t.name for t in asyncio.run(sec.mcp.list_tools())}
    assert tools == {"get_sec_financials", "get_sec_filings"}
