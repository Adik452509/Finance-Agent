"""Company snapshot: formatting, the strength / watch rules, gathering from tools, and the API. Offline."""

import asyncio
import json
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from src.analysis import snapshot as snap
from src.api import app as api
from src.api.team import Team
from src.db.models import Base
from src.db.session import make_engine


def year(fye, revenue, net_income, operating=None, fcf=None, debt=None, equity=None):
    return {"fiscal_year_end": fye, "revenue": revenue, "net_income": net_income, "operating_income": operating,
            "free_cash_flow": fcf, "operating_cash_flow": None, "cash_and_equivalents": None,
            "total_debt": debt, "shareholders_equity": equity}


GROWING = {"ok": True, "currency": "USD", "source": "Yahoo Finance", "as_of": "2026-11-10T08:00:00+00:00", "years": [
    year("2026-06-30", 300e9, 100e9, 130e9, 60e9, 40e9, 300e9),   # newest first, as Yahoo gives them
    year("2025-06-30", 250e9, 80e9, 110e9, 55e9, 45e9, 260e9),
    year("2024-06-30", 200e9, 70e9, 90e9, 50e9, 50e9, 220e9),
]}
QUOTE = {"ok": True, "name": "Example Corp", "price": 410.5, "currency": "USD", "market_cap": 3.05e12,
         "trailing_pe": 31.2, "as_of": "2026-11-10T08:00:00+00:00"}


def titles(items):
    return [i["title"] for i in items]


# --- formatting ------------------------------------------------------------------------------------


def test_rupees_are_shown_in_crore_with_indian_grouping():
    assert snap.money(Decimal("10572190000000"), "INR") == "₹10,57,219 crore"
    assert snap.money(-5e9, "INR") == "₹-500 crore"
    assert snap.money(331.84e9, "USD") == "331.84 B USD"
    assert snap.money(3.05e12, "USD") == "3.05 T USD"
    assert snap.money(None, "USD") is None


def test_growth_maths():
    assert snap.pct_change(Decimal(200), Decimal(250)) == 25
    assert snap.pct_change(Decimal(-100), Decimal(-50)) == 50       # a smaller loss is an improvement
    assert snap.pct_change(Decimal(0), Decimal(5)) is None
    assert round(snap.cagr(Decimal(100), Decimal(121), 2), 6) == 10
    assert snap.cagr(Decimal(-1), Decimal(5), 2) is None             # undefined across a sign change
    assert snap.fy_label("2025-03-31") == "FY2025"


# --- the rules ------------------------------------------------------------------------------------


def test_a_growing_profitable_low_debt_company():
    s = snap.build_snapshot("EXM", QUOTE, GROWING)
    assert [y["label"] for y in s["years"]] == ["FY2024", "FY2025", "FY2026"]  # oldest first, for charts
    assert titles(s["strengths"]) == ["Revenue is growing", "Steady multi-year growth", "Profit is growing",
                                      "High profitability", "Generates cash", "Low debt"]
    assert s["watch"] == []
    assert s["strengths"][0]["detail"].startswith("FY2026 (year ended 2026-06-30): 300.00 B USD, 20.0% more")
    assert s["growth"]["revenue_cagr"] == pytest.approx(22.47, abs=0.01)
    assert [k["value"] for k in s["kpis"]] == ["300.00 B USD", "33.3%", "60.00 B USD", "0.13×"]
    assert s["price"]["market_cap"] == "3.05 T USD" and s["price"]["pe"] == 31.2
    assert "31 years" in s["valuation"][0]
    assert all(i["rule"] and i["source"].startswith("Yahoo Finance annual statements") for i in s["strengths"])


def test_falls_losses_cash_burn_debt_and_a_high_pe_are_flagged():
    annual = {**GROWING, "years": [year("2026-03-31", 90e9, -4e9, -2e9, -1e9, 200e9, 100e9),
                                   year("2025-03-31", 100e9, 3e9, 5e9, 2e9, 150e9, 110e9)]}
    s = snap.build_snapshot("LOSS", {**QUOTE, "trailing_pe": 55}, annual)
    assert titles(s["watch"]) == ["Revenue fell", "The company made a loss", "Spends more cash than it makes",
                                  "High debt", "Priced for high expectations"]
    assert "10.0% lower than the year before" in s["watch"][0]["detail"]
    assert s["watch"][-1]["source"].startswith("Yahoo Finance, as of")      # the price rule cites the quote
    assert s["strengths"] == [] and s["growth"]["revenue_cagr"] is None   # two years: no multi-year average


def test_thin_margins_negative_equity_and_no_profit_pe():
    annual = {**GROWING, "years": [year("2026-12-31", 100e9, 2e9, 3e9, None, 50e9, -5e9),
                                   year("2025-12-31", 100e9, 3e9, 4e9, None, 50e9, 10e9)]}
    s = snap.build_snapshot("THIN", {**QUOTE, "trailing_pe": -12}, annual)
    assert titles(s["watch"]) == ["Profit fell", "Thin profit margin", "Negative shareholders' equity"]
    assert s["valuation"] == ["No P/E: the company did not make a profit over the last year."]


def test_a_currency_mismatch_is_explained_and_rupee_figures_read_in_crore():
    annual = {**GROWING, "currency": "INR", "years": [year("2025-03-31", 9.8e12, 7e11), year("2024-03-31", 9e12, 6.9e11)]}
    s = snap.build_snapshot("RELIANCE.NS", {**QUOTE, "currency": "USD"}, annual)
    assert s["kpis"][0]["value"] == "₹9,80,000 crore"
    assert "Shares trade in USD while the statements are in INR." in s["valuation"]


def test_no_statements_is_an_error_not_an_empty_page():
    with pytest.raises(ValueError, match="No annual"):
        snap.build_snapshot("NONE", QUOTE, {"ok": True, "years": []})


def test_quote_pages_are_not_news():
    news = {"results": [{"title": "Reliance Industries Limited (RIGD.IL) stock price, news, quote and history",
                         "url": "https://uk.finance.yahoo.com/quote/RIGD.IL", "source": "uk.finance.yahoo.com"},
                        {"title": "Reliance Industries Limited", "url": "https://tavily.com/x", "source": "tavily.com"},
                        {"title": "Reliance posts record results", "url": "https://news.example/a", "source": "news.example"}]}
    s = snap.build_snapshot("EXM", QUOTE, GROWING, news)
    assert [n["title"] for n in s["news"]] == ["Reliance posts record results"]


def test_the_snapshot_never_gives_a_verdict():
    s = snap.build_snapshot("EXM", QUOTE, GROWING)
    text = json.dumps(s).lower()
    for verdict in ("you should buy", "you should sell", "strong buy", "we recommend", "undervalued"):
        assert verdict not in text
    assert "not investment advice" in s["disclaimer"]


# --- gathering from the tools ---------------------------------------------------------------------


class FakeTool:
    def __init__(self, name, reply):
        self.name, self.reply, self.calls = name, reply, []

    async def ainvoke(self, args):
        self.calls.append(args)
        if isinstance(self.reply, Exception):
            raise self.reply
        return json.dumps(self.reply)


def tools(**overrides):
    replies = {"get_quote": QUOTE, "get_annual_financials": GROWING,
               "search_news": {"ok": True, "results": [{"title": "Example Corp beats estimates", "source": "news.com",
                                                        "url": "https://news.com/a", "published_date": "2026-11-09"}]},
               "search_filings": {"ok": True, "results": [{"source_id": "EXM p12", "citation": "Example Corp 10-K, p. 12"}]}}
    replies.update(overrides)
    return {name: FakeTool(name, reply) for name, reply in replies.items()}


def test_fetch_gathers_price_statements_news_and_risk_factors():
    box = tools()
    s = asyncio.run(snap.fetch_snapshot(box, "EXM", {"EXM"}))
    assert box["get_annual_financials"].calls == [{"ticker": "EXM", "years": 5}]
    assert box["search_news"].calls[0]["query"] == "Example Corp company news"
    assert s["news"][0]["title"] == "Example Corp beats estimates"
    assert s["risk_factors"] == {"citation": "Example Corp 10-K, p. 12", "source_id": "EXM p12"}


def test_a_company_outside_the_library_skips_the_filings_and_a_news_failure_is_harmless():
    box = tools(search_news=RuntimeError("Tavily down"))
    s = asyncio.run(snap.fetch_snapshot(box, "EXM", {"AAPL"}))
    assert box["search_filings"].calls == [] and s["risk_factors"] is None and s["news"] == []


def test_missing_statements_raise_lookup_error():
    box = tools(get_annual_financials={"ok": False, "error": "no data for ZZZZ"})
    with pytest.raises(LookupError, match="no data for ZZZZ"):
        asyncio.run(snap.fetch_snapshot(box, "ZZZZ", set()))


# --- the API ------------------------------------------------------------------------------------


ALICE = {"X-User-Email": "alice@example.com"}


@pytest.fixture
def snapshot_client(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "internal_api_token", None)
    box = tools()

    class Toolbox:
        tools, failed = list(box.values()), {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    engine = make_engine(f"sqlite:///{(tmp_path / 'snap.db').as_posix()}")
    Base.metadata.create_all(engine)
    team = Team(toolbox_factory=Toolbox, graph_factory=lambda tools, on_event: object())
    with TestClient(api.create_app(team=team, sessions=sessionmaker(bind=engine, expire_on_commit=False),
                                   start_worker=False)) as client:
        client.box = box
        yield client


def test_the_snapshot_endpoint_builds_once_then_serves_from_cache(snapshot_client):
    first = snapshot_client.get("/companies/exm/snapshot", headers=ALICE)
    assert first.status_code == 200 and first.json()["ticker"] == "EXM"
    again = snapshot_client.get("/companies/EXM/snapshot", headers=ALICE)
    assert again.json() == first.json()
    assert len(snapshot_client.box["get_quote"].calls) == 1   # the second request used the cache


def test_the_snapshot_endpoint_rejects_bad_tickers_and_needs_a_user(snapshot_client):
    assert snapshot_client.get("/companies/EXM/snapshot").status_code == 401
    assert snapshot_client.get("/companies/..%2F.env/snapshot", headers=ALICE).status_code in (404, 422)
    assert snapshot_client.get("/companies/A B/snapshot", headers=ALICE).status_code == 422


def test_an_unknown_company_is_a_404(snapshot_client):
    snapshot_client.box["get_annual_financials"].reply = {"ok": False, "error": "no data for ZZZZ"}
    response = snapshot_client.get("/companies/ZZZZ/snapshot", headers=ALICE)
    assert response.status_code == 404 and "ZZZZ" in response.json()["detail"]
