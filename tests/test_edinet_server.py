"""EDINET MCP server with a fake EDINET API (httpx.MockTransport). Offline: no key needed."""

import asyncio
import io
import zipfile
from datetime import date, timedelta

import httpx
import pytest

from src.mcp_servers import edinet_server as ed

HEADER = ["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別", "期間・時点", "ユニットID", "単位", "値"]


def csv_zip(rows: list[tuple[str, str, str, str]], name="XBRL_TO_CSV/jpcrp030000-asr-001_E02144-000_2025-03-31_01_2025-06-18.csv"):
    """rows: (element, context, unit, value) -> a ZIP holding one UTF-16 tab-separated CSV, as EDINET sends."""
    lines = ["\t".join(f'"{h}"' for h in HEADER)]
    for element, context, unit, value in rows:
        lines.append("\t".join(f'"{x}"' for x in (element, "label", context, "", "", "", unit, "", value)))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, "\n".join(lines).encode("utf-16"))
        archive.writestr("XBRL_TO_CSV/jpaud-aar-cn-001.csv", "\t".join(HEADER).encode("utf-16"))  # audit report: ignored
    return buffer.getvalue()


S = "SummaryOfBusinessResults"
# Toyota-like IFRS summary (JPY), two years, plus non-consolidated rows that must be ignored.
TOYOTA_ROWS = [
    (f"jpcrp_cor:RevenueIFRS{S}", "CurrentYearDuration", "JPY", "48036704000000"),
    (f"jpcrp_cor:RevenueIFRS{S}", "Prior1YearDuration", "JPY", "45095325000000"),
    (f"jpcrp_cor:OperatingProfitLossIFRS{S}", "CurrentYearDuration", "JPY", "4795586000000"),
    (f"jpcrp_cor:ProfitLossAttributableToOwnersOfParentIFRS{S}", "CurrentYearDuration", "JPY", "4765086000000"),
    (f"jpcrp_cor:TotalAssetsIFRS{S}", "CurrentYearInstant", "JPY", "93601350000000"),
    (f"jpcrp_cor:DilutedEarningsPerShareIFRS{S}", "CurrentYearDuration", "JPYPerShares", "359.56"),
    (f"jpcrp_cor:NetSales{S}", "CurrentYearDuration_NonConsolidatedMember", "JPY", "12345"),
    (f"jpcrp_cor:TotalAssetsIFRS{S}", "Prior2YearInstant", "JPY", "－"),            # placeholder: not a value
    ("jppfs_cor:NetSales", "CurrentYearDuration", "JPY", "1"),                       # not the summary: ignored
]
ANNUAL_DOC = {"docID": "S100TEST", "secCode": "72030", "edinetCode": "E02144", "filerName": "トヨタ自動車株式会社",
              "docTypeCode": "120", "periodStart": "2024-04-01", "periodEnd": "2025-03-31",
              "submitDateTime": "2025-06-18 15:00", "csvFlag": "1", "withdrawalStatus": "0"}


class FakeEdinet:
    def __init__(self, status=200, lists=None, csv=None):
        self.status, self.lists, self.csv, self.requests = status, lists or {}, csv or {}, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"StatusCode": self.status, "message": "Access denied"})
        if request.url.path.endswith("documents.json"):
            day = request.url.params["date"]
            return httpx.Response(200, json={"metadata": {"status": "200"}, "results": self.lists.get(day, [])})
        doc_id = request.url.path.rsplit("/", 1)[-1]
        if doc_id in self.csv:
            return httpx.Response(200, content=self.csv[doc_id])
        return httpx.Response(200, json={"metadata": {"status": "404", "message": "not found"}})


@pytest.fixture
def edinet(monkeypatch, tmp_path):
    fake = FakeEdinet(csv={"S100TEST": csv_zip(TOYOTA_ROWS)})
    ed._cache.clear()
    monkeypatch.setattr(ed.settings, "edinet_api_key", "test-key-123")
    monkeypatch.setattr(ed, "_transport", httpx.MockTransport(fake))
    monkeypatch.setattr(ed, "MIN_REQUEST_GAP", 0)
    monkeypatch.setattr(ed, "INDEX_PATH", tmp_path / "index.json")
    ed.save_index({"synced": [], "docs": [ed._slim(ANNUAL_DOC)]})
    return fake


# --- parsing --------------------------------------------------------------------------------


def test_tokyo_codes_become_edinet_securities_codes():
    assert ed.sec_code("7203.T") == "72030" and ed.sec_code("7203") == "72030" and ed.sec_code("130a.t") == "130A0"
    with pytest.raises(ed.ToolInputError):
        ed.sec_code("AAPL")


def test_summary_figures_use_consolidated_rows_and_label_periods():
    rows, scope = ed.summary_figures(ed.read_csv_rows(csv_zip(TOYOTA_ROWS)), "2024-04-01", "2025-03-31", "annual")
    assert scope == "consolidated"
    current, prior = rows
    assert current["period_end"] == "2025-03-31" and prior["period_end"] == "2024-03-31"
    assert current["revenue"] == 48036704000000 and current["revenue_display"] == "48.04 T JPY"
    assert current["operating_income"] == 4795586000000 and current["net_income"] == 4765086000000
    assert current["diluted_eps"] == 359.56 and current["diluted_eps_display"] == "359.56 JPY per share"
    assert prior["revenue"] == 45095325000000 and "total_assets" not in prior  # "－" is not a value


def test_japanese_gaap_element_names_map_to_the_same_fields():
    rows = [(f"jpcrp_cor:NetSales{S}", "CurrentYearDuration", "JPY", "100"),
            (f"jpcrp_cor:OrdinaryIncomeLoss{S}", "CurrentYearDuration", "JPY", "20"),
            (f"jpcrp_cor:NetAssets{S}", "CurrentYearInstant", "JPY", "500"),
            (f"jpcrp_cor:NetCashProvidedByUsedInOperatingActivities{S}", "CurrentYearDuration", "JPY", "30")]
    row = ed.summary_figures(ed.read_csv_rows(csv_zip(rows)), "2024-04-01", "2025-03-31", "annual")[0][0]
    assert (row["revenue"], row["ordinary_income"], row["equity"], row["operating_cash_flow"]) == (100, 20, 500, 30)


def test_only_non_consolidated_figures_are_labelled_so():
    rows = [(f"jpcrp_cor:NetSales{S}", "CurrentYearDuration_NonConsolidatedMember", "JPY", "100")]
    result, scope = ed.summary_figures(ed.read_csv_rows(csv_zip(rows)), "2024-04-01", "2025-03-31", "annual")
    assert scope == "non-consolidated" and result[0]["revenue"] == 100


def test_semi_annual_periods():
    rows = [(f"jpcrp_cor:RevenueIFRS{S}", "InterimDuration", "JPY", "24000"),
            (f"jpcrp_cor:RevenueIFRS{S}", "Prior1InterimDuration", "JPY", "23000"),
            (f"jpcrp_cor:RevenueIFRS{S}", "Prior1YearDuration", "JPY", "48000")]
    out, _ = ed.summary_figures(ed.read_csv_rows(csv_zip(rows)), "2025-04-01", "2025-09-30", "semi_annual")
    assert [(r["period"], r["period_end"], r["revenue"]) for r in out] == [
        ("first half of the fiscal year", "2025-09-30", 24000),
        ("fiscal year", "2025-03-31", 48000),
        ("first half of the fiscal year", "2024-09-30", 23000)]


# --- index -----------------------------------------------------------------------------------


def test_sync_keeps_only_listed_companies_reports_and_resumes(edinet):
    day = date(2025, 6, 18)
    edinet.lists = {day.isoformat(): [
        {**ANNUAL_DOC, "docID": "S100NEW1", "secCode": "67580"},
        {**ANNUAL_DOC, "docID": "S100FUND", "secCode": None},                  # a fund: no securities code
        {**ANNUAL_DOC, "docID": "S100OTHR", "docTypeCode": "350"},              # not an annual / semi-annual report
        {**ANNUAL_DOC, "docID": "S100WDRN", "withdrawalStatus": "1"},           # withdrawn
    ]}
    index = ed.sync_index([day])
    assert {d["docID"] for d in index["docs"]} == {"S100TEST", "S100NEW1"} and index["synced"] == ["2025-06-18"]
    ed.sync_index([day])
    assert len(edinet.requests) == 1  # already synced: not fetched again


def test_today_is_always_re_read(edinet):
    ed.sync_index([date.today()])
    ed.sync_index([date.today()])
    assert len(edinet.requests) == 2 and date.today().isoformat() not in ed.load_index()["synced"]


# --- tools -----------------------------------------------------------------------------------


def test_annual_figures_end_to_end(edinet):
    r = ed.get_edinet_financials("7203.T", "annual", 2)
    assert r["ok"] and r["ticker"] == "7203.T" and r["currency"] == "JPY" and r["scope"] == "consolidated"
    assert r["rows"][0]["revenue"] == 48036704000000
    assert r["cite_as"] == "(EDINET, Annual Securities Report filed 2025-06-18)" and r["doc_id"] == "S100TEST"
    assert any("covers only" in w and "edinet_sync.py" in w for w in r["warnings"])  # small index: say so
    assert edinet.requests[0].url.params["Subscription-Key"] == "test-key-123"


def test_company_without_reports_is_not_found(edinet):
    r = ed.get_edinet_financials("6758.T")
    assert r["ok"] is False and r["error_type"] == "not_found"


def test_missing_key_is_not_configured_without_requests(monkeypatch, edinet):
    monkeypatch.setattr(ed.settings, "edinet_api_key", None)
    r = ed.get_edinet_financials("7203.T")
    assert r["error_type"] == "not_configured" and "EDINET_API_KEY" in r["error"] and edinet.requests == []


def test_rejected_key_is_reported_without_leaking_it(monkeypatch, edinet):
    monkeypatch.setattr(ed, "_transport", httpx.MockTransport(FakeEdinet(status=401)))
    r = ed.get_edinet_financials("7203.T")
    assert r["error_type"] == "not_configured" and "test-key-123" not in str(r)


def test_filings_are_listed_newest_first(edinet):
    semi = {**ANNUAL_DOC, "docID": "S100SEMI", "docTypeCode": "160", "periodStart": "2025-04-01",
            "periodEnd": "2025-09-30", "submitDateTime": "2025-11-13 15:00"}
    index = ed.load_index()
    index["docs"].append(ed._slim(semi))
    ed.save_index(index)
    r = ed.get_edinet_filings("7203")
    assert [(f["report"], f["filed"]) for f in r["filings"]] == [("Semi-annual Report", "2025-11-13"),
                                                                ("Annual Securities Report", "2025-06-18")]


@pytest.mark.parametrize("kwargs", [{"report": "quarterly"}, {"count": 0}, {"count": 6}])
def test_bad_arguments_are_rejected(edinet, kwargs):
    r = ed.get_edinet_financials("7203.T", **kwargs)
    assert r["error_type"] == "invalid_input" and edinet.requests == []


def test_tools_are_discoverable():
    assert {t.name for t in asyncio.run(ed.mcp.list_tools())} == {"get_edinet_financials", "get_edinet_filings"}


def test_recent_days_counts_back_from_today():
    assert ed.recent_days(3, until=date(2026, 1, 2)) == [date(2026, 1, 2), date(2026, 1, 1), date(2025, 12, 31)]
    assert ed.recent_days(1)[0] == date.today() and ed.recent_days(2)[1] == date.today() - timedelta(days=1)
