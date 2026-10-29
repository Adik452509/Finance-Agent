"""EDINET MCP server: official figures of Japanese listed companies from Japan's FSA (EDINET API v2).

Start it standalone with:
    python -m src.mcp_servers.edinet_server

The Japanese counterpart of the SEC server. Each Annual Securities Report (有価証券報告書)
has a standard "Summary of business results" (主要な経営指標等の推移): five years of
headline figures, tagged in XBRL. EDINET's type=5 download converts that XBRL to CSV, so
- like the SEC data - the figures are exact and need no PDF table reading.

Two EDINET quirks shape this module:
  - There is no "search by company" endpoint, only "all filings on date D". So we keep a
    local index of the daily lists (data/cache/edinet/index.json): built once with
    `python scripts/edinet_sync.py`, then topped up with the last few days on every query.
  - Quarterly reports were abolished in April 2024; the freshest official figures between
    annual reports are in the Semi-annual Report (半期報告書).

Needs EDINET_API_KEY in .env (free). The key is sent as a query parameter, so it must
never appear in a URL we return, a log line or an error message.
"""

import csv
import io
import json
import logging
import re
import sys
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from config.settings import settings
from src.mcp_servers.common import ErrorType, ToolInputError, fail, ok, safe_call

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("edinet_server")

API = "https://api.edinet-fsa.go.jp/api/v2"
SOURCE = "EDINET"
TIMEOUT_SECONDS = 60
MIN_REQUEST_GAP = 1.0            # no published limit; one request a second is polite
REFRESH_DAYS = 7                 # days re-checked on each query so new filings show up
FULL_INDEX_DAYS = 400            # an annual report is filed every ~365 days
INDEX_PATH = settings.data_dir / "cache" / "edinet" / "index.json"
DOC_CACHE_TTL = 24 * 3600

REPORTS = {  # docTypeCode -> (kind, English name)
    "120": ("annual", "Annual Securities Report"), "130": ("annual", "Amended Annual Securities Report"),
    "160": ("semi_annual", "Semi-annual Report"), "170": ("semi_annual", "Amended Semi-annual Report"),
}
MAX_COUNT = 5

# Output field -> lower-case prefixes of the XBRL element name (with "SummaryOfBusinessResults"
# removed). Prefixes, because the same item is tagged per accounting standard: NetSales (J-GAAP),
# RevenueIFRS, RevenuesUSGAAP... The first field whose prefix matches wins.
FIELD_PREFIXES = [
    ("diluted_eps", ("dilutedearnings",)),
    ("operating_cash_flow", ("netcashprovidedbyusedinoperatingactivities", "cashflowsfromusedinoperatingactivities")),
    ("cash_and_equivalents", ("cashandcashequivalents",)),
    ("net_income", ("profitlossattributabletoownersofparent", "netincomelossattributabletoownersofparent",
                    "netincomeloss", "profitloss")),
    ("ordinary_income", ("ordinaryincome",)),
    ("operating_income", ("operatingincome", "operatingprofit")),
    ("revenue", ("netsales", "revenue", "operatingrevenue", "totalrevenue", "salesandoperatingrevenue")),
    ("total_assets", ("totalassets",)),
    ("equity", ("equityattributabletoownersofparent", "netassets", "totalequity", "shareholdersequity")),
]
CONTEXT_RE = re.compile(r"^(CurrentYear|Prior(\d)Year|Interim|Prior(\d)Interim)(Duration|Instant)"
                        r"(_NonConsolidatedMember)?$")

mcp = FastMCP("edinet", instructions=(
    "Official figures of Japanese listed companies from their Annual Securities Reports and "
    "Semi-annual Reports filed with Japan's FSA (EDINET)."))

_transport: httpx.BaseTransport | None = None  # tests inject httpx.MockTransport here
_cache: dict[str, tuple[float, object]] = {}
_last_request = 0.0


class EdinetError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(f"EDINET returned {status}")
        self.status, self.message = status, message


# --- HTTP ----------------------------------------------------------------------------------


def _get(path: str, params: dict) -> httpx.Response:
    """Throttled GET with the key. Errors never include the key or the full URL."""
    global _last_request
    wait = MIN_REQUEST_GAP - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    with httpx.Client(timeout=TIMEOUT_SECONDS, transport=_transport) as client:
        response = client.get(f"{API}/{path}", params={**params, "Subscription-Key": settings.edinet_api_key or ""})
    _last_request = time.monotonic()
    if response.status_code != 200:
        raise EdinetError(response.status_code)
    return response


def document_list(day: date) -> list[dict]:
    """Every filing EDINET received on one day (metadata only)."""
    data = _get("documents.json", {"date": day.isoformat(), "type": 2}).json()
    status = str((data.get("metadata") or {}).get("status", "200"))
    if status != "200":  # EDINET also reports errors inside a 200 response
        raise EdinetError(int(status) if status.isdigit() else 500, str((data.get("metadata") or {}).get("message", "")))
    return data.get("results") or []


def document_csv(doc_id: str) -> bytes:
    """The filing's XBRL converted to CSV (a ZIP of UTF-16 tab-separated files)."""
    key = f"csv:{doc_id}"
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < DOC_CACHE_TTL:
        return hit[1]  # type: ignore[return-value]
    content = _get(f"documents/{doc_id}", {"type": 5}).content
    if not content.startswith(b"PK"):  # an error comes back as JSON instead of a ZIP
        raise EdinetError(404, "no CSV for this document")
    _cache[key] = (time.monotonic(), content)
    return content


def _failure(e: Exception) -> dict:
    if isinstance(e, EdinetError):
        if e.status == 401:
            return fail(ErrorType.NOT_CONFIGURED, "EDINET rejected the API key (401). Check EDINET_API_KEY in .env.")
        if e.status == 404:
            return fail(ErrorType.NOT_FOUND, "EDINET has no such document.")
        if e.status == 429:
            return fail(ErrorType.RATE_LIMITED, "EDINET rate limit hit. Retry in a minute.")
        return fail(ErrorType.UPSTREAM_ERROR, f"EDINET error {e.status}.")
    if isinstance(e, httpx.TimeoutException):
        return fail(ErrorType.UPSTREAM_ERROR, f"EDINET did not respond within {TIMEOUT_SECONDS}s.")
    return fail(ErrorType.UPSTREAM_ERROR, f"Network error reaching EDINET: {type(e).__name__}")


# --- local index of the daily filing lists ---------------------------------------------------


def load_index(path: Path | None = None) -> dict:
    path = path or INDEX_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"synced": [], "docs": []}


def save_index(index: dict, path: Path | None = None) -> None:
    path = path or INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic: the server and the sync script may both write


def _slim(doc: dict) -> dict | None:
    """Keep only listed companies' annual / semi-annual reports, and only the fields we use."""
    if doc.get("docTypeCode") not in REPORTS or not doc.get("secCode") or str(doc.get("withdrawalStatus", "0")) != "0":
        return None
    return {k: doc.get(k) for k in ("docID", "secCode", "edinetCode", "filerName", "docTypeCode", "periodStart",
                                    "periodEnd", "submitDateTime", "csvFlag", "docDescription")}


def sync_index(days: list[date], index: dict | None = None, path: Path | None = None,
               progress=None) -> dict:
    """Add the given days' filings to the index (resumable: saved after every day)."""
    index = index if index is not None else load_index(path)
    synced = set(index["synced"])
    known = {d["docID"] for d in index["docs"]}
    today = date.today().isoformat()
    for n, day in enumerate(days, start=1):
        iso = day.isoformat()
        if iso in synced and iso != today:  # today's list keeps growing: always re-read it
            continue
        for doc in document_list(day):
            slim = _slim(doc)
            if slim and slim["docID"] not in known:
                index["docs"].append(slim)
                known.add(slim["docID"])
        if iso != today:
            synced.add(iso)
        index["synced"] = sorted(synced)
        save_index(index, path)
        if progress:
            progress(n, len(days), iso)
    return index


def recent_days(count: int, until: date | None = None) -> list[date]:
    until = until or date.today()
    return [until - timedelta(days=i) for i in range(count)]


# --- parsing ------------------------------------------------------------------------------------


def sec_code(ticker: str) -> str:
    """'7203.T' or '7203' -> EDINET's 5-character securities code '72030'."""
    if not isinstance(ticker, str):
        raise ToolInputError("ticker must be a string such as '7203.T'")
    cleaned = ticker.strip().upper().removesuffix(".T")
    if not re.fullmatch(r"[0-9][0-9A-Z]{3}", cleaned):
        raise ToolInputError(f"'{ticker}' is not a Tokyo Stock Exchange code. Use the 4-character code, e.g. 7203.T")
    return cleaned + "0"


def _number(text: str) -> int | float | None:
    try:
        value = Decimal(text.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None  # "－", "" and other placeholders mean "not reported"
    return int(value) if value == value.to_integral_value() else float(value)


def _field(element: str) -> str | None:
    local = element.split(":")[-1].removesuffix("SummaryOfBusinessResults").lower()
    for field, prefixes in FIELD_PREFIXES:
        if local.startswith(prefixes):
            return field
    return None


def read_csv_rows(zip_bytes: bytes) -> list[dict]:
    """Rows of the main report's CSV as dicts with keys element, context, unit, value."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv") and "/jpcrp" in "/" + n.lower()]
        for name in names:
            text = archive.read(name).decode("utf-16")
            reader = csv.reader(io.StringIO(text), delimiter="\t")
            header = next(reader, [])

            def col(keyword: str, default: int) -> int:
                return next((i for i, h in enumerate(header) if keyword in h), default)

            # Documented order: 要素ID, 項目名, コンテキストID, 相対年度, 連結・個別, 期間・時点, ユニットID, 単位, 値
            e, c, u, v = col("要素ID", 0), col("コンテキストID", 2), col("ユニットID", 6), col("値", 8)
            for r in reader:
                if len(r) > max(e, c, u, v):
                    rows.append({"element": r[e].strip('"'), "context": r[c].strip('"'),
                                 "unit": r[u].strip('"'), "value": r[v].strip('"')})
    return rows


def _shift_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29 February
        return day.replace(year=day.year - years, day=28)


def summary_figures(rows: list[dict], period_start: str, period_end: str, kind: str) -> tuple[list[dict], str]:
    """The 'Summary of business results' as rows per period, newest first; plus the scope used."""
    values: dict[tuple[str, bool], dict[str, dict]] = {}  # (period key, consolidated) -> field -> {value, unit}
    for row in rows:
        if not row["element"].endswith("SummaryOfBusinessResults"):
            continue
        match = CONTEXT_RE.match(row["context"])
        field = _field(row["element"])
        value = _number(row["value"])
        if not match or not field or value is None:
            continue
        span, prior_y, prior_i, _, non_consolidated = match.groups()
        key = ("Y" if "Year" in span else "I", int(prior_y or prior_i or 0))
        slot = values.setdefault((f"{key[0]}{key[1]}", not non_consolidated), {})
        slot.setdefault(field, {"value": value, "unit": row["unit"]})

    consolidated = any(scope for _, scope in values)
    end, start = date.fromisoformat(period_end), date.fromisoformat(period_start)
    out = []
    for (period_key, scope), fields in values.items():
        if scope != consolidated:
            continue
        kind_code, back = period_key[0], int(period_key[1:])
        if kind_code == "Y":
            # In an annual report "current year" ends at period_end. In a semi-annual report the
            # year columns are full fiscal years: Prior1Year is the one that ended the day before
            # the half-year began (period_start).
            if kind == "annual":
                fy_end = _shift_years(end, back)
            else:
                fy_end = _shift_years(start - timedelta(days=1), max(back - 1, 0))
            row = {"period": "fiscal year", "period_end": fy_end.isoformat()}
        else:
            row = {"period": "first half of the fiscal year", "period_end": _shift_years(end, back).isoformat()}
        for field, item in fields.items():
            row[field] = item["value"]
            row[f"{field}_display"] = _display(item["value"], item["unit"])
        out.append(row)
    out.sort(key=lambda r: (r["period_end"], r["period"] != "fiscal year"), reverse=True)
    return out, "consolidated" if consolidated else "non-consolidated"


def _display(value: int | float, unit: str) -> str:
    if unit.upper().startswith("JPYPERSHARE"):
        return f"{value:,.2f} JPY per share"
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= divisor:
            return f"{value / divisor:,.2f} {suffix} {unit or 'JPY'}"
    return f"{value:,} {unit or 'JPY'}"


# --- tools ---------------------------------------------------------------------------------------


def _company_docs(code: str, index: dict, kind: str | None = None) -> list[dict]:
    docs = [d for d in index["docs"] if d.get("secCode") == code and (kind is None or REPORTS[d["docTypeCode"]][0] == kind)]
    return sorted(docs, key=lambda d: d.get("submitDateTime") or "", reverse=True)


def _precheck(ticker: str) -> tuple[str, dict | None]:
    code = sec_code(ticker)
    if not settings.edinet_api_key:
        return code, fail(ErrorType.NOT_CONFIGURED, "EDINET is not configured: add EDINET_API_KEY to .env (free key "
                                                    "from https://api.edinet-fsa.go.jp/api/auth/index.aspx?mode=1).")
    return code, None


def _fresh_index() -> tuple[dict, list[str]]:
    """The index, topped up with the last few days so a report filed today is found."""
    index = sync_index(recent_days(REFRESH_DAYS))
    notes = []
    if len(index["synced"]) < FULL_INDEX_DAYS - 35:
        notes.append(f"The EDINET filing index covers only {len(index['synced'])} days; reports filed earlier may be "
                     "missed. Run: python scripts/edinet_sync.py")
    return index, notes


def _label(doc: dict) -> str:
    return f"{REPORTS[doc['docTypeCode']][1]} filed {(doc.get('submitDateTime') or '')[:10]}"


@mcp.tool()
def get_edinet_financials(ticker: str, report: str = "annual", count: int = 5) -> dict:
    """Get OFFICIAL figures of a Japanese listed company from its latest filing with Japan's FSA.

    Use for exact revenue (net sales), operating income, ordinary income, net income
    attributable to owners, total assets, equity, operating cash flow, cash and diluted EPS
    of Tokyo-listed companies (7203.T Toyota, 6758.T Sony, 9984.T SoftBank Group ...),
    from the "Summary of business results" of the Annual Securities Report (5 fiscal
    years) or the Semi-annual Report (latest half year - Japan has no quarterly reports
    since April 2024).

    Args:
        ticker: Tokyo code, e.g. 7203.T (the ".T" is optional).
        report: "annual" (Annual Securities Report) or "semi_annual" (Semi-annual Report).
        count: how many periods to return, 1-5 (default 5).

    Amounts are raw JPY (not millions). Cite as (EDINET, <filing> filed <date>).
    """
    def compute() -> dict:
        if report not in ("annual", "semi_annual"):
            raise ToolInputError('report must be "annual" or "semi_annual"')
        if not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
            raise ToolInputError(f"count must be an integer from 1 to {MAX_COUNT}")
        code, problem = _precheck(ticker)
        if problem:
            return problem
        try:
            index, notes = _fresh_index()
            docs = [d for d in _company_docs(code, index, report) if d.get("csvFlag") in ("1", 1, True)]
            if not docs:
                return fail(ErrorType.NOT_FOUND, f"No {report.replace('_', '-')} report with data for {code[:4]} in the "
                                                 f"EDINET index. " + " ".join(notes))
            doc = docs[0]
            rows, scope = summary_figures(read_csv_rows(document_csv(doc["docID"])),
                                          doc["periodStart"], doc["periodEnd"], report)
        except (EdinetError, httpx.HTTPError) as e:
            return _failure(e)
        except zipfile.BadZipFile:
            return fail(ErrorType.UPSTREAM_ERROR, "EDINET returned a damaged CSV archive.")
        if not rows:
            return fail(ErrorType.NOT_FOUND, f"The {_label(doc)} has no machine-readable summary figures.")
        warnings_ = notes + ([] if scope == "consolidated" else
                             ["Only non-consolidated (parent company) figures are reported; label them so."])
        return ok(ticker=f"{code[:4]}.T", company=doc.get("filerName"), scope=scope, currency="JPY",
                  unit="raw JPY (not millions); EPS per share", rows=rows[:count],
                  reported_in=_label(doc), doc_id=doc["docID"], warnings=warnings_, source=SOURCE,
                  cite_as=f"(EDINET, {_label(doc)})",
                  as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    return safe_call("get_edinet_financials", compute, logger)


@mcp.tool()
def get_edinet_filings(ticker: str, count: int = 5) -> dict:
    """List a Japanese listed company's recent Annual Securities Reports and Semi-annual Reports.

    Use to check what is new: "has Sony filed its annual report yet?", "when was the last
    semi-annual report?". Returns the report type, period and filing date, newest first.

    Args:
        ticker: Tokyo code, e.g. 6758.T.
        count: number of filings, 1-10 (default 5).
    """
    def compute() -> dict:
        if not isinstance(count, int) or not 1 <= count <= 10:
            raise ToolInputError("count must be an integer from 1 to 10")
        code, problem = _precheck(ticker)
        if problem:
            return problem
        try:
            index, notes = _fresh_index()
        except (EdinetError, httpx.HTTPError) as e:
            return _failure(e)
        docs = _company_docs(code, index)[:count]
        if not docs:
            return fail(ErrorType.NOT_FOUND, f"No reports for {code[:4]} in the EDINET index. " + " ".join(notes))
        filings = [{"report": REPORTS[d["docTypeCode"]][1], "filed": (d.get("submitDateTime") or "")[:10],
                    "period_start": d.get("periodStart"), "period_end": d.get("periodEnd"), "doc_id": d["docID"]}
                   for d in docs]
        return ok(ticker=f"{code[:4]}.T", company=docs[0].get("filerName"), filings=filings, warnings=notes,
                  source=SOURCE, as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    return safe_call("get_edinet_filings", compute, logger)


if __name__ == "__main__":
    mcp.run()  # stdio transport
