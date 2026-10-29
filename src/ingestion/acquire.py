"""Fetch a company's latest annual report and register it for indexing (Phase 5 Step 3).

Pipeline position:  [acquire]  ->  loader  ->  chunker  ->  indexer  ->  Chroma

Sources:
  - SEC EDGAR: the latest 10-K / 20-F / 40-F of a US-listed company (an HTML document).
  - EDINET: the latest Annual Securities Report of a Tokyo-listed company (PDF, in Japanese).
  - A PDF URL you supply: any other company (e.g. Indian annual reports on company sites).

Only the most useful pages are indexed - management's discussion, financial statements,
business, risks - within a chunk budget, because the free Gemini tier embeds only ~1,000
chunks a day. The choice is stored in documents.json ("pages"), so re-indexing is repeatable.
"""

import json
import logging
import re
from datetime import date
from pathlib import Path

import httpx

from config.settings import settings
from src.ingestion.chunker import STATEMENT_RE, chunk_page, chunk_pages
from src.ingestion.indexer import make_batches
from src.ingestion.loader import DocumentMeta, Page, format_page_ranges, load_document

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024
DEFAULT_MAX_CHUNKS = 250
ANNUAL_SEC_FORMS = ("10-K", "20-F", "40-F")

# Page groups in priority order: filled until the chunk budget runs out.
SEC_PRIORITIES = {
    "10-K": ["7", "8", "fs", "1", "1A", "7A"],          # MD&A, statements, business, risks
    "20-F": ["5", "fs", "18", "4", "3"],                  # operating review, statements, company, risks
    "40-F": ["fs"],
}
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-d]?)\s*[.:\-–—]?\s+\S", re.I)
FS_RE = re.compile(r"notes to (the )?consolidated financial statements|segment information", re.I)
KEYWORDS = [  # page scoring for reports without "Item" headings (Indian / Japanese annual reports)
    "consolidated", "segment", "revenue", "profit", "balance sheet", "cash flow", "management discussion",
    "risk", "outlook", "operating income",
    "連結", "セグメント", "売上", "営業利益", "貸借対照表", "損益計算書", "キャッシュ・フロー", "事業等のリスク",
    "経営者による財政状態", "経営方針",
]
UNIT_PATTERNS = [(re.compile(r"in millions|millions of", re.I), "millions"),
                 (re.compile(r"in thousands|thousands of", re.I), "thousands"),
                 (re.compile(r"in crore|crore", re.I), "crore"),
                 (re.compile(r"百万円"), "millions")]

_transport: httpx.BaseTransport | None = None  # tests inject httpx.MockTransport here


class AcquireError(RuntimeError):
    """A filing could not be found or downloaded; the message says what to do."""


# --- downloading -------------------------------------------------------------------------------


def download(url: str, dest: Path, headers: dict | None = None, expect: bytes | None = None) -> Path:
    """Stream a file to disk with a size cap; written to a temp file first, so no half files."""
    if not url.startswith("https://"):
        raise AcquireError(f"Only https:// downloads are allowed, got {url[:80]!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    size = 0
    with httpx.Client(timeout=120, transport=_transport, headers=headers or {}, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise AcquireError(f"Download failed: HTTP {response.status_code} for {url[:120]}")
            with tmp.open("wb") as out:
                for block in response.iter_bytes():
                    size += len(block)
                    if size > MAX_DOWNLOAD_BYTES:
                        out.close()
                        tmp.unlink(missing_ok=True)
                        raise AcquireError(f"File is larger than {MAX_DOWNLOAD_BYTES // 2**20} MB; refusing it")
                    out.write(block)
    if expect and not tmp.read_bytes()[:1024].lstrip().startswith(expect):
        tmp.unlink(missing_ok=True)
        raise AcquireError(f"The downloaded file is not a {expect.decode(errors='ignore')} document")
    tmp.replace(dest)
    return dest


def _safe_name(*parts: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(p for p in parts if p)).strip("-")


# --- sources -----------------------------------------------------------------------------------


def sec_annual_report(ticker: str, raw_dir: Path | None = None) -> dict:
    """Download the latest 10-K / 20-F / 40-F; returns a registry entry (without "pages")."""
    from src.mcp_servers import sec_server as sec

    if not settings.sec_user_agent:
        raise AcquireError("SEC_USER_AGENT is missing in .env")
    try:
        cik, symbol, name, _ = sec._company(ticker)
        recent = sec._get_json(sec.SUBMISSIONS_URL.format(cik=cik), f"submissions:{cik}", 0)["filings"]["recent"]
    except LookupError as e:
        raise AcquireError(str(e)) from e
    for i, form in enumerate(recent.get("form", [])):
        if form in ANNUAL_SEC_FORMS:
            break
    else:
        raise AcquireError(f"No 10-K / 20-F / 40-F found among {symbol}'s recent SEC filings")

    accession = recent["accessionNumber"][i].replace("-", "")
    url = sec.ARCHIVE_URL.format(cik=cik, accession=accession) + recent["primaryDocument"][i]
    period = recent["reportDate"][i]
    suffix = Path(recent["primaryDocument"][i]).suffix.lower() or ".htm"
    path = download(url, (raw_dir or settings.raw_dir) / (_safe_name(symbol, form, period) + suffix),
                    headers={"User-Agent": settings.sec_user_agent})
    currency = "USD"
    if form != "10-K":  # foreign filers report in their own currency
        facts = sec._get_json(sec.FACTS_URL.format(cik=cik), f"facts:{cik}", sec.TTL["facts"])
        currency = sec.reporting_currency(facts.get("facts") or {}) or "USD"
    return {"file": path.name, "company": name.rstrip("/ ").title(), "ticker": symbol, "market": "USA",
            "doc_type": form, "fiscal_year_end": period, "currency": currency, "reporting_unit": "",
            "source": "sec", "source_url": url, "filed": recent["filingDate"][i]}


def edinet_annual_report(ticker: str, raw_dir: Path | None = None) -> dict:
    """Download the latest Annual Securities Report (PDF) of a Tokyo-listed company."""
    from src.mcp_servers import edinet_server as ed

    if not settings.edinet_api_key:
        raise AcquireError("EDINET_API_KEY is missing in .env")
    code = ed.sec_code(ticker)
    index, _ = ed._fresh_index()
    docs = [d for d in ed._company_docs(code, index, "annual")]
    if not docs:
        raise AcquireError(f"No Annual Securities Report for {code[:4]} in the EDINET index "
                           "(run: python scripts/edinet_sync.py)")
    doc = docs[0]
    response = ed._get(f"documents/{doc['docID']}", {"type": 2})
    if not response.content.startswith(b"%PDF"):
        raise AcquireError("EDINET did not return a PDF for this report")
    path = (raw_dir or settings.raw_dir) / (_safe_name(f"{code[:4]}.T", "ASR", doc["periodEnd"]) + ".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return {"file": path.name, "company": doc.get("filerName") or f"{code[:4]}.T", "ticker": f"{code[:4]}.T",
            "market": "Japan", "doc_type": "Annual Securities Report", "fiscal_year_end": doc["periodEnd"],
            "currency": "JPY", "reporting_unit": "", "source": "edinet", "source_url": f"EDINET {doc['docID']}",
            "filed": (doc.get("submitDateTime") or "")[:10]}


def pdf_from_url(url: str, entry: dict, raw_dir: Path | None = None) -> dict:
    """Download a PDF you point to; `entry` supplies company, ticker, market, doc_type, fiscal_year_end, currency."""
    name = _safe_name(entry["ticker"], entry["doc_type"], entry["fiscal_year_end"]) + ".pdf"
    path = download(url, (raw_dir or settings.raw_dir) / name, expect=b"%PDF")
    return {**entry, "file": path.name, "reporting_unit": entry.get("reporting_unit", ""), "source": "url",
            "source_url": url, "filed": entry.get("filed") or date.today().isoformat()}


# --- page selection ------------------------------------------------------------------------------


def item_of_pages(pages: list[Page]) -> dict[int, set[str]]:
    """Which 10-K / 20-F "Item" each page belongs to (a page can start a new item midway)."""
    current, out = None, {}
    for page in pages:
        found = [m.group(1).upper() for line in page.text.splitlines() if len(line) < 150
                 for m in [ITEM_RE.match(line)] if m]
        items = {current} if current else set()
        if len(set(found)) >= 5:          # the table of contents lists every item: not a real start
            out[page.page] = items
            continue
        if found:
            items |= set(found)
            current = found[-1]
        out[page.page] = items
    return out


def _chunks(page: Page) -> int:
    return len(chunk_page(page))


def choose_pages(pages: list[Page], doc_type: str, max_chunks: int = DEFAULT_MAX_CHUNKS) -> list[int]:
    """Page numbers to index: the most useful pages that fit in `max_chunks`."""
    if sum(_chunks(p) for p in pages) <= max_chunks:
        return [p.page for p in pages]

    groups: list[list[Page]] = []
    if doc_type in SEC_PRIORITIES:
        items = item_of_pages(pages)
        for group in SEC_PRIORITIES[doc_type]:
            if group == "fs":
                groups.append([p for p in pages if STATEMENT_RE.search(p.text[:400]) or FS_RE.search(p.text[:400])])
            else:
                groups.append([p for p in pages if group in items.get(p.page, set())])
    else:
        scored = sorted(pages, key=lambda p: -sum(p.text.lower().count(k.lower()) for k in KEYWORDS))
        groups.append([p for p in scored if any(k.lower() in p.text.lower() for k in KEYWORDS)])

    chosen, budget = set(), max_chunks
    for group in groups:
        for page in group:
            if page.page in chosen:
                continue
            cost = _chunks(page)
            if cost > budget:
                continue
            chosen.add(page.page)
            budget -= cost
    return sorted(chosen)


def detect_unit(pages: list[Page], currency: str) -> str:
    """'USD millions' from the statements' "(in millions ...)" headers; a pointer if unclear."""
    counts = {unit: 0 for _, unit in UNIT_PATTERNS}
    for page in pages:
        for pattern, unit in UNIT_PATTERNS:
            counts[unit] += len(pattern.findall(page.text))
    unit, hits = max(counts.items(), key=lambda kv: kv[1])
    return f"{currency} {unit}" if hits else f"{currency} (see each statement's header)"


def prepare(entry: dict, max_chunks: int = DEFAULT_MAX_CHUNKS, raw_dir: Path | None = None) -> tuple[dict, dict]:
    """Choose pages and the reporting unit for a downloaded filing; returns (entry, stats)."""
    meta = DocumentMeta(**{k: v for k, v in entry.items() if k != "pages"})
    pages = load_document(meta, raw_dir)
    if not pages:
        raise AcquireError(f"No readable text in {entry['file']} (a scanned PDF needs OCR)")
    chosen = choose_pages(pages, entry["doc_type"], max_chunks)
    kept = [p for p in pages if p.page in chosen]
    entry = {**entry, "pages": format_page_ranges(chosen)}
    if not entry.get("reporting_unit"):
        entry["reporting_unit"] = detect_unit(kept, entry["currency"])
    chunks = chunk_pages(kept)
    # One batch a minute, capped by count AND characters (the indexer's own batching).
    stats = {"pages_total": len(pages), "pages_chosen": len(kept), "chunks": len(chunks),
             "minutes": len(make_batches(chunks))}
    return entry, stats


# --- registry --------------------------------------------------------------------------------------


def _base(ticker: str) -> str:
    return ticker.strip().upper().split(".")[0]


def _in_database(path: Path | None) -> bool:
    return path is None and settings.filings_store == "postgres"


def registered(ticker: str, path: Path | None = None) -> list[dict]:
    if _in_database(path):
        from src.db.repository import documents_for_ticker
        from src.db.session import session_scope

        with session_scope() as db:
            return documents_for_ticker(db, ticker)
    path = path or settings.documents_registry
    return [e for e in json.loads(path.read_text(encoding="utf-8")) if _base(e["ticker"]) == _base(ticker)]


def register(entry: dict, path: Path | None = None) -> list[dict]:
    """Add the entry, replacing the same company's older entries; returns the replaced ones."""
    if _in_database(path):  # deployed: the documents table, so an added company survives restarts
        from src.db.repository import replace_document
        from src.db.session import session_scope

        with session_scope() as db:
            return replace_document(db, entry)
    path = path or settings.documents_registry
    entries = json.loads(path.read_text(encoding="utf-8"))
    replaced = [e for e in entries if _base(e["ticker"]) == _base(entry["ticker"])]
    entries = [e for e in entries if e not in replaced] + [entry]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return replaced


def latest_filing_date(entry: dict) -> str | None:
    """Filing date of the newest annual report the source now has (None if unknown)."""
    if entry.get("source") == "sec":
        from src.mcp_servers import sec_server as sec

        cik, _, _, _ = sec._company(entry["ticker"])
        recent = sec._get_json(sec.SUBMISSIONS_URL.format(cik=cik), f"submissions:{cik}", 0)["filings"]["recent"]
        return next((recent["filingDate"][i] for i, f in enumerate(recent["form"]) if f in ANNUAL_SEC_FORMS), None)
    if entry.get("source") == "edinet":
        from src.mcp_servers import edinet_server as ed

        docs = ed._company_docs(ed.sec_code(entry["ticker"]), ed._fresh_index()[0], "annual")
        return (docs[0].get("submitDateTime") or "")[:10] if docs else None
    return None
