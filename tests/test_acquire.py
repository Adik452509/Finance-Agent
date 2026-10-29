"""Step 3: HTML filings in the loader, page selection, registry updates, downloads. Offline."""

import json

import httpx
import pytest

from src.ingestion import acquire
from src.ingestion.loader import (
    DocumentMeta, Page, format_page_ranges, html_pages, load_document, load_registry, parse_page_ranges,
)
from src.rag.retriever import RetrievedChunk

PAGE_BREAK = '<hr style="page-break-after:always"/>'
FILING = f"""<html><head><title>10-K</title><style>.x{{color:red}}</style></head><body>
<div style="display:none"><ix:header>hidden XBRL header 999,999</ix:header></div>
<div><p>MICROSOFT CORPORATION</p><p>FORM 10-K</p>
<p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 7. MD&amp;A</p><p>Item 7A. Market risk</p>
<p>Item 8. Financial Statements</p></div>
<div><p>Microsoft Corporation | 2026 Form 10-K | 1</p></div>{PAGE_BREAK}
<p>Item 1. Business</p><p>{"We build software. " * 40}</p>
<div><p>Microsoft Corporation | 2026 Form 10-K | 2</p></div>{PAGE_BREAK}
<p>ITEM 8. FINANCIAL STATEMENTS</p><p>INCOME STATEMENTS</p><p>(In millions)</p>
<table>
  <tr><td><p>Year Ended June 30,</p></td><td>2026</td><td>2025</td></tr>
  <tr><td>
      <div>Total revenue</div></td><td><span>$</span></td><td>331,839</td><td>$</td><td>281,724</td></tr>
  <tr><td>Other income</td><td>(</td><td>4,901</td><td>)</td></tr>
</table>
<div><p>Microsoft Corporation | 2026 Form 10-K | 3</p></div>{PAGE_BREAK}
</body></html>"""


@pytest.fixture
def filing(tmp_path):
    (tmp_path / "MSFT-10-K-2026-06-30.htm").write_text(FILING, encoding="utf-8")
    meta = DocumentMeta(file="MSFT-10-K-2026-06-30.htm", company="Microsoft Corp", ticker="MSFT", market="USA",
                        doc_type="10-K", fiscal_year_end="2026-06-30", currency="USD", reporting_unit="")
    return tmp_path, meta


# --- HTML in the loader ----------------------------------------------------------------------


def test_html_splits_at_page_breaks_and_hides_hidden_parts(filing):
    raw_dir, _ = filing
    pages = html_pages(raw_dir / "MSFT-10-K-2026-06-30.htm")
    assert len(pages) == 3
    assert "999,999" not in "".join(pages) and "color:red" not in "".join(pages)


def test_html_table_rows_stay_on_one_line(filing):
    # Seen live: one cell per line hid which year each number belonged to.
    raw_dir, meta = filing
    statement = load_document(meta, raw_dir)[2]
    assert "Year Ended June 30, 2026 2025" in statement.text
    assert "Total revenue $331,839 $281,724" in statement.text
    assert "Other income (4,901)" in statement.text


def test_html_pages_keep_printed_page_numbers(filing):
    raw_dir, meta = filing
    pages = load_document(meta, raw_dir)
    assert [(p.page, p.page_label) for p in pages] == [(1, "1"), (2, "2"), (3, "3")]
    assert all("Form 10-K |" not in p.text for p in pages)  # the footer itself is removed


def test_page_selection_is_applied(filing):
    raw_dir, meta = filing
    chosen = DocumentMeta(**{**meta.__dict__, "pages": "1,3"})
    assert [p.page for p in load_document(chosen, raw_dir)] == [1, 3]


def test_page_ranges_round_trip():
    assert parse_page_ranges("1-3,7, 9-10") == {1, 2, 3, 7, 9, 10}
    assert format_page_ranges([10, 1, 2, 3, 7, 9]) == "1-3,7,9-10"
    with pytest.raises(ValueError):
        parse_page_ranges("5-2")


def test_citation_says_pdf_only_for_pdfs():
    html = RetrievedChunk(id="a", text="", score=1, metadata={"file": "MSFT.htm", "company": "Microsoft", "doc_type": "10-K",
                                                             "fiscal_year_end": "2026-06-30", "page": 50, "page_label": "50"})
    pdf = RetrievedChunk(id="b", text="", score=1, metadata={**html.metadata, "file": "AAPL.pdf"})
    assert html.citation().endswith(", p. 50 (printed p. 50)") and ", PDF p. 50" in pdf.citation()


def test_optional_registry_fields_are_read_and_checked(tmp_path):
    base = {"file": "a.htm", "company": "A", "ticker": "A", "market": "USA", "doc_type": "10-K",
            "fiscal_year_end": "2026-06-30", "currency": "USD", "reporting_unit": "USD millions"}
    path = tmp_path / "documents.json"
    path.write_text(json.dumps([{**base, "source": "sec", "filed": "2026-07-29", "pages": "1-5"}]), encoding="utf-8")
    doc = load_registry(path)[0]
    assert doc.source == "sec" and doc.selected_pages() == {1, 2, 3, 4, 5}
    path.write_text(json.dumps([{**base, "pages": "x-1"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="bad page range"):
        load_registry(path)


def test_chunk_metadata_is_unchanged_for_hand_added_reports():
    meta = DocumentMeta(file="a.pdf", company="A", ticker="A", market="USA", doc_type="10-K",
                        fiscal_year_end="2025-09-27", currency="USD", reporting_unit="USD millions")
    assert set(Page(text="x", page=1, page_label=None, meta=meta).metadata()) == {
        "file", "company", "ticker", "market", "doc_type", "fiscal_year_end", "currency", "reporting_unit", "page"}


# --- page selection --------------------------------------------------------------------------


def page(n: int, text: str, meta=None) -> Page:
    meta = meta or DocumentMeta(file="x.htm", company="X", ticker="X", market="USA", doc_type="10-K",
                                fiscal_year_end="2026-06-30", currency="USD", reporting_unit="")
    return Page(text=text, page=n, page_label=None, meta=meta)


def test_items_are_tracked_across_pages_and_the_contents_page_is_ignored():
    pages = [page(1, "Item 1. Business\nItem 1A. Risk\nItem 7. MD&A\nItem 7A. Risk\nItem 8. Statements"),
             page(2, "Item 1. Business\ntext"), page(3, "more business"),
             page(4, "end of business\nItem 7. Management's Discussion"), page(5, "analysis")]
    items = acquire.item_of_pages(pages)
    assert items[1] == set() and items[3] == {"1"} and items[4] == {"1", "7"} and items[5] == {"7"}


def test_choose_pages_prefers_mdna_and_statements_within_the_budget():
    long = "word " * 400  # ~2,000 chars: about 2 chunks per page
    pages = [page(1, "Item 1. Business\n" + long), page(2, long),
             page(3, "Item 7. Management's Discussion\n" + long), page(4, long),
             page(5, "Item 8. Financial Statements\nCONSOLIDATED BALANCE SHEETS\n" + long)]
    cost = {p.page: acquire._chunks(p) for p in pages}
    chosen = acquire.choose_pages(pages, "10-K", max_chunks=cost[3] + cost[4] + cost[5])
    assert chosen == [3, 4, 5]  # MD&A and statements before the business section
    assert acquire.choose_pages(pages, "10-K", max_chunks=10_000) == [1, 2, 3, 4, 5]  # all fit: keep all


def test_reports_without_items_are_chosen_by_financial_keywords():
    pages = [page(1, "Chairman's letter. " * 50), page(2, "連結損益計算書 売上 営業利益 " * 30),
             page(3, "Consolidated balance sheet, segment revenue and profit. " * 20)]
    chosen = acquire.choose_pages(pages, "Annual Report", max_chunks=acquire._chunks(pages[1]) + acquire._chunks(pages[2]))
    assert chosen == [2, 3]


def test_reporting_unit_is_detected():
    assert acquire.detect_unit([page(1, "(In millions, except per share amounts)")], "USD") == "USD millions"
    assert acquire.detect_unit([page(1, "(単位：百万円)")], "JPY") == "JPY millions"
    assert acquire.detect_unit([page(1, "no unit here")], "INR") == "INR (see each statement's header)"


# --- registry ----------------------------------------------------------------------------------


def test_register_replaces_the_same_companys_older_report(tmp_path):
    path = tmp_path / "documents.json"
    path.write_text(json.dumps([{"ticker": "MSFT", "file": "old.htm"}, {"ticker": "AAPL", "file": "a.pdf"}]),
                    encoding="utf-8")
    replaced = acquire.register({"ticker": "MSFT", "file": "new.htm"}, path)
    assert replaced == [{"ticker": "MSFT", "file": "old.htm"}]
    assert [e["file"] for e in json.loads(path.read_text(encoding="utf-8"))] == ["a.pdf", "new.htm"]
    assert acquire.registered("msft", path) == [{"ticker": "MSFT", "file": "new.htm"}]


# --- downloads ---------------------------------------------------------------------------------


def test_only_https_downloads(tmp_path):
    with pytest.raises(acquire.AcquireError, match="https"):
        acquire.download("http://example.com/a.pdf", tmp_path / "a.pdf")


def test_download_checks_the_file_type_and_leaves_no_half_files(monkeypatch, tmp_path):
    monkeypatch.setattr(acquire, "_transport", httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>login</html>")))
    with pytest.raises(acquire.AcquireError, match="not a %PDF"):
        acquire.download("https://example.com/a.pdf", tmp_path / "a.pdf", expect=b"%PDF")
    assert list(tmp_path.iterdir()) == []


def test_download_size_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(acquire, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(acquire, "_transport", httpx.MockTransport(lambda r: httpx.Response(200, content=b"%PDF" + b"x" * 50)))
    with pytest.raises(acquire.AcquireError, match="larger than"):
        acquire.download("https://example.com/a.pdf", tmp_path / "a.pdf")
    assert list(tmp_path.iterdir()) == []


def test_sec_annual_report_downloads_the_latest_10k(monkeypatch, tmp_path):
    from src.mcp_servers import sec_server as sec
    from tests.test_sec_server import SUBMISSIONS, TICKERS

    def fake(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "submissions" in url:
            return httpx.Response(200, json=SUBMISSIONS)
        return httpx.Response(200, text=FILING)

    sec._cache.clear()
    monkeypatch.setattr(sec.settings, "sec_user_agent", "Test test@example.com")
    monkeypatch.setattr(sec, "_transport", httpx.MockTransport(fake))
    monkeypatch.setattr(sec, "MIN_REQUEST_GAP", 0)
    monkeypatch.setattr(acquire, "_transport", httpx.MockTransport(fake))
    entry = acquire.sec_annual_report("MSFT", raw_dir=tmp_path)
    assert entry["file"] == "MSFT-10-K-2026-06-30.htm" and (tmp_path / entry["file"]).exists()
    assert (entry["doc_type"], entry["filed"], entry["currency"], entry["source"]) == ("10-K", "2026-07-29", "USD", "sec")
    assert entry["source_url"].endswith("/789019/000119312526323660/msft-20260630.htm")

    prepared, stats = acquire.prepare(entry, raw_dir=tmp_path)
    assert prepared["reporting_unit"] == "USD millions" and prepared["pages"] == "1-3"
    assert stats["pages_chosen"] == 3 and stats["chunks"] >= 3
