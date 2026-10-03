"""Chunking rules: sizes, clean cut points, table headers, stable IDs. Offline."""

from src.ingestion.chunker import chunk_page, split_text, table_context
from src.ingestion.loader import DocumentMeta, Page

RELIANCE = DocumentMeta(
    file="ril.pdf", company="Reliance Industries Limited", ticker="RELIANCE.NS", market="India",
    doc_type="Annual Report", fiscal_year_end="2025-03-31", currency="INR", reporting_unit="INR crore",
)

RELIANCE_HEADER = (
    "CONSOLIDA TED FINANCIAL ST A TEMENTS\n"
    "CONSOLIDA TED FINANCIAL ST A TEMENTS\n"  # spreads print it once per half
    "(₹ in crore)\n"
    "Notes As at\n"
    "31st March, 2025\n"
    "As at\n"
    "31st March, 2024\n"
)

APPLE_HEADER = (
    "Apple Inc.\n"
    "CONSOLIDATED STATEMENTS OF OPERATIONS\n"
    "(In millions, except number of shares, which are reflected in thousands, and per-share amounts)\n"
    "Years ended\n"
    "September 27,\n"
    "2025\n"
    "September 28,\n"
    "2024\n"
    "Net sales:\n"
    "Products $ 307,003 $ 294,866 $ 298,085\n"
)


# --- split_text ----------------------------------------------------------------


def test_short_text_is_one_piece():
    assert split_text("Revenue grew.", size=100, overlap=10) == ["Revenue grew."]


def test_pieces_respect_size_and_cut_between_words():
    words = [f"word{i}" for i in range(2000)]
    pieces = split_text(" ".join(words), size=500, overlap=100)
    vocab = set(words)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 500
        tokens = piece.split()
        assert tokens[0] in vocab and tokens[-1] in vocab  # no half-words at the edges


def test_consecutive_pieces_overlap_and_advance():
    text = " ".join(f"word{i}" for i in range(2000))
    pieces = split_text(text, size=500, overlap=100)
    for a, b in zip(pieces, pieces[1:]):
        assert set(a.split()) & set(b.split()), "neighbours should share overlap text"
        assert text.index(b) > text.index(a), "each piece must start further on"


def test_figures_are_never_split():
    text = " ".join(["6,83,102"] * 400)
    for piece in split_text(text, size=300, overlap=50):
        assert set(piece.split()) == {"6,83,102"}


def test_prefers_paragraph_break():
    para1 = " ".join(["alpha"] * 60)
    para2 = " ".join(["beta"] * 60)
    pieces = split_text(f"{para1}\n\n{para2}", size=500, overlap=50)
    assert pieces[0] == para1


# --- table_context ------------------------------------------------------------


def test_indian_balance_sheet_header_keeps_dates_and_dedupes_scope():
    ctx = table_context(RELIANCE_HEADER + "Borrowings 20 1,10,631 1,01,910\n")
    assert ctx == (
        "CONSOLIDATED FINANCIAL STATEMENTS (₹ in crore) Notes As at 31st March, 2025 As at 31st March, 2024"
    )


def test_us_income_statement_header_keeps_column_order():
    ctx = table_context(APPLE_HEADER)
    assert "CONSOLIDATED STATEMENTS OF OPERATIONS" in ctx
    assert ctx.index("September 27,") < ctx.index("2025") < ctx.index("September 28,") < ctx.index("2024")


def test_prose_page_has_no_table_context():
    prose = "Reliance continued to capitalise on energy-saving opportunities across its businesses.\n" * 5
    assert table_context(prose) is None


# --- chunk_page ---------------------------------------------------------------


def _table_page(rows: int = 60) -> Page:
    body = "Borrowings 20 1,10,631 1,01,910\n" * rows
    return Page(text=RELIANCE_HEADER + body, page=99, page_label="194 195", meta=RELIANCE)


def test_later_chunks_repeat_table_header():
    chunks = chunk_page(_table_page(), size=300, overlap=50)
    assert len(chunks) > 2
    assert not chunks[0].text.startswith("[Table context")  # already has the real header
    for chunk in chunks[1:]:
        assert chunk.text.startswith(
            "[Table context from top of page: CONSOLIDATED FINANCIAL STATEMENTS (₹ in crore) "
            "Notes As at 31st March, 2025 As at 31st March, 2024]"
        )


def test_chunk_metadata_carries_citation_fields():
    meta = chunk_page(_table_page(), size=300, overlap=50)[1].metadata
    assert meta["ticker"] == "RELIANCE.NS"
    assert meta["page"] == 99 and meta["page_label"] == "194 195"
    assert meta["reporting_unit"] == "INR crore"


def test_embedding_text_names_company_year_and_units():
    chunk = chunk_page(_table_page(), size=300, overlap=50)[0]
    first_line = chunk.embedding_text.splitlines()[0]
    assert "Reliance Industries Limited" in first_line
    assert "2025-03-31" in first_line and "INR crore" in first_line


def test_ids_are_stable_and_content_sensitive():
    first = [c.id for c in chunk_page(_table_page(), size=300, overlap=50)]
    again = [c.id for c in chunk_page(_table_page(), size=300, overlap=50)]
    changed = [c.id for c in chunk_page(_table_page(rows=61), size=300, overlap=50)]
    assert first == again
    assert len(set(first)) == len(first)
    assert first != changed
