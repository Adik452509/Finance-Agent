"""Split cleaned pages into overlapping chunks ready for embedding.

Pipeline position:  PDF  ->  loader  ->  [chunker]  ->  indexer  ->  Chroma

Design rules:
  - A chunk never spans two pages, so every chunk cites exactly one page.
  - Cuts prefer natural breaks (blank line > line end > sentence > space),
    so a figure like "6,83,102" is never split in half.
  - On table pages, later chunks repeat the table's column header (units and
    periods). Without it a chunk holding "43,787,709 46,079,610" cannot say
    which year each number belongs to - and column order differs by company
    (Toyota lists the older year first; Apple and Reliance the newer).

Try it:
    python -m src.ingestion.chunker
"""

import hashlib
import re
from dataclasses import dataclass

from config.settings import settings
from src.ingestion.loader import Page

# Cut points searched from the end of the window back to this fraction of it,
# so chunks never come out tiny just because a break appeared early.
MIN_CUT_FRACTION = 0.5
# Preferred cut points, best first. The cut lands just after the separator.
SEPARATORS = ("\n\n", "\n", ". ", " ")

# Table headers live near the top of a page.
HEADER_SCAN_LINES = 15
MAX_CONTEXT_CHARS = 300

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
UNIT_RE = re.compile(r"\bin (millions|thousands|billions|crore|lakhs?)\b|\bmillions of yen\b", re.I)
PERIOD_RE = re.compile(r"\b(years? ended|months ended|as at|as of|fiscal year)\b", re.I)
# US style "September 27," / "March 31, 2025" and Indian style "31st March, 2025".
DATE_RE = re.compile(
    rf"^(({_MONTHS})\s+\d{{1,2}},?(\s+(19|20)\d{{2}})?"
    rf"|\d{{1,2}}(st|nd|rd|th)?\s+({_MONTHS}),?\s+(19|20)\d{{2}})$",
    re.I,
)
# "2025", "2025 2024 2023", "Notes 2024-25 2023-24"
YEARS_RE = re.compile(r"^(notes\s+)?((19|20)\d{2}(-\d{2})?\s*)+$", re.I)
STATEMENT_RE = re.compile(
    r"balance sheets?|income statement|statements? of (operations|income|profit and loss|"
    r"cash flows|financial position|comprehensive income|changes in equity)",
    re.I,
)
# Reliance's headers are extracted with broken spacing: "CONSOLIDA TED FINANCIAL ST A TEMENTS".
SCOPE_HEADERS = {
    "consolidatedfinancialstatements": "CONSOLIDATED FINANCIAL STATEMENTS",
    "standalonefinancialstatements": "STANDALONE FINANCIAL STATEMENTS",
}


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str  # what gets stored and shown as the source quote
    page: Page

    @property
    def metadata(self) -> dict[str, str | int]:
        return {**self.page.metadata(), "chunk_id": self.id}

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model: chunk plus a document header.

        A chunk like "Total net sales 416,161" never says "Apple" or "2025".
        Prepending who/when/units makes it findable by questions that do.
        """
        m = self.page.meta
        header = (
            f"{m.company} ({m.ticker}) {m.doc_type}, fiscal year ended {m.fiscal_year_end}. "
            f"Figures in {m.reporting_unit}."
        )
        return f"{header}\n{self.text}"


def table_context(text: str) -> str | None:
    """Column-header lines (scope, statement, units, periods) from the top of a page.

    Returns None for prose pages: a header is only built when the page shows
    units or period labels, i.e. it actually looks like a financial table.
    """
    parts: list[str] = []
    looks_like_table = False
    for line in text.splitlines()[:HEADER_SCAN_LINES]:
        squashed = line.replace(" ", "").lower()
        scope = next((name for key, name in SCOPE_HEADERS.items() if key in squashed), None)
        if scope:
            parts.append(scope)
        elif len(line) <= 120 and STATEMENT_RE.search(line):
            parts.append(line)
        elif UNIT_RE.search(line) or (len(line) <= 60 and PERIOD_RE.search(line)):
            parts.append(line)
            looks_like_table = True
        elif len(line) <= 30 and (DATE_RE.match(line) or YEARS_RE.match(line)):
            parts.append(line)
            looks_like_table = True

    if not looks_like_table:
        return None
    # Two-page spreads repeat the scope header on each half; keep the first.
    # Period/date lines are NOT de-duplicated: "As at ... As at ..." pairs
    # carry column order.
    seen_scope: set[str] = set()
    kept: list[str] = []
    for part in parts:
        if part in SCOPE_HEADERS.values():
            if part in seen_scope:
                continue
            seen_scope.add(part)
        kept.append(part)
    return " ".join(kept)[:MAX_CONTEXT_CHARS]


def _find_cut(text: str, start: int, end: int) -> int:
    """Best cut position in text[start:end], searching backwards from end."""
    floor = start + int((end - start) * MIN_CUT_FRACTION)
    for sep in SEPARATORS:
        pos = text.rfind(sep, floor, end)
        if pos != -1:
            return pos + len(sep)
    return end  # no break at all: hard cut


def _next_start(text: str, start: int, cut: int, overlap: int) -> int:
    """Where the next chunk begins: `overlap` chars back, moved to a word boundary."""
    target = max(cut - overlap, start + 1)  # always move forward
    for sep in ("\n", " "):
        pos = text.find(sep, target, cut)
        if pos != -1:
            return pos + 1
    return cut


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into pieces of at most `size` chars with ~`overlap` shared chars."""
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            pieces.append(text[start:])
            break
        cut = _find_cut(text, start, end)
        pieces.append(text[start:cut])
        start = _next_start(text, start, cut, overlap)
    return [p.strip() for p in pieces if p.strip()]


def _chunk_id(page: Page, index: int, text: str) -> str:
    """Stable ID: same file + page + content => same ID on every run.

    The indexer uses this to skip chunks already embedded (resumable runs).
    The content hash means re-chunking with new settings yields new IDs
    instead of silently reusing vectors of different text.
    """
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", page.meta.file.rsplit(".", 1)[0])
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-p{page.page:03d}-c{index:02d}-{digest}"


def chunk_page(page: Page, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap

    context = table_context(page.text)
    chunks: list[Chunk] = []
    for i, piece in enumerate(split_text(page.text, size, overlap)):
        # The first piece already contains the page's header lines.
        if i > 0 and context:
            piece = f"[Table context from top of page: {context}]\n{piece}"
        chunks.append(Chunk(id=_chunk_id(page, i, piece), text=piece, page=page))
    return chunks


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    return [chunk for page in pages for chunk in chunk_page(page)]


if __name__ == "__main__":
    from collections import Counter

    from src.ingestion.loader import load_all

    chunks = chunk_pages(load_all())
    lengths = [len(c.text) for c in chunks]
    print(f"{len(chunks)} chunks  (min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)} chars)")
    for file, n in Counter(c.page.meta.file for c in chunks).items():
        print(f"  {file:45} {n:4} chunks")
    with_ctx = sum(c.text.startswith("[Table context") for c in chunks)
    print(f"  chunks with repeated table header: {with_ctx}")
    print(f"  duplicate IDs: {len(chunks) - len({c.id for c in chunks})}")

    sample = next(c for c in chunks if c.page.meta.ticker == "7203.T" and c.text.startswith("[Table context"))
    print(f"\nSample: {sample.id}")
    print(sample.text[:400])
