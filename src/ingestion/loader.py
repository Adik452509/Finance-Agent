"""Load registered PDFs into clean, per-page records with document metadata.

Pipeline position:  PDF  ->  [loader]  ->  chunker  ->  indexer  ->  Chroma

Each PDF must be listed in config/documents.json. That registry, not the file
name, is the source of truth for company, ticker, fiscal year and units.

Try it:
    python -m src.ingestion.loader
"""

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config.settings import settings

logger = logging.getLogger(__name__)

# pypdf logs harmless warnings ("Ignoring wrong pointing object") on many PDFs.
logging.getLogger("pypdf").setLevel(logging.ERROR)

REQUIRED_FIELDS = (
    "file", "company", "ticker", "market", "doc_type",
    "fiscal_year_end", "currency", "reporting_unit",
)
# Written by scripts/add_company.py: where the filing came from, and which pages to index.
OPTIONAL_FIELDS = ("source", "source_url", "filed", "pages")
HTML_SUFFIXES = (".htm", ".html")

# Pages with less text than this after cleaning (chart-only pages, blank
# separators) are skipped: they cost embedding quota and never answer anything.
MIN_PAGE_CHARS = 100

# Header/footer detection looks only at the first/last few lines of each page.
EDGE_LINES = 3
# A boilerplate line must repeat on at least this share of pages.
REPEAT_THRESHOLD = 0.25

# A line that is only a page number: "23", "F-12", or a spread like "194 195".
PAGE_NUMBER_RE = re.compile(r"^(?:F-)?\d{1,4}(?: \d{1,4})?$")


@dataclass(frozen=True)
class DocumentMeta:
    """One entry of config/documents.json."""

    file: str
    company: str
    ticker: str
    market: str
    doc_type: str
    fiscal_year_end: str
    currency: str
    reporting_unit: str
    source: str = ""        # "sec", "edinet", "url" - or "" for a PDF added by hand
    source_url: str = ""
    filed: str = ""         # filing date, used to spot newer reports
    pages: str = ""         # "1-40,55-90": index only these pages ("" = all)

    def path(self, raw_dir: Path | None = None) -> Path:
        return (raw_dir or settings.raw_dir) / self.file

    def selected_pages(self) -> set[int] | None:
        return parse_page_ranges(self.pages) if self.pages else None


@dataclass(frozen=True)
class Page:
    """Cleaned text of one PDF page plus everything needed to cite it."""

    text: str
    page: int  # 1-based PDF page index: what a PDF viewer jumps to
    page_label: str | None  # number printed on the page, e.g. "23", "F-12", "194 195"
    meta: DocumentMeta

    def metadata(self) -> dict[str, str | int]:
        """Flat metadata for the vector store (Chroma rejects None values)."""
        meta = {k: v for k, v in asdict(self.meta).items() if k in REQUIRED_FIELDS or (v and k != "pages")}
        data: dict[str, str | int] = {**meta, "page": self.page}
        if self.page_label:
            data["page_label"] = self.page_label
        return data


# --- Registry ----------------------------------------------------------------


def registry_entries() -> list[dict]:
    """The raw registry: the documents table (FILINGS_STORE=postgres) or config/documents.json."""
    if settings.filings_store == "postgres":
        from src.db.repository import document_entry, list_documents
        from src.db.session import session_scope

        with session_scope() as db:
            return [document_entry(d) for d in list_documents(db)]
    return json.loads(settings.documents_registry.read_text(encoding="utf-8"))


def load_registry(path: Path | None = None) -> list[DocumentMeta]:
    """Read and validate the registry (a given JSON file, or wherever FILINGS_STORE says)."""
    entries = json.loads(path.read_text(encoding="utf-8")) if path else registry_entries()
    path = path or settings.documents_registry  # only used to name the source in error messages

    docs: list[DocumentMeta] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
        if missing:
            raise ValueError(f"{path.name} entry #{i + 1} is missing: {', '.join(missing)}")

        # Guardrail: only bare file names, so a registry entry can never point
        # outside data/raw (e.g. "../../.env").
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError(f"{path.name} entry #{i + 1}: 'file' must be a plain file name, got {entry['file']!r}")

        if entry["file"] in seen:
            raise ValueError(f"{path.name} lists {entry['file']!r} more than once")
        seen.add(entry["file"])

        optional = {f: str(entry[f]) for f in OPTIONAL_FIELDS if entry.get(f)}
        if optional.get("pages"):
            parse_page_ranges(optional["pages"])  # fail here, with the entry number, not mid-ingest
        docs.append(DocumentMeta(**{f: str(entry[f]) for f in REQUIRED_FIELDS}, **optional))
    return docs


def parse_page_ranges(spec: str) -> set[int]:
    """'1-3,7' -> {1, 2, 3, 7}."""
    pages: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ValueError(f"bad page range {part!r} in {spec!r}")
        start, end = int(match.group(1)), int(match.group(2) or match.group(1))
        if end < start:
            raise ValueError(f"bad page range {part!r} in {spec!r}")
        pages.update(range(start, end + 1))
    return pages


def format_page_ranges(pages: list[int]) -> str:
    """[1, 2, 3, 7] -> '1-3,7' (the compact form stored in documents.json)."""
    parts, ordered = [], sorted(set(pages))
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        parts.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(parts)


def check_raw_dir(docs: list[DocumentMeta], raw_dir: Path | None = None) -> None:
    """Fail on registered files that are missing; warn on PDFs nobody registered."""
    raw_dir = raw_dir or settings.raw_dir
    missing = [d.file for d in docs if not d.path(raw_dir).exists()]
    if missing:
        raise FileNotFoundError(f"Registered but not found in {raw_dir}: {', '.join(missing)}")

    registered = {d.file for d in docs}
    for pdf in sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in (".pdf", *HTML_SUFFIXES)):
        if pdf.name not in registered:
            logger.warning("Skipping %s: not listed in %s", pdf.name, settings.documents_registry.name)


# --- Text cleaning -----------------------------------------------------------


def normalize_text(text: str, currency: str) -> str:
    """Fix extraction artefacts without changing any figures."""
    text = text.replace("\u00a0", " ").replace("\t", " ")
    # Dot leaders in tables ("Sales revenues ........ 43,787,709") are noise.
    # 4+ dots only, so a real ellipsis "..." survives.
    text = re.sub(r"\.{4,}", " ", text)

    if currency == "INR":
        # The rupee sign is extracted as "C" or "`" in Reliance's fonts.
        text = re.sub(r"\(\s*[C`]\s*in crore\s*\)", "(₹ in crore)", text)
        text = re.sub(r"`(?=\s?\d)", "₹", text)

    # Re-join words hyphenated across a line break ("reve-\nnue" -> "revenue").
    # Lowercase on both sides only, so ranges like "2024-\n25" stay intact.
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _signature(line: str) -> str:
    """Line with digits masked, so 'Form 10-K | 23' and '| 24' compare equal."""
    return re.sub(r"\d+", "#", line)


def find_boilerplate(pages_lines: list[list[str]], company: str) -> set[str]:
    """Signatures of repeated header/footer lines that mention the company.

    Requiring the company name keeps meaningful repeated headers such as
    'CONSOLIDATED FINANCIAL STATEMENTS' vs 'STANDALONE FINANCIAL STATEMENTS',
    which tell a reader which kind of figures a page contains.
    """
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        counts.update({_signature(ln) for ln in lines[:EDGE_LINES] + lines[-EDGE_LINES:]})

    company_word = company.split()[0].lower()  # "apple", "reliance", "toyota"
    min_pages = max(2, int(len(pages_lines) * REPEAT_THRESHOLD))
    return {
        sig for sig, n in counts.items()
        if n >= min_pages and company_word in sig.lower()
    }


def strip_edges(lines: list[str], boilerplate: set[str]) -> tuple[list[str], str | None]:
    """Remove page numbers and boilerplate from a page's edges.

    Returns the remaining lines and the printed page label, if one was found.
    """
    label: str | None = None
    edge = set(range(min(EDGE_LINES, len(lines)))) | set(range(max(0, len(lines) - EDGE_LINES), len(lines)))

    kept: list[str] = []
    for i, line in enumerate(lines):
        if i in edge and PAGE_NUMBER_RE.match(line):
            label = label or line
            continue
        if i in edge and _signature(line) in boilerplate:
            # Footers like "Apple Inc. | 2025 Form 10-K | 23" carry the page number.
            trailing = re.search(r"\|\s*(\d{1,4})$", line)
            if trailing and not label:
                label = trailing.group(1)
            continue
        kept.append(line)
    return kept, label


# --- Loading -----------------------------------------------------------------


class _HTMLText(HTMLParser):
    """Visible text of a filing's HTML, split into pages at its page breaks.

    SEC filings are HTML; their print layout marks each page end with a CSS page break
    (`<hr style="page-break-after:always">`), and the footer above it carries the
    printed page number - so pages, and citations, match the printed document.
    """

    BLOCKS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}
    HIDDEN = {"script", "style", "head", "title", "ix:header"}
    VOID = {"br", "hr", "img", "meta", "link", "input"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: list[list[str]] = [[]]
        self._stack: list[tuple[str, bool, bool]] = []  # (tag, hidden, page break after)
        self._hidden = 0
        self._in_cell = 0  # inside <td>/<th>: <div>/<p> wrappers must not break the row's line

    def _break(self) -> None:
        if any(s.strip() for s in self.pages[-1]):
            self.pages.append([])

    def handle_starttag(self, tag, attrs):
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        hidden = tag in self.HIDDEN or "display:none" in style
        after = "page-break-after:always" in style or "break-after:page" in style
        if "page-break-before:always" in style or "break-before:page" in style:
            self._break()
        if tag in ("td", "th"):
            self.pages[-1].append(" ")
            self._in_cell += 1
        elif tag in self.BLOCKS and (not self._in_cell or tag == "tr"):
            self.pages[-1].append("\n")
        if tag in self.VOID:
            if after:
                self._break()
            return
        self._stack.append((tag, hidden, after))
        self._hidden += hidden

    def handle_endtag(self, tag):
        while self._stack:  # tolerate unclosed tags: pop until the matching one
            open_tag, hidden, after = self._stack.pop()
            self._hidden -= hidden
            if open_tag in ("td", "th"):
                self._in_cell = max(0, self._in_cell - 1)
            if after:
                self._break()
            if open_tag == tag:
                break
        if tag in self.BLOCKS and not self._in_cell:
            self.pages[-1].append("\n")

    def handle_data(self, data):
        if not self._hidden:
            # As in a browser, line breaks in the HTML source are just spaces; lines come from tags.
            self.pages[-1].append(re.sub(r"\s+", " ", data))

    def text_pages(self) -> list[str]:
        pages = []
        for parts in self.pages:
            text = "".join(parts)
            text = re.sub(r"\(\s+(?=\d)|(?<=\d)\s+\)", lambda m: m.group().strip(), text)  # "(4,901 )" -> "(4,901)"
            text = re.sub(r"\$\s+(?=[\d(])", "$", text)                                    # "$ 64,696" -> "$64,696"
            pages.append(text)
        while pages and not pages[-1].strip():  # whatever follows the last page break
            pages.pop()
        return pages


def html_pages(path: Path) -> list[str]:
    parser = _HTMLText()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    return parser.text_pages()


def load_document(meta: DocumentMeta, raw_dir: Path | None = None) -> list[Page]:
    """Extract, clean and return the usable pages of one registered PDF or HTML filing."""
    path = meta.path(raw_dir)
    if path.suffix.lower() in HTML_SUFFIXES:
        raw_pages = html_pages(path)
    else:
        try:
            reader = PdfReader(path)
            raw_pages = [page.extract_text() or "" for page in reader.pages]
        except PdfReadError as e:
            raise RuntimeError(f"Could not read {path.name}: {e}") from e

    pages_lines = [
        [ln for ln in normalize_text(t, meta.currency).splitlines() if ln]
        for t in raw_pages
    ]
    boilerplate = find_boilerplate(pages_lines, meta.company)

    selected = meta.selected_pages()
    pages: list[Page] = []
    skipped = 0
    for number, lines in enumerate(pages_lines, start=1):
        if selected is not None and number not in selected:
            continue
        kept, label = strip_edges(lines, boilerplate)
        text = "\n".join(kept)
        if len(text) < MIN_PAGE_CHARS:
            skipped += 1
            continue
        pages.append(Page(text=text, page=number, page_label=label, meta=meta))

    logger.info("%s: %d pages kept, %d near-empty skipped", meta.file, len(pages), skipped)
    return pages


def load_all(raw_dir: Path | None = None) -> list[Page]:
    """Load every registered document."""
    docs = load_registry()
    check_raw_dir(docs, raw_dir)
    pages: list[Page] = []
    for meta in docs:
        pages.extend(load_document(meta, raw_dir))
    return pages


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    all_pages = load_all()
    by_file = Counter(p.meta.file for p in all_pages)
    for file, n in by_file.items():
        chars = sum(len(p.text) for p in all_pages if p.meta.file == file)
        print(f"{file:45} {n:4} pages  {chars:>10,} chars")
    sample = next(p for p in all_pages if p.meta.ticker == "RELIANCE.NS" and "Balance Sheet" in p.text)
    print(f"\nSample: {sample.meta.company}, PDF page {sample.page} (printed {sample.page_label})")
    print(sample.metadata())
    print(sample.text[:500])
