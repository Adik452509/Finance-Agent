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

    def path(self, raw_dir: Path | None = None) -> Path:
        return (raw_dir or settings.raw_dir) / self.file


@dataclass(frozen=True)
class Page:
    """Cleaned text of one PDF page plus everything needed to cite it."""

    text: str
    page: int  # 1-based PDF page index: what a PDF viewer jumps to
    page_label: str | None  # number printed on the page, e.g. "23", "F-12", "194 195"
    meta: DocumentMeta

    def metadata(self) -> dict[str, str | int]:
        """Flat metadata for the vector store (Chroma rejects None values)."""
        data: dict[str, str | int] = {**asdict(self.meta), "page": self.page}
        if self.page_label:
            data["page_label"] = self.page_label
        return data


# --- Registry ----------------------------------------------------------------


def load_registry(path: Path | None = None) -> list[DocumentMeta]:
    """Read and validate config/documents.json."""
    path = path or settings.documents_registry
    entries = json.loads(path.read_text(encoding="utf-8"))

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

        docs.append(DocumentMeta(**{f: str(entry[f]) for f in REQUIRED_FIELDS}))
    return docs


def check_raw_dir(docs: list[DocumentMeta], raw_dir: Path | None = None) -> None:
    """Fail on registered files that are missing; warn on PDFs nobody registered."""
    raw_dir = raw_dir or settings.raw_dir
    missing = [d.file for d in docs if not d.path(raw_dir).exists()]
    if missing:
        raise FileNotFoundError(f"Registered but not found in {raw_dir}: {', '.join(missing)}")

    registered = {d.file for d in docs}
    for pdf in sorted(raw_dir.glob("*.pdf")):
        if pdf.name not in registered:
            logger.warning("Skipping %s: not listed in %s", pdf.name, settings.documents_registry.name)


# --- Text cleaning -----------------------------------------------------------


def normalize_text(text: str, currency: str) -> str:
    """Fix extraction artefacts without changing any figures."""
    text = text.replace(" ", " ").replace("\t", " ")
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


def load_document(meta: DocumentMeta, raw_dir: Path | None = None) -> list[Page]:
    """Extract, clean and return the usable pages of one registered PDF."""
    path = meta.path(raw_dir)
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

    pages: list[Page] = []
    skipped = 0
    for number, lines in enumerate(pages_lines, start=1):
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
