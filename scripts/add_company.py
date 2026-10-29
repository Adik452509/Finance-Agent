"""Add (or update) a company's latest annual report in the filings index.

    python scripts/add_company.py MSFT --dry-run       # download + choose pages, embed nothing
    python scripts/add_company.py MSFT                 # US-listed: latest 10-K / 20-F from the SEC
    python scripts/add_company.py 6758.T               # Tokyo-listed: Annual Securities Report from EDINET
    python scripts/add_company.py INFY.NS --pdf-url https://.../annual-report.pdf \\
        --company "Infosys Limited" --market India --doc-type "Annual Report" \\
        --fiscal-year-end 2025-03-31 --currency INR --unit "INR crore"

Only the most useful pages are embedded (--max-chunks, default 250), because the free
Gemini tier embeds ~1,000 chunks a day. Re-running resumes; a newer report replaces the
older one. Restart the app afterwards so the agents see the new company.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402
from src.ingestion.acquire import (  # noqa: E402
    DEFAULT_MAX_CHUNKS, AcquireError, edinet_annual_report, pdf_from_url, prepare, register, registered,
    sec_annual_report,
)
from src.ingestion.chunker import chunk_pages  # noqa: E402
from src.ingestion.indexer import get_collection, index_chunks  # noqa: E402
from src.ingestion.loader import DocumentMeta, load_document  # noqa: E402

warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")


def fetch(args) -> dict:
    if args.pdf_url:
        missing = [n for n in ("company", "market", "doc_type", "fiscal_year_end", "currency") if not getattr(args, n)]
        if missing:
            raise AcquireError("--pdf-url also needs: " + ", ".join("--" + m.replace("_", "-") for m in missing))
        return pdf_from_url(args.pdf_url, {"company": args.company, "ticker": args.ticker.upper(),
                                           "market": args.market, "doc_type": args.doc_type,
                                           "fiscal_year_end": args.fiscal_year_end, "currency": args.currency,
                                           "reporting_unit": args.unit or ""})
    if args.ticker.upper().endswith(".T"):
        return edinet_annual_report(args.ticker)
    if "." in args.ticker:
        raise AcquireError(f"{args.ticker} is not US- or Tokyo-listed: download its annual report yourself and "
                           "pass the link with --pdf-url (plus --company, --market, --doc-type, --fiscal-year-end, "
                           "--currency).")
    return sec_annual_report(args.ticker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker")
    parser.add_argument("--dry-run", action="store_true", help="download and choose pages, but embed nothing")
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--force", action="store_true", help="replace a report you added by hand")
    for name in ("pdf-url", "company", "market", "doc-type", "fiscal-year-end", "currency", "unit"):
        parser.add_argument(f"--{name}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    existing = registered(args.ticker)
    if any(not e.get("source") for e in existing) and not args.force:
        print(f"{args.ticker} already has a report added by hand ({existing[0]['file']}). Use --force to replace it.")
        return 1
    try:
        entry = fetch(args)
        entry, stats = prepare(entry, args.max_chunks)
    except AcquireError as e:
        print(f"Could not add {args.ticker}: {e}")
        return 1

    print(f"\n{entry['company']} ({entry['ticker']}) {entry['doc_type']} for the fiscal year ended "
          f"{entry['fiscal_year_end']}, filed {entry['filed']}")
    print(f"  file: {entry['file']}   unit: {entry['reporting_unit']}")
    print(f"  pages: {stats['pages_chosen']} of {stats['pages_total']} chosen ({entry['pages']})")
    print(f"  chunks to embed: {stats['chunks']} (~{stats['minutes']} min on the free tier, one batch a minute)")
    if any(e.get("filed") == entry["filed"] and e.get("file") == entry["file"] for e in existing):
        print("  already registered: indexing resumes / checks what is missing")
    if args.dry_run:
        print("\n--dry-run: documents.json and the index were not changed.")
        return 0

    replaced = register(entry)
    collection = get_collection()
    for old in replaced:
        if old["file"] != entry["file"]:
            collection.delete(where={"file": old["file"]})
            print(f"  replaced older report {old['file']}")
            if old.get("source"):  # only files this script downloaded are deleted
                (settings.raw_dir / old["file"]).unlink(missing_ok=True)

    meta = DocumentMeta(**entry)
    report = index_chunks(chunk_pages(load_document(meta)), collection)
    print(f"\nEmbedded now: {report.embedded}   already indexed: {report.already_indexed}   "
          f"in collection: {collection.count()}")
    if not report.complete:
        print(f"STOPPED - {report.stopped_reason}\nRe-run the same command to continue.")
        return 1
    print("COMPLETE. Restart the app (or re-run your script) so the agents see the new report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
