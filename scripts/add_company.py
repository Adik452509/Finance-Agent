"""Add (or update) a company's latest annual report in the filings index.

    python scripts/add_company.py MSFT --dry-run       # download + choose pages, embed nothing
    python scripts/add_company.py MSFT                 # US-listed: latest 10-K / 20-F from the SEC
    python scripts/add_company.py 6758.T               # Tokyo-listed: Annual Securities Report from EDINET
    python scripts/add_company.py INFY.NS --pdf-url https://.../annual-report.pdf \\
        --company "Infosys Limited" --market India --doc-type "Annual Report" \\
        --fiscal-year-end 2025-03-31 --currency INR --unit "INR crore"

Only the most useful pages are embedded (--max-chunks, default 250), because the free
Gemini tier embeds ~1,000 chunks a day. Re-running resumes; a newer report replaces the
older one. Restart the app afterwards so the agents see the new company. The same work
runs in the deployed app as an admin job (src/ingestion/acquire.add_company).
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.acquire import DEFAULT_MAX_CHUNKS, AcquireError, add_company  # noqa: E402

warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")


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

    pdf = None
    if args.pdf_url:
        missing = [n for n in ("company", "market", "doc_type", "fiscal_year_end", "currency") if not getattr(args, n)]
        if missing:
            print("--pdf-url also needs: " + ", ".join("--" + m.replace("_", "-") for m in missing))
            return 1
        pdf = {"url": args.pdf_url, "company": args.company, "market": args.market, "doc_type": args.doc_type,
               "fiscal_year_end": args.fiscal_year_end, "currency": args.currency, "reporting_unit": args.unit or ""}
    try:
        result = add_company(args.ticker, pdf=pdf, max_chunks=args.max_chunks, force=args.force,
                             dry_run=args.dry_run, progress=lambda message: print(f"  {message}", flush=True))
    except AcquireError as e:
        print(f"Could not add {args.ticker}: {e}")
        return 1

    entry = result["entry"]
    print(f"\n  file: {entry['file']}   unit: {entry['reporting_unit']}   pages: {entry['pages']}")
    if args.dry_run:
        print("\n--dry-run: the registry and the index were not changed.")
        return 0
    for old in result.get("replaced", []):
        print(f"  replaced older report {old}")
    print(f"\nEmbedded now: {result['embedded']}   already indexed: {result['already_indexed']}")
    if not result["complete"]:
        print(f"STOPPED - {result['stopped_reason']}\nRe-run the same command to continue.")
        return 1
    print("COMPLETE. Restart the app (or re-run your script) so the agents see the new report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
