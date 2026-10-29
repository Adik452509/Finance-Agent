"""Check whether the companies added with add_company.py have published a newer annual report.

    python scripts/check_new_filings.py            # report only
    python scripts/check_new_filings.py --update   # also fetch and index the newer reports

Meant to run once a day (e.g. Windows Task Scheduler). Reports added by hand or from a
--pdf-url are skipped: there is no source to ask.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from src.ingestion.acquire import latest_filing_date  # noqa: E402
from src.ingestion.loader import load_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="fetch and index newer reports")
    args = parser.parse_args()

    newer = []
    for doc in load_registry(settings.documents_registry):
        if doc.source not in ("sec", "edinet"):
            continue
        try:
            latest = latest_filing_date({"source": doc.source, "ticker": doc.ticker})
        except Exception as e:  # one company's problem must not stop the others
            print(f"  {doc.ticker:10} could not check: {type(e).__name__}: {str(e)[:120]}")
            continue
        status = "NEWER REPORT AVAILABLE" if latest and latest > doc.filed else "up to date"
        print(f"  {doc.ticker:10} indexed: filed {doc.filed}   latest: {latest or '?'}   {status}")
        if latest and latest > doc.filed:
            newer.append(doc.ticker)

    if newer and args.update:
        for ticker in newer:
            print(f"\nUpdating {ticker} ...")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "add_company.py"), ticker], check=False)
    elif newer:
        print(f"\n{len(newer)} newer report(s). Run with --update to index them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
