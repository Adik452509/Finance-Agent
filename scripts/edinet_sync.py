"""Build / update the local index of EDINET filings (Japanese companies' official reports).

EDINET can only list filings day by day, so the EDINET server looks companies up in a local
index of those lists. Run this once (about 400 requests, ~7-10 minutes); after that the
server tops up the last few days by itself. Safe to stop and re-run: it resumes.

    python scripts/edinet_sync.py              # last 400 days
    python scripts/edinet_sync.py --days 30    # just the last month
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402
from src.mcp_servers.edinet_server import (  # noqa: E402
    FULL_INDEX_DAYS, INDEX_PATH, EdinetError, load_index, recent_days, sync_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=FULL_INDEX_DAYS, help="how many days back to cover")
    args = parser.parse_args()
    if not settings.edinet_api_key:
        print("EDINET_API_KEY is missing in .env (free key: https://api.edinet-fsa.go.jp/api/auth/index.aspx?mode=1)")
        return 1

    before = load_index()
    todo = [d for d in recent_days(args.days) if d.isoformat() not in set(before["synced"])]
    print(f"Index: {len(before['synced'])} days, {len(before['docs'])} reports. Fetching {len(todo)} day(s)...")

    def progress(n: int, total: int, day: str) -> None:
        if n % 25 == 0 or n == total:
            print(f"  {n}/{total} (last: {day})", flush=True)

    try:
        index = sync_index(todo, before, progress=progress)
    except EdinetError as e:
        print(f"Stopped: EDINET returned {e.status}. {'Check EDINET_API_KEY.' if e.status == 401 else 'Re-run to resume.'}")
        return 1
    except KeyboardInterrupt:
        print("Stopped. Progress is saved; re-run to resume.")
        return 1
    print(f"Done: {len(index['synced'])} days, {len(index['docs'])} annual / semi-annual reports -> {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
