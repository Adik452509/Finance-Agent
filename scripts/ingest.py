"""Index the registered PDFs into Chroma.

Run from the project root:
    python scripts/ingest.py --dry-run   # counts + time estimate, spends no quota
    python scripts/ingest.py             # embed what's missing (resumable)
    python scripts/ingest.py --reset     # wipe the collection and rebuild

Exits with code 0 when every chunk is indexed, 1 if the run stopped early.
"""

import argparse
import logging
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

# scripts/ is not a package: make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from src.ingestion.chunker import chunk_pages  # noqa: E402
from src.ingestion.indexer import get_collection, index_chunks  # noqa: E402
from src.ingestion.loader import load_all  # noqa: E402

warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="show what would be embedded, spend no quota")
    parser.add_argument("--reset", action="store_true", help="delete the collection and rebuild from scratch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    chunks = chunk_pages(load_all())
    print(f"\n{len(chunks)} chunks from {len({c.page.meta.file for c in chunks})} documents "
          f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    for file, n in Counter(c.page.meta.file for c in chunks).items():
        print(f"  {file:45} {n:4} chunks")

    if args.dry_run:
        if args.reset:
            print("\n--dry-run ignores --reset; nothing was deleted.")
        # Opening the collection reads state but embeds nothing.
        collection = get_collection()
        stored = set(collection.get(ids=[c.id for c in chunks], include=[])["ids"])
        todo = len(chunks) - len(stored)
        batches = math.ceil(todo / settings.embed_batch_size)
        print(f"\nAlready indexed: {len(stored)}   To embed: {todo}   "
              f"Estimated time: ~{batches} min ({batches} batches of {settings.embed_batch_size})")
        return 0

    collection = get_collection(reset=args.reset)
    report = index_chunks(chunks, collection)

    print("\n" + "-" * 60)
    print(f"Chunks total     : {report.total}")
    print(f"Already indexed  : {report.already_indexed}")
    print(f"Embedded now     : {report.embedded}")
    print(f"Stale removed    : {report.removed_stale}")
    print(f"In collection    : {collection.count()}")
    if report.complete:
        print("Status           : COMPLETE")
        return 0
    print(f"Status           : STOPPED - {report.stopped_reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
