"""Copy the local filings library (Chroma + documents.json) into Postgres + pgvector (Phase 6.2).

    python scripts/migrate_to_postgres.py            # migrate, then verify
    python scripts/migrate_to_postgres.py --verify   # only compare the two stores

The vectors are COPIED, not re-created, so no Gemini quota is spent. Resumable: chunks
already in Postgres are skipped. Needs DATABASE_URL (Supabase) in .env. Afterwards set
FILINGS_STORE=postgres to make the app read from Postgres.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402

PAGE = 100  # chunks per round trip: ~1.2 MB of vectors in binary


def upgrade_schema() -> None:
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config(str(settings.project_root / "alembic.ini")), "head")


def copy_chunks(chroma, pg) -> int:
    done = set(pg.get(include=[])["ids"])
    total = chroma.count()
    print(f"Chunks: {total} in Chroma, {len(done)} already in Postgres")
    copied, offset = 0, 0
    while offset < total:
        page = chroma.get(limit=PAGE, offset=offset, include=["embeddings", "documents", "metadatas"])
        offset += len(page["ids"])
        todo = [i for i, id_ in enumerate(page["ids"]) if id_ not in done]
        if todo:
            pg.upsert(ids=[page["ids"][i] for i in todo], embeddings=[page["embeddings"][i] for i in todo],
                      documents=[page["documents"][i] for i in todo], metadatas=[page["metadatas"][i] for i in todo])
            copied += len(todo)
        print(f"  {offset}/{total} read, {copied} copied", flush=True)
        if not page["ids"]:
            break
    return copied


def copy_registry(chroma) -> None:
    import json

    from src.db.repository import list_documents, replace_document
    from src.db.session import session_scope

    entries = json.loads(settings.documents_registry.read_text(encoding="utf-8"))
    per_file = Counter(m["file"] for m in chroma.get(include=["metadatas"])["metadatas"])
    with session_scope() as db:
        for entry in entries:
            replace_document(db, entry, status="ready", chunks=per_file.get(entry["file"], 0))
        print("Documents table:", ", ".join(f"{d.ticker} ({d.chunks} chunks)" for d in list_documents(db)))


def verify(chroma, pg) -> bool:
    """Same chunks per company, and the same search results for real stored vectors."""
    ok = True
    chroma_files = Counter(m["ticker"] for m in chroma.get(include=["metadatas"])["metadatas"])
    pg_files = Counter(m["ticker"] for m in pg.get(include=["metadatas"])["metadatas"])
    for ticker in sorted(set(chroma_files) | set(pg_files)):
        same = chroma_files[ticker] == pg_files[ticker]
        ok &= same
        print(f"  {ticker:12} Chroma {chroma_files[ticker]:5}   Postgres {pg_files[ticker]:5}   {'OK' if same else 'MISMATCH'}")

    # Search parity without spending Gemini quota: use stored chunk vectors as the queries.
    sample = chroma.get(limit=5, offset=137, include=["embeddings", "metadatas"])
    agree = 0
    for vector, meta in zip(sample["embeddings"], sample["metadatas"]):
        where = {"ticker": meta["ticker"]}
        a = chroma.query(query_embeddings=[vector], n_results=5, where=where, include=["distances"])
        b = pg.query(query_embeddings=[vector], n_results=5, where=where, include=["distances"])
        overlap = len(set(a["ids"][0]) & set(b["ids"][0]))
        agree += a["ids"][0][0] == b["ids"][0][0]
        print(f"  search for a {meta['ticker']} chunk: same best match: {a['ids'][0][0] == b['ids'][0][0]}, "
              f"top-5 overlap {overlap}/5, best distance Chroma {a['distances'][0][0]:.4f} / pgvector {b['distances'][0][0]:.4f}")
    ok &= agree == len(sample["ids"])
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="only compare the two stores")
    args = parser.parse_args()
    if not (settings.database_url or "").startswith(("postgresql", "postgres")):
        print("DATABASE_URL in .env must be the Supabase Postgres URL.")
        return 1
    print("Target:", settings.masked_database_url())

    from src.rag.pgstore import open_pg_collection
    from src.rag.store import open_chroma_collection

    if not args.verify:
        upgrade_schema()
    chroma, pg = open_chroma_collection(), open_pg_collection()
    if not args.verify:
        copied = copy_chunks(chroma, pg)
        print(f"Copied {copied} chunk(s) (~{copied * 12.3 / 1024:.0f} MB of vectors)")
        copy_registry(chroma)
    print("\nVerifying:")
    ok = verify(chroma, pg)
    print("\nMIGRATION VERIFIED - set FILINGS_STORE=postgres to use it." if ok else "\nVERIFY FAILED - see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
