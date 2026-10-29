"""filing chunks with pgvector embeddings (Phase 6.2, Postgres only)

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-29

Why no approximate (HNSW) index: pgvector indexes at most 2,000 dimensions and
gemini-embedding-001 gives 3,072. We don't need one: searches are filtered by ticker
first (a btree index), leaving a few hundred rows, and an exact cosine scan over those
takes milliseconds - and is exact, unlike Chroma's approximate HNSW.

On SQLite (the laptop) this migration does nothing: there the filings stay in Chroma.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DIMENSIONS = 3072  # gemini-embedding-001


def _postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _postgres():
        return
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(f"""
        CREATE TABLE filing_chunks (
            id        TEXT PRIMARY KEY,               -- the chunker's stable id (file-page-chunk-hash)
            file      TEXT NOT NULL,
            ticker    TEXT NOT NULL,
            document  TEXT NOT NULL,                  -- the chunk text shown to the agents
            metadata  JSONB NOT NULL,                 -- company, doc_type, page, units ... (as in Chroma)
            embedding VECTOR({DIMENSIONS}) NOT NULL
        )""")
    op.execute("CREATE INDEX ix_filing_chunks_ticker ON filing_chunks (ticker)")
    op.execute("CREATE INDEX ix_filing_chunks_file ON filing_chunks (file)")
    # Which embedding model built the vectors: mixing models makes similarity meaningless.
    op.execute("CREATE TABLE filing_store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def downgrade() -> None:
    if not _postgres():
        return
    op.execute("DROP TABLE IF EXISTS filing_store_meta")
    op.execute("DROP TABLE IF EXISTS filing_chunks")
