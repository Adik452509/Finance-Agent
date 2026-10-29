"""The filings vector store on Postgres + pgvector (FILINGS_STORE=postgres, Phase 6.2).

PgCollection offers the small part of Chroma's Collection API the project uses - get,
query, upsert, delete, count, name, metadata - so the retriever, indexer, filings server
and scripts work unchanged on either store (src/rag/store.py picks one).

Vectors travel in pgvector's binary format (4 bytes per number instead of ~11 characters
of text), which matters for 3,072-dimension embeddings over a mobile connection.
"""

import json
import logging

import numpy as np
from pgvector.psycopg import register_vector
from sqlalchemy import Engine, event

from config.settings import settings

logger = logging.getLogger(__name__)

# Chroma `where` keys that are real columns; anything else is looked up in the metadata JSON.
COLUMNS = {"ticker", "file"}


class StoreNotReady(RuntimeError):
    """The pgvector tables don't exist yet: run `alembic upgrade head`."""


def where_sql(where: dict | None) -> tuple[str, list]:
    """Translate the Chroma filters this project uses into SQL.

    Supported: {"k": v}, {"k": {"$eq": v}}, {"k": {"$in": [...]}}, {"$and": [...]}.
    Values are always bound as parameters, never pasted into the SQL.
    """
    if not where:
        return "TRUE", []
    if "$and" in where:
        parts = [where_sql(w) for w in where["$and"]]
        return " AND ".join(f"({sql})" for sql, _ in parts), [p for _, params in parts for p in params]
    clauses, params = [], []
    for key, condition in where.items():
        if not key.replace("_", "").isalnum():
            raise ValueError(f"unsupported filter key {key!r}")
        target = key if key in COLUMNS else "metadata->>%s"
        key_params = [] if key in COLUMNS else [key]
        if isinstance(condition, dict):
            (op, value), = condition.items()
            if op == "$in":
                clauses.append(f"{target} = ANY(%s)")
                params += key_params + [[str(v) for v in value]]
                continue
            if op != "$eq":
                raise ValueError(f"unsupported filter operator {op!r}")
            condition = value
        clauses.append(f"{target} = %s")
        params += key_params + [str(condition)]
    return " AND ".join(clauses), params


class PgCollection:
    def __init__(self, engine: Engine, name: str = "filing_chunks") -> None:
        self.engine = engine
        self.name = name

        @event.listens_for(engine, "connect")
        def _register(dbapi_connection, _record):  # numpy arrays <-> pgvector, in binary
            register_vector(dbapi_connection)

    # -- plumbing -----------------------------------------------------------------------------

    def _run(self, sql: str, params: list | tuple = (), *, many: list | None = None, fetch: bool = True):
        raw = self.engine.raw_connection()
        try:
            with raw.cursor(binary=True) as cur:
                if many is not None:
                    cur.executemany(sql, many)
                    rows = []
                else:
                    cur.execute(sql, params)
                    rows = cur.fetchall() if fetch and cur.description else []
            raw.commit()
            return rows
        except Exception as e:
            raw.rollback()
            if "filing_chunks" in str(e) and "does not exist" in str(e):
                raise StoreNotReady("The pgvector tables are missing. Run: alembic upgrade head") from e
            raise
        finally:
            raw.close()

    # -- Chroma-compatible API ----------------------------------------------------------------

    @property
    def metadata(self) -> dict:
        rows = self._run("SELECT key, value FROM filing_store_meta")
        return dict(rows)

    def ensure_embedding_model(self, model: str) -> None:
        """Record the model on first use; refuse to mix models afterwards (like store.check_embedding_model)."""
        stored = self.metadata.get("embedding_model")
        if stored is None:
            self._run("INSERT INTO filing_store_meta (key, value) VALUES ('embedding_model', %s) "
                      "ON CONFLICT (key) DO NOTHING", [model], fetch=False)
        elif stored != model:
            raise RuntimeError(f"The pgvector store was built with {stored!r}, but settings use {model!r}. "
                               "Vectors from different models can't be compared.")

    def count(self) -> int:
        return self._run("SELECT count(*) FROM filing_chunks")[0][0]

    def get(self, ids: list[str] | None = None, where: dict | None = None, limit: int | None = None,
            include: list[str] | None = None) -> dict[str, list]:
        include = ["documents", "metadatas"] if include is None else include
        sql, params = where_sql(where)
        if ids is not None:
            if not ids:
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
            sql, params = f"({sql}) AND id = ANY(%s)", params + [list(ids)]
        columns = ["id"] + [c for c, inc in (("document", "documents"), ("metadata", "metadatas"),
                                              ("embedding", "embeddings")) if inc in include]
        query = f"SELECT {', '.join(columns)} FROM filing_chunks WHERE {sql} ORDER BY id"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = self._run(query, params)
        out: dict[str, list] = {"ids": [r[0] for r in rows]}
        for i, column in enumerate(columns[1:], start=1):
            key = {"document": "documents", "metadata": "metadatas", "embedding": "embeddings"}[column]
            out[key] = [r[i] for r in rows]
        return out

    def query(self, query_embeddings: list, n_results: int = 5, where: dict | None = None,
              include: list[str] | None = None) -> dict[str, list]:
        """Exact cosine search (pgvector `<=>` is 1 - cosine similarity, like Chroma's distance)."""
        sql, params = where_sql(where)
        out: dict[str, list] = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        for vector in query_embeddings:
            rows = self._run(f"SELECT id, document, metadata, embedding <=> %s AS distance FROM filing_chunks "
                             f"WHERE {sql} ORDER BY distance LIMIT {int(n_results)}",
                             [np.asarray(vector, dtype=np.float32)] + params)
            out["ids"].append([r[0] for r in rows])
            out["documents"].append([r[1] for r in rows])
            out["metadatas"].append([r[2] for r in rows])
            out["distances"].append([float(r[3]) for r in rows])
        return out

    def upsert(self, ids: list[str], embeddings: list, documents: list[str], metadatas: list[dict]) -> None:
        rows = [(i, str(m.get("file", "")), str(m.get("ticker", "")), d, json.dumps(m),
                 np.asarray(e, dtype=np.float32))
                for i, e, d, m in zip(ids, embeddings, documents, metadatas)]
        self._run("INSERT INTO filing_chunks (id, file, ticker, document, metadata, embedding) "
                  "VALUES (%s, %s, %s, %s, %s::jsonb, %s) ON CONFLICT (id) DO UPDATE SET "
                  "file = EXCLUDED.file, ticker = EXCLUDED.ticker, document = EXCLUDED.document, "
                  "metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding", many=rows)

    def delete(self, ids: list[str] | None = None, where: dict | None = None) -> None:
        if ids is None and not where:
            raise ValueError("delete needs ids or where (refusing to delete every chunk)")
        sql, params = where_sql(where)
        if ids is not None:
            sql, params = f"({sql}) AND id = ANY(%s)", params + [list(ids)]
        self._run(f"DELETE FROM filing_chunks WHERE {sql}", params, fetch=False)

    def reset(self) -> None:
        """Delete every chunk (ingest.py --reset)."""
        self._run("TRUNCATE filing_chunks", fetch=False)
        self._run("DELETE FROM filing_store_meta", fetch=False)


def open_pg_collection(reset: bool = False) -> PgCollection:
    from src.db.session import make_engine

    collection = PgCollection(make_engine(settings.app_database_url))
    if reset:
        collection.reset()
    collection.ensure_embedding_model(settings.embedding_model)
    return collection
