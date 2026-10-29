"""Phase 6.2: the pgvector store's filter translation, store selection, and the registry in the database.

The SQL itself is exercised live by scripts/migrate_to_postgres.py (it verifies search parity
against Chroma); these tests stay offline.
"""

import json

import pytest
from sqlalchemy.orm import sessionmaker

from src.db import repository as repo
from src.db.models import Base
from src.db.session import make_engine
from src.rag.pgstore import where_sql


def test_filters_become_parameterised_sql():
    assert where_sql(None) == ("TRUE", [])
    assert where_sql({"ticker": "MSFT"}) == ("ticker = %s", ["MSFT"])
    assert where_sql({"ticker": {"$in": ["AAPL", "7203.T"]}}) == ("ticker = ANY(%s)", [["AAPL", "7203.T"]])
    assert where_sql({"page": {"$eq": 84}}) == ("metadata->>%s = %s", ["page", "84"])
    sql, params = where_sql({"$and": [{"ticker": "RELIANCE.NS"}, {"page": 130}]})
    assert sql == "(ticker = %s) AND (metadata->>%s = %s)" and params == ["RELIANCE.NS", "page", "130"]


def test_filter_values_are_never_pasted_into_the_sql():
    sql, params = where_sql({"ticker": "x'; DROP TABLE filing_chunks; --"})
    assert "DROP" not in sql and params == ["x'; DROP TABLE filing_chunks; --"]
    with pytest.raises(ValueError):
        where_sql({"ticker; DROP TABLE x": "a"})
    with pytest.raises(ValueError):
        where_sql({"ticker": {"$regex": ".*"}})


def test_local_store_stays_the_default(monkeypatch):
    from config.settings import Settings

    assert Settings(_env_file=None).filings_store == "local"
    with pytest.raises(ValueError, match="needs DATABASE_URL"):
        Settings(_env_file=None, filings_store="postgres", database_url=None)


@pytest.fixture
def db_registry(tmp_path, monkeypatch):
    """FILINGS_STORE=postgres, but with a temporary SQLite app database."""
    from src.db import session as session_module

    engine = make_engine(f"sqlite:///{(tmp_path / 'reg.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(session_module, "_factory", lambda: factory)
    monkeypatch.setattr(repo.settings, "filings_store", "postgres")
    yield factory
    engine.dispose()


ENTRY = {"file": "MSFT-10-K-2026-06-30.htm", "company": "Microsoft Corp", "ticker": "MSFT", "market": "USA",
         "doc_type": "10-K", "fiscal_year_end": "2026-06-30", "currency": "USD", "reporting_unit": "USD millions",
         "source": "sec", "source_url": "https://www.sec.gov/x", "filed": "2026-07-29", "pages": "1-99"}


def test_registry_lives_in_the_database_in_postgres_mode(db_registry):
    from src.ingestion import acquire
    from src.ingestion.loader import load_registry

    assert acquire.register(ENTRY) == []
    newer = {**ENTRY, "file": "MSFT-10-K-2027-06-30.htm", "fiscal_year_end": "2027-06-30", "filed": "2027-07-28"}
    replaced = acquire.register(newer)
    assert [e["file"] for e in replaced] == ["MSFT-10-K-2026-06-30.htm"]
    docs = load_registry()
    assert [(d.ticker, d.file, d.source, d.selected_pages() == set(range(1, 100))) for d in docs] == [
        ("MSFT", "MSFT-10-K-2027-06-30.htm", "sec", True)]
    assert acquire.registered("msft")[0]["filed"] == "2027-07-28"


def test_explicit_registry_files_still_work_in_postgres_mode(db_registry, tmp_path):
    from src.ingestion.loader import load_registry

    path = tmp_path / "documents.json"
    path.write_text(json.dumps([{k: v for k, v in ENTRY.items() if k not in ("source", "source_url", "filed", "pages")}]),
                    encoding="utf-8")
    assert [d.ticker for d in load_registry(path)] == ["MSFT"]


def test_document_status_and_chunk_count(db_registry):
    with db_registry() as db:
        repo.replace_document(db, ENTRY)
        repo.set_document_status(db, ENTRY["file"], "ready", chunks=228)
        db.commit()
        doc = repo.list_documents(db)[0]
        assert (doc.status, doc.chunks) == ("ready", 228)
