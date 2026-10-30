"""Database engine and sessions.

    from src.db.session import session_scope
    with session_scope() as db:
        user = repository.get_or_create_user(db, "a@b.com")

The URL comes from settings (DATABASE_URL, or a local SQLite file). Tables are created by
Alembic migrations (`alembic upgrade head`), not by the app, so a deployed database is
always changed deliberately and its history is versioned.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings


def make_engine(url: str | None = None) -> Engine:
    url = url or settings.app_database_url
    if url.startswith("sqlite"):
        if url.startswith("sqlite:///") and ":memory:" not in url:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # SQLite ignores foreign keys unless asked
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    # Supabase's session pooler: keep a small pool and drop connections the pooler closed.
    # connect_timeout: seen live, a network that blocks port 5432 made connecting hang forever.
    return create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True, pool_recycle=300,
                         connect_args={"connect_timeout": 15})


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


@lru_cache(maxsize=1)
def _factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """A session that commits on success and rolls back on any error."""
    session = (factory or _factory())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
