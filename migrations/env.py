"""Alembic environment: migrates the app database named by config.settings.

    alembic upgrade head                              # create / update the tables
    alembic revision --autogenerate -m "add x"        # after changing src/db/models.py

The URL comes from settings (DATABASE_URL in .env, or the local SQLite file) and is passed
straight to the engine - never through alembic.ini, which would store the password in a
committed file (and ConfigParser would also choke on a '%' in it).
"""

from logging.config import fileConfig

from alembic import context

from config.settings import settings
from src.db.models import Base
from src.db.session import make_engine

config = context.config
if config.config_file_name is not None:
    # disable_existing_loggers=False: Python's default would silence every logger already
    # created (seen in the tests: the retriever's warnings vanished after a migration ran).
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Write the SQL to stdout instead of running it (alembic upgrade head --sql)."""
    context.configure(url=settings.app_database_url, target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"}, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = config.attributes.get("url") or settings.app_database_url  # tests pass a temporary database
    engine = make_engine(url)
    try:
        with engine.connect() as connection:
            # SQLite can't ALTER most things in place: batch mode rebuilds the table instead.
            context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True,
                              compare_type=True)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
