"""Phase 6.1 database layer: repository rules and the Alembic migration. Offline (SQLite)."""

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

from src.db import repository as repo
from src.db.models import Base, Conversation, Feedback, Message
from src.db.session import make_engine, session_scope


@pytest.fixture
def db(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def two_users(db):
    return repo.get_or_create_user(db, "Aditya@Example.com", "Aditya"), repo.get_or_create_user(db, "other@example.com")


# --- users ------------------------------------------------------------------------------------


def test_users_are_created_once_per_email(db):
    first = repo.get_or_create_user(db, "  Aditya@Example.COM ", "Aditya")
    again = repo.get_or_create_user(db, "aditya@example.com")
    assert first.id == again.id and again.email == "aditya@example.com" and again.name == "Aditya"
    with pytest.raises(ValueError):
        repo.get_or_create_user(db, "not-an-email")


def test_admins_come_from_settings(db, monkeypatch):
    monkeypatch.setattr(repo.settings, "admin_emails", "boss@example.com, other@x.com")
    assert repo.get_or_create_user(db, "Boss@example.com").is_admin
    assert not repo.get_or_create_user(db, "user@example.com").is_admin


# --- conversations and ownership ------------------------------------------------------------------


def test_one_user_can_never_touch_another_users_chat(db, two_users):
    alice, mallory = two_users
    chat = repo.create_conversation(db, alice.id)
    repo.add_message(db, chat, "user", "Apple revenue?")
    assert repo.get_conversation(db, mallory.id, chat.id) is None
    assert repo.list_messages(db, mallory.id, chat.id) is None
    assert repo.rename_conversation(db, mallory.id, chat.id, "hacked") is False
    assert repo.delete_conversation(db, mallory.id, chat.id) is False
    assert repo.list_conversations(db, mallory.id) == []
    assert repo.get_conversation(db, alice.id, chat.id).title == "Apple revenue?"


def test_sidebar_lists_the_most_recent_chat_first(db, two_users):
    alice, _ = two_users
    old, new = repo.create_conversation(db, alice.id), repo.create_conversation(db, alice.id)
    repo.add_message(db, new, "user", "second")
    repo.add_message(db, old, "user", "first, but answered later")  # activity moves it to the top
    assert [c.id for c in repo.list_conversations(db, alice.id)] == [old.id, new.id]


def test_first_question_becomes_the_title(db, two_users):
    alice, _ = two_users
    chat = repo.create_conversation(db, alice.id)
    repo.add_message(db, chat, "user", "What was   Microsoft's Intelligent Cloud revenue in fiscal 2026 and its share?")
    repo.add_message(db, chat, "user", "and in 2025?")
    assert chat.title == "What was Microsoft's Intelligent Cloud revenue in fiscal 2026"[:repo.TITLE_CHARS]
    assert repo.rename_conversation(db, alice.id, chat.id, "  MSFT cloud  ") and chat.title == "MSFT cloud"


def test_deleting_a_chat_deletes_its_messages_and_feedback(db, two_users):
    alice, _ = two_users
    chat = repo.create_conversation(db, alice.id)
    answer = repo.add_message(db, chat, "assistant", "41.52%")
    repo.set_feedback(db, alice.id, answer.id, 1)
    db.commit()
    assert repo.delete_conversation(db, alice.id, chat.id)
    db.commit()
    assert db.query(Conversation).count() == 0 and db.query(Message).count() == 0 and db.query(Feedback).count() == 0


# --- messages -----------------------------------------------------------------------------------


def test_messages_round_trip_with_their_json(db, two_users):
    alice, _ = two_users
    chat = repo.create_conversation(db, alice.id)
    repo.add_message(db, chat, "user", "Intelligent Cloud share?")
    repo.add_message(db, chat, "assistant", "41.52% of revenue", citations={"MSFT p84": "Microsoft 10-K, p. 84"},
                     trace={"plan": {"steps": [{"kind": "filings"}]}, "warnings": []},
                     usage={"researcher": 18260, "critic": 985}, tokens=19245)
    db.commit()
    question, answer = repo.list_messages(db, alice.id, chat.id)
    assert (question.role, answer.role) == ("user", "assistant")
    assert answer.citations["MSFT p84"] == "Microsoft 10-K, p. 84" and answer.usage["critic"] == 985
    with pytest.raises(ValueError):
        repo.add_message(db, chat, "system", "nope")


def test_recent_turns_give_context_for_follow_ups(db, two_users):
    alice, _ = two_users
    chat = repo.create_conversation(db, alice.id)
    repo.add_message(db, chat, "user", "and 2025?", standalone_question="What was Apple's revenue in fiscal 2025?")
    repo.add_message(db, chat, "assistant", "failed", status="error")  # failed answers are not context
    repo.add_message(db, chat, "assistant", "416,161")
    assert repo.recent_turns(db, chat.id) == [("user", "What was Apple's revenue in fiscal 2025?"),
                                              ("assistant", "416,161")]
    assert len(repo.recent_turns(db, chat.id, limit=1)) == 1


# --- daily quota ------------------------------------------------------------------------------------


def test_quota_counts_questions_per_day(db, two_users, monkeypatch):
    monkeypatch.setattr(repo.settings, "daily_question_quota", 2)
    alice, _ = two_users
    day = date(2026, 10, 29)
    assert repo.check_quota(db, alice, day) == (True, 2)
    repo.record_question(db, alice.id, tokens=12_000, today=day)
    repo.record_question(db, alice.id, tokens=8_000, today=day)
    assert repo.check_quota(db, alice, day) == (False, 0)
    assert repo.usage_today(db, alice.id, day) == (2, 20_000)
    assert repo.check_quota(db, alice, day + timedelta(days=1)) == (True, 2)  # a new day resets it


def test_per_user_quota_and_unlimited_admins(db, two_users, monkeypatch):
    alice, _ = two_users
    alice.daily_quota = 30
    assert repo.quota_for(alice) == 30
    monkeypatch.setattr(repo.settings, "admin_emails", alice.email)
    assert repo.quota_for(repo.get_or_create_user(db, alice.email)) == repo.UNLIMITED


# --- answer cache -------------------------------------------------------------------------------------


def test_same_question_in_other_words_hits_the_cache(db):
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
    repo.cache_put(db, "What was Apple's revenue?", {"report": "416,161"}, ttl_seconds=900, now=now)
    assert repo.cache_get(db, "  what was apple's   REVENUE?? ", now=now + timedelta(minutes=5)) == {"report": "416,161"}
    assert repo.cache_get(db, "What was Apple's profit?", now=now) is None


def test_expired_answers_are_not_served_and_can_be_purged(db):
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
    repo.cache_put(db, "Toyota share price?", {"report": "3,105"}, ttl_seconds=900, now=now)
    assert repo.cache_get(db, "Toyota share price?", now=now + timedelta(minutes=16)) is None
    assert repo.purge_expired_cache(db, now=now + timedelta(minutes=16)) == 1


def test_a_fresh_answer_replaces_the_old_one(db):
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
    repo.cache_put(db, "Toyota share price?", {"report": "3,105"}, ttl_seconds=900, now=now)
    repo.cache_put(db, "Toyota share price?", {"report": "3,150"}, ttl_seconds=900, now=now + timedelta(hours=1))
    assert repo.cache_get(db, "Toyota share price?", now=now + timedelta(hours=1, minutes=1)) == {"report": "3,150"}


# --- feedback ---------------------------------------------------------------------------------------


def test_feedback_only_on_your_own_answers_and_rating_again_replaces(db, two_users):
    alice, mallory = two_users
    chat = repo.create_conversation(db, alice.id)
    question = repo.add_message(db, chat, "user", "q")
    answer = repo.add_message(db, chat, "assistant", "a")
    assert repo.set_feedback(db, mallory.id, answer.id, -1) is False      # not her chat
    assert repo.set_feedback(db, alice.id, question.id, 1) is False       # questions can't be rated
    assert repo.set_feedback(db, alice.id, answer.id, 1) and repo.set_feedback(db, alice.id, answer.id, -1, "wrong year")
    rows = db.query(Feedback).all()
    assert len(rows) == 1 and rows[0].rating == -1 and rows[0].comment == "wrong year"
    with pytest.raises(ValueError):
        repo.set_feedback(db, alice.id, answer.id, 5)


# --- jobs -------------------------------------------------------------------------------------------


def test_jobs_are_claimed_once_in_order_and_finish(db, two_users):
    alice, _ = two_users
    first = repo.create_job(db, "add_company", {"ticker": "NVDA"}, created_by=alice.id)
    second = repo.create_job(db, "add_company", {"ticker": "GOOGL"})
    first.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db.flush()
    claimed = repo.claim_next_job(db)
    assert claimed.id == first.id and claimed.status == "running" and claimed.started_at
    assert repo.claim_next_job(db).id == second.id
    assert repo.claim_next_job(db) is None  # nothing left queued
    repo.update_job(db, first.id, progress="Batch 8/8", status="done", result={"chunks": 228})
    job = repo.get_job(db, first.id)
    assert (job.status, job.progress, job.result) == ("done", "Batch 8/8", {"chunks": 228}) and job.finished_at


# --- sessions and settings -----------------------------------------------------------------------------


def test_session_scope_rolls_back_on_error(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 's.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with pytest.raises(RuntimeError):
        with session_scope(factory) as db:
            repo.get_or_create_user(db, "a@example.com")
            raise RuntimeError("boom")
    with session_scope(factory) as db:
        assert repo.list_conversations(db, "nobody") == [] and db.query(repo.User).count() == 0
    engine.dispose()


def test_database_url_uses_psycopg_and_never_shows_the_password(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "database_url", "postgresql://postgres.abc:S3cr3t!pw@aws-0-ap-south-1.pooler.supabase.com:5432/postgres")
    assert settings.app_database_url.startswith("postgresql+psycopg://postgres.abc:")
    assert "S3cr3t" not in settings.masked_database_url() and "***@aws-0-ap-south-1" in settings.masked_database_url()
    monkeypatch.setattr(settings, "database_url", None)
    assert settings.app_database_url.endswith("/data/app.db") and settings.app_database_url.startswith("sqlite:///")


# --- migration ------------------------------------------------------------------------------------------


def test_migration_builds_exactly_the_models_schema_and_can_be_undone(tmp_path):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    from config.settings import settings

    url = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
    config = Config(str(settings.project_root / "alembic.ini"))
    config.attributes["url"] = url
    command.upgrade(config, "head")

    engine = make_engine(url)
    with engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []  # models == migrations
    assert set(inspect(engine).get_table_names()) >= {"users", "conversations", "messages", "feedback", "usage",
                                                      "answer_cache", "documents", "jobs"}
    command.downgrade(config, "base")
    assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    engine.dispose()
