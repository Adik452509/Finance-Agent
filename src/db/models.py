"""Database tables of the deployed app (Phase 6).

One schema for both databases: SQLite on a laptop, Postgres (Supabase) when deployed.
IDs are UUID strings made in Python and JSON columns use the generic JSON type (JSONB on
Postgres), so nothing here depends on one database's features. The filing chunks table
(pgvector, Postgres only) arrives in Step 6.2.

    users ─┬─< conversations ─< messages ─< feedback
           ├─< usage (one row per user per day)
           └─< jobs (e.g. "add company", run by the backend's worker)
    documents   (the filings library; replaces config/documents.json in 6.2)
    answer_cache (identical questions answered without spending tokens)
"""

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import JSON, Boolean, Date, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    """Someone who signed in (Google login gives the email)."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    name: Mapped[str | None] = mapped_column(String(200))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    daily_quota: Mapped[int | None] = mapped_column(Integer)  # None = the app default
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    """One chat in the sidebar."""

    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user_updated", "user_id", "updated_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan",
                                                     order_by="Message.created_at")


class Message(Base):
    """A user question or an assistant answer, with everything needed to redraw it later."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16))                 # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)                    # the question, or the final report
    standalone_question: Mapped[str | None] = mapped_column(Text)  # a follow-up rewritten in full (6.4)
    status: Mapped[str] = mapped_column(String(16), default="ok")  # "ok" | "error"
    citations: Mapped[dict | None] = mapped_column(JSONType)       # source id -> full citation
    trace: Mapped[dict | None] = mapped_column(JSONType)           # plan, tool calls, warnings, critique
    usage: Mapped[dict | None] = mapped_column(JSONType)           # tokens per agent
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    from_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Feedback(Base):
    """A thumbs up / down on an answer: real-world evaluation data."""

    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("message_id", "user_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    rating: Mapped[int] = mapped_column(Integer)                  # +1 or -1
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Usage(Base):
    """Questions and tokens per user per day, for the daily quota."""

    __tablename__ = "usage"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    questions: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)


class AnswerCache(Base):
    """A finished answer, reusable for the same question until it expires."""

    __tablename__ = "answer_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256 of the normalised question
    question: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSONType)                 # report, citations, trace, usage
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Document(Base):
    """One indexed annual report (the same fields as config/documents.json)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    file: Mapped[str] = mapped_column(String(255), unique=True)
    company: Mapped[str] = mapped_column(String(200))
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    market: Mapped[str] = mapped_column(String(50))
    doc_type: Mapped[str] = mapped_column(String(50))
    fiscal_year_end: Mapped[str] = mapped_column(String(10))
    currency: Mapped[str] = mapped_column(String(10))
    reporting_unit: Mapped[str] = mapped_column(String(80))
    source: Mapped[str] = mapped_column(String(20), default="")     # "sec" | "edinet" | "url" | "" (by hand)
    source_url: Mapped[str] = mapped_column(Text, default="")
    filed: Mapped[str] = mapped_column(String(10), default="")
    pages: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="registered")  # registered | indexing | ready | error
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    """Long work (e.g. adding a company: ~8 min of embedding) done by the backend's worker."""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_status_created", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(32))                  # e.g. "add_company"
    args: Mapped[dict] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued | running | done | failed
    progress: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[dict | None] = mapped_column(JSONType)
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
