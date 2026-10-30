"""All reads and writes of the app database, in one place.

Security rule: every function that touches a user's data takes that user's id and filters by
it, so one user can never read, rename, delete or rate another user's chats - even with a
guessed conversation id. The backend (6.3) only ever calls these functions.

Times are UTC; "today" for the quota is the UTC date. Comparisons with stored times happen
in SQL, because SQLite hands back naive datetimes while Postgres hands back aware ones.
"""

import hashlib
import re
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from config.settings import settings
from src.db.models import AnswerCache, Conversation, Document, Feedback, Job, Message, Usage, User, utcnow

TITLE_CHARS = 60
UNLIMITED = 10**6


def _today() -> date:
    return datetime.now(timezone.utc).date()


# --- users -----------------------------------------------------------------------------------


def admin_emails() -> set[str]:
    return {e.strip().lower() for e in (settings.admin_emails or "").split(",") if e.strip()}


def get_or_create_user(db: Session, email: str, name: str | None = None) -> User:
    """The signed-in user, created on first login. Admin rights come from ADMIN_EMAILS in .env."""
    email = email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError("a valid email is required")
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, name=name)
        db.add(user)
    elif name and name != user.name:
        user.name = name
    user.is_admin = email in admin_emails()
    user.last_seen_at = utcnow()
    db.flush()
    return user


# --- conversations -----------------------------------------------------------------------------


def create_conversation(db: Session, user_id: str, title: str | None = None) -> Conversation:
    conversation = Conversation(user_id=user_id, title=(title or "New chat")[:TITLE_CHARS])
    db.add(conversation)
    db.flush()
    return conversation


def list_conversations(db: Session, user_id: str, limit: int = 50) -> list[Conversation]:
    """The sidebar: most recently active first."""
    query = (select(Conversation).where(Conversation.user_id == user_id)
             .order_by(Conversation.updated_at.desc()).limit(limit))
    return list(db.scalars(query))


def get_conversation(db: Session, user_id: str, conversation_id: str) -> Conversation | None:
    """None if it does not exist OR belongs to someone else (callers can't tell which - on purpose)."""
    return db.scalar(select(Conversation).where(Conversation.id == conversation_id,
                                                Conversation.user_id == user_id))


def rename_conversation(db: Session, user_id: str, conversation_id: str, title: str) -> bool:
    conversation = get_conversation(db, user_id, conversation_id)
    if conversation is None or not title.strip():
        return False
    conversation.title = title.strip()[:TITLE_CHARS]
    return True


def delete_conversation(db: Session, user_id: str, conversation_id: str) -> bool:
    conversation = get_conversation(db, user_id, conversation_id)
    if conversation is None:
        return False
    db.delete(conversation)  # messages and their feedback go with it
    return True


# --- messages ---------------------------------------------------------------------------------


def add_message(db: Session, conversation: Conversation, role: str, content: str, **fields) -> Message:
    """Append a message; the first question also becomes the chat's title."""
    if role not in ("user", "assistant"):
        raise ValueError(f"unknown role {role!r}")
    message = Message(conversation_id=conversation.id, role=role, content=content, **fields)
    db.add(message)
    conversation.updated_at = utcnow()
    if role == "user" and conversation.title == "New chat":
        conversation.title = " ".join(content.split())[:TITLE_CHARS] or "New chat"
    db.flush()
    return message


def list_messages(db: Session, user_id: str, conversation_id: str) -> list[Message] | None:
    """A chat's messages in order; None if the chat is not this user's."""
    if get_conversation(db, user_id, conversation_id) is None:
        return None
    query = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    return list(db.scalars(query))


def recent_turns(db: Session, conversation_id: str, limit: int = 6) -> list[tuple[str, str]]:
    """The last few (role, text) pairs, oldest first: context for rewriting a follow-up (6.4)."""
    query = (select(Message.role, Message.standalone_question, Message.content)
             .where(Message.conversation_id == conversation_id, Message.status == "ok")
             .order_by(Message.created_at.desc()).limit(limit))
    rows = list(db.execute(query))[::-1]
    return [(role, standalone or content) for role, standalone, content in rows]


# --- daily quota ----------------------------------------------------------------------------------


def quota_for(user: User) -> int:
    if user.is_admin:
        return UNLIMITED
    return user.daily_quota if user.daily_quota is not None else settings.daily_question_quota


def usage_today(db: Session, user_id: str, today: date | None = None) -> tuple[int, int]:
    """(questions, tokens) used today."""
    row = db.get(Usage, (user_id, today or _today()))
    return (row.questions, row.tokens) if row else (0, 0)


def check_quota(db: Session, user: User, today: date | None = None) -> tuple[bool, int]:
    """(may ask another question, questions left today)."""
    used, _ = usage_today(db, user.id, today)
    left = max(quota_for(user) - used, 0)
    return left > 0, left


def record_question(db: Session, user_id: str, tokens: int, today: date | None = None) -> None:
    """Count one answered question (cached answers count too: the quota is about fairness)."""
    day = today or _today()
    row = db.get(Usage, (user_id, day))
    if row is None:
        row = Usage(user_id=user_id, day=day, questions=0, tokens=0)
        db.add(row)
    row.questions += 1
    row.tokens += max(int(tokens or 0), 0)
    db.flush()


# --- answer cache -----------------------------------------------------------------------------------


def normalize_question(question: str) -> str:
    """'  What was Apple's revenue?? ' and 'what was apple's revenue' are the same question."""
    return re.sub(r"[\s?.!]+$", "", " ".join(question.lower().split()))


def cache_key(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def cache_get(db: Session, question: str, now: datetime | None = None) -> dict | None:
    """A still-valid cached answer (and count the hit), or None."""
    now = now or utcnow()
    row = db.scalar(select(AnswerCache).where(AnswerCache.key == cache_key(question),
                                              AnswerCache.expires_at > now))
    if row is None:
        return None
    row.hits += 1
    return row.payload


def cache_put(db: Session, question: str, payload: dict, ttl_seconds: int, now: datetime | None = None) -> None:
    now = now or utcnow()
    key = cache_key(question)
    row = db.get(AnswerCache, key)
    if row is None:
        row = AnswerCache(key=key, question=question, payload=payload, created_at=now,
                          expires_at=now + timedelta(seconds=ttl_seconds))
        db.add(row)
    else:  # a fresh answer replaces an expired (or older) one
        row.question, row.payload, row.hits = question, payload, 0
        row.created_at, row.expires_at = now, now + timedelta(seconds=ttl_seconds)
    db.flush()


def purge_expired_cache(db: Session, now: datetime | None = None) -> int:
    rows = list(db.scalars(select(AnswerCache).where(AnswerCache.expires_at <= (now or utcnow()))))
    for row in rows:
        db.delete(row)
    return len(rows)


# --- feedback -------------------------------------------------------------------------------------


def set_feedback(db: Session, user_id: str, message_id: str, rating: int, comment: str | None = None) -> bool:
    """Rate an answer in one of this user's own chats (rating again replaces the old one)."""
    if rating not in (1, -1):
        raise ValueError("rating must be +1 or -1")
    owned = db.scalar(select(Message).join(Conversation)
                      .where(Message.id == message_id, Message.role == "assistant", Conversation.user_id == user_id))
    if owned is None:
        return False
    row = db.scalar(select(Feedback).where(Feedback.message_id == message_id, Feedback.user_id == user_id))
    if row is None:
        db.add(Feedback(message_id=message_id, user_id=user_id, rating=rating, comment=comment))
    else:
        row.rating, row.comment = rating, comment
    db.flush()
    return True


# --- documents (the filings library when FILINGS_STORE=postgres) ------------------------------------------

DOCUMENT_FIELDS = ("file", "company", "ticker", "market", "doc_type", "fiscal_year_end", "currency",
                   "reporting_unit", "source", "source_url", "filed", "pages")


def _base_ticker(ticker: str) -> str:
    return ticker.strip().upper().split(".")[0]


def document_entry(doc: Document) -> dict:
    """A row as the same dict config/documents.json holds."""
    return {f: getattr(doc, f) for f in DOCUMENT_FIELDS}


def list_documents(db: Session) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.created_at, Document.file)))


def documents_for_ticker(db: Session, ticker: str) -> list[dict]:
    return [document_entry(d) for d in list_documents(db) if _base_ticker(d.ticker) == _base_ticker(ticker)]


def replace_document(db: Session, entry: dict, status: str = "registered", chunks: int = 0) -> list[dict]:
    """Add a report, replacing the same company's older one(s); returns the replaced entries."""
    replaced = [d for d in list_documents(db) if _base_ticker(d.ticker) == _base_ticker(entry["ticker"])]
    old = [document_entry(d) for d in replaced]
    for doc in replaced:
        db.delete(doc)
    db.flush()
    db.add(Document(**{f: str(entry.get(f) or "") for f in DOCUMENT_FIELDS}, status=status, chunks=chunks))
    db.flush()
    return old


def set_document_status(db: Session, file: str, status: str, chunks: int | None = None) -> None:
    doc = db.scalar(select(Document).where(Document.file == file))
    if doc is not None:
        doc.status = status
        if chunks is not None:
            doc.chunks = chunks


# --- jobs (background work) ---------------------------------------------------------------------------


def create_job(db: Session, kind: str, args: dict, created_by: str | None = None) -> Job:
    job = Job(kind=kind, args=args, created_by=created_by)
    db.add(job)
    db.flush()
    return job


def claim_next_job(db: Session) -> Job | None:
    """Take the oldest queued job. The conditional UPDATE means two workers never take the same one."""
    candidate = db.scalar(select(Job.id).where(Job.status == "queued").order_by(Job.created_at).limit(1))
    if candidate is None:
        return None
    claimed = db.execute(update(Job).where(Job.id == candidate, Job.status == "queued")
                         .values(status="running", started_at=utcnow()))
    if claimed.rowcount != 1:
        return None  # another worker got there first
    db.flush()
    return db.get(Job, candidate, populate_existing=True)


def update_job(db: Session, job_id: str, *, progress: str | None = None, status: str | None = None,
               result: dict | None = None) -> None:
    job = db.get(Job, job_id)
    if job is None:
        return
    if progress is not None:
        job.progress = progress
    if result is not None:
        job.result = result
    if status is not None:
        job.status = status
        if status in ("done", "failed"):
            job.finished_at = utcnow()
    db.flush()


def get_job(db: Session, job_id: str) -> Job | None:
    return db.get(Job, job_id)


def list_jobs(db: Session, limit: int = 20) -> list[Job]:
    return list(db.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit)))
