"""The backend API (Phase 6.3).

    uvicorn src.api.app:app --host 127.0.0.1 --port 8000

Who may call it: only the Streamlit frontend. The backend listens on localhost and every
request must carry X-Api-Token (INTERNAL_API_TOKEN); the frontend adds X-User-Email after
Google login (6.6), and the backend trusts that header only because of the token.

Asking a question streams Server-Sent Events. The answer is produced by a background task,
not by the HTTP request: if the browser disconnects (Streamlit reruns, a closed tab) the
research still finishes and is saved, so the chat shows it when reopened.
"""

import asyncio
import hmac
import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from urllib.parse import unquote

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from src.api import schemas
from src.api.chat import answer_record, friendly_error, run_question
from src.api.team import Team
from src.api.worker import job_worker
from src.db import repository as repo
from src.db.session import session_scope
from src.memory.cache_policy import cache_ttl
from src.memory.followup import standalone_question

logger = logging.getLogger(__name__)


# --- dependencies -----------------------------------------------------------------------------


class Caller:
    def __init__(self, email: str, name: str | None) -> None:
        self.email, self.name = email, name


def caller(x_api_token: str | None = Header(default=None), x_user_email: str | None = Header(default=None),
           x_user_name: str | None = Header(default=None)) -> Caller:
    expected = settings.internal_api_token
    if expected and not hmac.compare_digest(x_api_token or "", expected):  # constant time: no timing leaks
        raise HTTPException(401, "invalid API token")
    if not x_user_email:
        raise HTTPException(401, "X-User-Email is required")
    return Caller(x_user_email, unquote(x_user_name) if x_user_name else None)  # names may hold non-ASCII


@contextmanager
def database(request: Request) -> Iterator[Session]:
    with session_scope(request.app.state.sessions) as db:
        yield db


def user_in(db: Session, who: Caller):
    try:
        return repo.get_or_create_user(db, who.email, who.name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def admin_in(db: Session, who: Caller):
    user = user_in(db, who)
    if not user.is_admin:
        raise HTTPException(403, "admins only")
    return user


# --- answering --------------------------------------------------------------------------------


def cacheable(final: dict) -> bool:
    """Only a complete, checked answer is reused.

    Not a doubtful one (critic fail), and not one with gaps: seen live, an honest "not found"
    (the researcher's reply came back empty) passed the critic and would have been served for 7 days.
    """
    if (final.get("critique") or {}).get("verdict") != "pass":
        return False
    notes = " ".join(f.get("facts", "") for f in final.get("findings") or [])
    return "NOT FOUND" not in notes and "UNVERIFIED" not in notes


def _progress(agent: str, label: str, text: str) -> dict:
    return {"type": "progress", "agent": agent, "label": label, "text": text, "detail": [], "warnings": []}


async def produce_answer(app: FastAPI, user_id: str, conversation_id: str, question_id: str, question: str,
                         turns: list[tuple[str, str]], emit) -> None:
    """Background task: understand, answer (from cache or research), save, then close the stream."""
    state = app.state
    record, status, counts = None, "ok", True
    try:
        # 1. A follow-up ("and in 2024?") becomes a standalone question, using the chat history.
        standalone, memory_tokens, warning = await standalone_question(state.rewriter, question, turns)
        if standalone != question:
            emit(_progress("memory", "Memory", f"Understood as: {standalone}"))
            with session_scope(state.sessions) as db:
                repo.set_standalone_question(db, question_id, standalone)
        if warning:
            emit(_progress("memory", "Memory", warning))

        # 2. The same question answered recently: reuse it (no tokens, no quota).
        with session_scope(state.sessions) as db:
            cached = repo.cache_get(db, standalone)
        if cached is not None:
            record = {**cached, "tokens": 0, "usage": {}, "from_cache": True}
            counts = False
            emit(_progress("cache", "Cache", "Answered from a recent identical question (no new research needed)"))
        else:
            # 3. Research it.
            if state.slots.locked():
                emit({"type": "queued", "text": "Another question is being researched; yours is next."})
            async with state.slots:  # Groq's per-minute limit: only a few runs at a time
                if not await state.team.wait_ready(timeout=90):
                    raise RuntimeError(state.team.error or "the research team is not ready")
                state.active += 1
                try:
                    final = await run_question(state.team, standalone, emit, settings.question_timeout_seconds)
                finally:
                    state.active -= 1
            record = answer_record(final, extra_usage={"memory": memory_tokens} if memory_tokens else None)
            if not record["content"]:
                raise RuntimeError("the team produced no report")
            if cacheable(final):
                with session_scope(state.sessions) as db:
                    repo.cache_put(db, standalone, {k: record[k] for k in ("content", "citations", "trace")},
                                   cache_ttl(standalone))
    except Exception as e:
        logger.exception("Question failed")
        record, status = {"content": friendly_error(e), "trace": {"error": f"{type(e).__name__}: {str(e)[:300]}"}}, "error"
        emit({"type": "error", "text": record["content"]})
    try:
        with session_scope(state.sessions) as db:
            conversation = repo.get_conversation(db, user_id, conversation_id)
            if conversation is not None:  # it may have been deleted while we worked
                message = repo.add_message(db, conversation, "assistant", status=status, **record)
                if status == "ok" and counts:
                    repo.record_question(db, user_id, record.get("tokens", 0))  # failures and cache hits are free
                emit({"type": "answer", "message": schemas.MessageOut.model_validate(message).model_dump(mode="json")})
    except Exception:
        logger.exception("Could not save the answer")
        emit({"type": "error", "text": "The answer could not be saved."})
    finally:
        emit(None)  # end of stream
        maybe_reload(app)


def maybe_reload(app: FastAPI) -> None:
    """Pick up newly added companies once no question is running."""
    if app.state.team.needs_reload and app.state.active == 0:
        logger.info("Reloading the research team to include new companies")
        app.state.team.request_reload()


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"


# --- the app ----------------------------------------------------------------------------------


def default_rewriter():
    from config.llm import get_chat_model

    return get_chat_model("default")  # the small model: a rewrite is a ~1k-token job


def create_app(team: Team | None = None, sessions: sessionmaker[Session] | None = None,
               start_worker: bool = True, rewriter=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = app.state
        state.team = team or Team()
        state.rewriter = rewriter or default_rewriter
        if sessions is not None:
            state.sessions = sessions
        else:
            from src.db.session import _factory

            state.sessions = _factory()
        state.slots = asyncio.Semaphore(settings.max_parallel_questions)
        state.active = 0
        state.tasks = set()
        background = [asyncio.create_task(state.team.run())]  # starts the MCP servers (~10-20 s)

        def after_job(_kind: str, status: str) -> None:
            if status == "done":
                state.team.needs_reload = True
                maybe_reload(app)

        if start_worker:
            background.append(asyncio.create_task(job_worker(state.sessions, on_done=after_job)))
        yield
        state.team.request_stop()
        for task in background[1:]:
            task.cancel()
        await asyncio.wait(background, timeout=30)

    app = FastAPI(title="Financial Research Agent API", version="0.6.3", lifespan=lifespan,
                  docs_url="/docs" if settings.app_env == "dev" else None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict:
        """Liveness for the container: no auth, no details."""
        return {"status": "ok"}

    @app.get("/status")
    async def status(request: Request, who: Caller = Depends(caller)) -> dict:
        team_ = request.app.state.team
        return {"team_ready": team_.ready, "team_error": team_.error, "unavailable_servers": team_.failed,
                "tools": len(team_.tools), "filings_store": settings.filings_store,
                "database": settings.masked_database_url(), "questions_running": request.app.state.active}

    @app.get("/me", response_model=schemas.Me)
    def me(request: Request, who: Caller = Depends(caller)):
        with database(request) as db:
            user = user_in(db, who)
            _, left = repo.check_quota(db, user)
            return schemas.Me(email=user.email, name=user.name, is_admin=user.is_admin,
                              questions_left_today=left, daily_quota=repo.quota_for(user))

    # -- conversations --

    @app.get("/conversations", response_model=list[schemas.ConversationOut])
    def conversations(request: Request, who: Caller = Depends(caller)):
        with database(request) as db:
            return [schemas.ConversationOut.model_validate(c) for c in repo.list_conversations(db, user_in(db, who).id)]

    @app.post("/conversations", response_model=schemas.ConversationOut, status_code=201)
    def new_conversation(body: schemas.NewConversation, request: Request, who: Caller = Depends(caller)):
        with database(request) as db:
            return schemas.ConversationOut.model_validate(repo.create_conversation(db, user_in(db, who).id, body.title))

    @app.get("/conversations/{conversation_id}", response_model=schemas.ConversationDetail)
    def conversation(conversation_id: str, request: Request, who: Caller = Depends(caller)):
        with database(request) as db:
            user = user_in(db, who)
            found = repo.get_conversation(db, user.id, conversation_id)
            if found is None:
                raise HTTPException(404, "conversation not found")
            messages = repo.list_messages(db, user.id, conversation_id) or []
            return schemas.ConversationDetail(**schemas.ConversationOut.model_validate(found).model_dump(),
                                              messages=[schemas.MessageOut.model_validate(m) for m in messages])

    @app.patch("/conversations/{conversation_id}", response_model=schemas.ConversationOut)
    def rename(conversation_id: str, body: schemas.Rename, request: Request, who: Caller = Depends(caller)):
        with database(request) as db:
            user = user_in(db, who)
            if not repo.rename_conversation(db, user.id, conversation_id, body.title):
                raise HTTPException(404, "conversation not found")
            return schemas.ConversationOut.model_validate(repo.get_conversation(db, user.id, conversation_id))

    @app.delete("/conversations/{conversation_id}", status_code=204)
    def delete(conversation_id: str, request: Request, who: Caller = Depends(caller)):
        with database(request) as db:
            if not repo.delete_conversation(db, user_in(db, who).id, conversation_id):
                raise HTTPException(404, "conversation not found")

    @app.post("/conversations/{conversation_id}/messages")
    async def ask(conversation_id: str, body: schemas.Ask, request: Request, who: Caller = Depends(caller)):
        """Ask a question: Server-Sent Events with live progress, ending with the saved answer."""
        app_ = request.app
        with database(request) as db:
            user = user_in(db, who)
            conversation_ = repo.get_conversation(db, user.id, conversation_id)
            if conversation_ is None:
                raise HTTPException(404, "conversation not found")
            allowed, _ = repo.check_quota(db, user)
            if not allowed:
                raise HTTPException(429, f"Daily limit of {repo.quota_for(user)} questions reached. "
                                         "It resets at midnight UTC.")
            question = " ".join(body.question.split())
            turns = repo.recent_turns(db, conversation_id)  # read BEFORE saving: the question is not its own context
            saved = repo.add_message(db, conversation_, "user", question)
            user_id, question_message = user.id, schemas.MessageOut.model_validate(saved).model_dump(mode="json")

        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(produce_answer(app_, user_id, conversation_id, saved.id, question, turns,
                                                  queue.put_nowait))
        app_.state.tasks.add(task)  # keep a reference: the task outlives this request if the client leaves
        task.add_done_callback(app_.state.tasks.discard)

        async def events():
            yield sse({"type": "question", "message": question_message})
            while (event := await queue.get()) is not None:
                yield sse(event)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/messages/{message_id}/feedback", status_code=204)
    def feedback(message_id: str, body: schemas.FeedbackIn, request: Request, who: Caller = Depends(caller)):
        if body.rating not in (1, -1):
            raise HTTPException(422, "rating must be +1 or -1")
        with database(request) as db:
            if not repo.set_feedback(db, user_in(db, who).id, message_id, body.rating, body.comment):
                raise HTTPException(404, "answer not found")

    # -- companies and admin --

    @app.get("/companies", response_model=list[schemas.CompanyOut])
    def companies(who: Caller = Depends(caller)):
        from src.ingestion.loader import load_registry

        return [schemas.CompanyOut(ticker=d.ticker, company=d.company, doc_type=d.doc_type,
                                   fiscal_year_end=d.fiscal_year_end, filed=d.filed, source=d.source)
                for d in load_registry()]

    @app.post("/admin/companies", response_model=schemas.JobOut, status_code=202)
    def add_company(body: schemas.AddCompany, request: Request, who: Caller = Depends(caller)):
        """Queue "add this company's latest annual report" (~8 min); poll /admin/jobs/{id}."""
        with database(request) as db:
            admin = admin_in(db, who)
            args = body.model_dump(mode="json", exclude_none=True)
            return schemas.JobOut.model_validate(repo.create_job(db, "add_company", args, created_by=admin.id))

    @app.get("/admin/jobs", response_model=list[schemas.JobOut])
    def jobs(request: Request, who: Caller = Depends(caller)):
        with database(request) as db:
            admin_in(db, who)
            return [schemas.JobOut.model_validate(j) for j in repo.list_jobs(db)]

    @app.get("/admin/jobs/{job_id}", response_model=schemas.JobOut)
    def job(job_id: str, request: Request, who: Caller = Depends(caller)):
        with database(request) as db:
            admin_in(db, who)
            found = repo.get_job(db, job_id)
            if found is None:
                raise HTTPException(404, "job not found")
            return schemas.JobOut.model_validate(found)

    @app.post("/admin/reload", status_code=202)
    def reload(request: Request, who: Caller = Depends(caller)) -> dict:
        """Restart the MCP servers (e.g. after adding a company) once no question is running."""
        with database(request) as db:
            admin_in(db, who)
        request.app.state.team.needs_reload = True
        maybe_reload(request.app)
        return {"reload": "started" if request.app.state.active == 0 else "after the running questions finish"}

    return app


app = create_app()
