"""Background worker: runs queued jobs (today: "add_company", ~8 minutes of embedding).

A job row is the hand-off: the admin page creates it, this loop claims it, runs it in a
thread (the work is blocking: downloads, PDF parsing, embedding calls) and writes its
progress back, so the page can poll /admin/jobs/{id}. A job survives a closed browser.
"""

import asyncio
import logging
from collections.abc import Callable

from sqlalchemy.orm import Session, sessionmaker

from src.db import repository as repo
from src.db.session import session_scope

logger = logging.getLogger(__name__)

POLL_SECONDS = 5


def _run_add_company(args: dict, progress: Callable[[str], None]) -> dict:
    from src.ingestion.acquire import add_company

    result = add_company(args["ticker"], pdf=args.get("pdf"), max_chunks=args.get("max_chunks", 250),
                         force=args.get("force", False), progress=progress)
    return {k: v for k, v in result.items() if k != "entry"} | {"company": result["entry"]["company"],
                                                                 "fiscal_year_end": result["entry"]["fiscal_year_end"]}


JOB_RUNNERS: dict[str, Callable[[dict, Callable[[str], None]], dict]] = {"add_company": _run_add_company}


async def run_one_job(sessions: sessionmaker[Session], runners: dict | None = None) -> str | None:
    """Claim and run the oldest queued job; returns its status, or None if there was nothing to do."""
    runners = runners or JOB_RUNNERS
    with session_scope(sessions) as db:
        job = repo.claim_next_job(db)
        if job is None:
            return None
        job_id, kind, args = job.id, job.kind, dict(job.args)

    def progress(message: str) -> None:  # called from the worker thread: its own short session
        with session_scope(sessions) as db:
            repo.update_job(db, job_id, progress=message[:500])

    runner = runners.get(kind)
    try:
        if runner is None:
            raise ValueError(f"unknown job kind {kind!r}")
        result = await asyncio.to_thread(runner, args, progress)
        status = "done" if result.get("complete", True) else "failed"
        message = "Done" if status == "done" else f"Stopped: {result.get('stopped_reason')}"
    except Exception as e:  # the job fails, the worker keeps running
        logger.exception("Job %s (%s) failed", job_id, kind)
        result, status = {"error": f"{type(e).__name__}: {str(e)[:300]}"}, "failed"
        message = result["error"]
    with session_scope(sessions) as db:
        repo.update_job(db, job_id, status=status, progress=message[:500], result=result)
    return status


async def job_worker(sessions: sessionmaker[Session], on_done: Callable[[str, str], None] | None = None,
                     poll_seconds: float = POLL_SECONDS) -> None:
    """Run jobs forever (cancelled when the backend stops)."""
    while True:
        try:
            status = await run_one_job(sessions)
        except Exception:  # e.g. the database is briefly unreachable
            logger.exception("Job worker error")
            status = None
        if status is None:
            await asyncio.sleep(poll_seconds)
        elif on_done:
            on_done("job", status)
