"""Phase 6.3 backend API with a fake research team and a temporary SQLite database. Offline."""

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from src.api import app as api
from src.api.chat import answer_record, describe_update
from src.api.team import Team
from src.api.worker import run_one_job
from src.db import repository as repo
from src.db.models import Base, Message
from src.db.session import make_engine

ALICE = {"X-User-Email": "alice@example.com", "X-User-Name": "Alice%20S%C3%B8rensen"}  # URL-encoded non-ASCII name
BOB = {"X-User-Email": "bob@example.com"}

FINAL = {"question": "q", "plan": {"steps": [{"id": 1, "kind": "filings", "ticker": "MSFT", "goal": "Total revenue"}],
                                    "needs_calculation": True},
         "route": "analyst", "findings": [{"step_id": 1, "facts": "- Total revenue: 331,839 USD millions [MSFT p84]"}],
         "analysis": "ANALYSIS:\n- share: 41.52%", "report": "Intelligent Cloud was 41.52% of revenue.",
         "citations": {"MSFT p84": "Microsoft Corp 10-K (FY ended 2026-06-30), p. 84"},
         "tool_calls": [{"agent": "researcher", "name": "search_filings", "ok": True}],
         "warnings": [], "critique": {"verdict": "pass", "issues": []}, "critic_rounds": 1,
         "usage": {"planner": 1000, "researcher": 8000, "critic": 900}}


class FakeToolbox:
    tools, failed = [], {"edinet": "disabled"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


class FakeGraph:
    def __init__(self, on_event, fail=None, delay=0.0, verdict="pass"):
        self.on_event, self.fail, self.delay, self.verdict = on_event, fail, delay, verdict
        self.researched: list[str] = []  # every question the "team" was asked to research

    async def astream(self, inputs, stream_mode):
        self.researched.append(inputs["question"])
        yield "updates", {"planner": {"plan": FINAL["plan"], "warnings": []}}
        self.on_event("call", {"agent": "researcher", "name": "search_filings", "args": {"query": "revenue"}})
        self.on_event("result", {"agent": "researcher", "name": "search_filings", "ok": True, "seconds": 0.4})
        await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        critique = {"verdict": self.verdict, "issues": [] if self.verdict == "pass" else [{"claim": "c", "problem": "p"}]}
        yield "updates", {"researcher": {"findings": FINAL["findings"]}}
        yield "updates", {"critic": {"critique": critique, "critic_rounds": 1}}
        yield "values", {**FINAL, "critique": critique}


def no_rewrite_expected():
    raise AssertionError("a first question must not build the rewrite model")


def make_client(tmp_path, fail=None, delay=0.0, verdict="pass", rewriter=no_rewrite_expected):
    engine = make_engine(f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    graphs: list[FakeGraph] = []

    def graph_factory(tools, on_event):
        graphs.append(FakeGraph(on_event, fail, delay, verdict))
        return graphs[-1]

    team = Team(toolbox_factory=FakeToolbox, graph_factory=graph_factory)
    client = TestClient(api.create_app(team=team, sessions=sessions, start_worker=False, rewriter=rewriter))
    client.graphs = graphs
    return client, sessions


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "internal_api_token", None)
    test_client, sessions = make_client(tmp_path)
    with test_client:
        test_client.sessions = sessions
        yield test_client


def events_of(response) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in response.iter_lines() if line.startswith("data: ")]


def ask(client, conversation_id, question="What share of revenue was Intelligent Cloud?", headers=ALICE):
    with client.stream("POST", f"/conversations/{conversation_id}/messages", json={"question": question},
                       headers=headers) as response:
        return response.status_code, (events_of(response) if response.status_code == 200 else response.read())


# --- access ------------------------------------------------------------------------------------


def test_health_needs_no_auth_but_everything_else_does(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/me").status_code == 401  # no user header


def test_the_api_token_is_enforced_when_set(client, monkeypatch):
    monkeypatch.setattr(api.settings, "internal_api_token", "s3cret-token-0123456789abcdef0123")
    assert client.get("/me", headers=ALICE).status_code == 401
    assert client.get("/me", headers={**ALICE, "X-Api-Token": "wrong"}).status_code == 401
    assert client.get("/me", headers={**ALICE, "X-Api-Token": "s3cret-token-0123456789abcdef0123"}).status_code == 200


def test_production_refuses_to_start_without_a_strong_token():
    from config.settings import Settings

    with pytest.raises(ValueError, match="INTERNAL_API_TOKEN"):
        Settings(_env_file=None, app_env="production", internal_api_token="short")


def test_me_creates_the_user_and_shows_the_quota(client, monkeypatch):
    monkeypatch.setattr(api.settings, "daily_question_quota", 3)
    me = client.get("/me", headers=ALICE).json()
    assert me == {"email": "alice@example.com", "name": "Alice Sørensen", "is_admin": False,
                  "questions_left_today": 3, "daily_quota": 3}


# --- conversations ------------------------------------------------------------------------------


def test_conversation_lifecycle_and_ownership(client):
    chat = client.post("/conversations", json={}, headers=ALICE).json()
    assert chat["title"] == "New chat"
    assert [c["id"] for c in client.get("/conversations", headers=ALICE).json()] == [chat["id"]]
    assert client.get("/conversations", headers=BOB).json() == []
    for method, body in (("get", None), ("patch", {"title": "x"}), ("delete", None)):
        assert client.request(method, f"/conversations/{chat['id']}", json=body, headers=BOB).status_code == 404
    assert client.patch(f"/conversations/{chat['id']}", json={"title": "MSFT cloud"}, headers=ALICE).json()["title"] == "MSFT cloud"
    assert client.delete(f"/conversations/{chat['id']}", headers=ALICE).status_code == 204
    assert client.get(f"/conversations/{chat['id']}", headers=ALICE).status_code == 404


# --- asking -------------------------------------------------------------------------------------------


def test_asking_streams_progress_and_saves_the_answer(client):
    chat = client.post("/conversations", json={}, headers=ALICE).json()
    status, events = ask(client, chat["id"])
    assert status == 200
    kinds = [e["type"] for e in events]
    assert kinds[0] == "question" and kinds[-1] == "answer" and "tool" in kinds and "tool_result" in kinds
    assert [e["agent"] for e in events if e["type"] == "progress"] == ["planner", "researcher", "critic"]
    answer = events[-1]["message"]
    assert answer["role"] == "assistant" and answer["content"] == FINAL["report"] and answer["tokens"] == 9900
    assert answer["citations"]["MSFT p84"].startswith("Microsoft Corp 10-K")

    detail = client.get(f"/conversations/{chat['id']}", headers=ALICE).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["title"].startswith("What share of revenue")
    assert detail["messages"][1]["trace"]["findings"][0]["facts"].startswith("- Total revenue")
    assert client.get("/me", headers=ALICE).json()["questions_left_today"] == 9  # one question counted


def test_a_failed_run_is_saved_as_an_error_and_costs_no_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "internal_api_token", None)
    failing, _ = make_client(tmp_path, fail=Exception("Rate limit reached on tokens per day (TPD)"))
    with failing:
        chat = failing.post("/conversations", json={}, headers=ALICE).json()
        _, events = ask(failing, chat["id"])
        assert any(e["type"] == "error" and "budget is used up" in e["text"] for e in events)
        assert events[-1]["message"]["status"] == "error"
        assert failing.get("/me", headers=ALICE).json()["questions_left_today"] == 10


def test_the_daily_limit_is_enforced(client, monkeypatch):
    monkeypatch.setattr(api.settings, "daily_question_quota", 1)
    chat = client.post("/conversations", json={}, headers=ALICE).json()
    assert ask(client, chat["id"])[0] == 200
    status, body = ask(client, chat["id"])
    assert status == 429 and b"Daily limit of 1" in body


def test_asking_in_someone_elses_chat_is_not_found(client):
    chat = client.post("/conversations", json={}, headers=ALICE).json()
    assert ask(client, chat["id"], headers=BOB)[0] == 404
    assert ask(client, chat["id"], question="hi")[0] == 422  # too short to be a question


def test_the_answer_is_saved_even_if_the_browser_disconnects(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "internal_api_token", None)
    slow, sessions = make_client(tmp_path, delay=0.5)
    with slow:
        chat = slow.post("/conversations", json={}, headers=ALICE).json()
        with slow.stream("POST", f"/conversations/{chat['id']}/messages", json={"question": "Apple revenue?"},
                         headers=ALICE) as response:
            next(response.iter_lines())  # read the first event, then walk away
        for _ in range(50):
            with sessions() as db:
                if db.query(Message).filter_by(role="assistant").count():
                    break
            time.sleep(0.1)
        roles = [m["role"] for m in slow.get(f"/conversations/{chat['id']}", headers=ALICE).json()["messages"]]
        assert roles == ["user", "assistant"]


# --- follow-ups and the answer cache (6.4) -------------------------------------------------------------


def test_a_follow_up_is_rewritten_with_the_chat_history(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from tests.conftest import scripted

    monkeypatch.setattr(api.settings, "internal_api_token", None)
    model = scripted(AIMessage(content="What was Apple's net income in fiscal 2024?",
                               usage_metadata={"input_tokens": 600, "output_tokens": 20, "total_tokens": 620}))
    client, sessions = make_client(tmp_path, rewriter=lambda: model)
    with client:
        chat = client.post("/conversations", json={}, headers=ALICE).json()
        ask(client, chat["id"], question="What was Apple's net income in fiscal 2025?")
        _, events = ask(client, chat["id"], question="and in 2024?")
        assert any(e.get("agent") == "memory" and "fiscal 2024" in e["text"] for e in events)
        assert client.graphs[0].researched == ["What was Apple's net income in fiscal 2025?",
                                               "What was Apple's net income in fiscal 2024?"]
        history = model.seen[0][1].content
        assert "User: What was Apple's net income in fiscal 2025?" in history and "Latest message: and in 2024?" in history
        assert "This is an automated research summary" not in history  # the disclaimer is noise
        messages = client.get(f"/conversations/{chat['id']}", headers=ALICE).json()["messages"]
        assert messages[-1]["usage"]["memory"] == 620
        with sessions() as db:
            follow_up = db.query(Message).filter_by(content="and in 2024?").one()
            assert follow_up.standalone_question == "What was Apple's net income in fiscal 2024?"


def test_the_same_question_is_answered_from_the_cache_for_free(client):
    first = client.post("/conversations", json={}, headers=ALICE).json()
    ask(client, first["id"], question="What was Apple's net income in fiscal 2025?")
    other = client.post("/conversations", json={}, headers=BOB).json()  # another user, other words
    _, events = ask(client, other["id"], question="  what was Apple's net income in fiscal 2025 ", headers=BOB)
    answer = events[-1]["message"]
    assert answer["from_cache"] and answer["tokens"] == 0 and answer["content"] == FINAL["report"]
    assert any(e.get("agent") == "cache" for e in events)
    assert len(client.graphs[0].researched) == 1                              # researched once only
    assert client.get("/me", headers=BOB).json()["questions_left_today"] == 10  # a cache hit costs no quota


def test_a_doubtful_answer_is_never_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "internal_api_token", None)
    client, _ = make_client(tmp_path, verdict="fail")
    with client:
        for _ in range(2):
            chat = client.post("/conversations", json={}, headers=ALICE).json()
            ask(client, chat["id"], question="What was Apple's net income in fiscal 2025?")
        assert len(client.graphs[0].researched) == 2


# --- feedback --------------------------------------------------------------------------------------


def test_feedback_on_your_own_answer(client):
    chat = client.post("/conversations", json={}, headers=ALICE).json()
    answer_id = ask(client, chat["id"])[1][-1]["message"]["id"]
    assert client.post(f"/messages/{answer_id}/feedback", json={"rating": -1, "comment": "wrong year"},
                       headers=BOB).status_code == 404
    assert client.post(f"/messages/{answer_id}/feedback", json={"rating": 5}, headers=ALICE).status_code == 422
    assert client.post(f"/messages/{answer_id}/feedback", json={"rating": 1}, headers=ALICE).status_code == 204


# --- companies, admin, jobs ------------------------------------------------------------------------------


def test_companies_are_listed(client):
    tickers = {c["ticker"] for c in client.get("/companies", headers=ALICE).json()}
    assert {"AAPL", "RELIANCE.NS", "7203.T"} <= tickers


def test_only_admins_queue_jobs_and_the_worker_runs_them(client, monkeypatch):
    monkeypatch.setattr(api.settings, "admin_emails", "alice@example.com")
    body = {"ticker": "NVDA", "max_chunks": 200}
    assert client.post("/admin/companies", json=body, headers=BOB).status_code == 403
    job = client.post("/admin/companies", json=body, headers=ALICE).json()
    assert job["status"] == "queued" and job["args"]["ticker"] == "NVDA"

    def fake_runner(args, progress):
        progress("Embedding: batch 1/1")
        return {"complete": True, "embedded": 120, "company": "Nvidia Corp"}

    assert asyncio.run(run_one_job(client.sessions, {"add_company": fake_runner})) == "done"
    done = client.get(f"/admin/jobs/{job['id']}", headers=ALICE).json()
    assert done["status"] == "done" and done["result"]["embedded"] == 120 and done["finished_at"]
    assert [j["id"] for j in client.get("/admin/jobs", headers=ALICE).json()] == [job["id"]]
    assert client.get("/admin/jobs", headers=BOB).status_code == 403


def test_a_failing_job_is_marked_failed(client):
    with client.sessions() as db:
        job = repo.create_job(db, "add_company", {"ticker": "ZZZZ"})
        db.commit()

    def broken(args, progress):
        raise RuntimeError("ZZZZ is not in the SEC's company list")

    assert asyncio.run(run_one_job(client.sessions, {"add_company": broken})) == "failed"
    with client.sessions() as db:
        assert "not in the SEC" in repo.get_job(db, job.id).result["error"]


def test_bad_company_requests_are_rejected(client, monkeypatch):
    monkeypatch.setattr(api.settings, "admin_emails", "alice@example.com")
    assert client.post("/admin/companies", json={"ticker": "MSFT; rm -rf /"}, headers=ALICE).status_code == 422
    pdf = {"url": "http://insecure.example.com/a.pdf", "company": "X", "market": "India", "doc_type": "AR",
           "fiscal_year_end": "2025-03-31", "currency": "INR"}
    assert client.post("/admin/companies", json={"ticker": "X.NS", "pdf": pdf}, headers=ALICE).status_code == 422


def test_status_shows_the_team(client):
    status = client.get("/status", headers=ALICE).json()
    assert status["team_ready"] and status["unavailable_servers"] == {"edinet": "disabled"}
    assert "***" in status["database"] or status["database"].startswith("sqlite")


# --- progress text -----------------------------------------------------------------------------------


def test_progress_lines_are_readable():
    assert describe_update("planner", {"plan": FINAL["plan"]})["text"] == "Planned 1 research step"
    assert describe_update("supervisor", {"route": "analyst"})["text"].startswith("Calculation needed")
    fail = describe_update("critic", {"critique": {"verdict": "fail", "issues": [{"claim": "c", "problem": "p"}]}})
    assert fail["text"] == "Found 1 issue(s) in the draft" and fail["detail"] == ["c: p"]
    assert describe_update("unknown_node", {}) is None
    record = answer_record(FINAL)
    assert record["tokens"] == 9900 and record["trace"]["route"] == "analyst"
