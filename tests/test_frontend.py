"""Phase 6.5 Research Desk: answer parsing, the API client, and the app rendering with a fake backend."""

from datetime import date

import httpx
import pytest

from frontend import notes
from frontend.api_client import ApiClient, ApiError, BackendDown, parse_sse

TRACE = {
    "analysis": "ANALYSIS:\n- Apple's net income as a percentage of total net sales in fiscal 2025: 26.92% "
                "(formula: part / total x 100; inputs: ...)",
    "findings": [{"step_id": 1, "facts":
                  "- Total net sales (fiscal year ended 2025-09-27, consolidated): 416,161 USD millions [AAPL p32]\n"
                  "- Net income (fiscal year ended 2025-09-27, consolidated): 112,010 USD millions [AAPL p32]\n"
                  "- UNVERIFIED (failed the source check, do not use): Revenue: 999,999 [AAPL p32]\n"
                  "- NOT FOUND: Services gross margin"}],
    "critique": {"verdict": "pass"},
}


# --- key figures -------------------------------------------------------------------------------


def test_key_figures_come_from_checked_evidence_only():
    figures = notes.key_figures(TRACE)
    assert [f["value"] for f in figures] == ["26.92%", "416,161", "112,010"]  # never the UNVERIFIED 999,999
    assert figures[1] == {"label": "Total net sales", "value": "416,161", "unit": "USD millions",
                          "context": "fiscal year ended 2025-09-27, consolidated"}


def test_figure_formats_seen_live():
    market = {"findings": [{"facts": "- Revenue (fiscal year ended 2026-03-31, consolidated): 20,158,000,000 USD "
                                     "(20.16 B USD) (Yahoo Finance, as of 2026-10-29)"}]}
    assert notes.key_figures(market)[0]["value"] == "20.16 B" and notes.key_figures(market)[0]["unit"] == "USD"
    round_brackets = {"findings": [{"facts": "- Revenue from Operations (2024-25, consolidated): 9,80,136 INR crore "
                                             "(RELIANCE.NS p100)"}]}
    assert notes.key_figures(round_brackets)[0]["unit"] == "INR crore"  # the citation is not part of the unit
    assembled = {"analysis": "ANALYSIS (assembled from calculator results):\n- percent_change(old_value=914472, "
                             "new_value=980136) = 7.18% (formula: (new - old) / |old| x 100)"}
    assert notes.key_figures(assembled) == [{"label": "Change", "value": "7.18%", "unit": "", "context": "calculated"}]
    assert notes.key_figures(None) == [] and notes.key_figures({"findings": [{"facts": "- Revenue: see above"}]}) == []


def test_the_disclaimer_is_split_off():
    body, disclaimer = notes.split_report("Revenue was 331,839.\n\nThis is an automated research summary of company "
                                          "filings, not financial advice.")
    assert body == "Revenue was 331,839." and disclaimer.startswith("This is an automated")
    assert notes.split_report("No disclaimer") == ("No disclaimer", None)


# --- sources ---------------------------------------------------------------------------------------


def test_sources_separate_cited_pages_from_pages_only_read():
    answer = {"content": "Revenue was 331,839 (Microsoft Corp 10-K (FY ended 2026-06-30), p. 84 (printed p. 84)). "
                         "The share price is $512 (Yahoo Finance, as of 2026-10-30). Azure grew (reuters.com, 29 Oct 2026).",
              "citations": {"MSFT p84": "Microsoft Corp 10-K (FY ended 2026-06-30), p. 84 (printed p. 84)",
                            "MSFT p50": "Microsoft Corp 10-K (FY ended 2026-06-30), p. 50 (printed p. 50)"}}
    items = notes.sources(answer)
    assert [(i["kind"], i["cited"]) for i in items] == [("Annual report", True), ("Annual report", False),
                                                       ("Market data", True), ("News", True)]
    assert notes.source_kinds(answer) == [("Annual report", ":material/description:"),
                                          ("Market data", ":material/candlestick_chart:"),
                                          ("News", ":material/newspaper:")]


def test_verification_badges_always_have_icon_and_words():
    assert notes.verification({"trace": TRACE})[0] == "Checked against sources"
    assert notes.verification({"trace": {"critique": {"verdict": "fail"}}})[2] == "orange"
    assert notes.verification({"status": "error"})[0] == "Not answered"


def test_history_is_grouped_by_day():
    chats = [{"id": "a", "updated_at": "2026-10-30T09:00:00+00:00"}, {"id": "b", "updated_at": "2026-10-29T22:00:00Z"},
             {"id": "c", "updated_at": "2026-10-25T10:00:00"}, {"id": "d", "updated_at": "2026-09-01T10:00:00+00:00"}]
    groups = notes.group_by_day(chats, today=date(2026, 10, 30))
    assert [(label, [c["id"] for c in items]) for label, items in groups] == [
        ("Today", ["a"]), ("Yesterday", ["b"]), ("Previous 7 days", ["c"]), ("Older", ["d"])]
    assert notes.seconds_between("2026-10-30T09:00:00Z", "2026-10-30T09:00:12.5+00:00") == 12.5


# --- the API client ------------------------------------------------------------------------------------


def test_sse_lines_become_events():
    lines = ['event: progress', 'data: {"type": "progress", "agent": "planner"}', '', 'data: not json', '',
             'data: {"type": "answer", "message": {"id": "m1"}}']
    assert [e["type"] for e in parse_sse(iter(lines))] == ["progress", "answer"]


def test_client_sends_identity_and_maps_errors():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        if request.url.path == "/me":
            return httpx.Response(429, json={"detail": "Daily limit of 10 questions reached."})
        return httpx.Response(200, json=[])

    client = ApiClient("http://api", "a@b.com", "Søren", token="t0k", transport=httpx.MockTransport(handler))
    assert client.conversations() == []
    assert seen["x-user-email"] == "a@b.com" and seen["x-user-name"] == "S%C3%B8ren" and seen["x-api-token"] == "t0k"
    with pytest.raises(ApiError) as error:
        client.me()
    assert error.value.status == 429 and "Daily limit" in error.value.detail


def test_a_backend_that_is_down_is_reported_plainly():
    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(BackendDown):
        ApiClient("http://api", "a@b.com", transport=httpx.MockTransport(down)).me()


# --- the app, rendered with a fake backend ------------------------------------------------------------------


class FakeApi:
    def __init__(self, *args, **kwargs):
        pass

    def me(self):
        return {"email": "dev@example.com", "name": "Dev", "is_admin": False, "questions_left_today": 7, "daily_quota": 10}

    def conversations(self):
        return [{"id": "c1", "title": "Apple net margin", "created_at": "2026-10-30T09:00:00Z",
                 "updated_at": "2026-10-30T09:00:13Z"}]

    def companies(self):
        return [{"ticker": "AAPL", "company": "Apple Inc.", "doc_type": "10-K", "fiscal_year_end": "2025-09-27",
                 "filed": "", "source": ""}]

    def conversation(self, conversation_id):
        return {"id": "c1", "title": "Apple net margin", "created_at": "2026-10-30T09:00:00Z",
                "updated_at": "2026-10-30T09:00:13Z", "messages": [
                    {"id": "q1", "role": "user", "content": "Apple's net margin in fiscal 2025?", "status": "ok",
                     "created_at": "2026-10-30T09:00:00Z"},
                    {"id": "a1", "role": "assistant", "status": "ok", "tokens": 13993, "from_cache": False,
                     "content": "Net income was 26.92% of net sales (Apple Inc. 10-K (FY ended 2025-09-27), PDF p. 32).\n\n"
                                "This is an automated research summary of company filings, not financial advice.",
                     "citations": {"AAPL p32": "Apple Inc. 10-K (FY ended 2025-09-27), PDF p. 32"},
                     "trace": TRACE, "usage": {"planner": 1300, "researcher": 9000, "critic": 900},
                     "created_at": "2026-10-30T09:00:13Z"}]}


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import frontend.session as session

    from config.settings import settings

    monkeypatch.setattr(session, "ApiClient", FakeApi)
    return AppTest.from_file(str(settings.project_root / "frontend" / "app.py"), default_timeout=30)


def test_the_desk_renders_with_examples_and_the_quota(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "What do you want to research?"
    labels = [b.label for b in app.button]
    assert any("Growth" in label for label in labels) and "Apple net margin" in labels
    assert "7 of 10 questions left today" in app.sidebar.get("progress")[0].proto.text


def test_an_answer_renders_as_a_research_note(app):
    app.session_state["conversation_id"] = "c1"
    app.run()
    assert not app.exception
    assert [m.label for m in app.metric][:3] == [
        "Apple's net income as a percentage of total net sales in fiscal 2025", "Total net sales", "Net income"]
    assert [m.value for m in app.metric][:3] == ["26.92%", "416,161", "112,010"]
    assert [t.label for t in app.tabs] == ["Sources (1)", "Research trail", "Usage"]
    assert any("Checked against sources" in m.value for m in app.markdown)  # st.badge renders as markdown
