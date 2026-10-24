"""The pipeline eval's question file and scoring logic. Offline (no pipeline run)."""

import json

import pytest

from config.settings import settings
from src.evaluation.checks import check_result, has_number

QUESTIONS = json.loads((settings.project_root / "tests" / "eval" / "pipeline_questions.json").read_text(encoding="utf-8"))["questions"]
DISCLAIMER = "\n\nThis is an automated research summary of company filings, not financial advice."


def final(report="", tools=(), route="analyst", warnings=()):
    return {"report": report + DISCLAIMER, "route": route, "warnings": list(warnings),
            "tool_calls": [{"name": t, "ok": True} for t in tools]}


# --- the question file ---------------------------------------------------------------------


def test_question_file_is_well_formed():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids)) >= 8
    for q in QUESTIONS:
        assert q["question"].strip() and set(q["expect"]) >= {"numbers", "must_not", "tools", "route"}
        assert q["expect"]["route"] in ("analyst", "writer", None)


def test_known_wrong_figures_are_not_also_expected():
    for q in QUESTIONS:
        assert not set(q["expect"]["numbers"]) & set(q["expect"]["must_not"]), q["id"]


# --- number matching ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, number, expected", [
    ("grew 7.18% in 2024-25", "7.18", True),
    ("grew 7.185%", "7.18", False),
    ("about 7.2%", "7.2", True),
    ("grew 7.25%", "7.2", False),
    ("9,801.36 billion", "9801.36", True),
    ("¥50,684,952 million", "50,684,952", True),
    ("fell by 21.46%", "21.46", True),
    ("19.50 %", "19.5", True),
    ("219.5", "19.5", False),
])
def test_has_number(text, number, expected):
    assert has_number(text, number) is expected


# --- scoring -----------------------------------------------------------------------------------

GROWTH = next(q for q in QUESTIONS if q["id"] == "ril_growth")["expect"]


def test_correct_run_passes():
    r = check_result(GROWTH, final("Revenue grew 7.18%.", tools=("search_filings", "percent_change")))
    assert r.passed and r.failures == []


def test_wrong_figure_fails():
    r = check_result(GROWTH, final("Revenue grew 48.11%.", tools=("search_filings", "percent_change")))
    assert not r.passed and any("missing" in f for f in r.failures) and any("48.11" in f for f in r.failures)


def test_right_number_without_the_calculator_fails():
    # The figure alone is not enough: it must come from the calculator tool.
    r = check_result(GROWTH, final("Revenue grew 7.18%.", tools=("search_filings",)))
    assert not r.passed and any("percent_change" in f for f in r.failures)


def test_failed_tool_calls_do_not_count():
    run = final("Revenue grew 7.18%.", tools=("search_filings",))
    run["tool_calls"].append({"name": "percent_change", "ok": False})
    assert not check_result(GROWTH, run).passed


def test_wrong_route_fails():
    news = next(q for q in QUESTIONS if q["id"] == "toyota_news")["expect"]
    r = check_result(news, final("Toyota launched X (reuters.com, 23 Oct 2026).", tools=("search_news",), route="analyst"))
    assert not r.passed and any("route" in f for f in r.failures)


def test_news_must_be_cited_with_site_and_date():
    news = next(q for q in QUESTIONS if q["id"] == "toyota_news")["expect"]
    assert check_result(news, final("Toyota launched X (reuters.com, Fri, 23 Oct 2026).", ("search_news",), "writer")).passed
    assert not check_result(news, final("Toyota launched X.", ("search_news",), "writer")).passed


def test_refusal_question():
    nf = next(q for q in QUESTIONS if q["id"] == "not_found_future_year")["expect"]
    assert check_result(nf, final("Fiscal 2031 figures are not available in the filings.", route="writer")).passed
    assert not check_result(nf, final("Apple's revenue grew 5% in fiscal 2031.", route="writer")).passed


def test_disclaimer_text_cannot_satisfy_a_refusal():
    nf = next(q for q in QUESTIONS if q["id"] == "not_found_future_year")["expect"]
    run = final("Revenue grew 5%.", route="writer")
    run["report"] += " Figures not available."  # appended after the disclaimer split point
    assert not check_result(nf, run).passed


def test_warnings_are_reported_but_do_not_fail():
    r = check_result(GROWTH, final("Grew 7.18%.", ("search_filings", "percent_change"), warnings=["analyst returned no text"]))
    assert r.passed and r.notes and "analyst returned no text" in r.notes[0]


def test_empty_report_fails():
    r = check_result({"numbers": [], "must_not": [], "tools": [], "route": None}, {"report": "No report could be produced."})
    assert not r.passed
