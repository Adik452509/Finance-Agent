"""Phase 5 critic, layer 1: code checks of the draft against the evidence. Offline."""

from src.agents.critic import code_checks, figures

FINDINGS = [{"step_id": 1, "kind": "filings", "goal": "Revenue both years", "facts":
             "- Revenue from Operations (2024-25, consolidated): 9,80,136 INR crore [RELIANCE.NS p100]\n"
             "- Revenue from Operations (2023-24, consolidated): 9,14,472 INR crore [RELIANCE.NS p100]"}]
ANALYSIS = "ANALYSIS:\n- Growth 2023-24 to 2024-25: 7.18% (formula: (new - old) / |old| x 100)"
CITATIONS = {"RELIANCE.NS p100": "Reliance Annual Report, PDF p. 100"}
GOOD = ("Reliance's consolidated revenue from operations grew 7.18% in 2024-25.\n"
        "- 2024-25: ₹9,80,136 crore [RELIANCE.NS p100]\n"
        "- 2023-24: ₹9,14,472 crore [RELIANCE.NS p100]\n"
        "- Growth: 7.18% (calculated from the figures above)")


def check(draft, findings=FINDINGS, analysis=ANALYSIS, citations=CITATIONS):
    return code_checks(draft, findings, analysis, citations)


def test_a_faithful_draft_has_no_issues():
    assert check(GOOD) == []


def test_invented_figure_is_caught():
    issues = check(GOOD.replace("7.18% in 2024-25", "7.18% in 2024-25, reaching ₹10,50,000 crore"))
    assert any("10,50,000" in i["problem"] and "not in the fact notes" in i["problem"] for i in issues)


def test_formatting_differences_are_not_issues():
    draft = ("Revenue grew 7.18%.\n- 2024-25: 980,136 crore [RELIANCE.NS p100]\n- 2023-24: 914472 crore [RELIANCE.NS p100]\n"
             "- 7.18% (calculated from the figures above)")
    assert check(draft) == []


def test_rounded_figure_is_caught():
    # The writer must copy figures exactly, not round them (7.2% is not what the calculator said).
    issues = check(GOOD.replace("Growth: 7.18%", "Growth: 7.2%"))
    assert any("7.2" in i["problem"] for i in issues)


def test_headline_may_repeat_a_figure_cited_below():
    assert check(GOOD) == []  # "grew 7.18%" in line 1 is uncited, but the bullet cites it


def test_figure_never_cited_is_caught():
    draft = "Revenue was ₹9,80,136 crore in 2024-25."
    assert any("never cited" in i["problem"] for i in check(draft))


def test_unverified_figure_is_caught():
    findings = [{"step_id": 1, "kind": "filings", "goal": "g",
                 "facts": "- UNVERIFIED (failed the source check, do not use): Revenue: 6,05,006 crore [RELIANCE.NS p100]"}]
    issues = check("Revenue was ₹6,05,006 crore [RELIANCE.NS p100].", findings=findings, analysis="")
    assert any("UNVERIFIED" in i["problem"] for i in issues)


def test_figure_on_both_verified_and_unverified_lines_is_allowed():
    findings = [{"step_id": 1, "kind": "filings", "goal": "g", "facts":
                 "- UNVERIFIED (failed the source check, do not use): x: 9,80,136 [RELIANCE.NS p100]\n"
                 "- Revenue (2024-25, consolidated): 9,80,136 crore [RELIANCE.NS p100]"}]
    assert check("Revenue was ₹9,80,136 crore [RELIANCE.NS p100].", findings=findings, analysis="") == []


def test_label_that_was_never_retrieved_is_caught():
    issues = check(GOOD.replace("2023-24: ₹9,14,472 crore [RELIANCE.NS p100]", "2023-24: ₹9,14,472 crore [RELIANCE.NS p999]"))
    assert any("[RELIANCE.NS p999] is not a retrieved source" in i["problem"] for i in issues)


def test_tool_name_label_is_caught():
    issues = check(GOOD + "\n- Source: [search_filings]")
    assert any("search_filings" in i["problem"] for i in issues)


def test_tickers_years_dates_and_times_are_not_figures():
    text = "Toyota (7203.T) and Tencent (0700.HK) in FY2026, 23 Oct 2026 20:17:49 GMT, 2024-25, 5 articles"
    assert figures(text) == []


def test_market_and_news_citations_count():
    findings = [{"step_id": 1, "kind": "market", "goal": "price",
                 "facts": "- Share price: 3,105 JPY (Yahoo Finance, as of 2026-10-25)"}]
    assert check("Toyota trades at ¥3,105 (Yahoo Finance, as of 2026-10-25).", findings=findings, analysis="") == []
    news = [{"step_id": 1, "kind": "news", "goal": "news",
             "facts": "- Toyota sold 201,306 vehicles (autoevolution.com, 23 Oct 2026)"}]
    assert check("Toyota sold 201,306 vehicles (autoevolution.com, 23 Oct 2026).", findings=news, analysis="") == []


def test_each_problem_is_reported_once():
    draft = "Revenue was ₹12,345 crore. Again ₹12,345 crore."
    assert len(check(draft)) == 1
