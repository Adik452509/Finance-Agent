"""Phase 5 critic: layer 1 (code checks) and layer 2 (LLM judge, scripted). Offline."""

import asyncio

from langchain_core.messages import AIMessage

from src.agents.critic import code_checks, figures, make_critic, parse_verdict
from tests.conftest import scripted

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


# --- layer 2: verdict parsing --------------------------------------------------------------------

PERIOD_ISSUE = ('{"verdict": "fail", "issues": [{"claim": "grew 7.18% in 2023-24", '
                '"problem": "the growth happened in 2024-25", "fix": "grew 7.18% in 2024-25"}]}')


def test_parse_pass_and_fail():
    assert parse_verdict('{"verdict": "pass", "issues": []}') == {"verdict": "pass", "issues": []}
    v = parse_verdict(PERIOD_ISSUE)
    assert v["verdict"] == "fail" and v["issues"][0]["source"] == "llm" and "2024-25" in v["issues"][0]["fix"]


def test_parse_handles_fences_and_normalises_contradictions():
    assert parse_verdict("```json\n" + PERIOD_ISSUE + "\n```")["verdict"] == "fail"
    # "pass" but issues listed -> fail; "fail" with nothing usable -> pass (nothing to fix)
    assert parse_verdict('{"verdict": "PASS", "issues": [{"problem": "wrong unit"}]}')["verdict"] == "fail"
    assert parse_verdict('{"verdict": "fail", "issues": [{"claim": "x"}]}') == {"verdict": "pass", "issues": []}


def test_parse_rejects_unusable_replies():
    for bad in ("looks fine to me", '{"verdict": "maybe"}', "[1, 2]", ""):
        assert parse_verdict(bad) is None


def test_parse_caps_the_number_of_issues():
    many = '{"verdict": "fail", "issues": [' + ",".join(f'{{"problem": "p{i}"}}' for i in range(9)) + "]}"
    assert len(parse_verdict(many)["issues"]) == 5


# --- layer 2: the critic node ----------------------------------------------------------------------

STATE = {"question": "Reliance revenue growth?", "findings": FINDINGS, "analysis": ANALYSIS,
         "citations": CITATIONS, "draft": GOOD}


def run_critic(llm, state=STATE):
    return asyncio.run(make_critic(llm)(state))


def test_code_issues_skip_the_llm_to_save_tokens():
    judge = scripted()
    out = run_critic(judge, {**STATE, "draft": "Revenue was ₹12,345 crore."})
    assert judge.seen == [] and out["warnings"] == []  # the judge was never contacted
    assert out["critique"]["verdict"] == "fail" and out["critique"]["checked_by"] == ["code"]
    assert out["critique"]["issues"][0]["source"] == "code" and "usage" not in out


def test_llm_judge_catches_a_meaning_level_error():
    judge = scripted(AIMessage(content=PERIOD_ISSUE, usage_metadata={"input_tokens": 1500, "output_tokens": 80, "total_tokens": 1580}))
    out = run_critic(judge)
    assert out["critique"]["verdict"] == "fail" and out["critique"]["checked_by"] == ["code", "llm"]
    assert out["critique"]["issues"][0]["problem"] == "the growth happened in 2024-25"
    assert out["usage"] == {"critic": 1580}
    sent = judge.seen[0][1].content
    assert "DRAFT:" in sent and "₹9,80,136" in sent and "ANALYSIS:" in sent


def test_clean_draft_passes_both_layers():
    out = run_critic(scripted(AIMessage(content='{"verdict": "pass", "issues": []}')))
    assert out["critique"] == {"verdict": "pass", "issues": [], "checked_by": ["code", "llm"]}


def test_unreadable_judge_reply_falls_back_to_code_checks():
    out = run_critic(scripted(AIMessage(content="I think it is mostly fine.")))
    assert out["critique"]["verdict"] == "pass" and out["critique"]["checked_by"] == ["code"]
    assert any("not valid JSON" in w for w in out["warnings"])


def test_judge_failure_does_not_block_the_report():
    from tests.test_nodes import TOO_LARGE, raising

    out = run_critic(raising(TOO_LARGE))
    assert out["critique"]["verdict"] == "pass" and any("judge failed" in w for w in out["warnings"])


def test_critic_without_llm_uses_code_checks_only():
    assert run_critic(None)["critique"]["checked_by"] == ["code"]
