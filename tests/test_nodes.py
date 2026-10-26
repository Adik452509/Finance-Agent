"""Phase 4 agents (planner, researcher, analyst, writer) with scripted models. Offline."""

import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from src.agents.nodes import (
    CALC_TOOLS, RESEARCH_TOOLS, companies_context, format_findings, make_analyst, make_planner,
    make_researcher, make_writer,
)
from src.observability.guardrails import DISCLAIMER
from tests.conftest import scripted, tool_call

RIL_CITATION = "Reliance Industries Limited Annual Report (FY ended 2025-03-31), PDF p. 100 (printed p. 196 197)"
GOOD_PLAN = ('{"steps": [{"kind": "filings", "ticker": "RELIANCE.NS", "goal": "Consolidated revenue from operations '
             'for 2024-25 and 2023-24"}], "needs_calculation": true, "calculation": "percent change"}')


def search_filings(query: str, ticker: str | None = None, k: int = 5) -> str:
    """Search annual reports."""
    return json.dumps({"ok": True, "results": [{"source_id": "RELIANCE.NS p100", "citation": RIL_CITATION,
                                                "text": "Revenue from Operations 25 9,80,136 9,14,472"}]})


def get_quote(ticker: str) -> str:
    """Live price."""
    return json.dumps({"ok": True, "price": 3105, "currency": "JPY", "as_of": "2026-10-24"})


def percent_change(old_value: str, new_value: str) -> str:
    """Percent change."""
    return json.dumps({"ok": True, "result_percent": 7.18})


TOOLS = [StructuredTool.from_function(f) for f in (search_filings, get_quote, percent_change)]
FACTS = "- Revenue from operations (2024-25, consolidated): ₹9,80,136 crore [RELIANCE.NS p100]"


def run(node, state):
    return asyncio.run(node(state))


def plan_state(steps=None, needs_calc=True):
    steps = steps or [{"id": 1, "kind": "filings", "ticker": "RELIANCE.NS", "goal": "Revenue both years"}]
    return {"question": "How much did Reliance's revenue grow?",
            "plan": {"steps": steps, "needs_calculation": needs_calc, "calculation": "percent change"}}


# --- planner ---------------------------------------------------------------------------------


def test_companies_context_lists_every_indexed_report():
    text = companies_context()
    for ticker in ("AAPL", "RELIANCE.NS", "7203.T"):
        assert ticker in text
    assert "INR crore" in text


def test_planner_produces_a_validated_plan_and_counts_tokens():
    model = scripted(AIMessage(content=GOOD_PLAN, usage_metadata={"input_tokens": 500, "output_tokens": 80, "total_tokens": 580}))
    out = run(make_planner(model, companies="- test company"), {"question": "Reliance revenue growth?"})
    assert out["plan"]["steps"][0]["ticker"] == "RELIANCE.NS" and out["plan"]["needs_calculation"]
    assert out["usage"] == {"planner": 580} and out["warnings"] == []


def test_planner_prompt_has_no_unfilled_placeholders():
    model = scripted(AIMessage(content=GOOD_PLAN))
    run(make_planner(model, companies="- Apple Inc. (AAPL)"), {"question": "q?"})
    system = model.seen[0][0].content
    assert "<<" not in system and "- Apple Inc. (AAPL)" in system and "1 to 5 steps" in system


def test_planner_retries_once_with_the_reason():
    model = scripted(AIMessage(content="Sure! First I will look at the annual report."), AIMessage(content=GOOD_PLAN))
    out = run(make_planner(model, companies=""), {"question": "Reliance revenue growth?"})
    assert out["plan"]["steps"][0]["kind"] == "filings"
    assert len(model.seen) == 2 and "not usable" in model.seen[1][-1].content


def test_planner_falls_back_after_two_failures():
    model = scripted(AIMessage(content="no plan"), AIMessage(content="still no plan"))
    out = run(make_planner(model, companies=""), {"question": "By what percentage did revenue grow?"})
    assert out["plan"]["steps"][0]["kind"] == "any" and out["plan"]["needs_calculation"]
    assert any("fallback" in w for w in out["warnings"])


# --- researcher ------------------------------------------------------------------------------


def test_researcher_writes_fact_notes_and_collects_citations():
    model = scripted(tool_call("search_filings", {"query": "revenue", "ticker": "RELIANCE.NS", "k": 3}), AIMessage(content=FACTS))
    out = run(make_researcher(model, TOOLS), plan_state())
    assert out["findings"][0]["facts"] == FACTS and out["findings"][0]["step_id"] == 1
    assert out["citations"] == {"RELIANCE.NS p100": RIL_CITATION}
    assert out["tool_calls"][0]["agent"] == "researcher" and out["warnings"] == []


def test_researcher_cannot_use_calculator_tools():
    model = scripted(tool_call("percent_change", {"old_value": "1", "new_value": "2"}), AIMessage(content="- NOT FOUND: x"))
    out = run(make_researcher(model, TOOLS), plan_state())
    assert out["tool_calls"][0]["ok"] is False and "Unknown tool" in out["tool_calls"][0]["error"]


def test_researcher_warns_when_no_tool_was_used():
    out = run(make_researcher(scripted(AIMessage(content=FACTS)), TOOLS), plan_state())
    assert any("no tool was used" in w for w in out["warnings"])


def test_researcher_handles_each_step_in_order():
    steps = [{"id": 1, "kind": "market", "ticker": "7203.T", "goal": "Toyota share price"},
             {"id": 2, "kind": "filings", "ticker": "7203.T", "goal": "Toyota operating income FY2025 and FY2026"}]
    model = scripted(tool_call("get_quote", {"ticker": "7203.T"}), AIMessage(content="- Price: ¥3,105 (Yahoo Finance, as of 2026-10-24)"),
                     tool_call("search_filings", {"query": "operating income", "ticker": "7203.T"}, call_id="c2"),
                     AIMessage(content="- Operating income FY2026: ¥3,766,216 million [7203.T p164]"))
    out = run(make_researcher(model, TOOLS), plan_state(steps))
    assert [f["step_id"] for f in out["findings"]] == [1, 2]
    assert [c["name"] for c in out["tool_calls"]] == ["get_quote", "search_filings"]
    assert "Toyota operating income" in model.seen[2][1].content  # step 2 got its own brief


# --- analyst ---------------------------------------------------------------------------------


def test_analyst_uses_calculator_and_reports():
    analysis = "ANALYSIS:\n- Revenue growth: 7.18% (formula: (new - old) / |old| x 100; inputs: ...)"
    model = scripted(tool_call("percent_change", {"old_value": "9,14,472", "new_value": "9,80,136"}), AIMessage(content=analysis))
    state = {**plan_state(), "findings": [{"step_id": 1, "kind": "filings", "goal": "Revenue", "facts": FACTS}]}
    out = run(make_analyst(model, TOOLS), state)
    assert out["analysis"] == analysis and out["tool_calls"][0]["agent"] == "analyst" and out["warnings"] == []
    assert "9,80,136" in model.seen[0][1].content  # the analyst received the fact notes


def test_analyst_empty_text_falls_back_to_calculator_results():
    # Seen live: the analyst called percent_change, then returned no text at all.
    model = scripted(tool_call("percent_change", {"old_value": "9,14,472", "new_value": "9,80,136"}), AIMessage(content=""))
    out = run(make_analyst(model, TOOLS), plan_state())
    assert out["analysis"].startswith("ANALYSIS (assembled from calculator results):")
    assert "percent_change(old_value=9,14,472, new_value=9,80,136) = 7.18%" in out["analysis"]
    assert any("assembled from the calculator results" in w for w in out["warnings"])


def test_fact_check_catches_figures_missing_from_the_cited_source():
    from src.agents.nodes import check_facts_against_sources

    passages = {"RELIANCE.NS p100": "CONSOLIDATED FINANCIAL STATEMENTS\nRevenue from Operations 25 9,80,136 9,14,472"}
    good = "- Revenue from operations (2024-25, consolidated): ₹9,80,136 crore [RELIANCE.NS p100]"
    bad = "- Revenue from operations (2024-25, consolidated): ₹9,99,999 crore [RELIANCE.NS p100]"
    assert check_facts_against_sources(good, passages) == []
    assert any("9,99,999" in w for w in check_facts_against_sources(bad, passages))
    # Western grouping of the same figure still matches; years are not treated as figures.
    assert check_facts_against_sources("- Revenue (FY 2024-25): 980,136 crore [RELIANCE.NS p100]", passages) == []


def test_fact_check_catches_standalone_figure_labelled_consolidated():
    # Seen live: p74 (STANDALONE notes) "Value of Sales 5,17,428" reported as consolidated revenue.
    from src.agents.nodes import check_facts_against_sources

    passages = {"RELIANCE.NS p74": "STANDALONE FINANCIAL STATEMENTS\nValue of Sales 5,17,428 5,13,313"}
    facts = "- Revenue from Operations (FY 2024-25, consolidated): 5,17,428 crore [RELIANCE.NS p74]"
    warnings = check_facts_against_sources(facts, passages)
    assert any("STANDALONE" in w for w in warnings)
    assert check_facts_against_sources(facts.replace("consolidated", "standalone"), passages) == []


SEGMENT_PAGE = ("CONSOLIDATED FINANCIAL STATEMENTS\n37. Segment Information\n"
                "2024-25 O2C Oil and Gas Retail Digital Services Others Unallocable Total\n"
                "Revenue from Operations (Net of GST)\n6,05,006 24,881 2,91,043 1,31,336 49,749 - 9,80,136")


def test_fact_check_catches_company_total_from_wrong_segment_column():
    # Seen live twice: O2C's 6,05,006 (first column) reported as the consolidated total.
    from src.agents.nodes import check_facts_against_sources

    passages = {"RELIANCE.NS p130": SEGMENT_PAGE}
    wrong = "- Revenue from Operations (2024-25, consolidated): 6,05,006 INR crore [RELIANCE.NS p130]"
    right = "- Revenue from Operations (2024-25, consolidated): 9,80,136 INR crore [RELIANCE.NS p130]"
    segment = "- Revenue from Operations (2024-25, Retail segment): 2,91,043 INR crore [RELIANCE.NS p130]"
    assert any("last ('Total') column (9,80,136)" in w for w in check_facts_against_sources(wrong, passages))
    assert check_facts_against_sources(right, passages) == []
    assert check_facts_against_sources(segment, passages) == []


def test_segment_check_ignores_ordinary_multi_year_rows():
    # Toyota's income statement lists 3 years; FY2025 is the middle column but that is correct.
    from src.agents.nodes import check_facts_against_sources

    passages = {"7203.T p164": "CONSOLIDATED STATEMENT OF INCOME\nOperating income 5,352,934 4,795,586 3,766,216"}
    facts = "- Operating income (FY2025, consolidated): 4,795,586 JPY millions [7203.T p164]"
    assert check_facts_against_sources(facts, passages) == []


class RaisingModel(scripted().__class__):
    """Scripted model where an Exception in the script is raised instead of returned."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        self.seen.append(list(messages))
        item = next(self.messages)
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def raising(*replies) -> RaisingModel:
    model = RaisingModel(messages=iter(replies))
    model.seen = []
    return model


TOO_LARGE = Exception("Error code: 413 - Request too large for model on tokens per minute (TPM): Limit 8000")


def test_failed_research_step_is_reported_and_the_next_step_still_runs():
    steps = [{"id": 1, "kind": "filings", "ticker": "RELIANCE.NS", "goal": "Consolidated revenue 2024-25"},
             {"id": 2, "kind": "filings", "ticker": "RELIANCE.NS", "goal": "Retail segment revenue 2024-25"}]
    model = raising(TOO_LARGE, tool_call("search_filings", {"query": "retail"}), AIMessage(content=FACTS))
    out = run(make_researcher(model, TOOLS), plan_state(steps))
    assert out["findings"][0]["facts"].startswith("- NOT FOUND: this research step failed")
    assert out["findings"][1]["facts"] == FACTS
    assert any("step 1 failed" in w and "413" in w for w in out["warnings"])


def test_failed_analyst_leaves_an_empty_analysis_with_a_warning():
    out = run(make_analyst(raising(TOO_LARGE), TOOLS), plan_state())
    assert out["analysis"] == "" and any("analyst failed" in w for w in out["warnings"])


def test_planner_api_failure_uses_the_fallback_plan():
    daily = Exception("Rate limit reached on tokens per day (TPD): Limit 200000")
    out = run(make_planner(raising(daily), companies=""), {"question": "By what percentage did revenue grow?"})
    assert out["plan"]["steps"][0]["kind"] == "any"
    assert any("daily token budget" in w for w in out["warnings"])


def segment_search(query: str, ticker: str | None = None, k: int = 5) -> str:
    """Search annual reports."""
    return json.dumps({"ok": True, "results": [{"source_id": "RELIANCE.NS p130", "citation": "RIL p130",
                                                "text": SEGMENT_PAGE}]})


SEGMENT_TOOLS = [StructuredTool.from_function(segment_search, name="search_filings"),
                 StructuredTool.from_function(percent_change)]
WRONG_TOTAL = "- Revenue from Operations (2024-25, consolidated): 6,05,006 INR crore [RELIANCE.NS p130]"
RIGHT_TOTAL = "- Revenue from Operations (2024-25, consolidated): 9,80,136 INR crore [RELIANCE.NS p130]"


def test_flagged_notes_trigger_one_corrected_retry():
    # The eval's failure: the check caught the O2C column, but nothing acted on it.
    model = scripted(tool_call("search_filings", {"query": "revenue"}), AIMessage(content=WRONG_TOTAL),
                     tool_call("search_filings", {"query": "revenue total"}, call_id="c2"), AIMessage(content=RIGHT_TOTAL))
    out = run(make_researcher(model, SEGMENT_TOOLS), plan_state())
    assert out["findings"][0]["facts"] == RIGHT_TOTAL
    assert any("retrying step 1" in w for w in out["warnings"])
    assert any("corrected after the source check" in w for w in out["warnings"])
    assert len(out["tool_calls"]) == 2
    assert "failed an automatic source check" in model.seen[2][1].content  # retry brief names the problem


def test_still_wrong_after_retry_is_quarantined_as_unverified():
    model = scripted(tool_call("search_filings", {"query": "revenue"}), AIMessage(content=WRONG_TOTAL),
                     tool_call("search_filings", {"query": "revenue"}, call_id="c2"), AIMessage(content=WRONG_TOTAL))
    out = run(make_researcher(model, SEGMENT_TOOLS), plan_state())
    assert out["findings"][0]["facts"].startswith("- UNVERIFIED (failed the source check, do not use):")
    assert any("marked UNVERIFIED" in w for w in out["warnings"])


def test_clean_notes_are_not_retried():
    model = scripted(tool_call("search_filings", {"query": "revenue"}), AIMessage(content=RIGHT_TOTAL))
    out = run(make_researcher(model, SEGMENT_TOOLS), plan_state())
    assert out["findings"][0]["facts"] == RIGHT_TOTAL and len(model.seen) == 2 and out["warnings"] == []


def test_correction_brief_says_where_to_look_instead():
    from src.agents.nodes import correction_brief, fact_problems

    segment = correction_brief(fact_problems(WRONG_TOTAL, {"RELIANCE.NS p130": SEGMENT_PAGE}))
    assert "consolidated statement of profit and loss" in segment
    standalone = correction_brief([(0, "researcher: a 'consolidated' figure cites a STANDALONE statement page (X p1)")])
    assert "CONSOLIDATED financial statements" in standalone


def test_only_flagged_lines_are_quarantined():
    from src.agents.nodes import fact_problems, mark_unverified

    facts = RIGHT_TOTAL + "\n" + WRONG_TOTAL
    marked = mark_unverified(facts, fact_problems(facts, {"RELIANCE.NS p130": SEGMENT_PAGE}))
    assert marked.splitlines()[0] == RIGHT_TOTAL
    assert marked.splitlines()[1].startswith("- UNVERIFIED")


def test_researcher_runs_the_fact_check():
    bad = "- Revenue (2024-25, consolidated): ₹1,23,456 crore [RELIANCE.NS p100]"
    model = scripted(tool_call("search_filings", {"query": "revenue", "ticker": "RELIANCE.NS"}), AIMessage(content=bad),
                     tool_call("search_filings", {"query": "revenue"}, call_id="c2"), AIMessage(content=bad))
    out = run(make_researcher(model, TOOLS), plan_state())
    assert any("1,23,456" in w for w in out["warnings"])


# --- LLM source check (Step 5.6) ----------------------------------------------------------------
# The eval's published error: a figure that IS on the page, but from another segment's column.
# The code checks can't see it (the number exists, the line says "segment"); the verifier can.

WRONG_RETAIL = "- Revenue from Operations (2024-25, Retail segment): 24,881 INR crore [RELIANCE.NS p130]"
RIGHT_RETAIL = "- Revenue from Operations (2024-25, Retail segment): 2,91,043 INR crore [RELIANCE.NS p130]"


def verdicts(*supported, tokens=0):
    results = [{"fact": i, "supported": s, "reason": "Oil and Gas column, not Retail" if not s else "ok"}
               for i, s in enumerate(supported, start=1)]
    return AIMessage(content=json.dumps({"results": results}),
                     usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens})


def test_code_checks_cannot_see_a_figure_from_another_segment_column():
    from src.agents.nodes import fact_problems

    assert fact_problems(WRONG_RETAIL, {"RELIANCE.NS p130": SEGMENT_PAGE}) == []


def test_source_check_flags_only_unsupported_filing_lines():
    from src.agents.nodes import source_check

    facts = "\n".join(["- Price: ¥3,105 (Yahoo Finance, as of 2026-10-24)", WRONG_RETAIL, RIGHT_TOTAL,
                       "- UNVERIFIED (failed the source check, do not use): x 1,234 [RELIANCE.NS p130]"])
    verifier = scripted(verdicts(False, True, tokens=3000))
    problems, tokens, warnings = asyncio.run(source_check(verifier, facts, {"RELIANCE.NS p130": SEGMENT_PAGE}))
    assert problems == [(1, "researcher: source check - Oil and Gas column, not Retail")]
    assert tokens == 3000 and warnings == []
    sent = verifier.seen[0][1].content
    assert "SOURCE PAGE (RELIANCE.NS p130)" in sent and "6,05,006 24,881 2,91,043" in sent
    assert "1. Revenue from Operations (2024-25, Retail segment): 24,881" in sent
    assert "Yahoo" not in sent and "UNVERIFIED" not in sent  # only filings lines that are still in play


def test_source_check_skips_the_call_when_nothing_cites_a_filing():
    from src.agents.nodes import source_check

    verifier = scripted()  # no replies: calling it would raise
    assert asyncio.run(source_check(verifier, "- Price: ¥3,105 (Yahoo Finance, as of 2026-10-24)", {})) == ([], 0, [])
    assert verifier.seen == []


def test_a_broken_verifier_never_blocks_the_notes():
    from src.agents.nodes import source_check

    passages = {"RELIANCE.NS p130": SEGMENT_PAGE}
    problems, _, warnings = asyncio.run(source_check(raising(TOO_LARGE), WRONG_RETAIL, passages))
    assert problems == [] and "source check failed" in warnings[0]
    problems, _, warnings = asyncio.run(source_check(scripted(AIMessage(content="looks fine")), WRONG_RETAIL, passages))
    assert problems == [] and "not valid JSON" in warnings[0]


def test_verifier_rejection_triggers_the_corrected_retry():
    model = scripted(tool_call("search_filings", {"query": "retail"}), AIMessage(content=WRONG_RETAIL),
                     tool_call("search_filings", {"query": "segment retail"}, call_id="c2"), AIMessage(content=RIGHT_RETAIL))
    verifier = scripted(verdicts(False, tokens=3000), verdicts(True, tokens=2500))
    out = run(make_researcher(model, SEGMENT_TOOLS, verifier=verifier), plan_state())
    assert out["findings"][0]["facts"] == RIGHT_RETAIL
    assert out["usage"] == {"researcher": 0, "verifier": 5500}
    assert any("corrected after the source check" in w for w in out["warnings"])
    assert "table header" in model.seen[2][1].content  # the retry brief says how to find the column


def test_verifier_rejection_after_the_retry_quarantines_the_line():
    model = scripted(tool_call("search_filings", {"query": "retail"}), AIMessage(content=WRONG_RETAIL),
                     tool_call("search_filings", {"query": "retail"}, call_id="c2"), AIMessage(content=WRONG_RETAIL))
    verifier = scripted(verdicts(False), verdicts(False))
    out = run(make_researcher(model, SEGMENT_TOOLS, verifier=verifier), plan_state())
    assert out["findings"][0]["facts"].startswith("- UNVERIFIED")
    assert any("source check" in w and "marked UNVERIFIED" in w for w in out["warnings"])


def test_verifier_is_not_called_when_the_code_checks_already_flag_the_notes():
    model = scripted(tool_call("search_filings", {"query": "revenue"}), AIMessage(content=WRONG_TOTAL),
                     tool_call("search_filings", {"query": "revenue total"}, call_id="c2"), AIMessage(content=RIGHT_TOTAL))
    verifier = scripted(verdicts(True))  # one reply: only the corrected notes reach it
    out = run(make_researcher(model, SEGMENT_TOOLS, verifier=verifier), plan_state())
    assert out["findings"][0]["facts"] == RIGHT_TOTAL and len(verifier.seen) == 1
    assert "9,80,136" in verifier.seen[0][1].content


def test_repeated_search_results_are_kept_once():
    from src.agents.nodes import collect_passages

    passages: dict[str, str] = {}
    result = segment_search("retail")
    collect_passages(result, passages)
    collect_passages(result, passages)
    assert passages["RELIANCE.NS p130"].count("Segment Information") == 1


def test_manual_unit_conversion_via_ratio_is_flagged():
    # Seen live: crore -> billion done as ratio(980136, 1000) = 980.136 (correct: 9,801.36).
    from src.agents.nodes import manual_conversion_warnings

    def ratio_call(denominator):
        return {"name": "ratio", "args": {"numerator": "980136", "denominator": denominator}, "ok": True}

    assert manual_conversion_warnings([ratio_call("1000")])
    assert manual_conversion_warnings([ratio_call("1,00,000")])
    assert manual_conversion_warnings([ratio_call(100.0)])
    assert manual_conversion_warnings([ratio_call("416161")]) == []   # a real ratio, e.g. net margin
    assert manual_conversion_warnings([ratio_call("1")]) == []
    assert manual_conversion_warnings([{"name": "convert_units", "args": {}, "ok": True}]) == []


def test_analyst_warns_without_calculator_and_cannot_search():
    out = run(make_analyst(scripted(AIMessage(content="ANALYSIS: about 7%")), TOOLS), plan_state())
    assert any("no calculator tool" in w for w in out["warnings"])
    model = scripted(tool_call("search_filings", {"query": "x"}), AIMessage(content="ANALYSIS: none"))
    assert run(make_analyst(model, TOOLS), plan_state())["tool_calls"][0]["ok"] is False


# --- writer ----------------------------------------------------------------------------------


DRAFT = "Revenue grew 7.18%.\n- 2024-25: ₹9,80,136 crore [RELIANCE.NS p100]"


def test_writer_produces_a_draft_with_short_labels():
    model = scripted(AIMessage(content=DRAFT, usage_metadata={"input_tokens": 900, "output_tokens": 100, "total_tokens": 1000}))
    state = {"question": "q", "findings": [], "analysis": "ANALYSIS: 7.18%"}
    out = run(make_writer(model), state)
    assert out["draft"] == DRAFT and "[RELIANCE.NS p100]" in out["draft"]  # not expanded yet: the critic needs labels
    assert out["usage"] == {"writer": 1000} and "revisions" not in out
    assert "ANALYSIS: 7.18%" in model.seen[0][1].content


def test_writer_revises_using_the_critics_issues():
    model = scripted(AIMessage(content="Revenue grew 7.18% in 2024-25 [RELIANCE.NS p100]."))
    state = {"question": "q", "draft": "Revenue grew 7.18% in 2023-24.", "revisions": 0,
             "critique": {"verdict": "fail", "issues": [{"claim": "grew 7.18% in 2023-24", "problem": "wrong period",
                                                         "fix": "the growth is for 2024-25"}]}}
    out = run(make_writer(model), state)
    sent = model.seen[0][1].content
    assert "Your previous draft:\nRevenue grew 7.18% in 2023-24." in sent and "wrong period" in sent
    assert out["revisions"] == 1


def test_writer_failure_keeps_the_previous_draft():
    out = run(make_writer(raising(TOO_LARGE)), {"question": "q", "draft": "old draft"})
    assert "draft" not in out and any("writer failed" in w for w in out["warnings"])


def test_finalize_expands_citations_and_adds_disclaimer():
    from src.agents.nodes import finalize

    out = finalize({"draft": DRAFT, "citations": {"RELIANCE.NS p100": RIL_CITATION}})
    assert f"({RIL_CITATION})" in out["report"] and out["report"].endswith(DISCLAIMER)
    assert out["warnings"] == []


def test_finalize_flags_unknown_labels_and_empty_drafts():
    from src.agents.nodes import finalize

    assert any("get_quote" in w for w in finalize({"draft": "Price ¥3,105 [get_quote]", "citations": {}})["warnings"])
    out = finalize({"draft": ""})
    assert out["report"].startswith("No report could be produced.") and any("empty report" in w for w in out["warnings"])


def test_finalize_adds_a_verification_note_when_the_critic_is_still_unhappy():
    from src.agents.nodes import finalize

    state = {"draft": "Revenue grew 9%.", "revisions": 2,
             "critique": {"verdict": "fail", "issues": [{"claim": "grew 9%", "problem": "not in the analysis"}]}}
    out = finalize(state)
    assert "**Verification note:**" in out["report"] and "grew 9%: not in the analysis" in out["report"]
    assert any("remain after 2 revision(s)" in w for w in out["warnings"])
    assert "Verification note" not in finalize({"draft": "ok", "critique": {"verdict": "pass", "issues": []}})["report"]


def test_tool_sets_do_not_overlap_and_format_findings():
    assert not RESEARCH_TOOLS & CALC_TOOLS
    assert "Step 1 (filings): Revenue" in format_findings([{"step_id": 1, "kind": "filings", "goal": "Revenue", "facts": FACTS}])
    assert format_findings([]) == "(no fact notes)"
