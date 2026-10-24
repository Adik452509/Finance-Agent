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


def test_researcher_runs_the_fact_check():
    model = scripted(tool_call("search_filings", {"query": "revenue", "ticker": "RELIANCE.NS"}),
                     AIMessage(content="- Revenue (2024-25, consolidated): ₹1,23,456 crore [RELIANCE.NS p100]"))
    out = run(make_researcher(model, TOOLS), plan_state())
    assert any("1,23,456" in w for w in out["warnings"])


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


def test_writer_expands_citations_and_adds_disclaimer():
    model = scripted(AIMessage(content="Revenue grew 7.18%.\n- 2024-25: ₹9,80,136 crore [RELIANCE.NS p100]",
                               usage_metadata={"input_tokens": 900, "output_tokens": 100, "total_tokens": 1000}))
    state = {"question": "q", "findings": [], "citations": {"RELIANCE.NS p100": RIL_CITATION}, "analysis": "ANALYSIS: 7.18%"}
    out = run(make_writer(model), state)
    assert f"({RIL_CITATION})" in out["report"] and out["report"].endswith(DISCLAIMER)
    assert out["usage"] == {"writer": 1000} and out["warnings"] == []
    assert "ANALYSIS: 7.18%" in model.seen[0][1].content


def test_writer_flags_unknown_labels_and_empty_reports():
    out = run(make_writer(scripted(AIMessage(content="Price ¥3,105 [get_quote]"))), {"question": "q", "citations": {}})
    assert any("get_quote" in w for w in out["warnings"])
    out = run(make_writer(scripted(AIMessage(content=""))), {"question": "q"})
    assert out["report"].startswith("No report could be produced.") and any("empty report" in w for w in out["warnings"])


def test_tool_sets_do_not_overlap_and_format_findings():
    assert not RESEARCH_TOOLS & CALC_TOOLS
    assert "Step 1 (filings): Revenue" in format_findings([{"step_id": 1, "kind": "filings", "goal": "Revenue", "facts": FACTS}])
    assert format_findings([]) == "(no fact notes)"
