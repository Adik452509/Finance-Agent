"""Phase 4 shared state: plan parsing (defensive) and reducers. Offline."""

import operator

import pytest

from src.agents.state import (
    MAX_STEPS, Plan, PlanError, add_usage, extract_json, fallback_plan, merge_dicts, parse_plan,
)

GOOD = """{"steps": [{"id": 1, "kind": "filings", "ticker": "RELIANCE.NS",
            "goal": "Find consolidated revenue from operations for 2024-25 and 2023-24"}],
           "needs_calculation": true, "calculation": "percent change 2023-24 -> 2024-25"}"""


def test_parses_a_clean_plan():
    plan, notes = parse_plan(GOOD)
    assert isinstance(plan, Plan) and notes == []
    assert plan.steps[0].kind == "filings" and plan.steps[0].ticker == "RELIANCE.NS"
    assert plan.needs_calculation and "percent change" in plan.calculation


def test_handles_code_fences_and_surrounding_prose():
    text = "Here is the plan:\n```json\n" + GOOD + "\n```\nLet me know!"
    plan, _ = parse_plan(text)
    assert plan.steps[0].ticker == "RELIANCE.NS"


def test_kind_synonyms_are_normalised():
    plan, _ = parse_plan('{"steps": [{"kind": "Price", "goal": "Toyota share price", "ticker": "7203.t"},'
                         '{"kind": "annual_report", "goal": "Toyota operating income FY2026"}]}')
    assert [s.kind for s in plan.steps] == ["market", "filings"]
    assert plan.steps[0].ticker == "7203.T"  # upper-cased


def test_calculation_steps_move_to_the_analyst():
    plan, notes = parse_plan('{"steps": [{"kind": "filings", "goal": "Reliance revenue both years"},'
                             '{"kind": "calculation", "goal": "compute growth percent"}]}')
    assert len(plan.steps) == 1
    assert plan.needs_calculation and plan.calculation == "compute growth percent"
    assert any("analyst" in n for n in notes)


def test_unknown_kind_becomes_any_with_a_note():
    plan, notes = parse_plan('{"steps": [{"kind": "telepathy", "goal": "find something useful"}]}')
    assert plan.steps[0].kind == "any" and notes


def test_too_many_steps_are_truncated_and_renumbered():
    steps = ",".join(f'{{"id": {i * 10}, "kind": "news", "goal": "news item {i}"}}' for i in range(8))
    plan, notes = parse_plan('{"steps": [' + steps + ']}')
    assert len(plan.steps) == MAX_STEPS
    assert [s.id for s in plan.steps] == [1, 2, 3, 4, 5]
    assert any("kept the first" in n for n in notes)


def test_invalid_ticker_is_dropped():
    plan, notes = parse_plan('{"steps": [{"kind": "market", "goal": "price please", "ticker": "AAPL; DROP"}]}')
    assert plan.steps[0].ticker is None and notes


@pytest.mark.parametrize("bad", ["", "no json here", "{not json}", "[1, 2]", '{"steps": []}',
                                 '{"steps": [{"kind": "calculation", "goal": "only maths"}]}',
                                 '{"steps": [{"kind": "news", "goal": "x"}]}'])
def test_unusable_plans_raise_plan_error(bad):
    with pytest.raises(PlanError):
        parse_plan(bad)


def test_extract_json_rejects_non_objects():
    with pytest.raises(PlanError):
        extract_json("[1, 2, 3]")


def test_fallback_plan_detects_calculation_questions():
    plan = fallback_plan("By what percentage did Reliance's revenue grow?")
    assert plan.steps[0].kind == "any" and plan.needs_calculation
    assert not fallback_plan("What is the latest news about Toyota?").needs_calculation


def test_langgraph_applies_the_reducers():
    from langgraph.graph import END, START, StateGraph

    from src.agents.state import ResearchState

    def first(state):
        return {"findings": [{"step_id": 1, "kind": "filings", "goal": "g1", "facts": "f1"}],
                "citations": {"A p1": "cite 1"}, "usage": {"researcher": 100}, "warnings": ["w1"]}

    def second(state):
        return {"findings": [{"step_id": 2, "kind": "market", "goal": "g2", "facts": "f2"}],
                "citations": {"B p2": "cite 2"}, "usage": {"researcher": 50, "writer": 10}, "warnings": ["w2"]}

    g = StateGraph(ResearchState)
    g.add_node("first", first)
    g.add_node("second", second)
    g.add_edge(START, "first")
    g.add_edge("first", "second")
    g.add_edge("second", END)
    out = g.compile().invoke({"question": "q"})

    assert [f["step_id"] for f in out["findings"]] == [1, 2]          # appended
    assert out["citations"] == {"A p1": "cite 1", "B p2": "cite 2"}    # merged
    assert out["usage"] == {"researcher": 150, "writer": 10}           # summed per agent
    assert out["warnings"] == ["w1", "w2"]


def test_reducers():
    assert merge_dicts({"A p1": "x"}, {"B p2": "y"}) == {"A p1": "x", "B p2": "y"}
    assert add_usage({"planner": 100}, {"planner": 50, "writer": 10}) == {"planner": 150, "writer": 10}
    assert add_usage(None, {"analyst": 5}) == {"analyst": 5}
    assert operator.add(["w1"], ["w2"]) == ["w1", "w2"]  # findings / tool_calls / warnings append
