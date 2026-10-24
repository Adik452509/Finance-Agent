"""Phase 4 tool loop: rounds, cap, errors, trimming, tokens, events. Offline."""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from src.agents.toolloop import STOP_TOOLS, parse_tool_result, run_tool_loop
from tests.conftest import scripted, tool_call


def percent_change(old_value: float, new_value: float) -> str:
    """Percent change."""
    return json.dumps({"ok": True, "result_percent": round((new_value - old_value) / old_value * 100, 2)})


def broken_tool(x: int) -> str:
    """Always fails."""
    raise RuntimeError("network down")


def big_tool(x: int) -> str:
    """Returns a huge payload."""
    return json.dumps({"ok": True, "text": "y" * 50_000})


TOOLS = [StructuredTool.from_function(f) for f in (percent_change, broken_tool, big_tool)]


def loop(model, max_rounds=3, events=None, **kwargs):
    on_event = (lambda kind, payload: events.append((kind, payload["name"]))) if events is not None else None
    return asyncio.run(run_tool_loop(model, TOOLS, [HumanMessage("question")], agent="analyst",
                                     max_rounds=max_rounds, on_event=on_event, **kwargs))


def test_plain_answer_needs_no_tools():
    result = loop(scripted(AIMessage(content="Just text.", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})))
    assert result.text == "Just text." and result.calls == [] and result.rounds == 0
    assert result.tokens == 10


def test_tool_result_is_fed_back_to_the_model():
    model = scripted(tool_call("percent_change", {"old_value": 914472, "new_value": 980136}),
                     AIMessage(content="Growth was 7.18%."))
    result = loop(model)
    assert result.text == "Growth was 7.18%."
    assert result.calls[0]["agent"] == "analyst" and result.calls[0]["ok"] is True
    second_call = model.seen[1]
    assert isinstance(second_call[-1], ToolMessage) and "7.18" in second_call[-1].content


def test_unknown_tool_becomes_an_error_result_not_a_crash():
    result = loop(scripted(tool_call("telepathy", {}), AIMessage(content="Could not do it.")))
    assert result.calls[0]["ok"] is False
    assert "Available tools" in result.calls[0]["error"] and "percent_change" in result.calls[0]["error"]
    assert result.text == "Could not do it."


def test_tool_exception_becomes_an_error_result_not_a_crash():
    result = loop(scripted(tool_call("broken_tool", {"x": 1}), AIMessage(content="Source unavailable.")))
    assert result.calls[0]["ok"] is False and "network down" in result.calls[0]["error"]
    assert result.text == "Source unavailable."


def test_round_cap_forces_a_final_answer_without_tools():
    calls = [tool_call("percent_change", {"old_value": 1, "new_value": 2}, call_id=f"c{i}") for i in range(2)]
    model = scripted(*calls, AIMessage(content="Final answer after the cap."))
    result = loop(model, max_rounds=2)
    assert result.hit_cap and result.rounds == 2 and len(result.calls) == 2
    assert result.text == "Final answer after the cap."
    assert model.seen[-1][-1].content == STOP_TOOLS


def test_oversized_tool_output_is_trimmed_for_the_model():
    model = scripted(tool_call("big_tool", {"x": 1}), AIMessage(content="done"))
    result = loop(model, max_tool_chars=1_000)
    sent = model.seen[1][-1].content
    assert len(sent) < 1_200 and "truncated" in sent
    assert len(result.results[0][1]) > 40_000  # the full content is still kept for the agent's own use


def test_tokens_are_summed_across_rounds():
    model = scripted(tool_call("percent_change", {"old_value": 1, "new_value": 2}, tokens=1200),
                     AIMessage(content="ok", usage_metadata={"input_tokens": 900, "output_tokens": 100, "total_tokens": 1000}))
    assert loop(model).tokens == 2200


def test_events_report_each_call_and_result():
    events = []
    loop(scripted(tool_call("percent_change", {"old_value": 1, "new_value": 2}), AIMessage(content="ok")), events=events)
    assert events == [("call", "percent_change"), ("result", "percent_change")]


def test_parse_tool_result():
    assert parse_tool_result('{"ok": true}') == (True, None)
    assert parse_tool_result('{"ok": false, "error": "bad"}') == (False, "bad")
    assert parse_tool_result("not json") == (None, None)
