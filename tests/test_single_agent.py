"""Single agent: tool loop, citation expansion and code-enforced guardrails. Offline.

A scripted fake model plays back tool calls, so the guardrails can be tested exactly.
"""

import asyncio
import itertools
import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from src.agents.single_agent import RECURSION_LIMIT, build_single_agent, expand_citations, run_agent
from src.observability.guardrails import DISCLAIMER

TOYOTA_CITATION = "Toyota Motor Corporation 20-F (FY ended 2026-03-31), PDF p. 164 (printed p. F-6)"


class ScriptedModel(GenericFakeChatModel):
    """Fake chat model that accepts tools (the agent binds them) and replays messages."""

    def bind_tools(self, tools, **kwargs):
        return self


def call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def fake_tools(quote_ok: bool = True) -> list[StructuredTool]:
    def get_quote(ticker: str) -> str:
        """Latest price for a ticker."""
        if not quote_ok:
            return json.dumps({"ok": False, "error_type": "not_found", "error": f"No market data for '{ticker}'."})
        return json.dumps({"ok": True, "ticker": ticker, "price": 2856.5, "currency": "JPY", "as_of": "2026-10-04"})

    def search_filings(query: str, ticker: str | None = None, k: int = 5) -> str:
        """Search annual reports."""
        return json.dumps({"ok": True, "results": [
            {"source_id": "7203.T p164", "citation": TOYOTA_CITATION, "text": "Total sales revenues ... 50,684,952"},
        ]})

    return [StructuredTool.from_function(get_quote), StructuredTool.from_function(search_filings)]


def run(messages, question="q", tools=None):
    agent = build_single_agent(tools or fake_tools(), ScriptedModel(messages=iter(messages)))
    return asyncio.run(run_agent(agent, question))


def test_agent_calls_tool_then_answers_with_disclaimer():
    result = run([call("get_quote", {"ticker": "7203.T"}),
                  AIMessage(content="Toyota trades at ¥2,856.5 (Yahoo Finance, as of 2026-10-04).")])
    assert result.tools_used == ["get_quote"]
    assert result.tool_calls[0].ok is True
    assert result.answer.endswith(DISCLAIMER)
    assert result.warnings == []


def test_filings_label_is_expanded_into_full_citation():
    result = run([call("search_filings", {"query": "revenue", "ticker": "7203.T"}),
                  AIMessage(content="Total sales revenues were ¥50,684,952 million [7203.T p164].")])
    assert f"({TOYOTA_CITATION})" in result.answer
    assert "[7203.T p164]" not in result.answer
    assert result.warnings == []


def test_citation_to_passage_never_retrieved_is_flagged():
    result = run([call("search_filings", {"query": "revenue"}),
                  AIMessage(content="Revenue was ¥50,684,952 million [7203.T p999].")])
    assert any("7203.T p999" in w for w in result.warnings)


def test_figures_without_any_tool_call_are_flagged():
    result = run([AIMessage(content="Apple's net sales were $416,161 million.")])
    assert result.tool_calls == []
    assert any("no tool was called" in w for w in result.warnings)


def test_tool_error_is_recorded_and_surfaced():
    result = run([call("get_quote", {"ticker": "TOYOTA"}), AIMessage(content="I couldn't find a quote.")],
                 tools=fake_tools(quote_ok=False))
    assert result.tool_calls[0].ok is False
    assert any("get_quote failed" in w for w in result.warnings)


def test_endless_tool_loop_is_stopped_by_step_limit():
    # Fresh message each time: LangGraph de-duplicates messages by id, so replaying
    # one object would replace history instead of adding steps (unlike a real model).
    forever = (call("get_quote", {"ticker": "7203.T"}, call_id=f"c{i}") for i in itertools.count())
    result = run(forever)
    assert result.stopped
    assert any(str(RECURSION_LIMIT) in w for w in result.warnings)
    assert 1 <= len(result.tool_calls) <= RECURSION_LIMIT


def test_expand_citations_handles_lists_and_leaves_markdown_links_alone():
    cites = {"AAPL p32": "Apple 10-K p. 32", "AAPL p39": "Apple 10-K p. 39"}
    text, unresolved = expand_citations("Sales [AAPL p32; AAPL p39]. See [docs](https://x.y).", cites)
    assert text == "Sales (Apple 10-K p. 32; Apple 10-K p. 39). See [docs](https://x.y)."
    assert unresolved == []


def test_malformed_source_labels_are_flagged():
    # Seen live: the model wrote "[7203.T p‑]" after a market-data figure.
    _, unresolved = expand_citations("Price ¥2,856.5 [7203.T p‑] and [S3].", {"AAPL p32": "x"})
    assert unresolved == ["7203.T p‑", "S3"]
