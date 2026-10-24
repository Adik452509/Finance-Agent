"""Phase 2 single agent: one ReAct loop that picks among the MCP tools by itself.

    async with MCPToolbox() as toolbox:
        agent = build_single_agent(toolbox.tools)
        run = await run_agent(agent, "What is Toyota's share price?")

ReAct = the model alternates "think -> call a tool -> read the result" until it can
answer. Phase 4 replaces this with a team of specialised agents; this is the baseline.

Guardrails enforced in code (not just asked for in the prompt):
  - a hard cap on loop steps, so a confused model can't call tools forever;
  - a warning when the answer contains figures but no tool was called (likely memory);
  - tool errors are recorded and surfaced;
  - the disclaimer is appended by code.
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from src.agents.citations import collect_citations as _collect_citations
from src.agents.citations import expand_citations  # noqa: F401  (re-exported for tests/scripts)
from src.agents.prompts import SINGLE_AGENT_PROMPT
from src.agents.toolloop import parse_tool_result
from src.observability.guardrails import with_disclaimer
from src.rag.qa import FIGURE_RE, YEAR_RE, _message_text, normalize_answer

with warnings.catch_warnings():
    # create_react_agent moved to langchain 1.x in LangGraph 1.0; we're on langchain-core 0.3.
    # Phase 4 replaces it with our own graph, so the deprecation is accepted here.
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

# LangGraph counts graph steps: each tool round is 2 (model -> tools), plus the final answer.
RECURSION_LIMIT = 15  # => at most ~7 tool rounds per question
STOPPED_MESSAGE = "I stopped before finishing: the research took too many steps. Try a narrower question."


@dataclass
class ToolCall:
    name: str
    args: dict
    ok: bool | None = None
    error: str | None = None
    seconds: float | None = None


@dataclass
class AgentRun:
    question: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stopped: bool = False

    @property
    def tools_used(self) -> list[str]:
        return [c.name for c in self.tool_calls]


def build_single_agent(tools: list[BaseTool], llm: BaseChatModel | None = None):
    if llm is None:
        from config.llm import get_chat_model

        llm = get_chat_model("default")
    with warnings.catch_warnings():
        # The deprecation fires when the agent is built (not at import); accepted until Phase 4.
        warnings.simplefilter("ignore")
        return create_react_agent(llm, tools, prompt=SINGLE_AGENT_PROMPT, name="single_agent")


def _has_figures(text: str) -> bool:
    return any(not YEAR_RE.match(m) for m in FIGURE_RE.findall(text))


async def run_agent(agent, question: str, on_event: Callable[[str, Any], None] | None = None) -> AgentRun:
    """Run one question; report each tool call/result through `on_event` as it happens."""
    emit = on_event or (lambda kind, payload: None)
    run = AgentRun(question=question, answer="")
    pending: dict[str, tuple[ToolCall, float]] = {}
    citations: dict[str, str] = {}
    final_text = ""

    try:
        async for update in agent.astream(
            {"messages": [HumanMessage(question)]},
            {"recursion_limit": RECURSION_LIMIT},
            stream_mode="updates",
        ):
            for node_output in update.values():
                for msg in (node_output or {}).get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            call = ToolCall(tc["name"], tc.get("args", {}))
                            run.tool_calls.append(call)
                            pending[tc["id"]] = (call, time.monotonic())
                            emit("call", call)
                    elif isinstance(msg, ToolMessage):
                        call, started = pending.pop(msg.tool_call_id, (None, None))
                        if call is not None:
                            call.ok, call.error = parse_tool_result(msg.content)
                            call.seconds = time.monotonic() - started
                            if call.name == "search_filings" and call.ok:
                                _collect_citations(msg.content, citations)
                            emit("result", call)
                    elif isinstance(msg, AIMessage):
                        final_text = _message_text(msg.content)
    except GraphRecursionError:
        run.stopped = True
        final_text = STOPPED_MESSAGE
        run.warnings.append(f"Stopped at the {RECURSION_LIMIT}-step limit.")

    answer, unresolved = expand_citations(normalize_answer(final_text), citations)
    run.answer = with_disclaimer(answer or "No answer was produced.")
    if not answer and not run.stopped:
        run.warnings.append("The model returned an empty answer (often a sign the request was near the "
                            "token limit). Retry, or use the reasoning model.")
    if unresolved:
        run.warnings.append(f"Citations that match no retrieved passage: {', '.join(sorted(set(unresolved)))}")
    if not run.tool_calls and _has_figures(final_text):
        run.warnings.append("Answer contains figures but no tool was called: it may come from model memory.")
    for call in run.tool_calls:
        if call.ok is False:
            run.warnings.append(f"Tool {call.name} failed: {call.error}")
    return run
