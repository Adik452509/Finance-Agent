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

import json
import logging
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from src.agents.prompts import SINGLE_AGENT_PROMPT
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
    return create_react_agent(llm, tools, prompt=SINGLE_AGENT_PROMPT, name="single_agent")


def _parse_tool_result(content: Any) -> tuple[bool | None, str | None]:
    """Our tools return JSON with ok/error; read it without trusting it."""
    try:
        data = json.loads(_message_text(content))
    except (json.JSONDecodeError, TypeError):
        return None, None
    if isinstance(data, dict) and "ok" in data:
        return bool(data["ok"]), (data.get("error") if not data["ok"] else None)
    return None, None


def _has_figures(text: str) -> bool:
    return any(not YEAR_RE.match(m) for m in FIGURE_RE.findall(text))


# [7203.T p164] or [AAPL p32, AAPL p39] - but not a markdown link "[text](url)".
BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,120})\](?!\()")
# Anything shaped like a source label, including malformed ones such as "7203.T p-".
SOURCE_ID_RE = re.compile(r"^[A-Z0-9.^-]+ p\S*$|^S\d+$")


def _collect_citations(content: Any, into: dict[str, str]) -> None:
    """Remember source_id -> citation from a search_filings result."""
    try:
        data = json.loads(_message_text(content))
    except (json.JSONDecodeError, TypeError):
        return
    for passage in (data.get("results") or []) if isinstance(data, dict) else []:
        if isinstance(passage, dict) and passage.get("source_id") and passage.get("citation"):
            into[passage["source_id"]] = passage["citation"]


def expand_citations(text: str, citations: dict[str, str]) -> tuple[str, list[str]]:
    """Replace [source_id] labels with full citations; report labels that match nothing.

    Done in code so the reader always sees the real source, whatever the model wrote.
    """
    unresolved: list[str] = []

    def replace(match: re.Match) -> str:
        parts = [p.strip() for p in re.split(r"[,;]", match.group(1))]
        if parts and all(p in citations for p in parts):
            return "(" + "; ".join(citations[p] for p in parts) + ")"
        unresolved.extend(p for p in parts if SOURCE_ID_RE.match(p) and p not in citations)
        return match.group(0)

    return BRACKET_RE.sub(replace, text), unresolved


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
                            call.ok, call.error = _parse_tool_result(msg.content)
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
    if unresolved:
        run.warnings.append(f"Citations that match no retrieved passage: {', '.join(sorted(set(unresolved)))}")
    if not run.tool_calls and _has_figures(final_text):
        run.warnings.append("Answer contains figures but no tool was called: it may come from model memory.")
    for call in run.tool_calls:
        if call.ok is False:
            run.warnings.append(f"Tool {call.name} failed: {call.error}")
    return run
