"""A small, explicit tool-calling loop shared by the Phase 4 agents.

    result = await run_tool_loop(llm, tools, messages, agent="researcher", max_rounds=3)

The loop:
    1. send the conversation to the LLM (with only THIS agent's tools bound)
    2. text reply             -> done
       tool calls             -> run each tool, append results, go to 1
    3. after max_rounds of tool calls, ask for a final answer with NO tools bound

Why not LangGraph's create_react_agent? It is deprecated in LangGraph 1.0, and we
need things it doesn't give us: per-agent token counts, trimming oversized tool
output (Groq free tier: 8,000 tokens per request), and a live event trace.

A tool error or an unknown tool name never crashes the loop: it becomes a tool
result the LLM can read and react to.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.agents.state import ToolCallRecord
from src.rag.qa import _message_text

DEFAULT_MAX_ROUNDS = 3
MAX_TOOL_CHARS = 10_000  # ~2,500 tokens; a k=3 filings search is ~8,000 chars
STOP_TOOLS = "Stop calling tools now. Answer using the information you already have."


@dataclass
class LoopResult:
    text: str
    calls: list[ToolCallRecord] = field(default_factory=list)
    results: list[tuple[str, str]] = field(default_factory=list)  # (tool name, raw content)
    tokens: int = 0
    rounds: int = 0       # rounds in which tools were called
    hit_cap: bool = False


def parse_tool_result(content: Any) -> tuple[bool | None, str | None]:
    """Our tools return JSON with ok/error; read it without trusting it."""
    try:
        data = json.loads(_message_text(content))
    except (json.JSONDecodeError, TypeError):
        return None, None
    if isinstance(data, dict) and "ok" in data:
        return bool(data["ok"]), (data.get("error") if not data["ok"] else None)
    return None, None


def _tokens(message: AIMessage) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("total_tokens", 0) or 0)


def _error_json(error_type: str, message: str) -> str:
    return json.dumps({"ok": False, "error_type": error_type, "error": message})


async def _run_tool(tool: BaseTool | None, name: str, args: dict, available: list[str]) -> str:
    if tool is None:
        return _error_json("invalid_input", f"Unknown tool {name!r}. Available tools: {', '.join(available)}")
    try:
        content = await tool.ainvoke(args)
    except Exception as e:  # a tool must never take the agent down
        return _error_json("upstream_error", f"{type(e).__name__}: {str(e)[:200]}")
    return content if isinstance(content, str) else _message_text(content)


def _trim(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + f"\n...[truncated {len(content) - limit} characters to fit the token budget]"


async def run_tool_loop(
    llm: BaseChatModel,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    *,
    agent: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_tool_chars: int = MAX_TOOL_CHARS,
    on_event: Callable[[str, dict], None] | None = None,
) -> LoopResult:
    """Run the think -> tool -> think loop; return the final text, calls and token count."""
    emit = on_event or (lambda kind, payload: None)
    by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools) if tools else llm
    msgs = list(messages)
    result = LoopResult(text="")

    while True:
        if result.rounds >= max_rounds:
            # Cap reached: one last call with no tools bound, so it cannot keep looping.
            result.hit_cap = True
            msgs.append(HumanMessage(STOP_TOOLS))
            final = await llm.ainvoke(msgs)
            result.tokens += _tokens(final)
            result.text = _message_text(final.content).strip()
            return result

        reply = await bound.ainvoke(msgs)
        result.tokens += _tokens(reply)
        msgs.append(reply)
        if not getattr(reply, "tool_calls", None):
            result.text = _message_text(reply.content).strip()
            return result

        result.rounds += 1
        for call in reply.tool_calls:
            name, args = call["name"], call.get("args", {}) or {}
            emit("call", {"agent": agent, "name": name, "args": args})
            started = time.monotonic()
            content = await _run_tool(by_name.get(name), name, args, list(by_name))
            ok, error = parse_tool_result(content)
            record: ToolCallRecord = {"agent": agent, "name": name, "args": args, "ok": ok,
                                      "error": error, "seconds": round(time.monotonic() - started, 2)}
            result.calls.append(record)
            result.results.append((name, content))
            msgs.append(ToolMessage(content=_trim(content, max_tool_chars), tool_call_id=call["id"], name=name))
            emit("result", record)
