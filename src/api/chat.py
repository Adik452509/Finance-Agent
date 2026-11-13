"""Run one question through the research team, as a stream of small progress events.

Events (sent to the browser as Server-Sent Events by app.py):
    {"type": "progress", "agent": "planner", "text": "Planned 2 research steps", ...}
    {"type": "tool", "agent": "researcher", "name": "search_filings", "args": {...}}
    {"type": "tool_result", "name": "search_filings", "ok": true, "seconds": 0.8}
    {"type": "answer", "message": {...}}   (from app.py, once the answer is saved)
    {"type": "error", "text": "..."}
"""

import asyncio
from collections.abc import Callable

from src.agents.nodes import describe_error
from src.api.team import Team, current_sink

Emit = Callable[[dict], None]

AGENT_LABELS = {"planner": "Planner", "researcher": "Researcher", "supervisor": "Supervisor", "analyst": "Analyst",
                "writer": "Writer", "critic": "Critic", "finalize": "Report"}


def describe_update(node: str, update: dict) -> dict | None:
    """One line for the live progress box, plus details the UI may expand."""
    text = None
    if node == "planner":
        steps = (update.get("plan") or {}).get("steps", [])
        text = f"Planned {len(steps)} research step{'s' * (len(steps) != 1)}"
        detail = [f"{s['kind']}: {s['goal']}" + (f" ({s['ticker']})" if s.get("ticker") else "") for s in steps]
    elif node == "researcher":
        findings = update.get("findings", [])
        text, detail = f"Collected fact notes for {len(findings)} step(s)", [f["facts"] for f in findings]
    elif node == "supervisor":
        text = "Calculation needed: handing over to the analyst" if update.get("route") == "analyst" \
            else "No calculation needed: handing over to the writer"
        detail = []
    elif node == "analyst":
        text, detail = "Calculated the figures", [update.get("analysis") or ""]
    elif node == "writer":
        text = f"Revised the draft (revision {update['revisions']})" if update.get("revisions") else "Wrote a draft"
        detail = []
    elif node == "critic":
        critique = update.get("critique") or {}
        if critique.get("verdict") == "pass":
            text, detail = "Checked the draft against the evidence: passed", []
        else:
            text = f"Found {len(critique.get('issues', []))} issue(s) in the draft"
            detail = [f"{i.get('claim', '')}: {i.get('problem', '')}" for i in critique.get("issues", [])]
    elif node == "finalize":
        text, detail = "Report ready", []
    if text is None:
        return None
    return {"type": "progress", "agent": node, "label": AGENT_LABELS.get(node, node), "text": text,
            "detail": detail, "warnings": update.get("warnings", [])}


def answer_record(final: dict, extra_usage: dict | None = None) -> dict:
    """What gets saved with the assistant's message: enough to redraw every tab later."""
    usage = {**(final.get("usage") or {}), **(extra_usage or {})}
    return {
        "content": final.get("report") or "",
        "citations": final.get("citations") or {},
        "usage": usage,
        "tokens": sum(usage.values()),
        "trace": {
            "plan": final.get("plan"), "route": final.get("route"), "findings": final.get("findings", []),
            "analysis": final.get("analysis"), "tool_calls": final.get("tool_calls", []),
            "warnings": final.get("warnings", []), "critique": final.get("critique"),
            "critic_rounds": final.get("critic_rounds", 0),
        },
    }


async def run_question(team: Team, question: str, emit: Emit, timeout: float) -> dict:
    """Run the graph, emitting progress; returns the final state. Raises on failure or timeout."""
    def sink(kind: str, payload: dict) -> None:  # live tool calls from inside the agents
        if kind == "call":
            emit({"type": "tool", "agent": payload.get("agent"), "name": payload.get("name"),
                  "args": payload.get("args", {})})
        elif kind == "result":
            emit({"type": "tool_result", "agent": payload.get("agent"), "name": payload.get("name"),
                  "ok": payload.get("ok"), "error": payload.get("error"), "seconds": payload.get("seconds")})

    async def go() -> dict:
        current_sink.set(sink)  # this task's own copy: other requests never see these events
        final: dict = {}
        async for mode, chunk in team.graph.astream({"question": question}, stream_mode=["updates", "values"]):
            if mode == "updates":
                for node, update in chunk.items():
                    event = describe_update(node, update or {})
                    if event:
                        emit(event)
            else:
                final = chunk
        return final

    return await asyncio.wait_for(go(), timeout)


class NoReport(RuntimeError):
    """The team ran but wrote nothing: every agent that mattered failed (and reported it as a warning).

    Seen live (2026-11-13, an invalid API key): each agent catches its own error so one bad step
    cannot crash a report - which also meant a run where EVERYTHING failed ended as a normal
    "No report could be produced." answer, saved as fine and counted against the user's daily limit.
    """

    def __init__(self, final: dict) -> None:
        warnings = [str(w) for w in final.get("warnings") or []]
        failed = [w for w in warnings if "failed" in w]
        # The most telling cause first: a spent daily budget explains every other failure.
        reason = next((w for w in warnings if "daily token budget" in w), failed[0] if failed else "the team produced no report")
        super().__init__(reason)
        self.final = final


def friendly_error(error: Exception) -> str:
    if isinstance(error, asyncio.TimeoutError):
        return "The research took too long and was stopped. Please try again or ask a narrower question."
    text = describe_error(error)
    if "daily token budget" in text:
        return "Today's AI budget is used up. Please try again tomorrow."
    if "AuthenticationError" in text or "Invalid API Key" in text or "Error code: 401" in text:
        # A configuration problem on our side: the raw provider message is of no use to a user.
        return "The AI service is not available right now (a configuration problem on our side). Please try again later."
    if isinstance(error, NoReport):
        return "Sorry, the research could not be completed, so there is no answer to show. Please try again."
    return f"Sorry, the research failed ({text}). Please try again."
