"""The Phase 4 multi-agent graph.

    START -> planner -> researcher -> supervisor --needs calculation--> analyst -> writer -> END
                                                 +--otherwise--------------------> writer -> END

The supervisor is deliberately plain code, not an LLM: routing on the plan's
needs_calculation flag is predictable, free (no tokens) and easy to debug.
An LLM supervisor would cost tokens on every question and could route wrongly.

Models are split by role so the work spreads across Groq's per-model daily budgets:
the cheap model plans and researches; the stronger model calculates and writes.

    async with MCPToolbox() as toolbox:
        graph = build_graph(toolbox.tools)
        final_state = await graph.ainvoke({"question": "..."})
        print(final_state["report"])
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from src.agents.nodes import CALC_TOOLS, Events, make_analyst, make_planner, make_researcher, make_writer, only
from src.agents.state import ResearchState

# agent -> config.llm role ("default" = gpt-oss-20b, "reasoning" = gpt-oss-120b on Groq)
DEFAULT_ROLES = {"planner": "default", "researcher": "default", "analyst": "reasoning", "writer": "reasoning"}
AGENTS = ("planner", "researcher", "analyst", "writer")


def make_supervisor(has_calculator: bool):
    """Decide who runs after the researcher. Plain code: no LLM, no tokens."""

    def supervisor(state: ResearchState) -> dict:
        if not (state.get("plan") or {}).get("needs_calculation"):
            return {"route": "writer"}
        if has_calculator:
            return {"route": "analyst"}
        return {"route": "writer",
                "warnings": ["supervisor: a calculation was needed but the calculator tools are unavailable"]}

    return supervisor


def default_llms(missing: set[str] | None = None) -> dict[str, BaseChatModel]:
    from config.llm import get_chat_model

    return {agent: get_chat_model(DEFAULT_ROLES[agent]) for agent in (missing or AGENTS)}


def build_graph(tools: list[BaseTool], llms: dict[str, BaseChatModel] | None = None,
                companies: str | None = None, on_event: Events = None):
    """Wire the four agents and the supervisor into a compiled LangGraph."""
    llms = dict(llms or {})
    missing = set(AGENTS) - set(llms)
    if missing:
        llms.update(default_llms(missing))

    g = StateGraph(ResearchState)
    g.add_node("planner", make_planner(llms["planner"], companies))
    g.add_node("researcher", make_researcher(llms["researcher"], tools, on_event=on_event))
    g.add_node("supervisor", make_supervisor(has_calculator=bool(only(tools, CALC_TOOLS))))
    g.add_node("analyst", make_analyst(llms["analyst"], tools, on_event=on_event))
    g.add_node("writer", make_writer(llms["writer"]))

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "supervisor")
    g.add_conditional_edges("supervisor", lambda state: state["route"], {"analyst": "analyst", "writer": "writer"})
    g.add_edge("analyst", "writer")
    g.add_edge("writer", END)
    return g.compile()
