"""The multi-agent graph (Phase 4 team + Phase 5 critic loop).

    START -> planner -> researcher -> supervisor --needs calculation--> analyst --+
                                                 +--otherwise------------------> writer -> critic
                                                                                   ^          |
                                                     fail, rounds left: revise  +--+          |
                                                                                              v
                                                         pass, or no rounds left -> finalize -> END

The supervisor and both routing decisions are plain code, not LLMs: predictable,
free (no tokens) and easy to debug. The revision cap is counted on the CRITIC's
rounds, so the loop always ends - even if the writer fails while revising.

finalize is code only: it expands citations and adds the disclaimer (and a
verification note if the critic was still unhappy), so the writer's draft keeps
short [TICKER pN] labels that the critic can check.

Models are split by role so the work spreads across Groq's per-model daily budgets:
the cheap model plans and researches; the stronger model calculates, writes and checks.

    async with MCPToolbox() as toolbox:
        graph = build_graph(toolbox.tools)
        final_state = await graph.ainvoke({"question": "..."})
        print(final_state["report"])
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from src.agents.critic import make_critic
from src.agents.nodes import (
    CALC_TOOLS, Events, finalize, make_analyst, make_planner, make_researcher, make_writer, only, unavailable_kinds,
)
from src.agents.state import ResearchState

# agent -> config.llm role ("default" = gpt-oss-20b, "reasoning" = gpt-oss-120b on Groq)
# The verifier re-reads the cited filing pages to check the researcher's figures (column, year, segment).
DEFAULT_ROLES = {"planner": "default", "researcher": "default", "verifier": "reasoning", "analyst": "reasoning",
                 "writer": "reasoning", "critic": "reasoning"}
AGENTS = ("planner", "researcher", "analyst", "writer", "critic")  # required; the verifier is optional
MAX_REVISIONS = 2  # the critic checks at most MAX_REVISIONS + 1 drafts


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


def route_after_critic(state: ResearchState, max_revisions: int = MAX_REVISIONS) -> str:
    """Send a failed draft back for revision while rounds remain; otherwise finish."""
    if (state.get("critique") or {}).get("verdict") != "fail":
        return "finalize"
    if state.get("critic_rounds", 0) > max_revisions:
        return "finalize"  # out of rounds: finalize adds a verification note
    return "writer"


def default_llms(missing: set[str] | None = None) -> dict[str, BaseChatModel]:
    from config.llm import get_chat_model

    return {agent: get_chat_model(DEFAULT_ROLES[agent]) for agent in (missing or DEFAULT_ROLES)}


def build_graph(tools: list[BaseTool], llms: dict[str, BaseChatModel] | None = None,
                companies: str | None = None, on_event: Events = None, max_revisions: int = MAX_REVISIONS):
    """Wire the agents, the supervisor and the critic loop into a compiled LangGraph.

    llms=None gives every role (verifier included) its default model. A partial dict gets the
    missing required agents filled in; the source check then runs only if it has a "verifier".
    """
    missing = set(DEFAULT_ROLES) if llms is None else set(AGENTS) - set(llms)
    llms = dict(llms or {})
    if missing:
        llms.update(default_llms(missing))

    g = StateGraph(ResearchState)
    g.add_node("planner", make_planner(llms["planner"], companies, unavailable=unavailable_kinds(tools)))
    g.add_node("researcher", make_researcher(llms["researcher"], tools, on_event=on_event,
                                             verifier=llms.get("verifier")))
    g.add_node("supervisor", make_supervisor(has_calculator=bool(only(tools, CALC_TOOLS))))
    g.add_node("analyst", make_analyst(llms["analyst"], tools, on_event=on_event))
    g.add_node("writer", make_writer(llms["writer"]))
    g.add_node("critic", make_critic(llms["critic"]))
    g.add_node("finalize", finalize)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "supervisor")
    g.add_conditional_edges("supervisor", lambda state: state["route"], {"analyst": "analyst", "writer": "writer"})
    g.add_edge("analyst", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", lambda state: route_after_critic(state, max_revisions),
                            {"writer": "writer", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()
