"""Phase 4 graph: the whole pipeline end-to-end with scripted models (one per agent). Offline."""

import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from src.agents.graph import DEFAULT_ROLES, build_graph, make_supervisor
from src.observability.guardrails import DISCLAIMER
from tests.conftest import scripted, tool_call

RIL_CITATION = "Reliance Industries Limited Annual Report (FY ended 2025-03-31), PDF p. 100 (printed p. 196 197)"
PLAN_WITH_CALC = ('{"steps": [{"kind": "filings", "ticker": "RELIANCE.NS", "goal": "Consolidated revenue from '
                  'operations for 2024-25 and 2023-24"}], "needs_calculation": true, "calculation": "percent change"}')
PLAN_NO_CALC = '{"steps": [{"kind": "news", "ticker": null, "goal": "Latest Toyota news"}], "needs_calculation": false}'
FACTS = ("- Revenue from operations (2024-25, consolidated): ₹9,80,136 crore [RELIANCE.NS p100]\n"
         "- Revenue from operations (2023-24, consolidated): ₹9,14,472 crore [RELIANCE.NS p100]")


def usage(n):
    return {"input_tokens": n, "output_tokens": 0, "total_tokens": n}


def search_filings(query: str, ticker: str | None = None, k: int = 5) -> str:
    """Search annual reports."""
    text = "CONSOLIDATED FINANCIAL STATEMENTS (₹ in crore)\nRevenue from Operations 25 9,80,136 9,14,472"
    return json.dumps({"ok": True, "results": [{"source_id": "RELIANCE.NS p100", "citation": RIL_CITATION, "text": text}]})


def search_news(query: str, days: int = 7, max_results: int = 5) -> str:
    """Search news."""
    return json.dumps({"ok": True, "results": [{"title": "Toyota news", "source": "reuters.com", "published_date": "2026-10-23"}]})


def percent_change(old_value: str, new_value: str) -> str:
    """Percent change."""
    return json.dumps({"ok": True, "result_percent": 7.18})


RESEARCH = [StructuredTool.from_function(f) for f in (search_filings, search_news)]
ALL_TOOLS = RESEARCH + [StructuredTool.from_function(percent_change)]


def run(graph, question="How much did Reliance's revenue grow from 2023-24 to 2024-25?"):
    return asyncio.run(graph.ainvoke({"question": question}))


GOOD_DRAFT = "Revenue grew 7.18%, from ₹9,14,472 crore to ₹9,80,136 crore [RELIANCE.NS p100]."
WRONG_PERIOD_DRAFT = "Revenue grew 7.18% in 2023-24, from ₹9,14,472 crore to ₹9,80,136 crore [RELIANCE.NS p100]."
PASS = '{"verdict": "pass", "issues": []}'
FAIL_PERIOD = ('{"verdict": "fail", "issues": [{"claim": "grew 7.18% in 2023-24", '
               '"problem": "the growth happened in 2024-25", "fix": "grew 7.18% in 2024-25"}]}')


def calc_pipeline_models(writer_replies=None, critic_replies=None):
    return {
        "planner": scripted(AIMessage(content=PLAN_WITH_CALC, usage_metadata=usage(600))),
        "researcher": scripted(tool_call("search_filings", {"query": "revenue", "ticker": "RELIANCE.NS", "k": 3}, tokens=1500),
                               AIMessage(content=FACTS, usage_metadata=usage(2500))),
        "analyst": scripted(tool_call("percent_change", {"old_value": "9,14,472", "new_value": "9,80,136"}, tokens=900),
                            AIMessage(content="ANALYSIS:\n- Growth: 7.18% (formula: (new - old) / |old| x 100)", usage_metadata=usage(1100))),
        "writer": scripted(*(writer_replies or [AIMessage(content=GOOD_DRAFT, usage_metadata=usage(1300))])),
        "critic": scripted(*(critic_replies or [AIMessage(content=PASS, usage_metadata=usage(1500))])),
    }


def node_order(graph):
    async def order():
        return [node for update in [u async for u in graph.astream({"question": "q"}, stream_mode="updates")] for node in update]
    return asyncio.run(order())


def test_full_pipeline_with_calculation():
    final = run(build_graph(ALL_TOOLS, calc_pipeline_models(), companies="- Reliance (RELIANCE.NS)"))
    assert final["route"] == "analyst"
    assert final["plan"]["steps"][0]["ticker"] == "RELIANCE.NS"
    assert "₹9,80,136 crore" in final["findings"][0]["facts"]
    assert final["analysis"].startswith("ANALYSIS:")
    assert f"({RIL_CITATION})" in final["report"] and final["report"].endswith(DISCLAIMER)
    assert [c["agent"] for c in final["tool_calls"]] == ["researcher", "analyst"]
    assert final["usage"] == {"planner": 600, "researcher": 4000, "analyst": 2000, "writer": 1300, "critic": 1500}
    assert final["critique"]["verdict"] == "pass" and final["critic_rounds"] == 1
    assert final["warnings"] == []


def test_pipeline_runs_agents_in_the_right_order():
    graph = build_graph(ALL_TOOLS, calc_pipeline_models(), companies="")
    assert node_order(graph) == ["planner", "researcher", "supervisor", "analyst", "writer", "critic", "finalize"]


def test_critic_failure_sends_the_draft_back_and_the_revision_fixes_it():
    models = calc_pipeline_models(writer_replies=[AIMessage(content=WRONG_PERIOD_DRAFT), AIMessage(content=GOOD_DRAFT)],
                                  critic_replies=[AIMessage(content=FAIL_PERIOD), AIMessage(content=PASS)])
    graph = build_graph(ALL_TOOLS, models, companies="")
    final = run(graph)
    assert final["revisions"] == 1 and final["critic_rounds"] == 2
    assert final["critique"]["verdict"] == "pass" and "2023-24" not in final["report"]
    assert "Verification note" not in final["report"]
    assert "the growth happened in 2024-25" in models["writer"].seen[1][1].content  # revision got the issue


def test_loop_stops_after_max_revisions_with_a_verification_note():
    models = calc_pipeline_models(writer_replies=[AIMessage(content=WRONG_PERIOD_DRAFT)] * 3,
                                  critic_replies=[AIMessage(content=FAIL_PERIOD)] * 3)
    final = run(build_graph(ALL_TOOLS, models, companies=""))
    assert final["critic_rounds"] == 3 and final["revisions"] == 2
    assert "**Verification note:**" in final["report"] and "the growth happened in 2024-25" in final["report"]
    assert any("remain after 2 revision(s)" in w for w in final["warnings"])


def test_a_writer_that_keeps_failing_cannot_loop_forever():
    from tests.test_nodes import TOO_LARGE, raising

    models = calc_pipeline_models(critic_replies=[AIMessage(content=FAIL_PERIOD)] * 3)
    models["writer"] = raising(AIMessage(content=WRONG_PERIOD_DRAFT), TOO_LARGE, TOO_LARGE)
    final = run(build_graph(ALL_TOOLS, models, companies=""))
    assert final["critic_rounds"] == 3  # capped on the critic's rounds, not the writer's revisions
    assert sum("writer failed" in w for w in final["warnings"]) == 2
    assert "Verification note" in final["report"]


def test_route_after_critic():
    from src.agents.graph import route_after_critic

    assert route_after_critic({"critique": {"verdict": "pass"}, "critic_rounds": 1}) == "finalize"
    assert route_after_critic({"critique": {"verdict": "fail"}, "critic_rounds": 1}) == "writer"
    assert route_after_critic({"critique": {"verdict": "fail"}, "critic_rounds": 2}) == "writer"
    assert route_after_critic({"critique": {"verdict": "fail"}, "critic_rounds": 3}) == "finalize"
    assert route_after_critic({}) == "finalize"


def test_no_calculation_skips_the_analyst():
    models = {
        "planner": scripted(AIMessage(content=PLAN_NO_CALC)),
        "researcher": scripted(tool_call("search_news", {"query": "Toyota"}),
                               AIMessage(content="- Toyota revamps bZ4X (reuters.com, 2026-10-23)")),
        "analyst": scripted(),  # no replies: calling it would raise
        "writer": scripted(AIMessage(content="Toyota revamped its bZ4X (reuters.com, 2026-10-23).")),
        "critic": scripted(AIMessage(content=PASS)),
    }
    final = run(build_graph(ALL_TOOLS, models, companies=""), "What is the latest news about Toyota?")
    assert final["route"] == "writer" and "analysis" not in final
    assert "analyst" not in final["usage"] and final["warnings"] == []
    assert "reuters.com" in final["report"]


def test_calculation_without_calculator_tools_goes_to_writer_with_warning():
    models = calc_pipeline_models()
    models["analyst"] = scripted()  # must not be called
    final = run(build_graph(RESEARCH, models, companies=""))  # no calculator tools available
    assert final["route"] == "writer"
    assert any("calculator tools are unavailable" in w for w in final["warnings"])


def test_supervisor_routing_rules():
    assert make_supervisor(True)({"plan": {"needs_calculation": True}}) == {"route": "analyst"}
    assert make_supervisor(True)({"plan": {"needs_calculation": False}}) == {"route": "writer"}
    assert make_supervisor(True)({}) == {"route": "writer"}
    assert make_supervisor(False)({"plan": {"needs_calculation": True}})["route"] == "writer"


def test_roles_spread_across_two_models():
    assert DEFAULT_ROLES == {"planner": "default", "researcher": "default", "analyst": "reasoning",
                             "writer": "reasoning", "critic": "reasoning"}
