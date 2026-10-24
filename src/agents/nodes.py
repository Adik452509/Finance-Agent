"""The four Phase 4 agents, as LangGraph nodes.

Each make_*() factory takes the agent's model (and only the tools it is allowed
to use) and returns an async function: shared state in -> state update out.

    planner    no tools          question     -> plan (JSON, validated; retry once, then fallback)
    researcher 6 research tools  each step    -> short fact notes + citations (raw pages stay here)
    analyst    6 calculator tools fact notes  -> analysis (every number from a calculator tool)
    writer     no tools          notes+analysis -> report (citations expanded in code, disclaimer)

Keeping each request small is deliberate: Groq's free tier allows 8,000 tokens
per request, and a single agent with all 12 tools went over it in Phase 3.
"""

import json
import re
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from src.agents.citations import collect_citations, expand_citations
from src.agents.prompts import ANALYST_PROMPT, PLANNER_PROMPT, RESEARCHER_PROMPT, WRITER_PROMPT
from src.agents.state import MAX_STEPS, Plan, PlanError, ResearchState, fallback_plan, parse_plan
from src.agents.toolloop import count_tokens, run_tool_loop
from src.observability.guardrails import with_disclaimer
from src.rag.qa import _message_text, normalize_answer

RESEARCH_TOOLS = {"get_quote", "get_quarterly_financials", "list_supported_markets",
                  "search_filings", "list_filings", "search_news"}
CALC_TOOLS = {"percent_change", "cagr", "ratio", "share_of_total", "convert_units", "series_change"}

Events = Callable[[str, dict], None] | None


def companies_context() -> str:
    """One line per indexed annual report, so the planner never has to guess tickers."""
    from src.ingestion.loader import load_registry

    return "\n".join(
        f"- {d.company} ({d.ticker}, {d.market}): {d.doc_type}, fiscal year ended {d.fiscal_year_end}, "
        f"figures in {d.reporting_unit}"
        for d in load_registry()
    )


SOURCE_TAG_RE = re.compile(r"\[([A-Z0-9.^-]+ p\d+)\]")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def collect_passages(content: str, into: dict[str, str]) -> None:
    """Remember source_id -> passage text from a search_filings result (for fact checking)."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return
    for p in (data.get("results") or []) if isinstance(data, dict) else []:
        if isinstance(p, dict) and p.get("source_id"):
            into[p["source_id"]] = into.get(p["source_id"], "") + "\n" + str(p.get("text", ""))


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def check_facts_against_sources(facts: str, passages: dict[str, str]) -> list[str]:
    """Cheap grounding check on the researcher's notes (an early, code-only Critic).

    For every fact line that cites a filing passage:
      - each figure in the line (3+ digits, not a year) must appear in that passage;
      - a line saying "consolidated" must not cite a passage from the standalone statements.
    Seen live: a "consolidated" revenue figure that was really the standalone "Value of Sales".
    """
    warnings = []
    for line in facts.splitlines():
        tags = SOURCE_TAG_RE.findall(line)
        if not tags:
            continue
        text = "\n".join(passages.get(t, "") for t in tags)
        if not text.strip():
            continue  # cited a passage we never saw; the writer's citation check reports that
        source_numbers = {_digits(n) for n in NUMBER_RE.findall(text)}
        claim = SOURCE_TAG_RE.sub("", line)
        for figure in NUMBER_RE.findall(claim):
            d = _digits(figure)
            if len(d) >= 3 and not re.fullmatch(r"(19|20)\d{2}", d) and d not in source_numbers:
                warnings.append(f"researcher: {figure} is not in its cited source {', '.join(tags)}")
        upper = text.upper()
        if ("consolidated" in claim.lower() and "STANDALONE FINANCIAL STATEMENTS" in upper
                and "CONSOLIDATED FINANCIAL STATEMENTS" not in upper):
            warnings.append(f"researcher: a 'consolidated' figure cites a STANDALONE statement page ({', '.join(tags)})")
    return warnings


def format_findings(findings: list[dict]) -> str:
    if not findings:
        return "(no fact notes)"
    return "\n\n".join(f"Step {f['step_id']} ({f['kind']}): {f['goal']}\n{f['facts']}" for f in findings)


def only(tools: list[BaseTool], allowed: set[str]) -> list[BaseTool]:
    return [t for t in tools if t.name in allowed]


# --- planner -------------------------------------------------------------------------------


def make_planner(llm: BaseChatModel, companies: str | None = None):
    system = (PLANNER_PROMPT.replace("<<COMPANIES>>", companies if companies is not None else companies_context())
              .replace("<<MAX_STEPS>>", str(MAX_STEPS)))

    async def planner(state: ResearchState) -> dict:
        question = state["question"]
        messages = [SystemMessage(system), HumanMessage(question)]
        tokens, warnings = 0, []
        for attempt in (1, 2):
            reply = await llm.ainvoke(messages)
            tokens += count_tokens(reply)
            try:
                plan, notes = parse_plan(_message_text(reply.content))
                warnings += [f"planner: {n}" for n in notes]
                break
            except PlanError as e:
                if attempt == 1:  # one retry, telling the model exactly what was wrong
                    messages += [reply, HumanMessage(f"That plan was not usable ({e}). Reply with ONLY the JSON object.")]
                else:
                    plan = fallback_plan(question)
                    warnings.append(f"planner failed twice ({e}); using a one-step fallback plan")
        return {"plan": plan.model_dump(), "warnings": warnings, "usage": {"planner": tokens}}

    return planner


# --- researcher ----------------------------------------------------------------------------


def _step_brief(question: str, step: dict) -> str:
    ticker = f" Ticker: {step['ticker']}." if step.get("ticker") else ""
    return (f"Overall question (for context): {question}\n\n"
            f"Your step ({step['kind']}):{ticker} {step['goal']}")


def make_researcher(llm: BaseChatModel, tools: list[BaseTool], max_rounds: int = 3, on_event: Events = None):
    research_tools = only(tools, RESEARCH_TOOLS)

    async def researcher(state: ResearchState) -> dict:
        plan = Plan.model_validate(state["plan"])
        findings, calls, citations, warnings, tokens = [], [], {}, [], 0
        for step in plan.model_dump()["steps"]:  # sequential: parallel calls would burst the token limit
            messages = [SystemMessage(RESEARCHER_PROMPT), HumanMessage(_step_brief(state["question"], step))]
            result = await run_tool_loop(llm, research_tools, messages, agent="researcher",
                                         max_rounds=max_rounds, on_event=on_event)
            tokens += result.tokens
            calls += result.calls
            passages: dict[str, str] = {}
            for name, content in result.results:
                if name == "search_filings":
                    collect_citations(content, citations)
                    collect_passages(content, passages)
            warnings += check_facts_against_sources(result.text, passages)
            if not result.calls:
                warnings.append(f"researcher step {step['id']}: no tool was used, so its notes are unverified")
            if result.hit_cap:
                warnings.append(f"researcher step {step['id']}: stopped at the {max_rounds}-round tool limit")
            findings.append({"step_id": step["id"], "kind": step["kind"], "goal": step["goal"],
                             "facts": result.text or "- NOT FOUND: the researcher returned no notes"})
        return {"findings": findings, "citations": citations, "tool_calls": calls,
                "warnings": warnings, "usage": {"researcher": tokens}}

    return researcher


# --- analyst -------------------------------------------------------------------------------


def make_analyst(llm: BaseChatModel, tools: list[BaseTool], max_rounds: int = 4, on_event: Events = None):
    calc_tools = only(tools, CALC_TOOLS)

    async def analyst(state: ResearchState) -> dict:
        plan = state.get("plan", {})
        brief = (f"Question: {state['question']}\n\nCalculation to do: {plan.get('calculation') or 'whatever the question needs'}"
                 f"\n\nFact notes:\n{format_findings(state.get('findings', []))}")
        result = await run_tool_loop(llm, calc_tools, [SystemMessage(ANALYST_PROMPT), HumanMessage(brief)],
                                     agent="analyst", max_rounds=max_rounds, on_event=on_event)
        warnings = []
        if not result.calls:
            warnings.append("analyst: no calculator tool was used, so any computed numbers are unverified")
        warnings += [f"analyst: {c['name']} failed: {c['error']}" for c in result.calls if c["ok"] is False]
        warnings += manual_conversion_warnings(result.calls)

        analysis = result.text
        if not analysis.strip() and result.calls:
            # Seen live: the model called the calculator, then returned no text. The tool
            # results are the trustworthy part anyway, so assemble the analysis from them.
            analysis = analysis_from_tool_results(result.calls, result.results)
            warnings.append("analyst returned no text; analysis assembled from the calculator results")
        return {"analysis": analysis, "tool_calls": result.calls,
                "warnings": warnings, "usage": {"analyst": result.tokens}}

    return analyst


def manual_conversion_warnings(calls: list[dict]) -> list[str]:
    """Flag ratio() calls dividing by a power of ten: almost always a hand-made unit conversion.

    Seen live: "convert crore to billion" done as ratio(980136, 1000) -> 980.136,
    but 1 billion = 100 crore, so the right answer is 9,801.36 (10x off).
    """
    warnings = []
    for call in calls:
        if call["name"] != "ratio":
            continue
        raw = str(call["args"].get("denominator", "")).replace(",", "").strip()
        try:
            denominator = float(raw)
        except ValueError:
            continue
        if denominator >= 10 and float(denominator).is_integer() and str(int(denominator)).strip("0") == "1":
            warnings.append(f"analyst: ratio(..., {raw}) looks like a manual unit conversion; "
                            "convert_units should be used (e.g. 1 billion = 100 crore)")
    return warnings


def analysis_from_tool_results(calls: list[dict], results: list[tuple[str, str]]) -> str:
    """Deterministic ANALYSIS text built straight from successful calculator outputs."""
    lines = ["ANALYSIS (assembled from calculator results):"]
    for call, (_, content) in zip(calls, results):
        if call["ok"] is not True:
            continue
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue
        if "result_percent" in data:
            value = f"{data['result_percent']}%"
        elif "result" in data:
            value = str(data["result"])
        else:
            continue
        args = ", ".join(f"{k}={v}" for k, v in call["args"].items())
        lines.append(f"- {call['name']}({args}) = {value} (formula: {data.get('formula', 'n/a')})")
    return "\n".join(lines) if len(lines) > 1 else ""


# --- writer --------------------------------------------------------------------------------


def make_writer(llm: BaseChatModel):
    async def writer(state: ResearchState) -> dict:
        parts = [f"Question: {state['question']}", f"Fact notes:\n{format_findings(state.get('findings', []))}"]
        if state.get("analysis"):
            parts.append(state["analysis"])
        reply = await llm.ainvoke([SystemMessage(WRITER_PROMPT), HumanMessage("\n\n".join(parts))])
        text = normalize_answer(_message_text(reply.content))
        report, unresolved = expand_citations(text, state.get("citations", {}))

        warnings = []
        if not report:
            warnings.append("writer returned an empty report (often a sign of hitting a token limit)")
        if unresolved:
            warnings.append(f"writer: citations that match no retrieved passage: {', '.join(sorted(set(unresolved)))}")
        return {"report": with_disclaimer(report or "No report could be produced."),
                "warnings": warnings, "usage": {"writer": count_tokens(reply)}}

    return writer
