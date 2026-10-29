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
from datetime import date
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from src.agents.citations import collect_citations, expand_citations
from src.agents.prompts import (
    ANALYST_PROMPT, PLANNER_PROMPT, RESEARCHER_PROMPT, SOURCE_CHECK_PROMPT, WRITER_PROMPT,
)
from src.agents.state import MAX_STEPS, Plan, PlanError, ResearchState, extract_json, fallback_plan, parse_plan
from src.agents.toolloop import count_tokens, run_tool_loop
from src.observability.guardrails import with_disclaimer
from src.rag.qa import _message_text, normalize_answer

RESEARCH_TOOLS = {"get_quote", "get_quarterly_financials", "get_annual_financials", "list_supported_markets",
                  "search_filings", "list_filings", "get_sec_financials", "get_sec_filings", "search_news"}
CALC_TOOLS = {"percent_change", "cagr", "ratio", "share_of_total", "convert_units", "series_change"}
# Each plan step gets only its source's tools (an "any" step gets all of RESEARCH_TOOLS).
# Seen live: a market step for Microsoft still called search_filings, which could only fail.
STEP_TOOLS = {"market": {"get_quote", "get_quarterly_financials", "get_annual_financials", "list_supported_markets"},
              "filings": {"search_filings", "list_filings"},
              "sec": {"get_sec_financials", "get_sec_filings"},
              "news": {"search_news"}}

Events = Callable[[str, dict], None] | None


def companies_context() -> str:
    """One line per indexed annual report, so the planner never has to guess tickers."""
    from src.ingestion.loader import load_registry

    return "\n".join(
        f"- {d.company} ({d.ticker}, {d.market}): {d.doc_type}, fiscal year ended {d.fiscal_year_end}, "
        f"figures in {d.reporting_unit}"
        for d in load_registry()
    )


def indexed_tickers() -> set[str]:
    from src.ingestion.loader import load_registry

    return {d.ticker for d in load_registry()}


def reroute_unindexed(plan: dict, known: set[str]) -> tuple[dict, list[str]]:
    """Send filings steps for companies with no indexed annual report to a source that has them.

    Seen live: asked about Microsoft, the planner chose filings anyway; the search could only
    fail. A US ticker (no exchange suffix) goes to the SEC's official figures; anything else
    to market data, which covers any listed company.
    """
    known_bases = {_base_ticker(t) for t in known}
    notes = []
    for step in plan["steps"]:
        ticker = step.get("ticker")
        if step["kind"] == "filings" and ticker and _base_ticker(ticker) not in known_bases:
            if "." not in ticker:
                step["kind"] = "sec"
                step["goal"] += f" (no annual report for {ticker} is indexed: use its official SEC figures)"
                notes.append(f"planner: no annual report indexed for {ticker}; filings step switched to SEC data")
            else:
                step["kind"] = "market"
                step["goal"] += (f" (no annual report for {ticker} is indexed: use market data - annual and quarterly "
                                 f"statement figures from Yahoo Finance)")
                notes.append(f"planner: no annual report indexed for {ticker}; filings step switched to market data")
    return plan, notes


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
            text, seen = str(p.get("text", "")), into.get(p["source_id"], "")
            if text not in seen:  # the same chunk found by a second search: keep one copy
                into[p["source_id"]] = seen + "\n" + text


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def fact_problems(facts: str, passages: dict[str, str]) -> list[tuple[int, str]]:
    """Cheap grounding check on the researcher's notes (an early, code-only Critic).

    Returns (line index, problem) pairs. For every fact line that cites a filing passage:
      - each figure in the line (3+ digits, not a year) must appear in that passage;
      - a line saying "consolidated" must not cite a passage from the standalone statements;
      - a company total must not come from a non-last column of a segment table.
    Seen live: a "consolidated" revenue figure that was really the standalone "Value of Sales",
    and O2C's segment column reported as the company total.
    """
    problems: list[tuple[int, str]] = []
    for i, line in enumerate(facts.splitlines()):
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
                problems.append((i, f"researcher: {figure} is not in its cited source {', '.join(tags)}"))
        upper = text.upper()
        if ("consolidated" in claim.lower() and "STANDALONE FINANCIAL STATEMENTS" in upper
                and "CONSOLIDATED FINANCIAL STATEMENTS" not in upper):
            problems.append((i, f"researcher: a 'consolidated' figure cites a STANDALONE statement page ({', '.join(tags)})"))
        problems += [(i, w) for w in _segment_column_warnings(claim, text)]
    return problems


def _base_ticker(ticker: str) -> str:
    return ticker.strip().upper().split(".")[0]  # "RELIANCE.NS" and "reliance" are the same company


def company_problems(facts: str, ticker: str | None) -> list[tuple[int, str]]:
    """Fact lines for one company's step that cite ANOTHER company's filing page.

    A search without a ticker returns Apple / Reliance / Toyota pages for any question; a
    figure copied from them passes the other checks (the number IS on the cited page).
    """
    if not ticker:
        return []
    problems = []
    for i, line in enumerate(facts.splitlines()):
        others = [t for t in SOURCE_TAG_RE.findall(line) if _base_ticker(t.split(" p")[0]) != _base_ticker(ticker)]
        if others:
            problems.append((i, f"researcher: this step is about {ticker} but the line cites another company's "
                                f"filing ({', '.join(others)})"))
    return problems


def check_facts_against_sources(facts: str, passages: dict[str, str]) -> list[str]:
    return [message for _, message in fact_problems(facts, passages)]


def mark_unverified(facts: str, problems: list[tuple[int, str]]) -> str:
    """Quarantine flagged lines so the analyst and writer never use their figures."""
    flagged = {i for i, _ in problems}
    lines = facts.splitlines()
    for i in flagged:
        lines[i] = "- UNVERIFIED (failed the source check, do not use): " + lines[i].lstrip("- ").strip()
    return "\n".join(lines)


MAX_SOURCE_CHARS = 9_000  # ~2.5k tokens of page text: keeps the verifier's request far under 8k


async def source_check(verifier: BaseChatModel, facts: str,
                       passages: dict[str, str]) -> tuple[list[tuple[int, str]], int, list[str]]:
    """LLM check of filings fact lines against the page text they cite.

    Catches what the code checks can't: a figure that IS on the page but in another column,
    year or segment. Seen in the eval: Oil & Gas's 2023-24 value 24,331 reported as Retail
    2024-25; gpt-oss-120b rejected it (and wrong-year / O2C-as-total) 5/5 in a live test.
    Returns (problems as (line index, message), tokens used, warnings). Never blocks: on any
    failure it returns no problems plus a warning.
    """
    lines = facts.splitlines()
    candidates = [(i, line) for i, line in enumerate(lines)
                  if SOURCE_TAG_RE.search(line) and _big_numbers(SOURCE_TAG_RE.sub("", line))
                  and not line.lstrip("- ").startswith("UNVERIFIED")]
    if not candidates:
        return [], 0, []
    cited = list(dict.fromkeys(tag for _, line in candidates for tag in SOURCE_TAG_RE.findall(line)))
    cited = [tag for tag in cited if passages.get(tag, "").strip()]
    if not cited:
        return [], 0, []
    share = MAX_SOURCE_CHARS // len(cited)  # every cited page gets room, not just the first
    source_text = "\n\n".join(f"SOURCE PAGE ({tag}):\n{passages[tag].strip()[:share]}" for tag in cited)
    numbered = "\n".join(f"{n}. {line.lstrip('- ').strip()}" for n, (_, line) in enumerate(candidates, start=1))
    try:
        reply = await verifier.ainvoke([SystemMessage(SOURCE_CHECK_PROMPT),
                                        HumanMessage(f"{source_text}\n\nFACTS:\n{numbered}")])
    except Exception as e:
        return [], 0, [f"source check failed ({describe_error(e)}); relying on the code checks"]
    tokens = count_tokens(reply)
    try:
        results = extract_json(_message_text(reply.content)).get("results") or []
    except PlanError:
        return [], tokens, ["source check reply was not valid JSON; relying on the code checks"]
    problems = []
    for r in results:
        if not isinstance(r, dict) or r.get("supported") is not False:
            continue
        try:
            index = candidates[int(r.get("fact")) - 1][0]
        except (TypeError, ValueError, IndexError):
            continue
        problems.append((index, f"researcher: source check - {str(r.get('reason', 'not supported by the page'))[:160]}"))
    return problems, tokens, []


def correction_brief(problems: list[tuple[int, str]]) -> str:
    """Tell the retry WHAT was wrong and WHERE to look instead.

    Seen in the eval: a retry told only what was wrong picked the same segment-table figure again.
    """
    issues = "\n".join(f"- {message.removeprefix('researcher: ')}" for _, message in problems)
    hints = []
    if any("segment table" in m for _, m in problems):
        hints.append('Search for "consolidated statement of profit and loss" (or the consolidated balance '
                     "sheet) instead of the segment note, and take the company total from that statement.")
    if any("STANDALONE" in m for _, m in problems):
        hints.append('Search with the word "consolidated" in the query and use only pages from the '
                     "CONSOLIDATED financial statements.")
    if any("not in its cited source" in m for _, m in problems):
        hints.append("Copy each figure exactly as the cited passage shows it, and cite the passage it came from.")
    if any("another company's filing" in m for _, m in problems):
        hints.append("Only use passages from this step's company. If search_filings has no annual report for it, "
                     "write NOT FOUND for the filings figures - never use another company's figures.")
    if any("source check" in m for _, m in problems):
        hints.append("Read the table header to find the column for the requested segment and year, and take the "
                     "figure from that exact column and row.")
    return ("\n\nYour previous notes for this step failed an automatic source check:\n" + issues +
            "\nWhat to do now: " + " ".join(hints or ["Search again and write corrected notes."]))


def _big_numbers(line: str) -> list[str]:
    """Figures with 3+ digits that are not years: the 'columns' of a financial table row."""
    return [n for n in NUMBER_RE.findall(line)
            if len(_digits(n)) >= 3 and not re.fullmatch(r"(19|20)\d{2}", _digits(n))]


def _segment_column_warnings(claim: str, source: str) -> list[str]:
    """A company total taken from the wrong column of a segment table.

    Seen live (twice): Reliance's segment row is "6,05,006 24,881 2,91,043 1,31,336 49,749 - 9,80,136"
    (O2C, Oil & Gas, Retail, Digital, Others, Unallocable, TOTAL). Taking the FIRST number as the
    consolidated total gave 48.11% instead of 29.69%. The total is the row's last column.
    """
    lowered = claim.lower()
    if "segment" in lowered or not ("consolidated" in lowered or "total" in lowered):
        return []
    if "segment" not in source.lower():
        return []
    warnings = []
    for figure in _big_numbers(claim):
        for row in source.splitlines():
            columns = _big_numbers(row)
            if len(columns) >= 4 and _digits(figure) in {_digits(c) for c in columns[:-1]}:
                warnings.append(f"researcher: {figure} comes from a multi-column segment table row and is not its "
                                f"last ('Total') column ({columns[-1]}); a company total is usually the last column")
                break
    return warnings


def format_findings(findings: list[dict]) -> str:
    if not findings:
        return "(no fact notes)"
    return "\n\n".join(f"Step {f['step_id']} ({f['kind']}): {f['goal']}\n{f['facts']}" for f in findings)


def only(tools: list[BaseTool], allowed: set[str]) -> list[BaseTool]:
    return [t for t in tools if t.name in allowed]


# --- planner -------------------------------------------------------------------------------


def make_planner(llm: BaseChatModel, companies: str | None = None, known_tickers: set[str] | None = None):
    system = (PLANNER_PROMPT.replace("<<COMPANIES>>", companies if companies is not None else companies_context())
              .replace("<<MAX_STEPS>>", str(MAX_STEPS)))
    known = known_tickers if known_tickers is not None else indexed_tickers()

    async def planner(state: ResearchState) -> dict:
        question = state["question"]
        # The date lets "latest year" questions pick the right year (seen live: the model assumed FY2025).
        messages = [SystemMessage(system), HumanMessage(f"Today's date: {date.today().isoformat()}\n\n{question}")]
        tokens, warnings = 0, []
        for attempt in (1, 2):
            try:
                reply = await llm.ainvoke(messages)
            except Exception as e:  # e.g. daily budget gone: research with a simple plan anyway
                plan = fallback_plan(question)
                warnings.append(f"planner failed ({describe_error(e)}); using a one-step fallback plan")
                break
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
        plan_dict, notes = reroute_unindexed(plan.model_dump(), known)
        return {"plan": plan_dict, "warnings": warnings + notes, "usage": {"planner": tokens}}

    return planner


# --- researcher ----------------------------------------------------------------------------


def _step_brief(question: str, step: dict) -> str:
    ticker = f" Ticker: {step['ticker']}." if step.get("ticker") else ""
    return (f"Today's date: {date.today().isoformat()}\nOverall question (for context): {question}\n\n"
            f"Your step ({step['kind']}):{ticker} {step['goal']}")


def describe_error(error: Exception) -> str:
    text = str(error)
    if "413" in text or "Request too large" in text:
        return "the request exceeded the model's per-request token limit (413)"
    if "per day" in text or "TPD" in text:
        return "the model's daily token budget is used up"
    return f"{type(error).__name__}: {text[:150]}"


def make_researcher(llm: BaseChatModel, tools: list[BaseTool], max_rounds: int = 3, on_event: Events = None,
                    compact_older_results: int = 1500, verifier: BaseChatModel | None = None):
    """verifier: optional stronger model for the LLM source check (run only when the code checks pass)."""
    research_tools = only(tools, RESEARCH_TOOLS)

    async def run_step(brief: str, step: dict):
        messages = [SystemMessage(RESEARCHER_PROMPT), HumanMessage(brief)]
        ticker = step.get("ticker")
        # A filings search without a ticker returns other companies' pages: pin it to the step's company.
        defaults = {"search_filings": {"ticker": ticker}} if ticker else None
        step_tools = only(research_tools, STEP_TOOLS[step["kind"]]) if step["kind"] in STEP_TOOLS else research_tools
        # Its server is down or disabled: better any real source than a researcher with no tools at all.
        step_tools = step_tools or research_tools
        return await run_tool_loop(llm, step_tools, messages, agent="researcher", max_rounds=max_rounds,
                                   compact_older_results=compact_older_results, on_event=on_event,
                                   default_args=defaults)

    async def researcher(state: ResearchState) -> dict:
        plan = Plan.model_validate(state["plan"])
        findings, calls, citations, warnings, tokens, verifier_tokens = [], [], {}, [], 0, 0
        for step in plan.model_dump()["steps"]:  # sequential: parallel calls would burst the token limit
            brief = _step_brief(state["question"], step)
            passages: dict[str, str] = {}
            facts, problems = "", []
            # Attempt 1, then (only if the source check flags the notes) one corrected attempt.
            # Seen in the eval: the checks caught both wrong answers, but nothing acted on them.
            for attempt in (1, 2):
                try:
                    result = await run_step(brief if attempt == 1 else brief + correction_brief(problems), step)
                except Exception as e:  # one failed step must not take down the whole report
                    reason = describe_error(e)
                    warnings.append(f"researcher step {step['id']} failed: {reason}")
                    if attempt == 1:
                        facts = f"- NOT FOUND: this research step failed ({reason})"
                    break
                tokens += result.tokens
                calls += result.calls
                for name, content in result.results:
                    if name == "search_filings":
                        collect_citations(content, citations)
                        collect_passages(content, passages)
                if not result.calls:
                    warnings.append(f"researcher step {step['id']}: no tool was used, so its notes are unverified")
                if result.hit_cap:
                    warnings.append(f"researcher step {step['id']}: stopped at the {max_rounds}-round tool limit")
                facts = result.text or "- NOT FOUND: the researcher returned no notes"
                problems = company_problems(facts, step.get("ticker")) + fact_problems(facts, passages)
                if not problems and verifier is not None:  # code checks first: free, and they catch the obvious
                    problems, used, notes = await source_check(verifier, facts, passages)
                    verifier_tokens += used
                    warnings += notes
                if not problems:
                    if attempt == 2:
                        warnings.append(f"researcher step {step['id']}: notes corrected after the source check")
                    break
                if attempt == 1:
                    warnings += [f"{m} (retrying step {step['id']})" for _, m in problems]

            if problems:  # still flagged after the retry: quarantine those lines
                facts = mark_unverified(facts, problems)
                warnings += [f"{m} - marked UNVERIFIED" for _, m in problems]
            findings.append({"step_id": step["id"], "kind": step["kind"], "goal": step["goal"], "facts": facts})
        usage = {"researcher": tokens, **({"verifier": verifier_tokens} if verifier_tokens else {})}
        return {"findings": findings, "citations": citations, "tool_calls": calls,
                "warnings": warnings, "usage": usage}

    return researcher


# --- analyst -------------------------------------------------------------------------------


def make_analyst(llm: BaseChatModel, tools: list[BaseTool], max_rounds: int = 4, on_event: Events = None):
    calc_tools = only(tools, CALC_TOOLS)

    async def analyst(state: ResearchState) -> dict:
        plan = state.get("plan", {})
        brief = (f"Question: {state['question']}\n\nCalculation to do: {plan.get('calculation') or 'whatever the question needs'}"
                 f"\n\nFact notes:\n{format_findings(state.get('findings', []))}")
        try:
            result = await run_tool_loop(llm, calc_tools, [SystemMessage(ANALYST_PROMPT), HumanMessage(brief)],
                                         agent="analyst", max_rounds=max_rounds, on_event=on_event)
        except Exception as e:  # the writer can still report the researched figures
            return {"analysis": "", "warnings": [f"analyst failed: {describe_error(e)}; no calculation available"]}
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


def critic_issues(state: ResearchState) -> list[dict]:
    """Issues from the critic's latest verdict, if it failed the draft."""
    critique = state.get("critique") or {}
    return critique.get("issues", []) if critique.get("verdict") == "fail" else []


def make_writer(llm: BaseChatModel):
    """Writes the DRAFT (citations stay as [TICKER pN] so the critic can check them).

    If the critic failed the previous draft, the writer revises it using the critic's issues.
    """

    async def writer(state: ResearchState) -> dict:
        parts = [f"Question: {state['question']}", f"Fact notes:\n{format_findings(state.get('findings', []))}"]
        if state.get("analysis"):
            parts.append(state["analysis"])
        issues = critic_issues(state)
        revising = bool(issues and state.get("draft"))
        if revising:
            listed = "\n".join(f"- {i.get('claim', '')}: {i.get('problem', '')}" + (f" (fix: {i['fix']})" if i.get("fix") else "")
                               for i in issues)
            parts.append(f"Your previous draft:\n{state['draft']}\n\nA reviewer checked it against the fact notes "
                         f"and analysis and found these problems:\n{listed}\n\nRewrite the answer so every problem "
                         "is fixed. Use only the fact notes and analysis above.")
        try:
            reply = await llm.ainvoke([SystemMessage(WRITER_PROMPT), HumanMessage("\n\n".join(parts))])
        except Exception as e:  # keep the previous draft rather than lose the answer
            return {"warnings": [f"writer failed: {describe_error(e)}"]}

        update = {"draft": normalize_answer(_message_text(reply.content)), "usage": {"writer": count_tokens(reply)}}
        if revising:
            update["revisions"] = state.get("revisions", 0) + 1
        return update

    return writer


def finalize(state: ResearchState) -> dict:
    """Code only: expand citations, report what couldn't be verified, add the disclaimer."""
    draft = state.get("draft", "")
    report, unresolved = expand_citations(draft, state.get("citations", {}))
    warnings = []
    if not report:
        warnings.append("writer returned an empty report (often a sign of hitting a token limit)")
    if unresolved:
        warnings.append(f"writer: citations that match no retrieved passage: {', '.join(sorted(set(unresolved)))}")

    issues = critic_issues(state)
    if report and issues:  # the critic still wasn't satisfied after the allowed revisions
        report += "\n\n**Verification note:** an automatic review could not confirm these points:\n" + "\n".join(
            f"- {i.get('claim', '')}: {i.get('problem', '')}" for i in issues)
        warnings.append(f"critic: {len(issues)} issue(s) remain after {state.get('revisions', 0)} revision(s)")
    return {"report": with_disclaimer(report or "No report could be produced."), "warnings": warnings}
