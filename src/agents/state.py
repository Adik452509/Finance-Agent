"""Shared state for the Phase 4 multi-agent graph, and the planner's plan format.

The agents never call each other. Each one reads the shared state and returns
an update; LangGraph merges updates using the reducer attached to each field:

    planner    -> plan
    researcher -> findings (compact fact notes), citations, tool_calls
    analyst    -> analysis, tool_calls
    writer     -> report
    everyone   -> warnings, usage (tokens per agent)

The plan comes from an LLM as JSON, so it is parsed defensively: code fences,
surrounding prose, synonyms ("price" -> "market") and stray "calculation"
steps are tidied up; anything unusable raises PlanError and the caller falls
back to fallback_plan() instead of crashing.
"""

import json
import operator
import re
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, ValidationError

MAX_STEPS = 5
StepKind = Literal["market", "filings", "news", "any"]

# What the LLM might write -> what we use.
KIND_SYNONYMS = {
    "market": "market", "price": "market", "quote": "market", "quarterly": "market", "stock": "market",
    "filings": "filings", "filing": "filings", "annual_report": "filings", "report": "filings", "rag": "filings",
    "news": "news", "web": "news",
    "any": "any", "research": "any", "general": "any",
}
CALC_KINDS = {"calculation", "calculate", "calc", "analysis", "math", "compute"}
CALC_WORDS = re.compile(r"\b(percent|percentage|%|growth|grow|change|ratio|convert|cagr|share of|margin|compare)\b", re.I)
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=]{0,19}$")


class PlanError(ValueError):
    """The planner's output could not be turned into a usable plan."""


class PlanStep(BaseModel):
    id: int
    kind: StepKind
    goal: str = Field(min_length=3, max_length=400)
    ticker: str | None = None


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_STEPS)
    needs_calculation: bool = False
    calculation: str = ""  # what the analyst should compute, in plain words


# --- reducers -----------------------------------------------------------------------


def merge_dicts(old: dict, new: dict) -> dict:
    return {**(old or {}), **(new or {})}


def add_usage(old: dict, new: dict) -> dict:
    """Sum token counts per agent across updates."""
    total = dict(old or {})
    for agent, tokens in (new or {}).items():
        total[agent] = total.get(agent, 0) + tokens
    return total


# --- shared state ------------------------------------------------------------------------


class Finding(TypedDict):
    step_id: int
    kind: str
    goal: str
    facts: str  # compact bullet notes with [source] tags, written by the researcher


class ToolCallRecord(TypedDict):
    agent: str
    name: str
    args: dict
    ok: bool | None
    error: str | None
    seconds: float


class ResearchState(TypedDict, total=False):
    question: str
    plan: dict                                                  # Plan.model_dump()
    route: str                                                  # supervisor's decision: "analyst" or "writer"
    findings: Annotated[list[Finding], operator.add]
    analysis: str
    draft: str                                                  # writer's text, citations still as [TICKER pN]
    critique: dict                                              # critic's latest verdict (Phase 5)
    critic_rounds: int                                          # how many times the critic has checked a draft
    revisions: int                                              # how many times the writer has revised
    report: str                                                 # final text: citations expanded + disclaimer
    citations: Annotated[dict[str, str], merge_dicts]           # source_id -> full citation
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    warnings: Annotated[list[str], operator.add]
    usage: Annotated[dict[str, int], add_usage]                 # agent -> tokens used


# --- plan parsing ------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of LLM text (handles ```json fences and prose)."""
    if not isinstance(text, str) or "{" not in text:
        raise PlanError("no JSON object found in the planner output")
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise PlanError(f"planner output is not valid JSON: {e.msg}") from None
    if not isinstance(data, dict):
        raise PlanError("planner output must be a JSON object")
    return data


def parse_plan(text: str) -> tuple[Plan, list[str]]:
    """Turn planner text into a validated Plan. Returns (plan, notes about fixes made)."""
    data = extract_json(text)
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanError("plan has no steps")

    notes: list[str] = []
    needs_calc = bool(data.get("needs_calculation", False))
    calculation = str(data.get("calculation") or "")
    steps: list[dict] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "any")).strip().lower()
        goal = str(raw.get("goal", "")).strip()
        if kind in CALC_KINDS:
            # Calculation is the analyst's job, not a research step.
            needs_calc = True
            calculation = calculation or goal
            notes.append(f"moved calculation step to the analyst: {goal[:60]}")
            continue
        if kind not in KIND_SYNONYMS:
            notes.append(f"unknown step kind {kind!r} treated as 'any'")
        ticker = raw.get("ticker")
        ticker = str(ticker).strip().upper() if ticker else None
        if ticker and not TICKER_RE.match(ticker):
            notes.append(f"dropped invalid ticker {ticker!r}")
            ticker = None
        steps.append({"kind": KIND_SYNONYMS.get(kind, "any"), "goal": goal, "ticker": ticker})

    if not steps:
        raise PlanError("plan has no research steps")
    if len(steps) > MAX_STEPS:
        notes.append(f"plan had {len(steps)} steps; kept the first {MAX_STEPS}")
        steps = steps[:MAX_STEPS]
    for i, step in enumerate(steps, start=1):
        step["id"] = i  # renumber: the LLM's own ids can't be trusted

    try:
        plan = Plan(steps=steps, needs_calculation=needs_calc, calculation=calculation)
    except ValidationError as e:
        raise PlanError(f"plan failed validation: {e.errors()[0]['msg']}") from None
    return plan, notes


def fallback_plan(question: str) -> Plan:
    """A safe one-step plan when the planner fails: research everything, calculate if asked."""
    return Plan(
        steps=[PlanStep(id=1, kind="any", goal=question[:400] if len(question) >= 3 else "general research")],
        needs_calculation=bool(CALC_WORDS.search(question)),
        calculation=question[:400] if CALC_WORDS.search(question) else "",
    )
