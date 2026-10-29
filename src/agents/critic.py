"""The Phase 5 critic: checks the writer's draft against the evidence it was given.

Chain of verification:
    filing pages --[researcher fact check, nodes.py]--> fact notes --[THIS critic]--> draft

Layer 1 (this module, code only, free and exact):
  - every figure in the draft comes from the fact notes or the analysis (nothing invented);
  - every figure is cited at least once in the draft (a headline sentence may repeat a
    figure that a bullet below cites);
  - no figure from a line marked UNVERIFIED is used;
  - every [TICKER pN] label points to a passage that was actually retrieved.
Layer 2 (Step 5.3): an LLM judge for meaning-level problems (wrong period, scope, unit).

An issue is a dict: {"claim", "problem", "fix", "source": "code" | "llm"}.
"""

import re
from decimal import Decimal, InvalidOperation

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.citations import SOURCE_ID_RE
from src.agents.nodes import format_findings
from src.agents.prompts import CRITIC_PROMPT
from src.agents.state import PlanError, ResearchState, extract_json
from src.agents.toolloop import count_tokens
from src.rag.qa import _message_text

MAX_ISSUES = 5

LABEL_RE = re.compile(r"\[([^\[\]\n]{1,120})\](?!\()")
SOURCE_TAG_RE = re.compile(r"\[[A-Z0-9.^-]+ p\d+\]")
TICKER_RE = re.compile(r"\b\d{3,4}\.[A-Z]{1,3}\b|\b[A-Z]{2,10}\.[A-Z]{2}\b")   # 7203.T, 0700.HK, RELIANCE.NS
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")                        # 20:17:49 in news dates
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITED_RE = re.compile(r"\[[A-Z0-9.^-]+ p\d+\]|[(\[](?:Yahoo Finance|SEC EDGAR|EDINET)|\(calculated|\(<?[a-z0-9.-]+\.[a-z]{2,}>?,",
                      re.I)
UNVERIFIED = "UNVERIFIED"
PLACEHOLDER_LABELS = {"source", "sources", "citation", "citations", "ref", "reference", "cite"}


def _value(token: str) -> Decimal | None:
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def figures(text: str) -> list[tuple[str, Decimal]]:
    """Numbers that need a source: 3+ digits or a decimal, not years, tickers, times or labels."""
    cleaned = SOURCE_TAG_RE.sub(" ", text)
    cleaned = TICKER_RE.sub(" ", cleaned)
    cleaned = TIME_RE.sub(" ", cleaned)
    out = []
    for token in NUMBER_RE.findall(cleaned):
        token = token.rstrip(",")         # "FY2026," -> "2026" (a comma after a number is punctuation)
        digits = re.sub(r"\D", "", token)
        if re.fullmatch(r"(19|20)\d{2}", token):
            continue                      # a year
        if len(digits) < 3 and "." not in token:
            continue                      # small counts like "5 articles", "Q2", "25" in 2024-25
        value = _value(token)
        if value is not None:
            out.append((token, value))
    return out


def _evidence(findings: list[dict], analysis: str) -> tuple[set[Decimal], set[Decimal]]:
    """Values the draft may use, and values that appear ONLY on UNVERIFIED lines."""
    allowed, unverified = set(), set()
    lines = [line for f in findings for line in f.get("facts", "").splitlines()] + analysis.splitlines()
    for line in lines:
        target = unverified if UNVERIFIED in line else allowed
        target.update(v for _, v in figures(line))
    return allowed, unverified - allowed


def _claim(line: str) -> str:
    return " ".join(line.strip().lstrip("-*• ").split())[:140]


def code_checks(draft: str, findings: list[dict], analysis: str, citations: dict[str, str]) -> list[dict]:
    """Layer 1: deterministic checks of the draft against the fact notes and analysis."""
    issues: list[dict] = []
    allowed, unverified = _evidence(findings, analysis or "")
    lines = [line for line in draft.splitlines() if line.strip()]

    cited_values = set()
    for line in lines:
        if CITED_RE.search(line):
            cited_values.update(v for _, v in figures(line))

    reported: set[Decimal] = set()
    for line in lines:
        for token, value in figures(line):
            if value in reported:
                continue
            if value in unverified:
                issues.append({"claim": _claim(line), "problem": f"{token} comes from a line marked UNVERIFIED",
                               "fix": "remove it and say the figure could not be verified", "source": "code"})
                reported.add(value)
            elif value not in allowed:
                issues.append({"claim": _claim(line), "problem": f"{token} is not in the fact notes or the analysis",
                               "fix": "use only figures exactly as given in the fact notes or analysis", "source": "code"})
                reported.add(value)
            elif value not in cited_values:
                issues.append({"claim": _claim(line), "problem": f"{token} is never cited",
                               "fix": "add its source tag, or '(calculated from the figures above)' for computed values",
                               "source": "code"})
                reported.add(value)

    for line in lines:
        for label in LABEL_RE.findall(line):
            for part in (p.strip() for p in re.split(r"[,;]", label)):
                if part.lower() in PLACEHOLDER_LABELS:  # seen live: "returned NOT FOUND.[source]"
                    issues.append({"claim": _claim(line), "problem": f"[{part}] is a placeholder, not a source",
                                   "fix": "remove it; cite only the source tags given in the fact notes", "source": "code"})
                elif SOURCE_ID_RE.match(part) and part not in citations:
                    issues.append({"claim": _claim(line), "problem": f"[{part}] is not a retrieved source",
                                   "fix": "cite only the source tags given in the fact notes", "source": "code"})
    return issues


# --- Layer 2: LLM judge ------------------------------------------------------------------------


def parse_verdict(text: str) -> dict | None:
    """Turn the judge's reply into {"verdict", "issues"}; None if unusable.

    Normalised so the verdict always agrees with the issues: "pass" with issues
    becomes "fail"; "fail" without any usable issue becomes "pass" (nothing to fix).
    """
    try:
        data = extract_json(text)
    except PlanError:
        return None
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "fail"):
        return None
    issues, seen = [], set()
    for item in data.get("issues") or []:
        if isinstance(item, dict) and str(item.get("problem", "")).strip():
            key = (str(item.get("claim", "")).strip().lower(), str(item["problem"]).strip().lower())
            if key in seen:  # seen live: the same unit problem listed twice
                continue
            seen.add(key)
            issues.append({"claim": str(item.get("claim", ""))[:200], "problem": str(item["problem"])[:300],
                           "fix": str(item.get("fix", ""))[:300], "source": "llm"})
    issues = issues[:MAX_ISSUES]
    return {"verdict": "fail" if issues else "pass", "issues": issues}


def _judge_brief(state: ResearchState) -> str:
    return (f"Question: {state.get('question', '')}\n\nFACT NOTES:\n{format_findings(state.get('findings', []))}\n\n"
            f"ANALYSIS:\n{state.get('analysis') or '(none)'}\n\nDRAFT:\n{state.get('draft', '')}")


def make_critic(llm: BaseChatModel | None):
    """Critic node: code checks first; the LLM judge only if the code checks pass (saves tokens)."""

    async def critic(state: ResearchState) -> dict:
        issues = code_checks(state.get("draft", ""), state.get("findings", []), state.get("analysis", ""),
                             state.get("citations", {}))
        checked_by, warnings, tokens = ["code"], [], 0

        if not issues and llm is not None and state.get("draft", "").strip():
            try:
                reply = await llm.ainvoke([SystemMessage(CRITIC_PROMPT), HumanMessage(_judge_brief(state))])
                tokens = count_tokens(reply)
                verdict = parse_verdict(_message_text(reply.content))
                if verdict is None:
                    warnings.append("critic: the judge's reply was not valid JSON; using the code checks only")
                else:
                    issues = verdict["issues"]
                    checked_by.append("llm")
            except Exception as e:  # a broken judge must not block the report
                warnings.append(f"critic: the judge failed ({type(e).__name__}); using the code checks only")

        critique = {"verdict": "fail" if issues else "pass", "issues": issues, "checked_by": checked_by}
        update = {"critique": critique, "critic_rounds": state.get("critic_rounds", 0) + 1, "warnings": warnings}
        if tokens:
            update["usage"] = {"critic": tokens}
        return update

    return critic
