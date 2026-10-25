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

from src.agents.citations import SOURCE_ID_RE

LABEL_RE = re.compile(r"\[([^\[\]\n]{1,120})\](?!\()")
SOURCE_TAG_RE = re.compile(r"\[[A-Z0-9.^-]+ p\d+\]")
TICKER_RE = re.compile(r"\b\d{3,4}\.[A-Z]{1,3}\b|\b[A-Z]{2,10}\.[A-Z]{2}\b")   # 7203.T, 0700.HK, RELIANCE.NS
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")                        # 20:17:49 in news dates
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITED_RE = re.compile(r"\[[A-Z0-9.^-]+ p\d+\]|\(Yahoo Finance|\(calculated|\([a-z0-9.-]+\.[a-z]{2,},", re.I)
UNVERIFIED = "UNVERIFIED"


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
                if SOURCE_ID_RE.match(part) and part not in citations:
                    issues.append({"claim": _claim(line), "problem": f"[{part}] is not a retrieved source",
                                   "fix": "cite only the source tags given in the fact notes", "source": "code"})
    return issues
