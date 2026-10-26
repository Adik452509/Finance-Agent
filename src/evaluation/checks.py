"""Scoring for the Phase 4 pipeline eval (tests/eval/pipeline_questions.json).

Kept separate from the runner so the scoring itself is unit-tested offline:
a wrong checker would make every eval result meaningless.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal

REFUSAL_PHRASES = ("not found", "couldn't find", "could not find", "not available", "no data",
                   "not included", "not in the", "does not contain", "do not contain", "unavailable",
                   "no information", "not reported", "not provided", "could not be verified",
                   "could not verify", "cannot be determined", "could not be determined")
# (site, date) - the writer sometimes wraps the site in <angle brackets> (seen live).
NEWS_CITATION_RE = re.compile(r"\(<?[a-z0-9.-]+\.[a-z]{2,}>?,\s*[^)]*\d{4}[^)]*\)", re.I)


@dataclass
class CheckResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    wrong_figure: bool = False                          # a known-wrong figure reached the report
    critic: dict = field(default_factory=dict)          # see critic_metrics()


def critic_metrics(final: dict) -> dict:
    """How the Phase 5 critic loop went for one run.

    outcome: "first draft" (passed at once), "fixed" (passed after revision),
             "unresolved" (still failing: report has a verification note), or "not run".
    """
    rounds = final.get("critic_rounds", 0)
    verdict = (final.get("critique") or {}).get("verdict")
    if not rounds:
        outcome = "not run"
    elif verdict == "pass":
        outcome = "first draft" if rounds == 1 else "fixed"
    else:
        outcome = "unresolved"
    return {"outcome": outcome, "rounds": rounds, "revisions": final.get("revisions", 0),
            "open_issues": len((final.get("critique") or {}).get("issues", [])),
            "checked_by": (final.get("critique") or {}).get("checked_by", [])}


def body_of(report: str) -> str:
    """The report without the disclaimer (which could otherwise match phrases)."""
    return report.split("\n\nThis is an automated")[0]


def has_number(text: str, number: str) -> bool:
    """True if `number` appears in `text` with the same VALUE, whatever the formatting.

    9,801.36 == 9801.36; 50,684,952 == 50684952; 19.50 == 19.5. Compared by value, so
    7.2 does not match 7.25 and 19.5 does not match 219.5.
    """
    flat = text.replace(",", "").replace(" ", "").replace(" ", "")
    target = Decimal(number.replace(",", ""))
    return any(Decimal(token) == target for token in re.findall(r"\d+(?:\.\d+)?", flat))


def check_result(expect: dict, final: dict) -> CheckResult:
    """Score one pipeline run (its final state) against a question's expectations."""
    report = body_of(final.get("report", ""))
    failures, notes = [], []

    numbers = expect.get("numbers") or []
    wrong_figure = False
    if numbers and not any(has_number(report, n) for n in numbers):
        failures.append(f"expected figure missing (any of {numbers})")
        # Missing figure + no admission of a gap = a confidently WRONG answer, even when the
        # wrong number isn't one we predicted (seen in the eval: Oil & Gas's 24,331 as "Retail").
        if not any(p in report.lower() for p in REFUSAL_PHRASES):
            wrong_figure = True
    for wrong in expect.get("must_not") or []:
        if has_number(report, wrong.rstrip("%")):
            failures.append(f"contains known-wrong figure {wrong}")
            wrong_figure = True

    used = {c["name"] for c in final.get("tool_calls", []) if c.get("ok") is not False}
    missing_tools = [t for t in expect.get("tools") or [] if t not in used]
    if missing_tools:
        failures.append(f"required tools not used successfully: {missing_tools}")
    any_of = expect.get("tools_any") or []
    if any_of and not used & set(any_of):
        failures.append(f"none of these tools was used successfully: {any_of}")

    route = expect.get("route")
    if route and final.get("route") != route:
        failures.append(f"route was {final.get('route')!r}, expected {route!r}")

    if expect.get("refusal") and not any(p in report.lower() for p in REFUSAL_PHRASES):
        failures.append("expected the report to say the information is not available")

    if expect.get("cites_news") and not NEWS_CITATION_RE.search(report):
        failures.append("news items are not cited as (site, date)")

    if not report.strip() or report.startswith("No report could be produced"):
        failures.append("no report was produced")

    warnings = final.get("warnings", [])
    if warnings:
        notes.append(f"{len(warnings)} warning(s): " + " | ".join(w[:90] for w in warnings))
    return CheckResult(passed=not failures, failures=failures, notes=notes,
                       wrong_figure=wrong_figure, critic=critic_metrics(final))
