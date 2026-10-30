"""Pure helpers that turn a saved answer into what the Research Desk shows. No Streamlit here.

Key figures come only from the answer's own evidence - the researcher's VERIFIED fact notes
and the analyst's calculator results - copied as written, never recomputed or rounded, so a
tile can never show a number the critic didn't check.
"""

import re
from datetime import date, datetime, timezone

DISCLAIMER_RE = re.compile(r"\n*This is an automated research summary.*$", re.S)
FACT_RE = re.compile(r"^\s*-\s*(?P<metric>[^:(\n]{2,90}?)\s*(?:\((?P<context>[^)\n]*)\))?\s*:\s*(?P<value>.+)$")
# Where a figure ends and its citation / formula begins.
VALUE_END_RE = re.compile(r"\s+\[|\s+\((?:formula|Yahoo|SEC EDGAR|EDINET|calculated|source|<?[a-z0-9.-]+\.[a-z]{2,}>?,)"
                          r"|\s+\([A-Z0-9.^-]+ p\d+\)", re.I)
COMPACT_RE = re.compile(r"\((-?[\d.,]+\s*[KMBT]\s+[A-Z]{3})\)")   # "(20.16 B USD)"
SKIP_RE = re.compile(r"^\s*-\s*(NOT FOUND|UNVERIFIED)", re.I)
# A figure: optional currency sign, digits (with Indian or Western grouping), optional % or K/M/B/T.
NUMBER_UNIT_RE = re.compile(r"^([^\d\s-]{0,3}\s?-?[\d][\d,]*(?:\.\d+)?\s?(?:%|[KMBT]\b)?)\s*(.*)$")
CHANGE_RE = re.compile(r"\b(change|growth|grew|fell|increase|decrease)\b", re.I)


def split_report(report: str) -> tuple[str, str | None]:
    """(answer text, disclaimer) - the disclaimer is shown small, under the answer."""
    match = DISCLAIMER_RE.search(report or "")
    if not match:
        return report or "", None
    return report[:match.start()].rstrip(), match.group().strip()


TOOL_RESULT_RE = re.compile(r"^\s*-\s*(?P<tool>[a-z_]+)\((?P<args>[^)]*)\)\s*=\s*(?P<value>[^(\n]+)")
TOOL_LABELS = {"percent_change": "Change", "cagr": "CAGR", "ratio": "Ratio", "share_of_total": "Share of total",
               "convert_units": "Converted", "series_change": "Change"}


def _figure(line: str) -> dict | None:
    if SKIP_RE.match(line):
        return None
    tool = TOOL_RESULT_RE.match(line)  # an analysis assembled from calculator results: "percent_change(...) = 7.18%"
    if tool and tool.group("tool") in TOOL_LABELS:
        return {"label": TOOL_LABELS[tool.group("tool")], "value": tool.group("value").strip(), "unit": "",
                "context": "calculated"}
    match = FACT_RE.match(line)
    if not match:
        return None
    value = VALUE_END_RE.split(match.group("value"), maxsplit=1)[0].strip().rstrip(".;,")
    compact = COMPACT_RE.search(match.group("value"))
    if compact:
        value = compact.group(1)          # "20,158,000,000 USD (20.16 B USD)" -> "20.16 B USD"
    if not re.search(r"\d", value) or len(value) > 40:
        return None
    # "416,161 USD millions" -> number "416,161" + unit "USD millions": a tile shows the number big
    # and the unit small (seen live: the whole string was cut to "416,161 ..." in a quarter-width tile).
    parts = NUMBER_UNIT_RE.match(value)
    number, unit = (parts.group(1).strip(), (parts.group(2) or "").strip()) if parts else (value, "")
    metric = match.group("metric").strip().rstrip(".")
    return {"label": metric[:1].upper() + metric[1:], "value": number, "unit": unit,
            "context": (match.group("context") or "").strip()}


def key_figures(trace: dict | None, limit: int = 4) -> list[dict]:
    """Headline figures for the tile row: calculations first (they ARE the answer), then facts."""
    if not trace:
        return []
    lines = [line for line in (trace.get("analysis") or "").splitlines() if line.strip().startswith("-")]
    for finding in trace.get("findings") or []:
        lines += [line for line in (finding.get("facts") or "").splitlines() if line.strip().startswith("-")]
    figures, seen = [], set()
    for line in lines:
        figure = _figure(line)
        if figure and (figure["label"].lower(), figure["value"]) not in seen:
            seen.add((figure["label"].lower(), figure["value"]))
            figures.append(figure)
        if len(figures) == limit:
            break
    return figures


def is_change(figure: dict) -> bool:
    return bool(CHANGE_RE.search(figure["label"]))


SOURCE_KINDS = [  # (kind, icon, pattern in the citation text)
    ("SEC filing", ":material/account_balance:", re.compile(r"SEC EDGAR", re.I)),
    ("Japan FSA filing", ":material/account_balance:", re.compile(r"EDINET", re.I)),
    ("Market data", ":material/candlestick_chart:", re.compile(r"Yahoo Finance", re.I)),
]


def sources(answer: dict) -> list[dict]:
    """Every source behind an answer, with its kind and whether the answer cites it.

    The citations map holds every page the researcher READ; only those whose citation text
    appears in the answer are "cited" (seen live: 4 pages read, 1 cited - listing all 4 as
    sources overstated the evidence).
    """
    text = answer.get("content") or ""
    out = [{"kind": "Annual report", "icon": ":material/description:", "label": citation, "ref": ref,
            "cited": citation in text or f"[{ref}]" in text}
           for ref, citation in (answer.get("citations") or {}).items()]
    out.sort(key=lambda item: not item["cited"])
    seen = set()
    for match in re.finditer(r"\(((?:Yahoo Finance|SEC EDGAR|EDINET)[^)]*)\)", text):
        citation = match.group(1).strip()
        if citation not in seen:
            seen.add(citation)
            kind, icon = next((k, i) for k, i, p in SOURCE_KINDS if p.search(citation))
            out.append({"kind": kind, "icon": icon, "label": citation, "ref": "", "cited": True})
    for match in re.finditer(r"\(<?([a-z0-9.-]+\.[a-z]{2,})>?,\s*([^)]*\d{4}[^)]*)\)", text):
        citation = f"{match.group(1)}, {match.group(2).strip()}"
        if citation not in seen:
            seen.add(citation)
            out.append({"kind": "News", "icon": ":material/newspaper:", "label": citation, "ref": "", "cited": True})
    return out


def source_kinds(answer: dict) -> list[tuple[str, str]]:
    """Distinct (kind, icon) pairs of the CITED sources, in order of first use."""
    kinds: dict[str, str] = {}
    for source in sources(answer):
        if source["cited"]:
            kinds.setdefault(source["kind"], source["icon"])
    return list(kinds.items())


def _as_datetime(value) -> datetime:
    if isinstance(value, datetime):
        moment = value
    else:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def group_by_day(conversations: list[dict], today: date | None = None) -> list[tuple[str, list[dict]]]:
    """Sidebar sections: Today / Yesterday / Previous 7 days / Older (newest first, as given)."""
    today = today or datetime.now(timezone.utc).date()
    groups: dict[str, list[dict]] = {}
    for conversation in conversations:
        age = (today - _as_datetime(conversation["updated_at"]).date()).days
        label = "Today" if age <= 0 else "Yesterday" if age == 1 else "Previous 7 days" if age <= 7 else "Older"
        groups.setdefault(label, []).append(conversation)
    order = ["Today", "Yesterday", "Previous 7 days", "Older"]
    return [(label, groups[label]) for label in order if label in groups]


def seconds_between(earlier, later) -> float | None:
    try:
        return max((_as_datetime(later) - _as_datetime(earlier)).total_seconds(), 0.0)
    except (TypeError, ValueError):
        return None


def verification(answer: dict) -> tuple[str, str, str]:
    """(label, icon, badge color) for an answer's verification state - always icon + words."""
    if answer.get("status") == "error":
        return "Not answered", ":material/error:", "red"
    critique = (answer.get("trace") or {}).get("critique") or {}
    if critique.get("verdict") == "pass":
        return "Checked against sources", ":material/verified:", "green"
    if critique:
        return "Partly unverified - see note", ":material/warning:", "orange"
    return "Not checked", ":material/help:", "gray"
