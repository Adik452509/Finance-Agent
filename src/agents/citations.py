"""Citation handling shared by every agent that writes answers.

Filings passages carry a source_id like "7203.T p164". Agents cite it in square
brackets; code (not the model) expands it into the full citation, so the reader
always sees the real source - and any label that matches no retrieved passage
is reported instead of silently passing.
"""

import json
import re
from typing import Any

from src.rag.qa import _message_text

# [7203.T p164] or [AAPL p32, AAPL p39] - but not a markdown link "[text](url)".
BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,120})\](?!\()")
# Anything shaped like a source label, including malformed ones such as "7203.T p-",
# and tool names used as if they were sources, e.g. "[get_quote]" (seen live).
SOURCE_ID_RE = re.compile(r"^[A-Z0-9.^-]+ p\S*$|^S\d+$|^[a-z]+(?:_[a-z]+)+$")


def collect_citations(content: Any, into: dict[str, str]) -> None:
    """Remember source_id -> citation from a search_filings result."""
    try:
        data = json.loads(_message_text(content))
    except (json.JSONDecodeError, TypeError):
        return
    for passage in (data.get("results") or []) if isinstance(data, dict) else []:
        if isinstance(passage, dict) and passage.get("source_id") and passage.get("citation"):
            into[passage["source_id"]] = passage["citation"]


def expand_citations(text: str, citations: dict[str, str]) -> tuple[str, list[str]]:
    """Replace [source_id] labels with full citations; report labels that match nothing."""
    unresolved: list[str] = []

    def replace(match: re.Match) -> str:
        parts = [p.strip() for p in re.split(r"[,;]", match.group(1))]
        if parts and all(p in citations for p in parts):
            return "(" + "; ".join(citations[p] for p in parts) + ")"
        unresolved.extend(p for p in parts if SOURCE_ID_RE.match(p) and p not in citations)
        return match.group(0)

    return BRACKET_RE.sub(replace, text), unresolved
