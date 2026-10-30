"""How long a finished answer may be reused, by what the question asks about.

A share price changes every minute; Apple's fiscal 2025 net income never changes. Reusing an
answer costs nothing, so the rule is: as long as the underlying data can't have moved.
Only answers the critic passed are cached (see app.py) - a doubtful answer is never reused.
"""

import re

MINUTE, HOUR, DAY = 60, 3600, 86400

LIVE_RE = re.compile(r"\b(price|share price|stock|quote|market cap|valuation|p/?e|trading|today|now|current|"
                     r"news|this week|happening)\b", re.I)  # not "recent": "most recent quarter" is not live
RECENT_RE = re.compile(r"\b(latest|most recent|last|this) (quarter|results|filing|report|fiscal year|year)\b|\bquarter",
                       re.I)


def cache_ttl(question: str) -> int:
    """Seconds an answer to this (standalone) question stays valid."""
    if LIVE_RE.search(question):
        return 15 * MINUTE   # prices and news move all the time
    if RECENT_RE.search(question):
        return 6 * HOUR      # "latest quarter" changes when a new filing lands (checked a few times a day)
    return 7 * DAY           # a named past period: reported figures don't change
