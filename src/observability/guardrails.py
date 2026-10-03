"""Safety rules enforced in code rather than requested from the LLM.

A model can ignore an instruction; code cannot. Anything that must ALWAYS
happen (like the disclaimer) belongs here, not only in a prompt.
"""

DISCLAIMER = (
    "This is an automated research summary of company filings, not financial advice. "
    "Verify figures against the cited sources before relying on them."
)


def with_disclaimer(text: str) -> str:
    """Append the disclaimer exactly once."""
    if DISCLAIMER in text:
        return text
    return f"{text.rstrip()}\n\n{DISCLAIMER}"
