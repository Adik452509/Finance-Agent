"""Follow-up questions: rewrite "and in 2024?" into a question the research team can answer alone.

The team sees one question at a time and has no chat history (it shouldn't: history would cost
tokens at every agent). So the history is used once, here: a small model turns the latest
message into a standalone question, which is what the team researches, what is cached, and
what is stored as the message's standalone_question.

To save tokens the rewrite only runs when the message looks like a follow-up. Any failure
falls back to the original question - a slightly vague question beats no answer.
"""

import re
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.prompts import REWRITE_PROMPT
from src.agents.toolloop import count_tokens
from src.rag.qa import _message_text

TURN_CHARS = 600          # an earlier answer is long; its first lines name the company, metric and period
DISCLAIMER_RE = re.compile(r"\n+This is an automated research summary.*", re.S)
SHORT_WORDS = 12
# Words that point back at something said before.
REFERENCE_RE = re.compile(
    r"\b(it|its|it's|they|them|their|that|those|these|this|same|also|previous|prior|earlier|above|"
    r"what about|how about|and in|and for|and the|compared|compare|vs|versus|instead|again|then)\b", re.I)


def looks_like_follow_up(question: str, has_history: bool) -> bool:
    if not has_history:
        return False
    return len(question.split()) <= SHORT_WORDS or bool(REFERENCE_RE.search(question))


def _history(turns: list[tuple[str, str]]) -> str:
    lines = []
    for role, text in turns:
        text = DISCLAIMER_RE.sub("", text).strip()
        if len(text) > TURN_CHARS:
            text = text[:TURN_CHARS] + " ..."
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)


def _clean(reply: str) -> str | None:
    """The model's question, or None if the reply is not a usable single question."""
    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    if not lines:
        return None
    text = re.sub(r"^(standalone question|question|rewritten)\s*:\s*", "", lines[0], flags=re.I).strip(" \"'`*")
    return text if 3 <= len(text) <= 500 else None


async def standalone_question(get_llm: Callable[[], BaseChatModel], question: str,
                              turns: list[tuple[str, str]]) -> tuple[str, int, str | None]:
    """(question to research, tokens used, warning or None).

    get_llm is called only when a rewrite is needed, so a first question never builds a model.
    """
    if not looks_like_follow_up(question, bool(turns)):
        return question, 0, None
    try:
        reply = await get_llm().ainvoke([SystemMessage(REWRITE_PROMPT),
                                   HumanMessage(f"Conversation so far:\n{_history(turns)}\n\nLatest message: {question}")])
    except Exception as e:
        return question, 0, f"follow-up rewrite failed ({type(e).__name__}); researching the question as asked"
    rewritten = _clean(_message_text(reply.content))
    if rewritten is None:
        return question, count_tokens(reply), "follow-up rewrite was unusable; researching the question as asked"
    return rewritten, count_tokens(reply), None
