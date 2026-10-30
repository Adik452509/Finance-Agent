"""Phase 6.4: follow-up rewriting and how long answers are cached. Offline."""

import asyncio

import pytest
from langchain_core.messages import AIMessage

from src.memory.cache_policy import DAY, HOUR, MINUTE, cache_ttl
from src.memory.followup import looks_like_follow_up, standalone_question
from tests.conftest import scripted

HISTORY = [("user", "What was Apple's net income in fiscal 2025?"),
           ("assistant", "Apple's net income for fiscal 2025 was 112,010 USD millions.\n\nThis is an automated research "
                         "summary of company filings, not financial advice.")]


def rewrite(model_or_error, question="and in 2024?", turns=HISTORY):
    def get_llm():
        if isinstance(model_or_error, Exception):
            raise model_or_error
        return model_or_error
    return asyncio.run(standalone_question(get_llm, question, turns))


@pytest.mark.parametrize("question, expected", [
    ("and in 2024?", True),
    ("What about Microsoft?", True),
    ("How did its operating income change over the same period, and what drove it in the annual report?", True),
    ("What was Toyota's operating income in fiscal 2026 compared with fiscal 2025 in the 20-F filing report?", True),
    ("What were Reliance Industries' consolidated total assets as at 31st March 2025 in the annual report?", False),
])
def test_what_looks_like_a_follow_up(question, expected):
    assert looks_like_follow_up(question, has_history=True) is expected
    assert looks_like_follow_up(question, has_history=False) is False  # no history: nothing to refer to


def test_a_follow_up_is_rewritten():
    model = scripted(AIMessage(content="What was Apple's net income in fiscal 2024?",
                               usage_metadata={"input_tokens": 500, "output_tokens": 15, "total_tokens": 515}))
    assert rewrite(model) == ("What was Apple's net income in fiscal 2024?", 515, None)


def test_a_standalone_question_never_calls_the_model():
    question = "What were Reliance Industries' consolidated total assets as at 31st March 2025 in the annual report?"
    assert rewrite(RuntimeError("must not be called"), question=question) == (question, 0, None)


def test_labels_quotes_and_extra_lines_are_stripped():
    model = scripted(AIMessage(content='Standalone question: "What was Apple\'s net income in fiscal 2024?"\nExplanation: ...'))
    assert rewrite(model)[0] == "What was Apple's net income in fiscal 2024?"


def test_any_failure_falls_back_to_the_original_question():
    question, tokens, warning = rewrite(RuntimeError("Groq is down"))
    assert question == "and in 2024?" and tokens == 0 and "rewrite failed" in warning
    question, _, warning = rewrite(scripted(AIMessage(content="   ")))
    assert question == "and in 2024?" and "unusable" in warning


@pytest.mark.parametrize("question, ttl", [
    ("What is Toyota's current share price?", 15 * MINUTE),
    ("What is the latest news about Microsoft?", 15 * MINUTE),
    ("What were Microsoft's revenue and net income in its most recent quarter?", 6 * HOUR),
    ("By what percentage did Infosys's revenue grow in its latest fiscal year?", 6 * HOUR),
    ("What was Apple's net income in fiscal 2025?", 7 * DAY),
    ("What share of Reliance's 2024-25 revenue came from the Retail segment?", 7 * DAY),
])
def test_answers_are_reused_only_while_their_data_cannot_have_changed(question, ttl):
    assert cache_ttl(question) == ttl
