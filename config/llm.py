"""LLM provider factory.

The rest of the project gets a chat model ONLY through `get_chat_model()`.
Agents never import a provider class directly, so switching providers is a
one-line change in .env (LLM_PROVIDER=...) and no agent code changes.

    from config.llm import get_chat_model

    llm = get_chat_model()             # Planner / Researcher / Writer
    llm = get_chat_model("reasoning")  # Analyst / Critic
"""

from functools import lru_cache
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import settings

Role = Literal["default", "reasoning"]
VALID_ROLES: tuple[str, ...] = ("default", "reasoning")


@lru_cache(maxsize=None)
def get_chat_model(role: Role = "default", temperature: float = 0.0) -> BaseChatModel:
    """Return a chat model for the active provider and the requested role.

    Args:
        role: "default" for routine steps (cheap model), "reasoning" for
            analysis/critique steps (stronger model).
        temperature: 0.0 by default. Financial output must be reproducible -
            the same question should give the same numbers every run, or the
            Critic ends up chasing differences that are just randomness.

    The result is cached per (role, temperature), so every agent shares one
    client per role. Consequence: changes to .env are picked up only on the
    next process start, not mid-run.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Unknown role '{role}'. Expected one of: {', '.join(VALID_ROLES)}.")

    model = settings.default_model if role == "default" else settings.reasoning_model
    api_key = settings.require_provider_key()
    provider = settings.llm_provider

    # Imports live inside each branch (lazy imports): only the active
    # provider's SDK is loaded, which keeps startup fast and means an unused
    # provider package can be removed without breaking this module.
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=api_key, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=api_key, temperature=temperature)

    # Unreachable while settings.llm_provider is a Literal, but guards against
    # someone adding a provider to settings without adding a branch here.
    raise ValueError(f"No factory branch for provider '{provider}'.")
