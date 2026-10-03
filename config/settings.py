"""Central configuration for the Financial Research Analyst Agent.

This is the ONLY module that reads the environment / .env file.
Every other module does `from config.settings import settings` so that:
  - a missing key fails loudly here, at startup, instead of deep inside an agent
  - swapping LLM providers is a one-line change in .env, not a code change
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# settings.py lives in <project>/config/, so the root is two levels up.
# Deriving it from __file__ keeps the project portable (no hard-coded C:\ paths).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

Provider = Literal["gemini", "openai", "groq", "anthropic"]

# Which model to use for each provider, per role.
# "default" = cheap workhorse (Planner, Researcher, Writer)
# "reasoning" = stronger/pricier, used sparingly (Analyst, Critic)
MODEL_MAP: dict[str, dict[str, str]] = {
    # Free tier has no Pro quota (limit 0), so both roles use Flash.
    # On a paid key, set REASONING_MODEL_OVERRIDE=gemini-3.1-pro-preview in .env.
    "gemini": {
        "default": "gemini-3.8-flash",
        "reasoning": "gemini-3.8-flash",
    },
    "openai": {
        "default": "gpt-4o-mini",
        "reasoning": "gpt-4o",
    },
    "groq": {
        "default": "llama-3.3-70b-versatile",
        "reasoning": "llama-3.3-70b-versatile",
    },
    "anthropic": {
        "default": "claude-haiku-4-5-20251001",
        "reasoning": "claude-sonnet-5",
    },
}

# Each provider needs exactly one key; used to give a clear error message.
PROVIDER_KEY_NAMES: dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class Settings(BaseSettings):
    """Typed, validated configuration loaded from .env + environment variables.

    Field names map to env vars case-insensitively: `llm_provider` <- LLM_PROVIDER.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # don't crash on unrelated env vars on the machine
    )

    # --- Which LLM provider is active -------------------------------------
    llm_provider: Provider = "gemini"

    # --- API keys. All optional: you only hold the ones you actually use. --
    # str | None (not str) so the app starts with just one provider configured.
    google_api_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    tavily_api_key: str | None = None
    langsmith_api_key: str | None = None

    # --- Optional model overrides -----------------------------------------
    # Leave empty in .env to use the MODEL_MAP defaults above.
    default_model_override: str | None = None
    reasoning_model_override: str | None = None

    # --- Paths -------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"

    # --- Derived values ----------------------------------------------------

    @property
    def default_model(self) -> str:
        """Model name for routine steps, honouring an .env override."""
        return self.default_model_override or MODEL_MAP[self.llm_provider]["default"]

    @property
    def reasoning_model(self) -> str:
        """Model name for analysis/critique steps, honouring an .env override."""
        return self.reasoning_model_override or MODEL_MAP[self.llm_provider]["reasoning"]

    @property
    def active_api_key(self) -> str | None:
        """The API key belonging to the currently selected provider."""
        attr = PROVIDER_KEY_NAMES[self.llm_provider].lower()
        return getattr(self, attr)

    # --- Helpers -----------------------------------------------------------

    def masked_keys(self) -> dict[str, str]:
        """Report which secrets are configured WITHOUT revealing any value.

        Never print `settings` directly - pydantic would dump the raw keys.
        Use this instead when checking configuration.
        """
        names = [
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "GROQ_API_KEY",
            "ANTHROPIC_API_KEY",
            "TAVILY_API_KEY",
            "LANGSMITH_API_KEY",
        ]
        return {
            name: "set" if getattr(self, name.lower()) else "missing"
            for name in names
        }

    def require_provider_key(self) -> str:
        """Return the active provider's key, or raise a clear, actionable error.

        Call this before building an LLM client - not at import time, so that
        tooling (tests, linting) can import settings without any keys present.
        """
        key = self.active_api_key
        if not key:
            env_name = PROVIDER_KEY_NAMES[self.llm_provider]
            raise RuntimeError(
                f"LLM_PROVIDER is '{self.llm_provider}' but {env_name} is not set. "
                f"Add it to {self.project_root / '.env'} (never to .env.example)."
            )
        return key

    def ensure_dirs(self) -> None:
        """Create the local data folders if they don't exist yet."""
        for path in (self.data_dir, self.raw_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)


# The single shared instance imported across the project.
settings = Settings()
