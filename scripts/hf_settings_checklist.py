"""List what to enter in the Hugging Face Space's "Variables and secrets" - names only, never values.

    python scripts/hf_settings_checklist.py --space-url https://aditya09106-finance-research-agent.hf.space

Copy each value from your .env into the Space (Settings > Variables and secrets). Secrets are
hidden after saving; variables stay visible, so only non-sensitive settings go there.
"""

import argparse
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]

SECRETS = {  # name -> why the app needs it
    "GROQ_API_KEY": "the agents' AI model",
    "GOOGLE_API_KEY": "Gemini embeddings (searching filings)",
    "TAVILY_API_KEY": "news search",
    "DATABASE_URL": "Supabase: chats, cache, filings (session pooler URI, includes the DB password)",
    "GOOGLE_OAUTH_CLIENT_ID": "Google sign-in",
    "GOOGLE_OAUTH_CLIENT_SECRET": "Google sign-in",
    "AUTH_COOKIE_SECRET": "signs the login cookie",
    "INTERNAL_API_TOKEN": "lets only the app call the backend",
}
OPTIONAL_SECRETS = {"EDINET_API_KEY": "Japan filings (only if you enable the EDINET server)"}
VARIABLES = {
    "LLM_PROVIDER": "groq",
    "SEC_USER_AGENT": "from .env",
    "ADMIN_EMAILS": "from .env",
    "PUBLIC_APP_URL": "the Space's direct address (below)",
    "ALLOWED_EMAILS": "optional: limit access (empty = any Google account you added as a test user)",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space-url", required=True)
    args = parser.parse_args()
    env = dotenv_values(ROOT / ".env")
    missing = []

    print("SECRETS  (Settings > Variables and secrets > New secret) - copy the value from .env:")
    for name, why in SECRETS.items():
        ok = bool(env.get(name))
        missing += [] if ok else [name]
        print(f"  [{'x' if ok else ' '}] {name:28} {why}{'' if ok else '   <- MISSING in .env'}")
    for name, why in OPTIONAL_SECRETS.items():
        print(f"  [ ] {name:28} optional: {why}")

    print("\nVARIABLES  (Settings > Variables and secrets > New variable):")
    for name, hint in VARIABLES.items():
        value = args.space_url if name == "PUBLIC_APP_URL" else (env.get(name) or "") if hint == "from .env" else hint
        shown = value if name in ("LLM_PROVIDER", "PUBLIC_APP_URL", "ADMIN_EMAILS", "SEC_USER_AGENT") else hint
        print(f"  {name:28} = {shown}")
    print("\nAlready set by the Docker image (do not add): APP_ENV=production, FILINGS_STORE=postgres, PORT=7860")
    print(f"\nGoogle Cloud > Google Auth Platform > Clients > your client > add redirect URI:\n  "
          f"{args.space_url.rstrip('/')}/oauth2callback")
    if missing:
        print(f"\nMissing in .env: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
