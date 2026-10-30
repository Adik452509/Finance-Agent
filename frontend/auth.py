"""Google login for the Research Desk (Phase 6.6), on Streamlit's built-in OpenID Connect support.

Who the user is decides everything downstream - their chats, quota and admin rights - so:
  - only a Google-verified email is accepted, optionally limited to ALLOWED_EMAILS;
  - without login configured the app uses a fixed dev user, and only when APP_ENV=dev
    (settings refuse APP_ENV=production without login);
  - secrets live in .env only; write_secrets() renders Streamlit's .streamlit/secrets.toml
    (git-ignored) from them before Streamlit starts.
"""

import json
from pathlib import Path

from config.settings import settings

GOOGLE_METADATA = "https://accounts.google.com/.well-known/openid-configuration"
SECRETS_PATH = settings.project_root / ".streamlit" / "secrets.toml"


def secrets_toml() -> str:
    """The [auth] section Streamlit's st.login() reads (json.dumps gives valid TOML strings)."""
    values = {
        "redirect_uri": settings.public_app_url.rstrip("/") + "/oauth2callback",
        "cookie_secret": settings.auth_cookie_secret,
        "client_id": settings.google_oauth_client_id,
        "client_secret": settings.google_oauth_client_secret,
        "server_metadata_url": GOOGLE_METADATA,
    }
    lines = ["# Generated from .env by frontend/auth.py - do not edit, do not commit.", "[auth]"]
    lines += [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    return "\n".join(lines) + "\n"


def write_secrets(path: Path | None = None) -> bool:
    """Write .streamlit/secrets.toml if Google login is configured; returns whether it did."""
    path = path or SECRETS_PATH
    if not settings.google_login_configured:
        path.unlink(missing_ok=True)  # never leave stale credentials behind
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secrets_toml(), encoding="utf-8")
    return True


def allowed(email: str) -> bool:
    allow = {e.strip().lower() for e in settings.allowed_emails.split(",") if e.strip()}
    return not allow or email.lower() in allow


def resolve_identity(user) -> tuple[str | None, str | None, str | None]:
    """(email, name, problem) from Streamlit's st.user - or the dev user when login is off.

    problem is None when the person may use the app, else the reason they may not.
    """
    if not settings.google_login_configured:
        if settings.app_env != "dev":
            return None, None, "Sign-in is not configured."
        return settings.dev_user_email, settings.dev_user_name, None
    if not user.get("is_logged_in"):
        return None, None, "signed_out"
    email = (user.get("email") or "").strip().lower()
    if not email or user.get("email_verified") is False:
        return None, None, "Your Google account has no verified email address."
    if not allowed(email):
        return email, None, f"{email} is not on this app's access list."
    return email, user.get("name") or email.split("@")[0], None
