"""Per-browser-session state: who is signed in, the API client, and refreshing after changes."""

import streamlit as st

from config.settings import settings
from frontend.api_client import ApiClient, ApiError, BackendDown
from frontend.auth import resolve_identity


def streamlit_user() -> dict:
    """st.user as a plain dict (it's empty, apart from is_logged_in, when nobody is signed in)."""
    try:
        return st.user.to_dict()
    except Exception:  # login not configured: st.user has nothing to offer
        return {"is_logged_in": False}


def identity() -> tuple[str | None, str | None, str | None]:
    """(email, name, problem) of the person using the app - see frontend.auth.resolve_identity."""
    return resolve_identity(streamlit_user())


def api() -> ApiClient:
    """The API client for the signed-in person; rebuilt if the person changes (sign out, sign in)."""
    email, name, _ = identity()
    previous = st.session_state.get("user_email")
    if previous is not None and previous != email:  # another person in this browser: nothing carries over
        for key in ("api", "_me", "conversation_id", "pending_question"):
            st.session_state.pop(key, None)
    if "api" not in st.session_state:
        st.session_state["user_email"] = email
        st.session_state["api"] = ApiClient(settings.api_base_url, email, name, settings.internal_api_token)
    return st.session_state["api"]


def me() -> dict | None:
    """The user's profile and quota for this run (one call per rerun)."""
    if "_me" not in st.session_state:
        try:
            st.session_state["_me"] = api().me()
        except (ApiError, BackendDown):
            st.session_state["_me"] = None
    return st.session_state["_me"]


def refresh() -> None:
    """Re-run the page with fresh data from the backend."""
    st.session_state.pop("_me", None)
    st.rerun()


def open_conversation(conversation_id: str | None) -> None:
    if conversation_id:
        st.session_state["conversation_id"] = conversation_id
    else:
        st.session_state.pop("conversation_id", None)
