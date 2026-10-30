"""Per-browser-session state: who is signed in, the API client, and refreshing after changes."""

import streamlit as st

from config.settings import settings
from frontend.api_client import ApiClient, ApiError, BackendDown


def identity() -> tuple[str, str]:
    """(email, name) of the person using the app. Google login replaces this in 6.6."""
    return settings.dev_user_email, settings.dev_user_name


def api() -> ApiClient:
    if "api" not in st.session_state:
        email, name = identity()
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
