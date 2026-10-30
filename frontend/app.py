"""Research Desk - the Streamlit frontend (Phase 6.5).

    streamlit run frontend/app.py          # the backend must be running (scripts/run_dev.py starts both)

Talks only to the backend API; never imports the agents. Sidebar: new research, history
grouped by day, today's question meter. Pages: Research, Library, Admin (admins only).
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `streamlit run` puts only frontend/ on the path

from frontend import notes  # noqa: E402
from frontend.api_client import ApiError, BackendDown  # noqa: E402
from frontend.session import api, me, open_conversation  # noqa: E402
from frontend.views import admin, library, research  # noqa: E402

st.set_page_config(page_title="Research Desk · Financial Research Agent", page_icon=":material/query_stats:",
                   layout="wide", initial_sidebar_state="auto")  # auto: collapsed on phones
st.session_state.pop("_me", None)  # fresh profile and quota on every run


def new_research() -> None:
    open_conversation(None)
    st.session_state["go_to_research"] = True


def select(conversation_id: str) -> None:
    open_conversation(conversation_id)
    st.session_state["go_to_research"] = True  # e.g. clicked a chat while on the Library page


def sidebar(profile: dict) -> None:
    with st.sidebar:
        st.button("New research", icon=":material/add:", type="primary", width="stretch", on_click=new_research)
        try:
            conversations = api().conversations()
        except (ApiError, BackendDown):
            conversations = []
        current = st.session_state.get("conversation_id")
        for label, items in notes.group_by_day(conversations):
            st.caption(label)
            for conversation in items[:25]:
                st.button(conversation["title"], key=f"chat-{conversation['id']}", width="stretch",
                          type="secondary" if conversation["id"] == current else "tertiary",
                          on_click=select, args=(conversation["id"],),
                          icon=":material/chat_bubble:" if conversation["id"] == current else None)
        st.space("medium")
        quota, left = profile["daily_quota"], profile["questions_left_today"]
        if profile["is_admin"]:
            st.caption(":material/shield_person: Admin · no daily limit")
        else:
            st.progress(1 - left / quota if quota else 1.0,
                        text=f"{left} of {quota} questions left today")
            st.caption("Resets at midnight UTC · reused answers are free")
        st.caption(f":material/account_circle: {profile.get('name') or profile['email']}")


def backend_down(error: Exception) -> None:
    st.title("Research Desk")
    st.error(f"The research service is not running. {error}", icon=":material/cloud_off:")
    st.markdown("Start it with `python scripts/run_dev.py` (backend + this app), or run the backend alone with "
                "`uvicorn src.api.app:app --host 127.0.0.1 --port 8000`.")
    st.button("Try again", icon=":material/refresh:")


def main() -> None:
    try:
        api().me()
    except BackendDown as e:
        backend_down(e)
        return
    except ApiError as e:
        st.error(f"The research service refused this app ({e.status}): {e.detail}", icon=":material/lock:")
        return
    profile = me()
    # The wordmark is light text for the navy sidebar; when the sidebar is collapsed (phones) the
    # header is light, so show only the mark there (seen live: the wordmark was invisible).
    assets = Path(__file__).parent / "assets"
    st.logo(str(assets / "logo.svg"), size="large", icon_image=str(assets / "logo_mark.svg"))
    pages = [st.Page(research.page, title="Research", icon=":material/query_stats:", url_path="research", default=True),
             st.Page(library.page, title="Library", icon=":material/library_books:", url_path="library")]
    if profile["is_admin"]:
        pages.append(st.Page(admin.page, title="Admin", icon=":material/admin_panel_settings:", url_path="admin"))
    page = st.navigation(pages, position="sidebar")
    if st.session_state.pop("go_to_research", False) and page.url_path != "research":
        st.switch_page(pages[0])
    sidebar(profile)
    page.run()


main()
