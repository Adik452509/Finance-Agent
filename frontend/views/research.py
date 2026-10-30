"""The Research page: the desk (when empty), the conversation, and live research."""

import streamlit as st

from frontend import components, notes
from frontend.api_client import ApiError, BackendDown
from frontend.session import api, me, refresh

EXAMPLES = [
    ("Annual report", ":material/description:", "What was the revenue of Microsoft's Intelligent Cloud segment in fiscal 2026, and what share of total revenue was that?"),
    ("Growth", ":material/trending_up:", "By what percentage did Reliance's consolidated revenue from operations grow from 2023-24 to 2024-25?"),
    ("Latest quarter · SEC", ":material/account_balance:", "What were Microsoft's revenue and net income in its most recent quarter, and how did revenue change from a year earlier?"),
    ("Margins", ":material/percent:", "What was Apple's net income as a percentage of total net sales in fiscal 2025?"),
    ("Market snapshot", ":material/candlestick_chart:", "What is Toyota's current share price and market capitalisation?"),
    ("News", ":material/newspaper:", "What is the latest news about Reliance Industries?"),
]


def _ask(question: str) -> None:
    st.session_state["pending_question"] = question


def _feedback(message_id: str, key: str) -> None:
    value = st.session_state.get(key)
    if value is None:
        return
    try:
        api().feedback(message_id, 1 if value == 1 else -1)
        st.toast("Thanks - your feedback helps improve the answers.", icon=":material/thumb_up:")
    except (ApiError, BackendDown) as e:
        st.toast(f"Feedback not saved: {e}", icon=":material/error:")


def desk() -> None:
    """The empty state: what the desk can do, and where to start.

    Everything sits in the "desk" container: its stale elements are hidden by CSS (components.CSS),
    so it vanishes the moment a question starts instead of staying faded under the live research.
    """
    with st.container(key="desk"):
        _desk()


def _desk() -> None:
    with st.container(key="desk-hero"):
        components.eyebrow("Financial research agent")
        st.title("What do you want to research?")
        st.markdown("<span class='desk-muted'>A team of AI analysts plans the research, reads annual reports and "
                    "official filings, calculates with a real calculator and checks every figure against its source "
                    "before you see it.</span>", unsafe_allow_html=True)
    st.space("small")
    companies = _companies()
    coverage = st.columns(3)
    with coverage[0].container(border=True, height="stretch"):
        st.markdown(":material/description: **Annual reports**")
        st.caption(f"{len(companies)} indexed, cited to the page: " + ", ".join(c["ticker"] for c in companies[:6]))
    with coverage[1].container(border=True, height="stretch"):
        st.markdown(":material/account_balance: **Official SEC data**")
        st.caption("Live 10-K / 10-Q / 20-F figures for 10,000+ US-listed companies")
    with coverage[2].container(border=True, height="stretch"):
        st.markdown(":material/candlestick_chart: **Market & news**")
        st.caption("Prices, annual and quarterly statements for any listed company, plus recent news")
    st.space("small")
    components.eyebrow("Start with an example")
    with st.container(key="examples"):
        for row in range(0, len(EXAMPLES), 2):
            for column, (kind, icon, question) in zip(st.columns(2), EXAMPLES[row:row + 2]):
                column.button(f"**{kind}**  \n{question}", icon=icon, width="stretch", on_click=_ask, args=(question,),
                              key=f"example-{kind}")


def _companies() -> list[dict]:
    try:
        return _cached_companies(st.session_state.get("user_email", ""))
    except (ApiError, BackendDown):
        return []


@st.cache_data(ttl=120, show_spinner=False)
def _cached_companies(_user: str) -> list[dict]:
    return api().companies()


def conversation_header(conversation: dict) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader(conversation["title"], anchor=False)
        with st.popover("", icon=":material/more_horiz:"):
            title = st.text_input("Rename", value=conversation["title"], max_chars=200)
            if st.button("Save name", icon=":material/edit:", width="stretch") and title.strip():
                api().rename(conversation["id"], title.strip())
                refresh()
            if st.button("Delete research", icon=":material/delete:", type="primary", width="stretch"):
                api().delete(conversation["id"])
                st.session_state.pop("conversation_id", None)
                refresh()


def history(conversation: dict) -> None:
    previous_user = None
    for message in conversation["messages"]:
        if message["role"] == "user":
            previous_user = message
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(message["content"])
        else:
            seconds = notes.seconds_between(previous_user["created_at"], message["created_at"]) if previous_user else None
            with st.chat_message("assistant", avatar=":material/query_stats:"):
                components.answer_card(message, seconds, on_feedback=_feedback)


def research(conversation_id: str, question: str) -> None:
    """Show the question, stream the research live, then reload the saved conversation."""
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/query_stats:"):
        timeline = components.LiveTimeline()
        answer = None
        try:
            for event in api().ask(conversation_id, question):
                if event.get("type") == "answer":
                    answer = event["message"]
                else:
                    timeline.event(event)
        except ApiError as e:
            timeline.finish(None)
            st.warning(e.detail, icon=":material/block:" if e.status == 429 else ":material/error:")
            return
        except BackendDown as e:
            timeline.finish(None)
            st.error(str(e), icon=":material/cloud_off:")
            return
        timeline.finish(answer)
    refresh()  # redraw from the saved conversation: title, history and quota all update


def page() -> None:
    components.inject_css()
    profile = me()
    conversation_id = st.session_state.get("conversation_id")
    conversation = None
    if conversation_id:
        try:
            conversation = api().conversation(conversation_id)
        except ApiError:
            st.session_state.pop("conversation_id", None)  # deleted, or not yours
            conversation_id = None

    pending = st.session_state.pop("pending_question", None)
    left = profile["questions_left_today"] if profile else 0
    typed = st.chat_input("Ask about a company's results, segments, growth, latest quarter or news…",
                          disabled=left <= 0, max_chars=2000)
    question = typed or pending

    if conversation and (conversation["messages"] or question):
        conversation_header(conversation)
        history(conversation)
    elif not question:
        desk()
    if left <= 0:
        st.info("You've used today's questions. They reset at midnight UTC - previous research stays available.",
                icon=":material/hourglass_empty:")
    if question:
        if not conversation_id:
            conversation_id = api().new_conversation()["id"]
            st.session_state["conversation_id"] = conversation_id
        research(conversation_id, question)
