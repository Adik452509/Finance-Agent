"""The sign-in page (shown until a Google account signs in) and the access-denied page."""

import streamlit as st

from frontend import components


def page() -> None:
    components.inject_css()
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.space("large")
        st.image(str(components.ASSETS / "logo_mark.svg"), width=56)
        components.eyebrow("Financial research agent")
        st.title("Research Desk")
        st.markdown("<span class='desk-muted'>Research listed companies from their annual reports, official SEC "
                    "filings and live market data. A team of AI analysts plans the work, calculates with a real "
                    "calculator and checks every figure against its source.</span>", unsafe_allow_html=True)
        st.space("small")
        with st.container(border=True):
            st.markdown(":material/verified: **Every figure cited** to its page, filing or data source")
            st.markdown(":material/history: **Your research is saved** - pick up any chat where you left it")
            st.markdown(":material/lock: **Private to you** - nobody else sees your questions")
        st.space("small")
        st.button("Sign in with Google", icon=":material/login:", type="primary", width="stretch", on_click=st.login)
        st.caption("We use your Google account only to know it's you: your email and name. "
                   "Research summaries are automated and not financial advice.")


def denied(reason: str) -> None:
    components.inject_css()
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.space("large")
        st.title("No access")
        st.warning(reason, icon=":material/lock:")
        st.caption("Ask the owner of this Research Desk to add your Google account, or sign in with another one.")
        st.button("Sign out", icon=":material/logout:", on_click=st.logout)
