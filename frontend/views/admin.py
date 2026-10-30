"""The Admin page: system status, add a company's annual report, follow the jobs."""

import pandas as pd
import streamlit as st

from frontend import components
from frontend.api_client import ApiError, BackendDown
from frontend.session import api, me

STATUS_COLORS = {"queued": "gray", "running": "blue", "done": "green", "failed": "red"}
STATUS_ICONS = {"queued": ":material/schedule:", "running": ":material/progress_activity:",
                "done": ":material/check_circle:", "failed": ":material/error:"}


def system_status() -> None:
    try:
        status = api().status()
    except (ApiError, BackendDown) as e:
        st.error(f"Backend unreachable: {e}", icon=":material/cloud_off:")
        return
    columns = st.columns(4)
    columns[0].metric("Research team", "Ready" if status["team_ready"] else "Starting", border=True)
    columns[1].metric("Tools", status["tools"], border=True)
    columns[2].metric("Questions running", status["questions_running"], border=True)
    columns[3].metric("Filings store", status["filings_store"].capitalize(), border=True)
    if status.get("unavailable_servers"):
        st.caption("Unavailable data sources: " + ", ".join(f"{k} ({v[:60]})" for k, v in status["unavailable_servers"].items()))
    if status.get("team_error"):
        st.error(status["team_error"], icon=":material/error:")


def add_company_form() -> None:
    with st.form("add-company", border=True):
        st.markdown("**Add a company's latest annual report**")
        st.caption("US-listed: latest 10-K / 20-F from the SEC. Tokyo-listed: EDINET (needs its key). "
                   "Anything else: give a PDF link below. Takes ~8 minutes on the free embedding tier.")
        left, right = st.columns([2, 1])
        ticker = left.text_input("Ticker", placeholder="NVDA, GOOGL, INFY.NS…", max_chars=20)
        max_chunks = right.number_input("Chunk budget", min_value=10, max_value=600, value=250, step=10,
                                        help="Pages beyond the budget are skipped (Gemini free tier: ~1,000 a day).")
        with st.expander("Not US- or Tokyo-listed? Use a PDF link", icon=":material/link:"):
            pdf_url = st.text_input("PDF link (https://…)")
            c1, c2 = st.columns(2)
            company = c1.text_input("Company name")
            market = c2.text_input("Market", placeholder="India")
            c3, c4, c5 = st.columns(3)
            doc_type = c3.text_input("Document type", value="Annual Report")
            fiscal_year_end = c4.text_input("Fiscal year end", placeholder="2026-03-31")
            currency = c5.text_input("Currency", placeholder="INR", max_chars=3)
        submitted = st.form_submit_button("Queue job", icon=":material/add_circle:", type="primary")
    if not submitted:
        return
    if not ticker.strip():
        st.warning("Enter a ticker.", icon=":material/warning:")
        return
    body = {"ticker": ticker.strip().upper(), "max_chunks": int(max_chunks)}
    if pdf_url.strip():
        body["pdf"] = {"url": pdf_url.strip(), "company": company, "market": market, "doc_type": doc_type,
                       "fiscal_year_end": fiscal_year_end, "currency": currency.upper()}
    try:
        job = api().add_company(body)
        st.toast(f"Queued {body['ticker']}", icon=":material/schedule:")
        st.session_state["last_job"] = job["id"]
    except ApiError as e:
        st.error(e.detail, icon=":material/error:")


@st.fragment(run_every=5)
def jobs_table() -> None:
    """Refreshes itself every 5 s, so a running job's progress moves on its own."""
    try:
        jobs = api().jobs()
    except (ApiError, BackendDown) as e:
        st.error(str(e), icon=":material/cloud_off:")
        return
    if not jobs:
        st.caption("No jobs yet.")
        return
    for job in jobs[:8]:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.badge(job["status"].capitalize(), icon=STATUS_ICONS.get(job["status"]),
                         color=STATUS_COLORS.get(job["status"], "gray"))
                st.markdown(f"**{job['args'].get('ticker', job['kind'])}**")
                st.caption(pd.Timestamp(job["created_at"]).strftime("%d %b %H:%M UTC"))
            st.caption(job["progress"] or "Waiting for the worker…")


def page() -> None:
    components.inject_css()
    profile = me()
    if not profile or not profile["is_admin"]:
        st.warning("This page is for admins.", icon=":material/lock:")
        return
    components.eyebrow("Operations")
    st.title("Admin")
    system_status()
    st.space("small")
    left, right = st.columns([3, 2], gap="large")
    with left:
        add_company_form()
        if st.button("Reload research team", icon=":material/restart_alt:",
                     help="Restart the data servers so newly added companies are searchable. "
                          "Happens automatically after a job, once no question is running."):
            st.toast(api().reload()["reload"], icon=":material/restart_alt:")
    with right:
        st.markdown("**Jobs**")
        jobs_table()
