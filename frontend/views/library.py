"""The Library page: what the desk can research, from which sources, how fresh."""

import pandas as pd
import streamlit as st

from frontend import components
from frontend.api_client import ApiError, BackendDown
from frontend.session import api

SOURCES = pd.DataFrame([
    {"Source": "Annual reports (indexed)", "Coverage": "The companies listed below", "Freshness": "Latest annual report",
     "Best for": "Segments, notes, risks, strategy, exact reported line items - cited to the page"},
    {"Source": "SEC EDGAR", "Coverage": "10,000+ US-listed companies (and foreign ones with a US listing)",
     "Freshness": "Same day as each 10-K / 10-Q / 20-F", "Best for": "Official annual and quarterly figures"},
    {"Source": "Yahoo Finance", "Coverage": "Any listed company (US, India, Japan, Hong Kong, China…)",
     "Freshness": "Prices live; statements as Yahoo updates them", "Best for": "Price, market cap, P/E, yearly and quarterly statements"},
    {"Source": "News (Tavily)", "Coverage": "Global news sites", "Freshness": "Up to the last 30 days",
     "Best for": "Recent events, deals, management changes"},
])


def page() -> None:
    components.inject_css()
    components.eyebrow("Coverage")
    st.title("Library")
    st.caption("Every answer is built from these sources - and says which ones it used.")
    st.dataframe(SOURCES, hide_index=True, width="stretch")

    st.subheader("Indexed annual reports", anchor=False)
    try:
        companies = api().companies()
    except (ApiError, BackendDown) as e:
        st.error(f"Could not load the library: {e}", icon=":material/cloud_off:")
        return
    if not companies:
        st.info("No annual reports are indexed yet.", icon=":material/inventory_2:")
        return
    table = pd.DataFrame([{"Company": c["company"], "Ticker": c["ticker"], "Document": c["doc_type"],
                           "Fiscal year end": c["fiscal_year_end"], "Filed": c.get("filed") or "",
                           "Added by": {"sec": "SEC download", "edinet": "EDINET download", "url": "PDF link"}.get(c.get("source"), "Hand-picked")}
                          for c in companies])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Ticker": st.column_config.TextColumn(width="small")})
    st.caption(f"{len(companies)} reports. Ask about any other company too: official SEC figures and market data "
               "cover it, and an admin can add its annual report.")
