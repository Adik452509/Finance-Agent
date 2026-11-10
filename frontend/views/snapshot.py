"""The Company snapshot page: the facts that help someone think about a company - never a verdict.

Built for "should I invest in X?". The desk does not answer that question (it depends on the
person, and in India specific buy/sell calls need SEBI registration). It shows the evidence:
growth and margin charts, strengths and things to watch - each with the rule that produced it
and its source - valuation in plain words, recent news, the company's own risk factors, and
questions to ask yourself. Every number comes from the backend's data tools; no AI writes it.
"""

import altair as alt
import pandas as pd
import streamlit as st

from frontend import components, notes
from frontend.api_client import ApiError, BackendDown
from frontend.session import api

EXAMPLES = ["MSFT", "AAPL", "RELIANCE.NS", "TCS.NS", "7203.T"]


def _pick_example() -> None:
    choice = st.session_state.get("snapshot_example")
    if choice:
        st.session_state["snapshot_ticker"] = choice


@st.cache_data(ttl=600, show_spinner=False)
def _load(ticker: str, _user: str) -> dict:
    return api().snapshot(ticker)


def header(snap: dict) -> None:
    price = snap["price"]
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.subheader(snap["company"], anchor=False)
        st.badge(snap["ticker"], icon=":material/sell:", color="gray")
    cells = st.columns(3)
    value = price.get("value")
    cells[0].metric("Share price", f"{value:,.2f} {price.get('currency') or ''}".strip() if value is not None else "n/a")
    cells[1].metric("Market value", price.get("market_cap") or "n/a")
    cells[2].metric("P/E (last 12 months)", f"{price['pe']:.1f}" if price.get("pe") else "n/a",
                    help="Share price divided by the last year's earnings per share.")
    st.caption(f":material/schedule: {price.get('source', '')}")


def kpi_tiles(snap: dict) -> None:
    # The number big, its unit small beneath (seen live: "331.84 B USD" was cut to "331.84 B ..." in a tile).
    with st.container(key="kpis"):
        for column, kpi in zip(st.columns(4), snap["kpis"]):
            number, unit = notes.split_amount(kpi["value"] or "n/a")
            with column.container(border=True, height="stretch"):
                st.metric(kpi["label"], number)
                st.caption(" · ".join(x for x in (unit, kpi["context"]) if x))


def growth_chart(snap: dict) -> None:
    years = snap["years"]
    divisor, unit = notes.chart_scale(years, snap["currency"])
    rows = [{"Year": y["label"], "Measure": measure, "Amount": y[field] / divisor}
            for y in years for measure, field in (("Revenue", "revenue"), ("Net income", "net_income"))
            if y.get(field) is not None]
    if not rows:
        st.caption("No revenue or profit figures to chart.")
        return
    order = [y["label"] for y in years]
    # Two series: a legend, a fixed series order (palette slots 1 and 2), thin bars rounded at the data end.
    chart = (alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4)
             .encode(x=alt.X("Year:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.35),
                             axis=alt.Axis(labelAngle=0, ticks=False)),
                     xOffset=alt.XOffset("Measure:N", sort=["Revenue", "Net income"]),
                     y=alt.Y("Amount:Q", title=unit, axis=alt.Axis(grid=True, gridOpacity=0.4, tickCount=5,
                                                                  labelExpr="format(datum.value, ',')")),
                     color=alt.Color("Measure:N", sort=["Revenue", "Net income"],
                                     legend=alt.Legend(orient="top", title=None)),
                     tooltip=[alt.Tooltip("Year:N"), alt.Tooltip("Measure:N"),
                              alt.Tooltip("Amount:Q", title=unit, format=",.2f")])
             .properties(height=280).configure_view(strokeWidth=0))
    st.altair_chart(chart, width="stretch")


def margin_chart(snap: dict) -> None:
    years = snap["years"]
    rows = [{"Year": y["label"], "Margin": name, "Percent": y[field]}
            for y in years for name, field in (("Net margin", "net_margin"), ("Operating margin", "operating_margin"))
            if y.get(field) is not None]
    if not rows:
        st.caption("No margin figures to chart.")
        return
    order = [y["label"] for y in years]
    chart = (alt.Chart(pd.DataFrame(rows)).mark_line(point=alt.OverlayMarkDef(size=50), strokeWidth=2)
             .encode(x=alt.X("Year:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, ticks=False)),
                     y=alt.Y("Percent:Q", title="% of revenue", axis=alt.Axis(grid=True, gridOpacity=0.4, tickCount=5,
                                                                               labelExpr="datum.value + '%'")),
                     color=alt.Color("Margin:N", sort=["Net margin", "Operating margin"],
                                     legend=alt.Legend(orient="top", title=None)),
                     tooltip=[alt.Tooltip("Year:N"), alt.Tooltip("Margin:N"),
                              alt.Tooltip("Percent:Q", title="% of revenue", format=".1f")])
             .properties(height=280).configure_view(strokeWidth=0))
    st.altair_chart(chart, width="stretch")


def points(items: list[dict], icon: str, empty: str) -> None:
    if not items:
        st.caption(empty)
        return
    for item in items:
        with st.container(border=True):
            st.markdown(f"{icon} **{item['title']}**")
            st.markdown(item["detail"])
            st.caption(f"{item['rule']} · {item['source']}")


def news_and_risks(snap: dict) -> None:
    risk = snap.get("risk_factors")
    if risk and risk.get("citation"):
        st.markdown(f":material/description: The company's own **risk factors** are in its annual report: "
                    f"{notes.md_escape(risk['citation'])}. Ask about them on the Research page for a cited summary.")
    else:
        st.caption(":material/description: This company's annual report is not in the library, so its own risk "
                   "factors are not shown here. An admin can add the report from the Admin page.")
    news = snap.get("news") or []
    if not news:
        st.caption(":material/newspaper: No news articles found for the last two weeks.")
        return
    components.eyebrow("Recent news · last 14 days")
    for item in news:
        title = notes.md_escape(item.get("title")) or "Untitled article"
        meta = " · ".join(x for x in (item.get("source"), (item.get("published_date") or "")[:16]) if x)
        url = item.get("url") or ""
        line = f"[{title}]({url})" if url.startswith(("https://", "http://")) else title
        st.markdown(f"{line}  \n:gray[{notes.md_escape(meta)}]")
    st.caption("Headlines come from news sites, not from the company; read them with care.")


def snapshot_view(snap: dict) -> None:
    header(snap)
    st.info("This page lays out the facts to help you think a company through. It does **not** say whether to "
            "buy or sell - that depends on your goals, your time horizon and how much risk you can take.",
            icon=":material/balance:")
    kpi_tiles(snap)
    st.space("small")
    left, right = st.columns(2)
    with left:
        components.eyebrow("Growth")
        st.markdown("**Revenue and net income by year**")
        growth_chart(snap)
    with right:
        components.eyebrow("Profitability")
        st.markdown("**Margins: how much of each sale is kept**")
        margin_chart(snap)
    growth = snap.get("growth") or {}
    if growth.get("revenue_cagr") is not None:
        st.caption(f"Average revenue growth {growth['cagr_span']}: {growth['revenue_cagr']:.1f}% a year · "
                   f"{snap['source']}")
    else:
        st.caption(snap["source"])

    st.space("small")
    good, watch = st.columns(2)
    with good:
        st.subheader(":material/thumb_up: Strengths", anchor=False)
        points(snap["strengths"], ":green[:material/check_circle:]", "No strengths stood out under the rules used here.")
    with watch:
        st.subheader(":material/visibility: Things to watch", anchor=False)
        points(snap["watch"], ":orange[:material/warning:]", "Nothing flagged under the rules used here - which "
               "does not mean there are no risks.")
    st.caption("Each point comes from a simple, stated rule of thumb. What counts as high or low differs by "
               "industry, so compare with similar companies.")

    st.space("small")
    price_col, context_col = st.columns(2)
    with price_col:
        st.subheader(":material/sell: The price in plain words", anchor=False)
        for sentence in snap.get("valuation") or ["No valuation figures are available for this company."]:
            st.markdown(f"- {sentence}")
    with context_col:
        st.subheader(":material/newspaper: News and risks", anchor=False)
        news_and_risks(snap)

    st.space("small")
    st.subheader(":material/psychology_alt: Questions to ask yourself", anchor=False)
    st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(snap["questions"], 1)))
    if snap.get("warnings"):
        with st.expander(f"Data notes ({len(snap['warnings'])})", icon=":material/info:"):
            for warning in snap["warnings"]:
                st.caption(warning)
    st.caption(f":material/gavel: {snap['disclaimer']}")


def page() -> None:
    components.inject_css()
    components.eyebrow("Think it through")
    st.title("Company snapshot")
    st.caption("Growth, profit, cash, debt, valuation and news for one company - with the reasons behind every "
               "point. Facts to help you decide, never a buy or sell call.")
    ticker = st.session_state.get("snapshot_ticker", "")
    with st.form("snapshot-form", border=False, enter_to_submit=True):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            typed = st.text_input("Ticker", value=ticker, placeholder="e.g. MSFT, RELIANCE.NS, 7203.T", max_chars=20,
                                  help="Use the exchange suffix: .NS for India (NSE), .T for Japan, none for the US.")
            submitted = st.form_submit_button("Show snapshot", icon=":material/search:", type="primary")
    if submitted and typed.strip():
        ticker = st.session_state["snapshot_ticker"] = typed.strip().upper()
    st.pills("Examples", EXAMPLES, key="snapshot_example", on_change=_pick_example, label_visibility="collapsed")
    if not ticker:
        st.info("Pick an example or type a ticker to see its snapshot.", icon=":material/query_stats:")
        return
    try:
        with st.spinner(f"Gathering {ticker}'s statements, price and news…"):
            snap = _load(ticker, st.session_state.get("user_email", ""))
    except ApiError as e:
        icon = ":material/search_off:" if e.status in (404, 422) else ":material/error:"
        st.warning(f"No snapshot for {ticker}: {e.detail}", icon=icon)
        return
    except BackendDown as e:
        st.error(str(e), icon=":material/cloud_off:")
        return
    snapshot_view(snap)
