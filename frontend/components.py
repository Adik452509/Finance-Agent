"""Building blocks of the Research Desk: answer cards, live research timeline, sources, trail, usage."""

import time

import altair as alt
import pandas as pd
import streamlit as st

from frontend import notes

AGENT_ICONS = {"memory": ":material/history:", "cache": ":material/bolt:", "planner": ":material/checklist:",
               "researcher": ":material/manage_search:", "supervisor": ":material/alt_route:",
               "analyst": ":material/calculate:", "writer": ":material/edit_note:", "critic": ":material/fact_check:",
               "finalize": ":material/task_alt:"}
KIND_ICONS = {"filings": ":material/description:", "sec": ":material/account_balance:",
              "edinet": ":material/account_balance:", "market": ":material/candlestick_chart:",
              "news": ":material/newspaper:", "any": ":material/travel_explore:"}
AGENT_ORDER = ["memory", "planner", "researcher", "verifier", "analyst", "writer", "critic"]

CSS = """
<style>
/* Figures: proportional digits for tile values; aligned digits only in tables. */
[data-testid="stMetricValue"] { font-weight: 650; letter-spacing: -0.01em; }
[data-testid="stMetricLabel"] p { font-size: 0.82rem; }
/* Example-question cards: left-aligned, two-line, like research prompts. */
.st-key-examples button { justify-content: flex-start; text-align: left; min-height: 4.2rem; }
.st-key-examples button > div { justify-content: flex-start; width: 100%; }
.st-key-examples button [data-testid="stMarkdownContainer"] { text-align: left; }
.st-key-examples button p { white-space: normal; line-height: 1.35; text-align: left; }
/* The research note: a little air between its parts. */
.st-key-desk-hero h1 { font-size: 2.1rem; margin-bottom: 0.2rem; }
.desk-eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; font-weight: 650;
                opacity: 0.65; margin-bottom: 0.25rem; }
.desk-muted { opacity: 0.7; font-size: 0.85rem; }
/* Streamlit keeps the previous page's elements on screen, faded ("stale"), until a run ends - and a
   research run lasts 10-60 s. The desk must not linger under the live research: hide its stale parts. */
.st-key-desk [data-stale="true"] { display: none; }
[data-testid="stSidebar"] .stButton button { justify-content: flex-start; }
[data-testid="stSidebar"] .stButton button p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def eyebrow(text: str) -> None:
    st.html(f'<div class="desk-eyebrow">{text}</div>')


# --- the answer card ---------------------------------------------------------------------------


def figure_tiles(trace: dict | None) -> None:
    figures = notes.key_figures(trace)
    if not figures:
        return
    # Always a 4-column grid: one figure keeps a tile's size instead of stretching across the page.
    # Inside each tile: the label, the number big, then the unit and period small.
    for column, figure in zip(st.columns(4), figures):
        with column.container(border=True, height="stretch"):
            long_label = len(figure["label"]) > 30
            st.metric(figure["label"], figure["value"], help=figure["label"] if long_label else None)
            details = " · ".join(x for x in (figure["unit"], figure["context"]) if x)
            if details:
                st.caption(details)


def badge_row(message: dict, seconds: float | None) -> None:
    label, icon, color = notes.verification(message)
    with st.container(horizontal=True, gap="small"):
        st.badge(label, icon=icon, color=color)
        for kind, kind_icon in notes.source_kinds(message):
            st.badge(kind, icon=kind_icon, color="gray")
        if message.get("from_cache"):
            st.badge("Instant: reused a recent answer", icon=":material/bolt:", color="violet")
        facts = []
        if seconds is not None and not message.get("from_cache"):
            facts.append(f"{seconds:.0f} s")
        if message.get("tokens"):
            facts.append(f"{message['tokens']:,} tokens")
        if facts:
            st.caption(" · ".join(facts))


def _source_row(item: dict) -> None:
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.badge(item["kind"], icon=item["icon"], color="gray")
        st.markdown(item["label"] + (f" :gray[· `{item['ref']}`]" if item["ref"] else ""))


def sources_tab(message: dict) -> None:
    items = notes.sources(message)
    cited = [i for i in items if i["cited"]]
    read = [i for i in items if not i["cited"]]
    if not items:
        st.caption("No external sources: this answer was not based on documents or data.")
        return
    if cited:
        eyebrow("Cited in the answer")
        for item in cited:
            _source_row(item)
    if read:
        with st.expander(f"Also read during research ({len(read)})", icon=":material/menu_book:"):
            for item in read:
                _source_row(item)


def trail_tab(trace: dict | None) -> None:
    trace = trace or {}
    plan = trace.get("plan") or {}
    if trace.get("error"):
        st.error(trace["error"], icon=":material/error:")
    if plan.get("steps"):
        eyebrow("Plan")
        for step in plan["steps"]:
            ticker = f" · `{step['ticker']}`" if step.get("ticker") else ""
            st.markdown(f"{KIND_ICONS.get(step['kind'], '')} **{step['kind'].upper()}**{ticker} — {step['goal']}")
        if plan.get("needs_calculation"):
            st.markdown(f":material/calculate: **CALCULATION** — {plan.get('calculation') or 'yes'}")
    for finding in trace.get("findings") or []:
        eyebrow(f"Fact notes · step {finding.get('step_id', '')}")
        st.code(finding.get("facts", ""), language=None, wrap_lines=True)
    if trace.get("analysis"):
        eyebrow("Analyst")
        st.code(trace["analysis"], language=None, wrap_lines=True)
    calls = trace.get("tool_calls") or []
    if calls:
        eyebrow(f"Tool calls ({len(calls)})")
        table = pd.DataFrame([{"Agent": c.get("agent"), "Tool": c.get("name"), "OK": c.get("ok") is not False,
                               "Seconds": c.get("seconds"), "Arguments": ", ".join(f"{k}={v}" for k, v in (c.get("args") or {}).items()),
                               "Error": c.get("error") or ""} for c in calls])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"OK": st.column_config.CheckboxColumn(width="small"),
                                    "Seconds": st.column_config.NumberColumn(format="%.2f", width="small")})
    critique = trace.get("critique") or {}
    if critique:
        eyebrow("Critic")
        rounds = trace.get("critic_rounds", 1)
        by = " + ".join(critique.get("checked_by", [])) or "code"
        if critique.get("verdict") == "pass":
            st.success(f"Passed after {rounds} check(s) ({by}).", icon=":material/verified:")
        else:
            st.warning(f"Issues remained after {rounds} check(s) ({by}).", icon=":material/warning:")
            for issue in critique.get("issues", []):
                st.markdown(f"- **{issue.get('claim', '')}** — {issue.get('problem', '')}")
    if trace.get("warnings"):
        with st.expander(f"Guardrail notes ({len(trace['warnings'])})", icon=":material/shield:"):
            for warning in trace["warnings"]:
                st.caption(warning)


def usage_tab(message: dict) -> None:
    usage = message.get("usage") or {}
    if message.get("from_cache"):
        st.caption("Reused a recent identical answer: no tokens were spent.")
        return
    if not usage:
        st.caption("No usage recorded.")
        return
    order = {name: i for i, name in enumerate(AGENT_ORDER)}
    data = pd.DataFrame(sorted(({"Agent": k.capitalize(), "Tokens": v} for k, v in usage.items() if v),
                               key=lambda row: order.get(row["Agent"].lower(), 99)))
    st.metric("Tokens for this answer", f"{sum(usage.values()):,}", border=False)
    # One series: one hue, no legend; thin bars with rounded data-ends and a hover tooltip.
    chart = (alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=14, color="#2a78d6")
             .encode(x=alt.X("Tokens:Q", title=None, axis=alt.Axis(grid=True, gridOpacity=0.4, tickCount=4,
                                                                    labelExpr="format(datum.value, ',')")),
                     y=alt.Y("Agent:N", sort=None, title=None, axis=alt.Axis(ticks=False, domain=False)),
                     tooltip=[alt.Tooltip("Agent:N"), alt.Tooltip("Tokens:Q", format=",")])
             .properties(height=34 * len(data) + 20)
             .configure_view(strokeWidth=0))
    st.altair_chart(chart, width="stretch")


def answer_card(message: dict, seconds: float | None, on_feedback=None) -> None:
    body, disclaimer = notes.split_report(message.get("content", ""))
    badge_row(message, seconds)
    if message.get("status") == "error":
        st.error(body, icon=":material/error:")
        return
    figure_tiles(message.get("trace"))
    st.markdown(body)
    cited = sum(item["cited"] for item in notes.sources(message))
    tab_sources, tab_trail, tab_usage = st.tabs([f"Sources ({cited})", "Research trail", "Usage"])
    with tab_sources:
        sources_tab(message)
    with tab_trail:
        trail_tab(message.get("trace"))
    with tab_usage:
        usage_tab(message)
    with st.container(horizontal=True, vertical_alignment="center"):
        if disclaimer:
            st.caption(f":material/info: {disclaimer}")
        if on_feedback:
            key = f"feedback-{message['id']}"
            st.feedback("thumbs", key=key, on_change=on_feedback, args=(message["id"], key))


# --- live research timeline ---------------------------------------------------------------------------


class LiveTimeline:
    """Shows the research as it happens inside an st.status box."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.box = st.status("Researching…", expanded=True, state="running")
        self.tools = 0

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def event(self, event: dict) -> None:
        kind = event.get("type")
        with self.box:
            if kind == "progress":
                # No box.update() here: an update re-sends the whole box and wipes the steps
                # already written in it (seen live: only the title was left). Title changes once, in finish().
                icon = AGENT_ICONS.get(event.get("agent"), ":material/arrow_right:")
                st.markdown(f"{icon} **{event.get('label', '')}** — {event.get('text', '')}")
            elif kind == "tool":
                self.tools += 1
                args = event.get("args") or {}
                focus = args.get("ticker") or args.get("query") or ""
                query = args.get("query") if args.get("ticker") else ""
                detail = " · ".join(str(x) for x in (focus, f"“{query}”" if query else "") if x)
                st.caption(f":material/subdirectory_arrow_right: `{event.get('name')}` {detail}")
            elif kind == "tool_result" and event.get("ok") is False:
                st.caption(f":material/error: `{event.get('name')}` failed: {str(event.get('error'))[:120]}")
            elif kind == "queued":
                st.caption(f":material/hourglass_top: {event.get('text')}")
            elif kind == "error":
                st.error(event.get("text"), icon=":material/error:")

    def finish(self, message: dict | None) -> None:
        seconds = self.elapsed()
        if message is None or message.get("status") == "error":
            self.box.update(label="The research could not be completed", state="error", expanded=True)
        elif message.get("from_cache"):
            self.box.update(label="Reused a recent identical answer", state="complete", expanded=False)
        else:
            label, _, _ = notes.verification(message)
            self.box.update(label=f"Researched in {seconds:.0f} s · {self.tools} tool calls · {label.lower()}",
                            state="complete", expanded=False)
