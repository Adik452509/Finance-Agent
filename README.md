---
title: Research Desk
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Financial research agent with cited, reviewed answers
---

# Research Desk — a financial research agent

[![tests](https://github.com/Adik452509/Finance-Agent/actions/workflows/tests.yml/badge.svg)](https://github.com/Adik452509/Finance-Agent/actions/workflows/tests.yml)

Ask a question about a listed company — *"What share of Microsoft's revenue came from Intelligent
Cloud in fiscal 2026?"* — and a team of AI agents plans the research, reads annual reports and
official filings, calculates with a real calculator, and reviews the answer against its evidence
before you see it. Every figure in the answer carries its source: a page of the annual report, an
SEC filing, or the data provider and time.

**Status:** runs locally and in Docker. It is not hosted publicly yet.

## What it does

- **Multi-agent research team** (LangGraph): planner → researcher → analyst → writer → critic.
  A plain-code supervisor routes the work; no agent does arithmetic in its head.
- **Sources over MCP** (Model Context Protocol), one server each:
  annual reports (search over indexed filings, cited to the page) · SEC EDGAR (official 10-K /
  10-Q / 20-F figures) · market data (Yahoo Finance) · news (Tavily) · an exact calculator ·
  Japan's EDINET (built, switched off until a key is added).
- **Company snapshot page:** growth and margin charts, strengths and things to watch (each with
  the rule that produced it and its source), the valuation in plain words, recent news. No AI
  writes it, and it never says buy or sell.
- **Research Desk UI** (Streamlit): live research timeline, key-figure tiles, sources, the full
  research trail and token usage per answer, saved chats, follow-up questions, Google sign-in.
- **Backend** (FastAPI): streaming answers, an answer cache for repeated questions, background
  jobs to add a company's latest annual report, and two limits: a daily question limit per
  person (counted the moment a question is asked) and a shared daily token budget for everyone.

## What is checked, and what is not

| Figures from | Checks before the answer is shown |
|---|---|
| **Annual reports** (the indexed library) | The number must be on the cited page · right company · consolidated vs standalone · a segment total must be the total column · the column must be the year claimed · the row must be the line item asked for · a stronger model re-reads the cited page |
| **SEC, market data, news** | Copied from the data tool with its source and date. **Not yet compared with the tool's output by code** — the next item on the list |
| **Calculations** | Done only by a calculator tool (exact decimal arithmetic), never by a model |
| **The final answer** | A critic checks that every figure comes from the research notes or the calculator, is cited, and that no year is named that the evidence does not cover; then a model reviews period, scope, unit and direction |

A figure that fails a check is researched once more with a note saying what was wrong. If it
fails again it is marked unverified and may not be used — the answer says it could not be
verified instead of showing the number.

## Evaluation

Results are reported as three outcomes — **correct**, **honestly refused** ("could not find or
verify") and **wrong** — because a refusal and a confident wrong figure are very different things.

**Development set: 12 questions, 3 companies (Reliance, Apple, Toyota).** These questions were
written to find errors (older columns, segment tables, unit conversions, a year that has not
happened), and the fixes below were made while looking at them.

| Code | Correct | Refused | Wrong |
|---|---|---|---|
| Before the latest fixes (10–11 Nov 2026) | 9 | 1 | 2 |
| After them (13 Nov 2026) | 12 | 0 | 0 |

Read the second row with care:

- It is a **development-set result, not an accuracy figure.** The checks were tuned on these
  same 12 questions; a set of questions the fixes never saw is still to be built.
- **One run per question.** The research model does not always answer the same way: one question
  came out refused, wrong, wrong, then correct across four runs while it was being fixed.
- 10 of the 12 answers come from one full run; the 2 Toyota answers were re-run after a further
  fix to which source is used.
- No comparison yet between a plain single-step search and the full team. On 17 simple look-up
  questions the single-step search alone scores 17 of 17 (14 answered, 3 impossible questions
  refused), so the team's benefit — calculations, several sources, companies outside the
  library — still has to be shown with numbers.

**What the two wrong answers taught:** both cited a real number from the right page. One was the
previous year's column; the other was real 2025 growth labelled "fiscal 2031". A number being on
the page does not make it the right number, so the checks now also look at the year and the row.

## Limits

- Figures from SEC, market data and news are not yet checked against the tool output by code.
- The newest checks (year column, line item) are pattern rules built on three companies' report
  layouts; on other layouts they may refuse a correct figure or miss a wrong one.
- Market data comes from Yahoo Finance, an unofficial source that can differ from a company's
  own reported line items.
- On the free model tier the whole app can research roughly a dozen questions a day in total;
  the shared daily token budget pauses new research when that is spent.
- Research summaries are automated and are not financial advice.

## Architecture

```
Browser ─▶ Streamlit (Research Desk, :7860) ─HTTP+SSE─▶ FastAPI backend (127.0.0.1:8000)
                                                          ├─ LangGraph agent team
                                                          ├─ 6 MCP servers (filings, SEC, market, news, calculator, EDINET)
                                                          └─ background job worker
                                  Postgres + pgvector (Supabase): users, chats, cache, filings
```

One Docker image runs everything; the backend is reachable only inside the container. On a
laptop the same code runs on SQLite and a local vector store.

## Run it locally

```bash
python -m venv .venv && .venv/Scripts/activate      # Python 3.12
pip install -r requirements.txt
cp .env.example .env                                 # add your keys (see the comments)
alembic upgrade head
python scripts/run_dev.py --local-db                 # http://localhost:8501
```

Or with Docker: `docker compose up --build`.

## Tests

`pytest` runs 500+ offline tests (no API keys or network needed); they run on every push.
The evaluations above use real models: `python scripts/run_pipeline_eval.py` (12 questions) and
`python scripts/run_eval.py` (17 look-up questions).
