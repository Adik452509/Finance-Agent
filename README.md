---
title: Research Desk
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Financial research agent with cited, source-checked answers
---

# Research Desk — a financial research agent

[![tests](https://github.com/Adik452509/Finance-Agent/actions/workflows/tests.yml/badge.svg)](https://github.com/Adik452509/Finance-Agent/actions/workflows/tests.yml)

Ask a question about a listed company — *"What share of Microsoft's revenue came from Intelligent
Cloud in fiscal 2026?"* — and a team of AI agents plans the research, reads annual reports and
official filings, calculates with a real calculator, and **checks every figure against its source
before you see it**. Every number in the answer is cited to its page, filing or data source.

**Live app:** https://aditya09106-finance-research-agent.hf.space (sign in with Google)

## What it does

- **Multi-agent research team** (LangGraph): planner → researcher → analyst → writer → critic.
  A plain-code supervisor routes the work; no agent ever does arithmetic in its head.
- **Sources over MCP** (Model Context Protocol), one server each:
  annual reports (RAG over indexed filings, cited to the page) · SEC EDGAR (official 10-K / 10-Q /
  20-F figures, same day as filed) · market data (Yahoo Finance) · news (Tavily) · an exact
  calculator · Japan's EDINET (optional).
- **Guardrails built from real failures:** figures must appear on the cited page; consolidated vs
  segment vs standalone checks; an LLM source check for the right column and year; one corrected
  retry, then quarantine as UNVERIFIED; a critic that reviews the draft against the evidence.
- **Research Desk UI** (Streamlit): live research timeline, key-figure tiles, sources, the full
  research trail and token usage per answer, saved chats, follow-up questions, Google sign-in.
- **Backend** (FastAPI): streaming answers (SSE), per-user daily quota, an answer cache for
  repeated questions, background jobs to add a company's latest annual report.

## Architecture

```
Browser ─▶ Streamlit (Research Desk, :7860) ─HTTP+SSE─▶ FastAPI backend (127.0.0.1:8000)
                                                          ├─ LangGraph agent team
                                                          ├─ 6 MCP servers (filings, SEC, market, news, calculator, EDINET)
                                                          └─ background job worker
                                  Postgres + pgvector (Supabase): users, chats, cache, filings
```

One Docker image runs everything; the backend is reachable only inside the container.

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

`pytest` runs ~480 offline tests (no API keys or network needed).

---

*Research summaries are automated and are not financial advice.*
