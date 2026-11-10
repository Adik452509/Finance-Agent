"""Company snapshot: the facts that help someone think about a company - never a buy/sell call.

Built for "should I invest in X?": the app does not answer that (it depends on the person's
goals and risk, and in India specific buy/sell calls need SEBI registration). It shows the
evidence instead - growth, profitability, cash, debt, valuation, news, the company's own risk
factors - plus strengths and things to watch.

No language model is involved: figures come from the data tools, arithmetic is Decimal, and
every strength / concern is a stated rule with its threshold and the numbers that triggered
it, so a reader can check exactly why it is listed. The thresholds are general rules of
thumb, and the page says so: what counts as "high" differs by industry.
"""

import json
from datetime import date
from decimal import Decimal, InvalidOperation

# Rules of thumb (shown to the user next to every strength / concern they produce).
HIGH_NET_MARGIN, THIN_NET_MARGIN = Decimal(15), Decimal(5)            # percent
STRONG_CAGR = Decimal(5)                                               # percent a year
LOW_LEVERAGE, HIGH_LEVERAGE = Decimal("0.5"), Decimal("1.5")           # debt / equity
HIGH_PE = Decimal(40)

QUESTIONS = [
    "How long can you stay invested? Shares can fall sharply and stay down for years.",
    "How much of your savings would this be? Spreading money across many companies lowers the risk of one going wrong.",
    "Do you understand how this company makes money, and what could stop it?",
    "Is the price already high for its earnings (see the P/E), meaning a lot of good news is expected?",
    "What do the company's own risk factors and recent news say that these numbers don't show?",
]
DISCLAIMER = ("This snapshot is an automated summary of public financial data for education and research. It is not "
              "investment advice or a recommendation to buy, sell or hold any security. Past results do not predict "
              "future returns. Consider your own situation and consult a SEBI-registered investment adviser (or a "
              "licensed adviser in your country) before investing.")


def _d(value) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def pct_change(old: Decimal | None, new: Decimal | None) -> Decimal | None:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / abs(old) * 100


def cagr(first: Decimal | None, last: Decimal | None, years: int) -> Decimal | None:
    """Compound annual growth rate in percent; None unless both ends are positive."""
    if first is None or last is None or first <= 0 or last <= 0 or years <= 0:
        return None
    return (Decimal(str(float(last / first) ** (1 / years))) - 1) * 100


def ratio_pct(part: Decimal | None, whole: Decimal | None) -> Decimal | None:
    if part is None or whole is None or whole == 0:
        return None
    return part / whole * 100


def money(value: Decimal | float | None, currency: str) -> str | None:
    """Readable amount: Indian rupees in crore (how Indian reports state them), others K/M/B/T."""
    value = _d(value)
    if value is None:
        return None
    if currency == "INR":
        crore = value / Decimal(10**7)
        return f"₹{_indian_grouping(crore)} crore"
    for divisor, suffix in ((Decimal(10**12), "T"), (Decimal(10**9), "B"), (Decimal(10**6), "M")):
        if abs(value) >= divisor:
            return f"{value / divisor:,.2f} {suffix} {currency}"
    return f"{value:,.0f} {currency}"


def _indian_grouping(value: Decimal) -> str:
    """9801360 -> '98,01,360' (lakh/crore grouping), rounded to a whole number."""
    sign, n = ("-" if value < 0 else ""), int(abs(value).quantize(Decimal(1)))
    s = str(n)
    if len(s) <= 3:
        return sign + s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return sign + ",".join(([head] if head else []) + groups) + "," + tail


def fy_label(fiscal_year_end: str) -> str:
    return f"FY{date.fromisoformat(fiscal_year_end).year}"


def _pct(value: Decimal | None) -> str:
    return f"{value:.1f}%" if value is not None else "n/a"


def build_snapshot(ticker: str, quote: dict, annual: dict, news: dict | None = None,
                   risk_ref: dict | None = None) -> dict:
    """Assemble the snapshot from tool results (each a tool's JSON reply as a dict)."""
    if not annual.get("ok") or not annual.get("years"):
        raise ValueError(annual.get("error") or f"No annual financial statements found for {ticker}.")
    currency = annual.get("currency") or quote.get("financial_currency") or quote.get("currency") or ""
    rows = sorted(annual["years"], key=lambda r: r["fiscal_year_end"])  # oldest -> newest, for charts

    years = []
    for r in rows:
        revenue, net_income, operating = _d(r.get("revenue")), _d(r.get("net_income")), _d(r.get("operating_income"))
        debt, equity = _d(r.get("total_debt")), _d(r.get("shareholders_equity"))
        years.append({
            "fiscal_year_end": r["fiscal_year_end"], "label": fy_label(r["fiscal_year_end"]),
            "revenue": revenue, "net_income": net_income, "operating_income": operating,
            "net_margin": ratio_pct(net_income, revenue), "operating_margin": ratio_pct(operating, revenue),
            "free_cash_flow": _d(r.get("free_cash_flow")), "operating_cash_flow": _d(r.get("operating_cash_flow")),
            "cash": _d(r.get("cash_and_equivalents")), "debt": debt, "equity": equity,
            "debt_to_equity": (debt / equity) if debt is not None and equity not in (None, 0) and equity > 0 else None,
        })
    latest, previous, first = years[-1], (years[-2] if len(years) > 1 else None), years[0]
    span = len(years) - 1
    revenue_growth = pct_change(previous["revenue"], latest["revenue"]) if previous else None
    profit_growth = pct_change(previous["net_income"], latest["net_income"]) if previous else None
    revenue_cagr = cagr(first["revenue"], latest["revenue"], span) if span >= 2 else None
    source = f"{annual.get('source', 'Yahoo Finance')} annual statements, as of {str(annual.get('as_of', ''))[:10]}"
    period = f"{latest['label']} (year ended {latest['fiscal_year_end']})"

    strengths, watch = [], []

    def add(bucket, title, detail, rule):
        bucket.append({"title": title, "detail": detail, "rule": rule, "source": source})

    if revenue_growth is not None and revenue_growth > 0:
        add(strengths, "Revenue is growing", f"{period}: {money(latest['revenue'], currency)}, "
            f"{_pct(revenue_growth)} more than the year before.", "Rule: revenue up year on year")
    elif revenue_growth is not None and revenue_growth < 0:
        add(watch, "Revenue fell", f"{period}: {money(latest['revenue'], currency)}, {_pct(abs(revenue_growth))} lower "
            "than the year before.", "Rule: revenue down year on year")
    if revenue_cagr is not None:
        if revenue_cagr >= STRONG_CAGR:
            add(strengths, "Steady multi-year growth", f"Revenue grew {_pct(revenue_cagr)} a year on average from "
                f"{first['label']} to {latest['label']}.", f"Rule: average growth of {STRONG_CAGR}%+ a year")
        elif revenue_cagr < 0:
            add(watch, "Revenue shrinking over several years", f"Revenue fell {_pct(abs(revenue_cagr))} a year on "
                f"average from {first['label']} to {latest['label']}.", "Rule: negative average growth")
    if latest["net_income"] is not None and latest["net_income"] < 0:
        add(watch, "The company made a loss", f"{period}: net loss of {money(abs(latest['net_income']), currency)}.",
            "Rule: net income below zero")
    elif profit_growth is not None and profit_growth < 0:
        add(watch, "Profit fell", f"{period}: net income {money(latest['net_income'], currency)}, "
            f"{_pct(abs(profit_growth))} lower than the year before.", "Rule: net income down year on year")
    elif profit_growth is not None and profit_growth > 0:
        add(strengths, "Profit is growing", f"{period}: net income {money(latest['net_income'], currency)}, "
            f"{_pct(profit_growth)} more than the year before.", "Rule: net income up year on year")
    margin = latest["net_margin"]
    if margin is not None and latest["net_income"] is not None and latest["net_income"] >= 0:
        if margin >= HIGH_NET_MARGIN:
            add(strengths, "High profitability", f"Keeps {_pct(margin)} of revenue as profit in {latest['label']}.",
                f"Rule of thumb: net margin above {HIGH_NET_MARGIN}% is high for most industries")
        elif margin < THIN_NET_MARGIN:
            add(watch, "Thin profit margin", f"Keeps only {_pct(margin)} of revenue as profit in {latest['label']}.",
                f"Rule of thumb: net margin below {THIN_NET_MARGIN}% leaves little room for bad years")
    fcf = latest["free_cash_flow"]
    if fcf is not None and fcf > 0:
        add(strengths, "Generates cash", f"Free cash flow of {money(fcf, currency)} in {latest['label']} "
            "(cash left after investing in the business).", "Rule: positive free cash flow")
    elif fcf is not None and fcf < 0:
        add(watch, "Spends more cash than it makes", f"Free cash flow of {money(fcf, currency)} in {latest['label']}.",
            "Rule: negative free cash flow")
    de = latest["debt_to_equity"]
    if de is not None:
        if de <= LOW_LEVERAGE:
            add(strengths, "Low debt", f"Debt is {de:.2f}× shareholders' equity ({latest['label']}).",
                f"Rule of thumb: debt-to-equity below {LOW_LEVERAGE}")
        elif de >= HIGH_LEVERAGE:
            add(watch, "High debt", f"Debt is {de:.2f}× shareholders' equity ({latest['label']}).",
                f"Rule of thumb: debt-to-equity above {HIGH_LEVERAGE} raises risk when profits dip")
    elif latest["equity"] is not None and latest["equity"] <= 0:
        add(watch, "Negative shareholders' equity", f"Liabilities exceed assets in {latest['label']}.",
            "Rule: equity at or below zero")

    valuation = []
    pe = _d(quote.get("trailing_pe"))
    price_source = f"Yahoo Finance, as of {str(quote.get('as_of', ''))[:16].replace('T', ' ')} UTC"
    if pe is not None and pe > 0:
        valuation.append(f"Price is {pe:.1f}× the last year's earnings (P/E). Put simply, at today's price it would "
                         f"take about {pe:.0f} years of current profit to earn back the share price.")
        if pe >= HIGH_PE:
            add(watch, "Priced for high expectations", f"P/E of {pe:.1f}: investors pay a lot for each unit of "
                "profit, so the price assumes strong future growth.", f"Rule of thumb: P/E above {HIGH_PE}")
            watch[-1]["source"] = price_source
    elif pe is not None:
        valuation.append("No P/E: the company did not make a profit over the last year.")
    if quote.get("currency") and currency and quote.get("currency") != currency:
        valuation.append(f"Shares trade in {quote['currency']} while the statements are in {currency}.")

    latest_rev_label = money(latest["revenue"], currency)
    kpis = [
        {"label": "Revenue", "value": latest_rev_label, "context": f"{latest['label']} · {_pct(revenue_growth)} vs prior year"
         if revenue_growth is not None else latest["label"]},
        {"label": "Net margin", "value": _pct(margin), "context": f"{latest['label']} · net income / revenue"},
        {"label": "Free cash flow", "value": money(fcf, currency) or "n/a", "context": latest["label"]},
        {"label": "Debt / equity", "value": f"{de:.2f}×" if de is not None else "n/a", "context": latest["label"]},
    ]
    news_items = [{"title": n.get("title", ""), "source": n.get("source", ""), "url": n.get("url", ""),
                   "published_date": n.get("published_date")} for n in ((news or {}).get("results") or [])
                  if is_article(n)][:5]
    return {
        "ticker": ticker, "company": quote.get("name") or ticker, "currency": currency,
        "price": {"value": quote.get("price"), "currency": quote.get("currency"),
                  "market_cap": money(quote.get("market_cap"), quote.get("currency") or "") or quote.get("market_cap_display"),
                  "pe": float(pe) if pe is not None else None, "source": price_source},
        "kpis": kpis,
        "years": [{k: (float(v) if isinstance(v, Decimal) else v) for k, v in y.items()} for y in years],
        "growth": {"revenue_yoy": _float(revenue_growth), "profit_yoy": _float(profit_growth),
                   "revenue_cagr": _float(revenue_cagr), "cagr_span": f"{first['label']}-{latest['label']}" if span >= 2 else None},
        "strengths": strengths, "watch": watch, "valuation": valuation, "news": news_items,
        "risk_factors": risk_ref, "questions": QUESTIONS, "disclaimer": DISCLAIMER,
        "source": source, "warnings": list(annual.get("warnings") or []) + list(quote.get("warnings") or []),
    }


async def call_tool(tools: dict, name: str, args: dict) -> dict:
    """Call one data tool directly (no agent involved); a structured error if it is missing or fails."""
    from src.rag.qa import _message_text

    tool = tools.get(name)
    if tool is None:
        return {"ok": False, "error": f"{name} is not available"}
    try:
        content = await tool.ainvoke(args)
    except Exception as e:  # a missing source must not break the whole snapshot
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return parse_tool_json(content if isinstance(content, str) else _message_text(content))


async def fetch_snapshot(tools: dict, ticker: str, indexed: set[str]) -> dict:
    """Gather price, five years of statements, news and (if indexed) the risk-factors page; build the snapshot."""
    import asyncio

    quote, annual = await asyncio.gather(call_tool(tools, "get_quote", {"ticker": ticker}),
                                         call_tool(tools, "get_annual_financials", {"ticker": ticker, "years": 5}))
    if not annual.get("ok"):
        raise LookupError(annual.get("error") or f"No annual statements for {ticker}.")
    company = quote.get("name") or ticker
    # 8 asked, 5 shown: quote pages that come back as "news" are dropped (is_article).
    extra = [call_tool(tools, "search_news", {"query": f"{company} company news", "days": 14, "max_results": 8})]
    indexed_ticker = next((t for t in indexed if t.upper() == ticker.upper()), None)
    if indexed_ticker:
        extra.append(call_tool(tools, "search_filings", {"query": "risk factors principal risks and uncertainties",
                                                         "ticker": indexed_ticker, "k": 1}))
    results = await asyncio.gather(*extra)
    news = results[0] if results[0].get("ok") else None
    risk_ref = None
    if indexed_ticker and results[1].get("ok") and results[1].get("results"):
        hit = results[1]["results"][0]
        risk_ref = {"citation": hit.get("citation"), "source_id": hit.get("source_id")}
    return build_snapshot(ticker, quote if quote.get("ok") else {}, annual, news, risk_ref)


def is_article(item: dict) -> bool:
    """A news story, not a stock-quote page or a search engine's own page (seen live: both came back as 'news')."""
    url, title = str(item.get("url", "")).lower(), str(item.get("title", "")).lower()
    if "/quote/" in url or "tavily.com" in (item.get("source") or "") or "stock price, news, quote" in title:
        return False
    return bool(title.strip())


def _float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def parse_tool_json(content) -> dict:
    """A tool's reply as a dict (tools return JSON text); a structured error if it isn't JSON."""
    if isinstance(content, dict):
        return content
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {"ok": False, "error": "unreadable tool reply"}
    return data if isinstance(data, dict) else {"ok": False, "error": "unexpected tool reply"}
