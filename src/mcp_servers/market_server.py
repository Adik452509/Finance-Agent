"""Market Data MCP server: live prices and quarterly financials via yfinance.

Runs as its own process; agents reach it only through MCP. Start it standalone with:
    python -m src.mcp_servers.market_server

Rules for every tool here:
  - Never raise: return {"ok": False, "error": ...} so the agent can react.
  - Validate inputs: they come from an LLM, so treat them as untrusted.
  - Say which currency every number is in, and flag data that would mislead
    (currency mismatches, missing quarters, blanks) instead of hiding it.
  - stdout belongs to the MCP protocol (stdio transport). Nothing may print
    there, so logging goes to stderr and yfinance calls have stdout redirected.
"""

import contextlib
import logging
import math
import re
import sys
import time
import warnings
from datetime import date, datetime, timezone
from typing import Any, Callable

import yfinance as yf
from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("market_server")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")

SOURCE = "Yahoo Finance (via yfinance)"
CACHE_TTL_SECONDS = 300  # agents often ask for the same ticker several times in one run
MAX_QUARTERS = 8

# Letters, digits and . - ^ = : covers AAPL, RELIANCE.NS, 7203.T, 0700.HK, ^NSEI.
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=]{0,19}$")

REVENUE_ROWS = ("Total Revenue", "Operating Revenue")
NET_INCOME_ROWS = (
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income From Continuing Operation Net Minority Interest",
)

MARKETS = [
    {"market": "USA", "suffix": "(none)", "example": "AAPL"},
    {"market": "India - NSE", "suffix": ".NS", "example": "RELIANCE.NS"},
    {"market": "India - BSE", "suffix": ".BO", "example": "RELIANCE.BO"},
    {"market": "Japan - Tokyo", "suffix": ".T", "example": "7203.T"},
    {"market": "Hong Kong", "suffix": ".HK", "example": "0700.HK"},
    {"market": "China - Shanghai", "suffix": ".SS", "example": "600519.SS"},
    {"market": "China - Shenzhen", "suffix": ".SZ", "example": "000858.SZ"},
]

mcp = FastMCP(
    "market-data",
    instructions=(
        "Live stock prices and quarterly financials from Yahoo Finance. "
        "Every amount is in raw currency units (not thousands or millions); check the currency fields."
    ),
)

_cache: dict[tuple, tuple[float, dict]] = {}


# --- helpers -----------------------------------------------------------------


class ToolInputError(ValueError):
    """Bad arguments from the caller (the LLM): reported back, not logged as a crash."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        raise ToolInputError("ticker must be a string such as 'AAPL' or 'RELIANCE.NS'")
    cleaned = ticker.strip().upper()
    if not TICKER_RE.match(cleaned):
        raise ToolInputError(
            f"'{ticker}' is not a valid ticker. Use the exchange suffix, e.g. AAPL, RELIANCE.NS, 7203.T, 0700.HK "
            "(see list_supported_markets)."
        )
    return cleaned


def _clean_number(value: Any) -> float | None:
    """yfinance uses NaN for blanks. Blanks must stay blank (None), never 0."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def format_amount(value: float | None, currency: str) -> str | None:
    """'109.42 B USD' - a readable twin of the raw number, so units can't be misread."""
    if value is None:
        return None
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= divisor:
            return f"{value / divisor:,.2f} {suffix} {currency}"
    return f"{value:,.2f} {currency}"


def _months_apart(later: date, earlier: date) -> int:
    return (later.year * 12 + later.month) - (earlier.year * 12 + earlier.month)


def _first_row(stmt, names: tuple[str, ...]):
    for name in names:
        if name in stmt.index:
            return name, stmt.loc[name]
    return None, None


def _yahoo(call: Callable[[], Any]) -> Any:
    """Run a yfinance call with anything it prints diverted away from stdout."""
    with contextlib.redirect_stdout(sys.stderr):
        return call()


def _safe(tool_name: str, key: tuple, compute: Callable[[], dict]) -> dict:
    """Cache, then turn every failure into a structured error the agent can read."""
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]
    try:
        result = compute()
    except ToolInputError as e:
        return {"ok": False, "error": str(e), "error_type": "invalid_input"}
    except Exception as e:  # network errors, Yahoo throttling, schema changes
        logger.exception("%s failed", tool_name)
        return {"ok": False, "error": f"Market data unavailable: {type(e).__name__}: {str(e)[:200]}",
                "error_type": "upstream_error"}
    if result.get("ok"):
        _cache[key] = (time.monotonic(), result)
    return result


# --- pure builders (tested without network) ------------------------------------


def build_quote(ticker: str, info: dict) -> dict:
    name = info.get("longName") or info.get("shortName")
    if not name:
        return {"ok": False, "error_type": "not_found",
                "error": f"No market data for '{ticker}'. Check the ticker and suffix (see list_supported_markets)."}

    currency = info.get("currency")
    financial_currency = info.get("financialCurrency") or currency
    price = _clean_number(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap = _clean_number(info.get("marketCap"))

    warnings_: list[str] = []
    if currency and financial_currency and currency != financial_currency:
        warnings_.append(
            f"Shares trade in {currency} but financial statements are reported in {financial_currency}. "
            "Convert before combining price with statement figures (e.g. P/E, price-to-sales)."
        )
    if price is None:
        warnings_.append("No current price available.")

    return {
        "ok": True,
        "ticker": ticker,
        "name": name,
        "price": price,
        "currency": currency,
        "market_cap": market_cap,
        "market_cap_display": format_amount(market_cap, currency or ""),
        "trailing_pe": _clean_number(info.get("trailingPE")),
        "financial_currency": financial_currency,
        "exchange": info.get("exchange"),
        "warnings": warnings_,
        "source": SOURCE,
        "as_of": _now(),
        "note": "trailing_pe is as published by Yahoo, not recomputed here.",
    }


def build_quarterly(ticker: str, stmt, currency: str | None, quarters: int) -> dict:
    if stmt is None or getattr(stmt, "empty", True):
        return {"ok": False, "error_type": "not_found",
                "error": f"No quarterly financial statements available for '{ticker}'."}

    rev_name, revenue = _first_row(stmt, REVENUE_ROWS)
    ni_name, net_income = _first_row(stmt, NET_INCOME_ROWS)
    if revenue is None and net_income is None:
        return {"ok": False, "error_type": "not_found",
                "error": f"Quarterly statements for '{ticker}' have no revenue or net income rows."}

    columns = list(stmt.columns)[:quarters]  # newest first
    rows, blanks = [], 0
    for col in columns:
        rev = _clean_number(revenue[col]) if revenue is not None else None
        ni = _clean_number(net_income[col]) if net_income is not None else None
        blanks += (rev is None) + (ni is None)
        rows.append({
            "quarter_end": col.date().isoformat(),
            "revenue": rev,
            "revenue_display": format_amount(rev, currency or ""),
            "net_income": ni,
            "net_income_display": format_amount(ni, currency or ""),
        })

    ends = [c.date() for c in columns]
    gaps = [f"{earlier.isoformat()} -> {later.isoformat()}"
            for later, earlier in zip(ends, ends[1:]) if _months_apart(later, earlier) != 3]

    warnings_: list[str] = []
    if gaps:
        warnings_.append(
            "Quarters are not consecutive (missing between " + ", ".join(gaps) + "). "
            "Do not compute quarter-over-quarter growth across a gap."
        )
    if blanks:
        warnings_.append(f"{blanks} value(s) are missing (null). Report them as unavailable, never as zero.")
    if len(columns) < quarters:
        warnings_.append(f"Only {len(columns)} quarter(s) available; {quarters} requested.")

    return {
        "ok": True,
        "ticker": ticker,
        "currency": currency,
        "unit": "raw currency units (not thousands or millions)",
        "quarters": rows,
        "consecutive": not gaps,
        "rows_used": {"revenue": rev_name, "net_income": ni_name},
        "warnings": warnings_,
        "source": SOURCE,
        "as_of": _now(),
    }


# --- MCP tools --------------------------------------------------------------------


@mcp.tool()
def get_quote(ticker: str) -> dict:
    """Get the latest stock price, market cap and trailing P/E for one company.

    Use for "current price", "market cap", "valuation" questions. Not for
    revenue or profit history (use get_quarterly_financials) and not for
    anything stated in annual reports (use the filings tools).

    Args:
        ticker: Yahoo ticker WITH exchange suffix: AAPL (USA), RELIANCE.NS (India),
            7203.T (Japan), 0700.HK (Hong Kong). Call list_supported_markets if unsure.

    Returns price in the trading `currency`. If `financial_currency` differs,
    statement figures are in another currency - see `warnings`.
    """
    def compute() -> dict:
        symbol = normalize_ticker(ticker)
        return build_quote(symbol, _yahoo(lambda: yf.Ticker(symbol).info) or {})

    return _safe("get_quote", ("quote", str(ticker).strip().upper()), compute)


@mcp.tool()
def get_quarterly_financials(ticker: str, quarters: int = 4) -> dict:
    """Get quarterly revenue and net income for one company, newest quarter first.

    Use for growth and profitability trends ("last 4 quarters", "QoQ", "YoY").
    Amounts are raw currency units in the company's REPORTING currency.

    Args:
        ticker: Yahoo ticker with exchange suffix, e.g. AAPL, RELIANCE.NS, 7203.T.
        quarters: how many recent quarters to return, 1-8 (default 4).

    Always read `warnings` and `consecutive`: Yahoo sometimes skips quarters or
    leaves values blank (null). Never treat null as zero, and never compute
    quarter-over-quarter growth across a missing quarter.
    """
    def compute() -> dict:
        symbol = normalize_ticker(ticker)
        if not isinstance(quarters, int) or not 1 <= quarters <= MAX_QUARTERS:
            raise ToolInputError(f"quarters must be an integer from 1 to {MAX_QUARTERS}")
        tk = yf.Ticker(symbol)
        stmt = _yahoo(lambda: tk.quarterly_income_stmt)
        info = _yahoo(lambda: tk.info) or {}
        currency = info.get("financialCurrency") or info.get("currency")
        return build_quarterly(symbol, stmt, currency, quarters)

    return _safe("get_quarterly_financials", ("quarterly", str(ticker).strip().upper(), quarters), compute)


@mcp.tool()
def list_supported_markets() -> dict:
    """List ticker formats per stock market, plus companies that have annual reports indexed.

    Call this before guessing a ticker: the suffix decides the market
    (RELIANCE.NS is India's NSE; RELIANCE alone is not valid).
    """
    def compute() -> dict:
        try:
            from src.ingestion.loader import load_registry

            indexed = [{"company": d.company, "ticker": d.ticker, "market": d.market} for d in load_registry()]
        except Exception as e:  # registry problems must not break market data
            logger.warning("Could not read document registry: %s", e)
            indexed = []
        return {"ok": True, "markets": MARKETS, "companies_with_filings": indexed, "source": SOURCE}

    return _safe("list_supported_markets", ("markets",), compute)


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
