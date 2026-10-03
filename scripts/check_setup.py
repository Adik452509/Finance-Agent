"""Phase 0 smoke test: confirms config, market data, and the LLM all work.

Run from the project root:
    python scripts/check_setup.py

Exits with code 0 if every check passes, 1 otherwise.

Besides "does it work", this also surfaces the data-quality traps the later
phases must handle: currency mismatches, missing quarters, and blank values.
"""

import math
import sys
import warnings
from datetime import date
from pathlib import Path

# scripts/ is not a package, so make the project root importable
# (lets us do `from config.settings import settings`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from config.llm import get_chat_model  # noqa: E402
from config.settings import settings  # noqa: E402

# yfinance triggers a harmless pandas deprecation warning on every call.
# Must come AFTER the imports: filters added later take priority, and
# importing yfinance installs filters of its own.
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")

console = Console()

# One company per market we support.
TICKERS: dict[str, str] = {
    "AAPL": "USA",
    "RELIANCE.NS": "India",
    "7203.T": "Japan",
    "0700.HK": "China / HK",
}
QUARTERS_TO_SHOW = 4


def human(value: float | None) -> str:
    """Format a raw currency amount as e.g. '3.09 T'. Returns 'n/a' for blanks."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= divisor:
            return f"{value / divisor:.2f} {suffix}"
    return f"{value:,.0f}"


def months_apart(later: date, earlier: date) -> int:
    return (later.year * 12 + later.month) - (earlier.year * 12 + earlier.month)


def check_config() -> bool:
    console.rule("[bold]1. Configuration")
    console.print(f"Provider        : {settings.llm_provider}")
    console.print(f"Default model   : {settings.default_model}")
    console.print(f"Reasoning model : {settings.reasoning_model}")
    # masked_keys() reports set/missing only, so no secret is printed.
    for name, state in settings.masked_keys().items():
        colour = "green" if state == "set" else "dim"
        console.print(f"  {name:18} [{colour}]{state}[/{colour}]")
    try:
        settings.require_provider_key()
        console.print("[green]Active provider key present.[/green]")
        return True
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return False


def check_ticker(ticker: str, market: str) -> bool:
    """Print price + recent quarterly revenue for one ticker; flag data traps."""
    tk = yf.Ticker(ticker)
    info = tk.info
    name = info.get("longName") or info.get("shortName")
    if not name:
        console.print(f"[red]{ticker}: no data returned (bad ticker or Yahoo throttling).[/red]")
        return False

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    trade_ccy = info.get("currency", "?")
    report_ccy = info.get("financialCurrency", "?")

    console.print(f"\n[bold]{name}[/bold]  ({ticker}, {market})")
    console.print(f"  Price: {price} {trade_ccy}    Financials reported in: {report_ccy}")

    warnings_found: list[str] = []
    if trade_ccy != report_ccy:
        warnings_found.append(
            f"Price is in {trade_ccy} but financials are in {report_ccy}; "
            "convert before computing P/E or market-cap ratios."
        )

    stmt = tk.quarterly_income_stmt
    if stmt.empty or "Total Revenue" not in stmt.index:
        console.print("  [yellow]No quarterly revenue available.[/yellow]")
        return True  # price worked; missing fundamentals is a data gap, not a setup failure

    revenue = stmt.loc["Total Revenue"].head(QUARTERS_TO_SHOW)
    quarter_ends = [ts.date() for ts in revenue.index]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Quarter end")
    table.add_column(f"Revenue ({report_ccy})", justify="right")
    for q_end, value in zip(quarter_ends, revenue.tolist()):
        table.add_row(str(q_end), human(value))
    console.print(table)

    blanks = int(revenue.isna().sum())
    if blanks:
        warnings_found.append(f"{blanks} of {len(revenue)} quarters have no revenue value.")

    # Consecutive quarter-ends should be exactly 3 months apart.
    gaps = [
        f"{earlier} -> {later}"
        for later, earlier in zip(quarter_ends, quarter_ends[1:])
        if months_apart(later, earlier) != 3
    ]
    if gaps:
        warnings_found.append(
            f"Missing quarter(s) between {', '.join(gaps)}; "
            "these rows are not consecutive, so QoQ growth would be wrong."
        )

    for w in warnings_found:
        console.print(f"  [yellow]! {w}[/yellow]")
    return True


def check_market_data() -> bool:
    console.rule("[bold]2. Market data (yfinance)")
    ok = True
    for ticker, market in TICKERS.items():
        try:
            ok = check_ticker(ticker, market) and ok
        except Exception as e:  # network errors, Yahoo blocking, schema changes
            console.print(f"[red]{ticker}: {type(e).__name__}: {e}[/red]")
            ok = False
    return ok


def check_llm() -> bool:
    console.rule("[bold]3. LLM")
    try:
        reply = get_chat_model().invoke("Reply with exactly one word: OK")
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        console.print(f"Reply from {settings.default_model}: [green]{text.strip()}[/green]")
        return True
    except Exception as e:
        # Keep the message short: provider errors can be long, and must never
        # include the key (LangChain doesn't put it in error text).
        console.print(f"[red]{type(e).__name__}: {str(e)[:200]}[/red]")
        return False


def main() -> int:
    results = {
        "Configuration": check_config(),
        "Market data": check_market_data(),
        "LLM": check_llm(),
    }
    console.rule("[bold]Summary")
    for name, ok in results.items():
        console.print(f"  {name:14} {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
