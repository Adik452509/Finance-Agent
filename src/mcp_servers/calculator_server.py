"""Calculator MCP server: safe, explicit financial arithmetic.

Start it standalone with:
    python -m src.mcp_servers.calculator_server

Why a calculator tool at all? LLMs are bad at arithmetic on big numbers
(980136 / 914472 - 1 is easy to get subtly wrong). The agent must call these
tools instead of computing in its head, and every result carries its formula
and inputs so a reader (or the Phase 5 Critic) can re-check it.

Safety rules:
  - NO eval()/exec(): each operation is an explicit function.
  - Decimal arithmetic (no float rounding surprises), results rounded at the end.
  - Inputs are validated: numbers only (Indian/Western comma strings accepted),
    no bool/NaN/infinity, sane magnitude limits.
  - Refuses to mix currencies; refuses growth across missing periods.
  - stdout belongs to MCP (stdio); logs go to stderr.
"""

import logging
import math
import sys
from datetime import date
from decimal import Decimal, InvalidOperation, getcontext

from mcp.server.fastmcp import FastMCP

from src.mcp_servers.common import ToolInputError, ok, safe_call

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("calculator_server")

getcontext().prec = 28
MAX_ABS = Decimal("1e18")      # larger than any real figure in raw units (¥50 trillion = 5e13)
MAX_POINTS = 40
DECIMALS = 4

UNIT_FACTORS: dict[str, Decimal] = {
    "raw": Decimal(1),
    "thousand": Decimal(10) ** 3,
    "lakh": Decimal(10) ** 5,
    "million": Decimal(10) ** 6,
    "crore": Decimal(10) ** 7,
    "billion": Decimal(10) ** 9,
    "trillion": Decimal(10) ** 12,
}
EXPECTED_MONTHS = {"quarterly": 3, "annual": 12}

mcp = FastMCP(
    "calculator",
    instructions="Exact financial arithmetic. Always use these tools instead of computing numbers yourself.",
)


# --- helpers ------------------------------------------------------------------


def to_decimal(name: str, value) -> Decimal:
    """Parse a number safely. Accepts 980136, 9.8e5, "9,80,136", "980,136"."""
    if isinstance(value, bool) or value is None:
        raise ToolInputError(f"{name} must be a number, got {value!r}")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ToolInputError(f"{name} must be a finite number")
    text = str(value).strip().replace(",", "").replace("_", "")
    try:
        number = Decimal(text)
    except InvalidOperation:
        raise ToolInputError(f"{name} must be a number, got {value!r}") from None
    if not number.is_finite():
        raise ToolInputError(f"{name} must be a finite number")
    if abs(number) > MAX_ABS:
        raise ToolInputError(f"{name} is implausibly large ({value!r})")
    return number


def rnd(number: Decimal, places: int = DECIMALS) -> float:
    return float(round(number, places))


def months_apart(later: date, earlier: date) -> int:
    return (later.year * 12 + later.month) - (earlier.year * 12 + earlier.month)


# --- tools ----------------------------------------------------------------------


@mcp.tool()
def percent_change(old_value: float | str, new_value: float | str) -> dict:
    """Percentage change from old_value to new_value: (new - old) / |old| x 100.

    Use for growth questions: YoY, QoQ, "how much did revenue grow".
    Both values must be in the SAME unit and currency (e.g. both INR crore).

    Args:
        old_value: earlier period value, e.g. 914472 or "9,14,472".
        new_value: later period value, e.g. 980136 or "9,80,136".
    """
    def compute() -> dict:
        old = to_decimal("old_value", old_value)
        new = to_decimal("new_value", new_value)
        if old == 0:
            raise ToolInputError("percent change from zero is undefined; report the absolute change instead")
        warnings = []
        if old < 0:
            warnings.append("The old value is negative: the sign of a percent change can mislead; also report the absolute change.")
        pct = (new - old) / abs(old) * 100
        return ok(result_percent=rnd(pct, 2), absolute_change=rnd(new - old),
                  direction="increase" if new > old else ("decrease" if new < old else "no change"),
                  formula="(new - old) / |old| x 100", inputs={"old_value": str(old), "new_value": str(new)},
                  warnings=warnings)

    return safe_call("percent_change", compute, logger)


@mcp.tool()
def cagr(start_value: float | str, end_value: float | str, years: float | str) -> dict:
    """Compound annual growth rate: (end / start) ^ (1 / years) - 1, as a percent.

    Use for multi-year growth ("average annual growth over 3 years").
    Both values must be positive and in the same unit/currency.

    Args:
        start_value: value at the start, e.g. 383285.
        end_value: value at the end, e.g. 416161.
        years: number of years between them, e.g. 2.
    """
    def compute() -> dict:
        start = to_decimal("start_value", start_value)
        end = to_decimal("end_value", end_value)
        n = to_decimal("years", years)
        if start <= 0 or end <= 0:
            raise ToolInputError("CAGR needs positive start and end values")
        if not 0 < n <= 100:
            raise ToolInputError("years must be between 0 and 100")
        rate = (float(end / start) ** (1 / float(n)) - 1) * 100  # fractional power: float is fine here
        return ok(result_percent=round(rate, 2), formula="(end / start) ^ (1 / years) - 1",
                  inputs={"start_value": str(start), "end_value": str(end), "years": str(n)}, warnings=[])

    return safe_call("cagr", compute, logger)


@mcp.tool()
def ratio(numerator: float | str, denominator: float | str,
          numerator_currency: str = "", denominator_currency: str = "") -> dict:
    """numerator / denominator, e.g. price / EPS (P/E), debt / equity, net income / revenue.

    Pass the currency of each side when they are money amounts: mixing currencies
    (e.g. HKD price with CNY earnings) is refused, because the result would be wrong.

    Args:
        numerator: top of the fraction.
        denominator: bottom of the fraction (not zero).
        numerator_currency: optional ISO code, e.g. "JPY".
        denominator_currency: optional ISO code, e.g. "JPY".
    """
    def compute() -> dict:
        top = to_decimal("numerator", numerator)
        bottom = to_decimal("denominator", denominator)
        nc, dc = numerator_currency.strip().upper(), denominator_currency.strip().upper()
        if nc and dc and nc != dc:
            raise ToolInputError(f"currency mismatch: {nc} vs {dc}. Convert to one currency first.")
        if bottom == 0:
            raise ToolInputError("denominator is zero")
        return ok(result=rnd(top / bottom), formula="numerator / denominator",
                  inputs={"numerator": str(top), "denominator": str(bottom)}, warnings=[])

    return safe_call("ratio", compute, logger)


@mcp.tool()
def share_of_total(part: float | str, total: float | str) -> dict:
    """What percent `part` is of `total`, e.g. a segment's share of revenue, or a margin.

    Args:
        part: e.g. Retail segment revenue 291043.
        total: e.g. total revenue 980136.
    """
    def compute() -> dict:
        p = to_decimal("part", part)
        t = to_decimal("total", total)
        if t == 0:
            raise ToolInputError("total is zero")
        warnings = ["part is larger than total - check the inputs"] if abs(p) > abs(t) else []
        return ok(result_percent=rnd(p / t * 100, 2), formula="part / total x 100",
                  inputs={"part": str(p), "total": str(t)}, warnings=warnings)

    return safe_call("share_of_total", compute, logger)


@mcp.tool()
def convert_units(value: float | str, from_unit: str, to_unit: str) -> dict:
    """Convert a number between scales: raw, thousand, lakh, million, crore, billion, trillion.

    Use before comparing figures reported in different scales, e.g. INR crore vs
    raw rupees, or USD millions vs USD billions. This does NOT convert currencies.

    Args:
        value: the number, e.g. 980136.
        from_unit: its scale, e.g. "crore".
        to_unit: target scale, e.g. "billion".
    """
    def compute() -> dict:
        number = to_decimal("value", value)
        src, dst = from_unit.strip().lower(), to_unit.strip().lower()
        for unit in (src, dst):
            if unit not in UNIT_FACTORS:
                raise ToolInputError(f"unknown unit {unit!r}. Valid: {', '.join(UNIT_FACTORS)}")
        converted = number * UNIT_FACTORS[src] / UNIT_FACTORS[dst]
        return ok(result=rnd(converted), formula=f"value x {UNIT_FACTORS[src]} / {UNIT_FACTORS[dst]}",
                  inputs={"value": str(number), "from_unit": src, "to_unit": dst}, warnings=[])

    return safe_call("convert_units", compute, logger)


@mcp.tool()
def series_change(points: list[dict], frequency: str = "quarterly") -> dict:
    """Period-over-period percent changes for a dated series, skipping gaps.

    Use for QoQ (frequency="quarterly") or YoY on annual data (frequency="annual").
    A pair of periods that are not exactly one period apart is reported as a gap
    and NOT computed - growth across a missing quarter would be wrong.

    Args:
        points: list of {"period_end": "YYYY-MM-DD", "value": number or null}, any order.
        frequency: "quarterly" or "annual".
    """
    def compute() -> dict:
        freq = frequency.strip().lower()
        if freq not in EXPECTED_MONTHS:
            raise ToolInputError('frequency must be "quarterly" or "annual"')
        if not isinstance(points, list) or not 2 <= len(points) <= MAX_POINTS:
            raise ToolInputError(f"points must be a list of 2 to {MAX_POINTS} items")

        parsed, skipped = [], []
        for i, point in enumerate(points):
            if not isinstance(point, dict) or "period_end" not in point:
                raise ToolInputError(f"points[{i}] must look like {{'period_end': 'YYYY-MM-DD', 'value': 123}}")
            try:
                end = date.fromisoformat(str(point["period_end"]))
            except ValueError:
                raise ToolInputError(f"points[{i}].period_end must be YYYY-MM-DD") from None
            if point.get("value") is None:
                skipped.append(end.isoformat())
                continue
            parsed.append((end, to_decimal(f"points[{i}].value", point["value"])))
        parsed.sort()

        changes, gaps = [], []
        for (prev_end, prev), (cur_end, cur) in zip(parsed, parsed[1:]):
            label = f"{prev_end.isoformat()} -> {cur_end.isoformat()}"
            if months_apart(cur_end, prev_end) != EXPECTED_MONTHS[freq]:
                gaps.append(label)
                continue
            if prev == 0:
                changes.append({"periods": label, "percent": None, "note": "previous value is zero"})
                continue
            changes.append({"periods": label, "percent": rnd((cur - prev) / abs(prev) * 100, 2)})

        warnings = []
        if gaps:
            warnings.append(f"Not consecutive, not computed: {', '.join(gaps)}")
        if skipped:
            warnings.append(f"Missing values skipped (null, not zero): {', '.join(skipped)}")
        return ok(changes=changes, gaps=gaps, frequency=freq,
                  formula="(current - previous) / |previous| x 100", warnings=warnings)

    return safe_call("series_change", compute, logger)


if __name__ == "__main__":
    mcp.run()  # stdio transport
