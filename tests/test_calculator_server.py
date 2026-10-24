"""Calculator MCP server: exact maths, validation, gaps, currencies. Offline."""

import asyncio

import pytest

from src.mcp_servers import calculator_server as calc


def test_percent_change_on_real_reliance_figures():
    # Reliance revenue from operations: 9,14,472 (2023-24) -> 9,80,136 (2024-25) crore
    r = calc.percent_change("9,14,472", "9,80,136")
    assert r["ok"] and r["result_percent"] == 7.18
    assert r["absolute_change"] == 65664 and r["direction"] == "increase"
    assert r["formula"].startswith("(new - old)")


def test_percent_change_decrease_toyota_operating_income():
    r = calc.percent_change(4795586, 3766216)   # exact: -21.4649...%
    assert r["result_percent"] == -21.46 and r["direction"] == "decrease"


def test_percent_change_from_zero_is_refused():
    r = calc.percent_change(0, 5)
    assert r["ok"] is False and r["error_type"] == "invalid_input"


def test_percent_change_negative_base_warns():
    r = calc.percent_change(-100, 50)
    assert r["ok"] and r["warnings"]


@pytest.mark.parametrize("bad", [True, None, "abc", float("nan"), float("inf"), "1e30"])
def test_bad_numbers_rejected(bad):
    r = calc.percent_change(bad, 10)
    assert r["ok"] is False and r["error_type"] == "invalid_input"


def test_cagr():
    r = calc.cagr(383285, 416161, 2)  # Apple net sales FY2023 -> FY2025
    assert r["ok"] and r["result_percent"] == 4.2


def test_cagr_needs_positive_values():
    assert calc.cagr(-1, 10, 2)["error_type"] == "invalid_input"


def test_ratio_and_currency_guard():
    assert calc.ratio(112010, 416161)["result"] == 0.2692
    r = calc.ratio(421.2, 30, numerator_currency="HKD", denominator_currency="CNY")
    assert r["ok"] is False and "currency mismatch" in r["error"]


def test_ratio_zero_denominator():
    assert calc.ratio(1, 0)["error_type"] == "invalid_input"


def test_share_of_total_retail_segment():
    assert calc.share_of_total(291043, 980136)["result_percent"] == 29.69


def test_convert_crore_to_raw_and_billion():
    assert calc.convert_units(980136, "crore", "raw")["result"] == 9801360000000
    assert calc.convert_units(980136, "crore", "billion")["result"] == 9801.36


def test_convert_unknown_unit_lists_valid_ones():
    r = calc.convert_units(1, "gazillion", "raw")
    assert r["ok"] is False and "crore" in r["error"]


def test_series_change_skips_reliance_missing_quarter():
    points = [
        {"period_end": "2026-06-30", "value": 3.09e12},
        {"period_end": "2026-03-31", "value": 2.94e12},
        {"period_end": "2025-12-31", "value": 2.65e12},
        {"period_end": "2025-06-30", "value": 2.44e12},   # Sep 2025 missing
    ]
    r = calc.series_change(points, "quarterly")
    assert [c["periods"] for c in r["changes"]] == ["2025-12-31 -> 2026-03-31", "2026-03-31 -> 2026-06-30"]
    assert r["gaps"] == ["2025-06-30 -> 2025-12-31"]
    assert r["warnings"]


def test_series_change_null_is_skipped_not_zero():
    points = [{"period_end": "2026-06-30", "value": 10}, {"period_end": "2026-03-31", "value": None},
              {"period_end": "2025-12-31", "value": 8}]
    r = calc.series_change(points, "quarterly")
    assert r["changes"] == []          # the only neighbours are 6 months apart -> gap
    assert any("null, not zero" in w for w in r["warnings"])


def test_series_change_rejects_bad_input():
    assert calc.series_change([{"period_end": "30/06/2026", "value": 1}, {"period_end": "2026-03-31", "value": 2}])["error_type"] == "invalid_input"
    assert calc.series_change([{"period_end": "2026-03-31", "value": 2}])["error_type"] == "invalid_input"
    assert calc.series_change([], "weekly")["error_type"] == "invalid_input"


def test_tools_are_discoverable():
    names = {t.name for t in asyncio.run(calc.mcp.list_tools())}
    assert names == {"percent_change", "cagr", "ratio", "share_of_total", "convert_units", "series_change"}
