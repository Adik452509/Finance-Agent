"""Loader rules: text cleanup, registry guardrails, header/footer removal. Offline."""

import json

import pytest

from src.ingestion.loader import find_boilerplate, load_registry, normalize_text, strip_edges

ENTRY = {
    "file": "apple.pdf", "company": "Apple Inc.", "ticker": "AAPL", "market": "USA",
    "doc_type": "10-K", "fiscal_year_end": "2025-09-27", "currency": "USD", "reporting_unit": "USD millions",
}


# --- normalize_text -----------------------------------------------------------


def test_rupee_symbol_repaired_for_inr_documents():
    assert normalize_text("(C in crore)", "INR") == "(₹ in crore)"
    assert normalize_text("Total ` 1,234", "INR") == "Total ₹ 1,234"


def test_rupee_fix_not_applied_to_other_currencies():
    assert normalize_text("(C in crore)", "USD") == "(C in crore)"


def test_dot_leaders_removed_but_ellipsis_kept():
    assert normalize_text("Sales revenues .......... 43,787,709", "JPY") == "Sales revenues 43,787,709"
    assert normalize_text("and so on...", "USD") == "and so on..."


def test_hyphenated_words_rejoined_but_year_ranges_kept():
    assert normalize_text("reve-\nnue", "USD") == "revenue"
    assert normalize_text("FY 2024-\n25", "INR") == "FY 2024-\n25"


def test_tabs_and_nonbreaking_spaces_collapsed():
    assert normalize_text("Property,\tPlant\u00a0and   Equipment", "INR") == "Property, Plant and Equipment"


def test_figures_are_unchanged():
    line = "Revenue from Operations 25 9,80,136 9,14,472 (2,433)"
    assert normalize_text(line, "INR") == line


# --- registry -----------------------------------------------------------------


def _write(tmp_path, entries):
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_valid_registry_loads(tmp_path):
    docs = load_registry(_write(tmp_path, [ENTRY]))
    assert docs[0].ticker == "AAPL" and docs[0].reporting_unit == "USD millions"


@pytest.mark.parametrize("bad_file", ["../.env", "..\\.env", "sub/apple.pdf", "C:/secret.pdf"])
def test_registry_rejects_paths_outside_raw_dir(tmp_path, bad_file):
    with pytest.raises(ValueError, match="plain file name"):
        load_registry(_write(tmp_path, [{**ENTRY, "file": bad_file}]))


def test_registry_reports_missing_fields(tmp_path):
    entry = {k: v for k, v in ENTRY.items() if k != "reporting_unit"}
    with pytest.raises(ValueError, match="reporting_unit"):
        load_registry(_write(tmp_path, [entry]))


def test_registry_rejects_duplicate_files(tmp_path):
    with pytest.raises(ValueError, match="more than once"):
        load_registry(_write(tmp_path, [ENTRY, ENTRY]))


# --- headers / footers ---------------------------------------------------------


def test_repeated_footer_with_company_name_is_boilerplate():
    pages = [["CONSOLIDATED STATEMENTS", f"body text {i}", f"Apple Inc. | 2025 Form 10-K | {i}"] for i in range(8)]
    boilerplate = find_boilerplate(pages, "Apple Inc.")
    assert "Apple Inc. | # Form #-K | #" in boilerplate
    # Repeated section headers carry meaning (consolidated vs standalone): keep them.
    assert "CONSOLIDATED STATEMENTS" not in boilerplate


def test_strip_edges_removes_footer_and_reads_its_page_number():
    lines = ["Risk Factors", "body", "Apple Inc. | 2025 Form 10-K | 23"]
    kept, label = strip_edges(lines, {"Apple Inc. | # Form #-K | #"})
    assert kept == ["Risk Factors", "body"]
    assert label == "23"


@pytest.mark.parametrize("number_line, label", [("23", "23"), ("F-12", "F-12"), ("194 195", "194 195")])
def test_strip_edges_reads_page_numbers(number_line, label):
    kept, found = strip_edges([number_line, "Balance Sheet", "body"], set())
    assert kept == ["Balance Sheet", "body"]
    assert found == label


def test_strip_edges_keeps_numbers_inside_the_page():
    lines = ["Heading", "a", "b", "c", "2025", "d", "e", "f", "Footer"]
    kept, _ = strip_edges(lines, set())
    assert "2025" in kept  # mid-page line, not an edge: could be a table year
