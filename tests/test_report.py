"""
Tests for report.py — summary table comparing NGP estimation methods vs IAU.
"""

import numpy as np
import pandas as pd
import pytest

from report import (
    build_summary_table, to_markdown, to_latex, save_report, IAU_ALPHA_H, IAU_DELTA_DEG,
    build_master_table,
)
from iau_forensics import KM2017_POLE


# ---------------------------------------------------------------------------
# build_summary_table
# ---------------------------------------------------------------------------

def _sample_results():
    return {
        "aprox_ar": {"alpha_NGP": 12.9, "delta_NGP": 27.0, "std_alpha": 0.1, "std_delta": 0.2},
        "ngp_3d_ransac": {"alpha_NGP": 12.86, "delta_NGP": 27.05},
    }


def test_build_summary_table_has_iau_row():
    table = build_summary_table(_sample_results())
    assert "IAU reference" in table["method"].values


def test_build_summary_table_iau_row_values():
    table = build_summary_table(_sample_results())
    iau_row = table[table["method"] == "IAU reference"].iloc[0]
    assert iau_row["alpha_NGP"] == pytest.approx(IAU_ALPHA_H)
    assert iau_row["delta_NGP"] == pytest.approx(IAU_DELTA_DEG)


def test_build_summary_table_has_error_vs_iau_column():
    table = build_summary_table(_sample_results())
    assert "error_vs_iau_deg" in table.columns


def test_build_summary_table_error_computed_correctly_known_input():
    """
    Known-input case: a method estimate exactly at the celestial pole
    (delta_NGP=90) has an angular separation from the IAU reference equal to
    (90 - IAU_DELTA_DEG) regardless of its alpha_NGP value.
    """
    results = {"pole_method": {"alpha_NGP": 0.0, "delta_NGP": 90.0}}
    table = build_summary_table(results)
    row = table[table["method"] == "pole_method"].iloc[0]
    expected = 90.0 - IAU_DELTA_DEG
    assert row["error_vs_iau_deg"] == pytest.approx(expected, abs=1e-6)


def test_build_summary_table_iau_row_error_is_zero():
    table = build_summary_table(_sample_results())
    iau_row = table[table["method"] == "IAU reference"].iloc[0]
    assert iau_row["error_vs_iau_deg"] == pytest.approx(0.0, abs=1e-9)


def test_build_summary_table_all_methods_present():
    table = build_summary_table(_sample_results())
    methods = set(table["method"].values)
    assert {"aprox_ar", "ngp_3d_ransac", "IAU reference"}.issubset(methods)


def test_build_summary_table_returns_dataframe():
    table = build_summary_table(_sample_results())
    assert isinstance(table, pd.DataFrame)


# ---------------------------------------------------------------------------
# to_markdown / to_latex
# ---------------------------------------------------------------------------

def test_to_markdown_nonempty_and_contains_method_names():
    table = build_summary_table(_sample_results())
    md = to_markdown(table)
    assert isinstance(md, str) and len(md) > 0
    assert "aprox_ar" in md
    assert "ngp_3d_ransac" in md
    assert "IAU reference" in md


def test_to_latex_nonempty_and_contains_method_names():
    table = build_summary_table(_sample_results())
    tex = to_latex(table)
    assert isinstance(tex, str) and len(tex) > 0
    assert "aprox_ar" in tex
    assert "ngp_3d_ransac" in tex


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def test_save_report_writes_both_files(tmp_path):
    table = build_summary_table(_sample_results())
    md_path = tmp_path / "summary_table.md"
    tex_path = tmp_path / "summary_table.tex"

    save_report(table, md_path=str(md_path), tex_path=str(tex_path))

    assert md_path.exists()
    assert tex_path.exists()
    assert len(md_path.read_text()) > 0
    assert len(tex_path.read_text()) > 0


def test_save_report_creates_parent_directories(tmp_path):
    table = build_summary_table(_sample_results())
    md_path = tmp_path / "nested" / "md" / "summary_table.md"
    tex_path = tmp_path / "nested" / "tex" / "summary_table.tex"

    save_report(table, md_path=str(md_path), tex_path=str(tex_path))

    assert md_path.exists()
    assert tex_path.exists()


# ---------------------------------------------------------------------------
# B7.1/B7.2/B7.3 — build_master_table (additive; must NOT alter anything above)
# ---------------------------------------------------------------------------

def _sample_master_results():
    return {
        ("great_circle", "disk_stars"): {
            "alpha_NGP": 12.9463, "delta_NGP": 26.4924,
            "sigma_stat_deg": 0.068, "sigma_syst_deg": 1.022, "sigma_total_deg": 1.024,
        },
        ("offset_plane", "disk_stars"): {
            "alpha_NGP": 12.9666, "delta_NGP": 25.6742,
            "z_sun_pc": -16.567, "zero_point_corrected": False,
        },
        ("great_circle", "cepheids"): {
            "alpha_NGP": 12.9485, "delta_NGP": 26.3204,
        },
    }


def test_build_master_table_old_api_unaffected():
    """
    B7.2 regression guard: old-API tests above this point must all still be
    collectible/runnable in the SAME test session as the new master-table
    tests below -- i.e. importing `build_master_table` alongside the old
    functions must not have broken anything. The 11 pre-existing tests in
    this module (test_build_summary_table_* / test_to_markdown_* /
    test_to_latex_* / test_save_report_*) are the actual regression guard;
    this test just documents the intent and re-asserts the old function
    still behaves identically after the B7 import change above.
    """
    table = build_summary_table(_sample_results())
    assert "IAU reference" in table["method"].values
    assert list(table.columns) == [
        "method", "alpha_NGP", "delta_NGP", "std_alpha", "std_delta", "error_vs_iau_deg",
    ]


def test_build_master_table_one_row_per_method_tracer():
    table = build_master_table(_sample_master_results())
    assert len(table) == 3
    assert isinstance(table, pd.DataFrame)


def test_build_master_table_has_required_columns():
    table = build_master_table(_sample_master_results())
    required = {
        "method", "tracer", "alpha_NGP", "delta_NGP",
        "sigma_stat_deg", "sigma_syst_deg", "sigma_total_deg",
        "zero_point_corrected", "error_vs_iau_deg", "error_vs_km2017_deg",
    }
    assert required.issubset(set(table.columns))


def test_build_master_table_method_tracer_values_present():
    table = build_master_table(_sample_master_results())
    pairs = set(zip(table["method"], table["tracer"]))
    assert ("great_circle", "disk_stars") in pairs
    assert ("offset_plane", "disk_stars") in pairs
    assert ("great_circle", "cepheids") in pairs


def test_build_master_table_missing_sigma_is_nan():
    table = build_master_table(_sample_master_results())
    row = table[(table["method"] == "offset_plane") & (table["tracer"] == "disk_stars")].iloc[0]
    assert np.isnan(row["sigma_stat_deg"])
    assert np.isnan(row["sigma_syst_deg"])
    assert np.isnan(row["sigma_total_deg"])


def test_build_master_table_present_sigma_is_not_nan():
    table = build_master_table(_sample_master_results())
    row = table[(table["method"] == "great_circle") & (table["tracer"] == "disk_stars")].iloc[0]
    assert row["sigma_stat_deg"] == pytest.approx(0.068)
    assert row["sigma_syst_deg"] == pytest.approx(1.022)
    assert row["sigma_total_deg"] == pytest.approx(1.024)


def test_build_master_table_zero_point_corrected_defaults_false():
    table = build_master_table(_sample_master_results())
    row = table[(table["method"] == "great_circle") & (table["tracer"] == "cepheids")].iloc[0]
    assert row["zero_point_corrected"] == False  # noqa: E712 (explicit bool check)


def test_build_master_table_zero_point_corrected_passthrough():
    table = build_master_table(_sample_master_results())
    row = table[(table["method"] == "offset_plane") & (table["tracer"] == "disk_stars")].iloc[0]
    assert row["zero_point_corrected"] == False  # noqa: E712


def test_build_master_table_error_vs_iau_matches_summary_table_convention():
    """Same known-input triangulation as test_build_summary_table_error_computed_correctly_known_input."""
    results = {("pole_method", "t"): {"alpha_NGP": 0.0, "delta_NGP": 90.0}}
    table = build_master_table(results)
    row = table.iloc[0]
    expected = 90.0 - IAU_DELTA_DEG
    assert row["error_vs_iau_deg"] == pytest.approx(expected, abs=1e-6)


def test_build_master_table_error_vs_km2017_zero_at_km2017_pole():
    km_alpha_h, km_delta_deg = KM2017_POLE[0] / 15.0, KM2017_POLE[1]
    results = {("m", "t"): {"alpha_NGP": km_alpha_h, "delta_NGP": km_delta_deg}}
    table = build_master_table(results)
    row = table.iloc[0]
    assert row["error_vs_km2017_deg"] == pytest.approx(0.0, abs=1e-6)


def test_build_master_table_empty_input_returns_empty_dataframe():
    table = build_master_table({})
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 0
    assert "method" in table.columns
