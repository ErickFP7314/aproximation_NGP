"""
Tests for report.py — summary table comparing NGP estimation methods vs IAU.
"""

import pandas as pd
import pytest

from report import build_summary_table, to_markdown, to_latex, save_report, IAU_ALPHA_H, IAU_DELTA_DEG


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
