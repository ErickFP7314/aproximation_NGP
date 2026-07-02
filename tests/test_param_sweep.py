"""
Tests for param_sweep.py — sensitivity sweeps over b_max, n, and delta_ra.

All tests use the synthetic galactic-plane dataset from conftest.py (offline).
Tests requiring real Gaia data are marked @pytest.mark.slow.
"""

import os
import pytest
import pandas as pd
import numpy as np

from param_sweep import sweep_b_max, sweep_n, sweep_delta_ra, run_all_sweeps

# ---------------------------------------------------------------------------
# sweep_b_max
# ---------------------------------------------------------------------------

def test_sweep_b_max_default_returns_19_rows(synthetic_disk_stars):
    """Default b_max_range=range(2,21) → 19 rows (one per b_max value 2-20)."""
    df = sweep_b_max(synthetic_disk_stars)
    assert len(df) == 19, f"Expected 19 rows, got {len(df)}"


def test_sweep_b_max_returns_expected_columns(synthetic_disk_stars):
    """sweep_b_max DataFrame must have exactly the five expected columns."""
    df = sweep_b_max(synthetic_disk_stars)
    expected = {"b_max", "alpha_ar", "delta_dec1", "delta_dec2", "n_used"}
    assert set(df.columns) == expected, f"Columns: {set(df.columns)}"


def test_sweep_b_max_n_used_is_monotone_nondecreasing(synthetic_disk_stars):
    """n_used must be non-decreasing as b_max increases."""
    df = sweep_b_max(synthetic_disk_stars)
    n_used = df["n_used"].values
    assert all(n_used[i] <= n_used[i + 1] for i in range(len(n_used) - 1)), (
        "n_used is not monotonically non-decreasing"
    )


def test_sweep_b_max_custom_range(synthetic_disk_stars):
    """Custom b_max_range controls the number of output rows."""
    df = sweep_b_max(synthetic_disk_stars, b_max_range=range(5, 11))
    assert len(df) == 6, f"Expected 6 rows for range(5,11), got {len(df)}"


# ---------------------------------------------------------------------------
# sweep_n
# ---------------------------------------------------------------------------

def test_sweep_n_returns_expected_columns(synthetic_disk_stars):
    """sweep_n DataFrame must have columns [n_frac, n_value, delta_dec1]."""
    df = sweep_n(synthetic_disk_stars)
    expected = {"n_frac", "n_value", "delta_dec1"}
    assert expected.issubset(set(df.columns)), f"Columns: {set(df.columns)}"


def test_sweep_n_default_fracs_returns_10_rows(synthetic_disk_stars):
    """Default n_fracs (10 fractions 0.05..0.50) → 10 rows."""
    df = sweep_n(synthetic_disk_stars)
    assert len(df) == 10, f"Expected 10 rows, got {len(df)}"


def test_sweep_n_n_value_is_positive(synthetic_disk_stars):
    """All n_value entries must be >= 1."""
    df = sweep_n(synthetic_disk_stars)
    assert (df["n_value"] >= 1).all(), "n_value must be >= 1"


def test_sweep_n_custom_fracs(synthetic_disk_stars):
    """Custom n_fracs controls the number of output rows."""
    df = sweep_n(synthetic_disk_stars, n_fracs=[0.1, 0.2, 0.3])
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"


# ---------------------------------------------------------------------------
# sweep_delta_ra
# ---------------------------------------------------------------------------

def test_sweep_delta_ra_default_returns_30_rows(synthetic_disk_stars):
    """Default delta_range = [0.05, 0.10, ..., 1.50] (30 steps) → 30 rows."""
    df = sweep_delta_ra(synthetic_disk_stars)
    assert len(df) == 30, f"Expected 30 rows, got {len(df)}"


def test_sweep_delta_ra_returns_expected_columns(synthetic_disk_stars):
    """sweep_delta_ra DataFrame must have columns [delta_ra, delta_dec2]."""
    df = sweep_delta_ra(synthetic_disk_stars)
    expected = {"delta_ra", "delta_dec2"}
    assert expected.issubset(set(df.columns)), f"Columns: {set(df.columns)}"


def test_sweep_delta_ra_first_value(synthetic_disk_stars):
    """First delta_ra value must be 0.05."""
    df = sweep_delta_ra(synthetic_disk_stars)
    assert abs(df["delta_ra"].iloc[0] - 0.05) < 1e-9


def test_sweep_delta_ra_last_value(synthetic_disk_stars):
    """Last delta_ra value must be 1.50."""
    df = sweep_delta_ra(synthetic_disk_stars)
    assert abs(df["delta_ra"].iloc[-1] - 1.50) < 1e-9


# ---------------------------------------------------------------------------
# run_all_sweeps
# ---------------------------------------------------------------------------

def test_run_all_sweeps_saves_csv(synthetic_disk_stars, tmp_path):
    """run_all_sweeps must write a CSV file at the given path."""
    output = str(tmp_path / "sweep.csv")
    run_all_sweeps(synthetic_disk_stars, output_path=output)
    assert os.path.isfile(output), f"Expected CSV at {output}"


def test_run_all_sweeps_returns_dict(synthetic_disk_stars, tmp_path):
    """run_all_sweeps must return a dict with keys 'b_max', 'n', 'delta_ra'."""
    output = str(tmp_path / "sweep.csv")
    result = run_all_sweeps(synthetic_disk_stars, output_path=output)
    assert isinstance(result, dict), "run_all_sweeps must return a dict"
    assert "b_max" in result
    assert "n" in result
    assert "delta_ra" in result


def test_run_all_sweeps_dict_values_are_dataframes(synthetic_disk_stars, tmp_path):
    """Each value in the returned dict must be a DataFrame."""
    output = str(tmp_path / "sweep.csv")
    result = run_all_sweeps(synthetic_disk_stars, output_path=output)
    for key, df in result.items():
        assert isinstance(df, pd.DataFrame), f"result['{key}'] is not a DataFrame"


def test_run_all_sweeps_creates_parent_directory(synthetic_disk_stars, tmp_path):
    """run_all_sweeps must create parent directories if they do not exist."""
    output = str(tmp_path / "nested" / "dir" / "sweep.csv")
    run_all_sweeps(synthetic_disk_stars, output_path=output)
    assert os.path.isfile(output)
