"""
Tests for bootstrap.py — bootstrap resampling of the 3D RANSAC NGP estimator.
"""

import json

import numpy as np
import pytest

from bootstrap import bootstrap_ngp, bootstrap_great_circle_pole, save_bootstrap_results


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_bootstrap_ngp_same_seed_gives_identical_result(synthetic_disk_stars):
    """Same seed twice -> identical alpha/delta summary statistics."""
    r1 = bootstrap_ngp(synthetic_disk_stars, n_samples=50, seed=42, ransac_max_trials=10)
    r2 = bootstrap_ngp(synthetic_disk_stars, n_samples=50, seed=42, ransac_max_trials=10)

    assert r1["alpha_mean"] == r2["alpha_mean"]
    assert r1["alpha_median"] == r2["alpha_median"]
    assert r1["alpha_ci95"] == r2["alpha_ci95"]
    assert r1["delta_mean"] == r2["delta_mean"]
    assert r1["delta_median"] == r2["delta_median"]
    assert r1["delta_ci95"] == r2["delta_ci95"]


def test_bootstrap_ngp_different_seed_gives_different_result(synthetic_disk_stars):
    """Different seeds should (almost certainly) give different results."""
    r1 = bootstrap_ngp(synthetic_disk_stars, n_samples=50, seed=42, ransac_max_trials=10)
    r2 = bootstrap_ngp(synthetic_disk_stars, n_samples=50, seed=7, ransac_max_trials=10)

    assert r1["alpha_mean"] != r2["alpha_mean"] or r1["delta_mean"] != r2["delta_mean"]


# ---------------------------------------------------------------------------
# Structure / contract
# ---------------------------------------------------------------------------

def test_bootstrap_ngp_returns_required_keys(synthetic_disk_stars):
    result = bootstrap_ngp(synthetic_disk_stars, n_samples=50, seed=42, ransac_max_trials=10)
    required = {
        "alpha_mean", "alpha_median", "alpha_ci95",
        "delta_mean", "delta_median", "delta_ci95",
        "n_samples",
    }
    assert set(result.keys()) == required


def test_bootstrap_ngp_respects_n_samples(synthetic_disk_stars):
    result = bootstrap_ngp(synthetic_disk_stars, n_samples=37, seed=1, ransac_max_trials=10)
    assert result["n_samples"] == 37


def test_bootstrap_ngp_ci95_bounds_are_ordered_and_contain_median(synthetic_disk_stars):
    result = bootstrap_ngp(synthetic_disk_stars, n_samples=80, seed=42, ransac_max_trials=10)

    alo, ahi = result["alpha_ci95"]
    dlo, dhi = result["delta_ci95"]

    assert alo <= result["alpha_median"] <= ahi
    assert dlo <= result["delta_median"] <= dhi


def test_bootstrap_ngp_ci95_contains_known_true_delta(synthetic_disk_stars):
    """
    Spec 6 style check (offline / fast version): the CI95 for delta should
    contain the synthetic dataset's known true NGP declination (27.11 deg),
    within the fixture's Gaussian scatter + outlier tolerance.
    """
    result = bootstrap_ngp(synthetic_disk_stars, n_samples=80, seed=42, ransac_max_trials=10)
    dlo, dhi = result["delta_ci95"]
    assert dlo - 2.0 <= 27.11 <= dhi + 2.0


# ---------------------------------------------------------------------------
# save_bootstrap_results
# ---------------------------------------------------------------------------

def test_save_bootstrap_results_json_round_trip(tmp_path, synthetic_disk_stars):
    result = bootstrap_ngp(synthetic_disk_stars, n_samples=30, seed=42, ransac_max_trials=10)
    out_path = tmp_path / "bootstrap_results.json"

    save_bootstrap_results(result, path=str(out_path))

    assert out_path.exists()
    with open(out_path) as f:
        loaded = json.load(f)

    assert loaded["n_samples"] == result["n_samples"]
    assert loaded["alpha_mean"] == pytest.approx(result["alpha_mean"])
    assert loaded["delta_mean"] == pytest.approx(result["delta_mean"])
    # Tuples become lists in JSON — round-trippable as 2-element sequences
    assert list(loaded["alpha_ci95"]) == pytest.approx(list(result["alpha_ci95"]))
    assert list(loaded["delta_ci95"]) == pytest.approx(list(result["delta_ci95"]))


def test_save_bootstrap_results_creates_parent_directory(tmp_path, synthetic_disk_stars):
    result = bootstrap_ngp(synthetic_disk_stars, n_samples=20, seed=42, ransac_max_trials=10)
    out_path = tmp_path / "nested" / "dir" / "bootstrap_results.json"

    save_bootstrap_results(result, path=str(out_path))

    assert out_path.exists()


# ---------------------------------------------------------------------------
# Slow / deferred — full 10,000-resample run
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_bootstrap_ngp_full_run_contains_iau_reference(synthetic_disk_stars):
    """
    Full-scale bootstrap (n_samples=10_000) — CI95 should contain the IAU
    reference NGP position (alpha=12.85h, delta=27.11deg). Slow: deferred
    from default test runs.
    """
    result = bootstrap_ngp(synthetic_disk_stars, n_samples=10_000, seed=42)
    alo, ahi = result["alpha_ci95"]
    dlo, dhi = result["delta_ci95"]
    assert alo <= 12.85 <= ahi
    assert dlo <= 27.11 <= dhi


# ---------------------------------------------------------------------------
# bootstrap_great_circle_pole — flagship distance-free estimator
# ---------------------------------------------------------------------------

def test_bootstrap_great_circle_pole_same_seed_gives_identical_result(synthetic_disk_stars):
    """Same seed twice -> identical alpha/delta summary statistics."""
    r1 = bootstrap_great_circle_pole(synthetic_disk_stars, n_samples=50, seed=42)
    r2 = bootstrap_great_circle_pole(synthetic_disk_stars, n_samples=50, seed=42)

    assert r1["alpha_mean"] == r2["alpha_mean"]
    assert r1["alpha_median"] == r2["alpha_median"]
    assert r1["alpha_ci95"] == r2["alpha_ci95"]
    assert r1["delta_mean"] == r2["delta_mean"]
    assert r1["delta_median"] == r2["delta_median"]
    assert r1["delta_ci95"] == r2["delta_ci95"]


def test_bootstrap_great_circle_pole_returns_required_keys(synthetic_disk_stars):
    result = bootstrap_great_circle_pole(synthetic_disk_stars, n_samples=50, seed=42)
    required = {
        "alpha_mean", "alpha_median", "alpha_ci95",
        "delta_mean", "delta_median", "delta_ci95",
        "n_samples",
    }
    assert set(result.keys()) == required


def test_bootstrap_great_circle_pole_respects_n_samples(synthetic_disk_stars):
    result = bootstrap_great_circle_pole(synthetic_disk_stars, n_samples=37, seed=1)
    assert result["n_samples"] == 37


def test_bootstrap_great_circle_pole_ci95_bounds_are_ordered_and_contain_median(synthetic_disk_stars):
    result = bootstrap_great_circle_pole(synthetic_disk_stars, n_samples=80, seed=42)

    alo, ahi = result["alpha_ci95"]
    dlo, dhi = result["delta_ci95"]

    assert alo <= result["alpha_median"] <= ahi
    assert dlo <= result["delta_median"] <= dhi


def test_bootstrap_great_circle_pole_ci95_contains_known_true_delta(synthetic_disk_stars):
    """CI95 for delta should contain the synthetic dataset's known true NGP
    declination (27.11 deg), within the fixture's scatter + outlier tolerance."""
    result = bootstrap_great_circle_pole(synthetic_disk_stars, n_samples=80, seed=42)
    dlo, dhi = result["delta_ci95"]
    assert dlo - 2.0 <= 27.11 <= dhi + 2.0
