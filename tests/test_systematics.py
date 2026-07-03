"""
Tests for systematics.py -- F5 systematic-error budget: OFAT systematics
matrix, sky-region jackknife, sigma_total = sigma_stat (+) sigma_syst, and
the extinction-asymmetry hemisphere-detection scenario (ngp-precision batch
B5).
"""

import numpy as np
import pandas as pd
import pytest

from bootstrap import bootstrap_great_circle_pole
from ngp_3d import great_circle_pole
from synthetic_catalog import synthetic_catalog
from systematics import (
    BUDGET_COLUMNS,
    combine_error_budget,
    extinction_asymmetry_check,
    sky_region_jackknife,
    systematics_grid,
)


# ---------------------------------------------------------------------------
# B5.1 -- clean synthetic data -> sigma_syst < 0.02 deg. Spec F5-R1-S1.
# ---------------------------------------------------------------------------

def test_systematics_grid_clean_synthetic_sigma_syst_small():
    df = synthetic_catalog(
        n=4000, seed=5, pole=(192.75, 27.11), outlier_fraction=0.0,
        dist_range_kpc=(0.2, 1.0),
    )

    table = systematics_grid(df, write_output=False)
    point_estimate = great_circle_pole(df)
    bs = bootstrap_great_circle_pole(df, n_samples=200, seed=5)

    budget = combine_error_budget(point_estimate, bs, table)

    assert budget["sigma_syst_deg"] < 0.02, (
        f"sigma_syst_deg={budget['sigma_syst_deg']:.5f}deg on a clean synthetic "
        f"catalog (no injected asymmetry) should be < 0.02deg"
    )
    assert budget["sigma_total_deg"] >= budget["sigma_stat_deg"]
    assert budget["n_variants"] == len(table)


# ---------------------------------------------------------------------------
# B5.2 -- hemisphere split detects an INJECTED asymmetry:
# |delta_pole| > 3 * sigma_stat. Spec F5-R1-S2. This is the key scientific
# validation: proving the method distinguishes a real systematic from noise.
# ---------------------------------------------------------------------------

def test_extinction_asymmetry_check_detects_injected_asymmetry():
    """
    Real dust extinction is not azimuthally uniform -- it concentrates along
    specific galactic-longitude "dust lanes" and is far worse on one side of
    the plane than the other. A purely |b|-fraction mask (azimuthally
    UNIFORM in galactic longitude `l`) turns out to leave the through-origin
    great-circle SVD fit essentially unbiased (verified numerically while
    building this test: `great_circle_pole` on a random b>0-thinned subsample
    reproduces the north-side population mean just as well as the full
    north population would, just noisier -- so a uniform-in-`l` mask changes
    N, not the expected pole). A physically realistic dust-lane mask MUST
    also be asymmetric in `l` to break the estimator's rotational symmetry
    and inject a genuine, detectable bias -- so this drops 90% of the
    b>0 & l<90deg stars (one dust lane), on top of the base synthetic
    catalog, rather than relying on `synthetic_catalog`'s built-in
    (azimuthally-uniform) `extinction_mask_fraction/region` alone.
    """
    df = synthetic_catalog(
        n=6000, seed=7, pole=(192.75, 27.11), outlier_fraction=0.0,
        dist_range_kpc=(0.2, 1.0),
    )
    rng = np.random.default_rng(1)
    dust_lane = (df["b"] > 0) & (df["l"] < 90)
    candidates = df.index[dust_lane]
    n_drop = int(round(len(candidates) * 0.9))
    drop_idx = rng.choice(np.asarray(candidates), size=n_drop, replace=False)
    df = df.drop(index=drop_idx).reset_index(drop=True)

    # Sanity: the injected dust-lane mask actually removed a large chunk of
    # the b>0 side, leaving a real N asymmetry between hemispheres.
    assert (df["b"] >= 0).sum() < (df["b"] < 0).sum()

    result = extinction_asymmetry_check(df, n_bootstrap=200, seed=7)

    assert result["delta_pole_deg"] > 3.0 * result["sigma_stat_deg"], (
        f"delta_pole_deg={result['delta_pole_deg']:.5f}deg should exceed "
        f"3*sigma_stat={3*result['sigma_stat_deg']:.5f}deg for an injected "
        f"hemisphere asymmetry"
    )
    assert result["flagged"] is True


def test_extinction_asymmetry_check_clean_data_not_flagged():
    """Negative control: no injected asymmetry -> should NOT be flagged
    (guards against a check that always fires regardless of input)."""
    df = synthetic_catalog(
        n=6000, seed=9, pole=(192.75, 27.11), outlier_fraction=0.0,
        dist_range_kpc=(0.2, 1.0),
    )
    result = extinction_asymmetry_check(df, n_bootstrap=200, seed=9)
    assert result["flagged"] is False


def test_extinction_asymmetry_check_missing_b_column_raises():
    df = pd.DataFrame({"ra": [1.0, 2.0, 3.0], "dec": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        extinction_asymmetry_check(df)


def test_extinction_asymmetry_check_insufficient_stars_raises():
    df = pd.DataFrame({
        "ra": [1.0, 2.0], "dec": [1.0, 2.0], "b": [1.0, -1.0],
    })
    with pytest.raises(ValueError):
        extinction_asymmetry_check(df, min_stars=3)


# ---------------------------------------------------------------------------
# B5.3 -- systematics_budget.csv/json schema contract. Spec F5-R1-S3.
# ---------------------------------------------------------------------------

def test_systematics_grid_output_schema(tmp_path):
    df = synthetic_catalog(
        n=2000, seed=11, outlier_fraction=0.0, dist_range_kpc=(0.2, 1.0),
    )
    csv_path = tmp_path / "systematics_budget.csv"
    json_path = tmp_path / "systematics_budget.json"

    table = systematics_grid(
        df, output_csv=str(csv_path), output_json=str(json_path),
    )

    assert list(table.columns) == BUDGET_COLUMNS
    assert csv_path.exists()
    assert json_path.exists()

    reloaded = pd.read_csv(csv_path)
    assert list(reloaded.columns) == BUDGET_COLUMNS
    assert len(reloaded) == len(table)

    import json as _json
    with open(json_path) as f:
        records = _json.load(f)
    assert isinstance(records, list)
    assert len(records) == len(table)
    assert set(records[0].keys()) == set(BUDGET_COLUMNS)


def test_systematics_grid_no_write_when_disabled(tmp_path):
    df = synthetic_catalog(
        n=1000, seed=13, outlier_fraction=0.0, dist_range_kpc=(0.2, 1.0),
    )
    csv_path = tmp_path / "should_not_exist.csv"
    systematics_grid(df, output_csv=str(csv_path), write_output=False)
    assert not csv_path.exists()


def test_systematics_grid_empty_raises():
    empty = pd.DataFrame({"ra": [], "dec": []})
    with pytest.raises(ValueError):
        systematics_grid(empty, write_output=False)


def test_systematics_grid_baseline_row_present():
    df = synthetic_catalog(
        n=1500, seed=17, outlier_fraction=0.0, dist_range_kpc=(0.2, 1.0),
    )
    table = systematics_grid(df, write_output=False)
    baseline_rows = table[
        table["b_max"].isna() & table["G_limit"].isna() & table["sn_cut"].isna()
        & (table["shell"] == "all") & (table["hemisphere"] == "all")
    ]
    assert len(baseline_rows) == 1
    assert baseline_rows.iloc[0]["method"] == "great_circle_svd"


# ---------------------------------------------------------------------------
# B5.5 -- bootstrap_great_circle_pole's own return signature is UNCHANGED;
# it is consumed as-is by combine_error_budget (regression). Spec F5-R2-S1.
# ---------------------------------------------------------------------------

def test_bootstrap_great_circle_pole_signature_unchanged_regression():
    df = synthetic_catalog(
        n=800, seed=19, outlier_fraction=0.0, dist_range_kpc=(0.2, 1.0),
    )
    bs = bootstrap_great_circle_pole(df, n_samples=50, seed=19)
    required = {
        "alpha_mean", "alpha_median", "alpha_ci95",
        "delta_mean", "delta_median", "delta_ci95", "n_samples",
    }
    assert set(bs.keys()) == required
    assert bs["n_samples"] == 50

    point_estimate = great_circle_pole(df)
    table = systematics_grid(df, write_output=False)
    budget = combine_error_budget(point_estimate, bs, table)
    assert budget["sigma_stat_deg"] >= 0.0


def test_combine_error_budget_empty_table_raises():
    point_estimate = {"alpha_NGP": 12.85, "delta_NGP": 27.13}
    bs = {"alpha_ci95": (12.8, 12.9), "delta_ci95": (27.0, 27.2)}
    empty_table = pd.DataFrame(columns=BUDGET_COLUMNS)
    with pytest.raises(ValueError):
        combine_error_budget(point_estimate, bs, empty_table)


# ---------------------------------------------------------------------------
# B5.6 -- sky_region_jackknife
# ---------------------------------------------------------------------------

def test_sky_region_jackknife_returns_expected_keys_and_positive_sigma():
    df = synthetic_catalog(
        n=3000, seed=23, outlier_fraction=0.0, dist_range_kpc=(0.2, 1.0),
    )
    result = sky_region_jackknife(df, n_regions=8)

    required = {
        "overall_alpha_NGP", "overall_delta_NGP", "region_estimates",
        "sigma_jackknife_deg", "n_regions_used",
    }
    assert set(result.keys()) == required
    assert result["n_regions_used"] >= 2
    assert result["sigma_jackknife_deg"] >= 0.0
    assert len(result["region_estimates"]) == result["n_regions_used"]


def test_sky_region_jackknife_empty_raises():
    empty = pd.DataFrame({"ra": [], "dec": []})
    with pytest.raises(ValueError):
        sky_region_jackknife(empty)


def test_sky_region_jackknife_missing_ra_raises():
    df = pd.DataFrame({"dec": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        sky_region_jackknife(df)
