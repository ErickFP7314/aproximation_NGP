"""
Tests for ngp_offset_plane.py — F1 free-offset plane fit (weighted
mean-centered PCA/TLS) recovering the pole AND z_sun, plus the delta(d)
distance-shell extrapolation cross-check (ngp-precision batch B1).
"""

import numpy as np
import pandas as pd
import pytest

from ngp_offset_plane import offset_plane_pole, delta_vs_distance_shells
from ngp_3d import great_circle_pole
from synthetic_catalog import synthetic_catalog


# ---------------------------------------------------------------------------
# B1.1 — injected pole ±0.05° AND z_sun ±3pc recovered (z_sun_pc=20).
# Spec F1-R1-S1.
# ---------------------------------------------------------------------------

def test_offset_plane_pole_recovers_injected_pole_and_z_sun():
    df = synthetic_catalog(
        n=2000, seed=2, z_sun_pc=20.0, outlier_fraction=0.0,
        dist_range_kpc=(1.0, 5.0),
    )
    truth = df.attrs["truth"]
    true_ra, true_dec = truth["pole"]

    result = offset_plane_pole(df)

    recovered_alpha_deg = result["alpha_NGP"] * 15.0
    assert abs(recovered_alpha_deg - true_ra) < 0.05, (
        f"alpha={recovered_alpha_deg:.4f}° vs injected {true_ra}°"
    )
    assert abs(result["delta_NGP"] - true_dec) < 0.05, (
        f"delta={result['delta_NGP']:.4f}° vs injected {true_dec}°"
    )
    assert abs(result["z_sun_pc"] - 20.0) < 3.0, (
        f"z_sun_pc={result['z_sun_pc']:.4f} vs injected 20.0"
    )


def test_offset_plane_pole_returns_required_keys():
    df = synthetic_catalog(
        n=500, seed=2, z_sun_pc=20.0, outlier_fraction=0.0,
        dist_range_kpc=(1.0, 5.0),
    )
    result = offset_plane_pole(df)
    required = {
        "alpha_NGP", "delta_NGP", "z_sun_pc", "z_sun_err_pc", "normal",
        "covariance", "n_stars", "method", "zero_point_corrected",
    }
    assert set(result.keys()) == required
    assert result["method"] == "offset_plane_tls"
    assert result["zero_point_corrected"] is False
    assert result["normal"].shape == (3,)
    assert result["covariance"].shape == (3, 3)


# ---------------------------------------------------------------------------
# B1.2 — z_sun=0 -> offset_plane_pole agrees with great_circle_pole within
# ±0.02°. Spec F1-R1-S2.
# ---------------------------------------------------------------------------

def test_offset_plane_pole_matches_great_circle_pole_when_z_sun_zero():
    df = synthetic_catalog(
        n=2000, seed=7, z_sun_pc=0.0, outlier_fraction=0.0,
        dist_range_kpc=(1.0, 5.0),
    )

    offset_result = offset_plane_pole(df)
    gc_result = great_circle_pole(df)

    da = abs(offset_result["alpha_NGP"] * 15.0 - gc_result["alpha_NGP"] * 15.0)
    dd = abs(offset_result["delta_NGP"] - gc_result["delta_NGP"])
    assert da < 0.02, f"alpha diff={da:.4f}°"
    assert dd < 0.02, f"delta diff={dd:.4f}°"
    assert abs(offset_result["z_sun_pc"]) < 3.0, (
        f"z_sun_pc={offset_result['z_sun_pc']:.4f} should be ~0"
    )


# ---------------------------------------------------------------------------
# B1.3 — single-distance-shell input raises
# ValueError("degenerate distance distribution"). Spec F1-R1-S3.
# ---------------------------------------------------------------------------

def test_offset_plane_pole_single_shell_raises_valueerror():
    df = synthetic_catalog(
        n=100, seed=3, dist_range_kpc=(0.3, 0.3), outlier_fraction=0.0,
    )
    with pytest.raises(ValueError, match="degenerate distance distribution"):
        offset_plane_pole(df)


def test_offset_plane_pole_empty_raises_valueerror():
    empty = pd.DataFrame({"ra": [], "dec": [], "parallax": []})
    with pytest.raises(ValueError):
        offset_plane_pole(empty)


def test_offset_plane_pole_missing_column_raises_valueerror():
    df = pd.DataFrame({"ra": [1.0, 2.0], "dec": [1.0, 2.0]})  # no parallax
    with pytest.raises(ValueError):
        offset_plane_pole(df)


def test_offset_plane_pole_nonpositive_parallax_raises_valueerror():
    df = pd.DataFrame({
        "ra": [1.0, 2.0, 3.0],
        "dec": [1.0, 2.0, 3.0],
        "parallax": [1.0, 0.0, 2.0],
    })
    with pytest.raises(ValueError):
        offset_plane_pole(df)


# ---------------------------------------------------------------------------
# B1.5 — delta_vs_distance_shells d->infinity extrapolation within ±0.1° of
# the true injected delta. Spec F1-R2-S1.
# ---------------------------------------------------------------------------

def test_delta_vs_distance_shells_extrapolation_recovers_true_delta():
    df = synthetic_catalog(
        n=6000, seed=4, pole=(200.0, 15.0), z_sun_pc=20.0,
        outlier_fraction=0.0, dist_range_kpc=(0.2, 5.0),
    )
    truth = df.attrs["truth"]
    _, true_dec = truth["pole"]

    result = delta_vs_distance_shells(df)

    assert abs(result["delta_inf"] - true_dec) < 0.1, (
        f"delta_inf={result['delta_inf']:.4f}° vs injected {true_dec}°"
    )
    assert len(result["shells"]) >= 2
    assert result["model"] == "linear_inv_d"
    for shell in result["shells"]:
        assert shell["n"] >= 200


def test_delta_vs_distance_shells_returns_required_keys():
    df = synthetic_catalog(
        n=6000, seed=4, pole=(200.0, 15.0), z_sun_pc=20.0,
        outlier_fraction=0.0, dist_range_kpc=(0.2, 5.0),
    )
    result = delta_vs_distance_shells(df)
    required = {"shells", "delta_inf", "delta_inf_err", "slope", "model"}
    assert set(result.keys()) == required
    for shell in result["shells"]:
        assert set(shell.keys()) == {"d_lo", "d_hi", "d_mean", "delta", "n"}


def test_delta_vs_distance_shells_insufficient_shells_raises_valueerror():
    """Fewer than 2 usable shells (e.g. a narrow distance range that only
    populates a single default shell bin) cannot support the delta(d)
    extrapolation and must raise."""
    df = synthetic_catalog(
        n=100, seed=3, dist_range_kpc=(0.3, 0.3), outlier_fraction=0.0,
    )
    with pytest.raises(ValueError):
        delta_vs_distance_shells(df)


# ---------------------------------------------------------------------------
# B1.7 — Regression: great_circle_pole(data/gaia_disk_stars.csv) still
# alpha=12.9463h+-0.001h, delta=26.49deg+-0.01deg. Spec F1-R3-S1.
# ---------------------------------------------------------------------------

def test_great_circle_pole_regression_on_real_gaia_cache():
    data = pd.read_csv("data/gaia_disk_stars.csv")
    result = great_circle_pole(data)
    assert abs(result["alpha_NGP"] - 12.9463) < 0.001, (
        f"alpha_NGP={result['alpha_NGP']:.6f}h"
    )
    assert abs(result["delta_NGP"] - 26.49) < 0.01, (
        f"delta_NGP={result['delta_NGP']:.6f}°"
    )


# ---------------------------------------------------------------------------
# B1.8 — F5.4 deferred scenario: injected z_sun recovered via
# offset_plane_pole within +-3pc (closes F5.4-R1-S3 now that the estimator
# exists).
# ---------------------------------------------------------------------------

def test_offset_plane_pole_closes_f5_4_z_sun_scenario():
    df = synthetic_catalog(
        n=2000, seed=5, z_sun_pc=15.0, outlier_fraction=0.0,
        dist_range_kpc=(0.5, 3.0),
    )
    assert df.attrs["truth"]["z_sun_pc"] == pytest.approx(15.0)

    result = offset_plane_pole(df)
    assert abs(result["z_sun_pc"] - 15.0) < 3.0, (
        f"z_sun_pc={result['z_sun_pc']:.4f} vs injected 15.0"
    )
