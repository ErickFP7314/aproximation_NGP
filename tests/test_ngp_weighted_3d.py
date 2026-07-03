"""
Tests for ngp_weighted_3d.py — F3 optional parallax zero-point correction +
covariance-weighted TLS/IRLS plane fit with combined z_sun (ngp-precision
batch B3).
"""

import time

import numpy as np
import pandas as pd
import pytest

import ngp_weighted_3d
from ngp_weighted_3d import (
    apply_parallax_zero_point,
    cartesian_covariances,
    weighted_tls_plane,
)
from ngp_offset_plane import offset_plane_pole
from synthetic_catalog import synthetic_catalog, _pole_basis, _cartesian_to_radec


# ---------------------------------------------------------------------------
# Test-only heteroscedastic-noise generator: perturbs ONLY the parallax
# (hence the line-of-sight distance) with a per-star sigma that scales with
# true distance, leaving the star's TRUE (noiseless) sky direction
# untouched. This is the exact physical noise model `weighted_tls_plane`
# assumes (rank-1, LOS-dominated covariance from parallax_error) — unlike
# `synthetic_catalog(heteroscedastic=True)`, whose ACTUAL injected position
# scatter is a fixed uniform-in-kpc angular term regardless of the
# `heteroscedastic` flag (that flag only scales the *reported*
# parallax_error column, not the true noise), which would not exercise a
# genuine weighted-vs-unweighted advantage. Reuses synthetic_catalog's own
# private pole-basis/direction helpers for consistency (no geometry
# reimplementation).
# ---------------------------------------------------------------------------

def _heteroscedastic_los_noise_catalog(
    n, seed, pole, z_sun_pc, dist_range_kpc,
    sigma_plx_base_mas=0.01, sigma_plx_scale=0.3,
):
    rng = np.random.default_rng(seed)
    pole_ra_deg, pole_dec_deg = pole
    pole_vec, u, v = _pole_basis(pole_ra_deg, pole_dec_deg)
    z_sun_kpc = z_sun_pc / 1000.0

    d_min, d_max = dist_range_kpc
    d_true = rng.uniform(d_min, d_max, n)
    theta = rng.uniform(0.0, 2 * np.pi, n)
    pts = d_true[:, None] * (
        np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    )
    pts = pts + z_sun_kpc * pole_vec

    ra_deg, dec_deg, _ = _cartesian_to_radec(pts)

    plx_true = 1.0 / d_true
    # Heteroscedastic: sigma_plx grows with TRUE distance (fainter/farther
    # stars have noisier Gaia parallaxes) — and the actual noise added below
    # is drawn from that same per-star sigma, so it genuinely matches what
    # `cartesian_covariances` assumes.
    sigma_plx = sigma_plx_base_mas * (1.0 + sigma_plx_scale * d_true)
    plx_obs = plx_true + rng.normal(0.0, sigma_plx)
    plx_obs = np.clip(plx_obs, 1e-6, None)

    return pd.DataFrame({
        "ra": ra_deg,
        "dec": dec_deg,
        "parallax": plx_obs,
        "parallax_error": sigma_plx,
    })


def _pole_angular_error_deg(alpha_h, delta_deg, true_ra_deg, true_dec_deg):
    ra1, dec1 = np.radians(alpha_h * 15.0), np.radians(delta_deg)
    ra2, dec2 = np.radians(true_ra_deg), np.radians(true_dec_deg)
    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    return float(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# B3.1/B3.2 — zero-point available (mocked) shifts distances
# corrected<uncorrected, flag True; unavailable (mocked ImportError) warns,
# no raise, flag False. Spec F3-R1-S1/S2.
# ---------------------------------------------------------------------------

def test_apply_parallax_zero_point_unavailable_warns_and_falls_back(monkeypatch):
    monkeypatch.setattr(ngp_weighted_3d, "HAVE_ZEROPOINT", False)
    df = pd.DataFrame({"parallax": [1.0, 2.0, 0.5]})

    with pytest.warns(UserWarning, match="gaiadr3-zeropoint"):
        corrected, flag = apply_parallax_zero_point(df)

    assert flag is False
    pd.testing.assert_series_equal(corrected["parallax"], df["parallax"])


def test_apply_parallax_zero_point_available_shifts_distances_corrected_lt_uncorrected(monkeypatch):
    monkeypatch.setattr(ngp_weighted_3d, "HAVE_ZEROPOINT", True)
    monkeypatch.setattr(
        ngp_weighted_3d, "_default_zpt_fn",
        lambda data: np.full(len(data), -0.02),  # typical DR3-like negative zpt
    )
    df = pd.DataFrame({"parallax": [1.0, 2.0, 0.5]})

    corrected, flag = apply_parallax_zero_point(df)

    assert flag is True
    assert np.allclose(corrected["parallax"].values, df["parallax"].values + 0.02)
    corrected_d = 1.0 / corrected["parallax"].values
    observed_d = 1.0 / df["parallax"].values
    assert np.all(corrected_d < observed_d), "corrected distances must SHRINK"


def test_apply_parallax_zero_point_injected_zpt_fn_bypasses_have_zeropoint(monkeypatch):
    monkeypatch.setattr(ngp_weighted_3d, "HAVE_ZEROPOINT", False)
    df = pd.DataFrame({"parallax": [1.0, 2.0]})

    corrected, flag = apply_parallax_zero_point(
        df, zpt_fn=lambda data: np.full(len(data), -0.01)
    )

    assert flag is True
    assert np.allclose(corrected["parallax"].values, df["parallax"].values + 0.01)


# ---------------------------------------------------------------------------
# weighted_tls_plane's own zero_point_corrected flag must reflect reality in
# every branch (never silently vanish — scientific-integrity requirement).
# ---------------------------------------------------------------------------

def test_weighted_tls_plane_zero_point_flag_false_when_package_unavailable(monkeypatch):
    monkeypatch.setattr(ngp_weighted_3d, "HAVE_ZEROPOINT", False)
    df = synthetic_catalog(n=300, seed=9, outlier_fraction=0.0, dist_range_kpc=(0.5, 2.0))

    with pytest.warns(UserWarning):
        result = weighted_tls_plane(df, zero_point=True)

    assert result["zero_point_corrected"] is False


def test_weighted_tls_plane_zero_point_flag_true_when_available(monkeypatch):
    monkeypatch.setattr(ngp_weighted_3d, "HAVE_ZEROPOINT", True)
    monkeypatch.setattr(
        ngp_weighted_3d, "_default_zpt_fn",
        lambda data: np.full(len(data), -0.01),
    )
    df = synthetic_catalog(n=300, seed=9, outlier_fraction=0.0, dist_range_kpc=(0.5, 2.0))

    result = weighted_tls_plane(df, zero_point=True)

    assert result["zero_point_corrected"] is True


def test_weighted_tls_plane_zero_point_flag_false_when_disabled():
    df = synthetic_catalog(n=300, seed=9, outlier_fraction=0.0, dist_range_kpc=(0.5, 2.0))
    result = weighted_tls_plane(df, zero_point=False)
    assert result["zero_point_corrected"] is False


# ---------------------------------------------------------------------------
# B3.4 — weighted TLS pole error < unweighted TLS (ngp_offset_plane.
# offset_plane_pole, uniform weights) under heteroscedastic parallax noise.
# STRICT comparison. Spec F3-R2-S1.
# ---------------------------------------------------------------------------

def test_weighted_tls_plane_beats_unweighted_under_heteroscedastic_noise():
    pole = (200.0, 20.0)
    df = _heteroscedastic_los_noise_catalog(
        n=4000, seed=13, pole=pole, z_sun_pc=10.0,
        dist_range_kpc=(0.3, 4.0), sigma_plx_scale=0.3,
    )

    weighted = weighted_tls_plane(df, zero_point=False)
    unweighted = offset_plane_pole(df)

    err_weighted = _pole_angular_error_deg(
        weighted["alpha_NGP"], weighted["delta_NGP"], *pole
    )
    err_unweighted = _pole_angular_error_deg(
        unweighted["alpha_NGP"], unweighted["delta_NGP"], *pole
    )

    assert err_weighted < err_unweighted, (
        f"weighted err={err_weighted:.6f}deg NOT < "
        f"unweighted err={err_unweighted:.6f}deg"
    )


def test_weighted_tls_plane_returns_required_keys():
    df = synthetic_catalog(n=300, seed=9, outlier_fraction=0.0, dist_range_kpc=(0.5, 2.0))
    result = weighted_tls_plane(df, zero_point=False)
    required = {
        "alpha_NGP", "delta_NGP", "z_sun_pc", "z_sun_err_pc", "normal",
        "covariance", "n_stars", "n_used", "zero_point_corrected",
        "sn_cut", "method",
    }
    assert set(result.keys()) == required
    assert result["method"] == "weighted_tls"
    assert result["normal"].shape == (3,)
    assert result["covariance"].shape == (3, 3)


# ---------------------------------------------------------------------------
# B3.6 — combined F1+F3 z_sun from a noisy 3D fit within +-5pc. Spec F3-R3-S1.
# ---------------------------------------------------------------------------

def test_weighted_tls_plane_recovers_combined_z_sun():
    pole = (192.75, 27.11)
    df = _heteroscedastic_los_noise_catalog(
        n=4000, seed=13, pole=pole, z_sun_pc=10.0,
        dist_range_kpc=(0.3, 4.0), sigma_plx_scale=0.3,
    )

    result = weighted_tls_plane(df, zero_point=False)

    assert abs(result["z_sun_pc"] - 10.0) < 5.0, (
        f"z_sun_pc={result['z_sun_pc']:.4f} vs injected 10.0"
    )


def test_weighted_tls_plane_with_offset_false_forces_zero_z_sun():
    df = synthetic_catalog(n=500, seed=2, z_sun_pc=0.0, outlier_fraction=0.0,
                            dist_range_kpc=(1.0, 5.0))
    result = weighted_tls_plane(df, zero_point=False, with_offset=False)
    assert result["z_sun_pc"] == 0.0


# ---------------------------------------------------------------------------
# B3.8/B3.9 — parallax_over_error_min (S/N) filter reduces N, documented
# not-worse than unfiltered. Spec F3-R4-S1.
# ---------------------------------------------------------------------------

def test_weighted_tls_plane_sn_filter_reduces_n_used_and_is_not_worse():
    pole = (192.75, 27.11)
    df = _heteroscedastic_los_noise_catalog(
        n=4000, seed=1, pole=pole, z_sun_pc=15.0,
        dist_range_kpc=(0.3, 5.0), sigma_plx_scale=0.5,
    )

    unfiltered = weighted_tls_plane(df, zero_point=False)
    filtered = weighted_tls_plane(df, zero_point=False, parallax_over_error_min=10.0)

    assert filtered["n_used"] < unfiltered["n_used"]
    assert filtered["n_used"] < filtered["n_stars"]
    assert filtered["sn_cut"] == 10.0
    assert unfiltered["sn_cut"] is None

    err_unfiltered = _pole_angular_error_deg(
        unfiltered["alpha_NGP"], unfiltered["delta_NGP"], *pole
    )
    err_filtered = _pole_angular_error_deg(
        filtered["alpha_NGP"], filtered["delta_NGP"], *pole
    )
    # "documented not worse than unfiltered" (spec F3-R4-S1): generous slack
    # since the S/N cut removes the noisiest stars from an already
    # inverse-variance-weighted fit, so it should not meaningfully hurt.
    assert err_filtered < err_unfiltered + 0.05, (
        f"filtered err={err_filtered:.5f}deg vs unfiltered={err_unfiltered:.5f}deg"
    )


def test_weighted_tls_plane_sn_filter_removing_everything_raises_valueerror():
    df = synthetic_catalog(n=50, seed=9, outlier_fraction=0.0, dist_range_kpc=(0.5, 2.0))
    with pytest.raises(ValueError):
        weighted_tls_plane(df, zero_point=False, parallax_over_error_min=1e9)


# ---------------------------------------------------------------------------
# cartesian_covariances — shape/PSD sanity, and input validation shared with
# weighted_tls_plane.
# ---------------------------------------------------------------------------

def test_cartesian_covariances_shape_and_psd():
    df = synthetic_catalog(n=100, seed=1, outlier_fraction=0.0, dist_range_kpc=(0.5, 2.0))
    cov = cartesian_covariances(df)
    assert cov.shape == (100, 3, 3)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10), "covariance matrices must be PSD"


def test_cartesian_covariances_missing_parallax_error_raises_valueerror():
    df = pd.DataFrame({"ra": [1.0, 2.0], "dec": [1.0, 2.0], "parallax": [1.0, 2.0]})
    with pytest.raises(ValueError):
        cartesian_covariances(df)


def test_weighted_tls_plane_empty_raises_valueerror():
    empty = pd.DataFrame({"ra": [], "dec": [], "parallax": [], "parallax_error": []})
    with pytest.raises(ValueError):
        weighted_tls_plane(empty)


def test_weighted_tls_plane_missing_parallax_error_raises_valueerror():
    df = pd.DataFrame({"ra": [1.0, 2.0, 3.0], "dec": [1.0, 2.0, 3.0], "parallax": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        weighted_tls_plane(df)


# ---------------------------------------------------------------------------
# B3.11 [SLOW] — 50k-star weighted_tls_plane <60s, on the real cached Gaia
# disk-star sample (data/gaia_disk_stars.csv, ~53k rows), read fully offline
# via gaia_fetcher.fetch_gaia_stars() (cache-first, no network call needed
# since the cache file already exists). Spec F3-R5-S1 (SHOULD, not MUST).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_weighted_tls_plane_50k_real_cache_performance():
    from gaia_fetcher import fetch_gaia_stars

    df = fetch_gaia_stars()  # cache-first; data/gaia_disk_stars.csv exists offline

    start = time.perf_counter()
    result = weighted_tls_plane(df, zero_point=False)
    elapsed = time.perf_counter() - start

    print(f"\nweighted_tls_plane on {len(df)} real Gaia stars took {elapsed:.3f}s")
    assert result["n_stars"] == len(df)
    assert elapsed < 60.0, f"weighted_tls_plane took {elapsed:.3f}s (>= 60s target)"
