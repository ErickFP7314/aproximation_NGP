"""
Tests for synthetic_catalog.py — F5.4 configurable ground-truth injector
(ngp-precision batch B0, the shared test substrate for B1/B3/B4).
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_catalog import synthetic_catalog
from ngp_3d import great_circle_pole

_LEGACY_COLUMNS = {
    "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmdec", "phot_g_mean_mag", "l", "b",
}
_NGP_RA_DEG = 192.75
_NGP_DEC_DEG = 27.11


# ---------------------------------------------------------------------------
# B0.1 — all-off params ≡ current synthetic_disk_stars fixture; determinism
# ---------------------------------------------------------------------------

def test_all_off_matches_legacy_fixture_columns_and_size():
    """Spec F5.4-R1-S1 — default (all-off) params reproduce the legacy
    fixture's shape: same columns, same N, parallax in the legacy [2,10] mas
    range, and known-outlier fraction of 10%."""
    df = synthetic_catalog()  # all defaults == legacy fixture params
    assert set(df.columns) == _LEGACY_COLUMNS
    assert len(df) == 500
    assert df["parallax"].min() > 1.5
    assert df["parallax"].max() < 11.0
    assert df.attrs["truth"]["outlier_fraction"] == pytest.approx(0.10)


def test_all_off_statistically_equivalent_to_legacy_fixture(synthetic_disk_stars):
    """Spec F5.4-R1-S1 — statistical equivalence: `synthetic_catalog()` with
    defaults yields the same pole-recovery behaviour (via great_circle_pole)
    as the legacy `synthetic_disk_stars` fixture, within the same tolerance
    the pre-existing regression suite already uses for that fixture."""
    df_new = synthetic_catalog(n=500, seed=42, pole=(_NGP_RA_DEG, _NGP_DEC_DEG))

    result_new = great_circle_pole(df_new)
    result_legacy = great_circle_pole(synthetic_disk_stars)

    expected_alpha_h = _NGP_RA_DEG / 15.0
    assert abs(result_new["alpha_NGP"] - expected_alpha_h) < 1.0
    assert abs(result_new["delta_NGP"] - _NGP_DEC_DEG) < 5.0
    # Both should be in the same ballpark (legacy fixture IS this function).
    assert abs(result_new["alpha_NGP"] - result_legacy["alpha_NGP"]) < 1e-9
    assert abs(result_new["delta_NGP"] - result_legacy["delta_NGP"]) < 1e-9


def test_seeded_determinism_identical_runs():
    """Cross-cutting-reproducibility-S1 — seed=42 run twice gives identical
    output (frame contents; attrs carry the same ground truth)."""
    df1 = synthetic_catalog(seed=42)
    df2 = synthetic_catalog(seed=42)
    pd.testing.assert_frame_equal(df1, df2)
    assert df1.attrs["truth"] == df2.attrs["truth"]


def test_different_seed_gives_different_output():
    """Sanity check: different seeds must not collapse to identical data."""
    df1 = synthetic_catalog(seed=42)
    df2 = synthetic_catalog(seed=7)
    assert not df1["ra"].equals(df2["ra"])


# ---------------------------------------------------------------------------
# B0.3 — injected pole recovered via great_circle_pole, warp/extinction off
# ---------------------------------------------------------------------------

def test_injected_pole_recovered_within_tolerance():
    """Spec F5.4-R1-S2 — a known, non-default injected pole (warp and
    extinction off, no outliers so the pure-geometry recovery is exercised)
    is recovered by `great_circle_pole` within ±0.05°."""
    injected_ra, injected_dec = 200.0, 15.0
    df = synthetic_catalog(
        n=500, seed=1, pole=(injected_ra, injected_dec),
        outlier_fraction=0.0, warp_amplitude_deg=0.0,
        extinction_mask_fraction=0.0,
    )
    result = great_circle_pole(df)

    recovered_alpha_deg = result["alpha_NGP"] * 15.0
    assert abs(recovered_alpha_deg - injected_ra) < 0.05, (
        f"alpha={recovered_alpha_deg:.4f}° vs injected {injected_ra}°"
    )
    assert abs(result["delta_NGP"] - injected_dec) < 0.05, (
        f"delta={result['delta_NGP']:.4f}° vs injected {injected_dec}°"
    )


def test_injected_pole_recovered_different_pole_and_seed():
    """Robustness check: recovery tolerance holds for another pole/seed
    combination, not just one lucky draw."""
    injected_ra, injected_dec = 45.0, -10.0
    df = synthetic_catalog(
        n=800, seed=99, pole=(injected_ra, injected_dec),
        outlier_fraction=0.0,
    )
    result = great_circle_pole(df)
    recovered_alpha_deg = result["alpha_NGP"] * 15.0
    # NGP normal sign flip: great_circle_pole always returns northern
    # hemisphere normal, so for an injected southern pole compare the
    # antipodal point (ra+180, -dec) instead.
    if result["delta_NGP"] >= 0 and injected_dec < 0:
        injected_ra = (injected_ra + 180.0) % 360.0
        injected_dec = -injected_dec
    assert abs(((recovered_alpha_deg - injected_ra + 180) % 360) - 180) < 0.05
    assert abs(result["delta_NGP"] - injected_dec) < 0.05


# ---------------------------------------------------------------------------
# Ground truth in attrs — z_sun plausibility (informational for B1)
# ---------------------------------------------------------------------------

def test_injected_z_sun_present_and_plausible_in_attrs():
    """Spec F5.4-R1-S3 (ground-truth side) — injected z_sun_pc is recorded
    verbatim in df.attrs["truth"], and is reflected in the data: the
    (distance-weighted) mean projection of star directions onto the pole
    is approximately z_sun_kpc (sanity check that the offset is actually
    baked into the geometry, not just metadata)."""
    z_sun_pc = 20.0
    df = synthetic_catalog(
        n=2000, seed=7, z_sun_pc=z_sun_pc, outlier_fraction=0.0,
        dist_range_kpc=(1.0, 5.0),
    )
    assert df.attrs["truth"]["z_sun_pc"] == pytest.approx(z_sun_pc)

    ra0 = np.radians(_NGP_RA_DEG)
    dec0 = np.radians(_NGP_DEC_DEG)
    pole_vec = np.array([
        np.cos(dec0) * np.cos(ra0),
        np.cos(dec0) * np.sin(ra0),
        np.sin(dec0),
    ])
    ra_rad = np.radians(df["ra"].values)
    dec_rad = np.radians(df["dec"].values)
    directions = np.column_stack([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad),
    ])
    d_kpc = 1.0 / df["parallax"].values
    points_kpc = directions * d_kpc[:, None]
    z_sun_estimate_kpc = np.mean(points_kpc @ pole_vec)

    assert abs(z_sun_estimate_kpc * 1000.0 - z_sun_pc) < 5.0


def test_degenerate_single_distance_shell_allowed_by_injector():
    """dist_range_kpc with d_min==d_max produces a single-distance-shell
    catalog (degeneracy is `offset_plane_pole`'s concern in B1, not the
    injector's — the injector must simply support building one)."""
    df = synthetic_catalog(n=100, seed=3, dist_range_kpc=(0.3, 0.3), outlier_fraction=0.0)
    d_kpc = 1.0 / df["parallax"].values
    assert np.allclose(d_kpc, 0.3, atol=1e-6)
