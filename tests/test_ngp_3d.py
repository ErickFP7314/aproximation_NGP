"""
Tests for ngp_3d.py — 3D RANSAC plane-fit NGP estimator.
"""

import pytest
import numpy as np
import pandas as pd

from ngp_3d import (
    coords_to_cartesian,
    fit_plane_ransac,
    normal_to_equatorial,
    ngp_3d_pipeline,
    great_circle_pole,
)

# IAU reference NGP position (matches conftest.synthetic_disk_stars ground truth)
_NGP_RA_DEG = 192.75
_NGP_DEC_DEG = 27.11


# ---------------------------------------------------------------------------
# coords_to_cartesian tests
# ---------------------------------------------------------------------------

def test_coords_to_cartesian_known_position():
    """S4.1 — ra=0°, dec=0°, parallax=1000 mas (d=1 kpc) → X≈1, Y≈0, Z≈0."""
    df = pd.DataFrame({"ra": [0.0], "dec": [0.0], "parallax": [1000.0]})
    result = coords_to_cartesian(df)
    assert abs(result["X"].iloc[0] - 1.0) < 1e-4
    assert abs(result["Y"].iloc[0] - 0.0) < 1e-4
    assert abs(result["Z"].iloc[0] - 0.0) < 1e-4


def test_coords_to_cartesian_nonpositive_parallax_raises():
    """S4.2 — any parallax <= 0 raises ValueError."""
    df = pd.DataFrame({
        "ra": [0.0, 10.0],
        "dec": [0.0, 5.0],
        "parallax": [1.0, 0.0],   # second row has parallax=0
    })
    with pytest.raises(ValueError):
        coords_to_cartesian(df)


def test_coords_to_cartesian_negative_parallax_raises():
    """Triangulate: negative parallax also raises ValueError."""
    df = pd.DataFrame({"ra": [0.0], "dec": [0.0], "parallax": [-1.0]})
    with pytest.raises(ValueError):
        coords_to_cartesian(df)


# ---------------------------------------------------------------------------
# fit_plane_ransac tests
# ---------------------------------------------------------------------------

def test_fit_plane_ransac_recovers_correct_normal():
    """S4.3 — 1000 pts on Z=0.5X + noise + 10% outliers → angle to true normal < 0.01 rad."""
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.uniform(-10, 10, n)
    Y = rng.uniform(-10, 10, n)
    Z_clean = 0.5 * X
    Z = Z_clean + rng.normal(0, 0.05, n)

    # 10% outliers
    n_out = n // 10
    out_idx = rng.choice(n, size=n_out, replace=False)
    Z[out_idx] += rng.uniform(5, 15, n_out)

    normal = fit_plane_ransac(X, Y, Z, rng=np.random.default_rng(42))

    # True plane Z = 0.5X + 0Y + 0  → normal ∝ (-0.5, 0, 1) → normalized
    true_normal = np.array([-0.5, 0.0, 1.0])
    true_normal /= np.linalg.norm(true_normal)

    # Angle between the two normals (take the acute angle)
    cos_angle = abs(np.dot(normal, true_normal))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    assert angle < 0.01, f"Angle between normals = {angle:.4f} rad (expected < 0.01)"


def test_fit_plane_ransac_too_few_points_raises_runtime_error():
    """S4.4 — fewer than 3 points raises RuntimeError with 'RANSAC' in message."""
    X = np.array([0.0, 1.0])
    Y = np.array([0.0, 0.0])
    Z = np.array([0.0, 0.0])
    with pytest.raises(RuntimeError, match="RANSAC"):
        fit_plane_ransac(X, Y, Z)


def test_fit_plane_ransac_refine_recovers_correct_normal():
    """User decision 2026-07-01 — refine=True (SVD/TLS refit) also recovers normal < 0.01 rad."""
    rng = np.random.default_rng(1)
    n = 1000
    X = rng.uniform(-10, 10, n)
    Y = rng.uniform(-10, 10, n)
    Z = 0.5 * X + rng.normal(0, 0.05, n)

    n_out = n // 10
    out_idx = rng.choice(n, size=n_out, replace=False)
    Z[out_idx] += rng.uniform(5, 15, n_out)

    normal = fit_plane_ransac(X, Y, Z, rng=np.random.default_rng(42), refine=True)

    true_normal = np.array([-0.5, 0.0, 1.0])
    true_normal /= np.linalg.norm(true_normal)

    cos_angle = abs(np.dot(normal, true_normal))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    assert angle < 0.01, f"Angle with refine=True = {angle:.4f} rad (expected < 0.01)"


# ---------------------------------------------------------------------------
# normal_to_equatorial tests
# ---------------------------------------------------------------------------

def test_normal_to_equatorial_north_pole():
    """S4.5 — normal (0, 0, 1) → delta_deg ≈ 90.0."""
    alpha_h, delta_deg = normal_to_equatorial(0, 0, 1)
    assert abs(delta_deg - 90.0) < 1e-6, f"delta_deg={delta_deg}"


def test_normal_to_equatorial_returns_in_range():
    """Triangulate: alpha in [0,24), delta in [-90,90] for arbitrary normal."""
    alpha_h, delta_deg = normal_to_equatorial(1, 1, 0.5)
    assert 0 <= alpha_h < 24
    assert -90 <= delta_deg <= 90


# ---------------------------------------------------------------------------
# ngp_3d_pipeline tests
# ---------------------------------------------------------------------------

def test_ngp_3d_pipeline_returns_required_keys(synthetic_disk_stars):
    """S4.6 — pipeline returns dict with exactly 4 required keys."""
    result = ngp_3d_pipeline(synthetic_disk_stars)
    required = {"alpha_NGP", "delta_NGP", "inlier_mask", "n_inliers"}
    assert set(result.keys()) == required, f"Keys: {set(result.keys())}"


def test_ngp_3d_pipeline_recovers_approximate_ngp(synthetic_disk_stars):
    """S4.6 — pipeline on synthetic disk stars yields alpha in [12,13.5]h, delta in [20,35]°."""
    result = ngp_3d_pipeline(synthetic_disk_stars, rng=np.random.default_rng(42))
    assert 12.0 <= result["alpha_NGP"] <= 13.5, f"alpha_NGP={result['alpha_NGP']:.3f}h"
    assert 20.0 <= result["delta_NGP"] <= 35.0, f"delta_NGP={result['delta_NGP']:.3f}°"


def test_ngp_3d_pipeline_refine_returns_required_keys(synthetic_disk_stars):
    """User decision — refine=True pipeline also returns all 4 keys with n_inliers > 0."""
    result = ngp_3d_pipeline(synthetic_disk_stars, rng=np.random.default_rng(42), refine=True)
    required = {"alpha_NGP", "delta_NGP", "inlier_mask", "n_inliers"}
    assert set(result.keys()) == required
    assert result["n_inliers"] > 0


def test_ngp_3d_pipeline_forwards_residual_threshold(synthetic_disk_stars):
    """Bugfix — residual_threshold is forwarded to fit_plane_ransac AND used
    consistently (not hardcoded 0.1) when recomputing the inlier mask.
    A very tight threshold (0.0001 kpc) should drastically shrink n_inliers
    compared to the loose default (0.1), proving the value is actually used."""
    loose = ngp_3d_pipeline(
        synthetic_disk_stars, rng=np.random.default_rng(42), residual_threshold=0.5
    )
    tight = ngp_3d_pipeline(
        synthetic_disk_stars, rng=np.random.default_rng(42), residual_threshold=0.01
    )
    assert tight["n_inliers"] < loose["n_inliers"], (
        f"tight={tight['n_inliers']} loose={loose['n_inliers']}"
    )


# ---------------------------------------------------------------------------
# great_circle_pole tests (B6.x — great-circle SVD flagship estimator)
# ---------------------------------------------------------------------------

def test_great_circle_pole_recovers_known_pole(synthetic_disk_stars):
    """Great-circle SVD on unit direction vectors (no distance) recovers the
    known NGP within a reasonable tolerance on the synthetic fixture
    (NGP at ra=192.75°, dec=27.11°, ~10% outliers)."""
    result = great_circle_pole(synthetic_disk_stars)

    expected_alpha_h = _NGP_RA_DEG / 15.0
    assert abs(result["alpha_NGP"] - expected_alpha_h) < 1.0, (
        f"alpha_NGP={result['alpha_NGP']:.3f}h expected~{expected_alpha_h:.3f}h"
    )
    assert abs(result["delta_NGP"] - _NGP_DEC_DEG) < 5.0, (
        f"delta_NGP={result['delta_NGP']:.3f}° expected~{_NGP_DEC_DEG:.3f}°"
    )


def test_great_circle_pole_returns_required_keys(synthetic_disk_stars):
    """Returns exactly alpha_NGP, delta_NGP, method."""
    result = great_circle_pole(synthetic_disk_stars)
    required = {"alpha_NGP", "delta_NGP", "method"}
    assert set(result.keys()) == required, f"Keys: {set(result.keys())}"


def test_great_circle_pole_method_label(synthetic_disk_stars):
    """method key is the literal string 'great_circle_svd'."""
    result = great_circle_pole(synthetic_disk_stars)
    assert result["method"] == "great_circle_svd"


def test_great_circle_pole_ranges(synthetic_disk_stars):
    """alpha_NGP in [0,24)h, delta_NGP in [-90,90]°, and northern hemisphere flip applied."""
    result = great_circle_pole(synthetic_disk_stars)
    assert 0 <= result["alpha_NGP"] < 24
    assert -90 <= result["delta_NGP"] <= 90
    assert result["delta_NGP"] >= 0  # fixture's true pole is in the north


def test_great_circle_pole_empty_raises_valueerror():
    """Empty input raises ValueError."""
    empty = pd.DataFrame({"ra": [], "dec": []})
    with pytest.raises(ValueError):
        great_circle_pole(empty)
