"""
Shared pytest fixtures for ngp-improvement test suite.
Provides a synthetic galactic-plane dataset with known ground truth.
"""

import numpy as np
import pandas as pd
import pytest


# IAU reference NGP position
_NGP_RA_DEG = 192.75   # alpha = 12h51m
_NGP_DEC_DEG = 27.11   # delta


def _make_synthetic_disk_stars(N: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate N synthetic stars on the Galactic plane with known NGP normal.

    The Galactic plane has its normal pointing toward (ra=192.75°, dec=27.11°).
    Stars are distributed with:
    - ra/dec near the plane (Gaussian scatter σ=0.5° around the plane)
    - parallax in [2, 10] mas
    - ~10% random outliers displaced far from the plane

    Returns a DataFrame with columns: ra, dec, parallax, l, b
    plus placeholder columns parallax_error, pmra, pmdec, phot_g_mean_mag.
    """
    rng = np.random.default_rng(seed)

    # Convert NGP pole to Cartesian unit vector
    ra0 = np.radians(_NGP_RA_DEG)
    dec0 = np.radians(_NGP_DEC_DEG)
    # Plane normal (pointing to NGP)
    pole = np.array([
        np.cos(dec0) * np.cos(ra0),
        np.cos(dec0) * np.sin(ra0),
        np.sin(dec0),
    ])

    # Build two orthogonal vectors in the plane
    # Use a reference vector not parallel to pole
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(pole, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(pole, ref)
    u /= np.linalg.norm(u)
    v = np.cross(pole, u)
    v /= np.linalg.norm(v)

    n_outliers = max(1, N // 10)
    n_inliers = N - n_outliers

    # Inlier parallaxes: uniform in [2, 10] mas
    plx_inliers = rng.uniform(2.0, 10.0, n_inliers)

    # Inlier stars: angles uniformly spread in the plane
    theta = rng.uniform(0, 2 * np.pi, n_inliers)
    r = plx_inliers  # scale doesn't matter for direction

    # Points in plane (Cartesian)
    pts_inliers = (
        r[:, None] * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    )
    # Add small scatter perpendicular to plane (σ ~ 0.5° in radians)
    scatter_std = np.radians(0.5)
    pts_inliers += rng.normal(0, scatter_std, (n_inliers, 3))

    # Outliers: random directions
    plx_outliers = rng.uniform(2.0, 10.0, n_outliers)
    phi_out = rng.uniform(0, np.pi, n_outliers)
    lam_out = rng.uniform(0, 2 * np.pi, n_outliers)
    pts_outliers = plx_outliers[:, None] * np.column_stack([
        np.sin(phi_out) * np.cos(lam_out),
        np.sin(phi_out) * np.sin(lam_out),
        np.cos(phi_out),
    ])

    pts = np.vstack([pts_inliers, pts_outliers])
    plx = np.concatenate([plx_inliers, plx_outliers])

    # Convert Cartesian → ra, dec
    norms = np.linalg.norm(pts, axis=1)
    pts_unit = pts / norms[:, None]

    dec_rad = np.arcsin(np.clip(pts_unit[:, 2], -1, 1))
    ra_rad = np.arctan2(pts_unit[:, 1], pts_unit[:, 0])
    ra_deg = np.degrees(ra_rad) % 360.0
    dec_deg = np.degrees(dec_rad)

    # Galactic coordinates: simplified approximation via NGP pole transform
    # Use the actual distance (1/parallax in kpc) for l, b
    # For test purposes use ra/dec with a simple galactic approx
    b_rad = np.arcsin(
        np.sin(dec_rad) * np.sin(dec0)
        + np.cos(dec_rad) * np.cos(dec0) * np.cos(ra_rad - ra0)
    )
    l_rad = np.arctan2(
        np.cos(dec_rad) * np.sin(ra_rad - ra0),
        np.cos(dec_rad) * np.sin(dec0) * np.cos(ra_rad - ra0)
        - np.sin(dec_rad) * np.cos(dec0),
    )
    b_deg = np.degrees(b_rad)
    l_deg = np.degrees(l_rad) % 360.0

    df = pd.DataFrame({
        "ra": ra_deg,
        "dec": dec_deg,
        "parallax": plx,
        "parallax_error": rng.uniform(0.01, 0.2, N),
        "pmra": rng.normal(0, 5, N),
        "pmdec": rng.normal(0, 5, N),
        "phot_g_mean_mag": rng.uniform(10, 15, N),
        "l": l_deg,
        "b": b_deg,
    })
    return df


@pytest.fixture
def synthetic_disk_stars():
    """
    Pytest fixture: 500-star synthetic galactic-plane dataset.
    Known NGP normal at ra=192.75°, dec=27.11°.
    Parallaxes in [2,10] mas; ~10% outliers.
    """
    return _make_synthetic_disk_stars(N=500, seed=42)
