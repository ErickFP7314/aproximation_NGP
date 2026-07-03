"""
ngp_offset_plane.py — F1: free-offset plane fit (weighted mean-centered
PCA / TLS) that recovers both the Galactic-pole direction AND the Sun's
height above the Galactic plane (z_sun), plus a distance-shell delta(d)
extrapolation cross-check.

Extends `ngp_3d.great_circle_pole` (through-origin, direction-only SVD):
instead of fitting a plane through the origin (n . r = 0, distance-free,
using unit direction vectors), this module fits

    n . r = z_sun_kpc

where r is the heliocentric Cartesian position (kpc) — i.e. the SAME SVD
machinery as `great_circle_pole`, but on actual 3D positions with a free
offset recovered from the (weighted) centroid instead of forcing the fit
through the origin (design ADR1, `sdd/ngp-precision/design`).

Public API
----------
offset_plane_pole(data, *, weights=None, min_distance_spread_kpc=0.05) -> dict
    {alpha_NGP, delta_NGP, z_sun_pc, z_sun_err_pc, normal, covariance,
     n_stars, method:"offset_plane_tls", zero_point_corrected:False}
    Raises ValueError("degenerate distance distribution") if the input's
    heliocentric distance spread is below `min_distance_spread_kpc` (a
    single distance shell cannot constrain both the pole tilt and the
    offset — see design ADR1/ADR2).

delta_vs_distance_shells(data, *, shell_edges_kpc=(0.0,0.5,1.0,2.0,3.0,5.0),
                          min_stars_per_shell=200, model="linear_inv_d") -> dict
    {shells:[{d_lo,d_hi,d_mean,delta,n}...], delta_inf, delta_inf_err,
     slope, model}
    Bins stars by heliocentric distance and, in each shell with enough
    stars, runs the through-origin `great_circle_pole` (direction-only) to
    get that shell's apparent delta. A free plane offset (z_sun) biases a
    through-origin fit by an angle ~ z_sun/d that vanishes as d -> infinity,
    so delta(d) extrapolated to 1/d -> 0 (weighted linear regression,
    weight ~ sqrt(n) per shell) recovers the true pole declination
    (design ADR2) independent of `offset_plane_pole`'s own estimate — a
    cross-check.

Unit note — IMPORTANT, and DIFFERENT from `ngp_3d.coords_to_cartesian`:
    Heliocentric distance is computed here as ``d_kpc = 1 / parallax_mas``
    — the physically correct relation (d_pc = 1000/parallax_mas implies
    d_kpc = 1/parallax_mas), matching `synthetic_catalog.py`'s injected
    ground truth (see its module docstring and B0 apply-progress decision
    #1: the injector's z_sun_pc is a constant physical offset along the
    pole, NOT rescaled by distance, so d must be in genuine kpc for the
    offset's 1/d falloff — and hence z_sun_pc itself — to come out right).

    `ngp_3d.coords_to_cartesian` instead computes ``d = 1000/parallax_mas``,
    which is numerically the distance in PARSECS mislabeled as "kpc" in
    that (frozen, pre-existing) module. That mislabeling is harmless there
    because `fit_plane_ransac`/`ngp_3d_pipeline` only ever use plane
    geometry/direction, never an absolute physical distance scale.  Here,
    z_sun_pc IS an absolute physical quantity, so `coords_to_cartesian` is
    deliberately NOT reused — a private, correctly-scaled conversion
    (`_heliocentric_cartesian_kpc`) is used instead.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ngp_3d import normal_to_equatorial, great_circle_pole

_REQUIRED_COLUMNS = ("ra", "dec", "parallax")


def _heliocentric_cartesian_kpc(data: pd.DataFrame):
    """
    (N,3) heliocentric Cartesian positions in kpc, plus the (N,) distance
    array. d_kpc = 1 / parallax_mas — see module docstring's Unit note.
    """
    plx = data["parallax"].values.astype(float)
    if np.any(plx <= 0):
        raise ValueError(
            "All parallax values must be > 0. Found non-positive parallax."
        )
    d_kpc = 1.0 / plx
    ra_rad = np.radians(data["ra"].values.astype(float))
    dec_rad = np.radians(data["dec"].values.astype(float))

    x = d_kpc * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_kpc * np.cos(dec_rad) * np.sin(ra_rad)
    z = d_kpc * np.sin(dec_rad)

    return np.column_stack([x, y, z]), d_kpc


def _check_required_columns(data: pd.DataFrame) -> None:
    if data is None or len(data) == 0:
        raise ValueError("Input DataFrame is empty.")
    for col in _REQUIRED_COLUMNS:
        if col not in data.columns:
            raise ValueError(f"DataFrame is missing required column: '{col}'")


def offset_plane_pole(
    data: pd.DataFrame,
    *,
    weights: Optional[np.ndarray] = None,
    min_distance_spread_kpc: float = 0.05,
) -> dict:
    """
    Fit the free-offset plane ``n . r = z_sun_kpc`` to heliocentric
    Cartesian star positions via weighted mean-centered PCA/TLS (ADR1):
    the plane normal `n` is the eigenvector of smallest variance of the
    (weighted) scatter matrix about the (weighted) centroid; the offset is
    recovered as ``z_sun_kpc = n . centroid`` (the centroid's projection
    onto the fitted normal).

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (deg), 'dec' (deg), 'parallax' (mas, > 0).
    weights : array-like of shape (N,), optional
        Per-star weights for the weighted PCA (e.g. inverse-variance).
        Defaults to uniform weights (plain PCA/TLS).
    min_distance_spread_kpc : float
        Degeneracy guard: if ``max(d_kpc) - min(d_kpc)`` is below this
        threshold the distance distribution cannot jointly constrain the
        pole tilt and the offset (a single distance shell is scale/offset
        degenerate) and a ValueError is raised.

    Returns
    -------
    dict with keys:
        alpha_NGP, delta_NGP : float — pole (hours, degrees)
        z_sun_pc, z_sun_err_pc : float — solar offset from the plane (pc)
        normal : np.ndarray(3,) — fitted plane unit normal
        covariance : np.ndarray(3,3) — (weighted) scatter matrix of the
            centered positions (informational; NOT the normal's own
            covariance)
        n_stars : int
        method : "offset_plane_tls"
        zero_point_corrected : False (no parallax zero-point correction
            is applied in this module — see `ngp_weighted_3d.py`, F3)

    Raises
    ------
    ValueError : empty/missing-column input, non-positive parallax, or a
        degenerate (single-shell) distance distribution.
    """
    _check_required_columns(data)
    pts, d_kpc = _heliocentric_cartesian_kpc(data)
    n = len(pts)

    spread = float(d_kpc.max() - d_kpc.min())
    if spread < min_distance_spread_kpc:
        raise ValueError("degenerate distance distribution")

    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()

    centroid = (w[:, None] * pts).sum(axis=0)
    centered = pts - centroid

    # Weighted PCA via SVD: scaling each centered row by sqrt(w_i) makes
    # weighted_pts^T @ weighted_pts equal the weighted scatter matrix, so
    # its right singular vectors are the weighted-PCA eigenvectors.
    weighted_pts = centered * np.sqrt(w)[:, None]
    _, _, Vt = np.linalg.svd(weighted_pts, full_matrices=False)
    normal = Vt[-1]  # smallest singular value -> plane normal

    if normal[2] < 0:
        normal = -normal  # flip to the northern hemisphere

    z_sun_kpc = float(np.dot(normal, centroid))
    z_sun_pc = z_sun_kpc * 1000.0

    # Quick analytic error (design ADR1's "quick check" alternative to a
    # full bootstrap): perpendicular scatter about the fitted plane,
    # divided by sqrt(effective N).
    perp_kpc = centered @ normal
    n_eff = 1.0 / np.sum(w ** 2)
    sigma_perp_kpc = float(np.std(perp_kpc))
    z_sun_err_pc = sigma_perp_kpc / np.sqrt(max(n_eff, 1.0)) * 1000.0

    covariance = (weighted_pts.T @ weighted_pts)

    alpha_h, delta_deg = normal_to_equatorial(*normal)

    return {
        "alpha_NGP": alpha_h,
        "delta_NGP": delta_deg,
        "z_sun_pc": z_sun_pc,
        "z_sun_err_pc": z_sun_err_pc,
        "normal": normal,
        "covariance": covariance,
        "n_stars": int(n),
        "method": "offset_plane_tls",
        "zero_point_corrected": False,
    }


def delta_vs_distance_shells(
    data: pd.DataFrame,
    *,
    shell_edges_kpc: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0),
    min_stars_per_shell: int = 200,
    model: str = "linear_inv_d",
) -> dict:
    """
    Bin stars into heliocentric-distance shells, fit the through-origin
    `great_circle_pole` (direction-only) within each shell, and extrapolate
    delta(d) to d -> infinity (design ADR2).

    A free plane offset z_sun biases a through-origin (distance-free) fit
    by an angle that scales ~ z_sun/d — vanishing as d grows — so
    ``delta = delta_inf + slope * (1/d_mean)`` fit by (sqrt(n)-)weighted
    linear regression across shells recovers the true pole declination at
    the intercept (1/d -> 0), independent of and as a cross-check for
    `offset_plane_pole`.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (deg), 'dec' (deg), 'parallax' (mas, > 0).
    shell_edges_kpc : sequence of float
        Distance-shell bin edges (kpc), ascending.
    min_stars_per_shell : int
        Shells with fewer stars than this are dropped (too noisy).
    model : str
        Extrapolation model label (currently only "linear_inv_d" is
        implemented).

    Returns
    -------
    dict with keys:
        shells : list[dict] — one entry per usable shell:
            {d_lo, d_hi, d_mean, delta, n}
        delta_inf, delta_inf_err : float — extrapolated delta at d -> inf
        slope : float — fitted slope (delta per unit 1/d)
        model : str

    Raises
    ------
    ValueError : empty/missing-column input, non-positive parallax, or
        fewer than 2 usable shells (cannot fit a line through < 2 points).
    """
    _check_required_columns(data)
    plx = data["parallax"].values.astype(float)
    if np.any(plx <= 0):
        raise ValueError(
            "All parallax values must be > 0. Found non-positive parallax."
        )
    d_kpc = 1.0 / plx

    edges = list(shell_edges_kpc)
    shells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (d_kpc >= lo) & (d_kpc < hi)
        n_shell = int(mask.sum())
        if n_shell < min_stars_per_shell:
            continue
        shell_result = great_circle_pole(data.loc[mask])
        shells.append({
            "d_lo": float(lo),
            "d_hi": float(hi),
            "d_mean": float(d_kpc[mask].mean()),
            "delta": shell_result["delta_NGP"],
            "n": n_shell,
        })

    if len(shells) < 2:
        raise ValueError(
            "degenerate distance distribution: fewer than 2 usable "
            "distance shells (need >= 2 to extrapolate delta(d))"
        )

    inv_d = np.array([1.0 / s["d_mean"] for s in shells])
    deltas = np.array([s["delta"] for s in shells])
    shell_weights = np.sqrt(np.array([s["n"] for s in shells], dtype=float))

    slope, intercept = np.polyfit(inv_d, deltas, 1, w=shell_weights)

    residuals = deltas - (slope * inv_d + intercept)
    dof = max(len(shells) - 2, 1)
    resid_var = float(np.sum((shell_weights * residuals) ** 2) / dof)
    design_x = np.column_stack([inv_d, np.ones_like(inv_d)])
    weighted_x = design_x * shell_weights[:, None]
    try:
        cov_beta = resid_var * np.linalg.inv(weighted_x.T @ weighted_x)
        delta_inf_err = float(np.sqrt(cov_beta[1, 1]))
    except np.linalg.LinAlgError:
        delta_inf_err = float("nan")

    return {
        "shells": shells,
        "delta_inf": float(intercept),
        "delta_inf_err": delta_inf_err,
        "slope": float(slope),
        "model": model,
    }
