"""
ngp_weighted_3d.py — F3: parallax zero-point correction (optional dependency)
+ per-star covariance-weighted total-least-squares (TLS/IRLS) plane fit, with
a free offset that also recovers z_sun from the 3D method (ngp-precision
batch B3).

This module is meant to RESCUE the distance-aware 3D approach, which was the
weakest method in `ngp-improvement` (`ngp_3d.ngp_3d_pipeline`'s RANSAC fit
biased alpha by tens of minutes due to unweighted 1/parallax distance
noise — see `ngp_3d.great_circle_pole`'s docstring). It does so with two
upgrades over the frozen `ngp_3d_pipeline`:

1. An optional Gaia DR3 parallax zero-point correction (`gaiadr3-zeropoint`,
   Lindegren et al. 2021) applied BEFORE converting parallax to distance.
   The dependency is OPTIONAL: if the package is not importable, a warning
   is emitted and the pipeline proceeds with uncorrected parallaxes — this
   must NEVER raise, and the `zero_point_corrected: bool` flag in every
   returned dict makes the fallback visible downstream (scientific-integrity
   requirement, design ADR5 / proposal `sdd/ngp-precision/proposal`).
2. A per-star WEIGHTED total-least-squares (TLS) plane fit via iteratively
   reweighted least squares (IRLS, design ADR6): each star's weight is the
   inverse of its perpendicular-to-plane variance, propagated from its
   Gaia `parallax_error` into a Cartesian covariance. This replaces
   `ngp_3d_pipeline`'s arbitrary `residual_threshold`/RANSAC inlier gate
   with a principled per-star uncertainty weighting, and — combined with a
   free offset (the same mean-centered-PCA trick as `ngp_offset_plane.py`,
   F1) — recovers z_sun from the 3D fit itself.

Units — CRITICAL, see `sdd/ngp-precision/gotcha-units-coords-to-cartesian`:
    Heliocentric distance is computed here as ``d_kpc = 1 / parallax_mas``
    (d_pc = 1000/parallax_mas => d_kpc = 1/parallax_mas), the SAME physically
    correct convention used by `synthetic_catalog.py` and
    `ngp_offset_plane._heliocentric_cartesian_kpc` — NOT
    `ngp_3d.coords_to_cartesian`, which computes ``d = 1000/parallax_mas``
    (numerically parsecs, mislabeled "kpc" in that frozen module). That
    mislabeling is harmless in `ngp_3d.py` because `great_circle_pole` and
    `fit_plane_ransac` only ever use plane geometry/direction, never an
    absolute physical distance scale — but it WOULD silently corrupt
    `z_sun_pc` (an absolute physical quantity) by a factor of 1000 if
    reused here. `_heliocentric_cartesian_kpc` is imported directly from
    `ngp_offset_plane.py` (single implementation, no duplication) rather
    than reimplemented.

Public API
----------
apply_parallax_zero_point(data, *, zpt_fn=None) -> tuple[pd.DataFrame, bool]
    Applies the Gaia DR3 parallax zero-point correction if available (or if
    `zpt_fn` is supplied directly, bypassing the optional-import check).
    Falls back to uncorrected parallax + a `UserWarning` (no exception) if
    the optional `gaiadr3-zeropoint` package is not importable.

cartesian_covariances(data) -> np.ndarray, shape (N, 3, 3), kpc^2
    Per-star heliocentric-Cartesian position covariance, rank-1 and
    LOS-dominated: ``C_i = sigma_d_i^2 * outer(u_i, u_i)`` where `u_i` is
    the unit line-of-sight direction and
    ``sigma_d_i = d_i^2 * parallax_error_i`` (exact under the
    `d_kpc = 1/parallax_mas` convention above — see the "Error propagation"
    note below for the derivation and why NO extra /1000 factor is needed).

weighted_tls_plane(data, *, parallax_over_error_min=None, zero_point=True,
                    zpt_fn=None, with_offset=True, max_iter=10, tol=1e-8) -> dict
    {alpha_NGP, delta_NGP, z_sun_pc, z_sun_err_pc, normal, covariance,
     n_stars, n_used, zero_point_corrected, sn_cut, method:"weighted_tls"}
    IRLS: unweighted-normal seed -> per-star weights from cartesian
    covariances -> weighted (mean-centered, if `with_offset`) scatter-matrix
    eigendecomposition -> repeat until the normal direction stabilizes or
    `max_iter` is reached. See design ADR6; rejected alternative: full
    per-star 3x3 covariance-matrix ML inversion (too slow / unneeded
    complexity for the accuracy gain — IRLS on the rank-1 LOS covariance is
    sufficient and O(N) per iteration).

Error propagation — sigma_d derivation
---------------------------------------
Under ``d_kpc = 1 / parallax_mas`` (this module's convention),
``d(d_kpc)/d(parallax_mas) = -1/parallax_mas^2 = -d_kpc^2`` EXACTLY (no unit
conversion constant needed, because the convention already folds the
pc/mas <-> kpc/mas relationship into the numeric identity
`d_kpc = 1/parallax_mas`: 1 mas of parallax corresponds to exactly 1 kpc).
Hence ``sigma_d_kpc = d_kpc^2 * sigma_parallax_mas`` (first-order/linear
error propagation, valid at the S/N>~5-10 regime this module targets via
`parallax_over_error_min`). NOTE: design doc `sdd/ngp-precision/design`
mentions a "/1000" factor in this formula's prose sketch — that referred to
`coords_to_cartesian`'s d=1000/parallax convention (parsecs, mislabeled
kpc); under the `d_kpc = 1/parallax_mas` convention adopted here (per the
units gotcha), the conversion constant is already 1 and NO extra /1000
scaling is applied. This is a deliberate, documented deviation from the
design's literal formula text, not from its intent (per-star sigma_d from
parallax_error, propagated to Cartesian).

Unweighted-baseline note (used by this module's own tests)
------------------------------------------------------------
`ngp_offset_plane.offset_plane_pole(data)` (uniform weights, free offset)
IS the unweighted TLS baseline used to demonstrate the weighted estimator's
advantage under heteroscedastic parallax noise (spec F3-R2-S1) — no
separate "unweighted" function is duplicated here.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ngp_3d import normal_to_equatorial
from ngp_offset_plane import _check_required_columns, _heliocentric_cartesian_kpc

_REQUIRED_COLUMNS = ("ra", "dec", "parallax", "parallax_error")

# ---------------------------------------------------------------------------
# Optional dependency: gaiadr3-zeropoint (Lindegren et al. 2021, DR3 parallax
# zero-point correction). NEVER hard-required: import failure must degrade
# gracefully (warn + proceed uncorrected), never raise. HAVE_ZEROPOINT is a
# module-level flag (rather than re-importing on every call) so tests can
# monkeypatch it directly to exercise both branches without needing the
# real package installed (see tests/test_ngp_weighted_3d.py).
# ---------------------------------------------------------------------------
try:
    import gaiadr3_zeropoint as _gaiadr3_zeropoint  # noqa: F401  (optional dep)
    HAVE_ZEROPOINT = True
except ImportError:
    _gaiadr3_zeropoint = None
    HAVE_ZEROPOINT = False


def _default_zpt_fn(data: pd.DataFrame) -> np.ndarray:
    """
    Real Gaia DR3 zero-point correction via the optional `gaiadr3-zeropoint`
    package. NETWORK/PACKAGE-only (mirrors `gaia_fetcher._default_gaia_query`
    and `tracer_fetcher._default_cepheid_query`'s convention of leaving the
    real-dependency branch untested by the offline unit suite); only
    reachable when `HAVE_ZEROPOINT` is True. Returns the per-star zero-point
    offset (mas) to be SUBTRACTED from the observed parallax.
    """
    # NOTE: the exact gaiadr3-zeropoint API requires nu_eff_used_in_astrometry
    # / pseudocolour / ecl_lat / astrometric_params_solved columns beyond this
    # module's minimal contract; wiring the exact call is deferred to when
    # the package is actually installed (B3.10 pins it as an optional extra
    # only). This placeholder keeps the "available" code path structurally
    # complete without depending on columns this module does not require.
    from gaiadr3_zeropoint import zpt  # noqa: PLC0415

    return zpt.get_zpt(
        data["phot_g_mean_mag"].values,
        data.get("nu_eff_used_in_astrometry"),
        data.get("pseudocolour"),
        data.get("ecl_lat"),
        data.get("astrometric_params_solved"),
    )


def apply_parallax_zero_point(
    data: pd.DataFrame,
    *,
    zpt_fn=None,
) -> Tuple[pd.DataFrame, bool]:
    """
    Apply the Gaia DR3 parallax zero-point correction, subtracting the
    per-star zero-point offset (mas) from the observed parallax:

        corrected_parallax = observed_parallax - zero_point

    The DR3 zero-point is typically NEGATIVE (~-0.017 mas on average,
    Lindegren et al. 2021), so subtracting it makes corrected_parallax >
    observed_parallax, which SHRINKS the inferred 1/parallax distance.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'parallax'. `zpt_fn`'s own column requirements (if
        provided) are the caller's responsibility.
    zpt_fn : callable, optional
        `pd.DataFrame -> np.ndarray` of per-star zero-point offsets (mas).
        If given, it is used directly regardless of `HAVE_ZEROPOINT` — this
        is the injection point used to test the "package available" branch
        without requiring the real optional dependency to be installed. If
        None, the real `gaiadr3-zeropoint` package is used when
        `HAVE_ZEROPOINT` is True; otherwise a warning is emitted and the
        input is returned unmodified.

    Returns
    -------
    (corrected_data, zero_point_corrected) : tuple[pd.DataFrame, bool]
        `zero_point_corrected` is True only if the correction was ACTUALLY
        applied (never silently swallowed — see module/design docstrings).
    """
    if zpt_fn is None:
        if not HAVE_ZEROPOINT:
            warnings.warn(
                "gaiadr3-zeropoint package not available; proceeding with "
                "UNCORRECTED parallaxes (zero_point_corrected=False). "
                "Install the optional 'gaiadr3-zeropoint' dependency for a "
                "zero-point-corrected z_sun_pc estimate (see requirements.txt).",
                UserWarning,
                stacklevel=2,
            )
            return data, False
        zpt_fn = _default_zpt_fn

    zpt = np.asarray(zpt_fn(data), dtype=float)
    corrected = data.copy()
    corrected["parallax"] = data["parallax"].values.astype(float) - zpt
    return corrected, True


def cartesian_covariances(data: pd.DataFrame) -> np.ndarray:
    """
    Per-star heliocentric-Cartesian position covariance (N, 3, 3), kpc^2:
    rank-1 and line-of-sight-dominated (parallax noise only moves a star
    along its own line of sight, to first order):

        C_i = sigma_d_i^2 * outer(u_i, u_i)

    where `u_i` is the unit LOS direction (cos dec cos ra, cos dec sin ra,
    sin dec) and `sigma_d_i = d_i^2 * parallax_error_i` — see the module
    docstring's "Error propagation" note for the derivation (and why no
    extra unit-conversion factor is needed under this module's
    `d_kpc = 1/parallax_mas` convention).

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (deg), 'dec' (deg), 'parallax' (mas, > 0),
        'parallax_error' (mas, > 0).

    Returns
    -------
    np.ndarray, shape (N, 3, 3), kpc^2.
    """
    _check_required_columns(data)
    if "parallax_error" not in data.columns:
        raise ValueError("DataFrame is missing required column: 'parallax_error'")

    pts, d_kpc = _heliocentric_cartesian_kpc(data)
    u = pts / d_kpc[:, None]
    sigma_plx = data["parallax_error"].values.astype(float)
    sigma_d_kpc = d_kpc ** 2 * sigma_plx

    # (N,3,3): outer product of each row of u with itself, scaled by sigma_d^2.
    cov = sigma_d_kpc[:, None, None] ** 2 * np.einsum("ni,nj->nij", u, u)
    return cov


def _initial_normal(pts: np.ndarray, with_offset: bool) -> np.ndarray:
    """Unweighted SVD seed for the IRLS loop (mean-centered if with_offset)."""
    if with_offset:
        centroid = pts.mean(axis=0)
        centered = pts - centroid
    else:
        centered = pts
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal


def weighted_tls_plane(
    data: pd.DataFrame,
    *,
    parallax_over_error_min: Optional[float] = None,
    zero_point: bool = True,
    zpt_fn=None,
    with_offset: bool = True,
    max_iter: int = 10,
    tol: float = 1e-8,
) -> dict:
    """
    Covariance-weighted total-least-squares (TLS) plane fit via IRLS
    (design ADR6), replacing `ngp_3d.ngp_3d_pipeline`'s RANSAC/
    `residual_threshold` inlier gate with per-star uncertainty weighting.
    With `with_offset=True` (default) the fitted plane has a free offset
    (mean-centered weighted PCA, same trick as `ngp_offset_plane.
    offset_plane_pole`), recovering a combined F1+F3 `z_sun_pc` from the 3D
    fit itself.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (deg), 'dec' (deg), 'parallax' (mas, > 0),
        'parallax_error' (mas, > 0).
    parallax_over_error_min : float, optional
        If given, restrict the fit to stars with
        `parallax / parallax_error > parallax_over_error_min` (S/N cut,
        e.g. 10). `n_used` reports the post-cut count; `sn_cut` records the
        threshold used (None if no cut applied).
    zero_point : bool
        If True (default), apply `apply_parallax_zero_point` before
        anything else (graceful no-op fallback if the optional package is
        unavailable — see that function's docstring). If False, the
        zero-point step is skipped entirely (no warning), and
        `zero_point_corrected` is False.
    zpt_fn : callable, optional
        Forwarded to `apply_parallax_zero_point` (injection point for
        testing/using a custom correction).
    with_offset : bool
        If True (default) fit a free-offset plane (mean-centered weighted
        PCA) and report `z_sun_pc`. If False, fit a through-origin plane
        (weighted `great_circle_pole` analogue; `z_sun_pc` is forced to 0).
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance on `1 - |cos(angle between successive
        normals)|`.

    Returns
    -------
    dict with keys:
        alpha_NGP, delta_NGP : float — pole (hours, degrees)
        z_sun_pc, z_sun_err_pc : float — solar offset (pc); 0.0/nan if
            `with_offset=False`
        normal : np.ndarray(3,)
        covariance : np.ndarray(3,3) — final weighted scatter matrix
        n_stars : int — input row count (pre-S/N-cut)
        n_used : int — row count actually used in the fit (post-S/N-cut)
        zero_point_corrected : bool — see `apply_parallax_zero_point`
        sn_cut : float or None
        method : "weighted_tls"

    Raises
    ------
    ValueError : empty/missing-column input, non-positive parallax, or an
        S/N cut that removes every star.
    """
    _check_required_columns(data)
    if "parallax_error" not in data.columns:
        raise ValueError("DataFrame is missing required column: 'parallax_error'")

    n_stars = len(data)

    zero_point_corrected = False
    working = data
    if zero_point:
        working, zero_point_corrected = apply_parallax_zero_point(data, zpt_fn=zpt_fn)

    sn_cut = parallax_over_error_min
    if parallax_over_error_min is not None:
        sn = working["parallax"].values.astype(float) / working["parallax_error"].values.astype(float)
        mask = sn > parallax_over_error_min
        working = working.loc[mask]

    n_used = len(working)
    if n_used == 0:
        raise ValueError(
            "No stars remain after the S/N cut "
            f"(parallax_over_error_min={parallax_over_error_min})."
        )

    pts, d_kpc = _heliocentric_cartesian_kpc(working)
    cov = cartesian_covariances(working)

    normal = _initial_normal(pts, with_offset)
    weighted_pts = None
    centroid = np.zeros(3)

    for _ in range(max_iter):
        sigma_perp2 = np.einsum("i,nij,j->n", normal, cov, normal)
        sigma_perp2 = np.clip(sigma_perp2, 1e-15, None)
        w = 1.0 / sigma_perp2
        w = w / w.sum()

        if with_offset:
            centroid = (w[:, None] * pts).sum(axis=0)
            centered = pts - centroid
        else:
            centroid = np.zeros(3)
            centered = pts

        weighted_pts = centered * np.sqrt(w)[:, None]
        _, _, Vt = np.linalg.svd(weighted_pts, full_matrices=False)
        new_normal = Vt[-1]
        if new_normal[2] < 0:
            new_normal = -new_normal

        cos_angle = np.clip(abs(np.dot(new_normal, normal)), -1.0, 1.0)
        converged = (1.0 - cos_angle) < tol
        normal = new_normal
        if converged:
            break

    if with_offset:
        z_sun_kpc = float(np.dot(normal, centroid))
        z_sun_pc = z_sun_kpc * 1000.0
        perp_kpc = (pts - centroid) @ normal
        n_eff = 1.0 / np.sum(w ** 2)
        sigma_perp_kpc = float(np.std(perp_kpc))
        z_sun_err_pc = sigma_perp_kpc / np.sqrt(max(n_eff, 1.0)) * 1000.0
    else:
        z_sun_pc = 0.0
        z_sun_err_pc = float("nan")

    covariance = weighted_pts.T @ weighted_pts
    alpha_h, delta_deg = normal_to_equatorial(*normal)

    return {
        "alpha_NGP": alpha_h,
        "delta_NGP": delta_deg,
        "z_sun_pc": z_sun_pc,
        "z_sun_err_pc": z_sun_err_pc,
        "normal": normal,
        "covariance": covariance,
        "n_stars": int(n_stars),
        "n_used": int(n_used),
        "zero_point_corrected": bool(zero_point_corrected),
        "sn_cut": sn_cut,
        "method": "weighted_tls",
    }
