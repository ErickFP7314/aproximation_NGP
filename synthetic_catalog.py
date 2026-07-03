"""
synthetic_catalog.py — configurable synthetic Gaia-like disk-star catalog
generator with known ground truth (F5.4).

This is the shared test substrate consumed by every later ngp-precision
estimator module (F1 ngp_offset_plane, F3 ngp_weighted_3d, F4 ngp_kinematic,
F5 systematics): a single, frozen parameter contract (see design §4 of
`sdd/ngp-precision/design`) that can inject a known pole, a solar offset from
the Galactic plane (z_sun), a toy Galactic warp, asymmetric extinction
masking, heteroscedastic parallax noise, and a physically-consistent
rotating-disk proper-motion / radial-velocity field.

With every new-effect parameter left at its default (all off), the output is
statistically equivalent to the legacy `tests/conftest.py::synthetic_disk_stars`
fixture from `ngp-improvement` (same N, seed, pole, outlier fraction and
parallax range) — that fixture is now a thin wrapper around this function.

Ground truth is stashed in ``df.attrs["truth"]`` so downstream tests can
assert against it directly instead of re-deriving it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# v[km/s] = pm[mas/yr] * _KM_S_PER_MAS_YR_KPC * d[kpc]  (standard astrometric
# transverse-velocity relation, e.g. Binney & Merrifield eq. 10.13).
_KM_S_PER_MAS_YR_KPC = 4.74047


def _pole_basis(pole_ra_deg: float, pole_dec_deg: float):
    """Orthonormal basis (pole, u, v) with `pole` as the plane normal."""
    ra0 = np.radians(pole_ra_deg)
    dec0 = np.radians(pole_dec_deg)
    pole = np.array([
        np.cos(dec0) * np.cos(ra0),
        np.cos(dec0) * np.sin(ra0),
        np.sin(dec0),
    ])
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(pole, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(pole, ref)
    u /= np.linalg.norm(u)
    v = np.cross(pole, u)
    v /= np.linalg.norm(v)
    return pole, u, v


def _cartesian_to_radec(pts: np.ndarray):
    norms = np.linalg.norm(pts, axis=1)
    unit = pts / norms[:, None]
    dec_rad = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
    ra_rad = np.arctan2(unit[:, 1], unit[:, 0])
    return (np.degrees(ra_rad) % 360.0), np.degrees(dec_rad), norms


def _radec_to_galactic(ra_deg, dec_deg, ra0, dec0):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    b_rad = np.arcsin(
        np.sin(dec_rad) * np.sin(dec0)
        + np.cos(dec_rad) * np.cos(dec0) * np.cos(ra_rad - ra0)
    )
    l_rad = np.arctan2(
        np.cos(dec_rad) * np.sin(ra_rad - ra0),
        np.cos(dec_rad) * np.sin(dec0) * np.cos(ra_rad - ra0)
        - np.sin(dec_rad) * np.cos(dec0),
    )
    return (np.degrees(l_rad) % 360.0), np.degrees(b_rad)


def synthetic_catalog(
    *,
    n: int = 500,
    seed: int = 42,
    pole: Tuple[float, float] = (192.75, 27.11),
    z_sun_pc: float = 0.0,
    warp_amplitude_deg: float = 0.0,
    warp_onset_radius_kpc: float = 8.0,
    extinction_mask_fraction: float = 0.0,
    extinction_mask_region: Optional[str] = None,
    parallax_error_min_mas: float = 0.01,
    parallax_error_max_mas: float = 0.2,
    heteroscedastic: bool = False,
    outlier_fraction: float = 0.10,
    dist_range_kpc: Tuple[float, float] = (0.1, 0.5),
    include_proper_motion: bool = False,
    rotation_axis: Optional[Sequence[float]] = None,
    v_circ_kms: float = 220.0,
    solar_motion: Tuple[float, float, float] = (11.1, 12.24, 7.25),
    include_rvs: bool = False,
    rvs_fraction: float = 0.0,
) -> pd.DataFrame:
    """
    Build a synthetic disk-star catalog with known ground truth.

    Parameters
    ----------
    n : total number of stars.
    seed : RNG seed (deterministic — same seed -> identical output).
    pole : (ra_deg, dec_deg) of the injected NGP / disk-normal direction.
    z_sun_pc : solar offset from the Galactic plane, in parsecs. The disk
        plane satisfies ``pole . r = z_sun_pc/1000`` (kpc) in the heliocentric
        frame; z_sun_pc=0 reduces to a through-origin great circle.
    warp_amplitude_deg, warp_onset_radius_kpc : toy linear-in-distance warp:
        for stars beyond `warp_onset_radius_kpc` (heliocentric distance, kpc)
        an extra out-of-plane displacement of
        ``tan(warp_amplitude_deg) * (d_kpc - warp_onset_radius_kpc)`` is added
        along the pole direction. With the default `dist_range_kpc` (nearby
        sample) this never triggers even if amplitude != 0.
    extinction_mask_fraction, extinction_mask_region : randomly *drop* this
        fraction of stars from the named region ("b_positive" -> b>0,
        "b_negative" -> b<0, None -> whole sky) to emulate dust extinction
        creating an asymmetric selection function.
    parallax_error_min_mas, parallax_error_max_mas : uniform parallax_error
        range (mas); if `heteroscedastic`, scaled up with distance.
    outlier_fraction : fraction of stars placed uniformly on the sky
        (isotropic "wrong" population), matching the legacy fixture's 10%.
    dist_range_kpc : (d_min, d_max) heliocentric distance span (kpc) for
        inliers; d_min == d_max produces a degenerate single-distance-shell
        catalog (used to exercise `offset_plane_pole`'s degeneracy guard).
    include_proper_motion : if True, populate physically-consistent pmra/pmdec
        for a disk rotating about `rotation_axis` (default: `pole`) at
        `v_circ_kms`, with the Sun's `solar_motion` reflex subtracted. If
        False (default), pmra/pmdec are unstructured placeholder noise,
        matching the legacy fixture.
    rotation_axis : (x, y, z) unit-ish vector; defaults to the pole direction.
    include_rvs, rvs_fraction : if True, populate `radial_velocity` (km/s) for
        a `rvs_fraction` random subsample of stars (NaN elsewhere).

    Returns
    -------
    pd.DataFrame with columns:
        ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, l, b
        [, radial_velocity]
    and ``df.attrs["truth"]`` holding the injected ground truth (pole,
    z_sun_pc, warp params, outlier_fraction, rotation_axis, v_circ_kms,
    solar_motion, seed).
    """
    rng = np.random.default_rng(seed)
    pole_ra_deg, pole_dec_deg = pole
    ra0 = np.radians(pole_ra_deg)
    dec0 = np.radians(pole_dec_deg)
    pole_vec, u, v = _pole_basis(pole_ra_deg, pole_dec_deg)

    z_sun_kpc = z_sun_pc / 1000.0

    n_outliers = int(round(n * outlier_fraction)) if outlier_fraction > 0 else 0
    if outlier_fraction > 0:
        n_outliers = max(1, n_outliers)
    n_inliers = n - n_outliers

    d_min, d_max = dist_range_kpc
    if d_min == d_max:
        d_kpc = np.full(n_inliers, d_min)
    else:
        d_kpc = rng.uniform(d_min, d_max, n_inliers)

    theta = rng.uniform(0.0, 2 * np.pi, n_inliers)
    pts = d_kpc[:, None] * (
        np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    )

    # z_sun offset: constant physical shift along the pole direction — this
    # is the free offset in the plane equation n.r = z_sun (kpc), NOT scaled
    # by distance, so its angular effect naturally shrinks as d -> infinity
    # (matches the delta_vs_distance_shells extrapolation in F1).
    pts = pts + z_sun_kpc * pole_vec

    # Toy linear warp beyond the onset radius.
    if warp_amplitude_deg != 0.0 and n_inliers > 0:
        beyond_kpc = np.clip(d_kpc - warp_onset_radius_kpc, 0.0, None)
        warp_kpc = np.tan(np.radians(warp_amplitude_deg)) * beyond_kpc
        pts = pts + warp_kpc[:, None] * pole_vec

    # Fixed ~0.5deg-equivalent angular scatter about the plane, referenced to
    # the median inlier distance (keeps recovered-pole precision comparable
    # to the legacy fixture regardless of the chosen dist_range_kpc).
    if n_inliers > 0:
        scatter_kpc = float(np.median(d_kpc)) * np.radians(0.5)
        pts = pts + rng.normal(0.0, scatter_kpc, pts.shape)

    if n_outliers > 0:
        if d_min == d_max:
            d_out = np.full(n_outliers, d_min)
        else:
            d_out = rng.uniform(d_min, d_max, n_outliers)
        phi_out = rng.uniform(0, np.pi, n_outliers)
        lam_out = rng.uniform(0, 2 * np.pi, n_outliers)
        pts_out = d_out[:, None] * np.column_stack([
            np.sin(phi_out) * np.cos(lam_out),
            np.sin(phi_out) * np.sin(lam_out),
            np.cos(phi_out),
        ])
        pts = np.vstack([pts, pts_out])
        d_all = np.concatenate([d_kpc, d_out])
    else:
        d_all = d_kpc

    ra_deg, dec_deg, _ = _cartesian_to_radec(pts)
    l_deg, b_deg = _radec_to_galactic(ra_deg, dec_deg, ra0, dec0)

    d_safe = np.clip(d_all, 1e-9, None)
    parallax_mas = 1.0 / d_safe

    if heteroscedastic:
        base_err = rng.uniform(parallax_error_min_mas, parallax_error_max_mas, n)
        parallax_error = base_err * (1.0 + d_safe / np.median(d_safe))
    else:
        parallax_error = rng.uniform(parallax_error_min_mas, parallax_error_max_mas, n)

    df = pd.DataFrame({
        "ra": ra_deg,
        "dec": dec_deg,
        "parallax": parallax_mas,
        "parallax_error": parallax_error,
        "phot_g_mean_mag": rng.uniform(10, 15, n),
        "l": l_deg,
        "b": b_deg,
    })

    axis_used = None
    if include_proper_motion:
        axis_used = (
            np.asarray(rotation_axis, dtype=float)
            if rotation_axis is not None
            else pole_vec
        )
        axis_used = axis_used / np.linalg.norm(axis_used)

        pos_unit = pts / np.linalg.norm(pts, axis=1)[:, None]
        v_rot = v_circ_kms * np.cross(np.tile(axis_used, (n, 1)), pos_unit)

        solar = np.asarray(solar_motion, dtype=float)
        solar_radial = (pos_unit @ solar)[:, None] * pos_unit
        solar_tan = solar[None, :] - solar_radial
        # Apparent velocity = true stellar velocity minus the Sun's own
        # tangential reflex motion.
        v_app = v_rot - solar_tan

        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        e_alpha = np.column_stack([-np.sin(ra_rad), np.cos(ra_rad), np.zeros(n)])
        e_delta = np.column_stack([
            -np.sin(dec_rad) * np.cos(ra_rad),
            -np.sin(dec_rad) * np.sin(ra_rad),
            np.cos(dec_rad),
        ])
        v_alpha = np.einsum("ij,ij->i", v_app, e_alpha)
        v_delta = np.einsum("ij,ij->i", v_app, e_delta)
        df["pmra"] = v_alpha / (_KM_S_PER_MAS_YR_KPC * d_safe)
        df["pmdec"] = v_delta / (_KM_S_PER_MAS_YR_KPC * d_safe)

        if include_rvs and rvs_fraction > 0:
            v_radial = np.einsum("ij,ij->i", v_app, pos_unit)
            n_rvs = int(round(n * rvs_fraction))
            rv_col = np.full(n, np.nan)
            if n_rvs > 0:
                rvs_idx = rng.choice(n, size=n_rvs, replace=False)
                rv_col[rvs_idx] = v_radial[rvs_idx]
            df["radial_velocity"] = rv_col
    else:
        # Back-compat placeholder: unstructured noise, not physically
        # meaningful — matches the legacy fixture's pmra/pmdec columns.
        df["pmra"] = rng.normal(0, 5, n)
        df["pmdec"] = rng.normal(0, 5, n)
        if include_rvs and rvs_fraction > 0:
            n_rvs = int(round(n * rvs_fraction))
            rv_col = np.full(n, np.nan)
            if n_rvs > 0:
                rvs_idx = rng.choice(n, size=n_rvs, replace=False)
                rv_col[rvs_idx] = rng.normal(0, 20, n_rvs)
            df["radial_velocity"] = rv_col

    if extinction_mask_fraction > 0:
        if extinction_mask_region == "b_positive":
            candidates = df.index[df["b"] > 0]
        elif extinction_mask_region == "b_negative":
            candidates = df.index[df["b"] < 0]
        else:
            candidates = df.index
        n_mask = int(round(len(candidates) * extinction_mask_fraction))
        if n_mask > 0:
            drop_idx = rng.choice(np.asarray(candidates), size=n_mask, replace=False)
            df = df.drop(index=drop_idx).reset_index(drop=True)

    df.attrs["truth"] = {
        "pole": (pole_ra_deg, pole_dec_deg),
        "z_sun_pc": z_sun_pc,
        "warp_amplitude_deg": warp_amplitude_deg,
        "warp_onset_radius_kpc": warp_onset_radius_kpc,
        "outlier_fraction": outlier_fraction,
        "n_outliers": n_outliers,
        "n_inliers": n_inliers,
        "dist_range_kpc": tuple(dist_range_kpc),
        "rotation_axis": tuple(axis_used) if axis_used is not None else None,
        "v_circ_kms": v_circ_kms,
        "solar_motion": tuple(solar_motion),
        "seed": seed,
    }
    return df
