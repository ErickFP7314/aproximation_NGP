"""
ngp_kinematic.py — F4: kinematic (proper-motion-based) North Galactic Pole
estimator — the disk's rotation axis, derived from the stars' TANGENTIAL
VELOCITY field, physically INDEPENDENT of every geometric/positional pole
estimator elsewhere in this study (`ngp_3d.great_circle_pole`,
`ngp_offset_plane.offset_plane_pole`, `ngp_weighted_3d.weighted_tls_plane`
all use star POSITIONS on the sky / in 3D space; this module uses star
VELOCITIES instead). Cross-agreement between this kinematic pole and the
geometric poles -- two physically unrelated observables converging on the
same axis -- is the scientific headline of the `ngp-precision` study
(`sdd/ngp-precision/design`, ADR7).

Public API
----------
kinematic_pole(data, *, solar_motion=(11.1,12.24,7.25), r_sun_kpc=8.122,
                v_circ_kms=220.0) -> dict
    {alpha_NGP, delta_NGP, normal, covariance, n_stars, solar_motion,
     method: "kinematic_pm"}
    Raises ValueError if pmra/pmdec are (numerically) all zero -- degenerate
    input, no rotation signal to fit an axis from.

kinematic_pole_rvs(data, *, solar_motion=..., r_sun_kpc=8.122,
                    v_circ_kms=220.0) -> dict | None
    3D angular-momentum variant, using the (non-NaN) `radial_velocity`
    subsample for the full 3D velocity vector. Returns None (and logs the
    reason) if that subsample is empty or the column is absent -- this is
    NOT an error, since RVS coverage is optional/sparse in real Gaia data
    (design F4-R2).

Method (design ADR7)
---------------------
A star on a purely circular disk orbit has velocity
    v_rot = v_circ * (axis x r_hat)
which is, by construction of the cross product, PERPENDICULAR to the
rotation axis `axis` for every star regardless of where it sits on its
orbit or how far it is from the Sun. So the axis that best explains an
ensemble of disk velocities is the one minimizing

    Sum_i (axis . v_i)^2

i.e. the eigenvector of SMALLEST eigenvalue of the velocity
outer-product/scatter matrix ``Sum_i v_i v_i^T`` -- equivalently the
right-singular-vector of smallest singular value of the (N,3) velocity
matrix. This is a through-the-origin direction fit, exactly like
`ngp_3d.great_circle_pole` (no mean-centering -- there is no "offset"
analogue here, unlike `ngp_offset_plane.offset_plane_pole`'s free-offset
position fit).

Before that fit, each star's APPARENT (Gaia-observed) tangential velocity
must have the Sun's own reflex motion added back, to recover the star's
velocity relative to the Galactic rest frame. This EXACTLY inverts
`synthetic_catalog.py`'s forward model (`include_proper_motion=True`
branch):

    v_app = v_rot - solar_tan       (synthetic_catalog.py forward model)
    =>  v_rot = v_app + solar_tan   (this module's inversion)

where `solar_tan` is the component of the Sun's (U, V, W) peculiar motion
PERPENDICULAR to the line of sight (`solar_motion` parameter here); its
parallel/radial component never shows up in proper motion and is
irrelevant to this tangential fit (it IS used by `kinematic_pole_rvs`,
which additionally needs the radial velocity).

Units: v_transverse[km/s] = 4.74047 * pm[mas/yr] * d[kpc] -- the same
`_KM_S_PER_MAS_YR_KPC` constant used by `synthetic_catalog.py`.
d_kpc = 1 / parallax_mas -- the physically-correct convention adopted
throughout `ngp-precision` (see `ngp_offset_plane.py`'s module docstring
"Unit note"), deliberately NOT `ngp_3d.coords_to_cartesian`'s mislabeled
``d = 1000/parallax`` (pc, not kpc) convention -- see B1 apply-progress
"units gotcha", still binding here since this module also needs a genuine
physical distance scale (to turn pm into a physical velocity), not just a
direction.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ngp_3d import normal_to_equatorial

logger = logging.getLogger(__name__)

# v[km/s] = pm[mas/yr] * _KM_S_PER_MAS_YR_KPC * d[kpc] -- identical constant
# and relation used by synthetic_catalog.py's forward model.
_KM_S_PER_MAS_YR_KPC = 4.74047

_REQUIRED_COLUMNS = ("ra", "dec", "parallax", "pmra", "pmdec")
_ZERO_PM_ATOL = 1e-8


def _check_required_columns(data: pd.DataFrame) -> None:
    if data is None or len(data) == 0:
        raise ValueError("Input DataFrame is empty.")
    for col in _REQUIRED_COLUMNS:
        if col not in data.columns:
            raise ValueError(f"DataFrame is missing required column: '{col}'")


def _sky_basis(ra_rad: np.ndarray, dec_rad: np.ndarray):
    """
    Per-star orthonormal (pos_unit, e_alpha, e_delta) tangent-sphere basis
    -- the EXACT same basis synthetic_catalog.py's forward model uses to
    project velocities into pmra/pmdec, needed here to invert that
    projection consistently.
    """
    pos_unit = np.column_stack([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad),
    ])
    e_alpha = np.column_stack([
        -np.sin(ra_rad), np.cos(ra_rad), np.zeros_like(ra_rad),
    ])
    e_delta = np.column_stack([
        -np.sin(dec_rad) * np.cos(ra_rad),
        -np.sin(dec_rad) * np.sin(ra_rad),
        np.cos(dec_rad),
    ])
    return pos_unit, e_alpha, e_delta


def _reflex_corrected_tangential_velocity(
    data: pd.DataFrame, solar_motion: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (v_rot_tan (N,3) km/s, pos_unit (N,3)): the Galactic-rest-frame
    tangential velocity (solar reflex added back) and the per-star
    line-of-sight unit vector, shared by `kinematic_pole` and
    `kinematic_pole_rvs`.
    """
    plx = data["parallax"].values.astype(float)
    if np.any(plx <= 0):
        raise ValueError(
            "All parallax values must be > 0. Found non-positive parallax."
        )
    d_kpc = 1.0 / plx

    ra_rad = np.radians(data["ra"].values.astype(float))
    dec_rad = np.radians(data["dec"].values.astype(float))
    pos_unit, e_alpha, e_delta = _sky_basis(ra_rad, dec_rad)

    pmra = data["pmra"].values.astype(float)
    pmdec = data["pmdec"].values.astype(float)
    v_alpha = pmra * _KM_S_PER_MAS_YR_KPC * d_kpc
    v_delta = pmdec * _KM_S_PER_MAS_YR_KPC * d_kpc
    v_app_tan = v_alpha[:, None] * e_alpha + v_delta[:, None] * e_delta

    solar = np.asarray(solar_motion, dtype=float)
    solar_radial = (pos_unit @ solar)[:, None] * pos_unit
    solar_tan = solar[None, :] - solar_radial

    # Invert synthetic_catalog.py's forward model: v_app = v_rot - solar_tan.
    v_rot_tan = v_app_tan + solar_tan
    return v_rot_tan, pos_unit


def kinematic_pole(
    data: pd.DataFrame,
    *,
    solar_motion: Tuple[float, float, float] = (11.1, 12.24, 7.25),
    r_sun_kpc: float = 8.122,
    v_circ_kms: float = 220.0,
) -> dict:
    """
    Estimate the kinematic NGP (disk rotation axis) from proper motions.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (deg), 'dec' (deg), 'parallax' (mas, > 0),
        'pmra', 'pmdec' (mas/yr, Gaia convention -- pmra already includes
        the cos(dec) factor).
    solar_motion : (U, V, W) km/s
        Sun's peculiar motion, subtracted (added back, from the observer's
        perspective) before fitting the axis. Explicit, documented
        parameter (design F4-R3): using the WRONG solar_motion leaves
        residual solar reflex in the fit and biases the recovered axis
        toward the solar apex direction.
    r_sun_kpc, v_circ_kms : float
        Accepted for API-signature parity with the design contract
        (`sdd/ngp-precision/design`, F4) and potential future weighting
        use (e.g. the F5 systematics batch). The axis fit itself is
        direction-only (scale- and Galactocentric-radius-invariant), so
        neither is currently used inside this function.

    Returns
    -------
    dict with keys:
        alpha_NGP, delta_NGP : float -- pole (hours, degrees)
        normal : np.ndarray(3,) -- fitted rotation-axis unit vector
        covariance : np.ndarray(3,3) -- velocity outer-product ("scatter")
            matrix used for the eigen-decomposition (informational)
        n_stars : int
        solar_motion : tuple(float, float, float) -- echoes the input
        method : "kinematic_pm"

    Raises
    ------
    ValueError : empty/missing-column input, non-positive parallax, or
        all-zero proper motions (degenerate -- no rotation signal).
    """
    _check_required_columns(data)

    pmra = data["pmra"].values.astype(float)
    pmdec = data["pmdec"].values.astype(float)
    if (
        np.allclose(pmra, 0.0, atol=_ZERO_PM_ATOL)
        and np.allclose(pmdec, 0.0, atol=_ZERO_PM_ATOL)
    ):
        raise ValueError(
            "All proper motions are zero -- no rotation signal to fit a "
            "kinematic pole from (degenerate input)."
        )

    v_rot_tan, _ = _reflex_corrected_tangential_velocity(data, solar_motion)

    # Axis minimizing Sum (n . v)^2 == smallest right-singular-vector of the
    # (N,3) velocity matrix (through-origin, no mean-centering -- same style
    # as ngp_3d.great_circle_pole's direction-only SVD).
    _, _, Vt = np.linalg.svd(v_rot_tan, full_matrices=False)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal  # flip to the northern hemisphere

    covariance = v_rot_tan.T @ v_rot_tan
    alpha_h, delta_deg = normal_to_equatorial(*normal)

    return {
        "alpha_NGP": alpha_h,
        "delta_NGP": delta_deg,
        "normal": normal,
        "covariance": covariance,
        "n_stars": int(len(data)),
        "solar_motion": tuple(float(x) for x in solar_motion),
        "method": "kinematic_pm",
    }


def kinematic_pole_rvs(
    data: pd.DataFrame,
    *,
    solar_motion: Tuple[float, float, float] = (11.1, 12.24, 7.25),
    r_sun_kpc: float = 8.122,
    v_circ_kms: float = 220.0,
) -> Optional[dict]:
    """
    3D angular-momentum variant of the kinematic pole, using the full 3D
    velocity (tangential + radial) for the subsample of stars with a valid
    (non-NaN) `radial_velocity`.

    For a star on a circular orbit its specific angular momentum
    ``L = r x v`` points along the rotation axis (any velocity component
    parallel to `r` -- i.e. purely radial -- contributes nothing to the
    cross product, so this variant is robust to the radial-velocity
    reflex correction even where that correction is small or noisy). The
    recovered pole is the direction of the mean ``L`` over the RVS
    subsample.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra', 'dec', 'parallax', 'pmra', 'pmdec' AND
        'radial_velocity' (km/s, NaN where unavailable) for at least one
        row to produce a result.
    solar_motion, r_sun_kpc, v_circ_kms : see `kinematic_pole`.

    Returns
    -------
    dict with keys:
        alpha_NGP, delta_NGP : float -- pole (hours, degrees)
        normal : np.ndarray(3,) -- mean angular-momentum unit vector
        n_stars : int -- total input rows
        n_used : int -- rows with valid `radial_velocity` actually used
        solar_motion : tuple(float, float, float)
        method : "kinematic_rvs_angmom"
    or None (with a logged reason, NOT an exception) if `radial_velocity`
    is absent or its non-NaN subsample is empty (design F4-R2 -- RVS
    coverage is optional/sparse in real Gaia data).

    Raises
    ------
    ValueError : degenerate RVS subsample with ~zero net angular momentum
        (no rotation signal); missing 'ra'/'dec'/'parallax'/'pmra'/'pmdec'
        columns; non-positive parallax.
    """
    if data is None or len(data) == 0 or "radial_velocity" not in data.columns:
        logger.info(
            "kinematic_pole_rvs: no 'radial_velocity' column present -- "
            "returning None (RVS variant unavailable)."
        )
        return None

    rv_mask = data["radial_velocity"].notna().values
    if not np.any(rv_mask):
        logger.info(
            "kinematic_pole_rvs: 'radial_velocity' subsample is empty "
            "(all NaN) -- returning None (RVS variant unavailable)."
        )
        return None

    sub = data.loc[rv_mask]
    _check_required_columns(sub)

    v_rot_tan, pos_unit = _reflex_corrected_tangential_velocity(sub, solar_motion)

    plx = sub["parallax"].values.astype(float)
    d_kpc = 1.0 / plx

    solar = np.asarray(solar_motion, dtype=float)
    solar_radial_scalar = pos_unit @ solar  # (N,) -- Sun's LOS-projected motion
    rv_app = sub["radial_velocity"].values.astype(float)
    # Invert: rv_app = v_star_radial - solar_radial_scalar.
    v_rot_radial = rv_app + solar_radial_scalar

    v_rot_3d = v_rot_tan + v_rot_radial[:, None] * pos_unit
    r_vec = d_kpc[:, None] * pos_unit  # kpc; only direction matters downstream

    angular_momentum = np.cross(r_vec, v_rot_3d)
    mean_L = angular_momentum.mean(axis=0)
    norm = np.linalg.norm(mean_L)
    if norm < 1e-12:
        raise ValueError(
            "Degenerate RVS angular-momentum fit: mean angular momentum "
            "is ~zero (no net rotation signal in the RVS subsample)."
        )
    normal = mean_L / norm
    if normal[2] < 0:
        normal = -normal  # flip to the northern hemisphere

    alpha_h, delta_deg = normal_to_equatorial(*normal)

    return {
        "alpha_NGP": alpha_h,
        "delta_NGP": delta_deg,
        "normal": normal,
        "n_stars": int(len(data)),
        "n_used": int(len(sub)),
        "solar_motion": tuple(float(x) for x in solar_motion),
        "method": "kinematic_rvs_angmom",
    }
