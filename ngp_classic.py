"""
ngp_classic.py — Clean-room reimplementation of three 2D NGP estimation methods.

All functions are pure (DataFrame in, dict out, no global state, no I/O).
No imports from the author's original files.

Public API:
    aprox_ar(data, ar_ref)   -> dict  (pair-symmetry; alpha_NGP in hours)
    aprox_ar_svd(data)       -> dict  (SVD great-circle normal; alpha_NGP in degrees)
    aprox_dec1(data, n)      -> dict
    aprox_dec2(data, delta)  -> dict

aprox_ar_svd and aprox_dec1/dec2 return:
    {alpha_NGP: float (degrees), delta_NGP: float (degrees),
     std_alpha: float, std_delta: float}

aprox_ar (pair-symmetry) returns:
    {alpha_NGP: float (hours), delta_NGP: None,
     std_alpha: float, method: "pair_symmetry"}
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_nonempty(data: pd.DataFrame, req_cols=("ra", "dec")) -> None:
    """Raise ValueError if data is empty or missing required columns."""
    if data is None or len(data) == 0:
        raise ValueError("Input DataFrame is empty.")
    for col in req_cols:
        if col not in data.columns:
            raise ValueError(f"DataFrame is missing required column: '{col}'")


def _normal_to_equatorial(normal: np.ndarray):
    """
    Convert a unit normal vector to equatorial coordinates (RA degrees, Dec degrees).
    Ensures the normal points to the north hemisphere (dec >= 0).
    """
    if normal[2] < 0:
        normal = -normal
    alpha_deg = float(np.degrees(np.arctan2(normal[1], normal[0])) % 360)
    delta_deg = float(np.degrees(np.arcsin(np.clip(normal[2], -1.0, 1.0))))
    return alpha_deg, delta_deg


def _bootstrap_std(ra_deg, dec_deg, estimator_fn, n_boot=30, seed=42):
    """
    Estimate std_alpha and std_delta via bootstrap resampling of an estimator_fn.

    estimator_fn(ra_rad, dec_rad) -> (alpha_deg, delta_deg)
    """
    rng = np.random.default_rng(seed)
    n = len(ra_deg)
    alphas, deltas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, d = estimator_fn(ra_deg[idx], dec_deg[idx])
        alphas.append(a)
        deltas.append(d)
    return float(np.std(alphas)), float(np.std(deltas))


# ---------------------------------------------------------------------------
# Method 1a — Right Ascension pair-symmetry (faithful paper reimplementation)
# ---------------------------------------------------------------------------

def aprox_ar(data: pd.DataFrame, ar_ref: float = 12.816) -> dict:
    """
    Estimate NGP right ascension using the pair-symmetry method.

    Groups stars by rounded declination (0.1° bins). For each group with ≥2 stars,
    splits RA values (in hours) into those below and above ar_ref. If both sides
    are non-empty, the midpoint (mean_below + mean_above) / 2 is recorded.
    Only groups with stars on BOTH sides of ar_ref contribute.
    The mean of all per-group midpoints is returned as alpha_NGP.

    Methodological note: This method requires ar_ref as a prior (the theoretical
    NGP RA in hours). The result is anchored to this prior — a limitation of the
    original pair-symmetry algorithm (automatedAR.py, L. Cano 2021). For a
    prior-free estimate, use aprox_ar_svd().

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (degrees) and 'dec' (degrees) columns.
    ar_ref : float
        Reference RA in hours to split groups. Default 12.816h (≈ IAU NGP RA).

    Returns
    -------
    dict with keys:
        alpha_NGP : float (hours) — mean of per-group RA midpoints
        delta_NGP : None — not estimated by this method
        std_alpha : float — std of per-group midpoints (spread proxy)
        method    : "pair_symmetry"

    Raises
    ------
    ValueError : if data is empty, missing columns, or no valid groups found.
    """
    _validate_nonempty(data, req_cols=("ra", "dec"))

    df = data[["ra", "dec"]].copy()
    # Convert RA from degrees to hours for comparison with ar_ref
    df["ar_h"] = df["ra"] / 15.0
    # Round declination to 1 decimal place for grouping (mirrors original)
    df["dec_round"] = df["dec"].round(1)

    midpoints = []
    for _dec_val, group in df.groupby("dec_round"):
        ars = group["ar_h"].values
        if len(ars) < 2:
            continue  # not enough stars in this dec group
        below = ars[ars < ar_ref]
        above = ars[ars > ar_ref]
        # ars_check equivalent: exclude groups where all values are on one side
        if len(below) == 0 or len(above) == 0:
            continue
        midpoint = (float(np.mean(below)) + float(np.mean(above))) / 2.0
        midpoints.append(midpoint)

    if not midpoints:
        raise ValueError(
            "No valid declination groups found for pair-symmetry estimation. "
            "Every dec group had all RA values on the same side of ar_ref. "
            "Try a different ar_ref or use aprox_ar_svd() for a prior-free estimate."
        )

    return {
        "alpha_NGP": float(np.mean(midpoints)),
        "delta_NGP": None,
        "std_alpha": float(np.std(midpoints)),
        "method": "pair_symmetry",
    }


# ---------------------------------------------------------------------------
# Method 1b — Right Ascension by great-circle normal (SVD / pair-symmetry spirit)
# ---------------------------------------------------------------------------

def aprox_ar_svd(data: pd.DataFrame) -> dict:
    """
    Estimate NGP position from the great-circle normal to the observed star positions.

    Projects RA/Dec onto unit vectors on the celestial sphere, then uses SVD to
    find the direction of minimum variance — the normal to the best-fit great circle
    (the galactic plane). This is the geometric spirit of the pair-symmetry method:
    stars in the galactic disk appear symmetrically about the NGP RA, so the
    symmetry axis equals the great-circle normal. Unlike aprox_ar(), this method
    requires no prior on the NGP RA.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (degrees) and 'dec' (degrees) columns.

    Returns
    -------
    dict with keys alpha_NGP (degrees), delta_NGP (degrees), std_alpha, std_delta.

    Raises
    ------
    ValueError : if data is empty or missing required columns.
    """
    _validate_nonempty(data, req_cols=("ra", "dec"))

    ra_deg = data["ra"].values.astype(float)
    dec_deg = data["dec"].values.astype(float)

    def _svd_estimate(ra_d, dec_d):
        ra_r = np.radians(ra_d)
        dec_r = np.radians(dec_d)
        X = np.column_stack([
            np.cos(dec_r) * np.cos(ra_r),
            np.cos(dec_r) * np.sin(ra_r),
            np.sin(dec_r),
        ])
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        normal = Vt[-1]  # direction with smallest singular value = great-circle normal
        return _normal_to_equatorial(normal)

    alpha_NGP, delta_NGP = _svd_estimate(ra_deg, dec_deg)
    std_alpha, std_delta = _bootstrap_std(ra_deg, dec_deg, _svd_estimate, n_boot=30)

    return {
        "alpha_NGP": alpha_NGP,
        "delta_NGP": delta_NGP,
        "std_alpha": std_alpha,
        "std_delta": std_delta,
    }


# ---------------------------------------------------------------------------
# Method 2 — Declination by top-n stars
# ---------------------------------------------------------------------------

def aprox_dec1(data: pd.DataFrame, n: int = None) -> dict:
    """
    Estimate NGP declination using the n highest-|dec| stars.

    For each star in the top-n, NGP dec ≈ 90 - |dec_star|.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' and 'dec' columns.
    n : int, optional
        Number of top stars to use.  Defaults to max(1, len(data)//10).

    Returns
    -------
    dict with keys alpha_NGP, delta_NGP, std_alpha, std_delta.

    Raises
    ------
    ValueError : if data is empty, n < 1, or n > len(data).
    """
    _validate_nonempty(data, req_cols=("ra", "dec"))

    N = len(data)
    if n is None:
        n = max(1, N // 10)

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if n > N:
        raise ValueError(f"n={n} exceeds number of available rows ({N}).")

    df = data[["ra", "dec"]].copy()
    df["abs_dec"] = df["dec"].abs()
    top_n = df.nlargest(n, "abs_dec")

    delta_estimates = (90.0 - top_n["abs_dec"].values).tolist()
    alpha_estimates = top_n["ra"].values.tolist()

    return {
        "alpha_NGP": float(np.mean(alpha_estimates)),
        "delta_NGP": float(np.mean(delta_estimates)),
        "std_alpha": float(np.std(alpha_estimates)),
        "std_delta": float(np.std(delta_estimates)),
    }


# ---------------------------------------------------------------------------
# Method 3 — Declination by RA window
# ---------------------------------------------------------------------------

def aprox_dec2(data: pd.DataFrame, delta: float) -> dict:
    """
    Estimate NGP declination by filtering stars within ±delta hours of the NGP RA.

    The NGP RA is estimated first (via aprox_ar_svd great-circle method), then stars
    within the window [ra_NGP - delta*15, ra_NGP + delta*15] degrees are selected.
    NGP dec ≈ 90 - |dec| for each selected star.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (degrees) and 'dec' (degrees) columns.
    delta : float
        Half-width of the RA window in hours (0 < delta <= 12).

    Returns
    -------
    dict with keys alpha_NGP, delta_NGP, std_alpha, std_delta.

    Raises
    ------
    ValueError : if data is empty, delta <= 0, or delta > 12.
    """
    _validate_nonempty(data, req_cols=("ra", "dec"))

    if delta <= 0:
        raise ValueError(f"delta must be > 0 (hours), got {delta}.")
    if delta > 12:
        raise ValueError(f"delta must be <= 12 hours, got {delta}.")

    df = data[["ra", "dec"]].copy()

    # Estimate NGP RA via great-circle normal (SVD; no globals, no prior required)
    ra_est = aprox_ar_svd(data)["alpha_NGP"]
    half_window_deg = delta * 15.0  # hours → degrees

    mask = (df["ra"] >= ra_est - half_window_deg) & (df["ra"] <= ra_est + half_window_deg)
    window_stars = df[mask]

    if len(window_stars) == 0:
        window_stars = df  # fallback to all data

    abs_dec = window_stars["dec"].abs().values
    delta_estimates = (90.0 - abs_dec).tolist()
    alpha_estimates = window_stars["ra"].values.tolist()

    return {
        "alpha_NGP": float(np.mean(alpha_estimates)),
        "delta_NGP": float(np.mean(delta_estimates)),
        "std_alpha": float(np.std(alpha_estimates)),
        "std_delta": float(np.std(delta_estimates)),
    }
