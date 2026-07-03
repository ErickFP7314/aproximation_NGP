"""
systematics.py — honest systematic-error budget for the NGP pole estimate
(ngp-precision batch B5 / spec capability F5).

Turns the ad-hoc uncertainty of earlier batches into a defensible error bar
by combining:

  1. A one-factor-at-a-time (OFAT) *systematics matrix*: the chosen pole
     estimator is re-run under a battery of analysis-choice variants
     (galactic-latitude cut b_max, magnitude limit G_limit, parallax S/N
     cut, distance shell, sky hemisphere) starting from a common
     no-cuts baseline. The spread of the resulting poles across variants
     is the systematic uncertainty, sigma_syst (see `combine_error_budget`).
     A full Cartesian product over every axis combination would be
     combinatorially large and mostly redundant (interactions between
     independent selection cuts are a second-order effect); OFAT from a
     shared baseline isolates each axis's own contribution while keeping
     runtime linear in the number of grid values.
  2. Leave-one-sky-region-out jackknife (`sky_region_jackknife`): an
     independent, non-parametric cross-check of the statistical
     uncertainty that does not rely on resampling with replacement.
  3. `combine_error_budget`: sigma_total = sqrt(sigma_stat**2 + sigma_syst**2),
     consuming `bootstrap.bootstrap_great_circle_pole`'s existing return
     shape UNCHANGED (frozen API, not modified by this module).
  4. `extinction_asymmetry_check`: the key scientific validation that the
     method can tell a *real* systematic (e.g. asymmetric dust extinction
     masking one galactic hemisphere) apart from ordinary sampling noise:
     it splits the sample by galactic-latitude sign and flags the pole
     difference as systematic only if it exceeds 3 * sigma_stat.

Public API:
    systematics_grid(data, ...)                                -> pd.DataFrame
    sky_region_jackknife(data, ...)                            -> dict
    combine_error_budget(point_estimate, bootstrap_result, systematics_table) -> dict
    extinction_asymmetry_check(data, ...)                      -> dict
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bootstrap import bootstrap_great_circle_pole
from ngp_3d import great_circle_pole

logger = logging.getLogger("systematics")

# Frozen output schema (design section 4 / spec F5).
BUDGET_COLUMNS = [
    "method", "tracer", "b_max", "G_limit", "sn_cut", "shell", "hemisphere",
    "alpha", "delta", "sigma_stat", "sigma_syst", "sigma_total",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _angular_separation_deg(alpha1_h, delta1_deg, alpha2_h, delta2_deg) -> float:
    """Great-circle angular separation (deg) between two (RA-hours, Dec-deg) poles."""
    ra1, dec1 = np.radians(alpha1_h * 15.0), np.radians(delta1_deg)
    ra2, dec2 = np.radians(alpha2_h * 15.0), np.radians(delta2_deg)
    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    return float(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))))


def _sigma_stat_deg_from_bootstrap(bootstrap_result: dict, delta_ngp_deg: float) -> float:
    """
    Convert a `bootstrap_great_circle_pole` result (alpha_ci95 in hours,
    delta_ci95 in degrees) into a single scalar angular sigma_stat (degrees),
    combining the RA and Dec 95%-CI half-widths (assumed ~Gaussian, so
    half-width / 1.96 = 1 sigma) in quadrature. The RA term is scaled by
    cos(delta) to convert an RA-hours spread into a true angular-on-sky
    spread near the pole's declination.
    """
    a_lo, a_hi = bootstrap_result["alpha_ci95"]
    d_lo, d_hi = bootstrap_result["delta_ci95"]
    sigma_alpha_deg = (a_hi - a_lo) * 15.0 / (2.0 * 1.96) * np.cos(np.radians(delta_ngp_deg))
    sigma_delta_deg = (d_hi - d_lo) / (2.0 * 1.96)
    return float(np.hypot(sigma_alpha_deg, sigma_delta_deg))


def _run_estimator_safe(estimator: Callable, subset: pd.DataFrame, min_stars: int):
    """Run `estimator(subset)`, returning None (+ log) if too few stars or it raises."""
    if subset is None or len(subset) < min_stars:
        logger.info(
            "systematics_grid: skipping variant with %d stars (< min_stars=%d)",
            0 if subset is None else len(subset), min_stars,
        )
        return None
    try:
        return estimator(subset)
    except Exception as exc:  # defensive: a degenerate cut must not crash the grid
        logger.info("systematics_grid: skipping variant, estimator raised: %s", exc)
        return None


def _distance_kpc(data: pd.DataFrame) -> np.ndarray:
    """d_kpc = 1 / parallax_mas (B1's established convention; NOT coords_to_cartesian's)."""
    return 1.0 / data["parallax"].values.astype(float)


# ---------------------------------------------------------------------------
# B5.4 — systematics_grid
# ---------------------------------------------------------------------------

def systematics_grid(
    data: pd.DataFrame,
    *,
    estimator: Callable[[pd.DataFrame], dict] = great_circle_pole,
    tracer: str = "all",
    b_max_values: Sequence[Optional[float]] = (None, 15.0, 10.0, 5.0),
    g_limit_values: Sequence[Optional[float]] = (None, 15.0, 14.0, 13.0),
    sn_cut_values: Sequence[Optional[float]] = (None, 5.0, 10.0, 20.0),
    shell_edges_kpc: Optional[Sequence[float]] = None,
    hemispheres: Sequence[str] = ("all", "north", "south"),
    quadrants: bool = False,
    min_stars: int = 20,
    output_csv: Optional[str] = "results/systematics_budget.csv",
    output_json: Optional[str] = "results/systematics_budget.json",
    write_output: bool = True,
) -> pd.DataFrame:
    """
    Re-run `estimator` under a one-factor-at-a-time grid of analysis-choice
    variants and return (and optionally persist) the resulting pole table.

    The FIRST element of `b_max_values`, `g_limit_values`, `sn_cut_values`
    and `hemispheres` is the "no cut" / baseline value for that axis
    (``None`` for the three numeric cuts, ``"all"`` for hemisphere). Each
    subsequent value produces exactly one row where ONLY that axis's cut is
    applied and every other axis stays at baseline — this keeps the grid
    linear in the number of values per axis (unlike a full Cartesian
    product) while still isolating each systematic's own contribution.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra', 'dec' (required by `estimator`), plus whichever
        of 'b', 'phot_g_mean_mag', 'parallax'/'parallax_error' are needed by
        the requested cut axes.
    estimator : callable
        Pole estimator, e.g. `ngp_3d.great_circle_pole` (default) or any
        other `ngp-precision` estimator sharing the `{alpha_NGP, delta_NGP}`
        result-dict contract.
    tracer : str
        Label recorded in the output 'tracer' column (this batch does not
        itself merge multiple tracer catalogs -- callers wanting a
        multi-tracer budget call this once per tracer DataFrame and
        concatenate the returned tables).
    b_max_values, g_limit_values, sn_cut_values : sequences
        OFAT grid values for |b| < b_max, phot_g_mean_mag < G_limit, and
        parallax/parallax_error > sn_cut respectively.
    shell_edges_kpc : sequence, optional
        If given, adds one row per distance shell (d_kpc = 1/parallax_mas,
        B1 convention), each using only stars in that shell (baseline for
        every other axis).
    hemispheres : sequence
        "all" (baseline) plus any of "north" (b >= 0), "south" (b < 0).
    quadrants : bool
        If True, also add four galactic-longitude-quadrant variants,
        recorded in the 'hemisphere' column as "quad0".."quad3" (0-90,
        90-180, 180-270, 270-360 deg in l).
    min_stars : int
        Variants with fewer stars than this are skipped (logged, not
        raised) rather than producing a noisy/degenerate row.
    output_csv, output_json : str, optional
        Destination paths. Parent directories are created automatically.
        Ignored if `write_output` is False or either is None.
    write_output : bool
        If False, no files are written (useful for tests).

    Returns
    -------
    pd.DataFrame with columns `BUDGET_COLUMNS` (see module docstring). The
    'sigma_stat'/'sigma_syst'/'sigma_total' columns are NaN here (per-row
    bootstrap CIs are not computed by the grid for performance reasons);
    use `combine_error_budget` to derive the scalar sigma_syst/sigma_total
    summary from this table plus a bootstrap sigma_stat.
    """
    if data is None or len(data) == 0:
        raise ValueError("Input DataFrame is empty.")

    rows = []

    def _add_row(subset, *, b_max=None, g_limit=None, sn_cut=None, shell="all", hemisphere="all"):
        result = _run_estimator_safe(estimator, subset, min_stars)
        if result is None:
            return
        rows.append({
            "method": result.get("method", getattr(estimator, "__name__", "estimator")),
            "tracer": tracer,
            "b_max": b_max,
            "G_limit": g_limit,
            "sn_cut": sn_cut,
            "shell": shell,
            "hemisphere": hemisphere,
            "alpha": result["alpha_NGP"],
            "delta": result["delta_NGP"],
            "sigma_stat": np.nan,
            "sigma_syst": np.nan,
            "sigma_total": np.nan,
        })

    # --- baseline (no cuts at all) ---
    _add_row(data)

    # --- b_max axis ---
    if "b" in data.columns:
        for b_max in b_max_values[1:]:
            if b_max is None:
                continue
            subset = data[data["b"].abs() < b_max]
            _add_row(subset, b_max=b_max)

    # --- G_limit axis ---
    if "phot_g_mean_mag" in data.columns:
        for g_limit in g_limit_values[1:]:
            if g_limit is None:
                continue
            subset = data[data["phot_g_mean_mag"] < g_limit]
            _add_row(subset, g_limit=g_limit)

    # --- sn_cut axis ---
    if "parallax_error" in data.columns:
        sn = data["parallax"].values.astype(float) / data["parallax_error"].values.astype(float)
        for sn_cut in sn_cut_values[1:]:
            if sn_cut is None:
                continue
            subset = data[sn > sn_cut]
            _add_row(subset, sn_cut=sn_cut)

    # --- distance-shell axis ---
    if shell_edges_kpc is not None and "parallax" in data.columns:
        d_kpc = _distance_kpc(data)
        edges = list(shell_edges_kpc)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (d_kpc >= lo) & (d_kpc < hi)
            subset = data[mask]
            _add_row(subset, shell=f"{lo:.2f}-{hi:.2f}")

    # --- hemisphere axis ---
    if "b" in data.columns:
        for hemi in hemispheres[1:]:
            if hemi == "north":
                subset = data[data["b"] >= 0]
            elif hemi == "south":
                subset = data[data["b"] < 0]
            else:
                continue
            _add_row(subset, hemisphere=hemi)

    # --- optional quadrant axis ---
    if quadrants and "l" in data.columns:
        edges = (0.0, 90.0, 180.0, 270.0, 360.0)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            subset = data[(data["l"] >= lo) & (data["l"] < hi)]
            _add_row(subset, hemisphere=f"quad{i}")

    table = pd.DataFrame(rows, columns=BUDGET_COLUMNS)

    if write_output and output_csv:
        parent = os.path.dirname(output_csv)
        if parent:
            os.makedirs(parent, exist_ok=True)
        table.to_csv(output_csv, index=False)
    if write_output and output_json:
        parent = os.path.dirname(output_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(table.to_dict(orient="records"), f, indent=2, default=lambda x: None if pd.isna(x) else x)

    return table


# ---------------------------------------------------------------------------
# B5.6 — sky_region_jackknife
# ---------------------------------------------------------------------------

def sky_region_jackknife(
    data: pd.DataFrame,
    *,
    estimator: Callable[[pd.DataFrame], dict] = great_circle_pole,
    n_regions: int = 12,
    min_stars_per_region: int = 5,
) -> dict:
    """
    Leave-one-sky-region-out jackknife of `estimator`'s pole.

    The sky is partitioned into `n_regions` equal-width right-ascension
    bins (0-360 deg). For each region i, the estimator is refit on ALL
    stars EXCEPT those in region i ("delete-one-group" jackknife), giving
    n_regions leave-one-out pole estimates. The classic delete-one-group
    jackknife variance formula is applied to the ANGULAR SEPARATION of
    each leave-one-out pole from the overall (all-region) pole, giving a
    single rotation-invariant sigma_jackknife (degrees) -- an independent,
    non-parametric cross-check of the bootstrap sigma_stat.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra', 'dec' (required by `estimator`).
    estimator : callable
        Pole estimator, default `ngp_3d.great_circle_pole`.
    n_regions : int
        Number of equal-width RA bins. Default 12 (30 deg each).
    min_stars_per_region : int
        Regions with fewer stars than this are skipped (their
        leave-one-out estimate would barely differ from the full-sample
        fit and contributes no information).

    Returns
    -------
    dict with keys:
        overall_alpha_NGP, overall_delta_NGP : float — fit on ALL data
        region_estimates : list[dict] — one {region, n_excluded, alpha_NGP,
            delta_NGP, offset_deg} per included region
        sigma_jackknife_deg : float
        n_regions_used : int

    Raises
    ------
    ValueError : if data is empty, missing 'ra', or fewer than 2 regions
        have enough stars to jackknife.
    """
    if data is None or len(data) == 0:
        raise ValueError("Input DataFrame is empty.")
    if "ra" not in data.columns:
        raise ValueError("DataFrame is missing required column: 'ra'")

    overall = estimator(data)
    overall_alpha, overall_delta = overall["alpha_NGP"], overall["delta_NGP"]

    ra = data["ra"].values.astype(float) % 360.0
    bin_width = 360.0 / n_regions
    region_idx = np.floor(ra / bin_width).astype(int)
    region_idx = np.clip(region_idx, 0, n_regions - 1)

    region_estimates = []
    offsets = []
    for region in range(n_regions):
        in_region = region_idx == region
        n_excluded = int(in_region.sum())
        if n_excluded < min_stars_per_region:
            continue
        remainder = data[~in_region]
        if len(remainder) < 3:
            continue
        loo = estimator(remainder)
        offset_deg = _angular_separation_deg(
            loo["alpha_NGP"], loo["delta_NGP"], overall_alpha, overall_delta
        )
        region_estimates.append({
            "region": region,
            "n_excluded": n_excluded,
            "alpha_NGP": loo["alpha_NGP"],
            "delta_NGP": loo["delta_NGP"],
            "offset_deg": offset_deg,
        })
        offsets.append(offset_deg)

    m = len(offsets)
    if m < 2:
        raise ValueError(
            f"Only {m} region(s) had >= {min_stars_per_region} stars; "
            "need at least 2 for a jackknife variance estimate."
        )

    offsets_arr = np.asarray(offsets, dtype=float)
    mean_offset = float(np.mean(offsets_arr))
    # Classic delete-one-group jackknife variance: (m-1)/m * sum((theta_i - theta_bar)^2)
    jackknife_var = (m - 1) / m * np.sum((offsets_arr - mean_offset) ** 2)
    sigma_jackknife_deg = float(np.sqrt(jackknife_var))

    return {
        "overall_alpha_NGP": overall_alpha,
        "overall_delta_NGP": overall_delta,
        "region_estimates": region_estimates,
        "sigma_jackknife_deg": sigma_jackknife_deg,
        "n_regions_used": m,
    }


# ---------------------------------------------------------------------------
# B5.6 — combine_error_budget
# ---------------------------------------------------------------------------

def combine_error_budget(
    point_estimate: dict,
    bootstrap_result: dict,
    systematics_table: pd.DataFrame,
) -> dict:
    """
    Combine a point estimate's bootstrap statistical uncertainty with the
    spread of the systematics grid into a single honest error budget.

    Parameters
    ----------
    point_estimate : dict
        A pole-estimator result dict (e.g. from `ngp_3d.great_circle_pole`),
        must contain 'alpha_NGP' (hours) and 'delta_NGP' (degrees).
    bootstrap_result : dict
        Output of `bootstrap.bootstrap_great_circle_pole` (or
        `bootstrap.bootstrap_ngp`) -- consumed AS-IS, its own signature is
        NOT modified by this module. Must contain 'alpha_ci95', 'delta_ci95'.
    systematics_table : pd.DataFrame
        Output of `systematics_grid` (or any table sharing its
        'alpha'/'delta' columns). sigma_syst is the standard deviation of
        each variant's angular separation from `point_estimate`'s pole --
        i.e. how much the pole moves as analysis choices change, which is
        exactly what defines a systematic (as opposed to statistical)
        uncertainty.

    Returns
    -------
    dict with keys:
        alpha_NGP, delta_NGP : float  (from point_estimate, passed through)
        sigma_stat_deg, sigma_syst_deg, sigma_total_deg : float
        n_variants : int  — number of systematics_table rows used

    Raises
    ------
    ValueError : if systematics_table is empty.
    """
    if systematics_table is None or len(systematics_table) == 0:
        raise ValueError("systematics_table is empty; cannot derive sigma_syst.")

    alpha_ngp = point_estimate["alpha_NGP"]
    delta_ngp = point_estimate["delta_NGP"]

    sigma_stat_deg = _sigma_stat_deg_from_bootstrap(bootstrap_result, delta_ngp)

    offsets = np.array([
        _angular_separation_deg(row.alpha, row.delta, alpha_ngp, delta_ngp)
        for row in systematics_table.itertuples()
    ])
    sigma_syst_deg = float(np.std(offsets)) if len(offsets) > 1 else 0.0

    sigma_total_deg = float(np.hypot(sigma_stat_deg, sigma_syst_deg))

    return {
        "alpha_NGP": alpha_ngp,
        "delta_NGP": delta_ngp,
        "sigma_stat_deg": sigma_stat_deg,
        "sigma_syst_deg": sigma_syst_deg,
        "sigma_total_deg": sigma_total_deg,
        "n_variants": int(len(systematics_table)),
    }


# ---------------------------------------------------------------------------
# B5.2 — extinction_asymmetry_check
# ---------------------------------------------------------------------------

def extinction_asymmetry_check(
    data: pd.DataFrame,
    *,
    estimator: Callable[[pd.DataFrame], dict] = great_circle_pole,
    split: str = "b_sign",
    sigma_stat_deg: Optional[float] = None,
    n_bootstrap: int = 200,
    seed: int = 42,
    min_stars: int = 3,
) -> dict:
    """
    Split the sample by galactic-hemisphere sign and check whether the
    resulting pole difference is larger than expected from statistical
    noise alone -- the key scientific validation that this method can
    distinguish a REAL systematic (e.g. dust extinction asymmetrically
    masking one hemisphere) from ordinary sampling noise.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra', 'dec', 'b' columns.
    estimator : callable
        Pole estimator, default `ngp_3d.great_circle_pole`.
    split : str
        Only "b_sign" (b >= 0 -> north, b < 0 -> south) is currently
        implemented.
    sigma_stat_deg : float, optional
        If given, used directly instead of bootstrapping. Otherwise
        `bootstrap.bootstrap_great_circle_pole` is run (n_bootstrap
        resamples) on the SMALLER of the two hemisphere subsets (the more
        conservative -- i.e. larger -- statistical uncertainty), and
        converted to a scalar angular sigma via
        `_sigma_stat_deg_from_bootstrap`.
    n_bootstrap : int
        Bootstrap resamples used to estimate sigma_stat_deg when not
        supplied directly. Default 200 (kept small so this stays fast
        enough for the default test suite).
    seed : int
        Bootstrap RNG seed (reproducibility).
    min_stars : int
        Minimum stars required in EACH hemisphere.

    Returns
    -------
    dict with keys:
        pole_north, pole_south : dict {alpha_NGP, delta_NGP, n_stars}
        delta_pole_deg : float — angular separation between the two poles
        sigma_stat_deg : float
        threshold_deg  : float — 3 * sigma_stat_deg
        flagged        : bool  — True if delta_pole_deg > threshold_deg

    Raises
    ------
    ValueError : if 'b' is missing, or either hemisphere has < min_stars.
    """
    if data is None or len(data) == 0:
        raise ValueError("Input DataFrame is empty.")
    if "b" not in data.columns:
        raise ValueError("DataFrame is missing required column: 'b'")
    if split != "b_sign":
        raise ValueError(f"Unsupported split={split!r}; only 'b_sign' is implemented.")

    north = data[data["b"] >= 0]
    south = data[data["b"] < 0]
    if len(north) < min_stars or len(south) < min_stars:
        raise ValueError(
            f"Insufficient stars per hemisphere (north={len(north)}, south={len(south)}, "
            f"min_stars={min_stars})."
        )

    pole_n = estimator(north)
    pole_s = estimator(south)

    delta_pole_deg = _angular_separation_deg(
        pole_n["alpha_NGP"], pole_n["delta_NGP"], pole_s["alpha_NGP"], pole_s["delta_NGP"]
    )

    if sigma_stat_deg is None:
        smaller = north if len(north) <= len(south) else south
        bs = bootstrap_great_circle_pole(smaller, n_samples=n_bootstrap, seed=seed)
        reference_delta = pole_n["delta_NGP"] if smaller is north else pole_s["delta_NGP"]
        sigma_stat_deg = _sigma_stat_deg_from_bootstrap(bs, reference_delta)

    threshold_deg = 3.0 * sigma_stat_deg
    flagged = bool(delta_pole_deg > threshold_deg)

    return {
        "pole_north": {
            "alpha_NGP": pole_n["alpha_NGP"], "delta_NGP": pole_n["delta_NGP"], "n_stars": len(north),
        },
        "pole_south": {
            "alpha_NGP": pole_s["alpha_NGP"], "delta_NGP": pole_s["delta_NGP"], "n_stars": len(south),
        },
        "delta_pole_deg": delta_pole_deg,
        "sigma_stat_deg": sigma_stat_deg,
        "threshold_deg": threshold_deg,
        "flagged": flagged,
    }
