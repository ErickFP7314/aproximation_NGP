"""
param_sweep.py — Sensitivity sweeps over b_max, n, and delta_ra parameters.

All functions are pure (DataFrame in, DataFrame/dict out, no hidden global I/O).
Calls to the estimators (aprox_ar, aprox_dec1, aprox_dec2) are wrapped in
try/except so individual parameter values that produce no valid groups return NaN
rather than crashing the sweep.

Public API:
    sweep_b_max(data, b_max_range, n_default, delta_ra_default) -> pd.DataFrame
    sweep_n(data, b_max, n_fracs)                               -> pd.DataFrame
    sweep_delta_ra(data, b_max, delta_range)                    -> pd.DataFrame
    run_all_sweeps(data, output_path)                           -> dict[str, pd.DataFrame]
"""

import os

import numpy as np
import pandas as pd

from ngp_classic import aprox_ar, aprox_dec1, aprox_dec2

# Default delta_range: 0.05, 0.10, ..., 1.50 (30 steps)
_DEFAULT_DELTA_RANGE = [round(0.05 * i, 10) for i in range(1, 31)]

# Default n_fracs: 0.05, 0.10, ..., 0.50 (10 fractions)
_DEFAULT_N_FRACS = [round(0.05 * i, 10) for i in range(1, 11)]


def _filter_by_b(data: pd.DataFrame, b_max: float) -> pd.DataFrame:
    """Return rows where |b| < b_max; requires column 'b' (galactic latitude)."""
    return data[np.abs(data["b"]) < b_max].copy()


# ---------------------------------------------------------------------------
# sweep_b_max
# ---------------------------------------------------------------------------

def sweep_b_max(
    data: pd.DataFrame,
    b_max_range=range(2, 21),
    n_default: int = None,
    delta_ra_default: float = 0.5,
) -> pd.DataFrame:
    """
    Sweep the b_max filter threshold and record NGP estimates from three methods.

    For each b_max value in b_max_range, stars with |b| < b_max are kept and
    passed to aprox_ar (pair-symmetry), aprox_dec1, and aprox_dec2.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra', 'dec', 'b' columns.
    b_max_range : iterable of int/float
        Galactic latitude cutoff values in degrees. Default range(2, 21).
    n_default : int, optional
        Fixed n for aprox_dec1. If None, each call uses the function's own default
        (max(1, len(filtered)//10)).
    delta_ra_default : float
        Half-width in hours for aprox_dec2. Default 0.5h.

    Returns
    -------
    pd.DataFrame with columns: b_max, alpha_ar, delta_dec1, delta_dec2, n_used.
        alpha_ar  : alpha_NGP in hours from aprox_ar (pair-symmetry)
        delta_dec1: delta_NGP in degrees from aprox_dec1
        delta_dec2: delta_NGP in degrees from aprox_dec2
        n_used    : number of stars passing the |b| < b_max filter
    """
    rows = []
    for b_max in b_max_range:
        filtered = _filter_by_b(data, b_max)
        n_used = len(filtered)

        # aprox_ar (pair-symmetry) → alpha_NGP in hours
        try:
            alpha_ar = aprox_ar(filtered)["alpha_NGP"]
        except (ValueError, Exception):
            alpha_ar = np.nan

        # aprox_dec1 → delta_NGP in degrees
        try:
            n = n_default if n_default is not None else max(1, n_used // 10)
            delta_dec1 = aprox_dec1(filtered, n=n)["delta_NGP"]
        except (ValueError, Exception):
            delta_dec1 = np.nan

        # aprox_dec2 → delta_NGP in degrees
        try:
            delta_dec2 = aprox_dec2(filtered, delta=delta_ra_default)["delta_NGP"]
        except (ValueError, Exception):
            delta_dec2 = np.nan

        rows.append({
            "b_max": float(b_max),
            "alpha_ar": alpha_ar,
            "delta_dec1": delta_dec1,
            "delta_dec2": delta_dec2,
            "n_used": n_used,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sweep_n
# ---------------------------------------------------------------------------

def sweep_n(
    data: pd.DataFrame,
    b_max: float = 15,
    n_fracs=None,
) -> pd.DataFrame:
    """
    Sweep the top-n fraction parameter of aprox_dec1.

    For each fraction in n_fracs, the effective n is max(1, int(N * frac)) where
    N = number of stars passing the |b| < b_max filter.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra', 'dec', 'b' columns.
    b_max : float
        Galactic latitude cut applied before sweeping. Default 15°.
    n_fracs : list of float, optional
        Fractions of the sample to use as n. Default [0.05, 0.10, ..., 0.50].

    Returns
    -------
    pd.DataFrame with columns: n_frac, n_value, delta_dec1.
    """
    if n_fracs is None:
        n_fracs = _DEFAULT_N_FRACS

    filtered = _filter_by_b(data, b_max)
    N = len(filtered)

    rows = []
    for frac in n_fracs:
        n_val = max(1, int(N * frac))
        try:
            delta_dec1 = aprox_dec1(filtered, n=n_val)["delta_NGP"]
        except (ValueError, Exception):
            delta_dec1 = np.nan
        rows.append({
            "n_frac": float(frac),
            "n_value": n_val,
            "delta_dec1": delta_dec1,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sweep_delta_ra
# ---------------------------------------------------------------------------

def sweep_delta_ra(
    data: pd.DataFrame,
    b_max: float = 15,
    delta_range=None,
) -> pd.DataFrame:
    """
    Sweep the RA-window half-width parameter of aprox_dec2.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra', 'dec', 'b' columns.
    b_max : float
        Galactic latitude cut applied before sweeping. Default 15°.
    delta_range : list of float, optional
        Half-widths in hours. Default [0.05, 0.10, ..., 1.50] (30 values, step 0.05).

    Returns
    -------
    pd.DataFrame with columns: delta_ra, delta_dec2.
    """
    if delta_range is None:
        delta_range = _DEFAULT_DELTA_RANGE

    filtered = _filter_by_b(data, b_max)

    rows = []
    for delta in delta_range:
        try:
            delta_dec2 = aprox_dec2(filtered, delta=delta)["delta_NGP"]
        except (ValueError, Exception):
            delta_dec2 = np.nan
        rows.append({
            "delta_ra": float(delta),
            "delta_dec2": delta_dec2,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# run_all_sweeps
# ---------------------------------------------------------------------------

def run_all_sweeps(
    data: pd.DataFrame,
    output_path: str = "results/param_sweep_results.csv",
) -> dict:
    """
    Run all three sweeps and save a combined CSV.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with 'ra', 'dec', 'b' columns.
    output_path : str
        Path where the combined CSV is written. Parent directories are created
        automatically (os.makedirs, exist_ok=True).

    Returns
    -------
    dict with keys 'b_max', 'n', 'delta_ra', each mapping to the sweep DataFrame.
    """
    df_bmax = sweep_b_max(data)
    df_n = sweep_n(data)
    df_delta = sweep_delta_ra(data)

    result = {
        "b_max": df_bmax,
        "n": df_n,
        "delta_ra": df_delta,
    }

    # Build a combined CSV with a sweep_type column for easy loading
    combined = pd.concat(
        [
            df_bmax.assign(sweep_type="b_max"),
            df_n.assign(sweep_type="n"),
            df_delta.assign(sweep_type="delta_ra"),
        ],
        ignore_index=True,
    )

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    combined.to_csv(output_path, index=False)

    return result
