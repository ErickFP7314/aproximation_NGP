"""
bootstrap.py — Bootstrap resampling of the 3D RANSAC NGP estimator.

Repeatedly resamples the input catalog with replacement and refits the
galactic plane via `ngp_3d.ngp_3d_pipeline`, collecting the distribution of
alpha_NGP (hours) and delta_NGP (degrees) to report mean, median and a 95%
confidence interval.

Public API:
    bootstrap_ngp(data, n_samples, seed, ransac_max_trials) -> dict
    bootstrap_great_circle_pole(data, n_samples, seed)      -> dict
    save_bootstrap_results(results, path)                  -> None
"""

import json
import os

import numpy as np
import pandas as pd

from ngp_3d import ngp_3d_pipeline, great_circle_pole


def bootstrap_ngp(
    data: pd.DataFrame,
    n_samples: int = 10_000,
    seed: int = 42,
    ransac_max_trials: int = 20,
) -> dict:
    """
    Bootstrap the 3D RANSAC NGP estimate by resampling `data` with replacement.

    For each of `n_samples` resamples (same size as `data`, drawn with
    replacement using `numpy.random.default_rng(seed)`), `ngp_3d_pipeline`
    is run with a per-resample-reduced `max_trials=ransac_max_trials` to keep
    runtime tractable at large n_samples.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (deg), 'dec' (deg), 'parallax' (mas, positive).
    n_samples : int
        Number of bootstrap resamples. Default 10_000.
    seed : int
        Seed for `numpy.random.default_rng`. Same seed -> identical result.
    ransac_max_trials : int
        RANSAC max_trials forwarded to `ngp_3d_pipeline` for each resample.
        Kept low (default 20) since the inlier fraction is already high after
        the first fit; the cost driver is n_samples * RANSAC fits.

    Returns
    -------
    dict with keys:
        alpha_mean, alpha_median : float (hours)
        alpha_ci95                : tuple[float, float]  (2.5th, 97.5th pct)
        delta_mean, delta_median : float (degrees)
        delta_ci95                : tuple[float, float]  (2.5th, 97.5th pct)
        n_samples                 : int

    Raises
    ------
    ValueError : if n_samples < 1 or data is empty.
    """
    if data is None or len(data) == 0:
        raise ValueError("Input DataFrame is empty.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

    rng = np.random.default_rng(seed)
    n_rows = len(data)

    alphas = np.empty(n_samples, dtype=float)
    deltas = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        idx = rng.integers(0, n_rows, size=n_rows)
        resample = data.iloc[idx]
        result = ngp_3d_pipeline(resample, rng=rng, max_trials=ransac_max_trials)
        alphas[i] = result["alpha_NGP"]
        deltas[i] = result["delta_NGP"]

    alpha_ci95 = tuple(np.percentile(alphas, [2.5, 97.5]).tolist())
    delta_ci95 = tuple(np.percentile(deltas, [2.5, 97.5]).tolist())

    return {
        "alpha_mean": float(np.mean(alphas)),
        "alpha_median": float(np.median(alphas)),
        "alpha_ci95": alpha_ci95,
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_ci95": delta_ci95,
        "n_samples": int(n_samples),
    }


def bootstrap_great_circle_pole(
    data: pd.DataFrame,
    n_samples: int = 2_000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap the flagship great-circle SVD NGP estimate (`ngp_3d.great_circle_pole`)
    by resampling `data` with replacement.

    Unlike `bootstrap_ngp` (3D RANSAC), each resample here is a plain SVD on
    unit direction vectors — no RANSAC trials, no distance/parallax — so this
    runs much faster per resample and does not need a `max_trials` knob.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'ra' (deg) and 'dec' (deg).
    n_samples : int
        Number of bootstrap resamples. Default 2_000.
    seed : int
        Seed for `numpy.random.default_rng`. Same seed -> identical result.

    Returns
    -------
    dict with keys:
        alpha_mean, alpha_median : float (hours)
        alpha_ci95                : tuple[float, float]  (2.5th, 97.5th pct)
        delta_mean, delta_median : float (degrees)
        delta_ci95                : tuple[float, float]  (2.5th, 97.5th pct)
        n_samples                 : int

    Raises
    ------
    ValueError : if n_samples < 1 or data is empty.
    """
    if data is None or len(data) == 0:
        raise ValueError("Input DataFrame is empty.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

    rng = np.random.default_rng(seed)
    n_rows = len(data)

    alphas = np.empty(n_samples, dtype=float)
    deltas = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        idx = rng.integers(0, n_rows, size=n_rows)
        resample = data.iloc[idx]
        result = great_circle_pole(resample)
        alphas[i] = result["alpha_NGP"]
        deltas[i] = result["delta_NGP"]

    alpha_ci95 = tuple(np.percentile(alphas, [2.5, 97.5]).tolist())
    delta_ci95 = tuple(np.percentile(deltas, [2.5, 97.5]).tolist())

    return {
        "alpha_mean": float(np.mean(alphas)),
        "alpha_median": float(np.median(alphas)),
        "alpha_ci95": alpha_ci95,
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_ci95": delta_ci95,
        "n_samples": int(n_samples),
    }


def save_bootstrap_results(results: dict, path: str = "results/bootstrap_results.json") -> None:
    """
    Write bootstrap results to a JSON file.

    Tuples (alpha_ci95, delta_ci95) are serialized as 2-element JSON arrays;
    on reload they come back as lists (JSON has no tuple type) but are
    round-trippable via `tuple(loaded["alpha_ci95"])`.

    Parameters
    ----------
    results : dict
        Output of `bootstrap_ngp`.
    path : str
        Destination path. Parent directories are created automatically.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    serializable = dict(results)
    serializable["alpha_ci95"] = list(results["alpha_ci95"])
    serializable["delta_ci95"] = list(results["delta_ci95"])

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
