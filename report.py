"""
report.py — Build a summary table comparing NGP estimation methods vs the
IAU reference position, and export it to Markdown / LaTeX.

Public API:
    build_summary_table(results_dict) -> pd.DataFrame
    to_markdown(table)                -> str
    to_latex(table)                   -> str
    save_report(table, md_path, tex_path) -> None
"""

import os

import numpy as np
import pandas as pd

# IAU reference NGP position (J2000): alpha = 12h51m26s = 12.85h,
# delta = +27deg07m42s = 27.128 deg (rounded to 27.13 elsewhere in the project).
# Kept consistent with the notebook and generate_artifacts.py so the published
# summary table's error column matches the notebook prose.
IAU_ALPHA_H = 12.85
IAU_DELTA_DEG = 27.13


def _angular_separation_deg(
    alpha_h: float,
    delta_deg: float,
    ref_alpha_h: float = IAU_ALPHA_H,
    ref_delta_deg: float = IAU_DELTA_DEG,
) -> float:
    """
    Great-circle angular separation (degrees) between (alpha_h, delta_deg)
    and a reference NGP position, defaulting to the IAU reference position
    (IAU_ALPHA_H, IAU_DELTA_DEG) so every pre-existing call site (which
    calls this with only the first two arguments) is completely unaffected
    by this additive signature extension (ngp-precision batch B7,
    `build_master_table`, needs a second reference — Karim & Mamajek 2017 —
    without duplicating this helper).

    Uses the standard spherical law of cosines:
        cos(sep) = sin(d1)*sin(d2) + cos(d1)*cos(d2)*cos(a1 - a2)
    with RA converted from hours to degrees before comparison.
    """
    a1 = np.radians(alpha_h * 15.0)
    a2 = np.radians(ref_alpha_h * 15.0)
    d1 = np.radians(delta_deg)
    d2 = np.radians(ref_delta_deg)

    cos_sep = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a1 - a2)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sep)))


def build_summary_table(results_dict: dict) -> pd.DataFrame:
    """
    Build a comparison table of NGP estimation methods vs the IAU reference.

    Parameters
    ----------
    results_dict : dict
        Maps method name (str) -> {alpha_NGP: float (hours), delta_NGP: float
        (degrees), std_alpha: float (optional), std_delta: float (optional)}.

    Returns
    -------
    pd.DataFrame with columns:
        method          : str  — method name, plus a synthetic "IAU reference" row
        alpha_NGP       : float (hours)
        delta_NGP       : float (degrees)
        std_alpha       : float or NaN
        std_delta       : float or NaN
        error_vs_iau_deg: float — great-circle angular separation (degrees)
                          from the IAU reference position. Chosen over
                          separate per-coordinate deltas because RA error is
                          not directly comparable to Dec error in degrees
                          without accounting for declination-dependent RA
                          scale; angular separation gives one physically
                          meaningful number. The IAU reference row itself has
                          error_vs_iau_deg == 0.
    """
    rows = []
    for method, vals in results_dict.items():
        alpha = float(vals["alpha_NGP"])
        delta = float(vals["delta_NGP"])
        rows.append({
            "method": method,
            "alpha_NGP": alpha,
            "delta_NGP": delta,
            "std_alpha": vals.get("std_alpha", np.nan),
            "std_delta": vals.get("std_delta", np.nan),
            "error_vs_iau_deg": _angular_separation_deg(alpha, delta),
        })

    rows.append({
        "method": "IAU reference",
        "alpha_NGP": IAU_ALPHA_H,
        "delta_NGP": IAU_DELTA_DEG,
        "std_alpha": np.nan,
        "std_delta": np.nan,
        "error_vs_iau_deg": 0.0,
    })

    return pd.DataFrame(rows, columns=[
        "method", "alpha_NGP", "delta_NGP", "std_alpha", "std_delta", "error_vs_iau_deg",
    ])


# ---------------------------------------------------------------------------
# F7 (ngp-precision batch B7) — master comparison table (method x tracer)
# ---------------------------------------------------------------------------

# Karim & Mamajek (2017, MNRAS 465, 472) independent modern NGP measurement,
# expressed in hours (192.729 deg / 15 = 12.8486 h) to match this module's
# alpha_NGP-in-hours convention — same constant as `iau_forensics.KM2017_POLE`
# (192.729, 27.084) deg, kept as a separate module-level default here (rather
# than importing iau_forensics) so `report.py` has no new hard dependency.
KM2017_ALPHA_H = 12.8486
KM2017_DELTA_DEG = 27.084


def build_master_table(
    method_tracer_results: dict,
    *,
    iau_pole: tuple = (IAU_ALPHA_H, IAU_DELTA_DEG),
    km2017_pole: tuple = (KM2017_ALPHA_H, KM2017_DELTA_DEG),
) -> pd.DataFrame:
    """
    Build the master publication comparison table: one row per
    (method, tracer) combination, benchmarked against BOTH the IAU
    reference position and the Karim & Mamajek (2017) independent modern
    measurement (design `sdd/ngp-precision/design` §4 F7; spec F7-R1-S1).

    This is ADDITIVE alongside `build_summary_table` (kept byte-for-byte
    unchanged, 11-test regression) — it does not replace it; the two
    tables serve different purposes (`build_summary_table`: single-method
    list vs IAU only; `build_master_table`: full method x tracer grid with
    the honest statistical/systematic error-budget columns from
    `systematics.combine_error_budget`).

    Parameters
    ----------
    method_tracer_results : dict
        Maps ``(method: str, tracer: str)`` -> a result dict. Each result
        dict must contain 'alpha_NGP' (hours) and 'delta_NGP' (degrees) --
        i.e. any raw estimator result dict from `ngp_3d.great_circle_pole`,
        `ngp_offset_plane.offset_plane_pole`, `ngp_weighted_3d.
        weighted_tls_plane`, `ngp_kinematic.kinematic_pole`, or the output
        of `systematics.combine_error_budget` all work unmodified as
        values here. Optional keys, defaulting to NaN (sigma_*) or False
        (zero_point_corrected) when absent from a given result dict:
            'sigma_stat_deg', 'sigma_syst_deg', 'sigma_total_deg'
            (present only for results that went through
            `combine_error_budget`) and 'zero_point_corrected' (present
            only for the two 3D methods that thread this provenance flag,
            design ADR5).
    iau_pole : tuple(alpha_h, delta_deg)
        Reference IAU NGP position. Defaults to the same
        (IAU_ALPHA_H, IAU_DELTA_DEG) constants used by
        `build_summary_table`, so `error_vs_iau_deg` is directly
        comparable between the two tables.
    km2017_pole : tuple(alpha_h, delta_deg)
        Karim & Mamajek (2017) reference NGP position. Defaults to
        (KM2017_ALPHA_H, KM2017_DELTA_DEG) == (192.729, 27.084) deg, the
        same value as `iau_forensics.KM2017_POLE`.

    Returns
    -------
    pd.DataFrame with columns:
        method, tracer                                  : str
        alpha_NGP (hours), delta_NGP (degrees)           : float
        sigma_stat_deg, sigma_syst_deg, sigma_total_deg  : float or NaN
        zero_point_corrected                             : bool
        error_vs_iau_deg, error_vs_km2017_deg             : float

    An empty `method_tracer_results` returns an empty DataFrame with the
    same columns (no rows) rather than raising, so callers can safely
    concatenate partial per-tracer results incrementally.
    """
    rows = []
    for (method, tracer), vals in method_tracer_results.items():
        alpha = float(vals["alpha_NGP"])
        delta = float(vals["delta_NGP"])
        rows.append({
            "method": method,
            "tracer": tracer,
            "alpha_NGP": alpha,
            "delta_NGP": delta,
            "sigma_stat_deg": vals.get("sigma_stat_deg", np.nan),
            "sigma_syst_deg": vals.get("sigma_syst_deg", np.nan),
            "sigma_total_deg": vals.get("sigma_total_deg", np.nan),
            "zero_point_corrected": bool(vals.get("zero_point_corrected", False)),
            "error_vs_iau_deg": _angular_separation_deg(alpha, delta, iau_pole[0], iau_pole[1]),
            "error_vs_km2017_deg": _angular_separation_deg(alpha, delta, km2017_pole[0], km2017_pole[1]),
        })

    return pd.DataFrame(rows, columns=[
        "method", "tracer", "alpha_NGP", "delta_NGP",
        "sigma_stat_deg", "sigma_syst_deg", "sigma_total_deg",
        "zero_point_corrected", "error_vs_iau_deg", "error_vs_km2017_deg",
    ])


def to_markdown(table: pd.DataFrame) -> str:
    """Render the summary table as a Markdown string."""
    return table.to_markdown(index=False)


def to_latex(table: pd.DataFrame) -> str:
    """Render the summary table as a publication-ready LaTeX ``table`` float."""
    tabular = table.to_latex(index=False, float_format="%.4f")
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Aproximaciones del Polo Norte Galactico (NGP) vs. valor de referencia IAU.}\n"
        "\\label{tab:ngp_comparison}\n"
        f"{tabular}"
        "\\end{table}\n"
    )


def save_report(
    table: pd.DataFrame,
    md_path: str = "results/summary_table.md",
    tex_path: str = "results/summary_table.tex",
) -> None:
    """
    Write the summary table to both Markdown and LaTeX files, creating
    parent directories automatically.
    """
    for path in (md_path, tex_path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    with open(md_path, "w") as f:
        f.write(to_markdown(table))

    with open(tex_path, "w") as f:
        f.write(to_latex(table))
