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


def _angular_separation_deg(alpha_h: float, delta_deg: float) -> float:
    """
    Great-circle angular separation (degrees) between (alpha_h, delta_deg)
    and the IAU reference NGP position (IAU_ALPHA_H, IAU_DELTA_DEG).

    Uses the standard spherical law of cosines:
        cos(sep) = sin(d1)*sin(d2) + cos(d1)*cos(d2)*cos(a1 - a2)
    with RA converted from hours to degrees before comparison.
    """
    a1 = np.radians(alpha_h * 15.0)
    a2 = np.radians(IAU_ALPHA_H * 15.0)
    d1 = np.radians(delta_deg)
    d2 = np.radians(IAU_DELTA_DEG)

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
