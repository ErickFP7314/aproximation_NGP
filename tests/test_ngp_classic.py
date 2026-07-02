"""
Tests for ngp_classic.py — three 2D NGP estimation methods.

aprox_ar_svd : SVD great-circle normal (clean-room reimplementation of pair-symmetry spirit)
aprox_ar     : paper pair-symmetry method (faithfully reimplemented)
aprox_dec1   : top-n declination method
aprox_dec2   : RA-window method
"""

import pytest
import pandas as pd
import numpy as np

from ngp_classic import aprox_ar, aprox_ar_svd, aprox_dec1, aprox_dec2

# IAU reference values
_NGP_RA_DEG = 192.75
_NGP_RA_H   = _NGP_RA_DEG / 15.0   # 12.85 hours
_NGP_DEC_DEG = 27.11
_TOLERANCE_DEG = 2.0

_AR_REF_H = 12.816  # hours — theoretical prior used by pair-symmetry

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"alpha_NGP", "delta_NGP", "std_alpha", "std_delta"}
_PAIR_SYM_KEYS = {"alpha_NGP", "delta_NGP", "std_alpha", "method"}


def _has_required_keys(result: dict) -> bool:
    return set(result.keys()) == _REQUIRED_KEYS


def _has_pair_sym_keys(result: dict) -> bool:
    return _PAIR_SYM_KEYS.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# Helpers to build pair-symmetry test DataFrames
# ---------------------------------------------------------------------------

def _symmetric_pairs(ar_ref_h: float, n_groups: int = 5) -> pd.DataFrame:
    """
    Build a DataFrame with n_groups dec values, each with 2 stars symmetric
    around ar_ref (one below, one above). Midpoints are exactly ar_ref.
    """
    ar_ref_deg = ar_ref_h * 15.0
    rows = []
    for i in range(n_groups):
        dec_val = float(10 + i * 10)      # 10, 20, 30, 40, 50
        offset_deg = (i + 1) * 5.0         # 5, 10, 15, 20, 25
        rows.append({"ra": ar_ref_deg - offset_deg, "dec": dec_val})
        rows.append({"ra": ar_ref_deg + offset_deg, "dec": dec_val})
    return pd.DataFrame(rows)


def _asymmetric_pairs(ar_ref_h: float) -> pd.DataFrame:
    """
    Build a DataFrame where midpoints are NOT all equal → non-zero std_alpha.
    """
    ar_ref_deg = ar_ref_h * 15.0
    rows = [
        # group dec=10: midpoint at ar_ref + 5°/15 h
        {"ra": ar_ref_deg - 10.0, "dec": 10.0},
        {"ra": ar_ref_deg + 20.0, "dec": 10.0},
        # group dec=20: midpoint at ar_ref + 12.5°/15 h
        {"ra": ar_ref_deg - 5.0,  "dec": 20.0},
        {"ra": ar_ref_deg + 30.0, "dec": 20.0},
        # group dec=30: midpoint at ar_ref (symmetric)
        {"ra": ar_ref_deg - 15.0, "dec": 30.0},
        {"ra": ar_ref_deg + 15.0, "dec": 30.0},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# aprox_ar_svd tests  (the renamed SVD great-circle normal estimator)
# ---------------------------------------------------------------------------

def test_aprox_ar_svd_recovers_ngp_within_tolerance(synthetic_disk_stars):
    """aprox_ar_svd on synthetic 500-star data should recover NGP within 2°."""
    result = aprox_ar_svd(synthetic_disk_stars)
    assert abs(result["alpha_NGP"] - _NGP_RA_DEG) < _TOLERANCE_DEG, (
        f"alpha_NGP={result['alpha_NGP']:.2f}° not within {_TOLERANCE_DEG}° of {_NGP_RA_DEG}"
    )
    assert abs(result["delta_NGP"] - _NGP_DEC_DEG) < _TOLERANCE_DEG, (
        f"delta_NGP={result['delta_NGP']:.2f}° not within {_TOLERANCE_DEG}° of {_NGP_DEC_DEG}"
    )


def test_aprox_ar_svd_empty_dataframe_raises_value_error():
    """aprox_ar_svd with empty DataFrame raises ValueError."""
    with pytest.raises(ValueError):
        aprox_ar_svd(pd.DataFrame(columns=["ra", "dec"]))


def test_aprox_ar_svd_returns_required_keys(synthetic_disk_stars):
    """aprox_ar_svd must return exactly the four standard keys."""
    result = aprox_ar_svd(synthetic_disk_stars)
    assert _has_required_keys(result), f"Unexpected keys: {set(result.keys())}"


# ---------------------------------------------------------------------------
# aprox_ar (pair-symmetry) tests
# ---------------------------------------------------------------------------

def test_aprox_ar_pair_symmetry_exact_midpoints():
    """
    With stars placed symmetrically around ar_ref, each group midpoint == ar_ref.
    Mean of midpoints must equal ar_ref within floating-point tolerance.
    """
    df = _symmetric_pairs(_AR_REF_H, n_groups=5)
    result = aprox_ar(df, ar_ref=_AR_REF_H)
    assert abs(result["alpha_NGP"] - _AR_REF_H) < 1e-9, (
        f"alpha_NGP={result['alpha_NGP']:.6f}h expected {_AR_REF_H}h"
    )


def test_aprox_ar_pair_symmetry_returns_expected_keys():
    """aprox_ar (pair-symmetry) must include alpha_NGP, delta_NGP, std_alpha, method."""
    df = _symmetric_pairs(_AR_REF_H)
    result = aprox_ar(df, ar_ref=_AR_REF_H)
    assert _has_pair_sym_keys(result), f"Missing keys in {set(result.keys())}"


def test_aprox_ar_pair_symmetry_delta_ngp_is_none():
    """delta_NGP is None because pair-symmetry does not estimate declination."""
    df = _symmetric_pairs(_AR_REF_H)
    result = aprox_ar(df, ar_ref=_AR_REF_H)
    assert result["delta_NGP"] is None


def test_aprox_ar_pair_symmetry_method_label():
    """method key must equal 'pair_symmetry'."""
    df = _symmetric_pairs(_AR_REF_H)
    result = aprox_ar(df, ar_ref=_AR_REF_H)
    assert result["method"] == "pair_symmetry"


def test_aprox_ar_pair_symmetry_std_alpha_nonzero_when_asymmetric():
    """Asymmetric pairs produce varying midpoints → std_alpha > 0."""
    df = _asymmetric_pairs(_AR_REF_H)
    result = aprox_ar(df, ar_ref=_AR_REF_H)
    assert result["std_alpha"] > 0.0, "std_alpha should be > 0 for asymmetric pairs"


def test_aprox_ar_pair_symmetry_excludes_one_sided_groups():
    """
    A group where all stars are on the same side of ar_ref must be excluded.
    Only the valid two-sided group contributes to the result.
    """
    ar_ref = _AR_REF_H
    ar_ref_deg = ar_ref * 15.0

    rows = [
        # dec=10 group: both stars below ar_ref → excluded
        {"ra": ar_ref_deg - 20.0, "dec": 10.0},
        {"ra": ar_ref_deg - 10.0, "dec": 10.0},
        # dec=20 group: one each side → included; midpoint == ar_ref
        {"ra": ar_ref_deg - 15.0, "dec": 20.0},
        {"ra": ar_ref_deg + 15.0, "dec": 20.0},
    ]
    df = pd.DataFrame(rows)
    result = aprox_ar(df, ar_ref=ar_ref)

    # Only 1 valid group → result is exactly that midpoint
    assert abs(result["alpha_NGP"] - ar_ref) < 1e-9
    # std of a single-element list is 0
    assert result["std_alpha"] == pytest.approx(0.0, abs=1e-12)


def test_aprox_ar_pair_symmetry_no_valid_groups_raises_value_error():
    """If every group fails the two-sided check, ValueError must be raised."""
    ar_ref = _AR_REF_H
    ar_ref_deg = ar_ref * 15.0

    # 3 groups each with 2 stars all below ar_ref
    rows = []
    for i in range(3):
        rows.append({"ra": ar_ref_deg - 20.0 - i * 5, "dec": float(10 + i * 10)})
        rows.append({"ra": ar_ref_deg - 10.0 - i * 5, "dec": float(10 + i * 10)})
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        aprox_ar(df, ar_ref=ar_ref)


def test_aprox_ar_pair_symmetry_empty_raises_value_error():
    """Empty DataFrame raises ValueError."""
    with pytest.raises(ValueError):
        aprox_ar(pd.DataFrame(columns=["ra", "dec"]), ar_ref=_AR_REF_H)


# ---------------------------------------------------------------------------
# aprox_dec1 tests (unchanged)
# ---------------------------------------------------------------------------

def test_aprox_dec1_n_exceeds_sample_raises_value_error():
    """S3.3 — n > len(data) raises ValueError."""
    small_df = pd.DataFrame({
        "ra": np.linspace(180, 200, 10),
        "dec": np.linspace(20, 35, 10),
    })
    with pytest.raises(ValueError):
        aprox_dec1(small_df, n=100)


def test_aprox_dec1_valid_n_returns_required_keys():
    """S3.4 — valid n returns dict with all four keys."""
    df = pd.DataFrame({
        "ra": np.linspace(180, 210, 50),
        "dec": np.linspace(15, 40, 50),
    })
    result = aprox_dec1(df, n=20)
    assert _has_required_keys(result), f"Unexpected keys: {set(result.keys())}"


def test_aprox_dec1_returns_required_keys(synthetic_disk_stars):
    """Triangulate: aprox_dec1 with default n on 500-star data returns all keys."""
    result = aprox_dec1(synthetic_disk_stars)
    assert _has_required_keys(result)


# ---------------------------------------------------------------------------
# aprox_dec2 tests (unchanged)
# ---------------------------------------------------------------------------

def test_aprox_dec2_delta_zero_raises_value_error(synthetic_disk_stars):
    """S3.5 — delta=0 raises ValueError."""
    with pytest.raises(ValueError):
        aprox_dec2(synthetic_disk_stars, delta=0)


def test_aprox_dec2_valid_delta_returns_required_keys():
    """S3.6 — valid delta returns dict with all four keys."""
    df = pd.DataFrame({
        "ra": np.random.default_rng(0).uniform(180, 210, 30),
        "dec": np.random.default_rng(1).uniform(15, 40, 30),
    })
    result = aprox_dec2(df, delta=1.0)
    assert _has_required_keys(result), f"Unexpected keys: {set(result.keys())}"


def test_aprox_dec2_returns_required_keys(synthetic_disk_stars):
    """Triangulate: aprox_dec2 with delta=1.0 on 500-star data returns all keys."""
    result = aprox_dec2(synthetic_disk_stars, delta=1.0)
    assert _has_required_keys(result)
