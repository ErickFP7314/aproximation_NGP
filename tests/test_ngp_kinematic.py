"""
Tests for ngp_kinematic.py — F4 kinematic (proper-motion-based) NGP
estimator: rotation-axis eigenvector fit, solar-motion reflex correction,
and the optional RVS 3D angular-momentum variant (ngp-precision batch B4).
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ngp_3d import normal_to_equatorial
from ngp_kinematic import kinematic_pole, kinematic_pole_rvs
from synthetic_catalog import synthetic_catalog


def _pole_angular_error_deg(alpha_h, delta_deg, true_ra_deg, true_dec_deg):
    ra1, dec1 = np.radians(alpha_h * 15.0), np.radians(delta_deg)
    ra2, dec2 = np.radians(true_ra_deg), np.radians(true_dec_deg)
    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    return float(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))))


def _true_ra_dec_deg(rotation_axis):
    alpha_h, delta_deg = normal_to_equatorial(*rotation_axis)
    return alpha_h * 15.0, delta_deg


# ---------------------------------------------------------------------------
# B4.1/B4.3 — recovers a known injected rotation axis within ±0.2°.
# Spec F4-R1-S1.
# ---------------------------------------------------------------------------

def test_kinematic_pole_recovers_injected_rotation_axis():
    df = synthetic_catalog(
        n=3000, seed=11, pole=(200.0, 15.0), outlier_fraction=0.0,
        dist_range_kpc=(0.2, 1.0), include_proper_motion=True,
        v_circ_kms=220.0,
    )
    true_axis = df.attrs["truth"]["rotation_axis"]
    true_ra_deg, true_dec_deg = _true_ra_dec_deg(true_axis)

    result = kinematic_pole(df)

    err = _pole_angular_error_deg(
        result["alpha_NGP"], result["delta_NGP"], true_ra_deg, true_dec_deg
    )
    assert err < 0.2, (
        f"kinematic pole error={err:.5f}deg vs injected axis "
        f"(ra={true_ra_deg:.4f}, dec={true_dec_deg:.4f})"
    )


def test_kinematic_pole_recovers_custom_rotation_axis_distinct_from_pole():
    """Rotation axis independently set apart from the positional `pole` --
    proves the estimator truly tracks the VELOCITY field, not the sky
    position distribution."""
    custom_axis = (0.1, 0.2, 0.95)  # need not be near `pole`
    df = synthetic_catalog(
        n=3000, seed=31, pole=(192.75, 27.11), outlier_fraction=0.0,
        dist_range_kpc=(0.2, 1.0), include_proper_motion=True,
        rotation_axis=custom_axis, v_circ_kms=220.0,
    )
    true_axis = df.attrs["truth"]["rotation_axis"]
    true_ra_deg, true_dec_deg = _true_ra_dec_deg(true_axis)

    result = kinematic_pole(df)

    err = _pole_angular_error_deg(
        result["alpha_NGP"], result["delta_NGP"], true_ra_deg, true_dec_deg
    )
    assert err < 0.2, f"kinematic pole error={err:.5f}deg"


def test_kinematic_pole_returns_required_keys():
    df = synthetic_catalog(
        n=300, seed=11, outlier_fraction=0.0, dist_range_kpc=(0.2, 1.0),
        include_proper_motion=True,
    )
    result = kinematic_pole(df)
    required = {
        "alpha_NGP", "delta_NGP", "normal", "covariance", "n_stars",
        "solar_motion", "method",
    }
    assert set(result.keys()) == required
    assert result["method"] == "kinematic_pm"
    assert result["normal"].shape == (3,)
    assert result["covariance"].shape == (3, 3)
    assert result["n_stars"] == len(df)
    assert result["solar_motion"] == (11.1, 12.24, 7.25)


# ---------------------------------------------------------------------------
# B4.2 — all-zero proper motion raises ValueError. Spec F4-R1-S2.
# ---------------------------------------------------------------------------

def test_kinematic_pole_zero_proper_motion_raises_valueerror():
    df = synthetic_catalog(
        n=200, seed=3, outlier_fraction=0.0, dist_range_kpc=(0.3, 1.0),
    )
    df["pmra"] = 0.0
    df["pmdec"] = 0.0
    with pytest.raises(ValueError):
        kinematic_pole(df)


def test_kinematic_pole_empty_raises_valueerror():
    empty = pd.DataFrame({
        "ra": [], "dec": [], "parallax": [], "pmra": [], "pmdec": [],
    })
    with pytest.raises(ValueError):
        kinematic_pole(empty)


def test_kinematic_pole_missing_column_raises_valueerror():
    df = pd.DataFrame({
        "ra": [1.0, 2.0], "dec": [1.0, 2.0], "parallax": [1.0, 2.0],
    })  # no pmra/pmdec
    with pytest.raises(ValueError):
        kinematic_pole(df)


def test_kinematic_pole_nonpositive_parallax_raises_valueerror():
    df = pd.DataFrame({
        "ra": [1.0, 2.0, 3.0], "dec": [1.0, 2.0, 3.0],
        "parallax": [1.0, 0.0, 2.0], "pmra": [1.0, 2.0, 3.0],
        "pmdec": [1.0, 2.0, 3.0],
    })
    with pytest.raises(ValueError):
        kinematic_pole(df)


# ---------------------------------------------------------------------------
# B4.4/B4.5 — RVS variant: empty/absent subsample -> None (+ logged reason,
# no exception). Spec F4-R2-S1.
# ---------------------------------------------------------------------------

def test_kinematic_pole_rvs_returns_none_when_no_rvs_column(caplog):
    df = synthetic_catalog(
        n=200, seed=3, outlier_fraction=0.0, include_proper_motion=True,
    )
    assert "radial_velocity" not in df.columns

    with caplog.at_level(logging.INFO, logger="ngp_kinematic"):
        result = kinematic_pole_rvs(df)

    assert result is None
    assert any("radial_velocity" in rec.message for rec in caplog.records)


def test_kinematic_pole_rvs_returns_none_when_rvs_fraction_zero(caplog):
    df = synthetic_catalog(
        n=200, seed=3, outlier_fraction=0.0, include_proper_motion=True,
        include_rvs=True, rvs_fraction=0.0,
    )

    with caplog.at_level(logging.INFO, logger="ngp_kinematic"):
        result = kinematic_pole_rvs(df)

    assert result is None


def test_kinematic_pole_rvs_returns_none_when_all_nan(caplog):
    df = synthetic_catalog(
        n=200, seed=3, outlier_fraction=0.0, include_proper_motion=True,
        include_rvs=True, rvs_fraction=0.5,
    )
    df["radial_velocity"] = np.nan

    with caplog.at_level(logging.INFO, logger="ngp_kinematic"):
        result = kinematic_pole_rvs(df)

    assert result is None


# ---------------------------------------------------------------------------
# RVS variant: non-empty subsample -> a result (not None), axis reasonably
# close to truth.
# ---------------------------------------------------------------------------

def test_kinematic_pole_rvs_returns_result_when_rvs_present():
    df = synthetic_catalog(
        n=3000, seed=17, pole=(200.0, 15.0), outlier_fraction=0.0,
        dist_range_kpc=(0.2, 1.0), include_proper_motion=True,
        v_circ_kms=220.0, include_rvs=True, rvs_fraction=0.5,
    )
    true_axis = df.attrs["truth"]["rotation_axis"]
    true_ra_deg, true_dec_deg = _true_ra_dec_deg(true_axis)

    result = kinematic_pole_rvs(df)

    assert result is not None
    assert result["method"] == "kinematic_rvs_angmom"
    assert result["n_used"] < result["n_stars"]

    err = _pole_angular_error_deg(
        result["alpha_NGP"], result["delta_NGP"], true_ra_deg, true_dec_deg
    )
    assert err < 0.2, f"RVS kinematic pole error={err:.5f}deg"


# ---------------------------------------------------------------------------
# B4.6/B4.7 — solar_motion is an explicit parameter that shifts the result
# in a documented direction: an INCORRECT solar_motion=(0,0,0) on data
# generated WITH the correct nonzero solar reflex must give a WORSE
# (larger angular error) result than using the correct solar_motion.
# Spec F4-R3-S1.
# ---------------------------------------------------------------------------

def test_kinematic_pole_solar_motion_reflex_correction_matters():
    true_solar_motion = (11.1, 12.24, 7.25)
    df = synthetic_catalog(
        n=3000, seed=23, pole=(200.0, 15.0), outlier_fraction=0.0,
        dist_range_kpc=(0.2, 1.0), include_proper_motion=True,
        v_circ_kms=220.0, solar_motion=true_solar_motion,
    )
    true_axis = df.attrs["truth"]["rotation_axis"]
    true_ra_deg, true_dec_deg = _true_ra_dec_deg(true_axis)

    correct = kinematic_pole(df, solar_motion=true_solar_motion)
    wrong = kinematic_pole(df, solar_motion=(0.0, 0.0, 0.0))

    err_correct = _pole_angular_error_deg(
        correct["alpha_NGP"], correct["delta_NGP"], true_ra_deg, true_dec_deg
    )
    err_wrong = _pole_angular_error_deg(
        wrong["alpha_NGP"], wrong["delta_NGP"], true_ra_deg, true_dec_deg
    )

    assert err_wrong > err_correct, (
        f"err_wrong={err_wrong:.5f}deg should exceed "
        f"err_correct={err_correct:.5f}deg -- reflex-subtraction wiring "
        f"must matter"
    )
    assert err_correct < 0.2
