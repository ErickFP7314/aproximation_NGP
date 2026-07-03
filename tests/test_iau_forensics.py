"""
Tests for iau_forensics.py -- F6 convention forensics: B1950->J2000 IAU pole
propagation and divergence decomposition (ngp-precision batch B6).
"""

import numpy as np
import pytest

from iau_forensics import (
    DEFAULT_FK4_FK5_ARTIFACT_DEG,
    IAU_B1950_POLE,
    IAU_J2000_POLE,
    KM2017_POLE,
    _angular_separation_deg,
    b1950_to_j2000_pole,
    decompose_divergence,
)


# ---------------------------------------------------------------------------
# B6.1/B6.2 -- b1950_to_j2000_pole. Spec F6-R1-S1.
# ---------------------------------------------------------------------------

def test_b1950_to_j2000_pole_matches_published_iau_j2000_pole():
    ra_j2000, dec_j2000 = b1950_to_j2000_pole(*IAU_B1950_POLE)

    sep_deg = _angular_separation_deg(ra_j2000, dec_j2000, *IAU_J2000_POLE)

    assert sep_deg < 0.001, (
        f"astropy FK4(B1950)->ICRS transform of the 1958 IAU pole "
        f"{IAU_B1950_POLE} gave ({ra_j2000:.6f}, {dec_j2000:.6f}) which is "
        f"{sep_deg:.6f} deg from the published J2000 pole {IAU_J2000_POLE}; "
        f"expected < 0.001 deg"
    )
    # Individually the RA/Dec components must also each be tight (not just
    # the combined angular separation, which could hide a large RA drift at
    # low cos(dec) weighting near a low-declination pole -- not the case
    # here at dec~27 deg, but checked explicitly for the exact spec values).
    assert abs(ra_j2000 - IAU_J2000_POLE[0]) < 0.001
    assert abs(dec_j2000 - IAU_J2000_POLE[1]) < 0.001


def test_b1950_to_j2000_pole_performs_a_real_transform_not_passthrough():
    """
    Triangulation: prove the function actually invokes astropy's coordinate
    transform (and isn't a hardcoded pass-through of the input), by checking
    the output measurably differs from the raw B1950 input by roughly the
    known ~0.6 deg B1950->J2000 galactic-pole shift.
    """
    ra_j2000, dec_j2000 = b1950_to_j2000_pole(*IAU_B1950_POLE)

    shift_deg = _angular_separation_deg(ra_j2000, dec_j2000, *IAU_B1950_POLE)

    assert shift_deg > 0.5, (
        f"Expected a real B1950->J2000 shift of several tenths of a degree "
        f"(known value ~0.6 deg), got only {shift_deg:.6f} deg -- looks like "
        f"a pass-through rather than an actual coordinate transform"
    )
    assert isinstance(ra_j2000, float) and isinstance(dec_j2000, float)


def test_b1950_to_j2000_pole_different_input_gives_different_output():
    """Triangulation: a different B1950 input must not collapse to the same output."""
    ra_a, dec_a = b1950_to_j2000_pole(*IAU_B1950_POLE)
    ra_b, dec_b = b1950_to_j2000_pole(0.0, 0.0)

    assert (ra_a, dec_a) != (ra_b, dec_b)
    # (0,0) B1950 should land far from the galactic pole region.
    assert _angular_separation_deg(ra_b, dec_b, ra_a, dec_a) > 50.0


# ---------------------------------------------------------------------------
# B6.3/B6.4 -- decompose_divergence. Spec F6-R2-S1.
# ---------------------------------------------------------------------------

def test_decompose_divergence_terms_sum_to_total_for_real_measured_pole():
    """
    Uses this project's actual `great_circle_pole` regression result on
    53,082 real Gaia DR3 stars (alpha_NGP=12.946260h -> 194.193900deg,
    delta_NGP=26.492430deg -- see apply-progress B1.7/B5.8) as `our_pole`.
    """
    our_pole = (12.946260 * 15.0, 26.492430)

    result = decompose_divergence(our_pole, IAU_J2000_POLE)

    terms_sum = (
        result["term_1958_measurement_deg"]
        + result["term_fk4_fk5_artifact_deg"]
        + result["term_gas_vs_stars_deg"]
    )
    assert abs(terms_sum - result["total_deg"]) < 0.001
    assert abs(result["residual_deg"]) < 0.001

    # total_deg must be a REAL, non-trivial angular separation (not 0/empty).
    assert result["total_deg"] > 0.5
    assert result["total_deg"] == pytest.approx(
        _angular_separation_deg(*our_pole, *IAU_J2000_POLE), abs=1e-9
    )


def test_decompose_divergence_synthetic_pole_equal_to_km2017_zeroes_gas_term():
    """
    Triangulation / spec sanity check: if OUR measured pole happened to
    exactly equal the Karim & Mamajek (2017) modern reference pole, the
    "real gas-vs-stellar-plane difference" term must vanish -- ALL of the
    divergence from the IAU pole is then explained by the known historical
    (1958-era + FK4->FK5) gap, by construction.
    """
    result = decompose_divergence(KM2017_POLE, IAU_J2000_POLE)

    assert result["term_gas_vs_stars_deg"] == pytest.approx(0.0, abs=1e-9)
    assert result["error_vs_km2017_deg"] == pytest.approx(0.0, abs=1e-9)
    # total_deg must equal the historical IAU-vs-KM2017 gap exactly in this case.
    assert result["total_deg"] == pytest.approx(
        _angular_separation_deg(*IAU_J2000_POLE, *KM2017_POLE), abs=1e-9
    )
    terms_sum = (
        result["term_1958_measurement_deg"]
        + result["term_fk4_fk5_artifact_deg"]
        + result["term_gas_vs_stars_deg"]
    )
    assert abs(terms_sum - result["total_deg"]) < 0.001


def test_decompose_divergence_our_pole_equal_to_iau_zeroes_total():
    """Triangulation: our_pole == iau_pole_j2000 exactly -> total_deg == 0."""
    result = decompose_divergence(IAU_J2000_POLE, IAU_J2000_POLE)

    assert result["total_deg"] == pytest.approx(0.0, abs=1e-9)
    terms_sum = (
        result["term_1958_measurement_deg"]
        + result["term_fk4_fk5_artifact_deg"]
        + result["term_gas_vs_stars_deg"]
    )
    assert abs(terms_sum - result["total_deg"]) < 0.001
    # gas-vs-stars term absorbs the (negative) remainder needed to close the
    # budget against a total of exactly 0.
    assert result["term_gas_vs_stars_deg"] == pytest.approx(
        -(result["term_1958_measurement_deg"] + result["term_fk4_fk5_artifact_deg"]),
        abs=1e-9,
    )


def test_decompose_divergence_term_fk4_fk5_artifact_uses_documented_default():
    """The FK4->FK5 artifact term must echo the documented Liu/Zhu/Zhang default."""
    result = decompose_divergence(KM2017_POLE, IAU_J2000_POLE)

    assert result["term_fk4_fk5_artifact_deg"] == pytest.approx(
        DEFAULT_FK4_FK5_ARTIFACT_DEG, abs=1e-12
    )
    # Documented magnitude: ~0.377 arcsec ~ 0.0001 deg -- tiny compared to
    # the ~0.05-0.1 deg differences seen elsewhere in this project.
    assert result["term_fk4_fk5_artifact_deg"] < 0.001


def test_decompose_divergence_custom_fk4_fk5_artifact_is_honored():
    """Triangulation: overriding fk4_fk5_artifact_deg changes the returned term."""
    custom_artifact = 0.5 / 3600.0  # deliberately different from the default
    result = decompose_divergence(
        KM2017_POLE, IAU_J2000_POLE, fk4_fk5_artifact_deg=custom_artifact
    )

    assert result["term_fk4_fk5_artifact_deg"] == pytest.approx(custom_artifact, abs=1e-12)
    terms_sum = (
        result["term_1958_measurement_deg"]
        + result["term_fk4_fk5_artifact_deg"]
        + result["term_gas_vs_stars_deg"]
    )
    assert abs(terms_sum - result["total_deg"]) < 0.001


def test_decompose_divergence_term_1958_measurement_dominates_historical_gap():
    """
    Background fact: the FK4->FK5 artifact (~0.0001 deg) is tiny relative to
    the ~0.12 deg IAU-vs-KM2017 historical gap, so term_1958_measurement_deg
    must dominate that gap (not the artifact term).
    """
    result = decompose_divergence(KM2017_POLE, IAU_J2000_POLE)

    assert result["term_1958_measurement_deg"] > 10 * result["term_fk4_fk5_artifact_deg"]
