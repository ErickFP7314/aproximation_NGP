"""
iau_forensics.py -- convention forensics: WHY does our measured NGP differ
from the 1958 IAU galactic-pole convention? (ngp-precision batch B6, spec
capability F6 "convention-forensics").

This module does NOT attempt a literal re-derivation of the 1958
Blaauw-et-al. photographic-plate reduction (the original plate measurements
and reduction pipeline are not reproducible from first principles today).
Instead it provides a QUANTITATIVE COMPARISON FRAMEWORK that:

  1. Reproduces the standard published J2000/ICRS IAU pole from the original
     1958 B1950/FK4 definition using astropy's coordinate machinery
     (`b1950_to_j2000_pole`) -- this isolates the (tiny) FK4->FK5/ICRS
     transformation step from everything else.
  2. Decomposes the divergence between OUR measured pole and the IAU J2000
     pole into an "error budget" of three named terms plus a modern
     independent cross-check against Karim & Mamajek (2017, MNRAS 465, 472)
     (`decompose_divergence`).

Key numbers used as defaults (see design doc / spec F6, and Liu, Zhu &
Zhang 2011, A&A 526, A16 for the FK4->FK5 axis-orthogonality mismatch):

  IAU_B1950_POLE  = (192.25, 27.4)         -- original 1958 definition (deg, FK4/B1950)
  IAU_J2000_POLE  = (192.859508, 27.128336) -- standard published J2000/ICRS conversion
  KM2017_POLE     = (192.729, 27.084)       -- Karim & Mamajek (2017) independent modern
                                                measurement from young Galactic tracers
  DEFAULT_FK4_FK5_ARTIFACT_DEG = 0.377/3600 -- Liu, Zhu & Zhang (2011): the FK4->FK5/ICRS
                                                transform of the galactic axes is not
                                                perfectly orthogonal; ~0.377 arcsec
                                                mismatch at J2000 (~0.0001 deg -- tiny
                                                compared to the ~0.05-0.1 deg differences
                                                seen elsewhere in this project)

IMPORTANT CAVEAT on "decomposition terms sum to total" (spec F6-R2-S1):
Angular separations on a sphere do NOT add linearly in general (they only
add exactly along a single great circle / when three poles are collinear).
We cannot re-derive the true 1958 measurement error independently, so
`decompose_divergence` implements this as an explicit ERROR BUDGET (in the
same spirit as `systematics.combine_error_budget`'s sigma_stat/sigma_syst
split): two terms are estimated from independent, physically-motivated
quantities (the historical IAU-vs-modern-truth gap, net of the tiny
FK4->FK5 artifact), and the THIRD term ("gas-vs-stellar-plane difference")
is DEFINED as the remainder needed to close the budget exactly against the
observed total. This makes the "terms sum to total" identity hold by
construction (residual_deg is a numerical-precision health check, not a
physical residual) while keeping every term individually interpretable:

    total_deg                 = angular_separation(our_pole, iau_pole_j2000)
    historical_total_deg      = angular_separation(iau_pole_j2000, km2017_pole)
    term_fk4_fk5_artifact_deg = DEFAULT_FK4_FK5_ARTIFACT_DEG (or override)
    term_1958_measurement_deg = historical_total_deg - term_fk4_fk5_artifact_deg
                                (clipped at 0; dominated by 1958-era measurement
                                 error since the FK4->FK5 artifact itself is
                                 only ~0.0001 deg -- Liu, Zhu & Zhang 2011)
    term_gas_vs_stars_deg     = total_deg - term_1958_measurement_deg
                                          - term_fk4_fk5_artifact_deg
                                (the part of OUR divergence from the IAU pole
                                 that is NOT explained by the known historical
                                 IAU-vs-modern-truth gap; a proxy for the real
                                 physical difference between the plane traced
                                 by our estimator and the modern young-stellar
                                 tracer plane, plus any of our own residual
                                 systematics -- see `error_vs_km2017_deg` for
                                 the direct, non-decomposed cross-check)

For an independent, non-decomposed sanity check, `error_vs_km2017_deg` is
also returned: the literal angular separation between OUR pole and the
Karim & Mamajek (2017) pole (NOT part of the additive budget).

Public API:
    b1950_to_j2000_pole(ra_b1950_deg, dec_b1950_deg) -> tuple[float, float]
    decompose_divergence(our_pole, iau_pole_j2000, *, km2017_pole=..., fk4_fk5_artifact_deg=...) -> dict
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# --- Reference poles (design section 4 / spec F6) --------------------------

IAU_B1950_POLE = (192.25, 27.4)           # original 1958 definition (deg, FK4/B1950)
IAU_J2000_POLE = (192.859508, 27.128336)  # standard published J2000/ICRS conversion
KM2017_POLE = (192.729, 27.084)           # Karim & Mamajek (2017), MNRAS 465, 472

# Liu, Zhu & Zhang (2011), A&A 526, A16: the FK4->FK5/ICRS transform of the
# galactic coordinate axes is not perfectly rigorous -- the transformed
# Galactic-center axis and NGP axis are not exactly orthogonal, with a
# ~0.377 arcsec mismatch at J2000.
DEFAULT_FK4_FK5_ARTIFACT_DEG = 0.377 / 3600.0


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _angular_separation_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    """
    Great-circle angular separation (deg) between two (RA-deg, Dec-deg)
    points, via the haversine formula. The haversine form (rather than the
    spherical law of cosines + arccos) stays numerically stable for very
    small separations -- this module compares poles that can be identical
    or only arcsecond-scale apart, where arccos(cos_sep~1) loses precision.
    """
    ra1, dec1 = np.radians(ra1_deg), np.radians(dec1_deg)
    ra2, dec2 = np.radians(ra2_deg), np.radians(dec2_deg)
    d_ra = ra1 - ra2
    d_dec = dec1 - dec2
    a = np.sin(d_dec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return float(np.degrees(c))


# ---------------------------------------------------------------------------
# B6.2 -- b1950_to_j2000_pole
# ---------------------------------------------------------------------------

def b1950_to_j2000_pole(ra_b1950_deg: float, dec_b1950_deg: float) -> Tuple[float, float]:
    """
    Transform a galactic-pole position from B1950/FK4 to J2000/ICRS using
    astropy's coordinate machinery (NOT a hand-rolled precession formula).

    Parameters
    ----------
    ra_b1950_deg, dec_b1950_deg : float
        Pole position in the B1950/FK4 frame (degrees).

    Returns
    -------
    (ra_j2000_deg, dec_j2000_deg) : tuple[float, float]
        Pole position transformed to ICRS (deg), which coincides with J2000
        to sub-milliarcsecond precision for this purpose. Reproduces the
        standard published J2000 IAU NGP position (192.859508, 27.128336)
        to << 0.001 deg when applied to `IAU_B1950_POLE`.
    """
    coord_b1950 = SkyCoord(
        ra=ra_b1950_deg * u.deg,
        dec=dec_b1950_deg * u.deg,
        frame="fk4",
        equinox="B1950",
    )
    coord_j2000 = coord_b1950.transform_to("icrs")
    return float(coord_j2000.ra.deg), float(coord_j2000.dec.deg)


# ---------------------------------------------------------------------------
# B6.4 -- decompose_divergence
# ---------------------------------------------------------------------------

def decompose_divergence(
    our_pole: Tuple[float, float],
    iau_pole_j2000: Tuple[float, float] = IAU_J2000_POLE,
    *,
    km2017_pole: Tuple[float, float] = KM2017_POLE,
    fk4_fk5_artifact_deg: float = DEFAULT_FK4_FK5_ARTIFACT_DEG,
) -> dict:
    """
    Decompose the angular divergence between OUR measured NGP pole and the
    published J2000 IAU pole into a three-term error budget, plus an
    independent cross-check against Karim & Mamajek (2017).

    Parameters
    ----------
    our_pole : tuple[float, float]
        (ra_deg, dec_deg) of our measured NGP pole. NOTE: this module works
        entirely in degrees; callers holding an estimator result with
        `alpha_NGP` in HOURS (e.g. `ngp_3d.great_circle_pole`) must convert
        via `ra_deg = alpha_NGP * 15.0` before calling.
    iau_pole_j2000 : tuple[float, float]
        (ra_deg, dec_deg) of the published J2000/ICRS IAU pole. Defaults to
        `IAU_J2000_POLE`.
    km2017_pole : tuple[float, float]
        (ra_deg, dec_deg) of the Karim & Mamajek (2017) modern independent
        pole estimate. Defaults to `KM2017_POLE`.
    fk4_fk5_artifact_deg : float
        The (tiny, ~fixed) FK4->FK5/ICRS transformation artifact from Liu,
        Zhu & Zhang (2011). Defaults to `DEFAULT_FK4_FK5_ARTIFACT_DEG`
        (0.377 arcsec ~ 0.0001 deg). Exposed as a parameter (not a hardcoded
        magic number) so callers can substitute a different published value.

    Returns
    -------
    dict with keys:
        total_deg                 : angular_separation(our_pole, iau_pole_j2000)
        term_1958_measurement_deg : historical IAU-vs-modern-truth gap, net
                                     of the FK4->FK5 artifact (dominated by
                                     1958-era measurement error)
        term_fk4_fk5_artifact_deg : the FK4->FK5/ICRS transform artifact
                                     (echoes `fk4_fk5_artifact_deg`)
        term_gas_vs_stars_deg     : remainder of `total_deg` not explained by
                                     the two terms above (see module
                                     docstring's CAVEAT -- this is the
                                     budget-closing term, a proxy for the
                                     real gas-vs-stellar-plane physical
                                     difference plus our own residual
                                     systematics)
        residual_deg              : total_deg - sum(the three terms above);
                                     by construction this is ~0 (floating-
                                     point level) -- it is a numerical
                                     sum-consistency check, NOT an
                                     independent physical residual
        error_vs_km2017_deg       : angular_separation(our_pole, km2017_pole)
                                     -- direct, non-decomposed cross-check
                                     against the modern independent
                                     reference pole
    """
    our_ra, our_dec = our_pole
    iau_ra, iau_dec = iau_pole_j2000
    km_ra, km_dec = km2017_pole

    total_deg = _angular_separation_deg(our_ra, our_dec, iau_ra, iau_dec)
    historical_total_deg = _angular_separation_deg(iau_ra, iau_dec, km_ra, km_dec)

    term_fk4_fk5_artifact_deg = float(fk4_fk5_artifact_deg)
    term_1958_measurement_deg = max(historical_total_deg - term_fk4_fk5_artifact_deg, 0.0)
    term_gas_vs_stars_deg = total_deg - term_1958_measurement_deg - term_fk4_fk5_artifact_deg

    residual_deg = total_deg - (
        term_1958_measurement_deg + term_fk4_fk5_artifact_deg + term_gas_vs_stars_deg
    )

    error_vs_km2017_deg = _angular_separation_deg(our_ra, our_dec, km_ra, km_dec)

    return {
        "total_deg": total_deg,
        "term_1958_measurement_deg": term_1958_measurement_deg,
        "term_fk4_fk5_artifact_deg": term_fk4_fk5_artifact_deg,
        "term_gas_vs_stars_deg": term_gas_vs_stars_deg,
        "residual_deg": residual_deg,
        "error_vs_km2017_deg": error_vs_km2017_deg,
    }
