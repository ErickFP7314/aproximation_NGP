"""
Shared pytest fixtures for the ngp-improvement / ngp-precision test suites.
Provides a synthetic galactic-plane dataset with known ground truth.

`_make_synthetic_disk_stars` / `synthetic_disk_stars` are now a thin,
back-compat wrapper around `synthetic_catalog.synthetic_catalog` (F5.4,
ngp-precision batch B0) — all new-effect parameters (z_sun, warp, extinction
mask, proper motion, RVS) default to "off", reproducing the original
ngp-improvement fixture's N=500/seed=42/pole=(192.75,27.11)/10%-outlier
behaviour statistically. See `synthetic_catalog.py` for the frozen parameter
contract shared by later ngp-precision batches (F1/F3/F4/F5).
"""

import pandas as pd
import pytest

from synthetic_catalog import synthetic_catalog

# IAU reference NGP position
_NGP_RA_DEG = 192.75   # alpha = 12h51m
_NGP_DEC_DEG = 27.11   # delta


def _make_synthetic_disk_stars(N: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate N synthetic stars on the Galactic plane with known NGP normal.

    The Galactic plane has its normal pointing toward (ra=192.75°, dec=27.11°).
    Stars are distributed with:
    - ra/dec near the plane (Gaussian scatter σ~0.5° around the plane)
    - parallax in [2, 10] mas
    - ~10% random outliers displaced far from the plane

    Returns a DataFrame with columns: ra, dec, parallax, l, b
    plus placeholder columns parallax_error, pmra, pmdec, phot_g_mean_mag.
    """
    return synthetic_catalog(n=N, seed=seed, pole=(_NGP_RA_DEG, _NGP_DEC_DEG))


@pytest.fixture
def synthetic_disk_stars():
    """
    Pytest fixture: 500-star synthetic galactic-plane dataset.
    Known NGP normal at ra=192.75°, dec=27.11°.
    Parallaxes in [2,10] mas; ~10% outliers.
    """
    return _make_synthetic_disk_stars(N=500, seed=42)
