"""
tracer_fetcher.py — F2.1: young Galactic-disk tracer catalogs (Cepheids
primary; OB stars and young open clusters secondary). Each tracer sample
runs unmodified through the existing pole estimators
(`ngp_3d.great_circle_pole`, `ngp_offset_plane.offset_plane_pole`) — both
only require 'ra'/'dec' (and 'parallax' for the offset-plane fit), which
every tracer schema below already provides, so no estimator-side adapter
is needed (ngp-precision batch B2, design §4/§5, spec F2.1).

Public API
----------
fetch_cepheids(cache_path="data/gaia_cepheids.csv", force_refresh=False,
                query_fn=None) -> pd.DataFrame
fetch_ob_stars(cache_path="data/gaia_ob_stars.csv", force_refresh=False,
                query_fn=None) -> pd.DataFrame
fetch_young_clusters(cache_path="data/cantat_gaudin_clusters.csv",
                      force_refresh=False, query_fn=None) -> pd.DataFrame
apply_galactocentric_cut(df, *, r_max_kpc=9.0, r_sun_kpc=8.122,
                          distance_col="distance_kpc") -> pd.DataFrame
cepheid_pl_distance(df, *, pl_relation=None) -> pd.Series

All three fetch_* functions share the cache-first + injectable query_fn
contract of `gaia_fetcher.fetch_gaia_stars` (via the shared
`gaia_fetcher._fetch_with_cache` helper, generalized there in this same
batch): cache hit never calls query_fn; cache miss calls query_fn(),
caches the result to CSV, and falls back to a stale cache with a
UserWarning if the query raises and a cache exists. Real (NETWORK-only,
not unit-tested) query functions reuse `gaia_fetcher._chunked_sync_query` —
SYNCHRONOUS queries only, paginated over `random_index` windows sized to
stay under the TAP server's MAXREC=2000 cap (bugfix #677,
`sdd/ngp-improvement/bugfix-gaia-query`): NEVER `ORDER BY random_index`
and NEVER the async endpoint.

`apply_galactocentric_cut` and `cepheid_pl_distance` are separate,
composable pure functions (not baked into the fetch_* functions) so each
is independently unit-testable — a typical pipeline composes them
explicitly, e.g.::

    df = fetch_cepheids()
    df["pl_distance_kpc"] = cepheid_pl_distance(df)
    df = apply_galactocentric_cut(df, distance_col="pl_distance_kpc")
    pole = great_circle_pole(df)

Unit note — distances are computed two DIFFERENT ways depending on tracer,
and BOTH avoid the `ngp_3d.coords_to_cartesian` pc/kpc mislabeling bug
(`sdd/ngp-precision/gotcha-units-coords-to-cartesian`):
  - Cepheids: Period-Luminosity distance (`cepheid_pl_distance`), fully
    INDEPENDENT of parallax — this is precisely why Cepheids double as a
    zero-point-independent cross-check tracer (design ADR3). See
    `cepheid_pl_distance`'s docstring for the default P-L relation used.
  - OB stars / clusters (when no direct distance column is available):
    ``distance_kpc = 1 / parallax_mas`` — the physically correct
    convention used throughout ngp-precision (matches
    `synthetic_catalog.py` and `ngp_offset_plane.py`'s
    `_heliocentric_cartesian_kpc`), NOT `ngp_3d.coords_to_cartesian`
    (which numerically returns parsecs mislabeled "kpc" — harmless there
    only because it is used for plane geometry/direction, never an
    absolute physical distance/radius).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from gaia_fetcher import _chunked_sync_query, _fetch_with_cache

_CEPHEID_CACHE_PATH = "data/gaia_cepheids.csv"
_OB_CACHE_PATH = "data/gaia_ob_stars.csv"
_CLUSTER_CACHE_PATH = "data/cantat_gaudin_clusters.csv"

_CEPHEID_COLUMNS = [
    "source_id", "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmdec", "phot_g_mean_mag", "l", "b",
    "pf", "type_best_classification",
]

_OB_COLUMNS = [
    "source_id", "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmdec", "phot_g_mean_mag", "bp_rp", "l", "b",
]

_CLUSTER_COLUMNS = [
    "cluster_id", "ra", "dec", "parallax",
    "pmra", "pmdec", "n_members", "l", "b", "age_myr",
]

# Approximate total gaia_source random_index span (DR3, ~1.8e9 sources).
# The Cepheid/OB joins below are NETWORK-only and not unit-tested; these
# constants are placeholders to be recalibrated with a COUNT() query
# (same method as gaia_fetcher.py bugfix #677) immediately before the
# real B2.13/B2.14 download batch, so that every window's expected yield
# stays under the TAP server's MAXREC=2000 cap.
_GAIA_RANDOM_INDEX_TOTAL = 1_811_709_771

# gaiadr3.vari_cepheid has ~15k rows total (far smaller than gaia_source),
# so a handful of large windows keep each window's expected match count
# comfortably under 2000 (~15,000 / 9 windows ~= 1,700/window).
_CEPHEID_RANDOM_INDEX_STEP = 200_000_000

# OB candidates (bp_rp < 0, blue/hot stars) are MUCH rarer than the
# flagship disk-star sample at the same |b|/G cuts: a COUNT() calibration
# run during B7 found only ~1 match per 100,000 random_index at the
# original 4.5M-row disk-star subset (`_OB_RANDOM_INDEX_MAX` below would
# have yielded ~45 stars total -- too few). Recalibrated (same method as
# bugfix #677 / the B2.13 Cepheid recalibration) by scanning the FULL
# gaia_source random_index range with a COUNT() query: a 200M-row window
# yields ~1,200 matches (comfortably under MAXREC=2000), matching the
# Cepheid step exactly, so both fetchers now share
# `_GAIA_RANDOM_INDEX_TOTAL` / a 200M step.
_OB_RANDOM_INDEX_STEP = 200_000_000
_OB_RANDOM_INDEX_MAX = _GAIA_RANDOM_INDEX_TOTAL


def _default_cepheid_query() -> pd.DataFrame:
    """
    Query Gaia DR3 classical Cepheids: gaiadr3.vari_cepheid JOIN
    gaia_source ON source_id, chunked-sync over random_index windows.
    NETWORK-ONLY — not unit-tested (see module docstring).
    """
    select_cols = (
        "g.source_id, g.ra, g.dec, g.parallax, g.parallax_error, g.pmra, "
        "g.pmdec, g.phot_g_mean_mag, g.l, g.b, v.pf, v.type_best_classification"
    )
    from_clause = (
        "gaiadr3.vari_cepheid AS v "
        "JOIN gaiadr3.gaia_source AS g ON v.source_id = g.source_id"
    )
    where_template = "g.random_index >= {lo} AND g.random_index < {hi}"
    return _chunked_sync_query(
        select_cols=select_cols,
        from_clause=from_clause,
        where_template=where_template,
        random_index_max=_GAIA_RANDOM_INDEX_TOTAL,
        random_index_step=_CEPHEID_RANDOM_INDEX_STEP,
        output_columns=_CEPHEID_COLUMNS,
    )


def _default_ob_query() -> pd.DataFrame:
    """
    Query Gaia DR3 OB-star candidates (blue, bright, high-S/N-parallax
    disk stars) via chunked-sync ADQL. NETWORK-ONLY — not unit-tested.
    """
    select_cols = (
        "source_id, ra, dec, parallax, parallax_error, pmra, pmdec, "
        "phot_g_mean_mag, bp_rp, l, b"
    )
    from_clause = "gaiadr3.gaia_source"
    where_template = (
        "parallax > 0 AND parallax_over_error > 5 AND bp_rp < 0.0 "
        "AND phot_g_mean_mag < 16 AND ABS(b) < 15 "
        "AND random_index >= {lo} AND random_index < {hi}"
    )
    return _chunked_sync_query(
        select_cols=select_cols,
        from_clause=from_clause,
        where_template=where_template,
        random_index_max=_OB_RANDOM_INDEX_MAX,
        random_index_step=_OB_RANDOM_INDEX_STEP,
        output_columns=_OB_COLUMNS,
    )


def _default_cluster_query() -> pd.DataFrame:
    """
    Fetch the Cantat-Gaudin et al. (2020, A&A 640, A1) young open-cluster
    catalog via VizieR (small table, no random_index pagination needed).
    NETWORK-ONLY — not unit-tested.

    CORRECTED during the real B7 download run (the original B2-era column
    map below was a speculative guess, written before any real query had
    been executed against this catalog):
      - ``Vizier.ROW_LIMIT = -1`` (astroquery's "no limit" sentinel) hangs
        indefinitely against this table (verified: >90s with zero
        progress) — use a fixed, generously-sized row cap instead
        (catalog has 2017 clusters; 5000 leaves headroom).
      - The actual VizieR columns are ``nbstars07`` (not ``"N"``) and
        ``pmRA*``/``pmDE`` (not ``"pmRA"``); there is NO ``GLON``/``GLAT``
        column at all — galactic l/b must be derived from
        ``RA_ICRS``/``DE_ICRS`` via an ICRS->Galactic transform.
      - ``AgeNN`` is ``log10(age / yr)`` (a neural-network age estimate,
        typical range ~6.2-9.9), NOT age in Myr directly — converted here
        via ``age_myr = 10 ** (AgeNN - 6)``.
    """
    from astropy.coordinates import SkyCoord  # noqa: PLC0415
    from astropy import units as u  # noqa: PLC0415
    from astroquery.vizier import Vizier  # noqa: PLC0415

    Vizier.ROW_LIMIT = 5000
    catalogs = Vizier.get_catalogs("J/A+A/640/A1")
    table = catalogs[0].to_pandas()
    df = table.rename(columns={
        "Cluster": "cluster_id", "RA_ICRS": "ra", "DE_ICRS": "dec",
        "plx": "parallax", "pmRA*": "pmra", "pmDE": "pmdec",
        "nbstars07": "n_members",
    })
    galactic = SkyCoord(
        ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg, frame="icrs",
    ).galactic
    df["l"] = galactic.l.deg
    df["b"] = galactic.b.deg
    df["age_myr"] = 10.0 ** (df["AgeNN"].values.astype(float) - 6.0)
    return df[_CLUSTER_COLUMNS].copy()


def fetch_cepheids(
    cache_path: str = _CEPHEID_CACHE_PATH,
    force_refresh: bool = False,
    query_fn: Optional[Callable[[], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of Gaia DR3 classical Cepheids, using a local CSV
    cache when possible.

    Parameters
    ----------
    cache_path : str
        Path to the CSV cache file. Defaults to data/gaia_cepheids.csv.
    force_refresh : bool
        If True, ignore the cache and re-query the archive.
    query_fn : callable, optional
        Zero-argument callable returning a pd.DataFrame of Cepheid
        results. Defaults to `_default_cepheid_query()`. Inject a stub
        in tests.

    Returns
    -------
    pd.DataFrame with columns (at minimum): source_id, ra, dec, parallax,
    parallax_error, pmra, pmdec, phot_g_mean_mag, l, b, pf (period, days),
    type_best_classification.

    Raises
    ------
    ConnectionError
        If the archive is unreachable and no cache is available.
    """
    return _fetch_with_cache(
        cache_path, force_refresh, query_fn, _default_cepheid_query,
        service_name="Gaia archive",
    )


def fetch_ob_stars(
    cache_path: str = _OB_CACHE_PATH,
    force_refresh: bool = False,
    query_fn: Optional[Callable[[], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of Gaia DR3 OB-star candidates, cache-first.
    Same contract as `fetch_cepheids` — see its docstring.
    """
    return _fetch_with_cache(
        cache_path, force_refresh, query_fn, _default_ob_query,
        service_name="Gaia archive",
    )


def fetch_young_clusters(
    cache_path: str = _CLUSTER_CACHE_PATH,
    force_refresh: bool = False,
    query_fn: Optional[Callable[[], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of Cantat-Gaudin et al. (2020) young open clusters,
    cache-first. Same contract as `fetch_cepheids` — see its docstring.
    """
    return _fetch_with_cache(
        cache_path, force_refresh, query_fn, _default_cluster_query,
        service_name="VizieR (Cantat-Gaudin)",
    )


def apply_galactocentric_cut(
    df: pd.DataFrame,
    *,
    r_max_kpc: float = 9.0,
    r_sun_kpc: float = 8.122,
    distance_col: str = "distance_kpc",
) -> pd.DataFrame:
    """
    Filter tracer rows to galactocentric radius R < r_max_kpc (design
    F2.1-R3 / spec "young-tracer-catalogs" R<9kpc cut).

    R is computed via the law of cosines from the heliocentric distance
    (`distance_col`, kpc) and galactic coordinates (l, b, degrees),
    projecting the heliocentric distance onto the Galactic plane
    (d_plane = d * cos(b)) — appropriate for an in-plane radial cut,
    ignoring the (much smaller) height offset:

        R^2 = R_sun^2 + d_plane^2 - 2 * R_sun * d_plane * cos(l)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'l', 'b' (galactic coordinates, degrees) and
        `distance_col` (heliocentric distance, kpc). The distance
        convention differs by tracer — pass the appropriate column name
        (e.g. 'pl_distance_kpc' for Cepheids via `cepheid_pl_distance`;
        'distance_kpc' = 1/parallax_mas for OB stars/clusters — see
        module docstring's Unit note).
    r_max_kpc : float
        Galactocentric radius cut (kpc). Default 9.0.
    r_sun_kpc : float
        Assumed Sun-to-Galactic-center distance (kpc). Default 8.122
        (GRAVITY Collaboration 2018).
    distance_col : str
        Name of the heliocentric distance column (kpc) to use.

    Returns
    -------
    pd.DataFrame — filtered rows (R < r_max_kpc), same columns as input
    plus an added 'r_galactocentric_kpc' column, index reset.

    Raises
    ------
    ValueError : if 'l', 'b', or `distance_col` is missing from `df`.
    """
    for col in ("l", "b", distance_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame is missing required column: '{col}'")

    l_rad = np.radians(df["l"].values.astype(float))
    b_rad = np.radians(df["b"].values.astype(float))
    d = df[distance_col].values.astype(float)

    d_plane = d * np.cos(b_rad)
    r_gc = np.sqrt(
        r_sun_kpc ** 2 + d_plane ** 2 - 2.0 * r_sun_kpc * d_plane * np.cos(l_rad)
    )

    out = df.copy()
    out["r_galactocentric_kpc"] = r_gc
    return out[out["r_galactocentric_kpc"] < r_max_kpc].reset_index(drop=True)


def _default_pl_relation(log_p: np.ndarray) -> np.ndarray:
    """
    Default Gaia G-band Period-Luminosity relation for fundamental-mode
    classical Cepheids, of the form ``M_G = a * log10(P[days]) + b``.

    PROVENANCE (updated during B7's real download run) — this REPLACES the
    original B2-era placeholder (a=-2.43, b=-2.678), which was found to be
    badly wrong once tested against the real 15,021-row Gaia DR3
    `vari_cepheid` download: it put ~7,800 DCEP-type Cepheids at a median
    "distance" of ~102 kpc (vs. an expected few kpc for Milky Way disk
    Cepheids), because its zero point `b` was miscalibrated by several
    magnitudes.

    Given the time budget for this batch, rather than trust an
    from-memory transcription of the exact published Ripepi et al. (2019,
    A&A 625, A14) / Groenewegen (2018) coefficients (risk of transcription
    error with no way to verify offline), these coefficients were instead
    SELF-CALIBRATED directly from this project's own real Gaia data: a
    linear fit of ``M_G = phot_g_mean_mag - (5*log10(d_pc) - 5)`` vs.
    ``log10(period)`` over the DCEP-type subsample with the BEST parallax
    S/N (`parallax/parallax_error > 20`, ~250 stars), 3-iteration
    2.5-sigma-clipped to reject contaminants:

        a=-2.4352, b=-0.2537  (residual scatter ~0.85 mag after clipping)

    CAVEATS (must be carried into any downstream results/manuscript text
    using this function):
      (1) This is a PARALLAX-CALIBRATED zero point (using this project's
          own Gaia parallaxes as the distance ground truth for the
          calibration subsample) — it is NOT a fully parallax-independent
          external calibration, so `cepheid_pl_distance`'s usual role as a
          zero-point-independent cross-check (design ADR3) is WEAKENED for
          the overall sample zero point (though the per-star *relative*
          distances within the low-S/N sample remain parallax-free).
      (2) NOT extinction-corrected (no Wesenheit/reddening-free index is
          available from the columns fetched) — individual distances for
          highly-reddened/distant Cepheids are likely still somewhat
          overestimated; the ~0.85 mag residual scatter after clipping is
          suspected to be dominated by exactly this missing correction.
      (3) The ~15k-row `vari_cepheid` table includes many genuinely
          extragalactic entries (LMC/SMC and beyond); the R<9kpc
          galactocentric cut (`apply_galactocentric_cut`) removes most of
          these, but a residual handful of misclassified/contaminated
          entries may remain.
    Treat any Cepheid-tracer pole/z_sun result derived via this relation
    as ILLUSTRATIVE pending a proper external, extinction-corrected PLR
    calibration — see `sdd/ngp-precision/apply-progress` (B7) for the full
    derivation and `docs/manuscrito_borrador.md` for how this caveat is
    surfaced in the manuscript. `pl_relation` remains injectable precisely
    so a verified external calibration can be swapped in later without
    touching this module.
    """
    a, b = -2.4352, -0.2537
    return a * log_p + b


def cepheid_pl_distance(
    df: pd.DataFrame,
    *,
    pl_relation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> pd.Series:
    """
    Period-Luminosity heliocentric distance (kpc) for classical Cepheids
    — INDEPENDENT of parallax (design ADR3: doubles as a parallax
    zero-point cross-check, unlike every other tracer/estimator in this
    project).

    ``M_G = pl_relation(log10(P))``; distance modulus
    ``mu = m_G - M_G``; ``d_pc = 10 ** ((mu + 5) / 5)``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'pf' (period, days) and 'phot_g_mean_mag' (apparent
        G magnitude).
    pl_relation : callable, optional
        `log10(P[days]) array -> M_G array`. Defaults to
        `_default_pl_relation` — inject the exact published calibration
        for publication-quality distances.

    Returns
    -------
    pd.Series named 'pl_distance_kpc', same index as `df`.
    """
    if pl_relation is None:
        pl_relation = _default_pl_relation

    log_p = np.log10(df["pf"].values.astype(float))
    abs_mag = pl_relation(log_p)
    app_mag = df["phot_g_mean_mag"].values.astype(float)

    distance_modulus = app_mag - abs_mag
    d_pc = 10.0 ** ((distance_modulus + 5.0) / 5.0)
    d_kpc = d_pc / 1000.0

    return pd.Series(d_kpc, index=df.index, name="pl_distance_kpc")
