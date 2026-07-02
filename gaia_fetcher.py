"""
gaia_fetcher.py — Download and cache Gaia DR3 disk stars.

Public API:
    fetch_gaia_stars(cache_path, force_refresh, query_fn) -> pd.DataFrame
    filter_valid_parallax(df) -> pd.DataFrame
    _default_gaia_query() -> pd.DataFrame   [NETWORK-only; not unit-tested]
"""

import os
import warnings

import pandas as pd

# Required output columns from the Gaia query
_GAIA_COLUMNS = [
    "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmdec", "phot_g_mean_mag", "l", "b",
]

_DEFAULT_CACHE_PATH = "data/gaia_disk_stars.csv"

# We sample a uniform random subset by filtering on `random_index` (a random
# permutation of the ~1.8e9 sources) rather than `ORDER BY random_index`, which
# forces a full-table sort and is rejected by the TAP server (HTTP 500). At the
# current selectivity (~1.17% of the index passes the astrophysical filters),
# random_index < 4.5e6 yields ~50k stars. This subset is spatially unbiased.
_RANDOM_INDEX_MAX = 4_500_000

_ADQL_QUERY = f"""
SELECT
    ra, dec, parallax, parallax_error, pmra, pmdec,
    phot_g_mean_mag, l, b
FROM gaiadr3.gaia_source
WHERE parallax > 0
  AND parallax_over_error > 5
  AND ABS(b) < 15
  AND phot_g_mean_mag < 15
  AND random_index < {_RANDOM_INDEX_MAX}
"""


# Synchronous TAP queries are capped at MAXREC=2000 rows server-side, and the
# asynchronous endpoint's result storage is intermittently unavailable for
# anonymous jobs (HTTP 500 "Path does not exists"). We therefore paginate over
# `random_index` windows with synchronous queries, keeping each window's expected
# yield (~1.17% selectivity) well under the 2000-row cap to avoid silent
# truncation. This is slower but robust to the async outage.
_RANDOM_INDEX_STEP = 100_000

_SELECT_COLS = "ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, l, b"


def _default_gaia_query() -> pd.DataFrame:
    """Query Gaia DR3 TAP service via chunked synchronous ADQL. NETWORK-ONLY — not unit-tested."""
    from astroquery.gaia import Gaia  # noqa: PLC0415

    frames = []
    for lo in range(0, _RANDOM_INDEX_MAX, _RANDOM_INDEX_STEP):
        hi = min(lo + _RANDOM_INDEX_STEP, _RANDOM_INDEX_MAX)
        query = (
            f"SELECT {_SELECT_COLS} "
            "FROM gaiadr3.gaia_source "
            "WHERE parallax > 0 AND parallax_over_error > 5 "
            "AND ABS(b) < 15 AND phot_g_mean_mag < 15 "
            f"AND random_index >= {lo} AND random_index < {hi}"
        )
        chunk = Gaia.launch_job(query).get_results().to_pandas()
        if len(chunk) >= 2000:  # MAXREC hit — window truncated, would bias sample
            warnings.warn(
                f"random_index window [{lo}, {hi}) hit the 2000-row cap; "
                "reduce _RANDOM_INDEX_STEP to avoid truncation",
                UserWarning,
                stacklevel=2,
            )
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    return df[_GAIA_COLUMNS].copy()


def fetch_gaia_stars(
    cache_path: str = _DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
    query_fn=None,
) -> pd.DataFrame:
    """
    Return a DataFrame of Gaia disk stars, using a local CSV cache when possible.

    Parameters
    ----------
    cache_path : str
        Path to the CSV cache file.  Defaults to data/gaia_disk_stars.csv.
    force_refresh : bool
        If True, ignore the cache and re-query the archive.
    query_fn : callable, optional
        Zero-argument callable that returns a pd.DataFrame of Gaia results.
        Defaults to _default_gaia_query().  Inject a stub in tests.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ConnectionError
        If the archive is unreachable and no cache is available.
    """
    if query_fn is None:
        query_fn = _default_gaia_query

    cache_exists = os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0

    # Cache-first: return immediately if cache is valid and no refresh requested
    if cache_exists and not force_refresh:
        return pd.read_csv(cache_path)

    # Attempt to fetch from archive
    try:
        df = query_fn()
    except Exception as exc:
        if cache_exists:
            warnings.warn(
                "Gaia archive unreachable — using cached data",
                UserWarning,
                stacklevel=2,
            )
            return pd.read_csv(cache_path)
        raise ConnectionError(
            f"Gaia archive unreachable and no local cache found at '{cache_path}': {exc}"
        ) from exc

    # Write to cache (create parent directory if needed)
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(cache_path, index=False)

    return df


def filter_valid_parallax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows where parallax > 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a 'parallax' column.

    Returns
    -------
    pd.DataFrame
    """
    return df[df["parallax"] > 0].copy()
