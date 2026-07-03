"""
gaia_fetcher.py — Download and cache Gaia DR3 disk stars.

Public API:
    fetch_gaia_stars(cache_path, force_refresh, query_fn) -> pd.DataFrame
    filter_valid_parallax(df) -> pd.DataFrame
    _default_gaia_query() -> pd.DataFrame   [NETWORK-only; not unit-tested]

Internal helpers reused by `tracer_fetcher.py` (ngp-precision batch B2,
F2.1) — generalized out of this module rather than duplicated:
    _chunked_sync_query(...) -> pd.DataFrame   [pagination logic; the
        `launch_fn` hook makes this unit-testable OFFLINE without hitting
        the real TAP service — see tests/test_gaia_fetcher.py]
    _fetch_with_cache(...) -> pd.DataFrame     [cache-first + injectable
        query_fn contract shared by every fetch_* function project-wide]
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


def _chunked_sync_query(
    *,
    select_cols: str,
    from_clause: str,
    where_template: str,
    random_index_max: int,
    random_index_step: int,
    output_columns: list,
    launch_fn=None,
) -> pd.DataFrame:
    """
    Generic chunked-SYNCHRONOUS ADQL query, paginated over `random_index`
    windows (bugfix `sdd/ngp-improvement/bugfix-gaia-query`, #677): the
    Gaia TAP async endpoint's result storage is unreliable for anonymous
    jobs (HTTP 500 "path does not exist"), and SYNCHRONOUS queries are
    capped at MAXREC=2000 rows server-side. We therefore paginate
    SYNCHRONOUS queries over `random_index` windows sized so each
    window's expected row yield stays well under 2000 — NEVER
    `ORDER BY random_index` (forces a full-table sort, HTTP 500) and
    NEVER `launch_job_async` (unreliable result storage).

    This is the pagination logic shared by `_default_gaia_query` (disk
    stars) and `tracer_fetcher.py`'s real Cepheid/OB-star query functions
    (ngp-precision F2.1) — extracted here so it is defined once.

    Parameters
    ----------
    select_cols : str
        Comma-separated SELECT column list (table-aliased if joining).
    from_clause : str
        FROM clause: a table name, or a JOIN expression.
    where_template : str
        WHERE clause with `{lo}`/`{hi}` placeholders for the random_index
        window bounds, e.g. "parallax > 0 AND random_index >= {lo} AND
        random_index < {hi}".
    random_index_max : int
        Upper bound of the random_index range to scan.
    random_index_step : int
        Window size (rows). Must be small enough that each window's
        expected match count stays under MAXREC=2000 for the given
        selectivity — calibrate with a COUNT() query beforehand.
    output_columns : list of str
        Final column order/selection for the returned DataFrame.
    launch_fn : callable, optional
        `str -> pd.DataFrame` — executes one ADQL query and returns its
        result as a DataFrame. Defaults to a real
        `astroquery.gaia.Gaia.launch_job(...).get_results().to_pandas()`
        call. Injectable so this pagination logic is unit-testable
        OFFLINE (see tests/test_gaia_fetcher.py) without importing
        astroquery or hitting the network.

    Returns
    -------
    pd.DataFrame
    """
    if launch_fn is None:
        def launch_fn(query: str) -> pd.DataFrame:
            from astroquery.gaia import Gaia  # noqa: PLC0415
            return Gaia.launch_job(query).get_results().to_pandas()

    frames = []
    for lo in range(0, random_index_max, random_index_step):
        hi = min(lo + random_index_step, random_index_max)
        where = where_template.format(lo=lo, hi=hi)
        query = f"SELECT {select_cols} FROM {from_clause} WHERE {where}"
        chunk = launch_fn(query)
        if len(chunk) >= 2000:  # MAXREC hit — window truncated, would bias sample
            warnings.warn(
                f"random_index window [{lo}, {hi}) hit the 2000-row cap; "
                "reduce random_index_step to avoid truncation",
                UserWarning,
                stacklevel=2,
            )
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=output_columns)
    df = pd.concat(frames, ignore_index=True)
    return df[output_columns].copy()


def _default_gaia_query() -> pd.DataFrame:
    """Query Gaia DR3 TAP service via chunked synchronous ADQL. NETWORK-ONLY — not unit-tested."""
    return _chunked_sync_query(
        select_cols=_SELECT_COLS,
        from_clause="gaiadr3.gaia_source",
        where_template=(
            "parallax > 0 AND parallax_over_error > 5 "
            "AND ABS(b) < 15 AND phot_g_mean_mag < 15 "
            "AND random_index >= {lo} AND random_index < {hi}"
        ),
        random_index_max=_RANDOM_INDEX_MAX,
        random_index_step=_RANDOM_INDEX_STEP,
        output_columns=_GAIA_COLUMNS,
    )


def _fetch_with_cache(
    cache_path: str,
    force_refresh: bool,
    query_fn,
    default_query_fn,
    *,
    service_name: str = "archive",
) -> pd.DataFrame:
    """
    Shared cache-first + injectable-query_fn contract used by every
    fetch_* function project-wide (`fetch_gaia_stars` here and
    `tracer_fetcher.fetch_cepheids`/`fetch_ob_stars`/`fetch_young_clusters`,
    ngp-precision batch B2).

    Parameters
    ----------
    cache_path : str
        Path to the CSV cache file.
    force_refresh : bool
        If True, ignore the cache and re-query.
    query_fn : callable or None
        Zero-argument callable returning a pd.DataFrame. If None,
        `default_query_fn` is used. Inject a stub in tests.
    default_query_fn : callable
        Zero-argument callable used when `query_fn` is None (the real,
        NETWORK-only query).
    service_name : str
        Human-readable name used in the fallback warning / ConnectionError
        message (e.g. "Gaia archive", "VizieR (Cantat-Gaudin)").

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ConnectionError
        If the query fails and no cache is available.
    """
    if query_fn is None:
        query_fn = default_query_fn

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
                f"{service_name} unreachable — using cached data",
                UserWarning,
                stacklevel=3,
            )
            return pd.read_csv(cache_path)
        raise ConnectionError(
            f"{service_name} unreachable and no local cache found at '{cache_path}': {exc}"
        ) from exc

    # Write to cache (create parent directory if needed)
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(cache_path, index=False)

    return df


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
    return _fetch_with_cache(
        cache_path, force_refresh, query_fn, _default_gaia_query,
        service_name="Gaia archive",
    )


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
