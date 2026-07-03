"""
Tests for gaia_fetcher.py — all network calls are replaced by an injectable query_fn stub.
"""

import warnings
import pytest
import pandas as pd

from gaia_fetcher import fetch_gaia_stars, filter_valid_parallax, _chunked_sync_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLS = [
    "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmdec", "phot_g_mean_mag", "l", "b",
]


def _make_stub_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid DataFrame that looks like a Gaia result."""
    return pd.DataFrame({
        "ra": [10.0] * n,
        "dec": [20.0] * n,
        "parallax": [3.0] * n,
        "parallax_error": [0.05] * n,
        "pmra": [1.0] * n,
        "pmdec": [-1.0] * n,
        "phot_g_mean_mag": [12.0] * n,
        "l": [100.0] * n,
        "b": [5.0] * n,
    })


def _raising_stub():
    raise ConnectionError("Gaia archive unreachable (stub)")


# ---------------------------------------------------------------------------
# Test 1: cache hit — stub must NOT be called
# ---------------------------------------------------------------------------

def test_cache_hit_no_network_call(tmp_path):
    """S2.2 — when cache exists the query_fn must never be called."""
    cache_file = tmp_path / "gaia_disk_stars.csv"
    # Write a valid cache
    _make_stub_df().to_csv(cache_file, index=False)

    def raising_stub():
        raise AssertionError("query_fn was called despite cache hit")

    result = fetch_gaia_stars(
        cache_path=str(cache_file),
        force_refresh=False,
        query_fn=raising_stub,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Test 2: cache miss — stub returns data; CSV written with required columns
# ---------------------------------------------------------------------------

def test_cache_miss_calls_stub_and_writes_csv(tmp_path):
    """S2.1 — on first call the stub is invoked and CSV is written."""
    cache_file = tmp_path / "gaia_disk_stars.csv"

    result = fetch_gaia_stars(
        cache_path=str(cache_file),
        force_refresh=False,
        query_fn=_make_stub_df,
    )

    assert cache_file.exists(), "CSV was not written"
    saved = pd.read_csv(cache_file)
    for col in _REQUIRED_COLS:
        assert col in saved.columns, f"Missing column: {col}"
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Test 3: force_refresh=True + archive down + cache present → warning + cached data
# ---------------------------------------------------------------------------

def test_force_refresh_archive_down_returns_cache_with_warning(tmp_path):
    """S2.3 — archive error during force_refresh falls back to cache with UserWarning."""
    cache_file = tmp_path / "gaia_disk_stars.csv"
    _make_stub_df().to_csv(cache_file, index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fetch_gaia_stars(
            cache_path=str(cache_file),
            force_refresh=True,
            query_fn=_raising_stub,
        )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5

    warning_messages = [str(w.message) for w in caught]
    assert any("Gaia archive unreachable" in m for m in warning_messages), (
        f"Expected 'Gaia archive unreachable' warning, got: {warning_messages}"
    )


# ---------------------------------------------------------------------------
# Test 4: archive down + no cache → ConnectionError
# ---------------------------------------------------------------------------

def test_archive_down_no_cache_raises_connection_error(tmp_path):
    """S2.4 — no cache + network failure → ConnectionError."""
    cache_file = tmp_path / "gaia_disk_stars.csv"

    with pytest.raises(ConnectionError):
        fetch_gaia_stars(
            cache_path=str(cache_file),
            force_refresh=False,
            query_fn=_raising_stub,
        )


# ---------------------------------------------------------------------------
# Test 5: filter_valid_parallax — keeps only parallax > 0
# ---------------------------------------------------------------------------

def test_filter_valid_parallax_removes_non_positive():
    """S2.5 — only parallax > 0 rows are kept."""
    df = pd.DataFrame({
        "ra": [1.0, 2.0, 3.0],
        "dec": [0.0, 0.0, 0.0],
        "parallax": [0.0, -0.5, 2.0],
    })
    result = filter_valid_parallax(df)
    assert len(result) == 1
    assert float(result["parallax"].iloc[0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 6: _chunked_sync_query — pagination logic, fully offline via
# injected launch_fn (ngp-precision B2: generalized out of
# _default_gaia_query for reuse by tracer_fetcher.py). Spec F2.1-R1-S2
# (chunked-sync query mechanism).
# ---------------------------------------------------------------------------

def test_chunked_sync_query_paginates_random_index_windows():
    """The WHERE clause seen by launch_fn must carry the correct
    non-overlapping, contiguous [lo, hi) random_index windows, and results
    from every window must be concatenated (no truncation, no gaps)."""
    seen_queries = []

    def fake_launch(query: str) -> pd.DataFrame:
        seen_queries.append(query)
        # One row per window so we can count windows via len(result).
        return pd.DataFrame({"a": [1], "b": [2]})

    result = _chunked_sync_query(
        select_cols="a, b",
        from_clause="some.table",
        where_template="random_index >= {lo} AND random_index < {hi}",
        random_index_max=250,
        random_index_step=100,
        output_columns=["a", "b"],
        launch_fn=fake_launch,
    )

    # 250 / 100 -> windows [0,100), [100,200), [200,250) = 3 windows
    assert len(seen_queries) == 3
    assert "random_index >= 0 AND random_index < 100" in seen_queries[0]
    assert "random_index >= 100 AND random_index < 200" in seen_queries[1]
    assert "random_index >= 200 AND random_index < 250" in seen_queries[2]
    assert len(result) == 3
    assert list(result.columns) == ["a", "b"]


def test_chunked_sync_query_warns_when_window_hits_maxrec_cap():
    """A window returning >= 2000 rows silently truncated by MAXREC must
    raise a UserWarning (bias-detection, not a silent wrong answer)."""
    def fake_launch(query: str) -> pd.DataFrame:
        return pd.DataFrame({"a": range(2000)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _chunked_sync_query(
            select_cols="a",
            from_clause="some.table",
            where_template="random_index >= {lo} AND random_index < {hi}",
            random_index_max=100,
            random_index_step=100,
            output_columns=["a"],
            launch_fn=fake_launch,
        )

    messages = [str(w.message) for w in caught]
    assert any("2000-row cap" in m for m in messages)


def test_chunked_sync_query_empty_range_returns_empty_dataframe_with_columns():
    """random_index_max <= 0 (or step >= max, degenerate range) must not
    crash — returns an empty DataFrame with the requested output columns."""
    def unreachable_launch(query: str) -> pd.DataFrame:
        raise AssertionError("launch_fn should never be called for an empty range")

    result = _chunked_sync_query(
        select_cols="a, b",
        from_clause="some.table",
        where_template="random_index >= {lo} AND random_index < {hi}",
        random_index_max=0,
        random_index_step=100,
        output_columns=["a", "b"],
        launch_fn=unreachable_launch,
    )
    assert len(result) == 0
    assert list(result.columns) == ["a", "b"]
