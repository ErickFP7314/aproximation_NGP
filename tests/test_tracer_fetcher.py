"""
Tests for tracer_fetcher.py — F2.1 young Galactic-disk tracer catalogs
(ngp-precision batch B2). Cepheids are the priority tracer (Period-
Luminosity distances, independent of parallax zero-point); OB stars and
young open clusters are secondary. All fast/unit tests are OFFLINE via an
injected `query_fn` stub — no real network call. Real downloads
(B2.13/B2.14) are NETWORK+SLOW and deferred — see the bottom of this file.
"""

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from tracer_fetcher import (
    fetch_cepheids,
    fetch_ob_stars,
    fetch_young_clusters,
    apply_galactocentric_cut,
    cepheid_pl_distance,
)
from ngp_3d import great_circle_pole


# ---------------------------------------------------------------------------
# Stub builders (column contract per design §5 / spec F2.1-R2)
# ---------------------------------------------------------------------------

_CEPHEID_REQUIRED_COLS = [
    "source_id", "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmdec", "phot_g_mean_mag", "l", "b",
    "pf", "type_best_classification",
]

_OB_REQUIRED_COLS = [
    "source_id", "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmdec", "phot_g_mean_mag", "bp_rp", "l", "b",
]

_CLUSTER_REQUIRED_COLS = [
    "cluster_id", "ra", "dec", "parallax",
    "pmra", "pmdec", "n_members", "l", "b", "age_myr",
]


def _make_cepheid_stub(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "source_id": np.arange(n),
        "ra": np.full(n, 190.0),
        "dec": np.full(n, 25.0),
        "parallax": np.full(n, 0.5),
        "parallax_error": np.full(n, 0.02),
        "pmra": np.full(n, 1.0),
        "pmdec": np.full(n, -1.0),
        "phot_g_mean_mag": np.full(n, 11.0),
        "l": np.full(n, 90.0),
        "b": np.full(n, 2.0),
        "pf": np.full(n, 10.0),
        "type_best_classification": ["DCEP_F"] * n,
    })


def _make_ob_stub(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "source_id": np.arange(n),
        "ra": np.full(n, 100.0),
        "dec": np.full(n, 10.0),
        "parallax": np.full(n, 1.0),
        "parallax_error": np.full(n, 0.05),
        "pmra": np.full(n, 2.0),
        "pmdec": np.full(n, -2.0),
        "phot_g_mean_mag": np.full(n, 9.0),
        "bp_rp": np.full(n, -0.1),
        "l": np.full(n, 45.0),
        "b": np.full(n, 1.0),
    })


def _make_cluster_stub(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "cluster_id": [f"C{i}" for i in range(n)],
        "ra": np.full(n, 80.0),
        "dec": np.full(n, 5.0),
        "parallax": np.full(n, 0.8),
        "pmra": np.full(n, 0.5),
        "pmdec": np.full(n, -0.5),
        "n_members": np.full(n, 150),
        "l": np.full(n, 200.0),
        "b": np.full(n, -1.0),
        "age_myr": np.full(n, 50.0),
    })


def _raising_stub():
    raise ConnectionError("archive unreachable (stub)")


# ---------------------------------------------------------------------------
# B2.1 — cache hit -> injected query_fn never invoked. Spec F2.1-R1-S1.
# ---------------------------------------------------------------------------

def test_fetch_cepheids_cache_hit_no_network_call(tmp_path):
    cache_file = tmp_path / "gaia_cepheids.csv"
    _make_cepheid_stub().to_csv(cache_file, index=False)

    def raising_stub():
        raise AssertionError("query_fn was called despite cache hit")

    result = fetch_cepheids(
        cache_path=str(cache_file), force_refresh=False, query_fn=raising_stub,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# B2.2 — cache miss -> stub invoked, results cached to CSV.
# Spec F2.1-R1-S2.
# ---------------------------------------------------------------------------

def test_fetch_cepheids_cache_miss_calls_stub_and_writes_csv(tmp_path):
    cache_file = tmp_path / "gaia_cepheids.csv"

    result = fetch_cepheids(
        cache_path=str(cache_file), force_refresh=False, query_fn=_make_cepheid_stub,
    )

    assert cache_file.exists(), "CSV was not written"
    saved = pd.read_csv(cache_file)
    for col in _CEPHEID_REQUIRED_COLS:
        assert col in saved.columns, f"Missing column: {col}"
    assert len(result) == 5


def test_fetch_cepheids_force_refresh_archive_down_falls_back_to_cache(tmp_path):
    cache_file = tmp_path / "gaia_cepheids.csv"
    _make_cepheid_stub().to_csv(cache_file, index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fetch_cepheids(
            cache_path=str(cache_file), force_refresh=True, query_fn=_raising_stub,
        )

    assert len(result) == 5
    messages = [str(w.message) for w in caught]
    assert any("unreachable" in m for m in messages)


def test_fetch_cepheids_archive_down_no_cache_raises_connection_error(tmp_path):
    cache_file = tmp_path / "gaia_cepheids.csv"
    with pytest.raises(ConnectionError):
        fetch_cepheids(cache_path=str(cache_file), force_refresh=False, query_fn=_raising_stub)


# ---------------------------------------------------------------------------
# B2.4 — Cepheid column contract. Spec F2.1-R2-S1.
# ---------------------------------------------------------------------------

def test_fetch_cepheids_column_contract(tmp_path):
    cache_file = tmp_path / "gaia_cepheids.csv"
    result = fetch_cepheids(
        cache_path=str(cache_file), force_refresh=False, query_fn=_make_cepheid_stub,
    )
    for col in ("source_id", "ra", "dec", "parallax", "pf"):
        assert col in result.columns


# ---------------------------------------------------------------------------
# B2.5 — R<9kpc galactocentric cut removes rows beyond R, count matches
# expectation. Spec F2.1-R3-S1.
# ---------------------------------------------------------------------------

def test_apply_galactocentric_cut_removes_distant_rows():
    # Row 0: l=0, b=0, d=1kpc -> R = |R_sun - d| = 7.122 kpc (inside 9kpc)
    # Row 1: l=0, b=0, d=20kpc -> R = |20 - 8.122| = 11.878 kpc (outside 9kpc)
    # Row 2: l=180, b=0, d=1kpc -> R = R_sun + d = 9.122 kpc (outside 9kpc,
    #        since 9.122 > 9.0)
    df = pd.DataFrame({
        "l": [0.0, 0.0, 180.0],
        "b": [0.0, 0.0, 0.0],
        "distance_kpc": [1.0, 20.0, 1.0],
    })
    result = apply_galactocentric_cut(df, r_max_kpc=9.0, r_sun_kpc=8.122)
    assert len(result) == 1
    assert result["r_galactocentric_kpc"].iloc[0] == pytest.approx(7.122, abs=1e-6)


def test_apply_galactocentric_cut_requires_distance_column():
    df = pd.DataFrame({"l": [0.0], "b": [0.0]})
    with pytest.raises(ValueError):
        apply_galactocentric_cut(df)


def test_apply_galactocentric_cut_custom_distance_col():
    df = pd.DataFrame({
        "l": [0.0], "b": [0.0], "pl_distance_kpc": [1.0],
    })
    result = apply_galactocentric_cut(df, distance_col="pl_distance_kpc")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# B2.6 — cepheid_pl_distance: injectable P-L relation.
# ---------------------------------------------------------------------------

def test_cepheid_pl_distance_matches_closed_form_with_custom_relation():
    df = pd.DataFrame({"pf": [10.0], "phot_g_mean_mag": [12.0]})

    # Custom, trivially-checkable P-L relation: M_G = -2.0 always.
    def constant_relation(log_p):
        return np.full_like(log_p, -2.0)

    result = cepheid_pl_distance(df, pl_relation=constant_relation)

    # mu = m - M = 12.0 - (-2.0) = 14.0 -> d_pc = 10**((14+5)/5) = 10**3.8
    expected_d_kpc = (10.0 ** ((12.0 - (-2.0) + 5.0) / 5.0)) / 1000.0
    assert result.iloc[0] == pytest.approx(expected_d_kpc, rel=1e-9)


def test_cepheid_pl_distance_default_relation_returns_positive_finite_distance():
    df = pd.DataFrame({"pf": [10.0, 20.0], "phot_g_mean_mag": [11.0, 10.5]})
    result = cepheid_pl_distance(df)
    assert (result > 0).all()
    assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# B2.7 — Cepheid CSV passed unmodified into great_circle_pole, returns a
# pole without raising (no estimator API modification needed: it only
# requires 'ra'/'dec', which every tracer schema already provides).
# Spec F2.1-R4-S1.
# ---------------------------------------------------------------------------

def test_cepheid_dataframe_runs_through_great_circle_pole_unmodified(tmp_path):
    cache_file = tmp_path / "gaia_cepheids.csv"
    df = fetch_cepheids(
        cache_path=str(cache_file), force_refresh=False,
        query_fn=lambda: _make_cepheid_stub(n=50),
    )
    result = great_circle_pole(df)
    assert "alpha_NGP" in result
    assert "delta_NGP" in result


# ---------------------------------------------------------------------------
# B2.9 — [SEC] OB column contract. Spec F2.1-R2-S1(OB).
# ---------------------------------------------------------------------------

def test_fetch_ob_stars_cache_miss_and_column_contract(tmp_path):
    cache_file = tmp_path / "gaia_ob_stars.csv"
    result = fetch_ob_stars(
        cache_path=str(cache_file), force_refresh=False, query_fn=_make_ob_stub,
    )
    assert cache_file.exists()
    for col in _OB_REQUIRED_COLS:
        assert col in result.columns


def test_fetch_ob_stars_cache_hit_no_network_call(tmp_path):
    cache_file = tmp_path / "gaia_ob_stars.csv"
    _make_ob_stub().to_csv(cache_file, index=False)

    def raising_stub():
        raise AssertionError("query_fn was called despite cache hit")

    result = fetch_ob_stars(cache_path=str(cache_file), query_fn=raising_stub)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# B2.11 — [SEC] Cluster column contract. Spec F2.1-R2-S1(clusters).
# ---------------------------------------------------------------------------

def test_fetch_young_clusters_cache_miss_and_column_contract(tmp_path):
    cache_file = tmp_path / "cantat_gaudin_clusters.csv"
    result = fetch_young_clusters(
        cache_path=str(cache_file), force_refresh=False, query_fn=_make_cluster_stub,
    )
    assert cache_file.exists()
    for col in _CLUSTER_REQUIRED_COLS:
        assert col in result.columns


def test_fetch_young_clusters_cache_hit_no_network_call(tmp_path):
    cache_file = tmp_path / "cantat_gaudin_clusters.csv"
    _make_cluster_stub().to_csv(cache_file, index=False)

    def raising_stub():
        raise AssertionError("query_fn was called despite cache hit")

    result = fetch_young_clusters(cache_path=str(cache_file), query_fn=raising_stub)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# B2.13 / B2.14 — [NETWORK][SLOW] real downloads — DEFERRED.
# Not executed in this batch. Run manually with RUN_NETWORK_TESTS=1 once
# network access + a valid TAP/VizieR session are available, to populate
# data/gaia_cepheids.csv (~3,400 rows expected post R<9kpc cut, per
# design §4/proposal), data/gaia_ob_stars.csv, and
# data/cantat_gaudin_clusters.csv.
# ---------------------------------------------------------------------------

_RUN_NETWORK = bool(os.environ.get("RUN_NETWORK_TESTS"))


@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_NETWORK,
    reason=(
        "B2.13 [NETWORK][SLOW] deferred: real Gaia DR3 "
        "gaiadr3.vari_cepheid JOIN gaia_source chunked-sync download. "
        "Run with RUN_NETWORK_TESTS=1 to populate data/gaia_cepheids.csv."
    ),
)
def test_real_cepheid_download_populates_cache():
    df = fetch_cepheids(force_refresh=True)
    assert len(df) > 0
    for col in _CEPHEID_REQUIRED_COLS:
        assert col in df.columns


@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_NETWORK,
    reason=(
        "B2.14 [NETWORK][SLOW][SEC] deferred: real OB-star Gaia query + "
        "Cantat-Gaudin (2020) VizieR cluster download. "
        "Run with RUN_NETWORK_TESTS=1 to populate data/gaia_ob_stars.csv "
        "and data/cantat_gaudin_clusters.csv."
    ),
)
def test_real_ob_and_cluster_downloads_populate_cache():
    ob = fetch_ob_stars(force_refresh=True)
    clusters = fetch_young_clusters(force_refresh=True)
    assert len(ob) > 0
    assert len(clusters) > 0
