"""Synthetic parity tests for ``iobrx.tme_cluster`` (Rust core vs the
ORIGINAL iobrpy tme_cluster functions).

No network, no official data; the whole file runs in seconds. The frozen
375x235 / 10x22 campaign gates (byte-identical output CSVs, sha256
8684ad4e… / 079ef917…) are validated by the porting harness recorded in the
module's provenance notes; here the ORIGINAL ``best_hartigan_run`` pipeline
(``backend='python'``) is the oracle on small synthetic frames.

Skipped while the native core / public wrapper wiring is pending
(``iobrx._rust.tme_kmeans_best`` missing or ``iobrx.tme_cluster`` not yet
exported); both appear once the Rust block is registered and the __init__
export lands.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pandas as pd
import pytest

import iobrx
from iobrx._backend import _native

_HAS_CORE = _native is not None and hasattr(_native, "tme_kmeans_best")
_HAS_API = hasattr(iobrx, "tme_cluster")

pytestmark = pytest.mark.skipif(
    not (_HAS_CORE and _HAS_API),
    reason="native tme_cluster core or public API wiring not built yet",
)


@pytest.fixture(scope="module")
def synth():
    """48 samples x 6 features with three planted well-separated blocks."""
    rng = np.random.default_rng(20260908)
    centers = np.array([[0.0, 0.0, 0.0, 0.0, 6.0, 6.0],
                        [5.0, 5.0, 5.0, 5.0, 0.0, 0.0],
                        [0.0, 6.0, 0.0, 6.0, 0.0, 6.0]])
    X = np.vstack([rng.normal(loc=c, scale=0.4, size=(16, 6)) for c in centers])
    df = pd.DataFrame(X, columns=[f"g{j}" for j in range(6)])
    df.insert(0, "sample", [f"S{i:03d}" for i in range(X.shape[0])])
    return df


def _run(synth, backend, **kw):
    return iobrx.tme_cluster(synth.copy(), id="sample", backend=backend, **kw)


def test_fast_matches_original_reference(synth):
    fast = _run(synth, "auto", min_nc=2, max_nc=5, n_threads=4)
    ref = _run(synth, "python", min_nc=2, max_nc=5)
    pd.testing.assert_frame_equal(fast, ref, check_exact=True)
    assert list(fast.columns[:2]) == ["ID", "cluster"]
    assert set(fast["cluster"]) <= {f"TME{i}" for i in range(1, 6)}


def test_deterministic_across_thread_counts(synth):
    ref = _run(synth, "auto", min_nc=2, max_nc=5, n_threads=1)
    for nt in (2, 8):
        pd.testing.assert_frame_equal(
            _run(synth, "auto", min_nc=2, max_nc=5, n_threads=nt), ref,
            check_exact=True)


def test_column_selection_and_no_scale(synth):
    for kw in (dict(features="1:4"), dict(pattern="g[0-2]"), dict(scale=False),
               dict(scale=False, features="2:6")):
        fast = _run(synth, "auto", min_nc=2, max_nc=4, n_threads=4, **kw)
        ref = _run(synth, "python", min_nc=2, max_nc=4, **kw)
        pd.testing.assert_frame_equal(fast, ref, check_exact=True, obj=str(kw))


def test_neighbor_seeds_k1_and_k_above_8(synth):
    # min_nc=2, max_nc=3 exercises the derived-seed neighbor runs k=1
    # (single cluster) and k=4; min_nc=3, max_nc=9 exercises k=10 clusters,
    # i.e. cluster ids >= 8 where CPython set iteration order in the
    # empty-cluster repair is NOT ascending (reproduced by the Rust core).
    for kw in (dict(min_nc=2, max_nc=3), dict(min_nc=3, max_nc=9)):
        fast = _run(synth, "auto", n_threads=4, **kw)
        ref = _run(synth, "python", **kw)
        pd.testing.assert_frame_equal(fast, ref, check_exact=True, obj=str(kw))


def test_nondefault_seed_nstart_tol(synth):
    for kw in (dict(seed=7), dict(nstart=1), dict(nstart=3, seed=99),
               dict(tol=1e-2), dict(max_iter=3)):
        fast = _run(synth, "auto", min_nc=2, max_nc=4, n_threads=4, **kw)
        ref = _run(synth, "python", min_nc=2, max_nc=4, **kw)
        pd.testing.assert_frame_equal(fast, ref, check_exact=True, obj=str(kw))


def test_csv_bytes_match(synth, tmp_path):
    a = tmp_path / "fast.csv"
    b = tmp_path / "ref.csv"
    _run(synth, "auto", min_nc=2, max_nc=5, n_threads=4).to_csv(a, sep=",", index=False)
    _run(synth, "python", min_nc=2, max_nc=5).to_csv(b, sep=",", index=False)
    assert a.read_bytes() == b.read_bytes()


def test_missing_order_matches_cpython():
    """The Rust replica of CPython's `set(range(k)) - set(present)` slot
    order (the empty-cluster repair iteration order) must equal the live
    interpreter's, including the non-ascending cases with ids >= 8."""
    rng = random.Random(5)
    # adversarial explicit cases: missing ids spanning the 8-slot boundary
    cases = [(10, [0, 1, 2, 4, 5, 6, 7, 8]),        # missing {3, 9} -> [9, 3]
             (12, list(range(12))[:3] + [4, 5, 6, 7, 8, 10, 11]),
             (9, [0, 1, 2, 3, 4, 5, 6, 7]),         # missing {8}
             (16, [0, 1, 2, 3, 5, 7, 11, 13])]
    # exhaustive small k
    for k in range(2, 10):
        for r in range(0, k + 1):
            for missing in itertools.combinations(range(k), r):
                cases.append((k, sorted(set(range(k)) - set(missing))))
    # randomized larger k
    for k in range(10, 31):
        for _ in range(60):
            present = sorted(rng.sample(range(k), rng.randint(0, k)))
            cases.append((k, present))
    for k, present in cases:
        expected = list(set(range(k)) - set(present))
        got = list(_native.tme_missing_order(k, present))
        assert got == expected, (k, present, expected, got)


def test_kl_index_matches_python_expression():
    rng = np.random.default_rng(3)
    for p in (6, 22, 235):
        for k in range(2, 9):
            w = rng.random(3) * 1e4 + 1.0
            e = 2 / p
            num = abs((k - 1) ** e * w[0] - k ** e * w[1])
            den = abs(k ** e * w[1] - (k + 1) ** e * w[2])
            expected = num / (den if den > 0 else 1e-8)
            got = _native.tme_kl_index(k, p, float(w[0]), float(w[1]), float(w[2]))
            assert float(got) == float(expected), (k, p, got, expected)


def test_k_larger_than_n_raises_like_original(synth):
    msg = "Cannot take a larger sample than population"
    with pytest.raises(ValueError, match=msg):
        _run(synth, "auto", min_nc=60, max_nc=60, n_threads=2)
    with pytest.raises(ValueError, match=msg):
        _run(synth, "python", min_nc=60, max_nc=60)


def test_all_features_filtered_raises(synth):
    df = synth[["sample"]].copy()
    df["c1"] = 1.0
    df["c2"] = 2.0
    msg = "No valid features remain after filtering"
    with pytest.raises(ValueError, match=msg):
        iobrx.tme_cluster(df.copy(), id="sample", backend="auto", n_threads=2)
    with pytest.raises(ValueError, match=msg):
        iobrx.tme_cluster(df.copy(), id="sample", backend="python")
