"""Parity gates for iobrx.nmf against the ORIGINAL iobrpy nmf CLI.

Upstream ``iobrpy.workflow.nmf`` is CLI-only (``iobrpy nmf`` / ``python -m
iobrpy.main nmf``), so the golden side here is the original CLI executed in a
subprocess on a small synthetic samples x features matrix; the iobrx side is
the API. Contract (per the port task): ``clusters.csv``,
``top_features_per_cluster.csv`` and ``pca_plot.png`` must be BYTE-identical
(sha256), including the upstream quirks reproduced on purpose:

  * top_features_per_cluster.csv is written before the output directory is
    created and the failure is swallowed -> with a non-existent ``outdir``
    the file is silently missing (bug-compat, pinned below);
  * ``random_state`` does not change the discrete outputs (nndsva init +
    shuffle=False; floats may differ ~1e-15 exactly as in the original);
  * ``kmax >= n_samples`` is clamped to ``n_samples - 1``.

Fast lane (default ``pytest -q``): synthetic 36 x 8 input, a handful of
original-CLI subprocess calls, well under 60 s. The frozen official input
(cibersort_stad10.csv x the docs command) is gated behind the ``full`` mark
and ``$IOBRX_NMF_FROZEN_INPUT``.

Until the strategy layer wires ``iobrx.nmf`` into ``src/iobrx/__init__.py``
(see research/build_nmf/INTEGRATION.md), the helper below falls back to the
identical-signature core ``iobrx._fast.nmf_fast.nmf_cluster``.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

CONTRACT_FILES = ("clusters.csv", "top_features_per_cluster.csv", "pca_plot.png")


def _iobrx_nmf(*args, **kwargs):
    """Public API if wired, otherwise the fast core (same signature)."""
    import iobrx

    fn = getattr(iobrx, "nmf", None)
    if fn is None:  # pre-integration fallback
        from iobrx._fast.nmf_fast import nmf_cluster as fn
    return fn(*args, **kwargs)


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _run_original_cli(csv_path, outdir, extra=()):
    """The golden side: original CLI in a fresh subprocess (as main.py does)."""
    os.makedirs(outdir, exist_ok=True)  # pre-created: avoid the silent-loss bug
    env = dict(os.environ, MPLBACKEND="Agg")
    cmd = [sys.executable, "-m", "iobrpy.main", "nmf",
           "-i", str(csv_path), "-o", str(outdir), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Deterministic 36 x 8 count-like matrix with 3 sample blocks."""
    rng = np.random.default_rng(20260908)
    n, p = 36, 8
    base = rng.poisson(5, size=(n, p)).astype(float)
    base[:12, 0:3] += 40          # block 1 signal
    base[12:24, 3:6] += 40        # block 2 signal
    base[24:, 6:8] += 40          # block 3 signal
    df = pd.DataFrame(
        base,
        index=[f"S{i:02d}" for i in range(n)],
        columns=[f"celltype_{j}" for j in range(p)],
    )
    d = tmp_path_factory.mktemp("nmf_parity")
    csv_path = d / "matrix.csv"
    df.to_csv(csv_path)
    return df, csv_path, d


def test_nmf_output_files_byte_identical(synthetic):
    """clusters.csv + top_features + pca_plot.png: sha256 equal to the CLI."""
    df, csv_path, d = synthetic
    orig_dir = d / "orig"
    port_dir = d / "ported"
    os.makedirs(port_dir, exist_ok=True)
    params = ["--kmin", "2", "--kmax", "4", "--log1p", "--normalize",
              "--max-iter", "500", "--random-state", "42"]
    _run_original_cli(csv_path, orig_dir, params)
    res = _iobrx_nmf(df.copy(), kmin=2, kmax=4, log1p=True, normalize=True,
                     max_iter=500, random_state=42, plot=True, outdir=str(port_dir))
    for name in CONTRACT_FILES:
        assert _sha(orig_dir / name) == _sha(port_dir / name), name
    # returned frames equal the written files, byte for byte
    assert res["clusters"].to_csv(index=False).encode() == (port_dir / "clusters.csv").read_bytes()
    assert res["top_features"].to_csv().encode() == (port_dir / "top_features_per_cluster.csv").read_bytes()
    assert 2 <= res["best_k"] <= 4


def test_nmf_features_selection_matches_cli(synthetic):
    """--features '2-6' column window: identical files via the API."""
    df, csv_path, d = synthetic
    orig_dir = d / "orig_feat"
    port_dir = d / "ported_feat"
    os.makedirs(port_dir, exist_ok=True)
    params = ["--kmin", "2", "--kmax", "3", "--features", "2-6", "--max-iter", "300"]
    _run_original_cli(csv_path, orig_dir, params)
    _iobrx_nmf(csv_path, features="2-6", kmin=2, kmax=3, max_iter=300,
               plot=True, outdir=str(port_dir))  # path input: original read_matrix
    for name in CONTRACT_FILES:
        assert _sha(orig_dir / name) == _sha(port_dir / name), name
    # 'm-n', 'm:n' and single 'm' parse identically (1-based, inclusive)
    a = _iobrx_nmf(df.copy(), features="2-6", kmin=2, kmax=2, max_iter=100)
    b = _iobrx_nmf(df.copy(), features="2:6", kmin=2, kmax=2, max_iter=100)
    assert a["clusters"].equals(b["clusters"])
    assert list(a["used_features"].columns) == [f"celltype_{j}" for j in range(1, 6)]
    for bad in ("0-2", "5-1", "abc", "99", "1-99-3"):
        with pytest.raises(ValueError):
            _iobrx_nmf(df.copy(), features=bad, kmin=2, kmax=2, max_iter=50)


def test_nmf_shift_and_negative_guard(synthetic):
    """Negative data: ValueError without shift (upstream message), works with."""
    df, _, d = synthetic
    neg = df.copy() - 100.0
    with pytest.raises(ValueError, match="NMF requires non-negative"):
        _iobrx_nmf(neg, kmin=2, kmax=2, max_iter=50)
    res = _iobrx_nmf(neg, kmin=2, kmax=2, max_iter=50, shift=0.5)
    assert res["used_features"].to_numpy().min() >= 0.0
    # non-negative data ignores shift entirely (upstream behavior)
    plain = _iobrx_nmf(df.copy(), kmin=2, kmax=2, max_iter=50)
    shifted = _iobrx_nmf(df.copy(), kmin=2, kmax=2, max_iter=50, shift=7.0)
    assert plain["clusters"].equals(shifted["clusters"])


def test_nmf_seed_invariance_and_kmax_clamp(synthetic):
    """random_state does not move labels/clusters; kmax clamps to n-1."""
    df, _, _ = synthetic
    a = _iobrx_nmf(df.copy(), kmin=2, kmax=4, max_iter=300, random_state=42)
    b = _iobrx_nmf(df.copy(), kmin=2, kmax=4, max_iter=300, random_state=99)
    assert np.array_equal(a["labels"], b["labels"])
    assert a["clusters"].equals(b["clusters"])
    tiny = df.iloc[:5]
    res = _iobrx_nmf(tiny.copy(), kmin=2, kmax=9, max_iter=200)
    assert max(res["k_results"]) <= 4  # kmax clamped to n_samples-1, k>=n skipped


def test_nmf_all_invalid_k_raises(synthetic):
    """Degenerate input: every k collapses to one argmax cluster -> RuntimeError."""
    df, _, _ = synthetic
    zeros = pd.DataFrame(0.0, index=df.index, columns=df.columns)
    with pytest.raises(RuntimeError, match="No successful results"):
        _iobrx_nmf(zeros, kmin=2, kmax=3, max_iter=50)


def test_nmf_outdir_missing_topfeatures_bugcompat(synthetic):
    """BUG-COMPAT pin: non-existent outdir silently loses top_features only."""
    df, _, d = synthetic
    out = d / "nodir_bug"  # deliberately NOT created
    res = _iobrx_nmf(df.copy(), kmin=2, kmax=3, max_iter=200, plot=True, outdir=str(out))
    assert os.path.isdir(out)
    assert not (out / "top_features_per_cluster.csv").exists()
    assert (out / "clusters.csv").exists() and (out / "pca_plot.png").exists()
    # the in-memory frame is still complete (the API route loses nothing)
    assert res["top_features"].shape[0] == res["best_k"]


def test_nmf_plot_false_skips_png_and_matplotlib(synthetic):
    """plot=False: no PNG in outdir, and matplotlib is never imported."""
    df, _, d = synthetic
    out = d / "noplot"
    os.makedirs(out, exist_ok=True)
    _iobrx_nmf(df.copy(), kmin=2, kmax=3, max_iter=200, plot=False, outdir=str(out))
    assert sorted(os.listdir(out)) == ["clusters.csv", "top_features_per_cluster.csv"]
    with pytest.raises(ValueError):
        _iobrx_nmf(df.copy(), kmin=2, kmax=2, max_iter=50, plot=True)  # needs outdir
    # fresh-process check: the API path must not pull matplotlib in
    code = "\n".join([
        "import sys",
        "import iobrx",
        "fn = getattr(iobrx, 'nmf', None)",
        "if fn is None:",
        "    from iobrx._fast.nmf_fast import nmf_cluster as fn",
        f"fn({str(synthetic[1])!r}, kmin=2, kmax=3, max_iter=200, plot=False)",
        "assert 'matplotlib' not in sys.modules, 'matplotlib leaked into the API path'",
        "print('LAZY-OK')",
    ])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, MPLBACKEND="Agg"))
    assert r.returncode == 0 and "LAZY-OK" in r.stdout, r.stdout + r.stderr


def test_nmf_backend_validation(synthetic):
    """backend='rust' fails clearly (no native kernel); bad values rejected."""
    df, _, _ = synthetic
    with pytest.raises(RuntimeError, match="pure-Python"):
        _iobrx_nmf(df.copy(), kmin=2, kmax=2, max_iter=50, backend="rust")
    with pytest.raises(ValueError, match="backend must be"):
        _iobrx_nmf(df.copy(), kmin=2, kmax=2, max_iter=50, backend="cuda")
    ok = _iobrx_nmf(df.copy(), kmin=2, kmax=2, max_iter=50, backend="python")
    assert ok["best_k"] == 2


@pytest.mark.full
def test_nmf_frozen_official_gate():
    """Official frozen gate: cibersort_stad10.csv x the docs command.

    Needs ``IOBRX_NMF_FROZEN_INPUT`` (the campaign's frozen CSV); skipped
    when unset. Golden = the original CLI run here, not stored hashes, so
    the gate is library-version independent.
    """
    frozen = os.environ.get("IOBRX_NMF_FROZEN_INPUT")
    if not frozen or not os.path.exists(frozen):
        pytest.skip("set IOBRX_NMF_FROZEN_INPUT to the frozen cibersort CSV")
    import tempfile

    params = ["--kmin", "2", "--kmax", "10", "--features", "1:22",
              "--max-iter", "10000", "--skip_k_2", "--random-state", "42"]
    with tempfile.TemporaryDirectory() as td:
        orig_dir = os.path.join(td, "orig")
        port_dir = os.path.join(td, "ported")
        os.makedirs(port_dir, exist_ok=True)
        _run_original_cli(frozen, orig_dir, params)
        res = _iobrx_nmf(frozen, kmin=2, kmax=10, features="1:22",
                         max_iter=10000, skip_k_2=True, random_state=42,
                         plot=True, outdir=port_dir)
        for name in CONTRACT_FILES:
            assert _sha(os.path.join(orig_dir, name)) == _sha(os.path.join(port_dir, name)), name
        assert res["best_k"] == 3  # campaign-recorded selection on this input
