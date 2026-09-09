"""merge_star_count parity gates (synthetic in-test fixture; default suite).

Compares the accelerated port ``iobrx._fast.merge_star_count_fast.merge_star_count``
against the ORIGINAL ``iobrpy.workflow.merge_star_count`` CLI on STAR
``*_ReadsPerGene.out.tab`` files generated below (200 genes x 3 samples,
4 leading bare-integer stat rows + 4 integer columns — the layout of the
frozen 8 x 60,004-row baseline fixture).

Contract under test (see research/merge/module_spec.md §2 and the port's
INTEGRATION.md):

* happy path — decompressed ``<project>.STAR.count.tsv.gz`` is TOKEN-for-
  TOKEN identical to the original after column alignment (row-label
  alignment is needed because the ORIGINAL anchors the union row order to
  whichever sample completes first, while the port anchors it to the
  sorted-first sample); the port's columns are deterministically sorted by
  sample name (the ORIGINAL's ``os.listdir`` + ``as_completed`` orders are
  nondeterministic by design); repeated runs are byte-stable; the pandas
  fallback is NOT used; discovery is single-level (nested decoys ignored).
* BUG-COMPAT gate — the four leading STAR stat rows are NOT skipped: they
  pollute the matrix as ``4 x n_samples`` NaN-padded rows with the source
  integers, exactly as upstream (the port must NOT fix this defect).
* same-stat values across samples (identical label sets) — no NaN is
  introduced, pandas keeps int64 columns and the file carries integer
  tokens (no ".0"); the port reproduces the dtype per column, still on the
  fast path, byte-identical after column alignment (row order is
  anchor-invariant when all label sets are equal).
* 2-column files (stride-1 fast branch), heterogeneous gene rows across
  samples (union grows; fast path, sort-normalized frame equality) and
  guard-tripping inputs (duplicate ID labels → the same ``InvalidIndexError``;
  pandas NA-token labels → fallback with the same NaN index entry).
* empty dir — notice printed, nothing written, ``None`` returned.

Run with the repo source on the path::

    PYTHONPATH=<repo>/src pytest -q tests/test_parity_merge_star_count.py
"""
from __future__ import annotations

import gzip
import io
import os
import sys

import pandas as pd
import pytest


def _mscf():
    return pytest.importorskip(
        "iobrx._fast.merge_star_count_fast",
        reason="needs PYTHONPATH=<repo>/src (or an installed iobrx)",
    )


def _orig_mod():
    return pytest.importorskip("iobrpy.workflow.merge_star_count")


# ---------------------------------------------------------------------------
# fixture generation (self-contained, seeded)
# ---------------------------------------------------------------------------
_SAMPLES = ("ALPHA", "MID", "ZEBRA")  # sorted order != creation order


def _gen_star_dir(root, mode="happy", n_genes=200, seed=20260908):
    """Write 3 flat STAR count files + decoys into ``root``.

    Layout per file: 4 bare-integer stat rows (same value in all 4 columns)
    + ``n_genes`` versioned-ENSG rows x 4 integer columns. Decoys: a nested
    dir holding a suffix-matching file (upstream discovery is NON-recursive
    and must ignore it), ``README.txt`` and ``ALPHA_quant.sf``.

    modes: happy | samestats (identical stat values in all samples -> equal
    label sets -> int64 columns) | twocol (2-column files) | het (ZEBRA
    genes reordered/trimmed/extended) | dup (one gene row duplicated in
    MID) | na (a 'NULL' gene id in ALPHA -> pandas NaN label).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    genes = [f"ENSG9000000{i:04d}.{i % 9 + 1}" for i in range(n_genes)]
    stats = {"ALPHA": [11111, 22222, 33333, 44444],
             "MID": [55555, 66666, 77777, 88888],
             "ZEBRA": [99999, 10101, 12121, 13131]}
    if mode == "samestats":
        stats = {s: [1111, 2222, 3333, 4444] for s in _SAMPLES}
    ncols = 2 if mode == "twocol" else 4
    for s in _SAMPLES:
        g = genes
        c = rng.poisson(5, n_genes)
        if mode == "het" and s == "ZEBRA":
            g = genes[7:] + genes[:7]          # reordered
            g = g[:-4]                          # 4 rows missing
            g = g + ["ENSG90000999998.1", "ENSG90000999999.1"]
            c = rng.poisson(5, len(g))
        if mode == "na" and s == "ALPHA":
            g = list(g)
            g[5] = "NULL"                       # pandas default NA token
        lines = ["\t".join([str(v)] * ncols) for v in stats[s]]
        for gi, ci in zip(g, c):
            rev = int(rng.binomial(ci, 0.5)) if ci > 0 else 0
            row = [gi, int(ci), rev, int(ci) - rev][:ncols]
            lines.append("\t".join(str(x) for x in row))
        path = os.path.join(root, f"{s}_ReadsPerGene.out.tab")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        if mode == "dup" and s == "MID":
            with open(path, "w") as f:
                f.write("\n".join(lines + [lines[4]]) + "\n")
    # decoys: nested suffix file (must NOT be found), unrelated names
    nested = os.path.join(root, "subdir")
    os.makedirs(nested, exist_ok=True)
    with open(os.path.join(nested, "DECOY_ReadsPerGene.out.tab"), "w") as f:
        f.write("1\t1\t1\t1\nENSG1\t2\t1\t1\n")
    with open(os.path.join(root, "README.txt"), "w") as f:
        f.write("ignore me\n")
    with open(os.path.join(root, "ALPHA_quant.sf"), "w") as f:
        f.write("Name\tLength\n")


def _run_original(root, project, monkeypatch):
    mod = _orig_mod()
    monkeypatch.setattr(
        sys, "argv",
        ["merge_star_count", "--path", str(root), "--project", project])
    mod.main()


def _run_fast(root, project, monkeypatch, verbose=False):
    """Run the port with a spy on the pandas-fallback entry point.

    Returns (merged_df, n_fallback_calls).
    """
    mscf = _mscf()
    calls = []
    real = mscf._merge_pandas_path

    def spy(*a, **k):
        calls.append(True)
        return real(*a, **k)

    monkeypatch.setattr(mscf, "_merge_pandas_path", spy)
    df = mscf.merge_star_count(str(root), project, verbose=verbose)
    return df, len(calls)


def _out(root, project):
    return os.path.join(root, f"{project}.STAR.count.tsv.gz")


def _decomp(path):
    with gzip.open(path, "rb") as f:
        return f.read()


def _tokens(gz_path):
    """Decompressed TSV -> (column-name list, {row-label bytes: token list}).

    Token-level (raw text) representation: the strictest parity contract —
    float/int text and empty NaN fields must match byte-for-byte.
    """
    data = _decomp(gz_path)
    lines = data.rstrip(b"\n").split(b"\n")
    cols = lines[0].split(b"\t")[1:]
    rows = {}
    for ln in lines[1:]:
        parts = ln.split(b"\t")
        rows[parts[0]] = parts[1:]
    return cols, rows


def _assert_token_parity(port_gz, orig_gz):
    """Align by column NAME and row LABEL; every cell token must be equal."""
    pc, pr = _tokens(port_gz)
    oc, orow = _tokens(orig_gz)
    assert set(pc) == set(oc), (pc, oc)
    assert set(pr) == set(orow), (len(pr), len(orow))
    perm = [oc.index(c) for c in pc]
    n = 0
    for lab, toks in pr.items():
        gtoks = orow[lab]
        assert len(toks) == len(gtoks)
        for k, j in enumerate(perm):
            assert toks[k] == gtoks[j], (lab, pc[k], toks[k], gtoks[j])
            n += 1
    assert n == len(pr) * len(pc)
    return n


def _aligned_bytes(orig_bytes, index, columns):
    """Re-serialize the original decompressed TSV in the port's row+column
    order with the same pandas writer -> byte-comparable."""
    df = pd.read_csv(io.BytesIO(orig_bytes), sep="\t", index_col=0)
    df = df.reindex(index=index)[list(columns)]
    buf = io.StringIO()
    df.to_csv(buf, sep="\t")
    return buf.getvalue().encode()


def _read_frames(port_gz, orig_gz):
    """Both outputs as DataFrames, original reindexed to the port's axes."""
    p = pd.read_csv(port_gz, sep="\t", index_col=0)
    o = pd.read_csv(orig_gz, sep="\t", index_col=0)
    o = o.reindex(index=p.index)[list(p.columns)]
    return p, o


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def test_happy_path_token_parity(tmp_path, monkeypatch):
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    fast_dir2 = tmp_path / "fast2"
    for d in (orig_dir, fast_dir, fast_dir2):
        _gen_star_dir(str(d), mode="happy")

    _run_original(orig_dir, "p", monkeypatch)
    df, n_fb = _run_fast(fast_dir, "p", monkeypatch, verbose=True)
    assert n_fb == 0, "happy path must not use the pandas fallback"

    # deterministic sorted columns (contract deviation from as_completed);
    # single-level discovery: the nested DECOY sample is NOT a column
    assert list(df.columns) == ["ALPHA", "MID", "ZEBRA"]
    assert df.index.name == "ID"
    assert df.shape == (212, 3)  # 4*3 stat rows + 200 genes (bug-compat)

    # exact union-append row order anchored to the sorted-first sample
    expect_rows = (["11111", "22222", "33333", "44444"]
                   + [f"ENSG9000000{i:04d}.{i % 9 + 1}" for i in range(200)]
                   + ["55555", "66666", "77777", "88888"]
                   + ["99999", "10101", "12121", "13131"])
    assert [str(i) for i in df.index] == expect_rows

    n = _assert_token_parity(_out(str(fast_dir), "p"), _out(str(orig_dir), "p"))
    assert n == 212 * 3

    # byte-level too: original re-serialized in the port's row+column order
    f_bytes = _decomp(_out(str(fast_dir), "p"))
    o_bytes = _decomp(_out(str(orig_dir), "p"))
    assert f_bytes == _aligned_bytes(o_bytes, df.index, df.columns)

    # returned frame == written file
    back = pd.read_csv(io.BytesIO(f_bytes), sep="\t", index_col=0)
    assert back.equals(df)

    # repeated run: identical decompressed bytes (determinism)
    _run_fast(fast_dir2, "p", monkeypatch)
    assert _decomp(_out(str(fast_dir), "p")) == _decomp(_out(str(fast_dir2), "p"))


def test_stat_row_pollution_bugcompat(tmp_path, monkeypatch):
    """The upstream defect is PRESERVED: stat rows are data rows."""
    d = tmp_path / "poll"
    _gen_star_dir(str(d), mode="happy", n_genes=50)
    df, n_fb = _run_fast(d, "p", monkeypatch)
    assert n_fb == 0
    stats = {"ALPHA": [11111, 22222, 33333, 44444],
             "MID": [55555, 66666, 77777, 88888],
             "ZEBRA": [99999, 10101, 12121, 13131]}
    gene_rows = [i for i in df.index.astype(str) if i.startswith("ENSG")]
    assert len(gene_rows) == 50
    assert df.shape == (62, 3)  # 50 genes + 12 unpurged stat rows
    for s, vals in stats.items():
        for v in vals:
            row = df.loc[str(v)]
            assert row[s] == float(v)          # value == source stat integer
            others = [c for c in df.columns if c != s]
            assert row[others].isna().all()    # NaN-padded in other samples
    # dtypes: every column carries NaN -> float64 upcast, ".0" tokens
    assert set(map(str, df.dtypes)) == {"float64"}
    raw = _decomp(_out(str(d), "p"))
    assert raw.split(b"\n")[1] == b"11111\t11111.0\t\t"


def test_same_stats_int64_tokens(tmp_path, monkeypatch):
    """Identical label sets -> no NaN -> pandas keeps int64 (integer tokens)."""
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    for d in (orig_dir, fast_dir):
        _gen_star_dir(str(d), mode="samestats", n_genes=60)

    _run_original(orig_dir, "p", monkeypatch)
    df, n_fb = _run_fast(fast_dir, "p", monkeypatch)
    assert n_fb == 0, "equal label sets stay on the fast path"
    assert df.shape == (64, 3)  # 4 shared stat rows + 60 genes
    assert set(map(str, df.dtypes)) == {"int64"}

    f_bytes = _decomp(_out(str(fast_dir), "p"))
    o_bytes = _decomp(_out(str(orig_dir), "p"))
    # row order is anchor-invariant here -> column-aligned byte equality
    assert f_bytes == _aligned_bytes(o_bytes, df.index, df.columns)
    first_row = f_bytes.split(b"\n")[1].split(b"\t")
    assert all(b"." not in tok for tok in first_row[1:]), first_row


def test_twocol_files_fast_path(tmp_path, monkeypatch):
    """2-column STAR files exercise the stride-1 branch of the fast parser."""
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    for d in (orig_dir, fast_dir):
        _gen_star_dir(str(d), mode="twocol", n_genes=40)

    _run_original(orig_dir, "p", monkeypatch)
    df, n_fb = _run_fast(fast_dir, "p", monkeypatch)
    assert n_fb == 0
    assert df.shape == (52, 3)
    n = _assert_token_parity(_out(str(fast_dir), "p"), _out(str(orig_dir), "p"))
    assert n == 52 * 3


def test_heterogeneous_rows_fast_union(tmp_path, monkeypatch):
    """Differing gene-row sets: the fast union reproduces concat semantics."""
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    for d in (orig_dir, fast_dir):
        _gen_star_dir(str(d), mode="het")

    _run_original(orig_dir, "p", monkeypatch)
    df, n_fb = _run_fast(fast_dir, "p", monkeypatch)
    assert n_fb == 0, "heterogeneous rows are handled by the fast union"

    # union rows: 200 + 12 stats + the 2 ZEBRA-only genes
    assert df.shape == (214, 3)
    assert df.isna().any().any()
    n = _assert_token_parity(_out(str(fast_dir), "p"), _out(str(orig_dir), "p"))
    assert n == 214 * 3

    # frame equality incl. dtypes and NaN pattern (the ORIGINAL anchors the
    # union row order to whichever sample completes first; sort-normalize)
    o = pd.read_csv(io.BytesIO(_decomp(_out(str(orig_dir), "p"))), sep="\t", index_col=0)
    f = pd.read_csv(io.BytesIO(_decomp(_out(str(fast_dir), "p"))), sep="\t", index_col=0)
    o = o[sorted(o.columns)].sort_index()
    f = f[sorted(f.columns)].sort_index()
    assert o.equals(f)

    # the port is deterministic on this input
    fast_dir2 = tmp_path / "fast2"
    _gen_star_dir(str(fast_dir2), mode="het")
    _run_fast(fast_dir2, "p", monkeypatch)
    assert _decomp(_out(str(fast_dir), "p")) == _decomp(_out(str(fast_dir2), "p"))


def test_duplicate_labels_same_error(tmp_path, monkeypatch):
    mscf = _mscf()
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    for d in (orig_dir, fast_dir):
        _gen_star_dir(str(d), mode="dup", n_genes=40)

    with pytest.raises(Exception) as e_orig:
        _run_original(orig_dir, "p", monkeypatch)
    calls = []
    real = mscf._merge_pandas_path

    def spy(*a, **k):
        calls.append(True)
        return real(*a, **k)

    monkeypatch.setattr(mscf, "_merge_pandas_path", spy)
    with pytest.raises(Exception) as e_fast:
        mscf.merge_star_count(str(fast_dir), "p", verbose=False)
    assert len(calls) == 1, "duplicate labels must route to the pandas fallback"
    assert type(e_orig.value) is type(e_fast.value)
    assert str(e_orig.value) == str(e_fast.value)
    # upstream wrote nothing; the port must not either
    assert not os.path.exists(_out(str(orig_dir), "p"))
    assert not os.path.exists(_out(str(fast_dir), "p"))


def test_na_token_label_fallback(tmp_path, monkeypatch):
    """A pandas NA token ('NULL') in the ID column routes to the fallback
    and reproduces the NaN index label + empty output field."""
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    for d in (orig_dir, fast_dir):
        _gen_star_dir(str(d), mode="na", n_genes=40)

    _run_original(orig_dir, "p", monkeypatch)
    df, n_fb = _run_fast(fast_dir, "p", monkeypatch)
    assert n_fb == 1, "NA-value ID tokens must route to the pandas fallback"
    assert df.index.isna().any(), "the NULL label becomes a NaN index entry"

    o = pd.read_csv(io.BytesIO(_decomp(_out(str(orig_dir), "p"))), sep="\t", index_col=0)
    f = pd.read_csv(io.BytesIO(_decomp(_out(str(fast_dir), "p"))), sep="\t", index_col=0)
    o = o[sorted(o.columns)].sort_index()
    f = f[sorted(f.columns)].sort_index()
    assert o.equals(f)
    # NaN label written as an empty field, as upstream
    labels = [ln.split(b"\t")[0] for ln in _decomp(_out(str(fast_dir), "p")).split(b"\n")[1:]]
    assert b"" in labels


def test_verbose_protocol(tmp_path, monkeypatch, capsys):
    d = tmp_path / "v"
    _gen_star_dir(str(d), mode="happy", n_genes=30)
    _run_fast(d, "p", monkeypatch, verbose=True)
    out = capsys.readouterr().out
    assert "Saving merged matrix..." in out
    assert "Head of merged file:" in out
    assert "Number of rows: 42" in out      # 30 genes + 12 stat rows
    assert "Number of columns: 3" in out
    assert f"Saved to: {_out(str(d), 'p')}" in out
    assert "IOBRpy: Immuno-Oncology Biological Research using Python" in out


def test_empty_dir_notice(tmp_path, monkeypatch, capsys):
    mscf = _mscf()
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    orig_dir.mkdir()
    fast_dir.mkdir()

    _run_original(orig_dir, "p", monkeypatch)
    o_out = capsys.readouterr().out
    ret = mscf.merge_star_count(str(fast_dir), "p")
    f_out = capsys.readouterr().out
    assert ret is None
    notice = ("No files ending with '_ReadsPerGene.out.tab' were found "
              "in the given path.")
    assert notice in o_out and notice in f_out
    assert list(orig_dir.iterdir()) == [] and list(fast_dir.iterdir()) == []


def test_wired_wrapper_when_available(tmp_path, monkeypatch):
    """Activates once the strategy layer wires ``iobrx.merge_star_count``.

    Checks the wrapper's backend contract: 'auto' (fast path) and 'python'
    (untouched upstream CLI via argv rewrite) produce the same aligned
    tokens; 'rust' fails clearly (pure-Python module, no native kernel).
    """
    import iobrx

    fn = getattr(iobrx, "merge_star_count", None)
    if fn is None:
        pytest.skip("iobrx.merge_star_count not wired yet (see INTEGRATION.md)")
    d_auto = tmp_path / "auto"
    d_py = tmp_path / "py"
    for d in (d_auto, d_py):
        _gen_star_dir(str(d), mode="happy", n_genes=50)
    fn(str(d_auto), "p", backend="auto", verbose=False)
    fn(str(d_py), "p", backend="python")
    n = _assert_token_parity(_out(str(d_auto), "p"), _out(str(d_py), "p"))
    assert n == 62 * 3
    with pytest.raises(RuntimeError):
        fn(str(d_auto), "p2", backend="rust")
