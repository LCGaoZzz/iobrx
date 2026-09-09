"""merge_salmon parity gates (synthetic in-test fixture; default suite).

Compares the accelerated port ``iobrx._fast.merge_salmon_fast.merge_salmon``
against the ORIGINAL ``iobrpy.workflow.merge_salmon`` CLI on quant.sf trees
generated below (200 transcripts x 3 samples, official 5-column layout,
GENCODE pipe Names, zero and scientific-notation TPM tokens — the token
shapes of the frozen 8 x 60k baseline fixture).

Contract under test (see research/merge/module_spec.md §1 and the port's
INTEGRATION.md):

* happy path — decompressed ``*_salmon_tpm.tsv.gz`` / ``*_salmon_count.tsv.gz``
  are BYTE-identical to the original after column alignment; the port's
  columns are deterministically sorted by sample name (the ORIGINAL's
  ``as_completed`` order is nondeterministic by design); repeated runs are
  content-stable; the pandas fallback is NOT used.
* guard-tripping inputs take the port's pandas fallback and reproduce
  ORIGINAL semantics: integer-form value columns (int64 dtype inference →
  integer tokens on disk), duplicate ``Name`` rows (concat accepts identical
  duplicated indexes), heterogeneous row sets (outer-join union: same rows,
  same values, same NaN pattern — the ORIGINAL anchors union row order to
  whichever sample completes first, the port anchors it to the sorted-first
  sample), and ``#``-comment headers (the same ``ValueError``).
* empty root — notice printed, nothing written, ``(None, None)`` returned.
* rust read path (when ``_rust.merge_salmon_parse`` is present) —
  ``engine='rust'`` output is byte-identical to ``engine='python'`` on
  happy and guard-tripping fixtures (the kernel declines whole runs to the
  sequential parser), reproduces the ``#``-header ``ValueError``, and is
  thread-count invariant (num_processes 1/2/16).

Run with the repo source on the path::

    PYTHONPATH=<repo>/src pytest -q tests/test_parity_merge_salmon.py
"""
from __future__ import annotations

import gzip
import io
import os
import sys

import pandas as pd
import pytest


def _msf():
    return pytest.importorskip(
        "iobrx._fast.merge_salmon_fast",
        reason="needs PYTHONPATH=<repo>/src (or an installed iobrx)",
    )


def _orig_mod():
    return pytest.importorskip("iobrpy.workflow.merge_salmon")


# ---------------------------------------------------------------------------
# fixture generation (self-contained, seeded)
# ---------------------------------------------------------------------------
def _pipe_name(i: int) -> str:
    return (f"ENST9000000{i:04d}.9|ENSG9000000{i:04d}.1|OTTHUMG{i:05d}.6|"
            f"OTTHUMT{i:05d}.6|SYN{i:05d}-201|SYN{i:05d}|12149|protein_coding")


def _gen_quant_tree(root, mode="happy", n_genes=200, seed=20260908):
    """Write 3 sample dirs (one nested) of official-format quant.sf.

    Sample names ALPHA / MID / ZEBRA: sorted order != creation order, and
    ZEBRA lives one directory deeper to exercise the recursive walk.
    modes: happy | int (integer-form NumReads) | het (ZEBRA rows reordered,
    trimmed and extended) | dup (first data row repeated) | hash ('#' banner).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    names = [_pipe_name(i) for i in range(n_genes)]
    layout = {"ALPHA": os.path.join(root, "ALPHA"),
              "MID": os.path.join(root, "batch2", "MID"),
              "ZEBRA": os.path.join(root, "batch2", "run9", "ZEBRA")}
    for s, sdir in layout.items():
        os.makedirs(sdir, exist_ok=True)
        nm = names
        if mode == "het" and s == "ZEBRA":
            nm = names[7:] + names[:7]          # reordered
            nm = nm[:-4]                        # 4 rows missing
            nm = nm + [_pipe_name(n_genes + 1), _pipe_name(n_genes + 2)]
        a = rng.random(len(nm))
        a[rng.random(len(nm)) < 0.3] = 0.0      # 30% zero rows -> "0.0" tokens
        tpm = (a / a.sum() * 1e6).round(6)
        tpm[tpm > 0] = np.minimum(tpm[tpm > 0], 1e6)
        if (tpm > 0).any():
            j = int(np.flatnonzero(tpm > 0)[0])
            tpm[j] = 1.8e-05                    # scientific-notation token
        nr = (tpm / 1e6 * 3e7 + rng.normal(0, 1, len(nm))).round(3)
        nr = np.maximum(nr, 0.0)
        if mode == "int":
            nr = rng.integers(0, 50, len(nm))   # integer-form tokens
        df = pd.DataFrame({"Name": nm,
                           "Length": rng.integers(200, 15000, len(nm)),
                           "EffectiveLength": (rng.random(len(nm)) * 1000).round(3) + 50,
                           "TPM": tpm, "NumReads": nr})
        path = os.path.join(sdir, "quant.sf")
        df.to_csv(path, sep="\t", index=False)
        if mode == "dup":
            lines = open(path).read().splitlines()
            with open(path, "w") as f:
                f.write("\n".join(lines + [lines[1]]) + "\n")
        if mode == "hash":
            body = open(path).read()
            with open(path, "w") as f:
                f.write("# Salmon v1.10.3\n" + body)


def _run_original(root, project, monkeypatch):
    mod = _orig_mod()
    monkeypatch.setattr(
        sys, "argv",
        ["merge_salmon", "--path_salmon", str(root), "--project", project])
    mod.main()


def _run_fast(root, project, monkeypatch, verbose=False):
    """Run the port with a spy on the pandas-fallback entry point.

    Returns (tpm_df, cnt_df, n_fallback_calls).
    """
    msf = _msf()
    calls = []
    real = msf._merge_pandas_path

    def spy(*a, **k):
        calls.append(True)
        return real(*a, **k)

    monkeypatch.setattr(msf, "_merge_pandas_path", spy)
    tpm_df, cnt_df = msf.merge_salmon(str(root), project, verbose=verbose)
    return tpm_df, cnt_df, len(calls)


def _decomp(path):
    with gzip.open(path, "rb") as f:
        return f.read()


def _aligned_to(orig_bytes, columns):
    """Re-serialize the original decompressed TSV with the port's column
    order using the same pandas writer → byte-comparable."""
    df = pd.read_csv(io.BytesIO(orig_bytes), sep="\t")
    assert set(df.columns) == set(columns), (list(df.columns), columns)
    buf = io.StringIO()
    df[list(columns)].to_csv(buf, sep="\t", index=False)
    return buf.getvalue().encode()


def _outputs(root, project):
    return (os.path.join(root, f"{project}_salmon_tpm.tsv.gz"),
            os.path.join(root, f"{project}_salmon_count.tsv.gz"))


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def test_happy_path_byte_parity(tmp_path, monkeypatch):
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    fast_dir2 = tmp_path / "fast2"
    for d in (orig_dir, fast_dir, fast_dir2):
        _gen_quant_tree(str(d), mode="happy")

    _run_original(orig_dir, "p", monkeypatch)
    tpm_df, cnt_df, n_fb = _run_fast(fast_dir, "p", monkeypatch, verbose=True)
    assert n_fb == 0, "happy path must not use the pandas fallback"

    # deterministic, sorted column order (contract deviation from as_completed)
    assert list(tpm_df.columns) == ["ALPHA", "MID", "ZEBRA"]
    assert tpm_df.index.name == "Name"
    assert tpm_df.shape == (200, 3) and cnt_df.shape == (200, 3)

    for tag, df in (("tpm", tpm_df), ("count", cnt_df)):
        o_bytes = _decomp(_outputs(str(orig_dir), "p")[0 if tag == "tpm" else 1])
        f_bytes = _decomp(_outputs(str(fast_dir), "p")[0 if tag == "tpm" else 1])
        assert f_bytes == _aligned_to(o_bytes, ["Name"] + list(df.columns)), tag
        # returned frame == written file
        back = pd.read_csv(io.BytesIO(f_bytes), sep="\t", index_col=0)
        assert back.equals(df), tag

    # repeated run: identical decompressed content (determinism)
    _run_fast(fast_dir2, "p", monkeypatch)
    for i in (0, 1):
        assert _decomp(_outputs(str(fast_dir), "p")[i]) == \
               _decomp(_outputs(str(fast_dir2), "p")[i])


def test_verbose_protocol(tmp_path, monkeypatch, capsys):
    d = tmp_path / "v"
    _gen_quant_tree(str(d), mode="happy", n_genes=30)
    _run_fast(d, "p", monkeypatch, verbose=True)
    out = capsys.readouterr().out
    assert "TPM head:" in out and "Count head:" in out
    assert "Saving TPM matrix..." in out and "Saving Count matrix..." in out
    assert f"Saved to (TPM): {os.path.join(str(d), 'p_salmon_tpm.tsv.gz')}" in out
    assert f"Saved to (Count): {os.path.join(str(d), 'p_salmon_count.tsv.gz')}" in out
    assert "IOBRpy: Immuno-Oncology Biological Research using Python" in out


def test_integer_numreads_fallback_keeps_int_tokens(tmp_path, monkeypatch):
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    for d in (orig_dir, fast_dir):
        _gen_quant_tree(str(d), mode="int", n_genes=60)

    _run_original(orig_dir, "p", monkeypatch)
    _, _, n_fb = _run_fast(fast_dir, "p", monkeypatch)
    assert n_fb == 1, "integer-form NumReads must route to the pandas fallback"

    o_cnt = _decomp(_outputs(str(orig_dir), "p")[1])
    f_cnt = _decomp(_outputs(str(fast_dir), "p")[1])
    cols = ["Name", "ALPHA", "MID", "ZEBRA"]
    assert f_cnt == _aligned_to(o_cnt, cols)
    # int64 tokens (no ".0") survived — the upstream dtype quirk is preserved
    first_row = o_cnt.split(b"\n")[1].split(b"\t")
    assert all(b"." not in tok for tok in first_row[1:])
    o_tpm = _decomp(_outputs(str(orig_dir), "p")[0])
    f_tpm = _decomp(_outputs(str(fast_dir), "p")[0])
    assert f_tpm == _aligned_to(o_tpm, cols)


def test_heterogeneous_rows_fallback_outer_join(tmp_path, monkeypatch):
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    for d in (orig_dir, fast_dir):
        _gen_quant_tree(str(d), mode="het")

    _run_original(orig_dir, "p", monkeypatch)
    tpm_df, cnt_df, n_fb = _run_fast(fast_dir, "p", monkeypatch)
    assert n_fb == 1, "differing row sets must route to the pandas fallback"

    # union rows: 200 + the 2 ZEBRA-only transcripts
    assert tpm_df.shape == (202, 3) and cnt_df.shape == (202, 3)
    assert tpm_df.isna().any().any(), "outer join must pad missing rows with NaN"

    for i, tag in enumerate(("tpm", "count")):
        o = pd.read_csv(io.BytesIO(_decomp(_outputs(str(orig_dir), "p")[i])), sep="\t")
        f = pd.read_csv(io.BytesIO(_decomp(_outputs(str(fast_dir), "p")[i])), sep="\t")
        o = o.set_index("Name")[sorted(set(o.columns) - {"Name"})].sort_index()
        f = f.set_index("Name")[sorted(set(f.columns) - {"Name"})].sort_index()
        assert o.equals(f), tag  # same rows, values, dtypes AND NaN pattern

    # the port is deterministic on this input (the ORIGINAL's union row
    # order depends on which sample completes first)
    fast_dir2 = tmp_path / "fast2"
    _gen_quant_tree(str(fast_dir2), mode="het")
    _run_fast(fast_dir2, "p", monkeypatch)
    for i in (0, 1):
        assert _decomp(_outputs(str(fast_dir), "p")[i]) == \
               _decomp(_outputs(str(fast_dir2), "p")[i])


def test_duplicate_names_fallback(tmp_path, monkeypatch):
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    for d in (orig_dir, fast_dir):
        _gen_quant_tree(str(d), mode="dup", n_genes=40)

    _run_original(orig_dir, "p", monkeypatch)
    _, _, n_fb = _run_fast(fast_dir, "p", monkeypatch)
    assert n_fb == 1, "duplicate Name labels must route to the pandas fallback"
    cols = ["Name", "ALPHA", "MID", "ZEBRA"]
    for i in (0, 1):
        o = _decomp(_outputs(str(orig_dir), "p")[i])
        f = _decomp(_outputs(str(fast_dir), "p")[i])
        assert f == _aligned_to(o, cols)
        assert o.count(b"\n") - 1 == 41  # 40 rows + the duplicated one


def test_hash_header_raises_same_valueerror(tmp_path, monkeypatch):
    msf = _msf()
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    for d in (orig_dir, fast_dir):
        _gen_quant_tree(str(d), mode="hash", n_genes=20)

    with pytest.raises(ValueError) as e_orig:
        _run_original(orig_dir, "p", monkeypatch)
    with pytest.raises(ValueError) as e_fast:
        msf.merge_salmon(str(fast_dir), "p", verbose=False)
    assert str(e_orig.value) == str(e_fast.value)
    # upstream wrote nothing; the port must not either
    for d in (orig_dir, fast_dir):
        assert not os.path.exists(_outputs(str(d), "p")[0])
        assert not os.path.exists(_outputs(str(d), "p")[1])


def test_empty_root_notice(tmp_path, monkeypatch, capsys):
    msf = _msf()
    orig_dir = tmp_path / "orig"
    fast_dir = tmp_path / "fast"
    orig_dir.mkdir()
    fast_dir.mkdir()

    _run_original(orig_dir, "p", monkeypatch)
    o_out = capsys.readouterr().out
    ret = msf.merge_salmon(str(fast_dir), "p")
    f_out = capsys.readouterr().out
    assert ret == (None, None)
    notice = "No quant.sf files were found under the given --path_salmon."
    assert notice in o_out and notice in f_out
    assert list(orig_dir.iterdir()) == [] and list(fast_dir.iterdir()) == []


def test_wired_wrapper_when_available(tmp_path, monkeypatch):
    """Activates once the strategy layer wires ``iobrx.merge_salmon``.

    Checks the wrapper's backend contract: 'auto' (native Rust read path
    when ``_rust.merge_salmon_parse`` is importable, else the sequential
    fast parser), explicit 'rust', and 'python' (untouched upstream CLI via
    argv rewrite) produce the same aligned bytes; 'auto' and 'rust' are
    byte-identical after decompression. 'rust' fails clearly only when the
    native extension/kernel itself is missing.
    """
    import iobrx

    fn = getattr(iobrx, "merge_salmon", None)
    if fn is None:
        pytest.skip("iobrx.merge_salmon not wired yet (see INTEGRATION.md)")
    has_kernel = _rust_kernel() is not None
    tags = ("auto", "py", "rust") if has_kernel else ("auto", "py")
    dirs = {}
    for tag in tags:
        d = tmp_path / tag
        _gen_quant_tree(str(d), mode="happy", n_genes=50)
        dirs[tag] = d
    fn(str(dirs["auto"]), "p", backend="auto", verbose=False)
    fn(str(dirs["py"]), "p", backend="python")
    if has_kernel:
        fn(str(dirs["rust"]), "p", backend="rust", verbose=False)
    for tag, d in dirs.items():
        for i in (0, 1):
            a = pd.read_csv(io.BytesIO(_decomp(_outputs(str(dirs["auto"]), "p")[i])), sep="\t")
            b = pd.read_csv(io.BytesIO(_decomp(_outputs(str(d), "p")[i])), sep="\t")
            b = b[list(a.columns)]
            assert a.equals(b), tag
    if has_kernel:
        for i in (0, 1):
            assert _decomp(_outputs(str(dirs["rust"]), "p")[i]) == \
                   _decomp(_outputs(str(dirs["auto"]), "p")[i])
    else:
        with pytest.raises(RuntimeError):
            fn(str(dirs["auto"]), "p2", backend="rust")


# ---------------------------------------------------------------------------
# rust read-path gates (active when _rust.merge_salmon_parse is present)
# ---------------------------------------------------------------------------
def _rust_kernel():
    try:
        import iobrx._rust as _ir

        return getattr(_ir, "merge_salmon_parse", None)
    except (ImportError, OSError):
        return None


def _run_fast_engine(root, project, monkeypatch, engine):
    """Run the port with a chosen read engine, spying on the pandas
    fallback and on the rust kernel resolution.

    Returns (tpm_df, cnt_df, n_pandas_fallback, rust_calls) where
    rust_calls holds the kernel's ok flag per call (empty for
    engine='python'). Spies are removed before returning so repeated
    calls in one test never chain.
    """
    msf = _msf()
    orig_pd = msf._merge_pandas_path
    orig_resolve = msf._rust_parse_fn
    pd_calls = []
    rust_calls = []

    def pd_spy(*a, **k):
        pd_calls.append(True)
        return orig_pd(*a, **k)

    def resolve_spy(strict):
        fn = orig_resolve(strict)
        if fn is None:
            return None

        def fn_spy(*a, **k):
            out = fn(*a, **k)
            rust_calls.append(bool(out[0]))
            return out

        return fn_spy

    monkeypatch.setattr(msf, "_merge_pandas_path", pd_spy)
    monkeypatch.setattr(msf, "_rust_parse_fn", resolve_spy)
    try:
        tpm_df, cnt_df = msf.merge_salmon(str(root), project, verbose=False,
                                          engine=engine)
    finally:
        monkeypatch.setattr(msf, "_merge_pandas_path", orig_pd)
        monkeypatch.setattr(msf, "_rust_parse_fn", orig_resolve)
    return tpm_df, cnt_df, len(pd_calls), rust_calls


def test_rust_engine_happy_path_byte_parity(tmp_path, monkeypatch):
    if _rust_kernel() is None:
        pytest.skip("_rust.merge_salmon_parse unavailable")
    import numpy as np

    py_dir = tmp_path / "py"
    rs_dir = tmp_path / "rs"
    orig_dir = tmp_path / "orig"
    for d in (py_dir, rs_dir, orig_dir):
        _gen_quant_tree(str(d), mode="happy")

    tpm_py, cnt_py, nfb_py, rc_py = _run_fast_engine(py_dir, "p", monkeypatch,
                                                     "python")
    tpm_rs, cnt_rs, nfb_rs, rc_rs = _run_fast_engine(rs_dir, "p", monkeypatch,
                                                     "rust")
    assert rc_py == [] and nfb_py == 0
    assert rc_rs == [True], "happy fixture must be accepted by the rust kernel"
    assert nfb_rs == 0

    # returned frames bitwise equal
    for a, b in ((tpm_rs, tpm_py), (cnt_rs, cnt_py)):
        assert list(a.columns) == list(b.columns)
        assert np.array_equal(np.ascontiguousarray(a.to_numpy()).view(np.uint64),
                              np.ascontiguousarray(b.to_numpy()).view(np.uint64))

    # decompressed file bytes equal (same to_csv writer downstream)
    for i in (0, 1):
        assert _decomp(_outputs(str(rs_dir), "p")[i]) == \
               _decomp(_outputs(str(py_dir), "p")[i])

    # and equal to the ORIGINAL after column alignment
    _run_original(orig_dir, "p", monkeypatch)
    for i, df in enumerate((tpm_rs, cnt_rs)):
        o_bytes = _decomp(_outputs(str(orig_dir), "p")[i])
        r_bytes = _decomp(_outputs(str(rs_dir), "p")[i])
        assert r_bytes == _aligned_to(o_bytes, ["Name"] + list(df.columns))


def test_rust_engine_declines_guard_inputs_like_python(tmp_path, monkeypatch):
    if _rust_kernel() is None:
        pytest.skip("_rust.merge_salmon_parse unavailable")
    for mode in ("int", "dup", "het"):
        py_dir = tmp_path / f"{mode}_py"
        rs_dir = tmp_path / f"{mode}_rs"
        for d in (py_dir, rs_dir):
            _gen_quant_tree(str(d), mode=mode, n_genes=60)
        _, _, nfb_py, rc_py = _run_fast_engine(py_dir, "p", monkeypatch,
                                               "python")
        _, _, nfb_rs, rc_rs = _run_fast_engine(rs_dir, "p", monkeypatch,
                                               "rust")
        assert rc_py == []
        assert rc_rs == [False], f"{mode}: kernel must decline guard inputs"
        assert nfb_rs == nfb_py == 1, f"{mode}: identical fallback chain"
        for i in (0, 1):
            assert _decomp(_outputs(str(rs_dir), "p")[i]) == \
                   _decomp(_outputs(str(py_dir), "p")[i]), mode


def test_rust_engine_hash_header_same_valueerror(tmp_path, monkeypatch):
    if _rust_kernel() is None:
        pytest.skip("_rust.merge_salmon_parse unavailable")
    msf = _msf()
    orig_dir = tmp_path / "orig"
    rs_dir = tmp_path / "rs"
    for d in (orig_dir, rs_dir):
        _gen_quant_tree(str(d), mode="hash", n_genes=20)
    with pytest.raises(ValueError) as e_orig:
        _run_original(orig_dir, "p", monkeypatch)
    with pytest.raises(ValueError) as e_rs:
        msf.merge_salmon(str(rs_dir), "p", verbose=False, engine="rust")
    assert str(e_orig.value) == str(e_rs.value)
    assert not os.path.exists(_outputs(str(rs_dir), "p")[0])
    assert not os.path.exists(_outputs(str(rs_dir), "p")[1])


def test_rust_engine_thread_invariance(tmp_path):
    if _rust_kernel() is None:
        pytest.skip("_rust.merge_salmon_parse unavailable")
    msf = _msf()
    dirs = []
    for w in (1, 2, 16):
        d = tmp_path / f"w{w}"
        _gen_quant_tree(str(d), mode="happy", n_genes=120)
        msf.merge_salmon(str(d), "p", num_processes=w, verbose=False,
                         engine="rust")
        dirs.append(d)
    for i in (0, 1):
        base = _decomp(_outputs(str(dirs[0]), "p")[i])
        for d in dirs[1:]:
            assert _decomp(_outputs(str(d), "p")[i]) == base
