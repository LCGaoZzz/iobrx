"""prepare_salmon parity: synthetic-input gates against the ORIGINAL iobrpy workflow.

Fast gates (no official data download needed): the accelerated
``iobrx._fast.prepare_salmon_fast.prepare_salmon`` path must be
byte-identical (output-file sha256) to the ORIGINAL
``iobrpy.workflow.prepare_salmon.prepare_salmon_tpm`` (a directly
importable function) on small synthetic GENCODE-style Salmon matrices
that exercise every branch:

* the in-place pipe-ID annotation parse (8-field ``Name``; symbol / ENST /
  ENSG selection incl. case-insensitive matching; ``.version`` stripping);
* ``remove_duplicate_genes`` groupby-mean dedup with the ORIGINAL
  alphabetical (groupby sort=True) row order;
* ``.tsv`` and ``.tsv.gz`` inputs (``compression='infer'``);
* the ALWAYS-comma-CSV output quirk (``to_csv`` without ``sep`` — even for
  a ``.tsv`` output path);
* the BUG-COMPAT swallowed-error contract: missing ``Name`` column,
  invalid ``return_feature``, and <8 pipe fields all print a red
  ``Error occurred: ...`` line, do NOT raise, return None and leave the
  output file unwritten (upstream CLI still exits rc=0);
* engine='original' routes to the untouched upstream function.

The official-data byte gate (fixture8_salmon_tpm.tsv.gz 60,000x8 ->
30,000x9 CSV, sha256 ``d5546a04...``) lives in the campaign scratch
(parity_gates.py) and the blind benchmark pass; it needs the frozen IOBRpy
baseline output, so it is not part of this fast file.
"""
from __future__ import annotations

import gzip
import hashlib

import numpy as np
import pandas as pd
import pytest

from iobrx._fast import prepare_salmon_fast as F

orig_tpm = pytest.importorskip("iobrpy.workflow.prepare_salmon").prepare_salmon_tpm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha(path) -> str:
    with open(str(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _gencode_name(i: int, t: int, sym: str) -> str:
    return (f"ENST9{i:010d}.{t}|ENSG8{i:010d}.{t}|OTTHUMG7{i:05d}.{t}|"
            f"OTTHUMT6{i:05d}.{t}|{sym}-20{t}|{sym}|{1000 + i}|protein_coding")


def _salmon_frame(n_genes=30, n_samples=4, seed=5):
    """GENCODE 8-field Names; every 3rd gene gets 2-3 transcripts (dups)."""
    rng = np.random.default_rng(seed)
    rows, vals = [], []
    for i in range(n_genes):
        sym = f"SYM{i:05d}"
        n_tx = (i % 3) + 1
        for t in range(1, n_tx + 1):
            rows.append(_gencode_name(i, t, sym))
            vals.append(np.round(rng.random(n_samples) * 120, 6))
    df = pd.DataFrame(vals, columns=[f"S{j}" for j in range(n_samples)])
    df.insert(0, "Name", rows)
    return df


def _write_tsv(df, path, gz=False):
    if gz:
        with gzip.open(str(path), "wt", newline="") as f:
            f.write(df.to_csv(sep="\t", index=False))
    else:
        df.to_csv(str(path), sep="\t", index=False)


# ---------------------------------------------------------------------------
# byte gates
# ---------------------------------------------------------------------------
def test_symbol_noversion_bytes_identical(tmp_path):
    df = _salmon_frame()
    inp = tmp_path / "eset.tsv"
    _write_tsv(df, inp)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    res = F.prepare_salmon(str(inp), str(fast_out),
                           return_feature="symbol", remove_version=True)
    orig_tpm(str(inp), str(orig_out), "symbol", True)
    assert _sha(fast_out) == _sha(orig_out)
    # contract: 30 unique symbols, alphabetical rows, Name + 4 sample cols
    out = pd.read_csv(fast_out)
    assert list(out["Name"]) == sorted(f"SYM{i:05d}" for i in range(30))
    assert list(out.columns) == ["Name", "S0", "S1", "S2", "S3"]
    pd.testing.assert_frame_equal(res.reset_index(drop=True), out)


def test_duplicate_values_are_group_means(tmp_path):
    df = _salmon_frame(n_genes=9, n_samples=3)
    inp = tmp_path / "eset.tsv"
    _write_tsv(df, inp)
    out = tmp_path / "fast.csv"
    F.prepare_salmon(str(inp), str(out), return_feature="symbol",
                     remove_version=True)
    got = pd.read_csv(out).set_index("Name")
    src = df.copy()
    src["sym"] = src["Name"].str.split("|").str[5]
    expect = src.groupby("sym")[["S0", "S1", "S2"]].mean()
    np.testing.assert_allclose(got.to_numpy(), expect.to_numpy(),
                               rtol=0, atol=1e-12)


def test_enst_ensg_and_case_insensitive_feature(tmp_path):
    df = _salmon_frame(n_genes=12, n_samples=2)
    inp = tmp_path / "eset.tsv"
    _write_tsv(df, inp)
    for rf in ("ENST", "ensg", "SYMBOL"):     # .lower() matching, as upstream
        fo, oo = tmp_path / f"fast_{rf}.csv", tmp_path / f"orig_{rf}.csv"
        F.prepare_salmon(str(inp), str(fo), return_feature=rf,
                         remove_version=False)
        orig_tpm(str(inp), str(oo), rf, False)
        assert _sha(fo) == _sha(oo), rf
        head = pd.read_csv(fo)["Name"].iloc[0]
        if rf.lower() == "enst":
            assert head.startswith("ENST9") and "." in head   # version kept
        if rf.lower() == "ensg":
            assert head.startswith("ENSG8")


def test_gz_input_bytes_identical(tmp_path):
    df = _salmon_frame(n_genes=15, n_samples=3)
    inp = tmp_path / "eset.tsv.gz"
    _write_tsv(df, inp, gz=True)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    F.prepare_salmon(str(inp), str(fast_out), return_feature="symbol",
                     remove_version=True)
    orig_tpm(str(inp), str(orig_out), "symbol", True)
    assert _sha(fast_out) == _sha(orig_out)


def test_output_always_comma_csv_even_for_tsv_path(tmp_path):
    df = _salmon_frame(n_genes=6, n_samples=2)
    inp = tmp_path / "eset.tsv"
    _write_tsv(df, inp)
    fo, oo = tmp_path / "fast.tsv", tmp_path / "orig.tsv"
    F.prepare_salmon(str(inp), str(fo))
    orig_tpm(str(inp), str(oo), "symbol", False)
    assert _sha(fo) == _sha(oo)
    first = fo.read_text().splitlines()[0]
    assert "," in first and "\t" not in first   # upstream quirk, preserved


# ---------------------------------------------------------------------------
# BUG-COMPAT swallowed errors (upstream try/except: red line, rc=0, no file)
# ---------------------------------------------------------------------------
def test_missing_name_column_swallowed(tmp_path, capsys):
    df = _salmon_frame(n_genes=5, n_samples=2).drop(columns=["Name"])
    inp = tmp_path / "noname.tsv"
    df["X"] = range(len(df))
    _write_tsv(df, inp)
    out = tmp_path / "never.csv"
    assert F.prepare_salmon(str(inp), str(out)) is None
    assert not out.exists()
    # upstream swallows identically
    out2 = tmp_path / "never2.csv"
    assert orig_tpm(str(inp), str(out2)) is None
    assert not out2.exists()


def test_invalid_return_feature_swallowed(tmp_path):
    df = _salmon_frame(n_genes=5, n_samples=2)
    inp = tmp_path / "eset.tsv"
    _write_tsv(df, inp)
    out = tmp_path / "never.csv"
    assert F.prepare_salmon(str(inp), str(out), return_feature="entrez") is None
    assert not out.exists()
    out2 = tmp_path / "never2.csv"
    assert orig_tpm(str(inp), str(out2), "entrez", False) is None
    assert not out2.exists()


def test_short_pipe_fields_swallowed(tmp_path):
    df = _salmon_frame(n_genes=5, n_samples=2)
    df["Name"] = df["Name"].str.split("|").str[:4].str.join("|")   # <8 fields
    inp = tmp_path / "short.tsv"
    _write_tsv(df, inp)
    out = tmp_path / "never.csv"
    assert F.prepare_salmon(str(inp), str(out)) is None
    assert not out.exists()
    out2 = tmp_path / "never2.csv"
    assert orig_tpm(str(inp), str(out2), "symbol", False) is None
    assert not out2.exists()


def test_engine_original_writes_same_bytes(tmp_path):
    df = _salmon_frame(n_genes=10, n_samples=3)
    inp = tmp_path / "eset.tsv"
    _write_tsv(df, inp)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    assert F.prepare_salmon(str(inp), str(fast_out)) is not None
    assert F.prepare_salmon(str(inp), str(orig_out), engine="original") is None
    assert _sha(fast_out) == _sha(orig_out)
