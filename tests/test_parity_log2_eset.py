"""log2_eset parity: synthetic-input gates against the ORIGINAL iobrpy workflow.

Fast gates (no official data download needed): the accelerated
``iobrx._fast.log2_eset_fast.log2_eset`` path must be byte-identical
(output-file sha256) to ``iobrpy.workflow.log2_eset.main()`` (argv-bridged)
on small synthetic matrices that exercise every branch:

* the three-level delimiter detection — csv.Sniffer success on comma /
  tab / semicolon files, and the extension-hint fallback;
* the output-separator rule (``.csv`` -> comma, ``.tsv`` -> tab, ambiguous
  extension mirrors the input delimiter);
* ``.gz`` transparent input (and decompressed-content equality for ``.gz``
  output, whose gzip bytes embed an mtime header and are therefore not
  run-to-run comparable by design — same as upstream);
* the all-numeric fast guard (skip ``df.apply(pd.to_numeric)`` — a
  provable identity on int/uint/float dtypes) AND the object-column path
  that coerces non-numeric cells to NaN with the ORIGINAL warning;
* negative values in ``[-1, 0)`` (warning, continue) and the hard
  ``min < -1`` error — the ORIGINAL ``sys.exit(1)`` becomes
  ``ValueError`` with the ORIGINAL message text (repo convention, cf.
  ``iobrx.nmf``); engine='original' keeps SystemExit;
* the empty-matrix error path.

The official-data byte gate (eset_stad.csv 60,483x10 official testdata ->
log2 CSV, sha256 ``edc2664a...``) lives in the campaign scratch
(parity_gates.py) and the blind benchmark pass; it needs the frozen IOBRpy
baseline output, so it is not part of this fast file.
"""
from __future__ import annotations

import gzip
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest

from iobrx._fast import log2_eset_fast as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha(path) -> str:
    with open(str(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _run_orig_main(inp, out):
    import importlib

    mod = importlib.import_module("iobrpy.workflow.log2_eset")
    saved = sys.argv
    try:
        sys.argv = ["log2_eset", "-i", str(inp), "-o", str(out)]
        mod.main()
    finally:
        sys.argv = saved


def _matrix(n=40, seed=11):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        np.round(rng.random((n, 4)) * 800, 6),
        index=[f"ENSG{i:011d}" for i in range(n)],
        columns=["a", "b", "c", "d"],
    )


# ---------------------------------------------------------------------------
# delimiter helpers (unit level)
# ---------------------------------------------------------------------------
def test_detect_sep_helpers(tmp_path):
    df = _matrix(n=8)
    p_csv = tmp_path / "x.csv"
    df.to_csv(p_csv)
    p_tsv = tmp_path / "x.tsv"
    df.to_csv(p_tsv, sep="\t")
    p_semi = tmp_path / "x_semi.csv"
    df.to_csv(p_semi, sep=";")
    assert F._detect_input_sep(str(p_csv)) == ","
    assert F._detect_input_sep(str(p_tsv)) == "\t"
    assert F._detect_input_sep(str(p_semi)) == ";"
    # output rule: .csv -> comma, .tsv -> tab, else mirror input
    assert F._choose_output_sep("o.csv", "\t") == ","
    assert F._choose_output_sep("o.tsv.gz", ",") == "\t"
    assert F._choose_output_sep("o.txt", ";") == ";"
    assert F._ext_lower("A.CSV.GZ") == ".csv"


# ---------------------------------------------------------------------------
# byte gates
# ---------------------------------------------------------------------------
def test_csv_bytes_identical(tmp_path):
    df = _matrix()
    inp = tmp_path / "in.csv"
    df.to_csv(inp)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    res = F.log2_eset(str(inp), str(fast_out))
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)
    # values are exactly log2(x+1) of the source
    np.testing.assert_allclose(res.to_numpy(), np.log2(df.to_numpy() + 1.0),
                               rtol=0, atol=0)
    assert res.index.name is None or res.index.name == df.index.name


def test_tsv_input_txt_output_mirrors_tab(tmp_path):
    df = _matrix(n=25)
    inp = tmp_path / "in.tsv"
    df.to_csv(inp, sep="\t")
    fast_out, orig_out = tmp_path / "fast.txt", tmp_path / "orig.txt"
    F.log2_eset(str(inp), str(fast_out))
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)
    assert "\t" in fast_out.read_text().splitlines()[0]


def test_semicolon_input_csv_output(tmp_path):
    df = _matrix(n=25)
    inp = tmp_path / "in_semi.txt"
    df.to_csv(inp, sep=";")
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    F.log2_eset(str(inp), str(fast_out))
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)


def test_gz_input_and_gz_output_decompressed(tmp_path):
    df = _matrix(n=30)
    inp = tmp_path / "in.csv.gz"
    with gzip.open(inp, "wt", newline="") as f:
        f.write(df.to_csv())
    # gz input -> plain csv output: file bytes comparable
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    F.log2_eset(str(inp), str(fast_out))
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)
    # gz output: gzip embeds mtime -> compare DECOMPRESSED content (upstream
    # has the same property)
    fgz, ogz = tmp_path / "fast.csv.gz", tmp_path / "orig.csv.gz"
    F.log2_eset(str(inp), str(fgz))
    _run_orig_main(inp, ogz)
    with gzip.open(fgz, "rb") as a, gzip.open(ogz, "rb") as b:
        assert hashlib.sha256(a.read()).hexdigest() == hashlib.sha256(b.read()).hexdigest()


def test_nonnumeric_coercion_bytes_identical(tmp_path):
    df = _matrix(n=20)
    raw = df.to_csv()
    lines = raw.splitlines()
    cells = lines[1].split(",")
    cells[2] = "not_a_number"          # one object cell -> NaN coercion path
    lines[1] = ",".join(cells)
    inp = tmp_path / "in_obj.csv"
    inp.write_text("\n".join(lines) + "\n")
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    F.log2_eset(str(inp), str(fast_out), verbose=True)
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)
    text = fast_out.read_text().splitlines()
    assert text[1].split(",")[2] == ""   # NaN -> empty token, as upstream


def test_negative_values_warning_continues(tmp_path):
    df = _matrix(n=15)
    df.iloc[0, 0] = -0.5                 # -1 <= min < 0 -> warning, continue
    inp = tmp_path / "in_neg.csv"
    df.to_csv(inp)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    F.log2_eset(str(inp), str(fast_out), verbose=True)
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)


# ---------------------------------------------------------------------------
# error semantics (ORIGINAL sys.exit(1) -> ValueError with the same message)
# ---------------------------------------------------------------------------
def test_min_below_minus_one_errors(tmp_path):
    df = _matrix(n=12)
    df.iloc[3, 2] = -2.0
    inp = tmp_path / "in_bad.csv"
    df.to_csv(inp)
    with pytest.raises(ValueError, match=r"log2\(x\+1\) is undefined for x < -1"):
        F.log2_eset(str(inp), str(tmp_path / "out.csv"))
    assert not (tmp_path / "out.csv").exists()
    # the ORIGINAL exits(1) on the same input; engine='original' keeps that
    with pytest.raises(SystemExit) as ei:
        F.log2_eset(str(inp), str(tmp_path / "out2.csv"), engine="original")
    assert ei.value.code == 1


def test_empty_matrix_errors(tmp_path):
    inp = tmp_path / "empty.csv"
    inp.write_text("gene,a,b\n")        # header only -> df.empty
    with pytest.raises(ValueError, match="empty after reading"):
        F.log2_eset(str(inp), str(tmp_path / "out.csv"))


def test_engine_original_writes_same_bytes(tmp_path):
    df = _matrix(n=18)
    inp = tmp_path / "in.csv"
    df.to_csv(inp)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    assert F.log2_eset(str(inp), str(fast_out)) is not None
    assert F.log2_eset(str(inp), str(orig_out), engine="original") is None
    assert _sha(fast_out) == _sha(orig_out)
