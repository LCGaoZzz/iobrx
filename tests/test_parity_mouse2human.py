"""mouse2human parity: synthetic-input gates against the ORIGINAL iobrpy workflow.

Fast gates (no official data download needed): the accelerated
``iobrx._fast.mouse2human_fast.mouse2human`` path must be byte-identical
(output-file sha256; ``.gz`` outputs compared decompressed — gzip embeds
an mtime, same property upstream) to
``iobrpy.workflow.mouse2human_eset.main()`` (argv-bridged) on small
synthetic mouse matrices drawn from the REAL packaged ``mus_human.pkl``
mapping, exercising every branch:

* matrix mode (``is_matrix=True``) with mouse-symbol index;
* table mode: ORIGINAL ``_remove_duplicate_genes`` (row-mean score,
  descending quicksort, keep first per symbol) incl. duplicated symbols
  with a deterministic higher-mean winner;
* one mouse -> many human symbols (row expansion), many mouse -> one
  human symbol (highest-row-mean selection, NOT averaging), unmapped
  mouse symbols silently dropped;
* the anno_eset tail filters via 1:1 mapped rows: all-zero row dropped,
  all-NA row dropped, first-sample-column-NA row dropped;
* separator inference (``.tsv`` -> tab; blank top-left header cell of the
  MANUAL writer) and ``.gz`` output compression;
* missing ``column_of_symbol`` in table mode -> ValueError with the
  ORIGINAL message;
* verbose gating: quiet by default (the anno_eset info prints are
  captured), ORIGINAL lines reproduced under ``verbose=True``;
* engine='original' routes to the untouched upstream CLI main().

The official-data byte gate (2,000 mouse symbols x 6 samples matrix-mode
fixture -> human_matrix.csv, sha256 ``174aec77...``) lives in the campaign
scratch (parity_gates.py) and the blind benchmark pass; it needs the
frozen IOBRpy baseline output, so it is not part of this fast file.
"""
from __future__ import annotations

import gzip
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest

from iobrx._fast import mouse2human_fast as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha(path) -> str:
    with open(str(path), "rb") as f:
        return _sha_bytes(f.read())


def _sha_decompressed(path) -> str:
    with gzip.open(str(path), "rb") as f:
        return _sha_bytes(f.read())


def _run_orig_main(argv_tail):
    import importlib

    mod = importlib.import_module("iobrpy.workflow.mouse2human_eset")
    saved = sys.argv
    try:
        sys.argv = ["mouse2human_eset"] + argv_tail
        mod.main()
    finally:
        sys.argv = saved


def _mapping():
    return F._load_mus_human_df()


def _pick_symbols():
    """Deterministic picks from the REAL mapping:
    one2many (mouse row count >= 2), many2one pair (two mouse symbols ->
    one human symbol), and three 1:1 pairs for the tail filters."""
    mh = _mapping()
    mus_counts = mh["gene_symbol_mus"].value_counts()
    hum_counts = mh["gene_symbol_human"].value_counts()
    one2many = sorted(mus_counts[mus_counts >= 2].index)[0]
    # many2one: a human symbol reached from >= 2 DIFFERENT mouse symbols
    grp = mh.groupby("gene_symbol_human")["gene_symbol_mus"].nunique()
    hum = sorted(grp[grp >= 2].index)[0]
    many2one = sorted(mh.loc[mh["gene_symbol_human"] == hum, "gene_symbol_mus"].unique())[:2]
    # 1:1 pairs: mouse appears once AND its human appears once
    oneto = mh[(mh["gene_symbol_mus"].map(mus_counts) == 1)
               & (mh["gene_symbol_human"].map(hum_counts) == 1)]
    pairs = [(r.gene_symbol_mus, r.gene_symbol_human)
             for r in oneto.head(3).itertuples()]
    return one2many, many2one, pairs


def _fixture_matrix(seed=3):
    """Mouse matrix covering every semantic branch (values round-trip
    CSV-exactly at 4 decimals)."""
    one2many, many2one, pairs = _pick_symbols()
    rng = np.random.default_rng(seed)
    idx = [one2many] + many2one + [p[0] for p in pairs] + ["ZZZ_UNMAPPABLE_99"]
    vals = np.round(rng.random((len(idx), 4)) * 500 + 1.0, 4)
    # pairs[0] -> all-zero row (dropped); pairs[1] -> all-NA (dropped);
    # pairs[2] -> NaN in the FIRST sample column (dropped)
    df = pd.DataFrame(vals, index=idx, columns=["M1", "M2", "M3", "M4"])
    base = 1 + len(many2one)
    df.iloc[base + 0, :] = 0.0                 # pairs[0]: all zero
    df.iloc[base + 1, :] = np.nan              # pairs[1]: all NA
    df.iloc[base + 2, 0] = np.nan              # pairs[2]: NA in first column
    # many2one winner: make the SECOND mouse row's mean strictly higher
    df.iloc[1, :] = np.round(df.iloc[1].to_numpy() + 100.0, 4)
    df.iloc[2, :] = np.round(df.iloc[2].to_numpy() * 0.1, 4)
    return df, one2many, many2one, pairs


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def test_mapping_cache_is_singleton():
    assert F._load_mus_human_df() is F._load_mus_human_df()
    assert {"gene_symbol_mus", "gene_symbol_human"} <= set(_mapping().columns)


def test_matrix_mode_bytes_identical(tmp_path):
    df, one2many, many2one, pairs = _fixture_matrix()
    inp = tmp_path / "mouse.csv"
    df.to_csv(inp)                       # pandas writes blank first cell (index name None)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    res = F.mouse2human(str(inp), str(fast_out), is_matrix=True)
    _run_orig_main(["-i", str(inp), "-o", str(orig_out), "--is_matrix"])
    assert _sha(fast_out) == _sha(orig_out)

    text = fast_out.read_text().splitlines()
    assert text[0].startswith(",")       # manual writer: blank top-left cell
    out = pd.read_csv(fast_out, index_col=0)
    humans = set(out.index)
    mh = _mapping()
    # one mouse -> many human: EVERY human symbol of that mouse is present
    assert set(mh.loc[mh["gene_symbol_mus"] == one2many, "gene_symbol_human"]) <= humans
    # many mouse -> one human: present exactly once, values = higher-mean row
    hum = mh.loc[mh["gene_symbol_mus"] == many2one[0], "gene_symbol_human"].iloc[0]
    rows = out.loc[[hum]]
    assert rows.shape[0] == 1
    np.testing.assert_allclose(rows.to_numpy(),
                               df.loc[many2one[0]].to_numpy()[None, :], rtol=0, atol=0)
    # unmapped mouse symbol silently dropped
    assert "ZZZ_UNMAPPABLE_99" not in humans
    # tail filters: all-zero / all-NA / first-col-NA human rows dropped
    for _, h in pairs:
        assert h not in humans
    # upstream anno_eset sorts the merged frame by _score DESCENDING when
    # duplicate human symbols exist (quicksort; verified tie-compatible in
    # anno_eset_fast) -> survivor row means are non-increasing
    means = out.mean(axis=1)
    assert (means.diff().dropna() <= 1e-12).all()
    pd.testing.assert_frame_equal(res, out.rename_axis("symbol"))


def test_table_mode_duplicates_bytes_identical(tmp_path):
    df, *_ = _fixture_matrix()
    sym_col = list(df.index) + list(df.index[:5])   # 5 duplicated symbols
    tbl = df.loc[sym_col].reset_index().rename(columns={"index": "SYMBOL"})
    # give the duplicate copies strictly LOWER means (deterministic winner)
    tbl.iloc[len(df):, 1:] = tbl.iloc[len(df):, 1:] * 0.01
    inp = tmp_path / "table.csv"
    tbl.to_csv(inp, index=False)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    F.mouse2human(str(inp), str(fast_out), is_matrix=False,
                  column_of_symbol="SYMBOL")
    _run_orig_main(["-i", str(inp), "-o", str(orig_out),
                    "--column_of_symbol", "SYMBOL"])
    assert _sha(fast_out) == _sha(orig_out)


def test_table_mode_missing_symbol_column_raises(tmp_path):
    df, *_ = _fixture_matrix()
    tbl = df.reset_index().rename(columns={"index": "SYMBOL"})
    inp = tmp_path / "table.csv"
    tbl.to_csv(inp, index=False)
    with pytest.raises(ValueError, match="column_of_symbol must be specified"):
        F.mouse2human(str(inp), str(tmp_path / "o.csv"), is_matrix=False)
    # ORIGINAL raises the same ValueError (inside its main())
    with pytest.raises(ValueError, match="column_of_symbol must be specified"):
        _run_orig_main(["-i", str(inp), "-o", str(tmp_path / "o2.csv")])


def test_tsv_separator_inference(tmp_path):
    df, *_ = _fixture_matrix()
    inp = tmp_path / "mouse.tsv"
    df.to_csv(inp, sep="\t")
    fast_out, orig_out = tmp_path / "fast.tsv", tmp_path / "orig.tsv"
    F.mouse2human(str(inp), str(fast_out), is_matrix=True)
    _run_orig_main(["-i", str(inp), "-o", str(orig_out), "--is_matrix"])
    assert _sha(fast_out) == _sha(orig_out)
    head = fast_out.read_text().splitlines()[0]
    assert head.startswith(",\t") or "\t" in head
    assert head.split("\t")[0] == ""      # blank first cell, tab separated


def test_gz_output_decompressed_equal(tmp_path):
    df, *_ = _fixture_matrix()
    inp = tmp_path / "mouse.csv"
    df.to_csv(inp)
    fgz, ogz = tmp_path / "fast.csv.gz", tmp_path / "orig.csv.gz"
    F.mouse2human(str(inp), str(fgz), is_matrix=True)
    _run_orig_main(["-i", str(inp), "-o", str(ogz), "--is_matrix"])
    # gzip bytes embed mtime (upstream too) -> compare decompressed content
    assert _sha_decompressed(fgz) == _sha_decompressed(ogz)


def test_dataframe_input_matches_file_path(tmp_path):
    df, *_ = _fixture_matrix()
    inp = tmp_path / "mouse.csv"
    df.to_csv(inp)                        # 4-decimal values round-trip exactly
    via_file = F.mouse2human(str(inp), None, is_matrix=True)
    via_df = F.mouse2human(df, None, is_matrix=True)
    pd.testing.assert_frame_equal(via_file, via_df)


def test_verbose_gating(tmp_path, capsys):
    df, *_ = _fixture_matrix()
    inp = tmp_path / "mouse.csv"
    df.to_csv(inp)
    F.mouse2human(str(inp), str(tmp_path / "q.csv"), is_matrix=True)
    quiet = capsys.readouterr().out
    assert "Annotation DataFrame shape" not in quiet
    assert "Converted matrix saved to" not in quiet
    F.mouse2human(str(inp), str(tmp_path / "v.csv"), is_matrix=True, verbose=True)
    loud = capsys.readouterr().out
    assert "Annotation DataFrame shape" in loud
    assert "Converted matrix saved to" in loud
    assert _sha(tmp_path / "q.csv") == _sha(tmp_path / "v.csv")


def test_engine_original_writes_same_bytes(tmp_path):
    df, *_ = _fixture_matrix()
    inp = tmp_path / "mouse.csv"
    df.to_csv(inp)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    assert F.mouse2human(str(inp), str(fast_out), is_matrix=True) is not None
    assert F.mouse2human(str(inp), str(orig_out), is_matrix=True,
                         engine="original") is None
    assert _sha(fast_out) == _sha(orig_out)


def test_engine_original_requires_paths():
    df, *_ = _fixture_matrix()
    with pytest.raises(ValueError, match="file path input"):
        F.mouse2human(df, None, is_matrix=True, engine="original")
