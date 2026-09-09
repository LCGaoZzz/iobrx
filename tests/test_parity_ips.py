"""IPS parity: synthetic-input gates against the ORIGINAL iobrpy workflow.

Fast gates (no official data download needed): the accelerated
``iobrx._fast.ips_fast.ips`` path must be byte-identical (output-file
sha256) to ``iobrpy.workflow.IPS.main()`` (argv-bridged — upstream exposes
the stage only as CLI main()) on small synthetic matrices that exercise
every branch:

* log2-scale input (``max <= 100`` passes through) and counts-scale input
  (``max > 100`` triggers the ORIGINAL ``np.log2(x+1)`` heuristic);
* ``.csv`` / ``.tsv`` input separator rules and ``.csv`` / ``.tsv`` output
  separator rules (extension-driven, unknown ext -> comma);
* missing single genes (group still present -> skipna group mean, warning
  text path) ;
* the BUG-COMPAT positional-slice crash: a wholly missing EC or SC block
  makes ``np.nanmean`` of an empty slice NaN and ``int(round(NaN...))``
  raises ``ValueError: cannot convert float NaN to integer`` — asserted
  identically for fast and ORIGINAL;
* result-frame contract: column order ``ID, MHC_IPS, EC_IPS, SC_IPS,
  CP_IPS, AZ_IPS, IPS_IPS`` (upstream EC/SC-before-CP reorder), float
  columns round(6), ``IPS_IPS`` int (banker's rounding statement kept
  verbatim);
* DataFrame input (iobrx extension) equals the file path when values
  round-trip CSV-exactly;
* engine='original' routes to the untouched upstream main().

The official-data byte gate (stad symbol-TPM 54,658x10 -> 10x6 CSV,
sha256 ``793b4718...``) lives in the campaign scratch (parity_gates.py)
and the blind benchmark pass; it needs the frozen IOBRpy baseline output,
so it is not part of this fast file.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

from iobrx._fast import ips_fast as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha(path) -> str:
    with open(str(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _run_orig_main(inp, out):
    """ORIGINAL CLI main() via argv bridge (upstream has no API function)."""
    import importlib

    mod = importlib.import_module("iobrpy.workflow.IPS")
    saved = sys.argv
    try:
        sys.argv = ["IPS", "--input", str(inp), "--output", str(out)]
        mod.main()
    finally:
        sys.argv = saved


def _ips_genes():
    return F._ips_genes()


def _full_panel_matrix(n_samples=4, scale="log2", seed=7):
    """All 160 unique IPS resource symbols; values round-trip CSV-exactly."""
    genes = list(dict.fromkeys(_ips_genes()["GENE"]))
    rng = np.random.default_rng(seed)
    if scale == "log2":
        vals = np.round(rng.random((len(genes), n_samples)) * 50, 6)
    else:  # counts scale -> triggers the max>100 log2 heuristic
        vals = np.round(rng.random((len(genes), n_samples)) * 5000, 4)
    cols = [f"S{i}" for i in range(n_samples)]
    return pd.DataFrame(vals, index=genes, columns=cols)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def test_resource_cache_is_singleton():
    assert F._ips_genes() is F._ips_genes()
    assert _ips_genes().shape == (162, 4)


def test_csv_log2_scale_bytes_identical(tmp_path):
    df = _full_panel_matrix()
    inp = tmp_path / "eset.csv"
    df.to_csv(inp)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    res = F.ips(str(inp), str(fast_out))
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)
    assert list(res.columns) == ["ID", "MHC_IPS", "EC_IPS", "SC_IPS",
                                 "CP_IPS", "AZ_IPS", "IPS_IPS"]
    assert res["IPS_IPS"].dtype.kind in "iu"
    # ID column = sample order preserved
    assert list(res["ID"]) == list(df.columns)


def test_tsv_counts_log2_heuristic_bytes_identical(tmp_path):
    df = _full_panel_matrix(n_samples=3, scale="counts")
    assert df.values.max() > 100  # heuristic WILL fire
    inp = tmp_path / "eset.tsv"
    df.to_csv(inp, sep="\t")
    fast_out, orig_out = tmp_path / "fast.tsv", tmp_path / "orig.tsv"
    F.ips(str(inp), str(fast_out))
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)
    # tsv output really is tab-separated
    assert "\t" in fast_out.read_text().splitlines()[0]


def test_unknown_extension_defaults_to_comma(tmp_path):
    df = _full_panel_matrix(n_samples=2)
    inp = tmp_path / "eset.dat"          # unknown ext -> read as comma
    df.to_csv(inp)                        # comma content
    fast_out, orig_out = tmp_path / "f.dat", tmp_path / "o.dat"
    F.ips(str(inp), str(fast_out))
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)
    assert "," in fast_out.read_text().splitlines()[0]


def test_missing_genes_within_groups_bugcompat(tmp_path):
    """Drop the last gene of every multi-gene NAME group: all 26 groups
    stay present, skipna group means apply, output stays byte-identical."""
    g = _ips_genes()
    drop = []
    for name, grp in g.groupby("NAME", sort=False):
        uniq = list(dict.fromkeys(grp["GENE"]))
        if len(uniq) >= 2:
            drop.append(uniq[-1])
    df = _full_panel_matrix(n_samples=3).drop(index=set(drop))
    inp = tmp_path / "eset.csv"
    df.to_csv(inp)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    res = F.ips(str(inp), str(fast_out), verbose=True)  # warning text path
    _run_orig_main(inp, orig_out)
    assert _sha(fast_out) == _sha(orig_out)
    assert res["IPS_IPS"].notna().all()


def test_missing_block_positional_slice_crash(tmp_path):
    """BUG-COMPAT: no EC/SC genes at all -> empty wg[20:24]/wg[24:26] ->
    nanmean NaN -> int(round(NaN*10/3)) ValueError, identically on both."""
    g = _ips_genes()
    ec_sc = set(g[g["CLASS"].isin(["EC", "SC"])]["GENE"].unique())
    df = _full_panel_matrix(n_samples=2)
    df = df.loc[[i for i in df.index if i not in ec_sc]]
    inp = tmp_path / "eset.csv"
    df.to_csv(inp)
    with pytest.warns(RuntimeWarning):        # "Mean of empty slice" (numpy)
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            F.ips(str(inp), str(tmp_path / "fast.csv"))
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            _run_orig_main(inp, tmp_path / "orig.csv")
    assert not (tmp_path / "fast.csv").exists()


def test_dataframe_input_matches_file_path(tmp_path):
    """iobrx extension: DataFrame input. Values round-trip CSV-exactly
    (6-decimal round), so the file path and the frame path must agree
    bit-for-bit (pandas CSV float parse drift only affects 17-digit text)."""
    df = _full_panel_matrix(n_samples=3)
    inp = tmp_path / "eset.csv"
    df.to_csv(inp)
    via_file = F.ips(str(inp), None)
    via_df = F.ips(df, None)
    pd.testing.assert_frame_equal(via_file, via_df)


def test_output_file_none_returns_frame_only():
    df = _full_panel_matrix(n_samples=2)
    res = F.ips(df, None)
    assert res.shape == (2, 7)


def test_engine_original_writes_same_bytes(tmp_path):
    df = _full_panel_matrix(n_samples=2)
    inp = tmp_path / "eset.csv"
    df.to_csv(inp)
    fast_out, orig_out = tmp_path / "fast.csv", tmp_path / "orig.csv"
    assert F.ips(str(inp), str(fast_out)) is not None
    assert F.ips(str(inp), str(orig_out), engine="original") is None
    assert _sha(fast_out) == _sha(orig_out)


def test_engine_original_requires_paths():
    with pytest.raises(ValueError, match="file path input"):
        F.ips(_full_panel_matrix(n_samples=1), None, engine="original")
