"""Official-data parity gates (pytest.mark.full).

Runs the 11 official IOBRpy workflow stages on the official test data and
compares the iobrx outputs against the ORIGINAL iobrpy implementations
executed on the fly. Expected: max abs diff 0.0 everywhere except CIBERSORT's
``P-value`` column (the ORIGINAL draws OS entropy for its permutations and is
not reproducible run-to-run by design; the formula and its 1/perm granularity
are identical).

Data resolution order for each frame:
  1. ``$IOBRX_TESTDATA/<name>.parquet``
  2. ``$IOBRX_TESTDATA/<name>.rda`` (first object; via pyreadr)
  3. download ``<name>.rda`` from the IOBR GitHub release ``data-v1.0``
     (https://github.com/IOBR/IOBR/releases/tag/data-v1.0) into a local cache

``eset_stad_symbol`` is not distributed (404 upstream): it is derived with the
ORIGINAL ``anno_eset`` from ``eset_stad`` x ``anno_grch38`` exactly as the
official baseline defines it, then shared by both sides.

Enable with::

    pytest -q -m full            # or  pytest -q --run-full
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.full

RELEASE = "https://github.com/IOBR/IOBR/releases/download/data-v1.0"
_CACHE = os.path.join(tempfile.gettempdir(), "iobrx_official_data")


def _load(name: str) -> pd.DataFrame:
    env = os.environ.get("IOBRX_TESTDATA")
    if env:
        pq = os.path.join(env, f"{name}.parquet")
        if os.path.exists(pq):
            return pd.read_parquet(pq)
        rda = os.path.join(env, f"{name}.rda")
        if os.path.exists(rda):
            import pyreadr

            return next(iter(pyreadr.read_r(rda).values()))
    pq = os.path.join(_CACHE, f"{name}.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    import urllib.request

    import pyreadr

    os.makedirs(_CACHE, exist_ok=True)
    rda = os.path.join(_CACHE, f"{name}.rda")
    urllib.request.urlretrieve(f"{RELEASE}/{name}.rda", rda)
    df = next(iter(pyreadr.read_r(rda).values()))
    df.to_parquet(pq)
    return df


@pytest.fixture(scope="module")
def data():
    from iobrpy.workflow.anno_eset import anno_eset as orig_anno

    imvigor = _load("imvigor210_eset")
    stad_raw = _load("eset_stad")
    anno = _load("anno_grch38")
    # derived exactly as the official baseline: ORIGINAL anno_eset('mean')
    stad_symbol = orig_anno(stad_raw.copy(), anno.copy(), symbol="symbol",
                            probe="id", method="mean")
    return imvigor, stad_raw, anno, stad_symbol


def _num_diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    num = b.select_dtypes(include=[np.number]).columns
    d = a[num].to_numpy(dtype=np.float64) - b[num].to_numpy(dtype=np.float64)
    obj = [c for c in b.columns if c not in num]
    obj_ok = all(a[c].astype(str).equals(b[c].astype(str)) for c in obj) if obj else True
    return {
        "index_equal": list(a.index) == list(b.index),
        "columns_equal": list(a.columns) == list(b.columns),
        "numeric_cells": int(d.size),
        "cells_bit_identical": int((d == 0).sum()),
        "max_abs_diff": float(np.abs(d).max()) if d.size else 0.0,
        "object_cols_equal": bool(obj_ok),
    }


def _assert_zero(res: dict, ctx: str):
    assert res["index_equal"] and res["columns_equal"] and res["object_cols_equal"], ctx
    assert res["max_abs_diff"] == 0.0 and res["cells_bit_identical"] == res["numeric_cells"], (
        f"{ctx}: {res}"
    )


@pytest.mark.parametrize("method", ["pca", "zscore", "ssgsea", "integration"])
def test_sig_scores(data, method):
    from iobrpy.workflow.calculate_sig_score import calculate_sig_score as orig

    import iobrx

    imvigor, *_ = data
    fast = iobrx.calculate_sig_score(imvigor.copy(), "signature_collection", method,
                                     mini_gene_count=3, adjust_eset=True, n_threads=8)
    slow = orig(imvigor.copy(), ["signature_collection"], method, 3, True, 1)
    _assert_zero(_num_diff(fast, slow), f"sig_{method}")


def test_anno_eset(data):
    from iobrpy.workflow.anno_eset import anno_eset as orig

    import iobrx

    _, stad_raw, anno, _ = data
    fast = iobrx.anno_eset(stad_raw.copy(), anno.copy(), symbol="symbol",
                           probe="id", method="mean")
    slow = orig(stad_raw.copy(), anno.copy(), symbol="symbol", probe="id",
                method="mean")
    _assert_zero(_num_diff(fast, slow), "anno_eset")


def test_cibersort(data):
    from iobrpy.workflow import cibersort as orig

    import iobrx

    _, _, _, stad_symbol = data
    fast = iobrx.cibersort(stad_symbol.copy(), perm=100, QN=True, n_threads=8)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mix.csv")
        stad_symbol.to_csv(path)                      # official input contract
        slow = orig.cibersort(path, perm=100, QN=True)
    nonp = [c for c in slow.columns if c != "P-value"]
    _assert_zero(_num_diff(fast[nonp], slow[nonp]), "cibersort(excl P-value)")
    pv = _num_diff(fast[["P-value"]], slow[["P-value"]])
    assert 0.0 <= float(np.abs(fast["P-value"] - slow["P-value"]).max()) <= 1.0
    assert pv["numeric_cells"] == fast.shape[0]


def test_epic(data):
    from iobrpy.workflow.epic import EPIC as orig

    import iobrx

    _, _, _, stad_symbol = data
    fast = iobrx.epic(stad_symbol.copy())
    slow = orig(bulk=stad_symbol.copy(), reference=iobrx._epic_tref())
    for key in ("mRNAProportions", "cellFractions", "fit_gof"):
        _assert_zero(_num_diff(fast[key], slow[key]), f"epic/{key}")


def test_quantiseq(data):
    from iobrpy.workflow.quantiseq import deconvolute_quantiseq_default as orig

    import iobrx

    _, _, _, stad_symbol = data
    qs_data = iobrx._quantiseq_data()
    fast = iobrx.quantiseq(stad_symbol.copy(), data=qs_data, arrays=False,
                           tumor=False, mRNAscale=True, method="lsei",
                           rmgenes="default")
    slow = orig(mix=stad_symbol.copy(), data=qs_data, arrays=False, tumor=False,
                mRNAscale=True, method="lsei", rmgenes="default")
    _assert_zero(_num_diff(fast, slow), "quantiseq")


def test_mcpcounter(data):
    from iobrpy.workflow.mcpcounter import MCPcounter_estimate as orig

    import iobrx

    _, _, _, stad_symbol = data
    fast = iobrx.mcpcounter(stad_symbol.copy(), features_type="HUGO_symbols")
    slow = orig(expression=stad_symbol.copy(), features_type="HUGO_symbols")
    _assert_zero(_num_diff(fast, slow), "mcpcounter")


def test_estimate(data):
    from iobrpy.workflow.estimate import estimate_score as orig

    import iobrx

    _, _, _, stad_symbol = data
    fast = iobrx.estimate_score(stad_symbol.copy(), platform="affy")
    slow = orig(input_df=stad_symbol.copy(), platform="affy")
    _assert_zero(_num_diff(fast, slow), "estimate")


def test_count2tpm(data):
    from iobrpy.workflow.count2tpm import count2tpm as orig

    import iobrx

    _, stad_raw, anno, _ = data
    # official call: anno passed but anno_gc_vm32=None -> the packaged-table
    # branch runs on BOTH sides (the upstream quirk), exactly like the gate
    fast = iobrx.count2tpm(stad_raw.copy(), anno_grch38=anno.copy(),
                           anno_gc_vm32=None, idType="Ensembl", org="hsa",
                           source="local", check_data=True, remove_version=True)
    slow = orig(stad_raw.copy(), anno.copy(), None, "Ensembl", "hsa", "local",
                None, "id", "symbol", "eff_length", True, True)
    _assert_zero(_num_diff(fast, slow), "count2tpm")
