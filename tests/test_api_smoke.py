"""Synthetic smoke tests: every public iobrx function runs, returns the
expected shape, and matches the ORIGINAL iobrpy implementation bit-for-bit
on inputs where a quick comparison is feasible.

No network, no official data downloads; runs in well under two minutes.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from importlib.resources import files

import iobrx

RES = files("iobrx").joinpath("_resources")


# ---------------------------------------------------------------------------
# shared synthetic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260907)


@pytest.fixture(scope="module")
def lm22():
    return pd.read_csv(RES.joinpath("lm22.txt"), sep=r"\s+", engine="python", index_col=0)


@pytest.fixture(scope="module")
def sig_collection():
    return pd.read_pickle(str(RES.joinpath("calculate_data.pkl")))["signature_collection"]


@pytest.fixture(scope="module")
def anno_grch38_df():
    """The packaged human annotation as a DataFrame (dataframe-branch input)."""
    import pickle

    with open(str(RES.joinpath("anno_eset.pkl")), "rb") as f:
        d = pickle.load(f)
    return pd.DataFrame(d["anno_grch38"])


def _lognormal_matrix(rng, index, columns, scale=8.0):
    vals = rng.lognormal(mean=0.0, sigma=1.0, size=(len(index), len(columns))) * scale
    return pd.DataFrame(vals, index=pd.Index(list(index)), columns=list(columns))


# ---------------------------------------------------------------------------
# cibersort
# ---------------------------------------------------------------------------
def test_cibersort_shape_and_original_parity(rng, lm22):
    orig = pytest.importorskip("iobrpy.workflow.cibersort")

    # real LM22 genes so the overlap is healthy; 4 synthetic mixtures
    genes = lm22.index[:400]
    w = rng.dirichlet(np.ones(lm22.shape[1]), size=4)          # 4 x 22
    base = lm22.loc[genes].to_numpy()                          # 400 x 22
    mix_vals = np.log2(base @ w.T + 1.0) + rng.normal(0, 0.05, (len(genes), 4))
    mix = pd.DataFrame(mix_vals, index=genes, columns=[f"S{i}" for i in range(4)])

    fast = iobrx.cibersort(mix, perm=5, QN=True, n_threads=4)
    assert fast.shape == (4, 22 + 3)
    assert list(fast.columns[-3:]) == ["P-value", "Correlation", "RMSE"]
    assert list(fast.index) == list(mix.columns)
    assert set(fast.iloc[:, :22].dtypes.astype(str)) == {"float32"}  # dtype parity

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mix.csv")
        mix.to_csv(path)                                       # official input contract
        slow = orig.cibersort(path, perm=5, QN=True)
    # weights / Correlation / RMSE must be bit-identical; P-value is unseeded
    # in the ORIGINAL (OS entropy) and excluded by design
    nonp = [c for c in fast.columns if c != "P-value"]
    pd.testing.assert_frame_equal(fast[nonp], slow[nonp], check_exact=True)
    assert ((fast["P-value"] >= 0) & (fast["P-value"] <= 1)).all()
    assert (fast["P-value"] * 5 == np.round(fast["P-value"] * 5)).all()  # 1/perm grid


def test_cibersort_absolute_shape(rng, lm22):
    genes = lm22.index[:300]
    mix = _lognormal_matrix(rng, genes, ["A", "B", "C"])
    out = iobrx.cibersort(mix, perm=2, QN=True, absolute=True, abs_method="sig.score",
                          n_threads=2)
    assert out.shape[1] == 22 + 4
    assert "Absolute_score_(sig_score)" in out.columns


# ---------------------------------------------------------------------------
# signature scoring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["pca", "zscore", "ssgsea", "integration"])
def test_calculate_sig_score_parity(method, rng, sig_collection):
    orig = pytest.importorskip("iobrpy.workflow.calculate_sig_score").calculate_sig_score

    genes = sorted({g for gl in sig_collection.values() for g in gl})
    eset = _lognormal_matrix(rng, genes[:2500], [f"P{i}" for i in range(12)])

    fast = iobrx.calculate_sig_score(eset, "signature_collection", method,
                                     mini_gene_count=3, adjust_eset=True, n_threads=2)
    slow = orig(eset.copy(), ["signature_collection"], method, 3, True, 1)
    assert fast.shape == slow.shape
    assert list(fast.columns) == list(slow.columns)
    assert list(fast["ID"]) == list(slow["ID"])
    assert fast.equals(slow)                                    # bit-identical


# ---------------------------------------------------------------------------
# count2tpm
# ---------------------------------------------------------------------------
def test_count2tpm_parity_and_packaged_mode(rng, anno_grch38_df):
    orig = pytest.importorskip("iobrpy.workflow.count2tpm").count2tpm

    anno = anno_grch38_df
    ids = anno["id"].dropna().sample(2000, random_state=7).tolist()
    counts = pd.DataFrame(
        rng.poisson(lam=50.0, size=(len(ids), 6)).astype(float),
        index=pd.Index(ids), columns=[f"C{i}" for i in range(6)],
    )

    fast = iobrx.count2tpm(counts.copy(), anno_grch38=anno.copy(), check_data=True,
                           remove_version=False)
    slow = orig(counts.copy(), anno.copy(), None, "Ensembl", "hsa", "local",
                None, "id", "symbol", "eff_length", True, False)
    pd.testing.assert_frame_equal(fast, slow, check_exact=True)

    # packaged-annotation branch (anno not a DataFrame -> packaged tables)
    pkg = iobrx.count2tpm(counts.copy(), check_data=False, remove_version=False)
    assert isinstance(pkg, pd.DataFrame) and pkg.shape[1] == 6
    assert pkg.index.name is None and pkg.notna().all().all()


# ---------------------------------------------------------------------------
# quanTIseq
# ---------------------------------------------------------------------------
def test_quantiseq_alias_builder_and_run(rng):
    _qs = pytest.importorskip("iobrpy.workflow.quantiseq")
    from iobrx._fast.quantiseq_fast import (
        _vectorized_build_hgnc_alias_map, _orig_build, deconvolute_quantiseq_default,
    )

    data = pd.read_pickle(str(RES.joinpath("quantiseq_data.pkl")))
    hgnc = data["HGNC_genenames_20170418"]

    vec = _vectorized_build_hgnc_alias_map(hgnc)
    ref = _orig_build(hgnc)
    assert vec == ref                                            # same key->value map
    assert list(vec.items()) == list(ref.items())                # same insertion order

    sig = data["TIL10_signature"]
    til_genes = sig.index.astype(str).tolist()      # signature genes are the INDEX
    mix = _lognormal_matrix(rng, til_genes, [f"Q{i}" for i in range(3)])
    out = iobrx.quantiseq(mix.copy(), data=data)
    assert out.shape[0] == 3
    direct = deconvolute_quantiseq_default(mix=mix.copy(), data=data)
    pd.testing.assert_frame_equal(out, direct, check_exact=True)


# ---------------------------------------------------------------------------
# EPIC
# ---------------------------------------------------------------------------
def test_epic_parity(rng):
    orig = pytest.importorskip("iobrpy.workflow.epic").EPIC
    from iobrx._fast.epic_fast import invalidate_cache

    tref = pd.read_pickle(str(RES.joinpath("epic_TRef_BRef.pkl")))["TRef"]
    ref_index = tref["refProfiles"].index
    genes = list(dict.fromkeys(list(tref["sigGenes"]) + list(ref_index[:800])))
    bulk = _lognormal_matrix(rng, genes, [f"E{i}" for i in range(4)])

    invalidate_cache()
    fast = iobrx.epic(bulk.copy())          # reference=None -> packaged TRef
    slow = orig(bulk=bulk.copy(), reference=tref)
    for key in ("mRNAProportions", "cellFractions", "fit_gof"):
        pd.testing.assert_frame_equal(fast[key], slow[key], check_exact=True)


# ---------------------------------------------------------------------------
# MCP-counter
# ---------------------------------------------------------------------------
def test_mcpcounter_parity(rng):
    orig = pytest.importorskip("iobrpy.workflow.mcpcounter").MCPcounter_estimate

    md = pd.read_pickle(str(RES.joinpath("mcp_data.pkl")))
    markers = md["genes"]["HUGO symbols"].astype(str).tolist()
    genes = markers + [f"FILLER{i}" for i in range(80)]
    expr = _lognormal_matrix(rng, genes, [f"M{i}" for i in range(5)])

    fast = iobrx.mcpcounter(expr.copy(), features_type="HUGO_symbols")
    slow = orig(expression=expr.copy(), features_type="HUGO_symbols")
    pd.testing.assert_frame_equal(fast, slow, check_exact=True)
    assert fast.shape[1] == 5


# ---------------------------------------------------------------------------
# ESTIMATE
# ---------------------------------------------------------------------------
def test_estimate_parity(rng):
    orig = pytest.importorskip("iobrpy.workflow.estimate").estimate_score

    common = pd.read_csv(RES.joinpath("common_genes.txt"), sep="\t", header=0, dtype=str)
    import pickle

    with open(str(RES.joinpath("estimate_data.pkl")), "rb") as f:
        si = pickle.load(f)["SI_geneset"]
    si_genes = set(si.iloc[:, 1:].to_numpy().ravel().tolist())
    genes = list(si_genes & set(common["GeneSymbol"]))[:1500]
    genes += list(common["GeneSymbol"].sample(500, random_state=3))
    genes = list(dict.fromkeys(genes))
    expr = _lognormal_matrix(rng, genes, [f"T{i}" for i in range(4)])

    fast = iobrx.estimate_score(expr.copy(), platform="affymetrix")
    slow = orig(input_df=expr.copy(), platform="affymetrix")
    pd.testing.assert_frame_equal(fast, slow, check_exact=True)
    assert "TumorPurity" in fast.index and "ESTIMATEScore" in fast.index


# ---------------------------------------------------------------------------
# anno_eset
# ---------------------------------------------------------------------------
def test_anno_eset_parity_and_builtin(rng, anno_grch38_df):
    orig = pytest.importorskip("iobrpy.workflow.anno_eset").anno_eset

    sub = anno_grch38_df.head(4000)
    probes = sub["id"].astype(str).tolist()
    eset = _lognormal_matrix(rng, probes, [f"S{i}" for i in range(6)])

    fast = iobrx.anno_eset(eset.copy(), sub.copy(), symbol="symbol", probe="id",
                           method="mean")
    slow = orig(eset.copy(), sub.copy(), symbol="symbol", probe="id", method="mean")
    pd.testing.assert_frame_equal(fast, slow, check_exact=True)
    assert fast.index.name == "symbol"

    builtin = iobrx.anno_eset(eset.copy(), "anno_grch38", symbol="symbol", probe="id")
    assert isinstance(builtin, pd.DataFrame) and builtin.shape[1] == 6


# ---------------------------------------------------------------------------
# threads helper
# ---------------------------------------------------------------------------
def test_set_threads():
    old = iobrx.get_threads()
    try:
        iobrx.set_threads(3)
        assert iobrx.get_threads() == 3
        assert iobrx.__version__
    finally:
        iobrx.set_threads(None)
        assert iobrx.get_threads() == old
