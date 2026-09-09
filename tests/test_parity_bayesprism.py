"""Parity gate for the accelerated bayesprism pipeline (quick, synthetic).

Compares ``iobrx._fast.bayesprism_fast`` against the ORIGINAL
``iobrpy.bayesprism`` pipeline IN-PROCESS on a small self-contained
synthetic configuration: 134 real gene symbols (taken from the frozen
campaign bulk input; they are guaranteed to pass the bundled
``cleanup_genes`` / ``select_gene_type`` filters), 120 synthetic cells over
8 states / 4 types (the ``key`` type deliberately has a SINGLE state), and
5 synthetic bulk samples built with a fixed ``default_rng`` seed.

The ORIGINAL CLI does NOT expose the Gibbs iteration parameters; both the
ORIGINAL Python API (``prism.Prism.run(gibbs_control=...)``) and the iobrx
API accept ``gibbs_control``, so this test shrinks the chains
(``chain.length=24, burn.in=12, thinning=2``) through that documented
passthrough instead of monkeypatching module constants.

Contract notes
--------------
* ``state_order='legacy'`` reproduces the ORIGINAL ``list(set(states))``
  merge order VERBATIM. Within one process (one PYTHONHASHSEED) the port
  must then be BIT-IDENTICAL to the ORIGINAL: theta / theta_cv / Z_tumor
  values (compared as raw float64 bit patterns, so identical NaN payloads
  count as equal) and the three output CSVs byte-for-byte.
* ``state_order='sorted'`` (the iobrx default) is the determinism FIX for
  the upstream PYTHONHASHSEED dependence (ORIGINAL:
  ``map_ = {cell_type: list(set(states))}`` — set iteration order follows
  the per-process hash seed, so merge sums — and occasionally discrete
  Gibbs draws — differ across processes). Sorted-mode outputs must be
  identical across calls WITHOUT pinning the hash seed. Multi-state merges
  sum in a different order than the legacy run, so the FINAL theta may
  legitimately differ at the documented ULP->discrete-flip level; the
  single-state ``key`` type's Z_tumor (initial posterior, no merge sum,
  hence map_-independent) MUST stay bit-identical across the two modes and
  is asserted so.
* Full-scale parity against the frozen PYTHONHASHSEED=0 gold-standard run
  (byte-identical CSVs in legacy mode, numeric error envelope in sorted
  mode) is verified by the campaign harness on the frozen 20-sample input,
  not by this quick test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 134 symbols from the frozen campaign bulk input (imvigor210 x BP_data
# intersection): all pass cleanup_genes' blacklist and gencode v22
# 'protein_coding' selection, so both sides run on a realistic gene set.
GENES = [
    "CHAF1A", "CDK2", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "CDKN2C",
    "GADD45G", "DBF4", "CHEK1", "DMC1", "TWIST2", "CNN1", "COL4A1",
    "CLDN4", "CLDN3", "CLDN7", "PARP1", "CTGF", "CTLA4", "CTPS1", "DUT",
    "E2F2", "E2F5", "TIGIT", "FANCA", "FAP", "FEN1", "FGFR3", "FOXF1",
    "SMUG1", "ORC6", "BLOC1S1", "SFN", "BLOC1S2", "CD274", "MSH6",
    "ANAPC4", "GZMB", "H2AFX", "HDAC1", "IFNG", "FAM101B", "CXCL10",
    "LAG3", "LIG1", "LOXL2", "MAD2L1", "MCM2", "MCM3", "MCM4", "MCM5",
    "MCM7", "CXCL9", "MNAT1", "NUDT1", "MYC", "NTHL1", "ROR2", "OGG1",
    "ORC1", "PCNA", "PDCD1", "ANAPC11", "PLK1", "FANCI", "PRF1", "NPLOC4",
    "TDP1", "PRKDC", "B2M", "POLD4", "RAD17", "RAD51", "RAD51C", "ACTA2",
    "RB1", "RBL1", "CCND1", "RFC2", "RFC3", "RFC4", "RFC5", "RPA1", "BLM",
    "SOX9", "BRCA1", "TAGLN", "TAP1", "TAP2", "BUB1", "BUB1B", "TFDP1",
    "TGFB3", "TNS1", "TPM1", "ACTG2", "TTK", "TWIST1", "UBE2N", "VEGFA",
    "VIM", "WEE1", "WNT5A", "WNT7B", "XRCC2", "YWHAH", "NEIL1", "SHFM1",
    "PDCD1LG2", "CDC45", "RAD54L", "HAVCR2", "CDC14A", "ADAM19", "CCNA2",
    "CCNA1", "CCNB1", "MBD4", "PKMYT1", "SMC3", "CCNB2", "EXO1", "BUB3",
    "PTTG1", "CD8A", "SH3PXD2A", "ESPL1", "CDK1", "ZEB2", "CDC6", "CDC20",
    "CDC25A", "CDH1",
]

STATES = ["Tumor_A", "T_mem", "T_naive", "B_1", "B_2", "M_1", "M_2", "M_3"]
TYPES = ["Tumor", "T_cells", "T_cells", "B_cells", "B_cells",
         "Myeloid", "Myeloid", "Myeloid"]
KEY = "Tumor"  # single-state key type: Z_tumor must be mode-invariant
N_THREADS = 4
_GENE_GROUP = ["Rb", "Mrp", "other_Rb", "chrM", "MALAT1", "chrX", "chrY"]


def _gibbs() -> dict:
    """Fresh reduced-chain control per run (the ORIGINAL Prism.run MUTATES
    the dict it is given — an upstream quirk — so never share one)."""
    return {"chain.length": 24, "burn.in": 12, "thinning": 2}


def _inputs():
    rng = np.random.default_rng(20260908)
    genes = GENES
    g = len(genes)
    lam = 2.0 + rng.exponential(20.0, size=g)

    n_cells = 120
    ref_arr = rng.poisson(lam=lam[None, :], size=(n_cells, g)).astype(np.int32)
    sc = pd.DataFrame(ref_arr, index=[f"cell_{i}" for i in range(n_cells)],
                      columns=genes)
    csl = [STATES[i % len(STATES)] for i in range(n_cells)]
    ctl = [TYPES[i % len(TYPES)] for i in range(n_cells)]

    n_s = 5
    bulk_arr = rng.poisson(lam=lam[:, None] * 3.0, size=(g, n_s))
    bulk = pd.DataFrame(bulk_arr, index=genes,
                        columns=[f"S{i}" for i in range(n_s)])
    return bulk, sc, csl, ctl


def _run_original(bulk, sc, csl, ctl):
    """The ORIGINAL pipeline steps (iobrpy.bayesprism), reduced chains."""
    from iobrpy.bayesprism import extract as o_extract
    from iobrpy.bayesprism import prism as o_prism
    from iobrpy.bayesprism import process_input as o_pi

    f1 = o_pi.cleanup_genes(sc, input_type="count.matrix", species="hs",
                            gene_group=_GENE_GROUP, exp_cells=5)
    f2 = o_pi.select_gene_type(f1, ["protein_coding"])
    pr = o_prism.Prism.new(reference=f2, input_type="count.matrix",
                           cell_type_labels=list(ctl),
                           cell_state_labels=list(csl),
                           key=KEY, mixture=bulk.T.astype(np.int32),
                           outlier_cut=0.01, outlier_fraction=0.1)
    bp = pr.run(n_cores=N_THREADS, update_gibbs=True,
                gibbs_control=_gibbs(), opt_control={})
    theta = o_extract.get_fraction(bp, which_theta="final",
                                   state_or_type="type").add_suffix("_BayesPrism")
    theta_cv = bp.posterior_theta_f.theta_cv.add_suffix("_BayesPrism")
    z = o_extract.get_exp(bp, state_or_type="type", cell_name=KEY)
    z = z.to_pandas() if hasattr(z, "to_pandas") else pd.DataFrame(z)
    return {"theta": theta, "theta_cv": theta_cv, "Z_tumor": z}


def _run_port(bulk, sc, csl, ctl, state_order, out_dir=None):
    from iobrx._fast.bayesprism_fast import bayesprism as bp_fast

    return bp_fast(bulk, sc_dat=sc, cell_state_labels=csl,
                   cell_type_labels=ctl, key=KEY, out_dir=out_dir,
                   n_threads=N_THREADS, state_order=state_order,
                   gibbs_control=_gibbs())


def _bits(df: pd.DataFrame) -> np.ndarray:
    """Raw float64 bit patterns (so identical NaN payloads compare equal)."""
    return np.ascontiguousarray(df.to_numpy(dtype=np.float64)).view(np.uint64)


def _assert_bit_identical(a: pd.DataFrame, b: pd.DataFrame, name: str):
    assert list(a.index) == list(b.index), f"{name}: index differs"
    assert list(a.columns) == list(b.columns), f"{name}: columns differ"
    va, vb = a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
    same = np.array_equal(
        np.ascontiguousarray(va).view(np.uint64),
        np.ascontiguousarray(vb).view(np.uint64),
    )
    assert same, (
        f"{name}: not bit-identical; max abs diff "
        f"{np.nanmax(np.abs(va - vb)) if va.size else 0.0}"
    )


def _write_csvs(res: dict, d):
    d.mkdir(parents=True, exist_ok=True)
    res["theta"].to_csv(d / "theta.csv")
    res["theta_cv"].to_csv(d / "theta_cv.csv")
    res["Z_tumor"].to_csv(d / "Z_tumor.csv")
    return d


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    bulk, sc, csl, ctl = _inputs()
    orig = _run_original(bulk, sc, csl, ctl)
    legacy = _run_port(bulk, sc, csl, ctl, "legacy")
    sorted1 = _run_port(bulk, sc, csl, ctl, "sorted")
    sorted2 = _run_port(bulk, sc, csl, ctl, "sorted")
    orig_dir = _write_csvs(orig, tmp_path_factory.mktemp("orig"))
    legacy_dir = _write_csvs(legacy, tmp_path_factory.mktemp("legacy"))
    return orig, legacy, sorted1, sorted2, orig_dir, legacy_dir


def test_legacy_bit_identical_to_original(runs):
    orig, legacy, *_ = runs
    for name in ("theta", "theta_cv", "Z_tumor"):
        _assert_bit_identical(orig[name], legacy[name], f"legacy/{name}")


def test_legacy_csv_bytes_identical_to_original(runs):
    *_, orig_dir, legacy_dir = runs
    for fname in ("theta.csv", "theta_cv.csv", "Z_tumor.csv"):
        a = (orig_dir / fname).read_bytes()
        b = (legacy_dir / fname).read_bytes()
        assert a == b, f"{fname}: CSV bytes differ (legacy vs ORIGINAL)"


def test_sorted_mode_deterministic_and_valid(runs):
    _, _, sorted1, sorted2, *_ = runs
    for name in ("theta", "theta_cv", "Z_tumor"):
        _assert_bit_identical(sorted1[name], sorted2[name], f"sorted-rerun/{name}")
    theta = sorted1["theta"]
    arr = theta.to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)
    np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-12)


def test_single_state_key_z_tumor_identical_across_modes(runs):
    """The key type has ONE state, so its merged Z involves no summation:
    Z_tumor comes straight out of the (map_-independent) initial Gibbs and
    must be bit-identical between 'sorted' and 'legacy'."""
    _, legacy, sorted1, *_ = runs
    _assert_bit_identical(legacy["Z_tumor"], sorted1["Z_tumor"],
                          "Z_tumor across state_order modes")


def test_output_contract(runs, tmp_path):
    from iobrx._fast.bayesprism_fast import bayesprism as bp_fast

    bulk, sc, csl, ctl = _inputs()
    res = bp_fast(bulk, sc_dat=sc, cell_state_labels=csl,
                  cell_type_labels=ctl, key=KEY, out_dir=tmp_path,
                  n_threads=N_THREADS, gibbs_control=_gibbs())
    for fname in ("theta.csv", "theta_cv.csv", "Z_tumor.csv"):
        assert (tmp_path / fname).exists(), fname
    theta_back = pd.read_csv(tmp_path / "theta.csv", index_col=0)
    assert list(theta_back.columns) == list(res["theta"].columns)
    assert all(c.endswith("_BayesPrism") for c in theta_back.columns)
    assert theta_back.shape == res["theta"].shape
    z_back = pd.read_csv(tmp_path / "Z_tumor.csv", index_col=0)
    assert z_back.shape == res["Z_tumor"].shape
    assert list(z_back.index) == list(res["theta"].index)  # bulk sample ids
