"""Parity gate for the NATIVE (Rust) bayesprism Gibbs kernel lane.

Same synthetic configuration as tests/test_parity_bayesprism.py (134 real
gene symbols, 120 cells / 8 states / 4 types with a single-state key, 5
bulk samples, reduced chains via the documented gibbs_control passthrough),
plus raw-RNG stream parity against numpy.

Contract
--------
* ``backend='rust'`` + ``state_order='legacy'`` must be BIT-IDENTICAL to the
  ORIGINAL iobrpy pipeline in-process (same PYTHONHASHSEED): theta /
  theta_cv / Z_tumor as raw float64 bit patterns and the three CSVs
  byte-for-byte. The native kernel ports the numpy 2.2.6 RNG chain
  (SeedSequence(123).spawn, the MT19937 pos=623 seeding quirk, sequential
  conditional-binomial multinomials, inversion/BTPE binomials,
  Marsaglia-Tsang gammas over ziggurat normals/exponentials, numpy pairwise
  vs strided-sequential summation orders, the phase-3 shared spawn(1)[0]
  seed quirk) — verified on 12.5M gold stream elements.
* ``state_order='sorted'``: the rust lane must be bit-identical to the
  fast-Python lane (the kernel replaces only the sampler) and deterministic
  across repeated calls.
* Raw parity helpers (``bp_seedseq_state624`` / ``bp_rng_doubles``) must
  match ``SeedSequence.generate_state(624)`` and
  ``Generator(MT19937(ss)).random(n)`` bit-for-bit.
* Non-default gibbs configurations (``fast.multinomial=True``,
  ``rng.backend='randomstate'``, non-int seeds) fall back to the verified
  fast-Python lane; the fallback must stay bit-identical to the ORIGINAL.

Marked `full`-free like the synthetic python-lane gate (no official data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# load the sibling synthetic-fixture module by file location (tests/ is not
# a package; robust under any pytest import mode)
import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location(
    "_bp_parity_fixture", _pl.Path(__file__).with_name("test_parity_bayesprism.py"))
_tbp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tbp)

KEY = _tbp.KEY
N_THREADS = _tbp.N_THREADS
_gibbs = _tbp._gibbs
_inputs = _tbp._inputs
_run_original = _tbp._run_original
_write_csvs = _tbp._write_csvs
_assert_bit_identical = _tbp._assert_bit_identical
_GENE_GROUP = _tbp._GENE_GROUP


def _rust_available() -> bool:
    try:
        from iobrx._fast.bayesprism_gibbs_rust import kernel_available
        return kernel_available()
    except Exception:
        return False


requires_rust = pytest.mark.skipif(
    not _rust_available(), reason="native Gibbs kernel (bp_gibbs_phase) unavailable")


def _run_rust(bulk, sc, csl, ctl, state_order, out_dir=None, gibbs=None):
    from iobrx._fast.bayesprism_gibbs_rust import bayesprism_rust

    return bayesprism_rust(bulk, sc_dat=sc, cell_state_labels=csl,
                           cell_type_labels=ctl, key=KEY, out_dir=out_dir,
                           n_threads=N_THREADS, state_order=state_order,
                           gibbs_control=gibbs if gibbs is not None else _gibbs())


def _run_py(bulk, sc, csl, ctl, state_order, out_dir=None, gibbs=None):
    from iobrx._fast.bayesprism_fast import bayesprism as bp_fast

    return bp_fast(bulk, sc_dat=sc, cell_state_labels=csl,
                   cell_type_labels=ctl, key=KEY, out_dir=out_dir,
                   n_threads=N_THREADS, backend="auto",
                   state_order=state_order,
                   gibbs_control=gibbs if gibbs is not None else _gibbs())


@requires_rust
def test_raw_rng_stream_parity_vs_numpy():
    """bp_seedseq_state624 / bp_rng_doubles == numpy, bit-for-bit."""
    import iobrx._rust as R

    for child in (-1, 0, 1, 3):
        if child < 0:
            ss = np.random.SeedSequence(123)
        else:
            ss = np.random.SeedSequence(123).spawn(child + 1)[child]
        ref_state = ss.generate_state(624, dtype=np.uint32)
        got_state = R.bp_seedseq_state624(123, child)
        assert np.array_equal(ref_state, got_state), f"child={child} state624"
        g = np.random.Generator(np.random.MT19937(ss))
        ref_d = g.random(50000)
        got_d = R.bp_rng_doubles(123, child, 50000)
        assert np.array_equal(ref_d.view(np.uint64), got_d.view(np.uint64)), \
            f"child={child} doubles"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    bulk, sc, csl, ctl = _inputs()
    orig = _run_original(bulk, sc, csl, ctl)
    rust_legacy = _run_rust(bulk, sc, csl, ctl, "legacy")
    rust_sorted1 = _run_rust(bulk, sc, csl, ctl, "sorted")
    rust_sorted2 = _run_rust(bulk, sc, csl, ctl, "sorted")
    py_sorted = _run_py(bulk, sc, csl, ctl, "sorted")
    orig_dir = _write_csvs(orig, tmp_path_factory.mktemp("orig"))
    rust_dir = _write_csvs(rust_legacy, tmp_path_factory.mktemp("rust"))
    return (orig, rust_legacy, rust_sorted1, rust_sorted2, py_sorted,
            orig_dir, rust_dir, (bulk, sc, csl, ctl))


@requires_rust
def test_rust_legacy_bit_identical_to_original(runs):
    orig, rust_legacy = runs[0], runs[1]
    for name in ("theta", "theta_cv", "Z_tumor"):
        _assert_bit_identical(orig[name], rust_legacy[name], f"rust-legacy/{name}")


@requires_rust
def test_rust_legacy_csv_bytes_identical_to_original(runs):
    orig_dir, rust_dir = runs[5], runs[6]
    for fname in ("theta.csv", "theta_cv.csv", "Z_tumor.csv"):
        assert (orig_dir / fname).read_bytes() == (rust_dir / fname).read_bytes(), fname


@requires_rust
def test_rust_sorted_matches_python_lane_and_is_deterministic(runs):
    _, _, rust_sorted1, rust_sorted2, py_sorted = runs[:5]
    for name in ("theta", "theta_cv", "Z_tumor"):
        _assert_bit_identical(py_sorted[name], rust_sorted1[name],
                              f"rust-vs-python-sorted/{name}")
        _assert_bit_identical(rust_sorted1[name], rust_sorted2[name],
                              f"rust-sorted-rerun/{name}")


@requires_rust
def test_rust_fallback_paths_stay_bit_identical(runs):
    """fast.multinomial / randomstate / non-int seed -> python-lane fallback,
    still bit-identical to the ORIGINAL run with the same control."""
    bulk, sc, csl, ctl = runs[7]
    for gc in ({"chain.length": 24, "burn.in": 12, "thinning": 2,
                "fast.multinomial": True},
               {"chain.length": 24, "burn.in": 12, "thinning": 2,
                "rng.backend": "randomstate"}):
        orig = _run_original_gc(bulk, sc, csl, ctl, gc)
        rust = _run_rust(bulk, sc, csl, ctl, "legacy", gibbs=dict(gc))
        for name in ("theta", "theta_cv", "Z_tumor"):
            _assert_bit_identical(orig[name], rust[name],
                                  f"fallback {gc}/{name}")


def _run_original_gc(bulk, sc, csl, ctl, gc):
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
                gibbs_control=dict(gc), opt_control={})
    theta = o_extract.get_fraction(bp, which_theta="final",
                                   state_or_type="type").add_suffix("_BayesPrism")
    theta_cv = bp.posterior_theta_f.theta_cv.add_suffix("_BayesPrism")
    z = o_extract.get_exp(bp, state_or_type="type", cell_name=KEY)
    z = z.to_pandas() if hasattr(z, "to_pandas") else pd.DataFrame(z)
    return {"theta": theta, "theta_cv": theta_cv, "Z_tumor": z}
