"""cibersort_fast: bit-exact, Rust+libsvm re-implementation of
IOBRpy's iobrpy.workflow.cibersort.cibersort() over a DataFrame input.

Bit-exactness strategy (all verified against the reference pipeline):
  - NuSVR solves run in vendored sklearn 1.7.2 libsvm (byte-identical
    svm.cpp) with sklearn's exact svm_parameter; kernel dots go through
    scipy's bundled openblas ddot, fetched from the very same .so the
    installed sklearn uses (ctypes address passed to the Rust module).
  - w = dual_coef @ SV and k = X @ w_use call numpy's bundled ILP64
    openblas dgemv with the exact layouts numpy picks (RowMajor/Trans for
    the C-contiguous support vectors, ColMajor/NoTrans for the F-order X
    that pandas .loc produces).
  - All numpy reductions (pairwise summation + 8192-element nditer chunking,
    axis-0 sequential accumulation, the F-layout mean/std tree) are ported
    bit-for-bit, including the f32 pairwise sum of the weights.
  - Quantile normalization reproduces np.argsort tie semantics by compiling
    numpy's vendored x86-simd-sort (AVX512 SKX variant, the kernel numpy
    dispatches to on AVX512F+DQ CPUs) into the crate.
  - Permutations use a seeded port of numpy SeedSequence+PCG64+Lemire
    integers, so p-values are reproducible run-to-run (the ORIGINAL is
    unseeded - SeedSequence() draws OS entropy - so its exact p-values are
    not reproducible by design; the formula p = count(null_r >= r)/perm is
    identical, granularity 1/perm).
"""
import ctypes
import os

import numpy as np
import pandas as pd

import iobrx._rust as _ir

from iobrx._sites import find_bundled_openblas as _find_bundled_openblas

_blas_inited = False


def _init_blas() -> None:
    """Resolve scipy_ddot_ (LP64) and scipy_cblas_dgemv64_ (ILP64) from the
    exact shared libraries the installed scipy/numpy use, then hand their
    addresses to the Rust module. dlopen returns the already-loaded handle,
    so the Rust code calls the very same BLAS kernels numpy/sklearn use."""
    global _blas_inited
    if _blas_inited:
        return
    scipy_hits, numpy_hits = _find_bundled_openblas()
    if not scipy_hits or not numpy_hits:
        raise RuntimeError("could not locate bundled openblas libraries for scipy/numpy")
    lib_s = ctypes.CDLL(scipy_hits[0])
    lib_n = ctypes.CDLL(numpy_hits[0])
    ddot_addr = ctypes.cast(lib_s.scipy_ddot_, ctypes.c_void_p).value
    gemv_addr = ctypes.cast(lib_n.scipy_cblas_dgemv64_, ctypes.c_void_p).value
    ok = _ir.init_blas(ddot_addr, gemv_addr)
    if ok != (True, True):
        raise RuntimeError(f"init_blas failed: {ok}")
    # Pin numpy's bundled openblas to 1 thread: our gemv calls (531x22) are
    # below any multithreaded size threshold numerically (verified bit-equal
    # at 1/2/8/64 threads) but its thread-team dispatch from inside hundreds
    # of rayon workers causes oversubscription. Restores ~30% wall time.
    try:
        setn = lib_n.scipy_openblas_set_num_threads64_
        setn.argtypes = [ctypes.c_int64]
        setn.restype = None
        setn(1)
    except AttributeError:
        pass
    _blas_inited = True


def _make_unique(index):
    """Same as iobrpy.workflow.cibersort.make_unique."""
    counts = {}
    out = []
    for name in index:
        c = counts.get(name, 0) + 1
        counts[name] = c
        out.append(name if c == 1 else f"{name}.{c}")
    return out


def _default_lm22() -> pd.DataFrame:
    from importlib.resources import files

    lm22_path = files("iobrpy.resources").joinpath("lm22.txt")
    return pd.read_csv(lm22_path, sep=r"\s+", engine="python", index_col=0)


def cibersort_fast(
    eset_df: pd.DataFrame,
    perm: int = 100,
    QN: bool = True,
    absolute: bool = False,
    abs_method: str = "sig.score",
    n_threads: int = 224,
    lm22_df: pd.DataFrame | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Drop-in fast cibersort over a genes x samples DataFrame.

    Returns a DataFrame indexed by sample with the signature's cell-type
    columns plus 'P-value', 'Correlation', 'RMSE' (and
    'Absolute_score_(<method>)' when absolute=True), matching the original's
    layout. Weights are float32-rounded exactly like the original.
    """
    _init_blas()
    if lm22_df is None:
        lm22_df = _default_lm22()

    # pandas-side index handling (original steps 1-2, 6 alignment)
    mix_df = eset_df.copy()
    mix_df.index = _make_unique(mix_df.index)
    sig_df = lm22_df.sort_index()
    mix_df = mix_df.sort_index()

    mix_y = np.ascontiguousarray(mix_df.to_numpy(dtype=np.float64))
    # ORIGINAL input semantics (bit-exactness): the reference cibersort() is
    # file-based - the official protocol feeds it eset.to_csv(temp.csv) and it
    # computes with pd.read_csv(path, sep=None, engine='python'), whose float
    # parse (pandas precise_xstrtod) is NOT the identity on the DataFrame's
    # float64 values (1-2 ulp drift on ~12% of cells; on the degenerate BLCA
    # mixture TCGA-2F-A9KR those last bits flip the NuSVR support set and move
    # 15/22 weights by up to 1.79e-3). Apply the exact same parse in memory
    # (validated bitwise against the real to_csv->read_csv round trip on the
    # full BLCA 205050-cell and STAD 501810-cell matrices, 0 differing cells).
    mix_y = _ir.csv_parse_roundtrip(mix_y, int(n_threads))
    common_mask = mix_df.index.isin(sig_df.index)
    if not common_mask.any():
        raise ValueError("No overlapping genes found between signature and mixture matrices.")
    sig_common = sig_df.loc[mix_df.index[common_mask]]
    sig_x = np.ascontiguousarray(sig_common.to_numpy(dtype=np.float64))

    # step 3 (exp2 heuristic) is applied HERE with numpy, not in Rust: the
    # original runs `if np.max(Y) < 50: np.exp2(Y, out=Y)`, and numpy's
    # vectorized exp2 differs from Rust's libm f64::exp2 by 1 ulp on ~5% of
    # inputs — on degenerate mixtures (few signature genes vs many cell
    # types) those last-bit differences flip NuSVR support sets and visibly
    # change weights. np.max's NaN propagation also matches the original
    # exactly (NaN -> no exp2). exp2_done tells cibersort_core to skip its
    # internal (non-numpy) exp2.
    exp2_done = bool(np.max(mix_y) < 50)
    if exp2_done:
        np.exp2(mix_y, out=mix_y)

    out = _ir.cibersort_core(
        mix_y,
        common_mask,
        sig_x,
        perm=int(perm),
        use_qn=bool(QN),
        n_threads=int(n_threads),
        seed=int(seed),
        absolute=bool(absolute),
        exp2_done=exp2_done,
    )

    n = mix_y.shape[1]
    c = sig_x.shape[1]
    weights = np.asarray(out["weights"]).reshape(n, c).astype(np.float32)
    rs = np.asarray(out["correlation"], dtype=np.float32).astype(np.float64)
    rmses = np.asarray(out["rmse"], dtype=np.float32).astype(np.float64)
    pvals = np.asarray(out["pvalue"], dtype=np.float64)

    colnames = list(sig_common.columns) + ["P-value", "Correlation", "RMSE"]
    if absolute:
        safe = abs_method.replace(".", "_")
        colnames.append(f"Absolute_score_({safe})")

    rows = []
    for i in range(n):
        w = weights[i]
        abs_score = None
        if absolute:
            if abs_method == "sig.score":
                # numpy NEP50: float32 array * python float -> float32
                ratio = float(out["y_col_medians"][i]) / float(out["y_median_full"])
                w = w * ratio
                abs_score = float(np.sum(w))
            elif abs_method == "no.sumto1":
                w = np.asarray(out["w_raw"]).reshape(n, c)[i].astype(np.float32)
                abs_score = float(out["w_raw_sums"][i])
            else:
                raise ValueError(f"unknown abs_method: {abs_method}")
        row = list(map(float, w)) + [float(pvals[i]), float(rs[i]), float(rmses[i])]
        if absolute:
            row.append(abs_score if abs_score is not None else float(np.sum(w)))
        rows.append(row)

    result = pd.DataFrame(rows, columns=colnames, index=mix_df.columns)
    # dtype parity with the reference output: weights float32, stats float64
    result[sig_common.columns] = result[sig_common.columns].astype(np.float32)
    return result
