"""sig_score_fast: composed signature-scoring entry for IOBRpy's
calculate_sig_score workflow (pca / zscore / ssgsea / integration).

Motivation (measured on the original module, official imvigor210_eset x
signature_collection):
  * every one of sig_score_pca / sig_score_zscore / sig_score_ssgsea calls
    preprocess_eset(eset, adjust_eset) itself, and sig_score_integration
    calls sig_score_pca + sig_score_zscore + preprocess_eset + gp.ssgsea,
    i.e. the SAME deterministic preprocessing is recomputed 3x per
    integration run (log2 heuristic + numeric/finite/sd>0 filtering over an
    872x348 frame);
  * the per-signature PCA loop is embarrassingly parallel (one independent
    PCA(n_components=1, svd_solver='full') per signature) and its heavy
    kernels (LAPACK dgesdd via scipy.linalg.svd inside sklearn PCA,
    numpy reductions) release the GIL, so joblib THREADS can overlap the
    Python/pandas/sklearn estimator overhead without pickling.

This module keeps output identity by construction:
  * preprocess_eset / filter_signatures / _merge_signature_groups are
    IMPORTED from the original calculate_sig_score module and reused
    unchanged - run ONCE per calculate_sig_score_fast call;
  * _pca_one / _zscore_one are pure functions of (eset2, name, genes) on
    the shared preprocessed frame, so joblib threads cannot change any
    result, only scheduling. Since 2026-09-07 both legs default to the
    Rust iobrx_rust kernels (pca_pc1 / rows_colmean_pairwise - bit-exact
    vs the sklearn/pandas originals for all 58 official signatures, GATE
    A/B in iobrx/bench/gate_pca_rust.py + gate_stage_rust_pca.py); the
    verbatim sklearn/pandas ports remain as in-module fallbacks
    (use_rust=False);
  * the ssgsea leg of method='integration' reproduces sig_score_integration
    literally: it does NOT go through sig_score_ssgsea (which double-filters
    signatures and enforces min_size = max(mini_gene_count, 5)); integration
    filters signatures ONCE against the RAW eset.index (>= mini_gene_count)
    and hands that dict straight to the ssGSEA core with
    min_size = mini_gene_count (gseapy's own internal load_gmt filter,
    min_size<=tag_len<=max_size against the PREPROCESSED index, still
    applies and is replicated literally by ssgsea_fast). The matrix fed to
    the core is the THIRD preprocess_eset(eset, adjust_eset) frame — the
    same deterministic frame the pca/zscore legs use. That core is now the
    Rust iobrx_rust.ssgsea_core via ssgsea_fast.ssgsea_fast (DEFAULT,
    _RUST_SSGSEA_DEFAULT = True): the rebuilt crate is bit-exact for BOTH
    semantics on the official data — refs/sig_ssgsea.parquet
    (iobrx/bench/rust_ssgsea_results.json) and refs/sig_integration.parquet
    (iobrx/bench/integration_rust_results.json). An earlier measurement
    against a pre-rebuild .so found 31/31 columns differing; superseded.
    The original gp.ssgsea call remains available via use_rust_ssgsea=False;
  * output column order and the TMEscore_CIR / TMEscore_plus derived
    columns are produced by the same statements as the original.

method='ssgsea' delegates to the original sig_score_ssgsea (or the rust
drop-in if importable) - preprocessing is shared by that callee itself.
"""
import warnings

import numpy as np
import pandas as pd
from importlib.resources import files
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from iobrpy.workflow.calculate_sig_score import (  # noqa: E402  (upstream, read-only)
    preprocess_eset,
    filter_signatures,
    _merge_signature_groups,
    sig_score_ssgsea as _sig_score_ssgsea_original,
)
from iobrx._sites import find_bundled_openblas as _find_bundled_openblas

try:
    import gseapy as gp
except ImportError:  # pragma: no cover - mirrors the original's guard
    gp = None

# Rust ssGSEA core (sibling deliverable iobrx/fast/ssgsea_fast.py: gseapy
# load_data/load_gmt literal ports + iobrx_rust.ssgsea_core + res2d
# reconstruction). The crate was rebuilt 2026-09-07 and the INSTALLED .so is
# bit-exact on the official imvigor210_eset x signature_collection data:
#   * method='ssgsea' semantics (double filter, min_size = max(3, 5)) ->
#     refs/sig_ssgsea.parquet, max abs diff 0.0
#     (iobrx/bench/rust_ssgsea_results.json)
#   * method='integration' semantics (single RAW-eset filter,
#     min_size = mini_gene_count, third preprocess feeds the core) ->
#     refs/sig_integration.parquet, max abs diff 0.0
#     (iobrx/bench/integration_rust_results.json)
# An earlier measurement that found 31/31 columns differing (max abs diff
# 0.4349) was taken against a pre-rebuild .so and is superseded.
# use_rust_ssgsea=False forces the original gp.ssgsea leg.
try:
    from iobrx._fast.ssgsea_fast import (  # type: ignore
        ssgsea_fast as _rust_ssgsea_res2d,
        sig_score_ssgsea_fast as _rust_sig_score_ssgsea,
    )
    _HAVE_RUST_SSGSEA = True
except Exception:  # pragma: no cover
    _rust_ssgsea_res2d = None
    _rust_sig_score_ssgsea = None
    _HAVE_RUST_SSGSEA = False

_RUST_SSGSEA_DEFAULT = True

# Rust PCA/zscore legs (iobrx_rust.pca_pc1 / rows_colmean_pairwise): the
# crate was rebuilt 2026-09-07 with a bit-exact port of the
# PCA(n_components=1, svd_solver='full') path (pandas z-chain orders +
# dgesdd jobz='S' called from the SAME dlopened scipy openblas via
# init_blas, lwork from the workspace query, svd_flip row-0 sign). GATE A
# (iobrx/bench/gate_pca_rust.py, fresh subprocess): all 58 official
# signatures bit-exact for the z intermediate, PC1 and mean_expr.
# _pca_one keeps the verbatim sklearn port as fallback (use_rust_pca=False
# or crate without pca_pc1).
try:
    import ctypes as _ctypes

    import iobrx._rust as _ir

    _pca_rust_ready = hasattr(_ir, "pca_pc1")
except Exception:  # pragma: no cover
    _ir = None
    _pca_rust_ready = False

_RUST_PCA_DEFAULT = True
_dgesdd_inited = False


def _ensure_rust_pca():
    """Resolve scipy_dgesdd_ from scipy's bundled openblas (the exact .so
    scipy.linalg.svd uses) and hand its address to the crate. Passing 0 for
    the other pointers leaves any previously initialized ddot/gemv64
    untouched."""
    global _dgesdd_inited
    if _dgesdd_inited or not _pca_rust_ready:
        return _pca_rust_ready
    hits, _ = _find_bundled_openblas()
    if not hits:
        return False
    try:
        lib = _ctypes.CDLL(hits[0])
        addr = _ctypes.cast(lib.scipy_dgesdd_, _ctypes.c_void_p).value
    except (OSError, AttributeError):
        return False
    _ir.init_blas(0, 0, addr)
    _dgesdd_inited = True
    return True


# ---------------------------------------------------------------------------
# per-signature scorers: verbatim ports of the original closures, sharing the
# ONE preprocessed eset2 (pure functions -> thread-safe, order-stable)
# ---------------------------------------------------------------------------
def _pca_one(eset2, name, genes, use_rust=None):
    """sig_score_pca._one on the shared preprocessed frame.

    Default: the Rust iobrx_rust.pca_pc1 leg (bit-exact for all 58 official
    signatures - pandas z-chain summation orders + dgesdd jobz='S' from the
    same scipy openblas + svd_flip sign; GATE A). The corrcoef/direction
    part stays in numpy on identical inputs, so it is bit-identical by
    construction. use_rust=False forces the verbatim sklearn port below.
    """
    valid = sorted(set(genes) & set(eset2.index))
    if len(valid) < 2:
        # Fallback: all zeros if not enough genes
        return name, np.zeros(len(eset2.columns), dtype=float)
    if use_rust is None:
        use_rust = _RUST_PCA_DEFAULT
    if use_rust and _pca_rust_ready and _ensure_rust_pca():
        x = np.ascontiguousarray(eset2.loc[valid].to_numpy(dtype=np.float64))
        pc1, mean_expr = _ir.pca_pc1(x)
        pc1 = np.asarray(pc1)
        mean_expr = np.asarray(mean_expr)
        corr = np.corrcoef(pc1, mean_expr)[0, 1]
        direction = np.sign(corr) if not np.isnan(corr) else 1.0
        return name, (pc1 * direction)

    tmp = eset2.loc[valid]               # genes × samples
    mat = tmp.T                          # samples × genes
    # z-score by gene
    mat = mat.sub(mat.mean(axis=0), axis=1)
    mat = mat.div(mat.std(axis=0, ddof=1).replace(0, np.nan), axis=1).fillna(0.0)

    pca = PCA(n_components=1, svd_solver='full', random_state=0)
    pc1 = pca.fit_transform(mat.values)[:, 0]   # length = n_samples

    mean_expr = tmp.mean(axis=0).values         # length = n_samples
    corr = np.corrcoef(pc1, mean_expr)[0, 1]
    direction = np.sign(corr) if not np.isnan(corr) else 1.0
    return name, (pc1 * direction)


def _zscore_one(eset2, name, genes, use_rust=None):
    """sig_score_zscore._one on the shared preprocessed frame.

    Default: the Rust iobrx_rust.rows_colmean_pairwise leg (pandas colwise
    pairwise sum order over the selected rows - bit-exact for all 58
    official signatures); use_rust=False forces the pandas verbatim.
    """
    valid = sorted(set(genes) & set(eset2.index))
    if len(valid) == 0:
        return name, np.zeros(len(eset2.columns), dtype=float)
    if use_rust is None:
        use_rust = _RUST_PCA_DEFAULT
    if use_rust and _pca_rust_ready:
        x = np.ascontiguousarray(eset2.loc[valid].to_numpy(dtype=np.float64))
        return name, np.asarray(_ir.rows_colmean_pairwise(x))
    mat = eset2.loc[valid]               # genes × samples
    return name, mat.mean(axis=0).values


def _attach_tme_contrasts(pdata, sigs):
    """Same TMEscore derived columns, same statements, as the original."""
    if {'TMEscoreA_CIR', 'TMEscoreB_CIR'}.issubset(sigs):
        pdata['TMEscore_CIR'] = pdata['TMEscoreA_CIR'] - pdata['TMEscoreB_CIR']
    if {'TMEscoreA_plus', 'TMEscoreB_plus'}.issubset(sigs):
        pdata['TMEscore_plus'] = pdata['TMEscoreA_plus'] - pdata['TMEscoreB_plus']
    return pdata


# ---------------------------------------------------------------------------
# scorers over an ALREADY-preprocessed frame (preprocessing done once by the
# caller); identical output to sig_score_pca / sig_score_zscore
# ---------------------------------------------------------------------------
def _sig_score_pca_on_eset2(eset, eset2, sig_dict, mini_gene_count, parallel_size=1):
    pdata = pd.DataFrame({'ID': eset.columns})
    min_size = max(mini_gene_count, 2)
    sigs = filter_signatures(sig_dict, eset2, min_size)
    items = list(sigs.items())

    if parallel_size and parallel_size > 1:
        # threads: each _one is independent and its numpy/LAPACK kernels
        # release the GIL; results come back in submission order, so the
        # output frame is identical to the serial run by construction.
        results = Parallel(n_jobs=int(parallel_size), prefer="threads")(
            delayed(_pca_one)(eset2, name, genes) for name, genes in items
        )
    else:
        results = [_pca_one(eset2, name, genes) for name, genes in items]

    for name, vec in results:
        pdata[name] = vec
    return _attach_tme_contrasts(pdata, sigs)


def _sig_score_zscore_on_eset2(eset, eset2, sig_dict, mini_gene_count, parallel_size=1):
    # straightforward: a column-mean per signature; parallel_size accepted
    # for signature compatibility (the per-signature op is microseconds).
    pdata = pd.DataFrame({'ID': eset.columns})
    min_size = max(mini_gene_count, 2)
    sigs = filter_signatures(sig_dict, eset2, min_size)

    results = [_zscore_one(eset2, name, genes) for name, genes in sigs.items()]

    for name, vec in results:
        pdata[name] = vec
    return _attach_tme_contrasts(pdata, sigs)


# ---------------------------------------------------------------------------
# ssGSEA leg shared by method='ssgsea'/'integration': rust core if available,
# else the ORIGINAL gp.ssgsea call with IOBRpy's exact arguments
# ---------------------------------------------------------------------------
def _ssgsea_res2d_shared(eset2, gene_sets, min_size, threads, use_rust=None):
    """Return a gseapy res2d-equivalent long DataFrame.

    Default: the Rust iobrx_rust.ssgsea_core (bit-exact on the official
    data for BOTH the method='ssgsea' and method='integration' semantics,
    see the module notes above). use_rust=False selects the literal
    gp.ssgsea call the original module makes instead.
    """
    if use_rust is None:
        use_rust = _RUST_SSGSEA_DEFAULT
    if use_rust and _HAVE_RUST_SSGSEA:
        return _rust_ssgsea_res2d(
            eset2,
            gene_sets,
            min_size=int(min_size),
            max_size=500,           # gp.ssgsea default, IOBRpy does not override
            weight=0.25,            # gp.ssgsea default for ssGSEA
            sample_norm_method="rank",
            correl_norm_type=None,  # gp.ssgsea default 'rank' -> CorrelType::Rank
            threads=int(threads),
        )
    if gp is None:
        raise ImportError("gseapy required for ssGSEA")
    print("Running ssGSEA (this may take a while)...")
    ss = gp.ssgsea(
        data=eset2,
        gene_sets=gene_sets,
        outdir=None,
        sample_norm_method='rank',    # rank-based kernel = Gaussian
        permutation_num=0,
        no_plot=True,
        threads=threads,
        min_size=min_size,
        ssgsea_norm=True
    )
    return ss.res2d


def _nes_from_res2d(res2d):
    """The original pivot dance: samples × terms + TME contrasts."""
    nes = res2d.pivot(index='Term', columns='Name', values='NES').T.reset_index()
    nes.rename(columns={'Name': 'ID'}, inplace=True)
    if 'TMEscoreA_CIR' in nes.columns and 'TMEscoreB_CIR' in nes.columns:
        nes['TMEscore_CIR'] = nes['TMEscoreA_CIR'] - nes['TMEscoreB_CIR']
    if 'TMEscoreA_plus' in nes.columns and 'TMEscoreB_plus' in nes.columns:
        nes['TMEscore_plus'] = nes['TMEscoreA_plus'] - nes['TMEscoreB_plus']
    return nes


# ---------------------------------------------------------------------------
# integration: ONE preprocess shared by all three scorers
# ---------------------------------------------------------------------------
def _sig_score_integration_fast(eset, sig_dict, mini_gene_count, adjust_eset, parallel_size,
                                use_rust_ssgsea=None):
    # (verbatim from sig_score_integration) filter against the RAW eset index
    filtered_sigs = {
        name: [g for g in genes if g in eset.index]
        for name, genes in sig_dict.items()
        if len([g for g in genes if g in eset.index]) >= mini_gene_count
    }

    # ---- the single shared preprocessing (the original does this 3 times) ----
    eset2 = preprocess_eset(eset, adjust_eset)

    p = _sig_score_pca_on_eset2(eset, eset2, filtered_sigs, mini_gene_count, parallel_size)
    p = p.set_index('ID').add_suffix('_PCA')

    z = _sig_score_zscore_on_eset2(eset, eset2, filtered_sigs, mini_gene_count, parallel_size)
    z = z.set_index('ID').add_suffix('_zscore')

    if gp is None or not filtered_sigs:
        reason = (
            "gseapy is not available"
            if gp is None
            else f"no signatures passed the mini_gene_count={mini_gene_count} filter"
        )
        warnings.warn(
            "Skipping ssGSEA in integration scoring because "
            f"{reason}; returning PCA and z-score results only.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.concat([p, z], axis=1).reset_index()

    res2d = _ssgsea_res2d_shared(eset2, filtered_sigs, mini_gene_count, parallel_size,
                                 use_rust=use_rust_ssgsea)
    nes = _nes_from_res2d(res2d)
    s = nes.set_index('ID').add_suffix('_ssGSEA')

    return pd.concat([p, z, s], axis=1).reset_index()


# ---------------------------------------------------------------------------
# public composed entry
# ---------------------------------------------------------------------------
def calculate_sig_score_fast(eset, signature_names, method, mini_gene_count=3,
                             adjust_eset=True, parallel_size=1, use_rust_ssgsea=None):
    """Drop-in for calculate_sig_score.calculate_sig_score with shared
    preprocessing and threaded per-signature PCA.

    Output DataFrames are IDENTICAL to the original for the same arguments
    (same functions on the same inputs; per-signature scoring is order-
    independent), verified bitwise against iobrx/refs/sig_*.parquet.

    use_rust_ssgsea: None -> module default (True: Rust iobrx_rust.ssgsea_core
    via ssgsea_fast — bit-exact vs refs for both the ssgsea and integration
    semantics on the official data; False forces the original gp.ssgsea leg).
    """
    resource_pkg = 'iobrpy.resources'
    resource_path = files(resource_pkg).joinpath('calculate_data.pkl')
    all_sigs = pd.read_pickle(resource_path)
    sig_dict = _merge_signature_groups(all_sigs, signature_names)
    if not isinstance(sig_dict, dict) or len(sig_dict) == 0:
        raise KeyError(f"No valid signatures found from groups: {signature_names}")

    use_rust = _RUST_SSGSEA_DEFAULT if use_rust_ssgsea is None else use_rust_ssgsea

    m = method.lower()
    if m == 'pca':
        eset2 = preprocess_eset(eset, adjust_eset)      # ONCE
        return _sig_score_pca_on_eset2(eset, eset2, sig_dict, mini_gene_count, parallel_size)
    if m == 'zscore':
        eset2 = preprocess_eset(eset, adjust_eset)      # ONCE
        return _sig_score_zscore_on_eset2(eset, eset2, sig_dict, mini_gene_count, parallel_size)
    if m == 'ssgsea':
        if use_rust and _HAVE_RUST_SSGSEA:
            return _rust_sig_score_ssgsea(eset, sig_dict, mini_gene_count, adjust_eset, parallel_size)
        return _sig_score_ssgsea_original(eset, sig_dict, mini_gene_count, adjust_eset, parallel_size)
    if m == 'integration':
        return _sig_score_integration_fast(eset, sig_dict, mini_gene_count, adjust_eset,
                                           parallel_size, use_rust_ssgsea=use_rust)
    raise ValueError("Unknown method")
