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

R5 glue hoist (2026-09-08, full-transcriptome scale): on the frozen
TPM_stad10 tme_profile input (~50k genes x 10 samples, signature=['all']
-> 12,959 merged signatures / ~12.6k surviving per leg) the per-signature
GIL glue dominated the wall clock (~120 s of the ~125 s step; the Rust
kernels themselves are <1 s): every one of the ~25k _pca_one/_zscore_one
calls rebuilt ``set(eset2.index)`` (~44k elements) and went through the
pandas ``.loc`` label-indexing machinery, and both the integration head
and upstream ``filter_signatures`` re-ran per-gene membership against the
Index engine (the integration head evaluates its list comprehension TWICE
per signature).  This module now hoists all of that into ONE memoized
per-frame context (_GlueCtx: frozenset(index), label->position dict,
lazily materialized full-frame float64 matrix), selects rows with numpy
positional take instead of ``.loc``, computes the RAW-index filter in a
single pass, shares the leg filter between the PCA and z-score legs of
integration, and builds each result frame with ONE DataFrame constructor
instead of ~12.5k per-column setitem inserts.  Output bytes are unchanged
by construction (see _GlueCtx docstring for the elementwise-cast /
row-order / duplicate-index arguments); non-unique indexes keep the
original ``.loc`` selection verbatim.

This module keeps output identity by construction:
  * preprocess_eset / _merge_signature_groups are IMPORTED from the
    original calculate_sig_score module and reused unchanged - run ONCE
    per calculate_sig_score_fast call; filter_signatures is replaced by
    _filter_signatures_fast, an output-identical replica whose membership
    test runs on the hoisted frozenset (the imported original is still
    used when the upstream default_debug flag is on, so its DEBUG prints
    are preserved, and by the original ssgsea path below);
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
import weakref

import numpy as np
import pandas as pd
from importlib.resources import files
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

try:
    from iobrpy.workflow.calculate_sig_score import (  # noqa: E402  (upstream, read-only)
        preprocess_eset,
        filter_signatures,
        _merge_signature_groups,
        sig_score_ssgsea as _sig_score_ssgsea_original,
    )
except ModuleNotFoundError:
    preprocess_eset = filter_signatures = None
    _merge_signature_groups = _sig_score_ssgsea_original = None


def _require_upstream():
    if preprocess_eset is None:
        raise ImportError(
            "signature scoring reuses upstream IOBRpy helpers; install the Python "
            "fallback backend with `pip install 'iobrx[python]'`."
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
# R5: per-frame glue context - hoists the per-signature set rebuilds, label
# lookups and float64 materialization out of the signature loops
# ---------------------------------------------------------------------------
_SIG_CTX_CACHE: dict = {}
_SIG_CTX_CACHE_MAX = 4


class _GlueCtx:
    """Precomputed per-frame glue for the signature loops.

    Output identity by construction:
      * ``index_set`` is ``frozenset(eset2.index)``. ``g in eset2.index``
        (Index engine hash lookup) and ``g in index_set`` agree for every
        hashable label (same hash/equality protocol, including the
        identity shortcut for NaN), so
        ``sorted(index_set.intersection(genes))`` is elementwise equal to
        the original ``sorted(set(genes) & set(eset2.index))``;
      * for a UNIQUE index, ``gene_pos`` maps each label to its row
        position and ``arr[positions]`` (numpy positional take, in
        label-sorted order) selects exactly the rows ``eset2.loc[valid]``
        selects, in the same order; the elementwise float64 cast commutes
        with row selection, so
        ``ascontiguousarray(arr[positions])`` is bit-identical to the
        original ``ascontiguousarray(eset2.loc[valid].to_numpy(
        dtype=np.float64))`` fed to the Rust kernels;
      * for a NON-UNIQUE index ``.loc`` pulls EVERY duplicate row per
        label, which a position map keyed by label does not express, so
        callers keep the original ``.loc`` selection verbatim (only the
        set hoist applies).
    """

    __slots__ = ("unique", "gene_pos", "index_set", "arr")

    def __init__(self, eset2):
        self.unique = bool(eset2.index.is_unique)
        if self.unique:
            self.gene_pos = {g: i for i, g in enumerate(eset2.index)}
            self.index_set = frozenset(self.gene_pos)
        else:
            self.gene_pos = None
            self.index_set = frozenset(eset2.index)
        self.arr = None      # lazily built full-frame float64 matrix


def _glue_ctx(eset2):
    """Memoized _GlueCtx, keyed by id() with a weakref identity guard and a
    dealloc callback, so a recycled id can never alias a stale context and
    entries are freed when their frame dies (FIFO bound as a backstop).
    Callers that use the historical 3-argument form (_pca_one(eset2, name,
    genes) - e.g. tme_profile_fast's process-split chunk workers) get the
    hoisted glue with no signature change: the context is built on the
    first call per frame and memoized for the rest of the loop."""
    key = id(eset2)
    ent = _SIG_CTX_CACHE.get(key)
    if ent is not None:
        ref, ctx = ent
        if ref() is eset2:
            return ctx
    if len(_SIG_CTX_CACHE) >= _SIG_CTX_CACHE_MAX:
        _SIG_CTX_CACHE.pop(next(iter(_SIG_CTX_CACHE)), None)

    def _drop(ref, key=key):
        ent = _SIG_CTX_CACHE.get(key)
        if ent is not None and ent[0] is ref:
            del _SIG_CTX_CACHE[key]

    ctx = _GlueCtx(eset2)
    _SIG_CTX_CACHE[key] = (weakref.ref(eset2, _drop), ctx)
    return ctx


def _ctx_arr(ctx, eset2):
    """The full-frame float64 matrix (built once per frame). For a
    consolidated float64 frame ``to_numpy(dtype=np.float64)`` is a
    zero-copy view; the elementwise cast is bit-equal to the original
    per-signature ``.to_numpy(dtype=np.float64)`` either way. Row layout
    does not matter: the per-signature fancy-index take below always
    returns a fresh C-contiguous copy."""
    arr = ctx.arr
    if arr is None:
        arr = eset2.to_numpy(dtype=np.float64)
        ctx.arr = arr
    return arr


def _row_positions(ctx, valid):
    """Row positions of the label-sorted `valid` genes (UNIQUE index)."""
    gene_pos = ctx.gene_pos
    return np.fromiter(map(gene_pos.__getitem__, valid), dtype=np.intp,
                       count=len(valid))


def _select_rows(ctx, eset2, valid):
    """Bit-identical replacement for
    ``np.ascontiguousarray(eset2.loc[valid].to_numpy(dtype=np.float64))``:
    positional take on the hoisted matrix for a unique index, the original
    ``.loc`` selection for a non-unique one (see _GlueCtx)."""
    if ctx.unique:
        x = _ctx_arr(ctx, eset2)[_row_positions(ctx, valid)]
    else:
        x = eset2.loc[valid].to_numpy(dtype=np.float64)
    return np.ascontiguousarray(x)


def _filter_signatures_fast(sig_dict, eset2, min_genes, ctx=None):
    """Output-identical replica of the original filter_signatures with the
    per-gene membership test on the hoisted frozenset instead of the Index
    engine: same dict insertion order, same per-signature list contents
    (duplicates and order within `genes` preserved exactly - no set() is
    applied here, exactly like the original). Defers to the imported
    original when the upstream default_debug flag is on so its DEBUG
    prints are not lost."""
    import iobrpy.workflow.calculate_sig_score as _orig_mod
    if getattr(_orig_mod, "default_debug", False):
        return filter_signatures(sig_dict, eset2, min_genes)
    if ctx is None:
        ctx = _glue_ctx(eset2)
    idx = ctx.index_set
    out = {}
    for name, genes in sig_dict.items():
        present = [g for g in genes if g in idx]
        if len(present) >= min_genes:
            out[name] = present
    return out


# ---------------------------------------------------------------------------
# per-signature scorers: verbatim ports of the original closures, sharing the
# ONE preprocessed eset2 (pure functions -> thread-safe, order-stable)
# ---------------------------------------------------------------------------
def _pca_one(eset2, name, genes, use_rust=None, ctx=None):
    """sig_score_pca._one on the shared preprocessed frame.

    Default: the Rust iobrx_rust.pca_pc1 leg (bit-exact for all 58 official
    signatures - pandas z-chain summation orders + dgesdd jobz='S' from the
    same scipy openblas + svd_flip sign; GATE A). The corrcoef/direction
    part stays in numpy on identical inputs, so it is bit-identical by
    construction. use_rust=False forces the verbatim sklearn port below.

    `ctx` is the memoized _GlueCtx (built on demand when None): the
    per-signature work is the label-sorted intersection plus a positional
    row take - no set(index) rebuild, no .loc machinery - and the matrix
    handed to the kernel is bit-identical (see _GlueCtx / _select_rows).
    """
    if ctx is None:
        ctx = _glue_ctx(eset2)
    valid = sorted(ctx.index_set.intersection(genes))
    if len(valid) < 2:
        # Fallback: all zeros if not enough genes
        return name, np.zeros(len(eset2.columns), dtype=float)
    if use_rust is None:
        use_rust = _RUST_PCA_DEFAULT
    if use_rust and _pca_rust_ready and _ensure_rust_pca():
        x = _select_rows(ctx, eset2, valid)
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


def _zscore_one(eset2, name, genes, use_rust=None, ctx=None):
    """sig_score_zscore._one on the shared preprocessed frame.

    Default: the Rust iobrx_rust.rows_colmean_pairwise leg (pandas colwise
    pairwise sum order over the selected rows - bit-exact for all 58
    official signatures); use_rust=False forces the pandas verbatim.
    `ctx`: see _pca_one (same hoisted glue, same identity argument).
    """
    if ctx is None:
        ctx = _glue_ctx(eset2)
    valid = sorted(ctx.index_set.intersection(genes))
    if len(valid) == 0:
        return name, np.zeros(len(eset2.columns), dtype=float)
    if use_rust is None:
        use_rust = _RUST_PCA_DEFAULT
    if use_rust and _pca_rust_ready:
        x = _select_rows(ctx, eset2, valid)
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
def _sig_score_pca_on_eset2(eset, eset2, sig_dict, mini_gene_count, parallel_size=1,
                            sigs=None):
    min_size = max(mini_gene_count, 2)
    if sigs is None:
        sigs = _filter_signatures_fast(sig_dict, eset2, min_size)
    items = list(sigs.items())

    ctx = _glue_ctx(eset2)
    if ctx.unique and _RUST_PCA_DEFAULT and _pca_rust_ready:
        _ctx_arr(ctx, eset2)             # materialize ONCE for all signatures

    if parallel_size and parallel_size > 1:
        # threads: each _one is independent and its numpy/LAPACK kernels
        # release the GIL; results come back in submission order, so the
        # output frame is identical to the serial run by construction.
        results = Parallel(n_jobs=int(parallel_size), prefer="threads")(
            delayed(_pca_one)(eset2, name, genes, None, ctx) for name, genes in items
        )
    else:
        results = [_pca_one(eset2, name, genes, None, ctx) for name, genes in items]

    # ONE DataFrame construction instead of one setitem insert per signature
    # (~12.5k inserts -> ~12.5k manager blocks on the full-transcriptome
    # 'all' workload): dict insertion order == the previous setitem order,
    # values/dtypes untouched, so the frame - and every byte written from
    # it - is identical, but consolidated into a single float64 block.
    cols = {'ID': eset.columns}
    for name, vec in results:
        cols[name] = vec
    pdata = pd.DataFrame(cols)
    return _attach_tme_contrasts(pdata, sigs)


def _sig_score_zscore_on_eset2(eset, eset2, sig_dict, mini_gene_count, parallel_size=1,
                               sigs=None):
    # straightforward: a column-mean per signature; parallel_size accepted
    # for signature compatibility (the per-signature op is microseconds).
    min_size = max(mini_gene_count, 2)
    if sigs is None:
        sigs = _filter_signatures_fast(sig_dict, eset2, min_size)

    ctx = _glue_ctx(eset2)
    if ctx.unique and _RUST_PCA_DEFAULT and _pca_rust_ready:
        _ctx_arr(ctx, eset2)             # materialize ONCE for all signatures

    results = [_zscore_one(eset2, name, genes, None, ctx) for name, genes in sigs.items()]

    cols = {'ID': eset.columns}
    for name, vec in results:
        cols[name] = vec
    pdata = pd.DataFrame(cols)
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
    # (semantics verbatim from sig_score_integration) filter against the RAW
    # eset index. The original's dict comprehension evaluates
    # `[g for g in genes if g in eset.index]` TWICE per signature (value +
    # condition); this single pass over a hoisted set(eset.index) produces
    # the identical dict - same key order, same list contents - with half
    # the membership work and no per-probe Index engine overhead.
    raw_index_set = set(eset.index)
    filtered_sigs = {}
    for name, genes in sig_dict.items():
        present = [g for g in genes if g in raw_index_set]
        if len(present) >= mini_gene_count:
            filtered_sigs[name] = present

    # ---- the single shared preprocessing (the original does this 3 times) ----
    eset2 = preprocess_eset(eset, adjust_eset)

    # both legs apply the SAME deterministic filter to the SAME dict/frame
    # (min_size = max(mini_gene_count, 2)): compute it once, share it.
    sigs = _filter_signatures_fast(filtered_sigs, eset2, max(mini_gene_count, 2))

    p = _sig_score_pca_on_eset2(eset, eset2, filtered_sigs, mini_gene_count, parallel_size,
                                sigs=sigs)
    p = p.set_index('ID').add_suffix('_PCA')

    z = _sig_score_zscore_on_eset2(eset, eset2, filtered_sigs, mini_gene_count, parallel_size,
                                   sigs=sigs)
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
    _require_upstream()
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
