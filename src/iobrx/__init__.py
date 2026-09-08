"""iobrx — bit-exact, multithreaded acceleration layer for IOBRpy.

``iobrx`` wraps the IOBRpy (https://pypi.org/project/iobrpy/) tumor
micro-environment workflows in accelerated implementations whose outputs are
bit-identical to the originals on the official validation data (all 11
official gates: max abs diff 0.0, cibersort's unseeded P-value column excepted
by design). Heavy solvers (CIBERSORT's NuSVR, ssGSEA, PCA signature scoring)
run in a Rust extension (rayon threads + vendored libsvm/OpenBLAS entry
points); the remaining stages are vectorized numpy/pandas with per-process
resource caching.

Quick start (downloads a public signature-scoring example)::

    import iobrx

    eset = iobrx.load_official("imvigor210_eset")  # $IOBRX_TESTDATA -> cache
    scores = iobrx.calculate_sig_score(eset, "signature_collection",
                                       method="pca")
    print(iobrx.backend_info())

The optional native extension uses NumPy's local sorting dispatch and does
not require AVX-512. Full-transcriptome deconvolution examples and measured
desktop timings are provided in the repository's executed tutorials.

Every function accepts DataFrames in the same orientation as IOBRpy
(genes/samples x samples/genes as documented per function) and returns frames
with the original layout, index, column order and dtypes. ``n_threads=None``
(the default) resolves to ``min(8, os.cpu_count())``; change it per call or
process-wide with :func:`set_threads`.
"""
from __future__ import annotations

import os
import sys

from iobrx._threads import get_threads, resolve_threads, set_threads
from iobrx._backend import backend_info, select_backend
from iobrx._testdata import MIRRORS, OfficialDataUnavailable, load_official

__version__ = "0.2.0"

# The optional native module and historical alias are managed by _backend.

__all__ = [
    "cibersort",
    "calculate_sig_score",
    "count2tpm",
    "quantiseq",
    "deconvolute_quantiseq",
    "epic",
    "mcpcounter",
    "estimate_score",
    "anno_eset",
    "bayesprism",
    "tme_profile",
    "nmf",
    "merge_salmon",
    "tme_cluster",
    "lr_cal",
    "ips",
    "log2_eset",
    "prepare_salmon",
    "mouse2human",
    "merge_star_count",
    "fastq_qc",
    "batch_salmon",
    "batch_star_count",
    "trust4",
    "spechla",
    "hla_typing",
    "runall",
    "load_official",
    "OfficialDataUnavailable",
    "MIRRORS",
    "set_threads",
    "get_threads",
    "backend_info",
    "__version__",
]

# ---------------------------------------------------------------------------
# lazy resource caches
# ---------------------------------------------------------------------------
_QUANTISEQ_DATA = None
_EPIC_TREF = None


def _quantiseq_data():
    global _QUANTISEQ_DATA
    if _QUANTISEQ_DATA is None:
        import pandas as pd
        from importlib.resources import files

        _QUANTISEQ_DATA = pd.read_pickle(
            str(files("iobrpy.resources").joinpath("quantiseq_data.pkl"))
        )
    return _QUANTISEQ_DATA


def _epic_tref():
    global _EPIC_TREF
    if _EPIC_TREF is None:
        import pandas as pd
        from importlib.resources import files

        _EPIC_TREF = pd.read_pickle(
            str(files("iobrpy.resources").joinpath("epic_TRef_BRef.pkl"))
        )["TRef"]
    return _EPIC_TREF


# ---------------------------------------------------------------------------
# CIBERSORT
# ---------------------------------------------------------------------------
def cibersort(
    eset,
    perm: int = 100,
    QN: bool = True,
    absolute: bool = False,
    abs_method: str = "sig.score",
    n_threads: int | None = None,
    backend: str = "auto",
):
    """CIBERSORT deconvolution against the LM22 signature (NuSVR), accelerated.

    Bit-exact drop-in for ``iobrpy.workflow.cibersort.cibersort`` on the
    official gates: weights / Correlation / RMSE are bit-identical; the
    P-value column uses the same formula on seeded permutations (the
    ORIGINAL draws OS entropy and is not reproducible run-to-run by design).

    Parameters
    ----------
    eset : pandas.DataFrame
        Mixture matrix, genes (index) x samples (columns), e.g. a
        symbol-aggregated expression matrix. Duplicated index entries are
        de-duplicated with the original ``make_unique`` ``name.2`` scheme.
    perm : int, default 100
        Number of permutations for the P-value estimate.
    QN : bool, default True
        Quantile-normalize the mixture before fitting. Retained for backward
        compatibility; RNA-seq workflows generally use False, while
        microarray workflows commonly use True. Match the data and protocol.
    absolute : bool, default False
        Also report the absolute-score column.
    abs_method : {'sig.score', 'no.sumto1'}, default 'sig.score'
        Absolute-mode scoring rule (as IOBRpy).
    n_threads : int or None, optional
        Rayon threads for the solver. ``None`` resolves to
        ``min(8, os.cpu_count())`` (see :func:`set_threads`).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        Auto uses native NuSVR when its BLAS requirements are available,
        otherwise the original Python workflow. Explicit rust fails clearly
        if unavailable. Native permutations use seed 0; the Python fallback
        retains the upstream unseeded P-values.

    Returns
    -------
    pandas.DataFrame
        One row per sample: 22 LM22 cell-type weight columns (float32, like
        the original) plus ``P-value``, ``Correlation``, ``RMSE`` and, when
        ``absolute=True``, ``Absolute_score_(<abs_method>)``.
    """
    threads = resolve_threads(n_threads)
    if select_backend(backend):
        from iobrx._fast.cibersort_fast import cibersort_fast, _init_blas
        try:
            _init_blas()
        except (RuntimeError, OSError, AttributeError) as exc:
            if backend == "rust":
                raise RuntimeError("Rust CIBERSORT needs compatible bundled OpenBLAS; use backend='python'") from exc
        else:
            return cibersort_fast(
                eset, perm=perm, QN=QN, absolute=absolute,
                abs_method=abs_method, n_threads=threads,
            )

    # Preserve the original file-input semantics in a private temporary
    # directory, including cleanup on exceptions and concurrent calls.
    import tempfile
    from pathlib import Path
    from iobrpy.workflow.cibersort import cibersort as original
    with tempfile.TemporaryDirectory(prefix="iobrx-") as directory:
        path = Path(directory) / "mixture.csv"
        eset.to_csv(path)
        return original(str(path), perm=perm, QN=QN, absolute=absolute,
                        abs_method=abs_method, n_jobs=threads)


# ---------------------------------------------------------------------------
# signature scoring
# ---------------------------------------------------------------------------
def calculate_sig_score(
    eset,
    signature,
    method: str,
    mini_gene_count: int = 3,
    adjust_eset: bool = True,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """Score gene signatures per sample (PCA / z-score / ssGSEA / integration).

    Drop-in for ``iobrpy.workflow.calculate_sig_score.calculate_sig_score``
    with shared preprocessing: the original recomputes ``preprocess_eset``
    up to three times per integration call; here it runs once. Outputs are
    bit-identical on the official imvigor210 x signature_collection gates
    (pca / zscore / ssgsea / integration, max abs diff 0.0).

    Parameters
    ----------
    eset : pandas.DataFrame
        Expression matrix, genes (index) x samples (columns).
    signature : str or list[str]
        Signature group name(s) bundled with IOBRpy, e.g.
        ``"signature_collection"`` (a str is wrapped into a one-element list,
        exactly like the original's ``signature_names`` argument).
    method : {'pca', 'zscore', 'ssgsea', 'integration'}
        Scoring method; ``integration`` concatenates PCA + z-score + ssGSEA
        columns (suffixed ``_PCA`` / ``_zscore`` / ``_ssGSEA``).
    mini_gene_count : int, default 3
        Minimum number of signature genes present in ``eset`` for the
        signature to be scored.
    adjust_eset : bool, default True
        Preprocess the matrix (log2 heuristic + finite/nonzero-sd feature
        filtering) exactly like the original.
    n_threads : int or None, optional
        Threads for the per-signature PCA leg (joblib threads) and the
        ssGSEA Rust core. ``None`` resolves to ``min(8, os.cpu_count())``.
    backend : {'auto', 'rust', 'python'}, default 'auto'
        Auto uses the optional extension, with the original workflow as
        fallback. Python explicitly selects the original IOBRpy workflow.

    Returns
    -------
    pandas.DataFrame
        One row per sample (``ID`` column first), one column per surviving
        signature; ``TMEscore_CIR`` / ``TMEscore_plus`` contrasts are appended
        when both constituents are present, matching the original.
    """
    names = [signature] if isinstance(signature, str) else list(signature)
    threads = resolve_threads(n_threads)
    if not select_backend(backend):
        from iobrpy.workflow.calculate_sig_score import calculate_sig_score as original
        return original(eset, names, method, mini_gene_count, adjust_eset, threads)
    from iobrx._fast.sig_score_fast import calculate_sig_score_fast
    return calculate_sig_score_fast(
        eset,
        names,
        method,
        mini_gene_count,
        adjust_eset,
        threads,
    )


# ---------------------------------------------------------------------------
# count -> TPM
# ---------------------------------------------------------------------------
def count2tpm(
    count_mat,
    anno_grch38=None,
    anno_gc_vm32=None,
    idType: str = "Ensembl",
    org: str = "hsa",
    source: str = "local",
    effLength_df=None,
    id_col: str = "id",
    gene_symbol_col: str = "symbol",
    length_col: str = "eff_length",
    check_data: bool = False,
    remove_version: bool = False,
):
    """Convert a raw count matrix to TPM (vectorized, bit-identical output).

    Drop-in for ``iobrpy.workflow.count2tpm.count2tpm`` — same index, column
    order, row order and float64 values on the official gate (48,058 x 10:
    max abs diff 0.0, all cells bit-identical).

    Parameters
    ----------
    count_mat : pandas.DataFrame
        Raw counts, features (index) x samples (columns). Feature identifiers
        follow ``idType`` (Ensembl IDs, Entrez IDs or gene symbols).
    anno_grch38 : pandas.DataFrame, optional
        Human gene annotation with (at least) ``id`` / ``eff_length`` /
        ``symbol`` (plus ``entrez`` / ``gc`` columns for those id types).
        When NOT a DataFrame (the default ``None``), the packaged tables
        from ``iobrpy.resources.count2tpm_data.pkl`` are used — this
        reproduces the upstream quirk that a non-DataFrame ``anno_grch38``
        is ignored in favor of the packaged tables.
    anno_gc_vm32 : pandas.DataFrame, optional
        Mouse annotation; same contract as ``anno_grch38`` (packaged table
        used when not a DataFrame).
    idType : {'Ensembl', 'entrez', 'symbol', 'mgi'}, default 'Ensembl'
        Identifier type of ``count_mat``'s index.
    org : {'hsa', 'mmus'}, default 'hsa'
        Organism.
    source : str, default 'local'
        Only ``'local'`` (packaged annotation) is implemented, as upstream.
    effLength_df : pandas.DataFrame, optional
        User-supplied effective lengths (``id_col`` / ``length_col`` /
        ``gene_symbol_col``); overrides the annotation tables when given.
    id_col, gene_symbol_col, length_col : str
        Column names inside ``effLength_df``.
    check_data : bool, default False
        Force the NA / non-numeric / infinite / zero-variance feature filter
        even when the matrix contains no NaN.
    remove_version : bool, default False
        Strip ``.N`` version suffixes from Ensembl-style index entries first.

    Returns
    -------
    pandas.DataFrame
        TPM matrix indexed by gene symbol with the original column order.
    """
    from iobrx._fast.count2tpm_fast import count2tpm as _count2tpm

    return _count2tpm(
        count_mat,
        anno_grch38=anno_grch38,
        anno_gc_vm32=anno_gc_vm32,
        idType=idType,
        org=org,
        source=source,
        effLength_df=effLength_df,
        id_col=id_col,
        gene_symbol_col=gene_symbol_col,
        length_col=length_col,
        check_data=check_data,
        remove_version=remove_version,
    )


# ---------------------------------------------------------------------------
# quanTIseq
# ---------------------------------------------------------------------------
def quantiseq(
    mix,
    data=None,
    arrays: bool = False,
    signame: str = "TIL10",
    tumor: bool = False,
    mRNAscale: bool = True,
    method: str = "lsei",
    rmgenes: str = "unassigned",
):
    """quanTIseq deconvolution (TIL10), with a memoized HGNC alias map.

    All numeric work runs in the ORIGINAL ``iobrpy.workflow.quantiseq``
    code; the only change is that the per-call HGNC alias-map rebuild
    (~2.5 s of ``iterrows`` per call) is built once per ``hgnc`` object with
    a vectorized builder producing an exactly-equal dict (same keys, values
    and insertion order). Official gate: max abs diff 0.0.

    Parameters
    ----------
    mix : pandas.DataFrame
        Mixture matrix, genes (index) x samples (columns), symbols.
    data : dict, optional
        The quanTIseq resource bundle. ``None`` loads
        ``iobrpy.resources.quantiseq_data.pkl`` once and reuses it.
    arrays : bool, default False
        Input is microarray data (affects gene mapping).
    signame : str, default 'TIL10'
        Signature name (upstream ships TIL10).
    tumor : bool, default False
        Include the tumor content fraction.
    mRNAscale : bool, default True
        Apply the mRNA-content rescaling.
    method : str, default 'lsei'
        Solver, as upstream.
    rmgenes : str, default 'unassigned'
        Genes to remove before fitting ('unassigned', 'default' or
        'none'), as upstream.

    Returns
    -------
    pandas.DataFrame
        One row per sample with the TIL10 cell-fraction columns.
    """
    from iobrx._fast.quantiseq_fast import deconvolute_quantiseq_default

    if data is None:
        data = _quantiseq_data()
    return deconvolute_quantiseq_default(
        mix,
        data=data,
        arrays=arrays,
        signame=signame,
        tumor=tumor,
        mRNAscale=mRNAscale,
        method=method,
        rmgenes=rmgenes,
    )


#: Alias matching the upstream function name.
deconvolute_quantiseq = quantiseq


# ---------------------------------------------------------------------------
# EPIC
# ---------------------------------------------------------------------------
def epic(
    bulk,
    reference=None,
    mRNA_cell=None,
    mRNA_cell_sub=None,
    sig_genes=None,
    scale_exprs: bool = True,
    with_other_cells: bool = True,
    constrained_sum: bool = True,
    range_based_optim: bool = False,
    solver: str = "SLSQP",
    init_jitter: float = 0.0,
    unlog_bulk: bool = False,
):
    """EPIC deconvolution with reference-side memoization.

    Drop-in for ``iobrpy.workflow.epic.EPIC`` with the same signature. On the
    default argument set the reference-derived work (merged references, sig
    genes, weights) is cached across calls; the per-sample solver loop is the
    upstream code, so results are bit-identical (official gate: max abs diff
    0.0 over mRNAProportions / cellFractions / fit_gof). Non-default
    arguments route to the untouched upstream implementation.

    Parameters
    ----------
    bulk : pandas.DataFrame
        Bulk expression, genes (index) x samples (columns), float64 symbols.
    reference : dict, optional
        EPIC reference dict with ``refProfiles`` / ``sigGenes`` (and optional
        ``refProfiles.var`` / ``mRNA_cell``). ``None`` loads the packaged
        ``TRef`` reference from ``iobrpy.resources.epic_TRef_BRef.pkl``.
    mRNA_cell, mRNA_cell_sub, sig_genes : optional
        As upstream (non-default values use the upstream code path).
    scale_exprs : bool, default True
        Scale counts per sample (upstream default).
    with_other_cells : bool, default True
        Report the uncharacterized-cell fraction.
    constrained_sum : bool, default True
        Constrain proportions to sum to <= 1.
    range_based_optim : bool, default False
        Upstream's range-based optimizer (forces the upstream code path).
    solver : str, default 'SLSQP'
        Upstream solver name (only used on the upstream path).
    init_jitter : float, default 0.0
        Upstream jitter (non-zero forces the upstream path).
    unlog_bulk : bool, default False
        2^x the bulk matrix first (upstream semantics).

    Returns
    -------
    dict
        ``{'mRNAProportions': DataFrame, 'cellFractions': DataFrame,
        'fit_gof': DataFrame}`` with the upstream layout.
    """
    from iobrx._fast.epic_fast import EPIC

    if reference is None:
        reference = _epic_tref()
    return EPIC(
        bulk,
        reference=reference,
        mRNA_cell=mRNA_cell,
        mRNA_cell_sub=mRNA_cell_sub,
        sig_genes=sig_genes,
        scale_exprs=scale_exprs,
        with_other_cells=with_other_cells,
        constrained_sum=constrained_sum,
        range_based_optim=range_based_optim,
        solver=solver,
        init_jitter=init_jitter,
        unlog_bulk=unlog_bulk,
    )


# ---------------------------------------------------------------------------
# MCP-counter
# ---------------------------------------------------------------------------
def mcpcounter(eset, features_type: str = "HUGO_symbols"):
    """MCP-counter cell population scoring (vectorized, bit-exact).

    Drop-in for ``iobrpy.workflow.mcpcounter.MCPcounter_estimate``. Marker
    loading is cached per ``features_type``; the gene selection keeps the
    exact upstream ``expression.index.intersection(genes)`` call so row order
    is unchanged. Official gate: max abs diff 0.0.

    Parameters
    ----------
    eset : pandas.DataFrame
        Expression matrix, genes (index) x samples (columns).
    features_type : {'HUGO_symbols', 'ENTREZ_ID', 'affy133P2_probesets'}, \
default 'HUGO_symbols'
        Identifier type of ``eset``'s index (as upstream).

    Returns
    -------
    pandas.DataFrame
        One row per cell population, columns = ``eset`` columns (the
        upstream transposed layout).
    """
    from iobrx._fast.mcpcounter_fast import MCPcounter_estimate

    return MCPcounter_estimate(expression=eset, features_type=features_type)


# ---------------------------------------------------------------------------
# ESTIMATE
# ---------------------------------------------------------------------------
def estimate_score(eset, platform: str = "affy"):
    """ESTIMATE stromal / immune scores and tumor purity (vectorized).

    Drop-in for ``iobrpy.workflow.estimate.estimate_score``. The
    per-column rank normalization is vectorized (bit-equal to the upstream
    per-column loop), and resource loading is cached instead of re-read from
    disk on every call. Official gate: max abs diff 0.0.

    Parameters
    ----------
    eset : pandas.DataFrame
        Expression matrix, genes (index) x samples (columns).
    platform : {'affy', 'affymetrix', ...}, default 'affy'
        Only the exact spelling ``'affymetrix'`` adds ``TumorPurity``, as
        upstream. The legacy ``'affy'`` default returns the three scores.

    Returns
    -------
    pandas.DataFrame
        Rows ``StromalSignature`` / ``ImmuneSignature`` / ``ESTIMATEScore``
        (+ ``TumorPurity`` for ``platform='affymetrix'``), columns = samples.
    """
    from iobrx._fast.estimate_fast import estimate_score as _estimate_score

    return _estimate_score(input_df=eset, platform=platform)


# ---------------------------------------------------------------------------
# annotation / ID conversion
# ---------------------------------------------------------------------------
def anno_eset(
    eset,
    annotation="anno_grch38",
    symbol: str = "symbol",
    probe: str = "id",
    method: str = "mean",
):
    """Aggregate probes to gene symbols (vectorized, bit-exact).

    Drop-in for ``iobrpy.workflow.anno_eset.anno_eset``. Duplicated probes
    are expanded, scored (mean/sd/sum across samples) and collapsed to the
    best-scoring row per symbol with the upstream sort/tie semantics
    (including the quicksort tie order and the strided mean summation
    order). Official gate (60,483 x 10 Ensembl matrix): max abs diff 0.0.

    Parameters
    ----------
    eset : pandas.DataFrame
        Expression matrix, probe/feature ids (index) x samples (columns).
    annotation : str or pandas.DataFrame, default 'anno_grch38'
        Built-in resource key (``'anno_grch38'``, ``'anno_rnaseq'``,
        ``'anno_hug133plus2'``, ``'anno_illumina'``) or a user-supplied
        annotation DataFrame containing the ``symbol`` / ``probe`` columns.
    symbol : str, default 'symbol'
        Gene-symbol column name in the annotation.
    probe : str, default 'id'
        Probe-id column name in the annotation.
    method : {'mean', 'sd', 'sum'}, default 'mean'
        Scoring rule for resolving multiple probes per symbol.

    Returns
    -------
    pandas.DataFrame
        Symbol-indexed expression matrix with the original sample columns.
    """
    from iobrx._fast.anno_eset_fast import anno_eset as _anno_eset

    return _anno_eset(eset, annotation, symbol=symbol, probe=probe, method=method)


# ---------------------------------------------------------------------------
# BayesPrism
# ---------------------------------------------------------------------------
def bayesprism(
    bulk,
    sc_dat=None,
    cell_state_labels=None,
    cell_type_labels=None,
    key: str = "Malignant_cells",
    out_dir=None,
    n_threads: int | None = None,
    backend: str = "auto",
    state_order: str = "sorted",
    outlier_cut: float = 0.01,
    outlier_fraction: float = 0.1,
    pseudo_min: float = 1e-8,
    gibbs_control: dict | None = None,
    opt_control: dict | None = None,
):
    """BayesPrism deconvolution (accelerated, determinism-fixed).

    Bit-exact drop-in for ``iobrpy.bayesprism.bayesprism.run_bayesprism``
    (the ``python -m iobrpy.main bayesprism`` CLI): with
    ``state_order='legacy'`` under the same ``PYTHONHASHSEED`` the three
    output CSVs are BYTE-identical to the original (verified: SHA-256 of
    theta.csv / theta_cv.csv / Z_tumor.csv equal to a frozen
    PYTHONHASHSEED=0 gold-standard run on the 20-sample imvigor210 x
    bundled BP_data configuration). The Gibbs sampler call sequence
    (per-gene sequential ``rng.multinomial``, ``rng.gamma`` Dirichlet,
    ``SeedSequence(123).spawn(n_samples)`` per-sample child streams, the
    phase-3 shared ``spawn(1)[0]`` seed quirk, fork-Pool per-sample
    parallelism) is preserved verbatim; the non-sampling hot spots
    (``select_gene_type``'s O(n^2) scan, ``cleanup_genes``'s boolean-frame
    sums, ``collapse``, ``merge_K``, the 60 MB sc_dat.csv parse) are
    vectorized/cached with bit-identical results.

    One documented determinism FIX vs upstream: the original builds
    ``map_ = {cell_type: list(set(states))}`` whose string-set iteration
    order follows the per-process ``PYTHONHASHSEED``, so the original's
    outputs are NOT reproducible across processes (merge-order ULPs
    occasionally flip discrete Gibbs draws). ``state_order='sorted'``
    (default) fixes the merge order semantically: iobrx outputs are then
    byte-identical across processes WITHOUT pinning the hash seed
    (verified), and on the frozen gold input theta/theta_cv come out
    100% bit-exact vs the PYTHONHASHSEED=0 original with Z_tumor within
    5.7e-14 abs (ULP-only, 0 discrete flips).

    Parameters
    ----------
    bulk : pandas.DataFrame or path-like
        Bulk expression matrix, GENES (index) x SAMPLES (columns) — the
        orientation of the CLI's bulk csv. Paths are parsed exactly like
        the original (sep by suffix, ``.gz`` support, ``index_col=0``);
        the frame is transposed and ``astype(np.int32)`` internally, as
        upstream.
    sc_dat : pandas.DataFrame or path-like, optional
        Single-cell reference count matrix, CELLS (index) x GENES
        (columns) — the upstream CLI help says "genes x cells" but the
        CODE requires cells x genes (rows align with the label files).
        ``None`` uses the bundled ``iobrpy.bayesprism/BP_data/sc_dat.csv``
        (original csv float semantics on first read, then process-memoized
        and pickle-cached under ``$IOBRX_CACHE`` or
        ``~/.cache/iobrx/bayesprism``).
    cell_state_labels, cell_type_labels : list-like or path-like, optional
        One label per reference cell (row order of ``sc_dat``). Paths are
        read header-less like the original; ``None`` uses the bundled
        BP_data label files.
    key : str, default 'Malignant_cells'
        Tumor cell type (as the CLI ``--key``); must be present in
        ``cell_type_labels`` when a custom reference is given.
    out_dir : path-like, optional
        When given, write ``theta.csv`` / ``theta_cv.csv`` / ``Z_tumor.csv``
        with the original output contract (``_BayesPrism`` column suffixes;
        ``Z_tumor.csv`` via ``xarray.to_pandas``, index name ``bulk_id``).
    n_threads : int or None, optional
        Worker processes for the per-sample Gibbs pools and the per-cell-type
        MAP optimization. ``None`` resolves to ``min(8, os.cpu_count())``
        (see :func:`set_threads`). The worker count does NOT affect output
        bits (seeds are assigned per sample up front; ``starmap`` preserves
        order — verified 8 vs 16 workers byte-identical).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        'python' runs the UNTOUCHED original iobrpy stages (including the
        hash-seed-dependent ``list(set(...))`` merge order). 'auto'/'rust'
        both take the accelerated pure-Python path — a native Gibbs kernel
        is a follow-up round; the value is still validated by
        ``select_backend``.
    state_order : {'sorted', 'legacy'}, default 'sorted'
        'sorted' = determinism fix (see above); 'legacy' reproduces the
        original ``list(set(states))`` expression verbatim (byte-identical
        to an original run only within the same ``PYTHONHASHSEED``).
    outlier_cut, outlier_fraction : float, default 0.01, 0.1
        Bulk outlier-gene filter (original ``Prism.new`` defaults).
    pseudo_min : float, default 1e-8
        Reference normalization pseudo-count (original default).
    gibbs_control, opt_control : dict, optional
        Passthrough to ``Prism.run`` (the CLI does not expose these), e.g.
        ``{'chain.length': 100, 'burn.in': 50, 'thinning': 2}``. Defaults
        are upstream's (chain 1000 / burn-in 500 / thinning 2 / seed 123 /
        generator backend / sequential multinomial; MAP optimizer / sigma 2).
        ``n.cores`` is filled from ``n_threads``. Upstream quirks preserved
        (e.g. ``optimizer='MLE'`` is broken under scipy>=1.16 — untouched).

    Returns
    -------
    dict
        ``{'theta': DataFrame, 'theta_cv': DataFrame, 'Z_tumor': DataFrame}``
        — exactly the frames written to the output CSVs: theta/theta_cv are
        samples x cell types with ``_BayesPrism``-suffixed columns;
        Z_tumor is samples x genes (index name ``bulk_id``), the INITIAL
        (pre-reference-update) tumor deconvolution, as upstream.
    """
    threads = resolve_threads(n_threads)
    if backend == "rust":
        from iobrx._backend import select_backend
        if select_backend("rust"):
            from iobrx._fast.bayesprism_gibbs_rust import bayesprism_rust

            return bayesprism_rust(
                bulk,
                sc_dat=sc_dat,
                cell_state_labels=cell_state_labels,
                cell_type_labels=cell_type_labels,
                key=key,
                out_dir=out_dir,
                n_threads=threads,
                state_order=state_order,
                outlier_cut=outlier_cut,
                outlier_fraction=outlier_fraction,
                pseudo_min=pseudo_min,
                gibbs_control=gibbs_control,
                opt_control=opt_control,
            )
    from iobrx._fast.bayesprism_fast import bayesprism as _bayesprism

    return _bayesprism(
        bulk,
        sc_dat=sc_dat,
        cell_state_labels=cell_state_labels,
        cell_type_labels=cell_type_labels,
        key=key,
        out_dir=out_dir,
        n_threads=threads,
        backend=backend,
        state_order=state_order,
        outlier_cut=outlier_cut,
        outlier_fraction=outlier_fraction,
        pseudo_min=pseudo_min,
        gibbs_control=gibbs_control,
        opt_control=opt_control,
    )


# ---------------------------------------------------------------------------
# tme_profile orchestrator
# ---------------------------------------------------------------------------
def tme_profile(
    input,
    output,
    threads: int = 1,
    *,
    signature="all",
    sig_method: str = "integration",
    mini_gene_count: int = 2,
    adjust_eset: bool = True,
    perm: int = 100,
    QN: bool = True,
    absolute: bool = False,
    abs_method: str = "sig.score",
    platform: str = "affymetrix",
    features: str = "HUGO_symbols",
    arrays: bool = True,
    signame: str = "TIL10",
    tumor: bool = True,
    mRNAscale: bool = True,
    quantiseq_method: str = "lsei",
    rmgenes: str = "unassigned",
    epic_reference: str = "TRef",
    data_type: str = "tpm",
    id_type: str = "symbol",
    cancer_type: str = "pancan",
    lr_verbose: bool = True,
    n_threads: int | None = None,
    backend: str = "auto",
    cibersort_backend: str = "original",
    parallel: bool = False,
    sig_procs: int | None = None,
    verbose: bool = True,
):
    """Run the whole tme_profile chain (sig scores + 6 deconvolutions + LR) in one process.

    Bit-exact drop-in for ``iobrpy.workflow.tme_profile`` (the
    ``iobrpy tme_profile`` CLI orchestrator): same directory layout and same
    output BYTES for ``01-signatures/calculate_sig_score.csv``, the six
    ``02-tme/*_results.csv``, ``02-tme/deconvo_merged.csv`` and
    ``03-LR_cal/lr_cal.csv`` — verified against the frozen TPM_stad10 gold
    run (7/9 files raw-byte identical; cibersort_results.csv and
    deconvo_merged.csv identical in every column EXCEPT
    ``P-value_CIBERSORT``, whose upstream permutations draw OS entropy and
    are not reproducible run-to-run by design — the same exception the
    original has against itself).  The nine serial ``subprocess`` cold starts
    of the CLI (~1.4-2.2 s import overhead each) are eliminated: fast
    sub-steps run through the iobrx bit-exact kernels, IPS / LR_cal / merge /
    mcpcounter-preprocess call the ORIGINAL code in-process, and the
    cibersort step calls the ORIGINAL solver on the ORIGINAL file path (see
    ``cibersort_backend``).

    Parameters mirror the defaults the original orchestrator injects into
    each sub-command's CLI (which differ from the workflow functions' own
    defaults — e.g. LR_cal CLI ``data_type='tpm'`` vs function ``'count'``,
    quantiseq CLI-injected ``arrays/tumor/scale_mrna``, estimate
    ``platform='affymetrix'``, mcpcounter ``features='HUGO_symbols'``, epic
    ``reference='TRef'``, sig_score ``signature='all'``,
    ``method='integration'``, ``mini_gene_count=2``, ``adjust_eset`` on).
    The CLI's free-form flag router (FLAG_BUCKETS / named blocks) maps 1:1
    onto these keyword parameters.

    Parameters
    ----------
    input : str or Path
        TPM matrix file (genes x samples; CSV/TSV, extension-driven sep
        inference per sub-step exactly as the original CLIs do).  File-based
        by design: each original sub-step parses the input with its OWN
        reader (engine/sep/cleaning differ), and pandas' float parsing is
        not correctly rounded on 17-digit text, so a shared in-memory frame
        is NOT byte-equivalent to the per-step file reads.
    output : str or Path
        Output directory; ``01-signatures/``, ``02-tme/``, ``03-LR_cal/`` are
        created as in the original.
    threads : int, default 1
        Mirrors ``--threads``: feeds calculate_sig_score ``parallel_size``
        and cibersort ``n_jobs`` only, as upstream.
    signature, sig_method, mini_gene_count, adjust_eset
        calculate_sig_score step parameters (CLI-injected defaults).
    perm, QN, absolute, abs_method : cibersort step (CLI defaults 100/True/False/'sig.score').
    platform : estimate step ('affymetrix', CLI-injected).
    features : mcpcounter step ('HUGO_symbols', CLI-injected).
    arrays, signame, tumor, mRNAscale, quantiseq_method, rmgenes
        quantiseq step (CLI-injected ``--arrays --tumor --scale_mrna``;
        module defaults TIL10/lsei/'unassigned').
    epic_reference : epic step ('TRef', CLI-injected).
    data_type, id_type, cancer_type, lr_verbose
        LR_cal step (CLI-injected ``tpm``/``symbol``/``pancan``/``--verbose``).
    n_threads : int or None, optional
        Repo-convention override: when not None it wins over ``threads``
        (resolved via :func:`resolve_threads`).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        Backend for the calculate_sig_score step (the only remaining
        iobrx-routed heavy kernel; rust ssGSEA/PCA legs are gate-verified and
        byte-verified on this chain's frozen input).
    cibersort_backend : {'original', 'python', 'auto', 'rust'}, default 'original'
        'original' calls ``iobrpy.workflow.cibersort.cibersort`` in-process
        on the input path — the only byte-exact route on the frozen input:
        the native backend deviates on a knife-edge sample (tie-heavy
        mixture column at the solver tolerance edge; 13 cells, max 2.5e-3,
        deterministic), and the 'python' fallback's DataFrame->temp-CSV
        round trip is not value-preserving (pandas float parsing shifts
        2,830/501,810 cells ~1 ulp on 17-digit text), which flips the same
        sample.  'auto'/'rust' opt into the native solver (faster, seeded
        P-values); 'python' into the iobrx fallback.  Details in
        ``iobrx._fast.tme_profile_fast``'s module docstring.
    parallel : bool, default False
        Extra mode (bytes unchanged, verified against the frozen gold):
        splits the integration legs of calculate_sig_score across
        ``sig_procs`` worker processes (the ~120 s GIL-bound per-signature
        pandas glue becomes truly parallel) and overlaps the six light
        sub-steps in a thread pool (IPS stays on the calling thread — its
        ``sys.argv`` bridge is process-global).  ~3.5x wall-clock on the
        frozen input (43.97 s vs 152.98 s original median).
    sig_procs : int or None, optional
        Worker processes for the split sig_score legs in parallel mode
        (default ``max(2, min(threads, 12))``; peak process/thread count
        stays <= threads + 6, within the 32-thread discipline at
        threads<=16).
    verbose : bool, default True
        Print per-step ``[ok]`` timing lines (stdout only; files unaffected).

    Returns
    -------
    dict
        ``{step_name: output_path}`` for the nine artifacts (the original
        CLI returns None; paths are also the contract layout on disk).

    Notes
    -----
    Hard-stop semantics match the original: the first failing step raises
    and the chain aborts (no resume), like ``assert rc == 0`` per step.
    """
    from iobrx._fast.tme_profile_fast import tme_profile_fast

    eff_threads = resolve_threads(n_threads) if n_threads is not None else int(threads)
    return tme_profile_fast(
        input,
        output,
        eff_threads,
        signature=signature,
        sig_method=sig_method,
        mini_gene_count=mini_gene_count,
        adjust_eset=adjust_eset,
        perm=perm,
        QN=QN,
        absolute=absolute,
        abs_method=abs_method,
        platform=platform,
        features=features,
        arrays=arrays,
        signame=signame,
        tumor=tumor,
        mRNAscale=mRNAscale,
        quantiseq_method=quantiseq_method,
        rmgenes=rmgenes,
        epic_reference=epic_reference,
        data_type=data_type,
        id_type=id_type,
        cancer_type=cancer_type,
        lr_verbose=lr_verbose,
        backend=backend,
        cibersort_backend=cibersort_backend,
        parallel=parallel,
        sig_procs=sig_procs,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# NMF clustering
# ---------------------------------------------------------------------------
def nmf(
    data,
    kmin: int = 2,
    kmax: int = 8,
    features: str | None = None,
    log1p: bool = False,
    normalize: bool = False,
    shift: float | None = None,
    random_state: int = 42,
    max_iter: int = 1000,
    skip_k_2: bool = False,
    plot: bool = False,
    outdir=None,
    verbose: bool = False,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """NMF clustering with silhouette-based k selection (lazy-import, bit-exact).

    Bit-exact port of ``iobrpy.workflow.nmf`` — the ``iobrpy nmf`` CLI step,
    which upstream exposes only as ``main()`` + argparse. On the official
    frozen gate (cibersort_stad10.csv, ``--kmin 2 --kmax 10 --features 1:22
    --max-iter 10000 --skip_k_2 --random-state 42``) the written
    ``clusters.csv``, ``top_features_per_cluster.csv`` and ``pca_plot.png``
    are byte-identical (sha256 equal) to the original CLI's, and
    ``verbose=True`` reproduces its stdout line for line. The k scan
    (``NMF(n_components=k, init='nndsvda', ...)`` with every other sklearn
    default, argmax(W) labels, ``silhouette_score`` selection with ties ->
    smaller k), the preprocessing chain (1-based inclusive ``features``
    column range -> log1p -> L1 row normalize -> non-negativity shift) and
    the plotting helpers are copied verbatim; unlike the CLI (which pays the
    full ``iobrpy.main`` import chain and always imports matplotlib), this
    API loads sklearn only on call and matplotlib only when ``plot=True``.

    Parameters
    ----------
    data : pandas.DataFrame or path-like
        Samples x features matrix with sample IDs as index (the content of
        the CLI's ``-i`` file). A path is read with the original
        ``read_matrix`` semantics — comma first, then a literal four-space
        ``sep`` fallback (the upstream quirk that cannot parse real tabs).
    kmin, kmax : int, default 2, 8
        Inclusive k-scan range. ``kmax >= n_samples`` is clamped to
        ``n_samples - 1`` as upstream; k < 2, k == 2 (with ``skip_k_2``)
        and k >= n_samples are skipped.
    features : str or None, default None
        Column window for the feature block: ``'m-n'`` / ``'m:n'`` / single
        ``'m'``, 1-based inclusive (the CLI's ``--features``); ``end``
        clamps to the column count; invalid ranges raise ValueError. None
        uses all columns.
    log1p : bool, default False
        Apply ``np.log1p`` after column selection, before ``normalize``.
    normalize : bool, default False
        Apply ``sklearn.preprocessing.normalize(X, norm='l1', axis=1)``.
    shift : float or None, default None
        Only when ``min(X) < 0``: add ``abs(min) + shift`` (upstream:
        ``--shift`` is ignored on non-negative data). Negative data without
        a shift raises ValueError with the upstream message.
    random_state : int, default 42
        Passed to NMF/PCA as upstream. Discrete outputs are seed-invariant
        (nndsva init + shuffle=False); floats may differ ~1e-15 across
        seeds exactly as in the original.
    max_iter : int, default 1000
        NMF iteration cap (the docs example uses 10000).
    skip_k_2 : bool, default False
        Skip k=2 in the scan (the CLI's ``--skip_k_2``).
    plot : bool, default False
        Draw ``pca_plot.png`` (PCA of W + MinCovDet(random_state=0) 90%
        ellipses, dpi=300, bbox_inches='tight') into ``outdir`` —
        byte-identical to the CLI's plot, which is always drawn. Requires
        ``outdir``; matplotlib is imported only here.
    outdir : path-like or None, default None
        When given, write ``clusters.csv`` and
        ``top_features_per_cluster.csv`` (plus ``pca_plot.png`` when
        ``plot=True``) byte-identically to the CLI. BUG-COMPAT: as
        upstream, the top-features file is written *before* the directory
        is created and the failure is swallowed — with a non-existent
        ``outdir`` it is silently missing while the other files are
        written. ``None`` keeps everything in memory (the returned frames
        are complete regardless).
    verbose : bool, default False
        Reproduce the CLI's stdout (per-k silhouette/rec_err, warnings,
        banner). The CLI always prints.
    n_threads : int or None, optional
        Accepted and validated for API uniformity, but the fit deliberately
        runs at ambient OpenMP/BLAS threading like the original CLI —
        pinning threads can change reduction order in the CD solver and
        break bit-exactness. The workload is small; nothing here benefits
        from threads.
    backend : {'auto', 'rust', 'python'}, default 'auto'
        nmf has no native kernel (pure-Python bit-exact port): ``'rust'``
        raises RuntimeError; ``'auto'`` / ``'python'`` run the same code.

    Returns
    -------
    dict
        ``clusters`` (DataFrame equal to ``clusters.csv``: sample, cluster,
        preprocessed feature columns), ``top_features`` (DataFrame equal to
        ``top_features_per_cluster.csv``: per-cluster top-15 H-row feature
        names, index name ``cluster``), ``best_k`` / ``best_silhouette`` /
        ``best_rec_err``, ``k_results`` (``{k: (silhouette, rec_err)}`` of
        the whole scan, incl. ``-inf`` entries), ``labels`` (0-based
        argmax(W)), ``W`` / ``H`` / ``model`` (best-k refit),
        ``pca_coords`` (PCA(n_components=2) of W, or None on failure),
        ``used_features`` (preprocessed matrix, original index/columns),
        ``outdir`` (abspath or None).

    Notes
    -----
    CLI ``sys.exit(1)`` paths raise instead: ``features`` parse errors and
    "no valid k in range" become ValueError / RuntimeError with the same
    message text. Reconstruction error is recorded per k but never
    participates in k selection, as upstream.
    """
    from iobrx._fast.nmf_fast import nmf_cluster

    return nmf_cluster(
        data, kmin=kmin, kmax=kmax, features=features, log1p=log1p,
        normalize=normalize, shift=shift, random_state=random_state,
        max_iter=max_iter, skip_k_2=skip_k_2, plot=plot, outdir=outdir,
        verbose=verbose, n_threads=n_threads, backend=backend,
    )


# ---------------------------------------------------------------------------
# merge salmon quant.sf
# ---------------------------------------------------------------------------
def merge_salmon(
    path_salmon,
    project,
    num_processes: int | None = None,
    n_threads: int | None = None,
    backend: str = "auto",
    verbose: bool = True,
):
    """Merge Salmon ``quant.sf`` dirs into TPM / count matrices, accelerated.

    Bit-exact drop-in for ``iobrpy.workflow.merge_salmon`` (CLI
    ``python -m iobrpy.main merge_salmon``): recursively discovers
    ``quant.sf`` under *path_salmon*, reads ``Name`` / ``TPM`` /
    ``NumReads`` (sample name = parent directory name) and writes
    ``<project>_salmon_tpm.tsv.gz`` and ``<project>_salmon_count.tsv.gz``
    (gzip level-5 TSVs, first column ``Name``) back into *path_salmon*.
    Decompressed contents are byte-identical to the original on the frozen
    8 x 60,000-transcript gate (480,000 cells per matrix bit-identical,
    max abs diff 0.0, aligned-content sha equal); output columns are
    deterministically sorted by sample name — the ORIGINAL's
    ``as_completed`` column order is nondeterministic run-to-run.

    Parameters
    ----------
    path_salmon : str
        Root directory searched recursively for ``quant.sf``; also the
        output destination (upstream forces output dir == input dir).
    project : str
        Output filename prefix.
    num_processes : int, optional
        Loader threads (upstream flag name). ``None`` → ``n_threads`` →
        ``os.cpu_count()``; capped at 32. Governs the fallback pandas
        loader, the dual-stream gzip writer, and (additionally capped at
        16) the Rust parse threads. The sequential pure-Python parser is
        deliberately single-threaded (its C-level byte surgery is GIL-bound
        and measured faster than any thread count: 0.43 s vs 2.20 s at 8
        threads on the 8-file gate).
    n_threads : int, optional
        iobrx-wide alias for ``num_processes``; ``num_processes`` wins
        when both are given.
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``auto`` runs the accelerated implementation with the native Rust
        read path when ``_rust.merge_salmon_parse`` is importable, else the
        sequential pure-Python fast parser; both fall back to the original
        pandas statements per input whenever their equivalence guards trip
        (bug-compatible errors, outer-join union semantics and int64 dtype
        quirks included), and all engines produce byte-identical files.
        ``rust`` forces the native read path (RuntimeError when the
        extension or the kernel is missing). ``python`` runs the untouched
        upstream CLI ``main()`` via argv rewrite. The pandas ``to_csv`` +
        gzip write path is common to the accelerated engines (the parity
        contract pins the upstream serialization bytes).
    verbose : bool, default True
        Reproduce the upstream console protocol (head previews, "Saving…"
        lines, output paths, IOBRpy banner).

    Returns
    -------
    (tpm_df, cnt_df) : tuple of pandas.DataFrame
        The matrices as written: transcripts (index name ``Name``) x
        samples (sorted columns), float64. ``(None, None)`` when no
        ``quant.sf`` was found (upstream prints a notice and writes
        nothing) or when ``backend='python'`` (upstream ``main()``
        returns nothing).
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    workers = num_processes if num_processes is not None else n_threads
    if backend == "python":
        from iobrx._fast.merge_salmon_fast import merge_salmon_original
        return merge_salmon_original(path_salmon, project, workers)
    if backend == "rust":
        select_backend("rust")  # house RuntimeError when the extension is missing
    from iobrx._fast.merge_salmon_fast import merge_salmon as _merge_salmon
    return _merge_salmon(path_salmon, project, workers, verbose=verbose,
                         engine=("rust" if backend == "rust" else "auto"))


# ---------------------------------------------------------------------------
# TME clustering
# ---------------------------------------------------------------------------
def tme_cluster(
    df,
    id: str | None = None,
    features: str | None = None,
    pattern: str | None = None,
    scale: bool = True,
    min_nc: int = 2,
    max_nc: int = 6,
    nstart: int = 10,
    max_iter: int = 10,
    tol: float = 1e-4,
    seed: int = 123,
    print_result: bool = False,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """TME k-means clustering with KL-index best-k selection (Rust core).

    Bit-exact drop-in for ``iobrpy.workflow.tme_cluster`` (the
    ``python -m iobrpy.main tme_cluster`` stage) on an in-memory table:
    the returned frame is exactly what the original CLI writes to its
    output file — ``ID``, ``cluster`` (``TME1``..``TMEk``), then the
    z-scored feature columns that survived the upstream
    ``feature_manipulation`` filter — and ``out.to_csv(path, sep=...,
    index=False)`` reproduces that file byte-for-byte (frozen campaign
    gates: 375x235 TME matrix and 10x22 CIBERSORT weights matrix,
    sha256-identical; per-k labels, withinss and KL values bit-identical).

    Parameters
    ----------
    df : pandas.DataFrame
        The parsed input table, samples x features with a sample-ID column
        (the first column by default, or the one named by ``id``) — exactly
        what ``pd.read_csv`` returns for the original CLI's input file.
    id : str, optional
        Sample-ID column name (upstream ``--id``); default: first column.
    features : str, optional
        1-based inclusive column range ``'m:n'`` (upstream ``--features``,
        applied after the ID column is dropped; only the ``m:n`` form is
        accepted, as upstream).
    pattern : str, optional
        Regex selecting feature columns by name (upstream ``--pattern``;
        ``features`` takes precedence).
    scale : bool, default True
        Z-score each feature (``std(ddof=1)``) and drop NA / non-numeric /
        infinite / zero-variance columns, as upstream ``--scale`` /
        ``--no-scale``.
    min_nc, max_nc : int, default 2, 6
        Inclusive k-means k-scan range (upstream ``--min_nc`` /
        ``--max_nc``). The KL-index neighbor k's (``min_nc-1`` when >= 1,
        and ``max_nc+1``) are run automatically with upstream's derived
        seeds (``RandomState(seed + 9973*k)``).
    nstart : int, default 10
        Random restarts per k; the best run is the strict minimum
        within-cluster sum of squares (first wins on ties), as upstream
        ``--nstart``.
    max_iter : int, default 10
        Hartigan-Wong move-loop iteration cap (upstream ``--max_iter``).
    tol : float, default 1e-4
        Strict improvement threshold for moving a point (upstream
        ``--tol``).
    seed : int, default 123
        ``np.random.RandomState`` seed for the initial-centre draws
        (upstream ``--seed``; the CLI front-end pins it to 123).
    print_result : bool, default False
        Print the per-k KL values, the chosen best k and the cluster
        counts (upstream ``--print_result``).
    n_threads : int or None, optional
        Rayon threads for the parallel nstart evaluation (results are
        thread-count invariant; the selection order is preserved).
        ``None`` resolves to ``min(8, os.cpu_count())`` (see
        :func:`set_threads`).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        'auto'/'rust' use the native core; 'python' runs the ORIGINAL
        ``iobrpy.workflow.tme_cluster`` functions in memory (bit-identical,
        ~10x slower on the frozen gates).

    Returns
    -------
    pandas.DataFrame
        ``ID``, ``cluster``, then the (scaled) feature columns that took
        part in the clustering, in the upstream row order.
    """
    threads = resolve_threads(n_threads)
    if select_backend(backend):
        from iobrx._fast.tme_cluster_fast import tme_cluster_fast

        return tme_cluster_fast(
            df, id=id, features=features, pattern=pattern, scale=scale,
            min_nc=min_nc, max_nc=max_nc, nstart=nstart, max_iter=max_iter,
            tol=tol, seed=seed, print_result=print_result, n_threads=threads,
        )
    from iobrx._fast.tme_cluster_fast import tme_cluster_reference

    return tme_cluster_reference(
        df, id=id, features=features, pattern=pattern, scale=scale,
        min_nc=min_nc, max_nc=max_nc, nstart=nstart, max_iter=max_iter,
        tol=tol, seed=seed, print_result=print_result,
    )


# ---------------------------------------------------------------------------
# LR pairs (ligand-receptor)
# ---------------------------------------------------------------------------
def lr_cal(
    eset,
    output_file=None,
    data_type: str = "count",
    id_type: str = "ensembl",
    cancer_type: str = "pancan",
    verbose: bool = False,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """Ligand-receptor pair expression matrix (min of log2 TPM), accelerated.

    Bit-exact drop-in for ``iobrpy.workflow.LR_cal.LR_cal``: output CSVs are
    byte-identical (sha256-equal) on the frozen official inputs — stad TPM
    54,658x10 -> 10x773 (``a49ddff5...``), eset_stad_symbol 50,181x10 ->
    10x773, imvigor210 872x348 -> 348x1 and the ENSG eset_stad degenerate
    10x0 case. The ORIGINAL's 81% hotspot (four per-gene pandas filter
    passes over 54,658 genes) becomes one Rust pass
    (``_rust.lr_gene_valid_mask``) replicating pandas' exact NaN / infinite /
    zero-variance semantics (nanvar two-pass with numpy pairwise sums); the
    905-pair loop becomes an ``np.fmin`` gather (bitwise == the ORIGINAL's
    skipna ``min(axis=0)``); the group_lrpairs merge (905 -> 813 -> 773)
    keeps the ORIGINAL's list semantics with column-position tracking; the
    tail (``insert(0, 'ID', ...)`` + ``to_csv``) runs the ORIGINAL
    statements, so the CSV text is byte-identical.

    Parameters
    ----------
    eset : str, path-like or pandas.DataFrame
        Expression matrix, genes (index) x samples (columns), HGNC-symbol
        index expected; or a path to it (``.tsv``/``.txt`` -> tab, anything
        else comma, exactly like the ORIGINAL ``detect_sep``). Linear-scale
        TPM is expected — the module always applies ``log2(x + 1)``
        internally (feeding log2-scale data double-logs it, as upstream).
    output_file : str, optional
        Output CSV/TSV path (separator chosen by extension, as the
        ORIGINAL). When None, nothing is written. The ORIGINAL takes it as
        a required positional; iobrx makes it optional.
    data_type : {'count', 'tpm'}, default 'count'
        As the ORIGINAL API (whose default is 'count' while its CLI defaults
        to 'tpm'). BUG-COMPATIBLE: 'count' performs the ORIGINAL's broken
        ``count2tpm(df, idType=..., org='hsa', source='local')`` call and
        raises the identical ``TypeError: count2tpm() missing 2 required
        positional arguments: 'anno_grch38' and 'anno_gc_vm32'`` (an iobrpy
        0.2.0 defect, deliberately not fixed here; it calls the REAL
        count2tpm, so an upstream fix propagates automatically).
    id_type : str, default 'ensembl'
        Passed to the (broken) count branch; unused on the tpm path.
    cancer_type : str, default 'pancan'
        Intercellular-network key (18 available, incl. per-cancer networks).
        An invalid key raises KeyError after gene filtering, as upstream.
    verbose : bool, default False
        Print the ORIGINAL's filter-count and ``[LOG]`` lines (the tqdm bar
        and the decorative banner are dropped, per iobrx convention).
    n_threads : int or None, optional
        Rayon threads for the Rust gene filter. ``None`` resolves to
        ``min(8, os.cpu_count())`` (see :func:`set_threads`). The per-row
        computation is pure, so results are thread-count invariant.
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``'python'`` runs the ORIGINAL ``compute_LR_pairs`` core;
        ``'auto'``/``'rust'`` use the accelerated core, which itself falls
        back to the ORIGINAL for frames the exactness guards reject
        (non-float64/int64 dtypes, duplicate gene labels or sample names).

    Returns
    -------
    pandas.DataFrame
        Samples x surviving LR pair/combo columns with the leading ``ID``
        column — the same frame the ORIGINAL writes to ``output_file``
        (the ORIGINAL itself returns None).
    """
    threads = resolve_threads(n_threads)
    from iobrx._fast.lr_cal_fast import LR_cal as _lr_cal

    engine = "fast" if select_backend(backend) else "original"
    return _lr_cal(
        eset,
        output_file=output_file,
        data_type=data_type,
        id_type=id_type,
        cancer_type=cancer_type,
        verbose=verbose,
        n_threads=threads,
        engine=engine,
    )


# ---------------------------------------------------------------------------
# IPS (immunophenoscore)
# ---------------------------------------------------------------------------
def ips(
    eset,
    output_file=None,
    verbose: bool = False,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """Immunophenoscore (Charoentong 2017 four-block design), lazy-import port.

    Bit-exact drop-in for the ``iobrpy.workflow.IPS`` CLI stage
    (``python -m iobrpy.main IPS``): the output CSV is byte-identical
    (sha256-equal) on the frozen official gate — stad symbol-TPM
    54,658x10 (count2tpm chain) -> 10x6 ``793b4718...``. Every numeric
    statement runs verbatim (pandas ``reindex``/``mean``/``std(ddof=1)``/
    ``groupby.agg`` whose Kahan-compensated group means naive numpy does
    not reproduce bit-for-bit, the ``max > 100 -> log2(x+1)`` heuristic,
    the POSITIONAL block slices ``MHC=wg[0:10] / CP=wg[10:20] /
    EC=wg[20:24] / SC=wg[24:26]`` and banker's-rounding
    ``int(round(AZ*10/3))``); the acceleration is eliminating the
    original's ~1.42 s ``iobrpy.main`` full-import cold-start floor
    (>95% of its wall clock — the per-sample loop itself costs only
    ~1.9 ms/sample), via lazy pandas/numpy imports and a per-process
    memoized ``IPS_genes.txt`` resource frame.

    BUG-COMPATIBLE (upstream defects preserved): missing NAME groups
    silently misalign the positional slices, and a wholly missing EC or
    SC block raises the upstream ``ValueError: cannot convert float NaN
    to integer`` (empty-slice ``nanmean``); z-scores use the WHOLE input
    matrix's per-sample mean/std, so results depend on the full input
    gene set and scale.

    Parameters
    ----------
    eset : str, path-like or pandas.DataFrame
        Expression matrix, genes (index, HGNC symbols) x samples; or a
        path to it (``.tsv``/``.txt`` -> tab, any other extension ->
        comma, exactly like the ORIGINAL).
    output_file : str, optional
        Output CSV/TSV path (``.tsv``/``.txt`` -> tab, else comma, as
        the ORIGINAL). When None, nothing is written.
    verbose : bool, default False
        Reproduce the ORIGINAL's missing-gene warning, ``Results saved
        to:`` line and IOBRpy banner (the tqdm bar and the dead-code
        ``groups``/``w_means`` dicts are dropped, per iobrx convention).
    n_threads : int or None, optional
        Accepted and validated for API uniformity; the workload is
        single-threaded (measured: threading pandas' small per-sample
        ops only adds overhead) and results are thread-count invariant.
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ips has no native kernel (pure-Python bit-exact port):
        ``'rust'`` raises RuntimeError; ``'auto'`` runs the accelerated
        path; ``'python'`` runs the UNTOUCHED upstream ``IPS.main()``
        via an argv bridge (file-path input and ``output_file``
        required; returns None, upstream console protocol included).

    Returns
    -------
    pandas.DataFrame or None
        ``ID, MHC_IPS, EC_IPS, SC_IPS, CP_IPS, AZ_IPS, IPS_IPS`` (the
        ORIGINAL's EC/SC-before-CP column reorder; float columns
        round(6), ``IPS_IPS`` int) — the same frame the ORIGINAL writes.
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "ips has no Rust kernel (pure-Python acceleration: the "
            "original's wall clock is >95% the ~1.4 s iobrpy.main import "
            "floor); use backend='auto' or backend='python'"
        )
    resolve_threads(n_threads)  # validate only; workload is single-threaded
    from iobrx._fast.ips_fast import ips as _ips

    return _ips(
        eset, output_file=output_file, verbose=verbose,
        engine="fast" if backend == "auto" else "original",
    )


# ---------------------------------------------------------------------------
# log2 expression set
# ---------------------------------------------------------------------------
def log2_eset(
    input,
    output,
    verbose: bool = False,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """Apply log2(x+1) to a genes x samples matrix, lazy-import port.

    Bit-exact drop-in for the ``iobrpy.workflow.log2_eset`` CLI stage
    (``python -m iobrpy.main log2_eset``): the output file is
    byte-identical (sha256-equal) on the frozen official gate —
    eset_stad.csv 60,483x10 official testdata -> log2 CSV
    ``edc2664a...``. The three-level delimiter sniffing (``csv.Sniffer``
    over the first 64 KiB -> character-frequency heuristic -> extension
    hint -> tab), ``pd.to_numeric(errors='coerce')`` coercion, the
    ``min < -1`` hard error, ``np.log2(df + 1.0)`` and the
    output-extension separator rule are all verbatim; the only
    optimization is a provable no-op guard (already int/uint/float
    columns skip the per-column ``apply`` — ``pd.to_numeric`` is the
    identity on those dtypes) and the elimination of the ~1.42 s
    ``iobrpy.main`` import floor via lazy imports.

    Parameters
    ----------
    input : str or path-like
        Input matrix (csv/tsv/txt/`;`/`|`, ``.gz`` supported); first
        column = gene IDs (index, name preserved).
    output : str or path-like
        Output path; ``.csv(.gz)`` -> comma, ``.tsv(.gz)`` -> tab,
        otherwise the input delimiter is mirrored (ORIGINAL rule;
        ``.gz`` output is compressed by pandas as upstream — gzip bytes
        embed an mtime, so compare decompressed content across runs).
    verbose : bool, default False
        Reproduce the ORIGINAL's stderr diagnostics (detected/output
        delimiter, coercion & negative-value warnings, ✅ Done line).
    n_threads : int or None, optional
        Accepted and validated for API uniformity; the workload is
        single-threaded (pandas parse/serialize bound).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        log2_eset has no native kernel: ``'rust'`` raises RuntimeError;
        ``'auto'`` runs the accelerated path; ``'python'`` runs the
        UNTOUCHED upstream CLI ``main()`` via an argv bridge (returns
        None; upstream ``sys.exit(1)`` semantics included).

    Returns
    -------
    pandas.DataFrame or None
        The log2(x+1) matrix as written (index = gene IDs).

    Notes
    -----
    The ORIGINAL's ``sys.exit(1)`` failure paths (unreadable/empty
    input, values < -1, write failure) raise ``ValueError`` carrying the
    ORIGINAL message text instead (repo convention, cf. ``iobrx.nmf`` —
    a SystemExit from a library call would kill interactive sessions).
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "log2_eset has no Rust kernel (pure-Python acceleration: "
            "pandas parse/serialize bound, and the parity contract "
            "requires the pandas to_csv writer); use backend='auto' or "
            "backend='python'"
        )
    resolve_threads(n_threads)  # validate only; workload is single-threaded
    from iobrx._fast.log2_eset_fast import log2_eset as _log2_eset

    return _log2_eset(
        input, output, verbose=verbose,
        engine="fast" if backend == "auto" else "original",
    )


# ---------------------------------------------------------------------------
# prepare salmon TPM matrix
# ---------------------------------------------------------------------------
def prepare_salmon(
    input,
    output,
    return_feature: str = "symbol",
    remove_version: bool = False,
    verbose: bool = False,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """Salmon TPM matrix -> deduplicated symbol/ENSG/ENST matrix, lazy-import port.

    Bit-exact drop-in for ``iobrpy.workflow.prepare_salmon.
    prepare_salmon_tpm`` (CLI ``python -m iobrpy.main prepare_salmon``):
    the output CSV is byte-identical (sha256-equal) on the frozen
    official gate — fixture8_salmon_tpm.tsv.gz 60,000 transcripts x 8
    samples -> 30,000 x 9 ``d5546a04...``. The GENCODE 8-field pipe-ID
    annotation is parsed IN PLACE from the ``Name`` column (the ORIGINAL
    reads no external annotation file), ``remove_duplicate_genes``
    keeps the per-column ``groupby.mean()`` loop verbatim (Kahan-
    compensated group means; rows come out in symbol-alphabetical
    groupby order), and the output is ALWAYS a comma-separated CSV
    regardless of the output extension (upstream quirk, preserved).
    The acceleration is eliminating the ~1.5 s ``iobrpy.main`` import
    floor (≈86% of the original's 1.742 s median) via lazy pandas
    import; tqdm bars are dropped and the ``>>>``/``Done!``/banner
    protocol reproduces under ``verbose=True``.

    BUG-COMPATIBLE (upstream defects preserved): the WHOLE flow sits in
    a try/except — any failure (missing ``Name`` column, invalid
    ``return_feature``, ``Name`` with <8 pipe fields, ...) only prints a
    red ``Error occurred: ...`` line, does NOT raise, leaves the output
    file unwritten and returns None (the upstream CLI still exits rc=0).

    Parameters
    ----------
    input : str or path-like
        merge_salmon TPM matrix (TSV or TSV.GZ) with the GENCODE-style
        pipe-ID ``Name`` column.
    output : str or path-like
        Output path — always written as comma-separated CSV with
        ``index=False`` (upstream quirk).
    return_feature : {'symbol', 'ENST', 'ENSG'}, default 'symbol'
        Pipe-field annotation to keep as ``Name`` (matched
        case-insensitively, as the upstream FUNCTION does; the CLI's
        argparse choices are case-sensitive — the function-level
        laxness is preserved).
    remove_version : bool, default False
        Strip ``.N`` version suffixes (``str.split('.').str[0]``).
    verbose : bool, default False
        Reproduce the ORIGINAL's ``>>>`` stdout protocol, green
        ``Done!`` and IOBRpy banner.
    n_threads : int or None, optional
        Accepted and validated for API uniformity; the workload is
        single-threaded (upstream has no concurrency).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        prepare_salmon has no native kernel: ``'rust'`` raises
        RuntimeError; ``'auto'`` runs the accelerated path;
        ``'python'`` calls the UNTOUCHED upstream ``prepare_salmon_tpm``
        (returns None).

    Returns
    -------
    pandas.DataFrame or None
        The deduplicated matrix as written (``Name`` + sample columns,
        alphabetical rows); None when the flow failed (error swallowed,
        bug-compatible).
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "prepare_salmon has no Rust kernel (pure-Python acceleration: "
            "string-parse + pandas-serialization bound); use "
            "backend='auto' or backend='python'"
        )
    resolve_threads(n_threads)  # validate only; workload is single-threaded
    from iobrx._fast.prepare_salmon_fast import prepare_salmon as _prepare_salmon

    return _prepare_salmon(
        input, output, return_feature=return_feature,
        remove_version=remove_version, verbose=verbose,
        engine="fast" if backend == "auto" else "original",
    )


# ---------------------------------------------------------------------------
# mouse -> human symbol conversion
# ---------------------------------------------------------------------------
def mouse2human(
    input,
    output=None,
    is_matrix: bool = False,
    column_of_symbol=None,
    sep=None,
    out_sep=None,
    verbose: bool = False,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """Convert mouse gene symbols to human symbols, lazy-import port.

    Bit-exact drop-in for the ``iobrpy.workflow.mouse2human_eset`` CLI
    stage: the output file is byte-identical (sha256-equal) on the
    frozen official gate — 2,000 mouse symbols x 6 samples matrix-mode
    fixture -> human_matrix.csv ``174aec77...``. Mapping runs through
    the packaged ``mus_human.pkl`` (20,342 rows, lazily memoized) with
    the ORIGINAL ``anno_eset(method='mean')`` semantics — repo-verified
    bit-exact ``iobrx._fast.anno_eset_fast`` core: unmapped mouse
    symbols silently dropped, one->many expansion, many->one keeps the
    highest-row-mean occurrence (selection, NOT averaging; with any
    duplicate symbol the frame is globally sorted by that score
    DESCENDING), all-zero / all-NA / first-column-NA rows filtered.
    Table mode (``is_matrix=False``) deduplicates first with the
    ORIGINAL ``_remove_duplicate_genes`` statements; output goes through
    the ORIGINAL manual writer (blank top-left header cell, NaN ->
    empty, ``str()`` full-precision values, ``.gz`` auto-compression,
    extension-inferred separators). The acceleration is eliminating the
    ~1.4 s ``iobrpy.main`` import floor plus lazy resource loading; the
    always-on upstream tqdm save bar is dropped and the info lines /
    banner reproduce under ``verbose=True``.

    Parameters
    ----------
    input : str, path-like or pandas.DataFrame
        Matrix mode: genes (mouse symbols, index) x samples. Table mode:
        a table containing the symbol column named by
        ``column_of_symbol``. Paths use the ORIGINAL separator rule
        (``.tsv/.tab/.txt(+.gz)`` -> tab, else comma; ``sep`` overrides).
    output : str, path-like or None
        Output path (``.gz`` auto-compresses — gzip bytes embed an
        mtime, so compare decompressed content across runs; separator
        by extension or ``out_sep``). When None, nothing is written.
    is_matrix : bool, default False
        Treat ``input`` as a symbol-indexed matrix (upstream
        ``--is_matrix``).
    column_of_symbol : str, optional
        Symbol column for table mode; required when ``is_matrix=False``
        (ValueError with the ORIGINAL message otherwise).
    sep, out_sep : str, optional
        Input/output separator overrides (upstream ``--sep``/``--out_sep``).
    verbose : bool, default False
        Reproduce the ORIGINAL's info lines (input shape/mode, mapping
        shape, ``anno_eset`` match-rate statistics, ``[iobrpy] Converted
        matrix saved to:`` and the IOBRpy banner).
    n_threads : int or None, optional
        Accepted and validated for API uniformity; the workload is
        single-threaded.
    backend : {'auto', 'rust', 'python'}, default 'auto'
        mouse2human has no native kernel: ``'rust'`` raises RuntimeError;
        ``'auto'`` runs the accelerated path; ``'python'`` runs the
        UNTOUCHED upstream CLI ``main()`` via an argv bridge (file-path
        ``input`` AND ``output`` required; returns None).

    Returns
    -------
    pandas.DataFrame or None
        Human-symbol-indexed matrix (index name ``symbol``) — the same
        frame the manual writer serializes.
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "mouse2human has no Rust kernel (pure-Python acceleration: "
            "mapping + manual-writer bound, and the parity contract "
            "requires the upstream str()-per-value writer); use "
            "backend='auto' or backend='python'"
        )
    resolve_threads(n_threads)  # validate only; workload is single-threaded
    from iobrx._fast.mouse2human_fast import mouse2human as _mouse2human

    return _mouse2human(
        input, output, is_matrix=is_matrix, column_of_symbol=column_of_symbol,
        sep=sep, out_sep=out_sep, verbose=verbose,
        engine="fast" if backend == "auto" else "original",
    )

# ---------------------------------------------------------------------------
# merge STAR gene counts
# ---------------------------------------------------------------------------
def merge_star_count(
    path,
    project,
    num_processes: int | None = None,
    n_threads: int | None = None,
    backend: str = "auto",
    verbose: bool = True,
):
    """Merge STAR ``*_ReadsPerGene.out.tab`` files into one count matrix, accelerated.

    Bit-exact drop-in for ``iobrpy.workflow.merge_star_count`` (CLI
    ``python -m iobrpy.main merge_star_count``): discovers
    ``*_ReadsPerGene.out.tab`` in *path* (single directory level,
    ``os.listdir``, non-recursive), reads column 0 (gene ID) and column 1
    (unstranded count; sample name = file name minus the suffix) and
    writes ``<project>.STAR.count.tsv.gz`` (gzip level-5 TSV, first column
    ``ID``) back into *path*. Decompressed contents are token-for-token
    identical to the original on the frozen 8 x 60,004-row gate (480,256
    cells bit-identical after column + row-label alignment, max abs diff
    0.0); output columns are deterministically sorted by sample name and
    the union row order is anchored to the sorted-first sample — the
    ORIGINAL's ``os.listdir`` + ``as_completed`` orders make BOTH its
    column order and its stat-row anchor nondeterministic run-to-run.

    BUG-COMPAT: STAR's four leading global-stat rows (bare integers:
    N_unmapped / N_multimapping / N_noFeature / N_ambiguous) are NOT
    skipped, exactly as upstream — they pollute the matrix as up to
    ``4 x n_samples`` NaN-padded rows ((60032, 8) on the 8-sample gate:
    60,000 gene rows + 32 stat rows, each stat row non-null only in its
    owning sample).

    Parameters
    ----------
    path : str
        Directory containing the STAR count files (non-recursive); also
        the output destination (upstream forces output dir == input dir).
    project : str
        Output filename prefix.
    num_processes : int, optional
        Loader threads for the pandas fallback (the upstream CLI has no
        such flag; it uses Python's ThreadPoolExecutor default). ``None``
        → ``n_threads`` → ``min(32, (os.cpu_count() or 1) + 4)``; capped
        at 32. The fast path spawns no threads (its C-level byte surgery
        is GIL-bound, same finding as merge_salmon).
    n_threads : int, optional
        iobrx-wide alias for ``num_processes``; ``num_processes`` wins
        when both are given.
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``auto`` runs the accelerated pure-Python implementation, which
        falls back to the original pandas statements for the whole run
        whenever its equivalence guards trip (bug-compatible errors and
        int64/float64 dtype quirks included). ``python`` runs the
        untouched upstream CLI ``main()`` via argv rewrite. ``rust``
        fails clearly — this stage has no native kernel (I/O +
        pandas-serialization bound, and the parity contract requires the
        pandas ``to_csv`` writer).
    verbose : bool, default True
        Reproduce the upstream console protocol ("Saving merged
        matrix...", head preview, row/column counts, "Saved to:", IOBRpy
        banner).

    Returns
    -------
    pandas.DataFrame or None
        The matrix as written: union of per-file ID labels (index name
        ``ID``; gene rows plus the unpurged stat rows) x samples (sorted
        columns); per-column dtype int64 (column covers the whole union)
        or float64 (NaN-padded), exactly as pandas concat produces.
        ``None`` when no STAR file was found (upstream prints a notice
        and writes nothing) or when ``backend='python'`` (upstream
        ``main()`` returns nothing).
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "merge_star_count has no Rust kernel (pure-Python "
            "acceleration: I/O + pandas-serialization bound); use "
            "backend='auto' or backend='python'"
        )
    workers = num_processes if num_processes is not None else n_threads
    if backend == "python":
        from iobrx._fast.merge_star_count_fast import merge_star_count_original
        return merge_star_count_original(path, project)
    from iobrx._fast.merge_star_count_fast import merge_star_count as _merge_star_count
    return _merge_star_count(path, project, workers, verbose=verbose)

# ---------------------------------------------------------------------------
# FASTQ QC (fastp + MultiQC)
# ---------------------------------------------------------------------------
def fastq_qc(
    path1_fastq,
    path2_fastp,
    num_threads: int | None = None,
    suffix1: str = "_1.fastq.gz",
    batch_size: int = 1,
    se: bool = False,
    length_required: int = 50,
    fastp_bin: str | None = None,
    multiqc_bin: str | None = None,
    n_threads: int | None = None,
    backend: str = "auto",
    verbose: bool = True,
):
    """FASTQ quality control with fastp + MultiQC (external-tool stage).

    Drop-in for ``iobrpy.workflow.fastq_qc`` (CLI ``python -m iobrpy.main
    fastq_qc``): discovers ``*<suffix1>`` under *path1_fastq*, runs fastp per
    sample through ``multiprocessing.Pool(batch_size)`` + ``imap_unordered``
    (the upstream scheduling primitives, verbatim), writes cleaned FASTQs +
    ``<sample>_fastp.html/.json`` + ``<sample>.task.complete`` resume markers
    into *path2_fastp*, then aggregates the fastp JSONs into
    ``<path2_fastp>/multiqc_report/multiqc_fastp_report.html``. The fastp and
    MultiQC command lines are token-for-token the upstream ones, so with the
    same binaries the cleaned FASTQ / JSON products are BYTE-identical to the
    original's (verified on the frozen 2 x 5 GB official-format gate: all 4
    cleaned fastq.gz, both fastp.json and both .task.complete sha256-equal;
    the human-facing HTMLs differ only in timestamps and fastp's own per-run
    7th-digit duplication-rate jitter, which ORIGINAL-vs-ORIGINAL reruns
    reproduce identically).

    Parameters
    ----------
    path1_fastq : str
        Directory of raw FASTQs (upstream ``--path1_fastq``).
    path2_fastp : str
        Output directory for cleaned FASTQs / reports / markers (created if
        missing; upstream ``--path2_fastp``).
    num_threads : int, optional
        Threads per fastp process (upstream ``--num_threads``, CLI default 8).
    suffix1 : str, default '_1.fastq.gz'
        R1 suffix; R2 inferred by the upstream ``suffix1.replace("1", "2")``.
    batch_size : int, default 1
        Pool size = concurrent samples (upstream CLI ``--batch_size`` default;
        the upstream PYTHON function default is 5 — this wrapper mirrors the
        CLI, as the whole port does).
    se : bool, default False
        Single-end mode (upstream ``--se``).
    length_required : int, default 50
        BUG-COMPATIBLE dead parameter: threaded through and never used,
        exactly as upstream (the fastp command disables length filtering).
    fastp_bin, multiqc_bin : str, optional
        Binary overrides; ``None`` = bare ``"fastp"`` / ``"multiqc"``
        resolved through the child-process PATH (the upstream resolution).
    n_threads : int, optional
        iobrx-wide alias for ``num_threads``; ``num_threads`` wins when both
        are given, else ``n_threads``, else the upstream CLI default 8.
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``auto`` runs the ported orchestration; ``python`` runs the untouched
        upstream CLI ``main()`` via argv rewrite (returns None, like
        upstream); ``rust`` fails clearly — this stage has no native kernel
        (the fastp/multiqc binaries do the work).
    verbose : bool, default True
        Reproduce the upstream console protocol ([Start]/[Done]/[Skip],
        output list, MultiQC notices, IOBRpy banner); per-sample fastp error
        messages print regardless.

    Returns
    -------
    dict or None
        ``{"results": [<per-sample dict: sample/status/outputs>, ...],
        "outputs": [<sorted unique cleaned-FASTQ paths>], "multiqc_report":
        <path or None>}`` (upstream returns None). ``None`` when
        ``backend='python'``.
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "fastq_qc has no Rust kernel (external-tool orchestration); "
            "use backend='auto' or backend='python'"
        )
    nt = num_threads if num_threads is not None else (
        n_threads if n_threads is not None else 8)
    if backend == "python":
        from iobrx._fast.fastq_qc_fast import fastq_qc_original
        return fastq_qc_original(path1_fastq, path2_fastp, nt, suffix1,
                                 batch_size, se, length_required)
    from iobrx._fast.fastq_qc_fast import fastq_qc as _fastq_qc
    return _fastq_qc(path1_fastq, path2_fastp, nt, suffix1, batch_size, se,
                     length_required, fastp_bin=fastp_bin,
                     multiqc_bin=multiqc_bin, verbose=verbose)


# ---------------------------------------------------------------------------
# batch salmon quant
# ---------------------------------------------------------------------------
def batch_salmon(
    index,
    path_fq,
    path_out,
    suffix1: str = "_1.fastq.gz",
    batch_size: int = 1,
    num_threads: int | None = None,
    gtf: str | None = None,
    salmon_bin: str | None = None,
    n_threads: int | None = None,
    backend: str = "auto",
    verbose: bool = True,
):
    """Batch-run Salmon quantification over paired-end FASTQs.

    Drop-in for ``iobrpy.workflow.batch_salmon`` (CLI ``python -m iobrpy.main
    batch_salmon``): R2 suffix inference, ``sorted(glob)`` R1 discovery,
    missing-R2 warn-and-skip, ``random.shuffle`` dispatch through
    ``multiprocessing.Pool(max(1, batch_size))`` + ``imap_unordered``, the
    upstream ``salmon quant -i <index> -l ISF --gcBias -1/-2 -p <threads>
    -o <out> --validateMappings [-g <gtf>]`` command line token-for-token,
    ``quant.sf`` + ``task.complete`` resume logic, and the friendly
    index-version-mismatch guidance on failure — all verbatim. With the same
    salmon binary, index and ``-p`` thread count, ``quant.sf`` /
    ``cmd_info.json`` / ``ambig_info.tsv`` / ``flenDist.txt`` are
    BYTE-identical to the original's (verified on the frozen official-format
    sample: sha256-equal; ``meta_info.json`` and ``salmon_quant.log`` carry
    run timestamps and match content-wise).

    Parameters
    ----------
    index : str
        Salmon index directory (upstream ``--index``).
    path_fq : str
        FASTQ directory; R1 = ``*<suffix1>`` (upstream ``--path_fq``).
    path_out : str
        Output root; one ``<sample_id>/`` per sample (``--path_out``).
    suffix1 : str, default '_1.fastq.gz'
        R1 suffix (``--suffix1``); an unterminable R2 suffix reproduces the
        upstream ValueError + CLI exit code 2.
    batch_size : int, default 1
        Pool size = concurrent samples (``--batch_size``; ``max(1, int(...))``
        as upstream).
    num_threads : int, optional
        Threads per salmon process (``--num_threads`` -> ``-p``, CLI default
        8). NOTE: salmon's online phase is thread-order sensitive — use the
        SAME value as any run you compare bytes against.
    gtf : str, optional
        Optional GTF for gene-level quant (``--gtf`` -> ``-g``).
    salmon_bin : str, optional
        Binary override; ``None`` = bare ``"salmon"`` through PATH (upstream).
    n_threads : int, optional
        iobrx-wide alias for ``num_threads`` (``num_threads`` wins, else
        ``n_threads``, else 8).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``python`` runs the untouched upstream CLI ``main()`` (argv rewrite);
        ``rust`` fails clearly (external-tool stage, no native kernel).
    verbose : bool, default True
        Upstream console protocol ([Plan]/[Start]/[Done]/[Skip], success
        summary, IOBRpy banner on success); stderr diagnostics ([warn]
        missing R2, [error], failure summary) print regardless, as upstream.

    Returns
    -------
    dict
        ``{"rc": 0|1|2, "failures": [(sample_id, message), ...], "n_pairs":
        int, "results": [(sample_id, ok, err), ...]}`` where ``rc`` is the
        exit code the upstream CLI produces (0 success, 1 sample failures,
        2 usage/discovery error) — RETURNED instead of ``sys.exit`` (the
        documented API deviation; ``backend='python'`` translates the
        upstream ``SystemExit`` into the same ``rc`` field).
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "batch_salmon has no Rust kernel (external-tool orchestration); "
            "use backend='auto' or backend='python'"
        )
    nt = num_threads if num_threads is not None else (
        n_threads if n_threads is not None else 8)
    if backend == "python":
        from iobrx._fast.batch_salmon_fast import (
            batch_salmon_original, _print_iobrpy_banner)
        res = batch_salmon_original(index, path_fq, path_out, suffix1,
                                    batch_size, nt, gtf)
        if res["rc"] == 0 and verbose:
            _print_iobrpy_banner()
        return res
    from iobrx._fast.batch_salmon_fast import batch_salmon as _batch_salmon
    return _batch_salmon(index, path_fq, path_out, suffix1, batch_size, nt,
                         gtf, salmon_bin=salmon_bin, verbose=verbose)


# ---------------------------------------------------------------------------
# batch STAR two-pass counting
# ---------------------------------------------------------------------------
def batch_star_count(
    index,
    path_fq,
    path_out,
    suffix1: str = "_1.fastq.gz",
    batch_size: int = 1,
    num_threads: int | None = None,
    star_bin: str | None = None,
    n_threads: int | None = None,
    backend: str = "auto",
    verbose: bool = True,
):
    """Batch-run STAR two-pass alignment + GeneCounts over FASTQ pairs.

    Drop-in for ``iobrpy.workflow.batch_star_count`` (CLI ``python -m iobrpy.main
    batch_star_count``): ``os.listdir`` + ``sorted`` + ``random.shuffle``
    discovery, ``RLIMIT_NOFILE`` raise per sample, the upstream STAR command
    line token-for-token (``--twopassMode Basic --runThreadN N
    --outBAMsortingThreadN N --outTmpDir <sample>_STARtmp --readFilesCommand
    zcat --outSAMtype BAM SortedByCoordinate --quantMode GeneCounts
    --limitBAMsortRAM 137438953472``), ``<sample>.task.complete`` + non-empty
    sorted-BAM resume logic, ``RuntimeError`` propagation on STAR failure and
    the final BAM/GeneCounts location summary — all verbatim. BUG-COMPATIBLE:
    despite the upstream docstrings, the batch loop is STRICTLY SEQUENTIAL
    (``batch_size`` only slices the shuffled list); the port keeps it that
    way. With the same STAR binary and ``--runThreadN``, ``ReadsPerGene.out.tab``
    / ``SJ.out.tab`` / junction lists are BYTE-identical and the BAM record
    stream is byte-identical (verified on the frozen official-format sample;
    the BAM header's @PG CL line embeds the thread count and paths, so the
    BAM sha matches only for identical ``num_threads`` and ``path_out``).

    Parameters
    ----------
    index : str
        STAR genome index directory (upstream ``--index`` -> ``--genomeDir``).
    path_fq : str
        FASTQ directory; R1 = ``*<suffix1>`` (upstream ``--path_fq``).
    path_out : str
        STAR output folder (upstream ``--path_out``; created if missing).
    suffix1 : str, default '_1.fastq.gz'
        R1 suffix; R2 via the upstream blunt ``suffix1.replace("1", "2")``.
    batch_size : int, default 1
        Upstream ``--batch_size``: chunk size of the SEQUENTIAL loop
        (bug-compatible — never actually concurrent).
    num_threads : int, optional
        ``--runThreadN`` / ``--outBAMsortingThreadN`` (upstream
        ``--num_threads``, CLI default 8).
    star_bin : str, optional
        Binary override; ``None`` = bare ``"STAR"`` through PATH (upstream;
        in the official toolchain that is the conda bash wrapper dispatching
        to the best SIMD build, e.g. STAR-avx2).
    n_threads : int, optional
        iobrx-wide alias for ``num_threads`` (``num_threads`` wins, else
        ``n_threads``, else 8).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``python`` runs the untouched upstream CLI ``main()`` (argv rewrite);
        ``rust`` fails clearly (external-tool stage, no native kernel).
    verbose : bool, default True
        Upstream console protocol ([Plan]/[Start]/[Done]/[Skip], output
        summary, IOBRpy banner); STAR's own output always streams to stderr
        (upstream ``stdout=sys.stderr``) and ``RuntimeError`` propagates.

    Returns
    -------
    dict
        ``{"rc": 0, "samples": [<sample_id>, ... in dispatch order]}``
        (upstream returns None). A failing STAR run raises ``RuntimeError``
        with the upstream message, aborting the stage.
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "batch_star_count has no Rust kernel (external-tool "
            "orchestration); use backend='auto' or backend='python'"
        )
    nt = num_threads if num_threads is not None else (
        n_threads if n_threads is not None else 8)
    if backend == "python":
        from iobrx._fast.batch_star_count_fast import (
            batch_star_count_original, _print_iobrpy_banner)
        res = batch_star_count_original(index, path_fq, path_out, suffix1,
                                        batch_size, nt)
        if verbose:
            _print_iobrpy_banner()
        return res
    from iobrx._fast.batch_star_count_fast import (
        batch_star_count as _batch_star_count)
    return _batch_star_count(index, path_fq, path_out, suffix1, batch_size,
                             nt, star_bin=star_bin, verbose=verbose)


# ---------------------------------------------------------------------------
# TRUST4 (TCR/BCR reconstruction)
# ---------------------------------------------------------------------------
def trust4(
    bam=None,
    r1=None,
    r2=None,
    ru=None,
    fqdir=None,
    f=None,
    ref=None,
    o=None,
    od=None,
    t: int | None = None,
    k: int | None = None,
    barcode=None,
    barcodeLevel=None,
    barcodeWhitelist=None,
    barcodeTranslate=None,
    UMI=None,
    readFormat=None,
    repseq: bool = False,
    contigMinCov: int | None = None,
    minHitLen: int | None = None,
    mateIdSuffixLen: int | None = None,
    skipMateExtension: bool = False,
    abnormalUnmapFlag: bool = False,
    assembleWithRef: bool = False,
    noExtraction: bool = False,
    outputReadAssignment: bool = False,
    stage: int | None = None,
    clean: int | None = None,
    extra=(),
    trust4_bin: str | None = None,
    n_threads: int | None = None,
    backend: str = "auto",
    verbose: bool = True,
):
    """TRUST4 TCR/BCR reconstruction (external-tool stage) + accelerated
    immune post-processing.

    Drop-in for ``iobrpy.workflow.trust4`` (CLI ``python -m iobrpy.main
    trust4 ...``). The FULL upstream CLI surface is mirrored as named
    arguments (``-b`` file/dir batch, ``-1/-2``, ``-u``, ``--fqdir``, ``-f``
    / ``--ref`` with the ``iobrpy.resources`` defaults ``hg38_bcrtcr.fa`` /
    ``human_IMGT+C.fa``, ``-o``/``--od``, ``-t``/``-k``, barcode/UMI
    options, mode flags, ``--stage``/``--clean``); anything else goes through
    ``extra`` (the upstream ``parse_known_args`` pass-through into every
    run-trust4 call). Runner resolution is verbatim: ``trust4_bin`` >
    ``$TRUST4_BIN`` > ``"run-trust4"`` on PATH (missing -> the upstream
    message + code 127). Batch modes keep the upstream SEQUENTIAL per-sample
    loop with ``<folder>.TRUST4.done`` resume flags and ``TRUST_<prefix>``
    output naming; after every mode the upstream staging pass collects all
    ``*_report.tsv`` under the output root and writes
    ``trust4_immdata.csv`` + ``trust4_immune_indices.csv``. The ONLY
    accelerated piece is that post-processing
    (``process_immune_data_batch_fast``: threaded per-file reads stored at
    their input index, whole-column hoist of the per-group string/float
    conversions, original numpy statement sequence per sample) — its two
    CSVs are BYTE-identical to the original's, including the degenerate
    frozen real-data gate (header-only report -> header-only immdata.csv and
    the 1-byte ``'\\n'`` indices file). TRUST4 data outputs
    (report.tsv/cdr3.out/airr/fa/fq) are the binary's own and byte-stable
    for fixed (binary, references, ``-t``).

    Parameters
    ----------
    bam : str, optional
        ``-b``: a BAM file (single run) or a directory of ``*.bam`` (batch).
    r1, r2 : str, optional
        ``-1``/``-2`` paired FASTQs (single run).
    ru : str, optional
        ``-u`` single-end FASTQ (single run).
    fqdir : str, optional
        ``--fqdir``: directory of ``*_1/_2.fastq(.gz)`` pairs (batch).
    f, ref : str, optional
        Reference FASTAs; ``None`` extracts the bundled IOBRpy defaults to
        atexit-cleaned temp dirs (upstream behaviour).
    o : str, optional
        Single-run: output PREFIX; batch modes: output ROOT directory.
    od : str, optional
        TRUST4-native output directory (single-run only, as upstream).
    t : int, optional
        ``-t`` threads for run-trust4 (no default — omitted when None, as
        upstream; the tool then uses its own default).
    k : int, optional
        ``-k`` starting k-mer size.
    barcode, barcodeLevel, barcodeWhitelist, barcodeTranslate, UMI,
    readFormat : str, optional
        Single-cell / barcode options (verbatim flags; ``barcodeLevel``
        choices ``cell``/``molecule`` are argparse-validated as upstream).
    repseq, skipMateExtension, abnormalUnmapFlag, assembleWithRef,
    noExtraction, outputReadAssignment : bool, default False
        Store-true mode flags (verbatim).
    contigMinCov, minHitLen, mateIdSuffixLen : int, optional
        Fine-control integers (verbatim).
    stage : {0,1,2,3}, optional
        Pipeline stage (argparse-validated, as upstream).
    clean : {0,1,2}, optional
        Intermediate cleanup level (argparse-validated, as upstream).
    extra : sequence of str, default ()
        Extra tokens appended to every run-trust4 command (the upstream
        unknown-args pass-through).
    trust4_bin : str, optional
        Runner override; ``None`` = ``$TRUST4_BIN`` or ``"run-trust4"``
        resolved with ``shutil.which`` (upstream resolution).
    n_threads : int, optional
        iobrx-wide alias for ``t`` (``t`` wins when both are given).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``auto``: ported orchestration + accelerated post-processing;
        ``python``: the untouched upstream ``main(argv)`` (original
        post-processing); ``rust`` fails clearly (no native kernel).
    verbose : bool, default True
        Gate the dispatch-level IOBRpy banner (upstream ``iobrpy.main``
        prints it after the trust4 stage exits, for ANY exit code). The
        stage's own protocol lines ([IOBRpy|trust4] ..., tqdm bars) are
        verbatim and always print.

    Returns
    -------
    dict
        ``{"rc": int}`` — the process exit code the upstream CLI produces
        (0 success; last failing sample's code in batch modes; 2 for
        argparse/usage errors; 127 runner-not-found; 130 on
        KeyboardInterrupt). The upstream ``sys.exit`` is caught and
        translated; nothing else differs.
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "trust4 has no Rust kernel (external-tool orchestration; the "
            "immune post-processing is accelerated pure Python); use "
            "backend='auto' or backend='python'"
        )
    argv: list[str] = []
    if bam is not None:
        argv += ["-b", str(bam)]
    if r1 is not None:
        argv += ["-1", str(r1)]
    if r2 is not None:
        argv += ["-2", str(r2)]
    if ru is not None:
        argv += ["-u", str(ru)]
    if fqdir is not None:
        argv += ["--fqdir", str(fqdir)]
    if f is not None:
        argv += ["-f", str(f)]
    if ref is not None:
        argv += ["--ref", str(ref)]
    if o is not None:
        argv += ["-o", str(o)]
    if od is not None:
        argv += ["--od", str(od)]
    tt = t if t is not None else n_threads
    if tt is not None:
        argv += ["-t", str(int(tt))]
    if k is not None:
        argv += ["-k", str(int(k))]
    if barcode is not None:
        argv += ["--barcode", str(barcode)]
    if barcodeLevel is not None:
        argv += ["--barcodeLevel", str(barcodeLevel)]
    if barcodeWhitelist is not None:
        argv += ["--barcodeWhitelist", str(barcodeWhitelist)]
    if barcodeTranslate is not None:
        argv += ["--barcodeTranslate", str(barcodeTranslate)]
    if UMI is not None:
        argv += ["--UMI", str(UMI)]
    if readFormat is not None:
        argv += ["--readFormat", str(readFormat)]
    if repseq:
        argv += ["--repseq"]
    if contigMinCov is not None:
        argv += ["--contigMinCov", str(int(contigMinCov))]
    if minHitLen is not None:
        argv += ["--minHitLen", str(int(minHitLen))]
    if mateIdSuffixLen is not None:
        argv += ["--mateIdSuffixLen", str(int(mateIdSuffixLen))]
    if skipMateExtension:
        argv += ["--skipMateExtension"]
    if abnormalUnmapFlag:
        argv += ["--abnormalUnmapFlag"]
    if assembleWithRef:
        argv += ["--assembleWithRef"]
    if noExtraction:
        argv += ["--noExtraction"]
    if outputReadAssignment:
        argv += ["--outputReadAssignment"]
    if stage is not None:
        argv += ["--stage", str(int(stage))]
    if clean is not None:
        argv += ["--clean", str(int(clean))]
    argv += [str(x) for x in extra]

    from iobrx._fast.trust4_fast import print_iobrpy_banner
    if backend == "python":
        from iobrx._fast.trust4_fast import trust4_original as _run
        engine_ok = False
    else:
        from iobrx._fast.trust4_fast import main as _t4_main
        _run = _t4_main
        engine_ok = True

    try:
        if engine_ok:
            _run(argv, trust4_bin=trust4_bin, engine="fast")
        else:
            _run(argv)
        rc = 0  # unreachable in practice: every upstream path sys.exit()s
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    if verbose:
        print_iobrpy_banner()
    return {"rc": rc}


# ---------------------------------------------------------------------------
# runall end-to-end orchestrator
# ---------------------------------------------------------------------------
def runall(
    mode,
    outdir,
    fastq,
    threads: int | None = None,
    batch_size: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    unknown=None,
    n_threads: int | None = None,
    backend: str = "auto",
    verbose: bool = True,
):
    """End-to-end FASTQ -> TME orchestrator (salmon/star), in-process port.

    Drop-in for ``iobrpy.workflow.runall`` (CLI ``python -m iobrpy.main
    runall``). The whole orchestration layer is VERBATIM upstream code: the
    ``salmon``/``star`` branching, the step order (fastq_qc ->
    batch_salmon/batch_star_count -> merge_salmon/merge_star_count ->
    prepare_salmon/count2tpm -> log2_eset -> calculate_sig_score ->
    cibersort/IPS/estimate/mcpcounter/quantiseq/epic -> deconvo merge ->
    LR_cal -> trust4), the numbered output layout (``01-qc`` ...
    ``07-TCRBCR``), the legacy sectioned passthrough parser and the
    long-flag auto-router (mode-aware ``--index``/``--project``/
    ``--remove_version``/``--suffix1`` routing, ``--method`` value
    disambiguation, legacy ``--num_threads``/``--parallel_size``/
    ``--num_processes`` absorption), every per-step default injection
    (``--project runall``, ``--return_feature symbol``,
    ``--remove_version``, ``--signature all --method integration
    --mini_gene_count 2 --adjust_eset``, ``--platform affymetrix``,
    ``--features HUGO_symbols``, ``--arrays --tumor --scale_mrna``,
    ``--reference TRef``, ``--data_type tpm --id_type symbol
    --cancer_type pancan --verbose``), the ``--resume`` done-flag checks,
    the ``--dry_run`` protocol, the inline pandas deconvolution merge and
    all ``[run]/[ok]/[ERROR]/[resume]/[done]`` console lines. Console
    parity is byte-identical to the original under ``--dry_run`` on both
    modes across default/rich-flag/legacy/sectioned/resume/error argv
    (13/13 gates, tests/test_parity_runall.py).

    THE ACCELERATION: upstream executes each step as ``subprocess.run(
    ["iobrpy", <step>, ...])`` — 9-10 child processes, each paying the
    ~1.4-2.2 s ``iobrpy.main`` full-import cold-start floor. This port
    keeps the constructed command lists byte-for-byte (so ``dry_run``
    output and the ``[run]`` headers are identical), parses each with a
    MIRROR of the exact ``iobrpy.main`` subparser for that step (pinning
    the CLI-level defaults — e.g. LR_cal ``data_type='tpm'`` where the
    plain function default is ``'count'``, epic ``solver='trust-constr'``,
    cibersort ``QN=True``), and dispatches to the already-ported ``iobrx``
    substep functions plus the ``iobrpy.main`` dispatch-layer I/O
    transformations (input-parse rules, ``_CIBERSORT``/``_estimate``/
    ``_MCPcounter``/``_quantiseq``/``_EPIC`` suffixes, transposes,
    ``index_label='ID'``, ``float_format='%.7f'``, separator-by-extension
    writes, banners) reproduced verbatim. External tools (fastp/multiqc/
    salmon/STAR/run-trust4) are still scheduled by the ported substeps
    exactly as upstream, so every substep's own parity contract carries
    over to the whole chain; what disappears is only the per-step Python
    cold start (speedup ceiling ~1 on tool-dominated runs — the value is
    the identical API plus in-process composition).

    Parameters
    ----------
    mode : {'salmon', 'star'}
        Quantification chain (upstream ``--mode``; other values reproduce
        the upstream argparse usage error as ``{"rc": 2}``).
    outdir : str or path-like
        Output root (upstream ``--outdir``); the numbered directories are
        created under it.
    fastq : str or path-like
        Raw FASTQ directory (upstream ``--fastq`` -> ``fastq_qc
        --path1_fastq``).
    threads : int, optional
        Unified concurrency (upstream ``--threads``): fastq_qc/
        batch_salmon/batch_star_count ``--num_threads``, merge_salmon
        ``--num_processes``, cibersort ``--threads``, calculate_sig_score
        ``--parallel_size``, trust4 ``-t``. ``None`` -> legacy flags in
        *unknown* -> 8 (upstream resolution).
    batch_size : int, optional
        Unified batch size (upstream ``--batch_size``): ``None`` -> legacy
        flag in *unknown* -> 1.
    resume : bool, default False
        Skip steps whose done-flags/outputs already exist (upstream
        ``--resume``).
    dry_run : bool, default False
        Print the step commands without executing them (upstream
        ``--dry_run``; BUG-COMPATIBLE: done-flag files and the
        deconvolution merge table are still written, as upstream).
    unknown : sequence of str, optional
        Extra CLI tokens after the mirrored flags — the upstream
        ``parse_known_args`` remainder: either the legacy "sectioned"
        style (``["fastq_qc", "--se", "cibersort", "--perm", "25"]``) or
        auto-routed long flags (``["--index", "/ref/salmon_idx",
        "--project", "prj"]``), including the legacy concurrency flags
        (``--num_threads``/``--parallel_size``/``--num_processes``/
        ``--batch_size``).
    n_threads : int, optional
        iobrx-wide alias for *threads* (``threads`` wins when both given).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``auto`` runs the in-process ported orchestration; ``python`` runs
        the UNTOUCHED upstream ``iobrpy.workflow.runall.main`` (which
        spawns the ``iobrpy`` child CLIs through subprocess — requires the
        ``iobrpy`` console script and the external tools on PATH);
        ``rust`` fails clearly (pure orchestration, no native kernel).
    verbose : bool, default True
        Reproduce the substep console protocol and dispatch-layer banners
        (the child CLIs' output). ``False`` silences those; the
        orchestrator's own ``[run]/[ok]/[resume]`` lines always print
        (verbatim upstream).

    Returns
    -------
    dict
        ``{"rc": int}`` — the process exit code the upstream CLI produces
        (0 success; a failing step's rc, e.g. batch_salmon's 1/2 or
        trust4's code; 2 for usage errors / a missing merged matrix) —
        RETURNED instead of ``sys.exit`` (the documented API deviation;
        ``iobrx._fast.runall_fast.main(argv)`` keeps the upstream
        ``sys.exit(rc)`` behaviour for CLI-style use).
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "runall has no Rust kernel (pure orchestration over the ported "
            "substeps; the external-tool binaries do the heavy work); use "
            "backend='auto' or backend='python'"
        )
    tt = threads if threads is not None else n_threads
    tokens = ["--mode", str(mode), "--outdir", str(outdir),
              "--fastq", str(fastq)]
    if tt is not None:
        tokens += ["--threads", str(int(tt))]
    if batch_size is not None:
        tokens += ["--batch_size", str(int(batch_size))]
    if resume:
        tokens.append("--resume")
    if dry_run:
        tokens.append("--dry_run")
    tokens += [str(t) for t in (unknown or ())]

    from iobrx._fast.runall_fast import runall_argv, runall_original
    if backend == "python":
        return {"rc": runall_original(tokens)}
    return {"rc": runall_argv(tokens, verbose=verbose)}


# ---------------------------------------------------------------------------
# SpecHLA (single-sample HLA typing — external-tool orchestration)
# ---------------------------------------------------------------------------
def spechla(
    name=None,
    read1=None,
    read2=None,
    outdir=None,
    threads: int | None = None,
    use_exon: int | None = None,
    extra=(),
    spec_hla_root: str | None = None,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """SpecHLA full-resolution HLA typing for one sample (external-tool
    stage).

    Drop-in for ``iobrpy.SpecHLA.SpecHLA`` (CLI ``python -m iobrpy.main
    spechla``). The CLI surface is mirrored 1:1 (``-n/--name``,
    ``-1/--read1``, ``-2/--read2``, ``-o/--outdir``, ``-j/--threads``
    CLI default 8, ``-u/--use-exon`` {0,1} CLI default 1 = exon/RNA);
    anything else goes through ``extra`` (the upstream ``main.py``
    ``parse_known_args`` pass-through). The ported orchestration is
    BYTE-VERBATIM upstream: pysam/biopython probes (pip auto-install),
    ``SpecHLA/bin`` + fermikit PATH prepending, ``libwfa2.so.0``
    LD_LIBRARY_PATH, tool probes (samtools/bwa/bowtie2/freebayes/bgzip/
    tabix), bcftools==1.21 enforcement (``$BCFTOOLS`` -> bundled
    ``SpecHLA/bin/bcftools`` -> PATH -> conda), bundled-blastn gate,
    vcflib==1.0.10 conda check, SpecHap/ExtractHAIRs build-dir gate
    (install_spechap.sh fallback), DRB bowtie2 index build-if-missing,
    ``bash script/whole/SpecHLA_RNAseq.sh -n -1 -2 -o -j -u`` and the
    ``hla_result_merged.txt`` merge. With the same binaries, PATH wiring
    and frozen inputs, every product (``<outdir>/<name>/hla.result.txt``,
    the merged table and all tool intermediates) is identical to the
    original's — the stage is orchestration-bound (speedup ~ 1 by
    contract); no periphery was re-implemented. The ONLY resolution
    difference: the SpecHLA asset root (``script/`` + ``db/`` + ``bin/``)
    comes from ``spec_hla_root=`` > ``$SPECHLA_ROOT`` > the installed
    ``iobrpy`` package (upstream takes ``dirname(__file__)`` — it lives
    inside iobrpy; iobrx resolves the same tree).

    Parameters
    ----------
    name : str
        ``-n`` sample name (upstream-required; missing -> argparse rc=2).
    read1, read2 : str
        ``-1``/``-2`` paired FASTQ(.gz) paths (upstream-required).
    outdir : str
        ``-o`` output directory; results land in ``<outdir>/<name>/``
        (upstream-required).
    threads : int, optional
        ``-j`` threads (omitted -> the upstream CLI default 8).
    use_exon : {0,1}, optional
        ``-u`` pipeline type: 1 = exon/RNA (upstream default), 0 = WGS /
        full-length.
    extra : sequence of str, default ()
        Extra tokens appended to the upstream argv (pass-through).
    spec_hla_root : str, optional
        SpecHLA asset-tree override (see above). IGNORED under
        ``backend='python'`` (the upstream resolves from its own package).
    n_threads : int, optional
        iobrx-wide alias for ``threads`` (``threads`` wins when both are
        given).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``auto``: the verbatim ported orchestration; ``python``: the
        untouched upstream ``main(argv)``; ``rust`` fails clearly (no
        native kernel — the external binaries do the work).

    Returns
    -------
    dict
        ``{"rc": int}`` — the exit code the upstream CLI produces (0 on
        success — upstream ``main`` returns normally and prints its own
        IOBRpy banner verbatim; failure paths keep the upstream
        ``sys.exit`` codes: dependency/tool/conda gates 1, argparse 2,
        a failing pipeline command its own return code).
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "spechla has no Rust kernel (SpecHLA external-tool "
            "orchestration; the scheduling logic is verbatim upstream "
            "Python); use backend='auto' or backend='python'"
        )
    argv: list[str] = []
    if name is not None:
        argv += ["-n", str(name)]
    if read1 is not None:
        argv += ["-1", str(read1)]
    if read2 is not None:
        argv += ["-2", str(read2)]
    if outdir is not None:
        argv += ["-o", str(outdir)]
    tt = threads if threads is not None else n_threads
    if tt is not None:
        argv += ["-j", str(int(tt))]
    if use_exon is not None:
        argv += ["-u", str(int(use_exon))]
    argv += [str(x) for x in extra]

    if backend == "python":
        from iobrx._fast.spechla_fast import spechla_original as _run
        run_kwargs = {}
    else:
        from iobrx._fast.spechla_fast import main as _run
        run_kwargs = {"spec_hla_root": spec_hla_root}

    try:
        _run(argv, **run_kwargs)
        rc = 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return {"rc": rc}


# ---------------------------------------------------------------------------
# Batch HLA typing (BAM dir -> ExtractHLAread + SpecHLA -> merged table)
# ---------------------------------------------------------------------------
def hla_typing(
    bam_dir=None,
    ref=None,
    outdir=None,
    threads: int | None = None,
    use_exon: int | None = None,
    extra=(),
    spec_hla_root: str | None = None,
    n_threads: int | None = None,
    backend: str = "auto",
):
    """Batch HLA typing from a directory of BAM files (external-tool
    stage).

    Drop-in for ``iobrpy.workflow.hla_typing`` (CLI ``python -m iobrpy.main
    hla_typing``). The CLI surface is mirrored 1:1 (``-b/--bam-dir``,
    ``-r/--ref`` {hg19,hg38}, ``-o/--outdir``, ``-j/--threads`` CLI
    default 8, ``-u/--use-exon`` {0,1} CLI default 1); anything else goes
    through ``extra``. The ported orchestration is BYTE-VERBATIM upstream:
    sorted ``*.bam`` discovery with the ``_Aligned.sortedByCoord.out.bam``
    sample-id rule, the ExtractHLAread dependency gate
    (libdeflate==1.25 / htslib==1.21 conda checks when inside a conda
    env; samtools + bamUtil ``bam`` on PATH), the SEQUENTIAL per-sample
    ``bash script/ExtractHLAread.sh -s -b -r -o`` phase with
    ``<id>.ExtractHLAread.done`` resume markers, the SpecHLA phase
    (environment ensured ONCE, then per-sample ``SpecHLA_RNAseq.sh`` with
    ``<id>.SpecHLA.done`` markers gated on ``hla.result.txt`` containing
    ``Sample=<id>``), and the ``<outdir>/hla_result_merged.txt`` writer
    (version line + single header + data lines, duplicate-header
    suppression). With the same binaries, PATH wiring and frozen inputs,
    every product is identical to the original's — orchestration-bound
    (speedup ~ 1 by contract). SpecHLA asset-root resolution:
    ``spec_hla_root=`` > ``$SPECHLA_ROOT`` > the installed ``iobrpy``
    package (upstream derives it from ``__file__`` inside iobrpy; the
    ported ``run_extraction`` threads the resolved root instead — same
    tree, same command tokens).

    Parameters
    ----------
    bam_dir : str
        ``-b`` directory of ``*.bam`` files (upstream-required; no BAMs
        -> argparse rc=2, as the CLI).
    ref : {'hg19','hg38'}
        ``-r`` reference genome for ExtractHLAread (upstream-required,
        argparse-validated).
    outdir : str
        ``-o`` root output directory; ``ExtractHLAread/`` and ``SpecHLA/``
        subfolders plus ``hla_result_merged.txt`` are created inside
        (upstream-required).
    threads : int, optional
        ``-j`` threads (omitted -> the upstream CLI default 8).
    use_exon : {0,1}, optional
        ``-u`` SpecHLA pipeline type: 1 = exon/RNA (upstream default),
        0 = WGS.
    extra : sequence of str, default ()
        Extra tokens appended to the upstream argv (pass-through).
    spec_hla_root : str, optional
        SpecHLA asset-tree override. IGNORED under ``backend='python'``.
    n_threads : int, optional
        iobrx-wide alias for ``threads`` (``threads`` wins when both are
        given).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        ``auto``: the verbatim ported orchestration; ``python``: the
        untouched upstream ``main(argv)``; ``rust`` fails clearly (no
        native kernel).

    Returns
    -------
    dict
        ``{"rc": int}`` — the exit code the upstream CLI produces (0 on
        success — upstream ``main`` returns normally after printing its
        IOBRpy banner verbatim; dependency gates exit 1, argparse 2).
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "hla_typing has no Rust kernel (ExtractHLAread + SpecHLA "
            "external-tool orchestration; the scheduling logic is "
            "verbatim upstream Python); use backend='auto' or "
            "backend='python'"
        )
    argv: list[str] = []
    if bam_dir is not None:
        argv += ["-b", str(bam_dir)]
    if ref is not None:
        argv += ["-r", str(ref)]
    if outdir is not None:
        argv += ["-o", str(outdir)]
    tt = threads if threads is not None else n_threads
    if tt is not None:
        argv += ["-j", str(int(tt))]
    if use_exon is not None:
        argv += ["-u", str(int(use_exon))]
    argv += [str(x) for x in extra]

    if backend == "python":
        from iobrx._fast.hla_typing_fast import hla_typing_original as _run
        run_kwargs = {}
    else:
        from iobrx._fast.hla_typing_fast import main as _run
        run_kwargs = {"spec_hla_root": spec_hla_root}

    try:
        _run(argv, **run_kwargs)
        rc = 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return {"rc": rc}
