"""iobrx — bit-exact, multithreaded acceleration layer for IOBRpy.

``iobrx`` wraps the IOBRpy (https://pypi.org/project/iobrpy/) tumor
micro-environment workflows in accelerated implementations whose outputs are
bit-identical to the originals on the official validation data (all 11
official gates: max abs diff 0.0, cibersort's unseeded P-value column excepted
by design). Heavy solvers (CIBERSORT's NuSVR, ssGSEA, PCA signature scoring)
run in a Rust extension (rayon threads + vendored libsvm/OpenBLAS entry
points); the remaining stages are vectorized numpy/pandas with per-process
resource caching.

Quick start::

    import iobrx, pandas as pd

    eset = pd.read_parquet("imvigor210_eset.parquet")        # genes x samples
    cib = iobrx.cibersort(eset, perm=100, QN=True)           # ~50x faster
    scores = iobrx.calculate_sig_score(eset, "signature_collection",
                                       method="pca")

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

__version__ = "0.1.0"

# Import the compiled extension eagerly (fail fast with a clear error if the
# Rust half is missing, e.g. a pure-python install) and register the
# historical top-level alias `iobrx_rust` for it, so `import iobrx_rust`
# keeps working in code written against the development stack once iobrx has
# been imported. The extension's canonical import path is `iobrx._rust`.
from . import _rust as _rust_ext  # noqa: E402

sys.modules.setdefault("iobrx_rust", _rust_ext)

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
    "set_threads",
    "get_threads",
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
        Quantile-normalize the mixture before fitting (recommended, and the
        only mode validated bit-exact on the official data).
    absolute : bool, default False
        Also report the absolute-score column.
    abs_method : {'sig.score', 'no.sumto1'}, default 'sig.score'
        Absolute-mode scoring rule (as IOBRpy).
    n_threads : int or None, optional
        Rayon threads for the solver. ``None`` resolves to
        ``min(8, os.cpu_count())`` (see :func:`set_threads`).

    Returns
    -------
    pandas.DataFrame
        One row per sample: 22 LM22 cell-type weight columns (float32, like
        the original) plus ``P-value``, ``Correlation``, ``RMSE`` and, when
        ``absolute=True``, ``Absolute_score_(<abs_method>)``.
    """
    from iobrx._fast.cibersort_fast import cibersort_fast

    return cibersort_fast(
        eset,
        perm=perm,
        QN=QN,
        absolute=absolute,
        abs_method=abs_method,
        n_threads=resolve_threads(n_threads),
    )


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

    Returns
    -------
    pandas.DataFrame
        One row per sample (``ID`` column first), one column per surviving
        signature; ``TMEscore_CIR`` / ``TMEscore_plus`` contrasts are appended
        when both constituents are present, matching the original.
    """
    from iobrx._fast.sig_score_fast import calculate_sig_score_fast

    names = [signature] if isinstance(signature, str) else list(signature)
    return calculate_sig_score_fast(
        eset,
        names,
        method,
        mini_gene_count,
        adjust_eset,
        resolve_threads(n_threads),
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
        ``'affy'`` / ``'affymetrix'`` adds the ``TumorPurity`` row (the
        cos-based conversion), as upstream.

    Returns
    -------
    pandas.DataFrame
        Rows ``StromalSignature`` / ``ImmuneSignature`` / ``ESTIMATEScore``
        (+ ``TumorPurity`` on affy platforms), columns = samples.
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
