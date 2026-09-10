"""bayesprism_fast: accelerated drop-in for the iobrpy ``bayesprism`` pipeline.

Bit-exact drop-in for ``iobrpy.bayesprism.bayesprism.run_bayesprism`` (the
``python -m iobrpy.main bayesprism`` CLI): same output contract
(``theta.csv`` / ``theta_cv.csv`` / ``Z_tumor.csv``, ``_BayesPrism`` column
suffixes, ``bulk_id`` Z index) with ONE documented determinism fix
(``state_order``, see below) and accelerated non-sampling stages.

Architecture (this round is the pure-Python lane; a native Gibbs kernel is a
follow-up round, so ``backend='auto'``/``'rust'`` both take this fast Python
path and ``backend='python'`` delegates to the ORIGINAL iobrpy code):

  * REUSED VERBATIM BY IMPORT from ``iobrpy.bayesprism`` (zero divergence
    risk on the numerically delicate parts):
      - ``optim``      — the MAP reference update (scipy ``CG`` per cell
        type, Pool-parallel, order-preserving ``starmap``). Its iteration
        path decides psi to the last ULP; not touched.
      - ``references`` — ``RefPhi`` / ``RefTumor`` containers. The ORIGINAL
        ``optim`` and the forked Gibbs ``isinstance`` checks must see the
        SAME class objects, so these are imported, never re-defined.
      - ``theta_post`` — trivial assembly of per-sample dicts into frames.
      - ``extract``    — ``get_fraction`` / ``get_exp`` (``get_exp`` asserts
        ``isinstance(bp, BayesPrism)``, so the ORIGINAL ``prism.BayesPrism``
        container class is imported and instantiated unchanged).
      - ``prism.BayesPrism`` — result container (see above).

  * FORKED WITH BIT-EXACT HOT-SPOT REPLACEMENTS (each deviation is provably
    value-identical; see comments at each site):
      - ``process_input``: ``select_gene_type`` (the O(n^2)
        ``list.index``/``in`` scans — ~2.0 s on the 4869-gene bundled
        reference — become cached first-occurrence dicts), ``cleanup_genes``
        (the 30M-element pandas ``(x > 0).sum(axis=0)`` becomes
        ``np.count_nonzero``; ``assign_category`` caches the gene tables and
        uses frozenset membership instead of per-category ``np.isin``),
        ``collapse`` (per-label boolean-mask copies + ``pd.concat`` chain
        become one stable argsort + ``np.add.reduceat`` — integer sums are
        exact and order-independent; for non-integer input each segment is a
        contiguous slice summed by the same numpy pairwise-summation as the
        original per-label ``np.sum``). ``norm_to_one``, ``validate_input``
        and ``filter_bulk_outlier`` are copied VERBATIM (float operation
        order matters).
      - ``gibbs``: copied verbatim EXCEPT the per-iteration ``if i in
        gibbs_idx`` (an O(len(gibbs_idx)) numpy elementwise comparison)
        becomes a precomputed ``frozenset`` membership test with identical
        truth values. The RNG call sequence is untouched: sequential
        ``rng.multinomial`` per gene (G calls/iteration), ``rng.gamma``
        Dirichlet, ``_make_rng`` = ``Generator(MT19937(SeedSequence))``,
        ``SeedSequence(123).spawn(n_samples)`` per-sample child streams for
        phase 1/3, the phase-3 per-sample ``spawn(1)[0]`` re-spawn quirk
        (EVERY sample gets the SAME child seed — preserved), the fork-Pool
        per-sample parallelism with order-preserving ``starmap``, the
        ``chunksize`` formula and the ``estimate_gibbs_time`` probe chains.
        Same numpy version => same streams => same draws, bit for bit.
      - ``joint_post``: ``JointPost.new`` verbatim; ``merge_K`` replaces the
        per-type xarray label ``.loc`` gathers with numpy fancy-index
        gathers into preallocated buffers. Addends keep the ORIGINAL order
        (the ``map_[type]`` state list), each subset is C-contiguous, so
        ``np.sum(axis=...)`` runs the same pairwise-summation tree over the
        same value sequence: bit-identical sums. The single-state
        ``np.squeeze`` quirk (which also squeezes n/g singleton dims) is
        preserved.
      - ``prism.Prism``: ``new``/``run`` copied verbatim (all prints,
        warnings, the ``pd.unique(list)`` FutureWarning quirk, the
        single-column mixture transpose quirk, the RangeIndex rename quirk,
        the mutable-default ``gibbs_control``/``opt_control`` quirk) with
        two changes: (1) the ``expressed_mask`` gene filter uses
        ``np.count_nonzero(arr > 0, axis=0)`` instead of
        ``np.sum(df > 0, axis=0)`` (identical mask, no pandas boolean
        frame); (2) the ``map_`` construction gains ``state_order``.

  * DETERMINISM FIX (``state_order``, the ONE intentional semantic change):
    the ORIGINAL builds
    ``map_ = {cell_type: list(set(states))}`` (prism.py ~L170). CPython set
    iteration order over strings depends on the per-process PYTHONHASHSEED,
    so ``merge_K``'s summation order — and hence the merged Z/theta bits —
    varies ACROSS PROCESSES; the ULP differences propagate into the MAP
    reference update and occasionally flip discrete Gibbs draws, moving one
    sample's whole theta row by ~5e-3 (verified: the ORIGINAL CLI run twice
    with a random hash seed gives different output SHA-256s; with
    PYTHONHASHSEED pinned it is bit-reproducible).
    ``state_order='sorted'`` (default) uses ``sorted(set(states))``: a
    semantic, hash-seed-independent order, making iobrx byte-identical
    across processes WITHOUT pinning PYTHONHASHSEED. ``state_order='legacy'``
    reproduces the ORIGINAL ``list(set(...))`` expression verbatim and — in
    a process with the same PYTHONHASHSEED as an ORIGINAL run — reproduces
    the ORIGINAL output bytes exactly (verified against the frozen
    PYTHONHASHSEED=0 gold-standard run: all three CSVs SHA-256-identical).

  * I/O ACCELERATION: the bundled 60 MB ``BP_data/sc_dat.csv`` (and any
    user-provided single-cell CSV) is parsed with the ORIGINAL
    ``pd.read_csv`` C-parser semantics on the FIRST read (float parsing
    unchanged — no parquet shortcut), then memoized per process and cached
    as a pickle keyed by (resolved path, mtime_ns, size, sep, compression,
    pandas/numpy versions) under ``$IOBRX_CACHE`` or
    ``~/.cache/iobrx/bayesprism``. Corrupt/unwritable caches silently fall
    back to the CSV parse. The ``txt/`` gene tables (gencode.v22 category
    table 5.7 MB, genelist tables) are parsed once per process.

Known upstream defects preserved bug-compatibly (NOT fixed, per porting
contract): phase-3 shared ``spawn(1)[0]`` child seed; ``--sc_dat`` CLI help
says "genes x cells" but the code requires cells x genes; the ``MLE``
optimizer path is broken under scipy>=1.16 (untouched — it lives in the
reused ORIGINAL ``optim``); ``pd.unique(list)`` FutureWarning;
``theta_cv`` of the merged cell-type posterior is dropped (``None``);
``Z_tumor`` comes from the INITIAL (pre-update) posterior while ``theta``
comes from the FINAL (post-update) one.
"""
from __future__ import annotations

import hashlib
import importlib
import multiprocessing
import os
import time
from datetime import datetime, timedelta
from itertools import compress, repeat
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from iobrx._resources import resource_path

__all__ = [
    "bayesprism",
    "run_bayesprism_fast",
    "PrismFast",
    "GibbsSamplerFast",
    "JointPostFast",
    "cleanup_genes",
    "select_gene_type",
    "norm_to_one",
    "collapse",
    "validate_input",
    "filter_bulk_outlier",
    "assign_category",
]

# The ORIGINAL run_bayesprism reference-cleaning constants (CLI surface).
_SPECIES = "hs"
_GENE_GROUP = ["Rb", "Mrp", "other_Rb", "chrM", "MALAT1", "chrX", "chrY"]
_EXP_CELLS = 5
_DEFAULT_KEY = "Malignant_cells"

# ---------------------------------------------------------------------------
# lazy access to the REUSED original modules (keeps `import iobrx` light)
# ---------------------------------------------------------------------------
_iobrpy_mods: dict = {}


def _bp_mod(name: str):
    """Import (once) ``iobrpy.bayesprism.<name>`` — the ORIGINAL modules."""
    if name not in _iobrpy_mods:
        _iobrpy_mods[name] = importlib.import_module(f"iobrpy.bayesprism.{name}")
    return _iobrpy_mods[name]


# ---------------------------------------------------------------------------
# per-process resource caches (repo convention: lazy, module level)
# ---------------------------------------------------------------------------
_GENCODE_TAB = None          # gencode.v22.broad.category.txt (DataFrame)
_GENCODE_FIRST: dict = {}    # column -> {value: first row index}
_GENELIST_TAB: dict = {}     # species -> genelist.<species>.new.txt (DataFrame)
_GENELIST_SETS: dict = {}    # (species, col) -> (categories, {cat: frozenset})
_BUNDLED: dict = {}          # bundled BP_data frames, memoized per process


def _bp_data_path(filename: str) -> Path:
    """Real filesystem path of a bundled ``BP_data`` file (repo convention:
    ``importlib.resources.files`` on the iobrpy package)."""
    from importlib.resources import files

    return Path(resource_path(f"BP_data/{filename}"))


def _txt_path(filename: str) -> str:
    from importlib.resources import files

    return resource_path(f"txt/{filename}")


def _gencode_table() -> pd.DataFrame:
    global _GENCODE_TAB
    if _GENCODE_TAB is None:
        # ORIGINAL: pd.read_table(gene_tab_path, sep="\t", header=None)
        _GENCODE_TAB = pd.read_table(
            _txt_path("gencode.v22.broad.category.txt"), sep="\t", header=None
        )
    return _GENCODE_TAB


def _gencode_first(col: int) -> dict:
    """{value: FIRST row index} for one gencode column — the dict form of the
    ORIGINAL's ``gene_list_N.index(x)`` linear scans (identical results:
    ``list.index`` returns the first occurrence)."""
    if col not in _GENCODE_FIRST:
        d = {}
        for i, v in enumerate(_gencode_table().iloc[:, col].tolist()):
            if v not in d:
                d[v] = i
        _GENCODE_FIRST[col] = d
    return _GENCODE_FIRST[col]


def _genelist_table(species: str) -> pd.DataFrame:
    if species not in _GENELIST_TAB:
        # ORIGINAL: pd.read_table(gene_list_path, sep="\t", header=None)
        _GENELIST_TAB[species] = pd.read_table(
            _txt_path(f"genelist.{species}.new.txt"), sep="\t", header=None
        )
    return _GENELIST_TAB[species]


def _genelist_sets(species: str, col: int):
    key = (species, col)
    if key not in _GENELIST_SETS:
        gene_list = _genelist_table(species)
        cats = gene_list.iloc[:, 0]
        # pandas .unique(): FIRST-occurrence order — same column order as the
        # ORIGINAL's gene_df["category"].unique() over the same table.
        categories = list(pd.unique(cats))
        genes_col = gene_list.iloc[:, col]
        sets = {c: frozenset(genes_col[cats == c].tolist()) for c in categories}
        _GENELIST_SETS[key] = (categories, sets)
    return _GENELIST_SETS[key]


# ---------------------------------------------------------------------------
# process_input fork (vectorized hot spots; float paths verbatim)
# ---------------------------------------------------------------------------
def norm_to_one(ref, pseudo_min):
    """VERBATIM copy of iobrpy.bayesprism.process_input.norm_to_one
    (float operation order is load-bearing: row-normalize with pseudo_min,
    then the hidden second branch overwriting rows whose min > 0)."""
    ref_index = ref.index
    ref_columns = ref.columns

    ref = ref.to_numpy()
    g = ref.shape[1]

    phi = ref / np.sum(ref, axis=1, keepdims=True) * (1 - pseudo_min * g) + pseudo_min

    min_value = np.min(ref, axis=1)
    which_row = min_value > 0
    if np.any(which_row):
        phi[which_row, :] = ref[which_row, :] / np.sum(ref[which_row, :], axis=1, keepdims=True)

    phi = pd.DataFrame(phi, index=ref_index, columns=ref_columns)
    return phi


def collapse(ref, labels):
    """Vectorized equivalent of process_input.collapse (same-label cell
    counts SUMMED into one row; label order = first occurrence; None labels
    dropped with the ORIGINAL warning).

    Bit-exactness: the ORIGINAL loops labels, builds a boolean mask and calls
    ``np.sum(ref.loc[mask].to_numpy(), axis=0)`` per label. For integer
    input (the pipeline's case: references are ``astype(np.int32)``
    upstream) the sums are exact and order-independent, so a one-hot
    ``S @ arr`` in float64 (every partial sum an integer < 2^53 — guarded by
    dtype width and a row-count cap) ``.astype(np.int64)`` yields the
    IDENTICAL values with the same accumulator widening ``np.sum`` applies;
    it measures ~5x faster than a sort+gather+``np.add.reduceat`` variant
    and ~3x faster than the per-label copies. For float input (where summation
    ORDER moves bits) and out-of-guard integers, one stable argsort groups
    the cells (cell order within a label preserved) and a single
    ``np.add.reduceat`` sums each contiguous segment with the same numpy
    pairwise summation over the same value sequence → identical bits.
    Pathological labels (NaN — where the ORIGINAL's ``label == x`` comparison
    is always False and yields an all-zero row — or unhashable) fall back to
    the ORIGINAL loop, copied verbatim.
    """
    assert ref.shape[0] == len(labels), "Error: nrow(ref) and length(labels) do not match!"

    non_na_idx = [x is not None for x in labels]
    if non_na_idx.count(False) > 0:
        print("Warning: NA found in the cell type/state labels. These cells will be excluded!")
    labels = list(compress(labels, non_na_idx))
    ref = ref.loc[non_na_idx, :]

    labels_seen = set()
    labels_uniq = [x for x in labels if not (x in labels_seen or labels_seen.add(x))]

    fast = True
    try:
        if pd.Index(labels).hasnans:  # NaN labels: ORIGINAL semantics give zero rows
            fast = False
    except TypeError:
        fast = False

    if fast and len(labels_uniq) > 0:
        pos = {lab: j for j, lab in enumerate(labels_uniq)}
        try:
            codes = np.fromiter((pos[x] for x in labels), dtype=np.intp, count=len(labels))
        except TypeError:  # unhashable labels
            codes = None
        if codes is not None:
            arr = ref.to_numpy()
            n_cells = arr.shape[0]
            k = len(labels_uniq)
            if (np.issubdtype(arr.dtype, np.signedinteger)
                    and arr.dtype.itemsize <= 4 and n_cells <= (1 << 20)):
                # Exact integer path: one-hot matmul. Every partial sum is an
                # integer < 2^53 (n_cells <= 2^20 rows x int32 magnitudes
                # <= 2^31), so float64 accumulation is EXACT and the result
                # equals the ORIGINAL's per-label np.sum (int64) value for
                # value, independent of summation order.
                S = np.zeros((k, n_cells))
                S[codes, np.arange(n_cells)] = 1.0
                sums = (S @ arr.astype(np.float64)).astype(np.int64)
            else:
                # Bit-exact generic path: stable argsort keeps cell order
                # within each label; each segment is a contiguous slice, so
                # np.add.reduceat runs the same pairwise summation as the
                # ORIGINAL np.sum over the same value sequence.
                if np.issubdtype(arr.dtype, np.signedinteger):
                    sum_dtype = np.dtype(np.int64) if arr.dtype.itemsize < 8 else None
                elif np.issubdtype(arr.dtype, np.unsignedinteger):
                    sum_dtype = np.dtype(np.uint64) if arr.dtype.itemsize < 8 else None
                else:
                    sum_dtype = None
                order = np.argsort(codes, kind="stable")
                counts = np.bincount(codes, minlength=k)
                starts = np.cumsum(counts) - counts
                sums = np.add.reduceat(arr[order], starts, axis=0, dtype=sum_dtype)
            ref_collapsed = pd.DataFrame(sums, index=labels_uniq, columns=ref.columns)
            return ref_collapsed

    # Fallback: ORIGINAL loop, verbatim.
    ref_collapsed = pd.DataFrame()
    for label_i in labels_uniq:
        indices = [label_i == i for i in labels]
        ref_label = ref.loc[indices].to_numpy()
        ref_collapsed = pd.concat([ref_collapsed, pd.Series(np.sum(ref_label, axis=0)).to_frame().T], ignore_index=True)
    ref_collapsed.index = labels_uniq
    ref_collapsed.columns = ref.columns

    return ref_collapsed


def validate_input(input):
    """VERBATIM copy of process_input.validate_input (including the
    isinstance check AFTER the ndarray use — upstream ordering quirk)."""
    input_ndarray = input.to_numpy()
    if np.max(input_ndarray) <= 1:
        print("Warning: input seems to be normalized.")
    elif np.max(input_ndarray) < 20:
        print("Warning: input seems to be log-transformed. Please double \
              check your input. Log transformation should be avoided")

    if np.any(input_ndarray < 0) or np.any(np.isinf(input_ndarray)):
        raise ValueError("Error: input contains negative, non-finite or \
                         non-numeric values. Please double check your \
                         input is unnormalized and untransformed raw count.")

    if input.columns.empty:
        raise ValueError("Error: please specify the colnames of mixture / \
                         reference using gene identifiers!")

    if not isinstance(input, pd.DataFrame):
        raise ValueError("Error: the type of mixture and reference need \
                         to be DataFrame!")


def filter_bulk_outlier(mixture, outlier_cut, outlier_fraction):
    """VERBATIM copy of process_input.filter_bulk_outlier (float order
    load-bearing: per-sample normalization then per-gene outlier fraction)."""
    mixture_ndarray = mixture.to_numpy()

    mixture_norm = mixture_ndarray / np.sum(mixture_ndarray, axis=1, keepdims=True)

    outlier_idx = np.sum(mixture_norm > outlier_cut, axis=0) / mixture_norm.shape[0] > outlier_fraction

    mixture = mixture.loc[:, ~outlier_idx]
    print("Number of outlier genes filtered from mixture =", np.sum(outlier_idx))
    return mixture


def assign_category(input_genes, species):
    """Cached/set-based equivalent of process_input.assign_category.

    Same returned boolean DataFrame (rows = input genes in input order,
    columns = categories in the gene table's first-occurrence order). The
    ORIGINAL's per-category ``np.isin`` becomes frozenset membership —
    identical booleans (duplicates inside a category are irrelevant for
    membership). Gene tables are parsed once per process.
    """
    assert species in ["hs", "mm"], "species must be 'hs' or 'mm'"

    # Detect whether input genes are mostly Ensembl IDs or gene symbols
    # (ORIGINAL detection expression, verbatim).
    use_ensembl = sum(gene[:3] == "ENS" for gene in input_genes) > len(input_genes) * 0.8

    if use_ensembl:
        print("EMSEMBLE IDs detected.")
        # Strip version suffix from Ensembl IDs (e.g. ENSG000001.1 -> ENSG000001)
        input_genes_short = [gene.split(".")[0] for gene in input_genes]
        col = 1  # gene_list: column 0 = category, column 1 = Ensembl ID
    else:
        print("Gene symbols detected.")
        input_genes_short = list(input_genes)
        col = 2  # gene_list: column 0 = category, column 2 = gene symbol

    categories, sets = _genelist_sets(species, col)

    n = len(input_genes_short)
    logic = np.empty((n, len(categories)), dtype=bool)
    for j, cat in enumerate(categories):
        s = sets[cat]
        logic[:, j] = np.fromiter((g in s for g in input_genes_short), dtype=bool, count=n)

    gene_group_matrix = pd.DataFrame(
        logic,
        index=input_genes,     # keep original gene names as index
        columns=categories,    # category names as columns
    )

    return gene_group_matrix


def cleanup_genes(input: pd.DataFrame, input_type, species, gene_group, exp_cells=1):
    """Equivalent of process_input.cleanup_genes with the hot spots
    vectorized: ``assign_category`` (cached tables + set membership) and the
    low-expression filter — the ORIGINAL's ``(input_filtered > 0).sum(axis=0)``
    materializes a pandas boolean frame (30M cells on the bundled reference,
    ~1 s); ``np.count_nonzero(arr > 0, axis=0)`` gives the identical counts.
    All prints and the GEP ``exp_cells`` clamp are preserved.
    """
    assert species in ["hs", "mm"]
    a = ["other_Rb", "chrM", "chrX", "chrY", "Rb", "Mrp", "act", "hb", "MALAT1"]
    assert all([g in a for g in gene_group])
    assert input_type in ["GEP", "count.matrix"]

    if input_type == "GEP":
        exp_cells = min(exp_cells, 1)
        print("As the input is a collpased GEP, exp.cells is set to min(exp.cells,1)")

    category_matrix = assign_category(input_genes=input.columns, species=species)
    category_matrix = category_matrix.loc[:, gene_group]

    print("number of genes filtered in each category: ")
    print(np.sum(category_matrix, axis=0))

    exclude_idx = np.sum(category_matrix, axis=1) > 0
    print("A total of", exclude_idx.sum(), "genes from", gene_group, "have been excluded")
    input_filtered = input.loc[:, ~exclude_idx]

    if exp_cells > 0:
        exclude_lowexp_idx = np.count_nonzero(input_filtered.to_numpy() > 0, axis=0) >= exp_cells
        print("A total of", (~exclude_lowexp_idx).sum(), "gene expressed in fewer than", exp_cells, "cells have been excluded")
        input_filtered = input_filtered.loc[:, exclude_lowexp_idx]
    else:
        print("A total of 0 lowly expressed genes have been excluded")

    return input_filtered


def select_gene_type(input: pd.DataFrame, gene_type):
    """Equivalent of process_input.select_gene_type with the O(n^2) scan
    removed. The ORIGINAL does, per input gene, ``x in gene_list_N`` (linear
    membership) AND ``gene_list_N.index(x)`` (another linear scan) over the
    ~60k-row gencode table — ~2.0 s for 4869 genes. Here a per-process
    cached {value: FIRST row index} dict answers both in O(1) with identical
    results (``list.index`` = first occurrence; missing genes are dropped
    exactly as the ORIGINAL's ``if x in`` filter does). Row gather, category
    filter and prints are verbatim.
    """
    assert all([g in ["protein_coding", "pseudogene", "lincRNA"] for g in gene_type])

    input_genes = input.columns
    gene_list = _gencode_table()

    if sum(1 for gene in input_genes if gene[:3] == "ENS") > len(input_genes) * 0.8:
        print("EMSEMBLE IDs detected.")
        input_genes = [gene.split(".")[0] for gene in input_genes]
        col = 7
    else:
        print("Gene symbols detected.")
        col = 4

    first_pos = _gencode_first(col)
    gene_match = [first_pos[x] for x in input_genes if x in first_pos]
    gene_df = gene_list.iloc[gene_match, [col, 8]]

    gene_df.columns = ["gene_name", "category"]

    selected_gene_idx = gene_df["category"].isin(gene_type)

    print("number of genes retained in each category: ")
    print(gene_df.loc[selected_gene_idx, "category"].value_counts())
    input_filtered = input.loc[:, selected_gene_idx.tolist()]
    return input_filtered


# ---------------------------------------------------------------------------
# gibbs fork — RNG call sequence VERBATIM (see module docstring)
# ---------------------------------------------------------------------------
def multinomial_rvs(count, p, rng=None, method="binomial"):
    """VERBATIM copy of iobrpy.bayesprism.gibbs.multinomial_rvs (both the
    sequential per-gene ``rng.multinomial`` path — the DEFAULT, whose RNG
    consumption defines the output — and the vectorized binomial-chain
    ``fast.multinomial`` path)."""
    if rng is None:
        rng = np.random.default_rng()

    p = np.asarray(p)
    count = np.array(count, copy=True)

    if method == "sequential":
        flat_samples = np.array(
            [rng.multinomial(int(n), prob) for n, prob in zip(count.reshape(-1), p.reshape(-1, p.shape[-1]))],
            dtype=int,
        )
        return flat_samples.reshape(p.shape)

    if method != "binomial":
        raise ValueError("Unsupported multinomial sampling method")

    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = rng.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


class GibbsSamplerFast:
    """Fork of iobrpy.bayesprism.gibbs.GibbsSampler.

    Every method is a verbatim copy EXCEPT the per-iteration retention test
    in ``sample_Z_theta_n`` / ``sample_theta_n``: the ORIGINAL's
    ``if i in gibbs_idx`` runs an elementwise numpy comparison against all
    250 retained indices every iteration; a precomputed frozenset of the
    same (unique, integer) indices gives identical truth values in O(1).
    Nothing in the RNG path is touched.
    """

    def __init__(self, reference, X, gibbs_control):
        self.reference = reference
        self.X = X
        self.gibbs_control = gibbs_control

    def get_gibbs_idx(gibbs_control):
        chain_length = gibbs_control['chain.length']
        burn_in = gibbs_control['burn.in']
        thinning = gibbs_control['thinning']
        all_idx = np.arange(0, chain_length)
        burned_idx = all_idx[int(burn_in):]
        thinned_idx = burned_idx[np.arange(0, len(burned_idx), thinning)]
        return thinned_idx

    def rdirichlet(alpha, rng=None, backend="generator"):
        """Dirichlet sampling using the provided RNG for determinism.
        (VERBATIM: one ``rng.gamma(alpha_vec, size=K)`` call + normalize.)"""
        if rng is None:
            rng = GibbsSamplerFast._make_rng(None, backend=backend)
        x = rng.gamma(alpha, size=len(alpha))
        return x / np.sum(x)

    def _make_rng(seed, backend="generator"):
        """VERBATIM copy of the ORIGINAL (Generator+MT19937(SeedSequence)
        main path; randomstate fallback; unseeded branches)."""
        backend = backend.lower()
        if backend not in {"generator", "randomstate"}:
            raise ValueError("Unsupported RNG backend; use 'generator' or 'randomstate'")

        if backend == "randomstate":
            if isinstance(seed, np.random.RandomState):
                return seed
            if isinstance(seed, np.random.SeedSequence):
                seed = int(seed.generate_state(1, dtype=np.uint32)[0])
            if isinstance(seed, np.random.Generator):
                seed = seed.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
            return np.random.RandomState(seed)

        # Generator backend
        if isinstance(seed, np.random.Generator):
            return seed
        if isinstance(seed, np.random.RandomState):
            seed = seed.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32)
        if isinstance(seed, np.random.SeedSequence):
            return np.random.Generator(np.random.MT19937(seed))
        if seed is None:
            return np.random.Generator(np.random.MT19937())
        return np.random.Generator(np.random.MT19937(seed))

    def _spawn_seeds(seed, n_children, backend="generator"):
        """VERBATIM copy: ``SeedSequence(seed).spawn(n_children)`` for the
        generator backend (per-sample child streams, phase 1/3), legacy
        ``RandomState.randint`` chain for the randomstate backend."""
        if seed is None:
            return [None] * n_children

        backend = backend.lower()
        if backend == "generator":
            seed_seq = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
            return seed_seq.spawn(n_children)

        if backend == "randomstate":
            base_rng = GibbsSamplerFast._make_rng(seed, backend=backend)
            return [base_rng.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32) for _ in range(n_children)]

        raise ValueError("Unsupported RNG backend; use 'generator' or 'randomstate'")

    def sample_Z_theta_n(
        X_n,
        phi,
        alpha,
        gibbs_idx,
        chain_length,
        seed=None,
        rng_backend="generator",
        compute_elbo=False,
        fast_multinomial=False,
    ):
        """VERBATIM except the frozenset retention test (see class docstring)."""
        rng = GibbsSamplerFast._make_rng(seed, backend=rng_backend)

        phi = phi.to_numpy()
        G = phi.shape[1]
        K = phi.shape[0]

        theta_n_i = np.repeat(1 / K, K)
        Z_n_i = np.empty((G, K))

        Z_n_sum = np.zeros((G, K))
        theta_n_sum = np.zeros(K)
        theta_n2_sum = np.zeros(K)

        multinom_coef = 0

        iterations = chain_length if chain_length is not None else (np.max(gibbs_idx) + 1)
        keep = frozenset(int(v) for v in np.asarray(gibbs_idx).ravel())

        for i in range(iterations):
            prob_mat = phi * theta_n_i[:, np.newaxis]
            prob_mat /= prob_mat.sum(axis=0, keepdims=True)

            Z_n_i = multinomial_rvs(
                count=X_n,
                p=prob_mat.T,
                rng=rng,
                method="binomial" if fast_multinomial else "sequential",
            )

            Z_nk_i = np.sum(Z_n_i, axis=0)
            theta_n_i = GibbsSamplerFast.rdirichlet(alpha=Z_nk_i + alpha, rng=rng, backend=rng_backend)

            if i in keep:
                Z_n_sum += Z_n_i
                theta_n_sum += theta_n_i
                theta_n2_sum += theta_n_i**2
                if compute_elbo:
                    import scipy
                    multinom_coef += np.sum(np.log(scipy.special.factorial(Z_nk_i))) - np.sum(
                        np.log(scipy.special.factorial(Z_n_i))
                    )

        samples_size = len(gibbs_idx)
        Z_n = Z_n_sum / samples_size
        theta_n = theta_n_sum / samples_size
        theta_cv_n = np.sqrt(theta_n2_sum / samples_size - (theta_n ** 2)) / theta_n
        gibbs_constant = multinom_coef / samples_size

        return {
            'Z_n': Z_n,
            'theta_n': theta_n,
            'theta.cv_n': theta_cv_n,
            'gibbs.constant': gibbs_constant,
        }

    def sample_theta_n(X_n, phi, alpha, gibbs_idx, chain_length, seed=None, rng_backend="generator", fast_multinomial=False):
        """VERBATIM except the frozenset retention test (see class docstring)."""
        rng = GibbsSamplerFast._make_rng(seed, backend=rng_backend)

        phi = phi.to_numpy()
        G = phi.shape[1]
        K = phi.shape[0]

        theta_n_i = np.repeat(1 / K, K)
        Z_n_i = np.empty((G, K))

        theta_n_sum = np.zeros(K)
        theta_n2_sum = np.zeros(K)

        iterations = chain_length if chain_length is not None else (np.max(gibbs_idx) + 1)
        keep = frozenset(int(v) for v in np.asarray(gibbs_idx).ravel())

        for i in range(iterations):
            prob_mat = phi * theta_n_i[:, np.newaxis]
            prob_mat /= prob_mat.sum(axis=0, keepdims=True)
            Z_n_i = multinomial_rvs(
                count=X_n,
                p=prob_mat.T,
                rng=rng,
                method="binomial" if fast_multinomial else "sequential",
            )

            theta_n_i = GibbsSamplerFast.rdirichlet(alpha=np.sum(Z_n_i, axis=0) + alpha, rng=rng, backend=rng_backend)

            if i in keep:
                theta_n_sum += theta_n_i
                theta_n2_sum += theta_n_i**2

        samples_size = len(gibbs_idx)
        theta_n = theta_n_sum / samples_size
        theta_cv_n = np.sqrt(theta_n2_sum / samples_size - (theta_n**2)) / theta_n

        return {'theta_n': theta_n, 'theta.cv_n': theta_cv_n}

    def my_seconds_to_period(x):
        days = round(x // (60 * 60 * 24))
        hours = round((x - days * 60 * 60 * 24) // (60 * 60))
        minutes = round((x - days * 60 * 60 * 24 - hours * 60 * 60) // 60) + 1
        days_str = '' if days == 0 else str(days) + 'days '
        hours_str = '' if (hours == 0 and days == 0) else str(hours) + 'hrs '
        minutes_str = '' if (minutes == 0 and days == 0 and hours == 0) else str(minutes) + 'mins'
        final_str = days_str + hours_str + minutes_str
        return final_str

    def estimate_gibbs_time(self, final, chain_length=50):
        """VERBATIM copy: 50-iteration single-sample probe chain (seed used
        directly, NOT spawned; result discarded — it cannot affect the main
        chains' streams) plus the ORIGINAL time-estimate prints."""
        references = _bp_mod("references")
        ref = self.reference
        X = self.X.to_numpy()
        gibbs_control = self.gibbs_control
        fast_mult = gibbs_control.get('fast.multinomial', False)
        rng_backend = gibbs_control.get('rng.backend', 'generator')
        ptm = time.process_time()

        if not final:
            assert isinstance(ref, references.RefPhi), "Gibbs is not final but ref is not refPhi"
            GibbsSamplerFast.sample_Z_theta_n(
                X_n=X[0, :],
                phi=ref.phi,
                alpha=gibbs_control['alpha'],
                gibbs_idx=GibbsSamplerFast.get_gibbs_idx(
                    {'chain.length': chain_length,
                     'burn.in': chain_length * gibbs_control['burn.in'] / gibbs_control['chain.length'],
                     'thinning': gibbs_control['thinning']}),
                chain_length=chain_length,
                seed=gibbs_control['seed'],
                rng_backend=rng_backend,
                compute_elbo=False,
                fast_multinomial=fast_mult,
            )
        else:
            if isinstance(ref, references.RefPhi):
                GibbsSamplerFast.sample_theta_n(
                    X_n=X[0, :],
                    phi=ref.phi,
                    alpha=gibbs_control['alpha'],
                    gibbs_idx=GibbsSamplerFast.get_gibbs_idx(
                        {'chain.length': chain_length,
                         'burn.in': chain_length * gibbs_control['burn.in'] / gibbs_control['chain.length'],
                         'thinning': gibbs_control['thinning']}),
                    chain_length=chain_length,
                    seed=gibbs_control['seed'],
                    rng_backend=rng_backend,
                    fast_multinomial=fast_mult,
                )
            if isinstance(ref, references.RefTumor):
                phi_1 = pd.concat([pd.DataFrame(ref.psi_mal.iloc[0, :]).T, ref.psi_env])
                nonzero_idx = np.max(phi_1, axis=0) > 0
                GibbsSamplerFast.sample_theta_n(
                    X_n=X[0, nonzero_idx],
                    phi=phi_1.loc[:, nonzero_idx],
                    alpha=gibbs_control['alpha'],
                    gibbs_idx=GibbsSamplerFast.get_gibbs_idx(
                        {'chain.length': chain_length,
                         'burn.in': chain_length*gibbs_control['burn.in'] / gibbs_control['chain.length'],
                         'thinning': gibbs_control['thinning']}),
                    chain_length=chain_length,
                    seed=gibbs_control['seed'],
                    rng_backend=rng_backend,
                    fast_multinomial=fast_mult,
                )

        total_time = time.process_time() - ptm
        estimated_time = gibbs_control['chain.length'] / chain_length * total_time * np.ceil(X.shape[0] / gibbs_control['n.cores']) * 2
        current_time = datetime.now()
        print("Current time: ", current_time)
        print("Estimated time to complete: ", GibbsSamplerFast.my_seconds_to_period(estimated_time))
        print("Estimated finishing time: ", current_time + timedelta(seconds=estimated_time))

    def run_gibbs_refPhi(self, final, compute_elbo):
        """VERBATIM copy (fork-Pool per-sample parallelism, order-preserving
        starmap, chunksize formula, ``SeedSequence(seed).spawn(n_samples)``
        child-stream assignment) — only the JointPost class is this fork's."""
        references = _bp_mod("references")
        assert isinstance(self.reference, references.RefPhi)
        phi = self.reference.phi
        X = self.X.to_numpy()
        gibbs_control = self.gibbs_control
        alpha = gibbs_control['alpha']
        fast_mult = gibbs_control.get('fast.multinomial', False)
        rng_backend = gibbs_control.get('rng.backend', 'generator')
        gibbs_idx = GibbsSamplerFast.get_gibbs_idx(gibbs_control)
        chain_length = gibbs_control['chain.length']
        seed = gibbs_control['seed']
        print("Start run...")

        ctx = multiprocessing.get_context("fork")
        chunk_size = max(1, int(np.ceil(X.shape[0] / (gibbs_control['n.cores'] * 4))))
        if not final:
            seeds = GibbsSamplerFast._spawn_seeds(seed, X.shape[0], backend=rng_backend)
            with ctx.Pool(processes=gibbs_control['n.cores']) as pool:
                X_input = [X[i, :] for i in np.arange(X.shape[0])]
                star_input = zip(
                    X_input,
                    repeat(phi),
                    repeat(alpha),
                    repeat(gibbs_idx),
                    repeat(chain_length),
                    seeds,
                    repeat(rng_backend),
                    repeat(compute_elbo),
                    repeat(fast_mult),
                )
                gibbs_list = pool.starmap(GibbsSamplerFast.sample_Z_theta_n, star_input, chunksize=chunk_size)
            return JointPostFast.new(self.X.index, self.X.columns, phi.index, gibbs_list)
        else:
            seeds = GibbsSamplerFast._spawn_seeds(seed, X.shape[0], backend=rng_backend)
            with ctx.Pool(processes=gibbs_control['n.cores']) as pool:
                X_input = [X[i, :] for i in np.arange(X.shape[0])]
                star_input = zip(
                    X_input,
                    repeat(phi),
                    repeat(alpha),
                    repeat(gibbs_idx),
                    repeat(chain_length),
                    seeds,
                    repeat(rng_backend),
                    repeat(fast_mult),
                )
                gibbs_list = pool.starmap(GibbsSamplerFast.sample_theta_n, star_input, chunksize=chunk_size)
            theta_post = _bp_mod("theta_post")
            return theta_post.ThetaPost.new(self.X.index, self.X.columns, gibbs_list)

    def run_gibbs_refTumor(self):
        """VERBATIM copy INCLUDING the upstream quirk that every sample's
        child seed is the SAME ``SeedSequence(seed).spawn(1)[0]`` object
        (re-spawned inside the per-sample loop, gibbs.py ~L393)."""
        references = _bp_mod("references")
        assert isinstance(self.reference, references.RefTumor)
        psi_mal = self.reference.psi_mal
        psi_env = self.reference.psi_env
        key = self.reference.key
        X = self.X.to_numpy()
        gibbs_control = self.gibbs_control
        alpha = gibbs_control['alpha']
        fast_mult = gibbs_control.get('fast.multinomial', False)
        rng_backend = gibbs_control.get('rng.backend', 'generator')
        gibbs_idx = GibbsSamplerFast.get_gibbs_idx(gibbs_control)
        chain_length = gibbs_control['chain.length']
        seed = gibbs_control['seed']
        print("Start run...")

        star_input = []
        for i in range(X.shape[0]):
            psi_mal_n = pd.DataFrame(psi_mal.iloc[i, :]).T
            phi_n = pd.concat([psi_mal_n, psi_env])
            nonzero_idx = np.max(phi_n, axis=0) > 0
            child_seed = GibbsSamplerFast._spawn_seeds(seed, 1, backend=rng_backend)[0]
            star_input.append((
                X[i, nonzero_idx],
                phi_n.loc[:, nonzero_idx],
                alpha,
                gibbs_idx,
                chain_length,
                child_seed,
                rng_backend,
                fast_mult,
            ))

        ctx = multiprocessing.get_context("fork")
        chunk_size = max(1, int(np.ceil(len(star_input) / (gibbs_control['n.cores'] * 4))))
        with ctx.Pool(processes=gibbs_control['n.cores']) as pool:
            gibbs_list = pool.starmap(GibbsSamplerFast.sample_theta_n, star_input, chunksize=chunk_size)

        theta_post = _bp_mod("theta_post")
        return theta_post.ThetaPost.new(self.X.index, [key] + list(psi_env.index), gibbs_list)

    def run(self, final, if_estimate=True, compute_elbo=False):
        references = _bp_mod("references")
        if final:
            print("Run Gibbs sampling using updated reference ...")
        else:
            print("Run Gibbs sampling...")

        if if_estimate:
            self.estimate_gibbs_time(final=final)
        if isinstance(self.reference, references.RefPhi):
            return GibbsSamplerFast.run_gibbs_refPhi(self, final=final, compute_elbo=compute_elbo)
        if isinstance(self.reference, references.RefTumor):
            return GibbsSamplerFast.run_gibbs_refTumor(self)


# ---------------------------------------------------------------------------
# joint_post fork — merge_K vectorized, sums bit-identical (see docstring)
# ---------------------------------------------------------------------------
class JointPostFast:
    """Fork of iobrpy.bayesprism.joint_post.JointPost (``new`` verbatim;
    ``merge_K`` vectorized with bit-identical per-type sums — the addend
    ORDER is the ORIGINAL ``map_[type]`` state-list order, and numpy
    pairwise summation over equal-length contiguous subsets is
    layout-deterministic)."""

    def __init__(self, Z, theta, theta_cv=None, constant=None):
        self.Z = Z
        self.theta = theta
        self.theta_cv = theta_cv
        self.constant = constant

    def new(bulk_id, gene_id, cell_type, gibbs_list):
        """VERBATIM copy of JointPost.new."""
        import xarray as xr

        n = len(bulk_id)
        h = len(gene_id)
        k = len(cell_type)
        assert len(gibbs_list) == n

        Z = np.zeros((n, h, k))
        theta = np.zeros((n, k))
        theta_cv = np.zeros((n, k))

        for i in range(n):
            Z[i, :, :] = gibbs_list[i]['Z_n']
            theta[i, :] = gibbs_list[i]['theta_n']

        if 'theta.cv_n' in gibbs_list[0] and gibbs_list[0]['theta.cv_n'] is not None:
            for i in range(n):
                theta_cv[i, :] = gibbs_list[i]['theta.cv_n']

        constant = sum(subdict["gibbs.constant"] for subdict in gibbs_list)

        Z = xr.DataArray(Z, coords=[bulk_id, gene_id, cell_type],
                         dims=['bulk_id', 'gene_id', 'cell_type'])
        theta = pd.DataFrame(theta, index=bulk_id, columns=cell_type)
        theta_cv = pd.DataFrame(theta_cv, index=bulk_id, columns=cell_type)

        return JointPostFast(Z, theta, theta_cv, constant)

    def merge_K(self, map_: dict):
        bulk_id = self.Z.coords['bulk_id'].values
        gene_id = self.Z.coords['gene_id'].values
        cell_type = self.Z.coords['cell_type'].values
        cell_type_merged = list(map_.keys())

        n = len(bulk_id)
        g = len(gene_id)
        k = len(cell_type)
        k_merged = len(cell_type_merged)

        assert sum([len(v) for v in map_.values()]) == k

        Zv = self.Z.values                      # (n, g, k), C-contiguous
        thetav = self.theta.to_numpy()          # (n, k)
        state_pos = {s: j for j, s in enumerate(cell_type)}

        Z_out = np.zeros((n, g, k_merged))
        theta_out = np.zeros((n, k_merged))

        for i in range(k_merged):
            cell_type_merged_k = cell_type_merged[i]
            cell_types_k = map_[cell_type_merged_k]
            idx = [state_pos[s] for s in cell_types_k]
            if len(cell_types_k) == 1:
                # np.squeeze quirk preserved verbatim (also squeezes n/g
                # singleton dims, exactly like the ORIGINAL).
                Z_out[:, :, i] = np.squeeze(Zv[:, :, idx])
                theta_out[:, i] = np.squeeze(thetav[:, idx])
            else:
                Z_out[:, :, i] = np.sum(Zv[:, :, idx], axis=2)
                theta_out[:, i] = np.sum(thetav[:, idx], axis=1)

        import xarray as xr
        Z = xr.DataArray(Z_out, coords=[('bulk_id', bulk_id), ('gene_id', gene_id), ('cell_type_merged', cell_type_merged)])
        theta = pd.DataFrame(theta_out, index=bulk_id, columns=cell_type_merged)

        # theta_cv intentionally dropped (upstream behaviour, preserved).
        return JointPostFast(Z, theta)


# ---------------------------------------------------------------------------
# prism fork — Prism.new/run with state_order; BayesPrism container REUSED
# ---------------------------------------------------------------------------
class PrismFast:
    """Fork of iobrpy.bayesprism.prism.Prism.

    ``new``/``run`` are verbatim copies (prints, warnings, quirks) except:
    the expressed-gene mask uses ``np.count_nonzero`` (identical mask), the
    preprocessing calls go to this module's vectorized ``process_input``
    fork, and ``new`` gains ``state_order`` ('sorted' = determinism fix,
    'legacy' = the ORIGINAL ``list(set(...))`` expression verbatim).
    The result container is the ORIGINAL ``prism.BayesPrism`` class so that
    ``extract.get_fraction``/``get_exp`` isinstance checks pass.
    """

    def __init__(self, phi_cellState, phi_cellType, map_, key, mixture):
        self.phi_cellState = phi_cellState
        self.phi_cellType = phi_cellType
        self.map = map_
        self.key = key
        self.mixture = mixture

    def valid_opt_control(control):
        """VERBATIM copy of Prism.valid_opt_control."""
        ctrl = {'maxit': 100000,
                'maximize': False,
                'trace': 0,
                'eps': 1e-07,
                'dowarn': True,
                'tol': 0,
                'maxNA': 500,
                'n.cores': 1,
                'optimizer': 'MAP',
                'sigma': 2}

        namc = list(control.keys())

        for name in namc:
            if name in ctrl.keys():
                ctrl[name] = control[name]
            else:
                raise ValueError("Unknown names in opt.control: {}".format(name))

        if ctrl['optimizer'] not in ["MAP", "MLE"]:
            raise ValueError("unknown names of optimizer: " + ctrl['optimizer'])

        if ctrl['optimizer'] == "MAP":
            if not isinstance(ctrl['sigma'], (int, float)):
                raise ValueError("sigma needs to be a numeric variable")
            else:
                if ctrl['sigma'] < 0:
                    raise ValueError("sigma needs to be positive")
        return ctrl

    def valid_gibbs_control(control):
        """VERBATIM copy of Prism.valid_gibbs_control (defaults: chain.length
        1000, burn.in 500, thinning 2, seed 123, alpha 1,
        fast.multinomial False, rng.backend 'generator')."""
        ctrl = {'chain.length': 1000,
                'burn.in': 500,
                'thinning': 2,
                'n.cores': 1,
                'seed': 123,
                'alpha': 1,
                'fast.multinomial': False,
                'rng.backend': 'generator'}

        namc = list(control.keys())

        for name in namc:
            if name in ctrl.keys():
                ctrl[name] = control[name]
            else:
                raise ValueError("Unknown names in opt.control: {}".format(name))

        if ctrl['alpha'] < 0:
            raise ValueError("alpha needs to be positive")

        if not isinstance(ctrl['fast.multinomial'], bool):
            raise ValueError("fast.multinomial must be a boolean flag")

        if ctrl['rng.backend'].lower() not in ['generator', 'randomstate']:
            raise ValueError("rng.backend must be either 'generator' or 'randomstate'")

        ctrl['rng.backend'] = ctrl['rng.backend'].lower()

        return ctrl

    def new(reference, input_type, cell_type_labels, cell_state_labels,
            key, mixture, outlier_cut=0.01, outlier_fraction=0.1, pseudo_min=1E-8,
            state_order="sorted"):
        """VERBATIM copy of Prism.new except the two documented changes
        (expressed_mask via count_nonzero; map_ via ``state_order``)."""
        import warnings

        if cell_state_labels is None:
            cell_state_labels = cell_type_labels

        print("number of cells in each cell state")
        print(pd.Series(cell_state_labels).value_counts().sort_values(ascending=False))
        if np.min(pd.Series(cell_state_labels).value_counts()) < 20:
            print("recommend to have sufficient number of cells in each cell state")

        if key is None:
            print("No tumor reference is speficied. Reference cell types are treated equally.")

        if len(cell_type_labels) != len(cell_state_labels):
            raise ValueError("Error: length of cell.type.labels and cell.state.labels do not match!")
        if len(cell_type_labels) != reference.shape[0]:
            raise ValueError("Error: length of cell.type.labels and nrow(reference) do not match!")

        type_to_state_mat = pd.DataFrame({"cell.type.labels": cell_type_labels, "cell.state.labels": cell_state_labels})
        type_to_state_mat = type_to_state_mat.drop_duplicates()
        if type_to_state_mat["cell.state.labels"].value_counts().max() > 1:
            raise ValueError("Error: one or more cell states belong to multiple cell types!")
        if len(pd.unique(cell_type_labels)) > len(pd.unique(cell_state_labels)):
            raise ValueError("Error: more cell types than states!")

        if not isinstance(mixture, pd.DataFrame):
            mixture = pd.DataFrame(mixture)
        if not isinstance(reference, pd.DataFrame):
            reference = pd.DataFrame(reference)

        if mixture.shape[1] == 1:
            mixture = mixture.T
            mixture = mixture.rename(index={0: ["mixture-1"]})

        if mixture.index.equals(pd.RangeIndex(mixture.shape[0])):
            mixture.index = [f"mixture-{i}" for i in range(len(mixture))]

        validate_input(reference)
        validate_input(mixture)

        input_type = input_type.lower()
        if input_type == "count.matrix":
            # Keep genes expressed in at least one cell to mirror R BayesPrism's
            # count-matrix filtering behaviour.
            # (FAST: np.count_nonzero on the numpy array instead of
            #  np.sum(df > 0, axis=0) — identical mask, no pandas boolean
            #  frame materialization.)
            expressed_mask = np.count_nonzero(reference.to_numpy() > 0, axis=0) > 0
            if expressed_mask.sum() < reference.shape[1]:
                print(f"Removing {reference.shape[1] - expressed_mask.sum()} genes not expressed in any cell")
            reference = reference.loc[:, expressed_mask]
            pseudo_min_to_use = pseudo_min
        elif input_type in ["tpm", "gep"]:
            # Collapsed/normalized profiles (e.g., TPM) only support removing
            # genes with zero abundance across all cell states/types and do not
            # add pseudo count during normalization.
            pseudo_min_to_use = 0
            if pseudo_min != 0:
                warnings.warn(
                    "pseudo_min is ignored for normalized input types (TPM/GEP); using 0 instead."
                )
            reference = reference.loc[:, np.sum(reference, axis=0) > 0]
        else:
            raise ValueError(
                "Unsupported input_type. Please specify one of ['count.matrix', 'TPM', 'GEP']."
            )

        mixture = filter_bulk_outlier(mixture, outlier_cut, outlier_fraction)

        _, gene_index, _ = np.intersect1d(reference.columns, mixture.columns, return_indices=True)
        gene_index.sort()
        gene_shared = [list(reference.columns)[i] for i in gene_index]
        if len(gene_shared) == 0:
            raise ValueError("Error: gene names of reference and mixture do not match!")
        if len(gene_shared) < 100:
            print("Warning: very few gene from reference and mixture match! Please double check your gene names.")

        ref_cs = collapse(reference, cell_state_labels)
        ref_ct = collapse(reference, cell_type_labels)

        print("Aligning reference and mixture...")
        ref_ct = ref_ct.loc[:, gene_shared]
        ref_cs = ref_cs.loc[:, gene_shared]
        mixture = mixture.loc[:, gene_shared]

        print("Normalizing reference...")
        ref_cs = norm_to_one(ref_cs, pseudo_min_to_use)
        ref_ct = norm_to_one(ref_ct, pseudo_min_to_use)

        if state_order == "sorted":
            # DETERMINISM FIX: sorted() instead of the ORIGINAL's
            # list(set(...)) whose string iteration order follows the
            # per-process PYTHONHASHSEED (see module docstring).
            map_ = {cell_type: sorted(set(cell_state_labels[i] for i, ct in enumerate(cell_type_labels) if ct == cell_type)) for cell_type in ref_ct.index}
        elif state_order == "legacy":
            # ORIGINAL expression, verbatim (hash-seed dependent order).
            map_ = {cell_type: list(set(cell_state_labels[i] for i, ct in enumerate(cell_type_labels) if ct == cell_type)) for cell_type in ref_ct.index}
        else:
            raise ValueError("state_order must be 'sorted' or 'legacy'")

        rf = _bp_mod("references")
        return PrismFast(
            rf.RefPhi(ref_cs, pseudo_min_to_use),
            rf.RefPhi(ref_ct, pseudo_min_to_use),
            map_,
            key,
            mixture
        )

    def run(self, n_cores=1, update_gibbs=True, gibbs_control={}, opt_control={}):
        """VERBATIM copy of Prism.run (mutable-default-dict quirk included)
        using GibbsSamplerFast / JointPostFast.merge_K and the ORIGINAL
        optim.update_reference + prism.BayesPrism."""
        _prism = _bp_mod("prism")
        optim = _bp_mod("optim")

        if 'n.cores' not in gibbs_control:
            gibbs_control['n.cores'] = n_cores
        if 'n.cores' not in opt_control:
            opt_control['n.cores'] = n_cores

        assert isinstance(update_gibbs, bool)
        assert isinstance(n_cores, int)

        opt_control = PrismFast.valid_opt_control(opt_control)
        gibbs_control = PrismFast.valid_gibbs_control(gibbs_control)

        if self.phi_cellState.pseudo_min == 0:
            gibbs_control['alpha'] = max(1, gibbs_control['alpha'])

        gibbsSampler_ini_cs = GibbsSamplerFast(reference=self.phi_cellState,
                                               X=self.mixture,
                                               gibbs_control=gibbs_control)

        jointPost_ini_cs = gibbsSampler_ini_cs.run(final=False)
        print("Now Merging...")
        jointPost_ini_ct = jointPost_ini_cs.merge_K(map_=self.map)

        if not update_gibbs:
            bp = _prism.BayesPrism(prism=self,
                                   posterior_initial_cellState=jointPost_ini_cs,
                                   posterior_initial_cellType=jointPost_ini_ct,
                                   control_param={'gibbs.control': gibbs_control,
                                                  'opt.control': opt_control,
                                                  'update.gibbs': update_gibbs})
            return bp
        else:
            psi = optim.update_reference(Z=jointPost_ini_ct.Z,
                                         phi_prime=self.phi_cellType,
                                         map=self.map,
                                         key=self.key,
                                         opt_control=opt_control)

            gibbsSampler_update = GibbsSamplerFast(reference=psi,
                                                   X=self.mixture,
                                                   gibbs_control=gibbs_control)

            theta_f = gibbsSampler_update.run(final=True)

            bp = _prism.BayesPrism(prism=self,
                                   posterior_initial_cellState=jointPost_ini_cs,
                                   posterior_initial_cellType=jointPost_ini_ct,
                                   control_param={'gibbs.control': gibbs_control,
                                                  'opt.control': opt_control,
                                                  'update.gibbs': update_gibbs},
                                   reference_update=psi,
                                   posterior_theta_f=theta_f)
            return bp


# ---------------------------------------------------------------------------
# I/O layer — ORIGINAL run_bayesprism contract + csv-first caching
# ---------------------------------------------------------------------------
def _read_user_csv(path: Path, **read_kwargs) -> pd.DataFrame:
    """VERBATIM copy of bayesprism._read_user_csv (sep by suffix, gz)."""
    suffixes = path.suffixes
    sep = "," if ".csv" in suffixes else "\t"
    compression = "gzip" if ".gz" in suffixes else None

    return pd.read_csv(path, sep=sep, compression=compression, **read_kwargs)


def _cache_dir() -> Optional[Path]:
    env = os.environ.get("IOBRX_CACHE")
    if env:
        base = Path(env)
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "iobrx"
    d = base / "bayesprism"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return d


def _read_sc_dat_file(path) -> pd.DataFrame:
    """Read a cells x genes single-cell count CSV with the ORIGINAL
    ``pd.read_csv`` C-parser float semantics (first read is ALWAYS csv —
    no parquet/float-parsing shortcut), ``.astype(np.int32)`` like the
    ORIGINAL, then memoized per process and pickle-cached on disk keyed by
    (resolved path, mtime_ns, size, sep, compression, pandas/numpy version).
    Corrupt or unwritable caches silently fall back to the csv parse."""
    path = Path(path)
    suffixes = path.suffixes
    sep = "," if ".csv" in suffixes else "\t"
    compression = "gzip" if ".gz" in suffixes else None

    try:
        resolved = path.resolve()
        st = resolved.stat()
        tag_src = (f"{resolved}|{st.st_mtime_ns}|{st.st_size}|{sep}|{compression}|"
                   f"{pd.__version__}|{np.__version__}")
        tag = hashlib.sha256(tag_src.encode()).hexdigest()[:32]
    except OSError:
        resolved, tag = None, None

    if resolved is not None and str(resolved) in _BUNDLED:
        return _BUNDLED[str(resolved)]

    cache_file = None
    if tag is not None:
        cdir = _cache_dir()
        if cdir is not None:
            cache_file = cdir / f"sc_dat_{tag}.pkl"
            if cache_file.exists():
                try:
                    df = pd.read_pickle(cache_file)
                    if isinstance(df, pd.DataFrame):
                        _BUNDLED[str(resolved)] = df
                        return df
                except Exception:
                    pass  # corrupt cache -> reparse from csv

    df = pd.read_csv(path, sep=sep, compression=compression, header=0, index_col=0).astype(np.int32)

    if cache_file is not None:
        try:
            df.to_pickle(cache_file)
        except Exception:
            pass  # unwritable cache is not an error
    if resolved is not None:
        _BUNDLED[str(resolved)] = df
    return df


def _load_bundled(name: str, kind: str):
    """Bundled BP_data loaders (kind: 'sc_dat' | 'labels'); memoized."""
    memo_key = f"bundled:{name}"
    if memo_key in _BUNDLED:
        return _BUNDLED[memo_key]
    path = _bp_data_path(name)
    if kind == "sc_dat":
        out = _read_sc_dat_file(path)
    else:
        # ORIGINAL: pd.read_csv(path, sep=",", header=None).iloc[:, 0]
        out = list(pd.read_csv(path, sep=",", header=None).iloc[:, 0])
    _BUNDLED[memo_key] = out
    return out


def _resolve_sc_dat(sc_dat) -> pd.DataFrame:
    if sc_dat is None:
        return _load_bundled("sc_dat.csv", "sc_dat")
    if isinstance(sc_dat, pd.DataFrame):
        return sc_dat.astype(np.int32)
    return _read_sc_dat_file(Path(sc_dat))


def _resolve_labels(labels, bundled_name: str) -> list:
    if labels is None:
        return _load_bundled(bundled_name, "labels")
    if isinstance(labels, (str, Path, os.PathLike)):
        # ORIGINAL: _read_user_csv(path, header=None).iloc[:, 0]
        return list(_read_user_csv(Path(labels), header=None).iloc[:, 0])
    return list(labels)


def _resolve_bulk(bulk) -> pd.DataFrame:
    """Bulk genes x samples frame; path inputs are read EXACTLY like the
    ORIGINAL CLI (sep by suffix, gzip, index_col=0)."""
    if isinstance(bulk, pd.DataFrame):
        return bulk
    path = Path(bulk)
    suffixes = path.suffixes
    compression = "gzip" if ".gz" in suffixes else None
    sep = "," if ".csv" in suffixes else "\t"
    return pd.read_csv(path, sep=sep, index_col=0, compression=compression)


def _pipeline_frames(bulk_df, sc_dat_df, cell_type_labels, cell_state_labels,
                     key, threads, outlier_cut, outlier_fraction, pseudo_min,
                     gibbs_control, opt_control, state_order=None, original=False):
    """Shared pipeline body: ORIGINAL run_bayesprism steps 2-5.

    ``original=True`` routes every stage through the UNTOUCHED iobrpy code
    (backend='python'); otherwise the forked fast stages run and
    ``state_order`` selects the map_ construction."""
    if original:
        pi = _bp_mod("process_input")
        prism_mod = _bp_mod("prism")
        sc_dat_filtered = pi.cleanup_genes(
            sc_dat_df,
            input_type="count.matrix",
            species=_SPECIES,
            gene_group=_GENE_GROUP,
            exp_cells=_EXP_CELLS,
        )
        sc_dat_filtered_pc = pi.select_gene_type(sc_dat_filtered, ["protein_coding"])
        mixture = bulk_df.T.astype(np.int32)
        my_prism = prism_mod.Prism.new(
            reference=sc_dat_filtered_pc,
            input_type="count.matrix",
            cell_type_labels=cell_type_labels,
            cell_state_labels=cell_state_labels,
            key=key,
            mixture=mixture,
            outlier_cut=outlier_cut,
            outlier_fraction=outlier_fraction,
            pseudo_min=pseudo_min,
        )
        bp_res = my_prism.run(n_cores=threads, update_gibbs=True,
                              gibbs_control=dict(gibbs_control), opt_control=dict(opt_control))
    else:
        sc_dat_filtered = cleanup_genes(
            sc_dat_df,
            input_type="count.matrix",
            species=_SPECIES,
            gene_group=_GENE_GROUP,
            exp_cells=_EXP_CELLS,
        )
        sc_dat_filtered_pc = select_gene_type(sc_dat_filtered, ["protein_coding"])
        mixture = bulk_df.T.astype(np.int32)
        my_prism = PrismFast.new(
            reference=sc_dat_filtered_pc,
            input_type="count.matrix",
            cell_type_labels=cell_type_labels,
            cell_state_labels=cell_state_labels,
            key=key,
            mixture=mixture,
            outlier_cut=outlier_cut,
            outlier_fraction=outlier_fraction,
            pseudo_min=pseudo_min,
            state_order=state_order,
        )
        bp_res = my_prism.run(n_cores=threads, update_gibbs=True,
                              gibbs_control=dict(gibbs_control), opt_control=dict(opt_control))

    extract = _bp_mod("extract")
    theta = extract.get_fraction(bp_res, which_theta="final", state_or_type="type")
    theta_cv = bp_res.posterior_theta_f.theta_cv
    Z_tumor = extract.get_exp(bp_res, state_or_type="type", cell_name=key)

    # ORIGINAL output contract (bayesprism.py step 6).
    theta_with_suffix = theta.add_suffix("_BayesPrism")
    theta_cv_with_suffix = theta_cv.add_suffix("_BayesPrism")
    if hasattr(Z_tumor, "to_pandas"):
        Z_tumor_df = Z_tumor.to_pandas()
    else:
        Z_tumor_df = pd.DataFrame(Z_tumor)

    return {"theta": theta_with_suffix, "theta_cv": theta_cv_with_suffix, "Z_tumor": Z_tumor_df}


def _write_outputs(res: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res["theta"].to_csv(out_dir / "theta.csv")
    res["theta_cv"].to_csv(out_dir / "theta_cv.csv")
    res["Z_tumor"].to_csv(out_dir / "Z_tumor.csv")


def run_bayesprism_fast(bulk, out_dir=None, n_cores=8, sc_dat=None,
                        cell_state_labels=None, cell_type_labels=None,
                        key=_DEFAULT_KEY, state_order="sorted",
                        gibbs_control=None, opt_control=None,
                        outlier_cut=0.01, outlier_fraction=0.1, pseudo_min=1e-8):
    """Accelerated equivalent of iobrpy.bayesprism.bayesprism.run_bayesprism
    with the same file-output contract; returns the three output frames
    (``theta`` / ``theta_cv`` / ``Z_tumor``) instead of ``None``.

    ``bulk``/``sc_dat`` accept paths (read with ORIGINAL csv semantics) or
    DataFrames (bulk: genes x samples; sc_dat: CELLS x genes — the code
    semantics; the upstream CLI help text has this backwards).
    """
    from iobrx._threads import resolve_threads

    threads = resolve_threads(n_cores)
    bulk_df = _resolve_bulk(bulk)
    sc_dat_df = _resolve_sc_dat(sc_dat)
    ctl = _resolve_labels(cell_type_labels, "cell_type_labels.csv")
    csl = _resolve_labels(cell_state_labels, "cell_state_labels.csv")

    res = _pipeline_frames(
        bulk_df, sc_dat_df, ctl, csl, key, threads,
        outlier_cut, outlier_fraction, pseudo_min,
        dict(gibbs_control or {}), dict(opt_control or {}),
        state_order=state_order, original=False,
    )
    if out_dir is not None:
        _write_outputs(res, out_dir)
    return res


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def bayesprism(
    bulk,
    sc_dat=None,
    cell_state_labels=None,
    cell_type_labels=None,
    key: str = _DEFAULT_KEY,
    out_dir=None,
    n_threads: Optional[int] = None,
    backend: str = "auto",
    state_order: str = "sorted",
    outlier_cut: float = 0.01,
    outlier_fraction: float = 0.1,
    pseudo_min: float = 1e-8,
    gibbs_control: Optional[dict] = None,
    opt_control: Optional[dict] = None,
) -> dict:
    """BayesPrism deconvolution, accelerated (pure-Python lane).

    Bit-exact drop-in for ``iobrpy.bayesprism.bayesprism.run_bayesprism``
    (the ``python -m iobrpy.main bayesprism`` CLI) on frozen inputs: with
    ``state_order='legacy'`` under the same PYTHONHASHSEED the three output
    CSVs are byte-identical to the ORIGINAL (verified SHA-256 against the
    PYTHONHASHSEED=0 gold-standard run); with the default
    ``state_order='sorted'`` the pipeline is deterministic ACROSS PROCESSES
    without pinning PYTHONHASHSEED (the ORIGINAL is not — its
    ``list(set(states))`` merge order follows the per-process hash seed,
    which occasionally flips discrete Gibbs draws; see the module
    docstring). The Gibbs sampler call sequence (per-gene sequential
    ``rng.multinomial``, ``rng.gamma`` Dirichlet, ``SeedSequence(123)
    .spawn(n_samples)`` child streams, the phase-3 shared ``spawn(1)[0]``
    seed quirk, fork-Pool per-sample parallelism) is preserved VERBATIM, so
    with equal merge order the samples are bit-identical.

    Parameters
    ----------
    bulk : pandas.DataFrame or path-like
        Bulk expression matrix, GENES (index) x SAMPLES (columns) — the same
        orientation as the CLI's bulk csv. Paths are parsed exactly like the
        ORIGINAL (sep by suffix, ``.gz`` support, ``index_col=0``) and the
        frame is transposed + ``astype(np.int32)`` internally, as upstream.
    sc_dat : pandas.DataFrame or path-like, optional
        Single-cell reference count matrix, CELLS (index) x GENES (columns)
        — the upstream CLI help says "genes x cells" but the CODE requires
        cells x genes (rows must align with the label files). ``None`` uses
        the bundled ``iobrpy.bayesprism/BP_data/sc_dat.csv`` (parsed with
        ORIGINAL csv float semantics on first read, then process-memoized
        and pickle-cached on disk).
    cell_state_labels, cell_type_labels : list-like or path-like, optional
        One label per reference cell (row order of ``sc_dat``). Paths are
        read header-less like the ORIGINAL; ``None`` uses the bundled
        BP_data label files.
    key : str, default 'Malignant_cells'
        Tumor cell type (as the CLI ``--key``; required semantics when a
        custom ``sc_dat`` is given — the key must exist in
        ``cell_type_labels``).
    out_dir : path-like, optional
        When given, write ``theta.csv`` / ``theta_cv.csv`` / ``Z_tumor.csv``
        into it with the ORIGINAL output contract (``_BayesPrism`` column
        suffixes; ``Z_tumor.csv`` from ``xarray.to_pandas``).
    n_threads : int or None, optional
        Worker processes for the per-sample Gibbs pools and the per-cell-type
        MAP optimization. ``None`` resolves to ``min(8, os.cpu_count())``
        (see :func:`iobrx.set_threads`). The worker count does NOT affect
        output bits (seeds are assigned per sample up front; ``starmap``
        preserves order).
    backend : {'auto', 'rust', 'python'}, default 'auto'
        'python' runs the UNTOUCHED ORIGINAL iobrpy stages (including its
        hash-seed-dependent ``list(set(...))`` merge order). 'auto'/'rust'
        both take this accelerated pure-Python path — a native Gibbs kernel
        is a follow-up round; ``select_backend`` still validates the value.
    state_order : {'sorted', 'legacy'}, default 'sorted'
        Order of the cell-state lists used by ``merge_K`` / the MAP update.
        'sorted' fixes the upstream PYTHONHASHSEED nondeterminism (deterministic
        across processes); 'legacy' reproduces the ORIGINAL
        ``list(set(states))`` expression verbatim (byte-identical to an
        ORIGINAL run only within the same PYTHONHASHSEED).
    outlier_cut, outlier_fraction : float, default 0.01, 0.1
        Bulk outlier-gene filter (as the ORIGINAL ``Prism.new`` defaults).
    pseudo_min : float, default 1e-8
        Reference normalization pseudo-count (as the ORIGINAL default).
    gibbs_control, opt_control : dict, optional
        Passthrough to ``Prism.run`` (e.g. ``{'chain.length': 100,
        'burn.in': 50, 'thinning': 2}``). The CLI does not expose these;
        defaults are the upstream ones (chain 1000 / burn-in 500 / thinning 2
        / seed 123 / generator backend / sequential multinomial; MAP
        optimizer / sigma 2). ``n.cores`` is filled from ``n_threads``.

    Returns
    -------
    dict
        ``{'theta': DataFrame, 'theta_cv': DataFrame, 'Z_tumor': DataFrame}``
        — the exact frames written to the output CSVs: theta/theta_cv are
        samples x cell types with ``_BayesPrism``-suffixed columns;
        Z_tumor is samples x genes (index name ``bulk_id``), the INITIAL
        (pre-reference-update) tumor deconvolution, as upstream.
    """
    from iobrx._backend import select_backend
    from iobrx._threads import resolve_threads

    use_fast = select_backend(backend)  # validates backend; no native bayesprism kernel yet
    if state_order not in ("sorted", "legacy"):
        raise ValueError("state_order must be 'sorted' or 'legacy'")
    threads = resolve_threads(n_threads)

    bulk_df = _resolve_bulk(bulk)
    sc_dat_df = _resolve_sc_dat(sc_dat)
    ctl = _resolve_labels(cell_type_labels, "cell_type_labels.csv")
    csl = _resolve_labels(cell_state_labels, "cell_state_labels.csv")

    res = _pipeline_frames(
        bulk_df, sc_dat_df, ctl, csl, key, threads,
        outlier_cut, outlier_fraction, pseudo_min,
        dict(gibbs_control or {}), dict(opt_control or {}),
        state_order=state_order, original=not use_fast,
    )
    if out_dir is not None:
        _write_outputs(res, out_dir)
    return res
