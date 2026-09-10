"""ssgsea_fast: bit-exact, Rust re-implementation of IOBRpy's
iobrpy.workflow.calculate_sig_score.sig_score_ssgsea() (which calls
gseapy 1.3.0 gp.ssgsea(..., sample_norm_method='rank', permutation_num=0,
ssgsea_norm=True) and pivots res2d).

Bit-exactness strategy (all mirrored from the installed gseapy 1.3.0 sources,
site-packages/gseapy/{__init__,ssgsea,base}.py + the vendored Rust core):
  - Python side replicates SingleSampleGSEA.load_data() (_load_data +
    _check_data: reset_index / set_index / dup-gene groupby-mean / inf
    replacement) and GSEAbase.load_gmt() dict filtering (set() dedup,
    min_size <= tag_len <= max_size, tag_len < n_genes, uppercase matching
    via check_uppercase) exactly, including the no-op dropna branch.
  - Rust iobrx_rust.ssgsea_core replicates stats.rs ss_gsea() for nperm=0:
    per-sample argsort(false) (stable ascending sort then reverse == gseapy
    utils.rs), pandas rank(method='average') tie semantics scaled by
    10000*rank/G, |x|^0.25 weighting, algorithm.rs fast_random_walk_ss
    closed-form ES with sequential f64 accumulation in ascending hit order,
    and the global min-max normalization nes = es / (max_es - min_es).
  - The res2d -> pivot(Term x Name NES).T.reset_index() dance is reproduced
    through the actual pandas pivot call, so column order / columns.name /
    dtypes are identical to the original by construction.
  - Output is thread-count invariant: every reduction is sequential per
    (sample, set) and the global max/min fold is order-independent.

Parameters that IOBRpy hardcodes for this path: weight=0.25 (gp.ssgsea
default), max_size=500 (default), correl_norm_type='rank' -> CorrelType::Rank
(no transform). `ssgsea_norm=True` is swallowed by gseapy's **kwargs and has
no effect (verified in gseapy 1.3.0 sources).
"""
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

import iobrx._rust as _ir

# upstream IOBRpy workflow helpers (read-only import; identical semantics)
try:
    from iobrpy.workflow.calculate_sig_score import (  # noqa: E402
        preprocess_eset,
        filter_signatures,
    )
except ModuleNotFoundError:
    preprocess_eset = filter_signatures = None


def _require_upstream():
    if preprocess_eset is None:
        raise ImportError(
            "signature scoring reuses upstream IOBRpy helpers; install the Python "
            "fallback backend with `pip install 'iobrx[python]'`."
        )

# sample_norm enum values understood by iobrx_rust.ssgsea_core
_SAMPLE_NORM = {None: 0, "custom": 0, "rank": 1, "log_rank": 2, "log": 3}
_CORREL_TYPE = {None: 0, "rank": 0, "symrank": 1, "zscore": 2}


# ------------------------------------------------------------------
# gseapy SingleSampleGSEA.load_data(): _load_data + _check_data, literal port
# ------------------------------------------------------------------
def _gseapy_reset_index(rank_metric: pd.DataFrame) -> pd.DataFrame:
    """gseapy GSEAbase._reset_index (pandas 3.0-compat variant, 1.3.0)."""
    if is_object_dtype(rank_metric.index.dtype) or is_string_dtype(rank_metric.index.dtype):
        try:
            pd.to_numeric(rank_metric.index)
        except (ValueError, TypeError):
            # non-numeric strings -> likely gene names -> moved to first column
            rank_metric = rank_metric.reset_index()
    return rank_metric


def _gseapy_check_data(exprs: pd.DataFrame) -> pd.DataFrame:
    """gseapy GSEAbase._check_data, literal port (bugs included)."""
    if exprs.iloc[:, 0].isnull().any():
        exprs.dropna(subset=[exprs.columns[0]])  # noqa: B018 -- result discarded in gseapy too
    if exprs.isnull().any().sum() > 0:
        exprs.dropna(how="all", inplace=True)
        exprs = exprs.fillna(0)
    exprs.set_index(keys=exprs.columns[0], inplace=True)
    df = exprs.select_dtypes(include=[np.number])
    if df.index.duplicated().sum() > 0:
        df = df.groupby(level=0).mean()
    if np.isinf(df).values.sum() > 0:
        col_min_max = {
            np.inf: df[np.isfinite(df)].max(),
            -np.inf: df[np.isfinite(df)].min(),
        }
        df = df.replace({col: col_min_max for col in df.columns})
    return df


def gseapy_load_data(eset2: pd.DataFrame) -> pd.DataFrame:
    """SingleSampleGSEA.load_data() over an already-preprocessed eset2."""
    exprs = _gseapy_reset_index(eset2.copy())
    return _gseapy_check_data(exprs)


# ------------------------------------------------------------------
# gseapy GSEAbase.check_uppercase + load_gmt dict-input path, literal port
# ------------------------------------------------------------------
def _is_entrez_id(idx) -> bool:
    try:
        int(idx)
        return True
    except Exception:
        return False


def _check_uppercase(gene_list) -> bool:
    if all([_is_entrez_id(g) for g in gene_list]):
        return False
    is_upper = [str(s).isupper() for s in gene_list]
    if sum(is_upper) / len(is_upper) >= 0.9:
        return True
    return False


def gseapy_filter_gene_sets(gmt_dict, gene_list, min_size, max_size):
    """GSEAbase.load_gmt for a dict gene_sets input.

    Returns (filtered dict name -> [gene], gene_dict used for tagging,
    gene_names list handed to the Rust core).
    """
    subsets = list(gmt_dict.keys())
    if not subsets:
        raise ValueError("Empty gene sets dictionary")
    gene_isupper = _check_uppercase(list(gene_list))
    gene_dict = {g: i for i, g in enumerate(gene_list)}

    sample_size = min(20, len(subsets))
    ups = [_check_uppercase(gmt_dict[s]) for s in subsets[:sample_size]]
    to_upper = (not gene_isupper) and all(ups)
    if to_upper:
        gene_names = [str(g).upper() for g in gene_list]
        gene_dict = {g: i for i, g in enumerate(gene_names)}
    else:
        gene_names = list(gene_list)

    out = {}
    for subset in subsets:
        subset_list = set(gmt_dict[subset])  # remove duplicates
        gene_overlap = [g for g in subset_list if g in gene_dict]
        tag_len = len(gene_overlap)
        if (min_size <= tag_len <= max_size) and tag_len < len(gene_names):
            out[subset] = gene_overlap
        # else: deleted, like load_gmt
    if not out:
        raise ValueError(
            "No gene sets passed through filtering condition !!! \n"
            "Hint 1: Try to lower min_size or increase max_size !\n"
            "Hint 2: Check gene symbols are identifiable to your gmt input.\n"
            "Hint 3: Gene symbols curated in Enrichr web services are all upcases.\n"
        )
    return out, gene_dict, gene_names


# ------------------------------------------------------------------
# core stage: drop-in replacement for the gp.ssgsea() call + NES pivot
# ------------------------------------------------------------------
def ssgsea_fast(
    eset2: pd.DataFrame,
    sigs: dict,
    min_size: int = 15,
    max_size: int = 500,
    weight: float = 0.25,
    sample_norm_method="rank",
    correl_norm_type=None,
    threads: int = 1,
):
    """Run the ssGSEA core on a preprocessed eset2 (genes x samples).

    Equivalent of::

        gp.ssgsea(data=eset2, gene_sets=sigs, outdir=None,
                  sample_norm_method='rank', permutation_num=0, no_plot=True,
                  threads=threads, min_size=min_size, ssgsea_norm=True)

    Returns a res2d-equivalent DataFrame with columns Name, Term, ES, NES
    (rows re-indexed like GSEAbase.to_df: sorted by |NES| desc, RangeIndex).
    """
    _require_upstream()
    data = gseapy_load_data(eset2)
    filtered, gene_dict, gene_names = gseapy_filter_gene_sets(
        sigs, data.index.tolist(), min_size, max_size
    )
    expr = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
    gene_sets = [
        (name, np.asarray([gene_dict[g] for g in genes], dtype=np.int32).tolist())
        for name, genes in filtered.items()
    ]
    out = _ir.ssgsea_core(
        expr,
        gene_sets,
        weight=float(weight),
        min_size=int(min_size),
        max_size=int(max_size),
        sample_norm=_SAMPLE_NORM[sample_norm_method],
        correl_type=_CORREL_TYPE[correl_norm_type],
        n_threads=int(threads),
    )
    names = list(out["names"])
    es = np.asarray(out["es"]).reshape(len(names), data.shape[1])
    nes_t = np.asarray(out["nes"]).reshape(data.shape[1], len(names))

    samples = [str(c) for c in data.columns]
    # res2d-equivalent long form, then the exact IOBRpy pivot dance
    res2d = pd.DataFrame(
        {
            "Name": np.repeat(samples, len(names)),
            "Term": names * len(samples),
            "ES": es.T.reshape(-1),     # sample-major, aligned with Name
            "NES": nes_t.reshape(-1),   # [N x n_sigs] row-major
        }
    )
    # GSEAbase.to_df reorders rows by |NES| descending (cosmetic for the pivot,
    # reproduced anyway so res2d matches the original row-for-row)
    res2d = res2d.reindex(res2d["NES"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return res2d


def sig_score_ssgsea_fast(eset, sig_dict, mini_gene_count, adjust_eset, parallel_size):
    """IDENTICAL-output replacement for calculate_sig_score.sig_score_ssgsea."""
    _require_upstream()
    # Preprocess like R (upstream functions, unchanged semantics)
    eset2 = preprocess_eset(eset, adjust_eset)
    # First filter with original threshold
    sigs = filter_signatures(sig_dict, eset2, mini_gene_count)
    # Then enforce min_size >= 5
    min_size = max(mini_gene_count, 5)
    sigs = filter_signatures(sig_dict, eset2, min_size)

    res2d = ssgsea_fast(
        eset2,
        sigs,
        min_size=min_size,
        max_size=500,           # gp.ssgsea default, IOBRpy does not override
        weight=0.25,            # gp.ssgsea default for ssGSEA
        sample_norm_method="rank",
        correl_norm_type=None,  # gp.ssgsea default 'rank' -> CorrelType::Rank
        threads=parallel_size,
    )

    # Pivot to samples x terms, keep Term as columns (original statements)
    nes = res2d.pivot(index="Term", columns="Name", values="NES").T.reset_index()
    nes.rename(columns={"Name": "ID"}, inplace=True)

    if "TMEscoreA_CIR" in nes.columns and "TMEscoreB_CIR" in nes.columns:
        nes["TMEscore_CIR"] = nes["TMEscoreA_CIR"] - nes["TMEscoreB_CIR"]
    if "TMEscoreA_plus" in nes.columns and "TMEscoreB_plus" in nes.columns:
        nes["TMEscore_plus"] = nes["TMEscoreA_plus"] - nes["TMEscoreB_plus"]
    return nes
