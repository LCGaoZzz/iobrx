"""epic_fast: cached + vectorized drop-in for iobrpy.workflow.epic.EPIC().

Same signature as upstream EPIC(bulk, reference, ...). Bit-exactness strategy
(gated against iobrx/refs/epic.pkl on the official eset_stad_symbol x TRef
call; per-frame max-abs-diff == 0 required):

  1. Reference-side work is invariant across calls with the same `reference`
     object and the same bulk gene universe. It is memoized in two levels:
       - level 1 (keyed by id(reference), strong ref held): merged duplicate
         rows of refProfiles / refProfiles.var (upstream merge_duplicates).
       - level 2 (keyed by (id(reference), hash(tuple(common)))): everything
         derived from `common = bulk.index.intersection(refP.index)` --
         sig-gene list, norm_ref, ref_s, refV_s, weights w, A/Aw/sqrtw.
     The per-call bulk index only feeds in through `common`, whose full tuple
     hash is part of the key (Python str hashes are cached, so this is a
     ~0.5ms C loop, vs 10ms for hash_pandas_object).
  2. Bulk-side pandas ops are replaced by value-identical numpy on the
     float64 values matrix:
       - isna().all(axis=1)  -> np.isnan(vals).all(axis=1)          [float64 only]
       - counts.loc[common].sum(axis=0).replace(0,1) -> vals[pos].sum(axis=0)
         with np.where(==0, 1.0)   (verified: ndarray col-sum is bit-identical
         to DataFrame.sum(axis=0) with pandas' non-bottleneck nanops)
       - counts.loc[sig].div(norm_fact, axis=1) * 1e6 -> (vals[pos_sig] /
         norm_fact) * 1e6  (elementwise ops, order preserved: divide first,
         then *1e6)
     Column uniqueness / float64 dtypes / unique bulk index are guarded; any
     violation routes to the upstream implementation (exact by construction).
  3. The per-sample solver loop is copied VERBATIM from upstream (scipy nnls,
     simplex projection with the NK floor, spearmanr/pearsonr/polyfit, the
     gof dict entries) so every float operation and summation order is
     unchanged. tqdm is dropped (it only rendered a progress bar).
  4. The tail (mRNAProportions/otherCells/cellFractions assembly) runs the
     identical pandas statements on the identically-shaped 10-row frames.

Non-default arguments (unlog_bulk, range_based_optim, sig_genes, mRNA_cell(_sub),
scale_exprs=False, with_other_cells=False, constrained_sum=False, init_jitter)
fall back to the upstream function.

Cache invalidation: call epic_fast.invalidate_cache() if you mutate a
`reference` dict (or its frames) in place between calls.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr

__all__ = ["EPIC", "invalidate_cache"]

# ---------------- DEFAULT mRNA PER CELL (verbatim from upstream) -----------
mRNA_cell_default = {
    'Bcells': 0.4016,
    'Macrophages': 1.4196,
    'Monocytes': 1.4196,
    'Neutrophils': 0.1300,
    'NKcells': 0.4396,
    'Tcells': 0.3952,
    'CD4_Tcells': 0.3952,
    'CD8_Tcells': 0.3952,
    'Thelper': 0.3952,
    'Treg': 0.3952,
    'otherCells': 0.4000,
    'default': 0.4000
}

_NK_FLOOR = 1e-12


def _import_original():
    from iobrpy.workflow import epic  # noqa
    return epic


def _merge_duplicates_values(df: pd.DataFrame):
    """Upstream merge_duplicates for unique indexes returns df unchanged;
    otherwise groupby(level=0, sort=False).median(numeric_only=True)."""
    if df.index.has_duplicates:
        n_dup = df.index.duplicated().sum()
        warnings.warn(
            f"There are {n_dup} duplicated gene names; using median."
        )
        return df.groupby(level=0, sort=False).median(numeric_only=True)
    return df


def _project_to_simplex_leq(x: np.ndarray, R: float = 1.0) -> np.ndarray:
    """Project x onto the nonnegative simplex of radius R (sum(x) <= R)."""
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= R:
        return x
    # Euclidean projection onto simplex (Duchi et al., 2008)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - R))[0][-1]
    theta = (cssv[rho] - R) / (rho + 1.0)
    return np.maximum(x - theta, 0.0)


def _find_nk_index(cell_types) -> int:
    """Best-effort index of NK column (handles naming variants)."""
    for i, name in enumerate(cell_types):
        s = str(name).lower().replace(' ', '').replace('_', '')
        if s.startswith('nk') or 'nkcell' in s or 'naturalkiller' in s:
            return i
    return -1


# ---------------- caches ---------------------------------------------------
_ref_cache = {}     # id(reference) -> entry holding a strong ref + merged frames
_common_cache = {}  # (id(reference), digest) -> derived reference-side bundle


def invalidate_cache():
    _ref_cache.clear()
    _common_cache.clear()


def _get_ref_entry(reference: dict):
    """Level-1 memo: merged-duplicate reference frames (upstream semantics)."""
    key = id(reference)
    ent = _ref_cache.get(key)
    if ent is not None and ent["reference"] is reference \
            and ent["refProfiles_src"] is reference.get("refProfiles") \
            and ent["var_src"] is reference.get("refProfiles.var") \
            and ent["sigGenes"] == reference.get("sigGenes") \
            and ent["var_present"] == bool(reference.get("var_present", False)):
        return ent

    refP0 = reference["refProfiles"]
    if not isinstance(refP0, pd.DataFrame):
        return None
    var_present = bool(reference.get("var_present", False))
    var0 = reference.get("refProfiles.var")
    use_var = var_present and var0 is not None

    refP = _merge_duplicates_values(refP0)
    refP = refP.loc[:, ~refP0.columns.duplicated()]
    if not (refP.columns.is_unique and all(dt == np.dtype("float64") for dt in refP.dtypes)
            and refP.index.is_unique):
        return None
    refV = None
    if use_var:
        refV = _merge_duplicates_values(var0).loc[:, refP.columns]
        if not (refV.columns.is_unique and all(dt == np.dtype("float64") for dt in refV.dtypes)
                and refV.index.is_unique):
            return None

    ent = {
        "reference": reference,
        "refProfiles_src": refP0,
        "var_src": var0,
        "sigGenes": reference.get("sigGenes"),
        "var_present": var_present,
        "use_var": use_var,
        "refP": refP,
        "refV": refV,
        "refP_vals": refP.to_numpy(dtype=np.float64, copy=False),
        "refV_vals": None if refV is None else refV.to_numpy(dtype=np.float64, copy=False),
        "refP_idx": refP.index,
        "n_celltypes": refP.shape[1],
    }
    _ref_cache[key] = ent
    return ent


def _get_common_bundle(reference, ent, common: pd.Index):
    """Level-2 memo: everything derived from `common` (sig genes, ref_s,
    weights, A/Aw/sqrtw). Returns None if it cannot be built exactly."""
    digest = hash((len(common), tuple(common)))
    key = (id(reference), digest)
    bundle = _common_cache.get(key)
    if bundle is not None and bundle["ref_entry"] is ent:
        return bundle

    refP, refP_vals = ent["refP"], ent["refP_vals"]
    sig = [g for g in ent["sigGenes"] if g in common]
    sig = list(dict.fromkeys(sig))
    if len(sig) < refP.shape[1]:
        raise ValueError(f"Only {len(sig)} signature genes < {refP.shape[1]} cell types.")

    # scale_counts(refP, sig, common): norm_ref = refP.loc[common].sum(axis=0)
    # .replace(0, 1.0); ref_s = refP.loc[sig].div(norm_ref, axis=1) * 1e6
    # NOTE: pandas sums column-major (pairwise per column); an F-order copy of
    # the gathered block reproduces that summation order bit-for-bit.
    refpos_common = refP.index.get_indexer(common)
    if (refpos_common < 0).any():
        return None
    norm_ref = np.asfortranarray(refP_vals[refpos_common]).sum(axis=0)
    norm_ref = np.where(norm_ref == 0, 1.0, norm_ref)
    refpos_sig = refP.index.get_indexer(sig)
    if (refpos_sig < 0).any():
        return None
    ref_s_vals = (refP_vals[refpos_sig] / norm_ref) * 1e6

    if ent["use_var"]:
        refV_vals = ent["refV_vals"]
        refV_s_vals = (refV_vals[refpos_sig] / norm_ref) * 1e6
        # upstream: w = (ref_s.div(refV_s + 1e-12)).sum(axis=1).to_numpy()
        # reproduced with the identical pandas statements on F-order frames
        # (pandas blocks are column-major; frame values are elementwise-equal)
        rs_df = pd.DataFrame(np.asfortranarray(ref_s_vals), index=sig,
                             columns=refP.columns)
        rv_df = pd.DataFrame(np.asfortranarray(refV_s_vals), index=sig,
                             columns=refP.columns)
        w = rs_df.div(rv_df + 1e-12).sum(axis=1, skipna=True).to_numpy()
        med_w = np.median(w[w > 0]) if np.any(w > 0) else 1.0
        w = np.minimum(w, 100.0 * med_w)
    else:
        w = np.ones(len(sig), dtype=float)

    sqrtw = np.sqrt(w, dtype=float)
    # F-order: upstream's ref_s.to_numpy(copy=False) is F-contiguous (pandas
    # blocks are column-major), and A.dot(x) / B slicing follow that layout —
    # BLAS gemv summation order (and thus the fit_gof bits) depends on it.
    A = np.asfortranarray(ref_s_vals)
    Aw = (A.T * sqrtw).T  # exact same op as upstream (w == 1 -> values unchanged)

    bundle = {
        "ref_entry": ent,
        "sig": sig,
        "gene_order": sig,
        "norm_ref": norm_ref,
        "w": w,
        "sqrtw": sqrtw,
        "A": A,
        "Aw": Aw,
        "cell_types": list(refP.columns),
        "nk_idx": _find_nk_index(list(refP.columns)),
    }
    _common_cache[key] = bundle
    return bundle


# ---------------- CORE: EPIC (fast path) -----------------------------------
def EPIC(bulk: pd.DataFrame,
         reference: dict,
         mRNA_cell=None,
         mRNA_cell_sub=None,
         sig_genes=None,
         scale_exprs=True,
         with_other_cells=True,
         constrained_sum=True,
         range_based_optim=False,
         solver='SLSQP',
         init_jitter=0.0,
         unlog_bulk=False):
    default_args = (
        mRNA_cell is None and not mRNA_cell_sub and sig_genes is None
        and scale_exprs and with_other_cells and constrained_sum
        and not range_based_optim and not unlog_bulk and init_jitter == 0.0
    )
    fast_ok = (
        default_args
        and isinstance(bulk, pd.DataFrame) and isinstance(reference, dict)
        and "refProfiles" in reference and "sigGenes" in reference
        and bulk.columns.is_unique
        and all(dt == np.dtype("float64") for dt in bulk.dtypes)
    )
    if fast_ok:
        ent = _get_ref_entry(reference)
        fast_ok = ent is not None
    if fast_ok:
        vals = bulk.to_numpy(dtype=np.float64, copy=False)
        na_all = np.isnan(vals).all(axis=1)
        if na_all.any():
            warnings.warn(f"{na_all.sum()} genes are NA in all bulk samples; removing.")
            bulk = bulk.loc[~na_all]
            vals = bulk.to_numpy(dtype=np.float64, copy=False)
        fast_ok = not bulk.index.has_duplicates
    if fast_ok:
        common = bulk.index.intersection(ent["refP_idx"])
        bundle = _get_common_bundle(reference, ent, common)
        fast_ok = bundle is not None
    if not fast_ok:
        return _import_original().EPIC(
            bulk=bulk, reference=reference, mRNA_cell=mRNA_cell,
            mRNA_cell_sub=mRNA_cell_sub, sig_genes=sig_genes,
            scale_exprs=scale_exprs, with_other_cells=with_other_cells,
            constrained_sum=constrained_sum, range_based_optim=range_based_optim,
            solver=solver, init_jitter=init_jitter, unlog_bulk=unlog_bulk)

    sig = bundle["sig"]
    gene_order = bundle["gene_order"]
    A = bundle["A"]
    Aw = bundle["Aw"]
    sqrtw = bundle["sqrtw"]
    cell_types = bundle["cell_types"]
    nk_idx = bundle["nk_idx"]

    # bulk-side scaling (value-identical to upstream scale_counts)
    # F-order copy: pandas sums column-major (pairwise) — see bundle builder.
    pos_common = bulk.index.get_indexer(common)
    norm_fact = np.asfortranarray(vals[pos_common]).sum(axis=0)
    norm_fact = np.where(norm_fact == 0, 1.0, norm_fact)
    pos_sig = bulk.index.get_indexer(gene_order)
    # F-order to match upstream B = bulk_s.loc[gene_order].to_numpy(copy=False),
    # which is F-contiguous (pandas block layout); feeds b = B[:, i].
    B = np.asfortranarray((vals[pos_sig] / norm_fact) * 1e6)

    n_samples = B.shape[1]
    nC = A.shape[1]

    mprops = np.empty((n_samples, nC), dtype=float)
    gof_list = []

    for i in range(n_samples):
        b = B[:, i]
        # Weighted NNLS: minimize ||sqrt(W)(A x - b)||^2  s.t. x >= 0
        bw = b * sqrtw
        x, _ = nnls(Aw, bw)

        # ---- NK-preserving projection under constraints ----
        if constrained_sum:
            if with_other_cells:
                if nk_idx >= 0:
                    eps = _NK_FLOOR
                    x = _project_to_simplex_leq(x, R=1.0 - eps)
                    x[nk_idx] += eps
                else:
                    x = _project_to_simplex_leq(x, R=1.0)
            else:
                s = x.sum()
                if s > 0:
                    x = x / s
                else:
                    x[:] = 0.0
        # ----------------------------------------------------

        mprops[i, :] = x

        # GOF metrics (verbatim from upstream)
        b_est = A.dot(x)
        sp = spearmanr(b, b_est)
        pe = pearsonr(b, b_est)
        try:
            a, b0 = np.polyfit(b, b_est, 1)
        except np.linalg.LinAlgError:
            a, b0 = np.nan, np.nan
        a0 = (np.sum(b * b_est) / np.sum(b * b)) if np.sum(b * b) else np.nan
        resid = (b_est - b) * sqrtw
        rmse = np.sqrt(np.mean(resid * resid))
        resid0 = (-b) * sqrtw
        rmse0 = np.sqrt(np.mean(resid0 * resid0))

        gof_list.append({
            'convergeCode': 0,
            'convergeMessage': 'nnls',
            'RMSE_weighted': rmse,
            'Root_mean_squared_geneExpr_weighted': rmse0,
            'spearmanR': sp.correlation, 'spearmanP': sp.pvalue,
            'pearsonR': pe.statistic, 'pearsonP': pe.pvalue,
            'regline_a_x': a,
            'regline_b': b0,
            'regline_a_x_through0': a0,
            'sum_mRNAProportions': np.sum(x)
        })

    mRNA_df = pd.DataFrame(mprops, index=bulk.columns, columns=cell_types)
    if with_other_cells:
        mRNA_df['otherCells'] = 1 - mRNA_df.sum(axis=1)
    # mRNA per cell
    mc = reference.get('mRNA_cell', mRNA_cell_default).copy()
    denom = [mc.get(c, mc.get('default', 1.0)) for c in mRNA_df.columns]
    cf_raw = mRNA_df.div(denom, axis=1)
    cf = cf_raw.div(cf_raw.sum(axis=1), axis=0)

    gof_df = pd.DataFrame(gof_list, index=bulk.columns)
    return {'mRNAProportions': mRNA_df, 'cellFractions': cf, 'fit_gof': gof_df}
