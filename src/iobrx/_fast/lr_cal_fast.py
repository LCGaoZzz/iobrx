"""lr_cal_fast: Rust + vectorized drop-in for iobrpy.workflow.LR_cal.LR_cal.

Bit-exactness strategy (gated on the frozen official inputs — stad TPM
54,658x10 (count2tpm B1 output), eset_stad_symbol 50,181x10, imvigor210
872x348 and the ENSG eset_stad 60,483x10 degenerate case: output CSV sha256
byte-identical to the ORIGINAL):

  1. feature_manipulation(is_matrix=True) — the ORIGINAL's 81% hotspot
     (four per-gene pandas passes over 54,658 genes) — becomes ONE Rust pass
     (`_rust.lr_gene_valid_mask`, rayon over genes) replicating, per gene
     row of the genes x samples matrix:
       dropna(how='any')   -> row has no NaN
       is_numeric_dtype    -> guarded Python-side: the fast path requires an
                              all-float64/int64 frame (df.T unifies numeric
                              dtypes to float64, so the ORIGINAL's per-gene
                              dtype check cannot differ); anything else
                              (object/bool/extension dtypes, duplicate
                              index/columns) routes to the ORIGINAL
                              compute_LR_pairs (exact by construction).
       np.isfinite().all() -> row all finite
       df[f].std() != 0    -> pandas nanvar on a NaN-free float64 Series is
                              a TWO-PASS reduction with numpy PAIRWISE sums
                              (bottleneck is absent in the validated env;
                              verified bit-identical vs pd.Series(x).std()
                              on 4000 adversarial random vectors covering
                              the n<8 / 8<=n<=128 / n>128 recursion
                              regimes):
                                mean = np.sum(x) / n
                                ssq  = np.sum((x - mean) ** 2)
                                std  = sqrt(ssq / (n - 1))
                              The Rust kernel reuses the crate's np_sum_f64
                              (numpy pairwise, 8192-chunked) — the same
                              kernel cibersort's zscore1d is gated on.
                              The ORIGINAL's `if df.shape[0] > 1` (shape[0]
                              of the TRANSPOSED frame = sample count) maps
                              to filter_zero_var = (S > 1).
     The no-.so fallback recomputes the same two-pass per row with 1-D
     np.sum calls (identical pairwise kernel). A vectorized
     vals.sum(axis=1) is NOT bit-exact: numpy's inner-axis reduce is
     sequential and differs from pairwise in the last ulp on ~20% of real
     rows (measured) — do not "optimize" it that way.
  2. log2(TPM + 1): np.log2(vals + 1) on the float64 values matrix —
     bit-identical to np.log2(df + 1).to_numpy() (verified). Values < -1
     produce NaN with the same numpy RuntimeWarning as the ORIGINAL
     (double-logging log2-scale input is upstream behaviour, preserved).
  3. Per-pair min: gene_expr.loc[[a, b]].min(axis=0) (skipna) is
     bit-identical to np.fmin(row_a, row_b) — verified on all 862 matched
     pairs of stad TPM and on the NaN-bearing imvigor210 double-log case.
     Absent genes give an all-NaN row exactly like the ORIGINAL; the
     ORIGINAL's O(G) `a in genes` list scans become a dict lookup.
  4. group_lrpairs merging: the ORIGINAL's 56-rule list surgery (rename
     EVERY occurrence of `main` to combo_name; remove EVERY occurrence of
     each involved pair; dedupe keeping the first) is replicated verbatim
     on the column-name list while tracking original column positions, so
     the incremental O(P) np.delete copies collapse into one final column
     gather — same surviving columns, same order, same values.
  5. Tail: the `~np.isnan(mat).any(axis=0)` column drop, the DataFrame
     assembly, `result.insert(0, 'ID', result.index)` and
     `to_csv(sep=detect_sep(out))` are the ORIGINAL statements, so the CSV
     text is byte-identical.

Bug-compatibility with iobrpy 0.2.0 (deliberately NOT fixed):
  * data_type='count' performs the ORIGINAL call
    count2tpm(df, idType=..., org='hsa', source='local'), which omits the
    two required positional arguments and therefore raises the same
    TypeError ("count2tpm() missing 2 required positional arguments:
    'anno_grch38' and 'anno_gc_vm32'"). If upstream ever fixes the
    signature, this path follows automatically (it calls the real
    function).
  * The API-level default data_type='count' (while the ORIGINAL CLI
    defaults to 'tpm') is preserved.
  * An invalid cancer_type raises KeyError from
    intercell_networks[cancer_type] AFTER gene filtering, as upstream.
  * A pair label with more than one '_' raises the same
    "too many values to unpack" ValueError as the ORIGINAL split.
Dropped from the ORIGINAL: the tqdm progress bar and the decorative
end-of-run banner (iobrx library convention); the verbose filter-count and
[LOG] lines are preserved.
"""
from __future__ import annotations
from iobrx._resources import resource_path

import os
import pickle

import numpy as np
import pandas as pd

__all__ = ["LR_cal", "compute_LR_pairs", "feature_manipulation",
           "detect_sep", "invalidate_cache"]

try:
    import iobrx._rust as _ir
    _RUST_READY = hasattr(_ir, "lr_gene_valid_mask")
except (ImportError, OSError):  # pragma: no cover - env without the .so
    _ir = None
    _RUST_READY = False

# ---------------------------------------------------------------------------
# lr_data.pkl: one ~0.14 s pickle.load per process (the ORIGINAL re-reads it
# on every call; on small matrices that load is its single largest cost).
# ---------------------------------------------------------------------------
_LR_DATA = None


def _lr_data() -> dict:
    global _LR_DATA
    if _LR_DATA is None:
        from importlib.resources import files

        with open(resource_path("lr_data.pkl"), "rb") as f:
            _LR_DATA = pickle.load(f)
    return _LR_DATA


def invalidate_cache():
    """Drop the cached lr_data.pkl bundle (e.g. after mutating it in place)."""
    global _LR_DATA
    _LR_DATA = None


def detect_sep(filename: str) -> str:
    """Verbatim port of the ORIGINAL detect_sep (.tsv/.txt -> tab, else comma)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.tsv', '.txt']:
        return '\t'
    return ','


# ---------------------------------------------------------------------------
# feature_manipulation (is_matrix=True path) — bit-exact fast replacement
# ---------------------------------------------------------------------------
def _std_zero_two_pass(x: np.ndarray) -> bool:
    """pandas Series(x).std() == 0 via the exact nanvar two-pass order
    (1-D np.sum = numpy pairwise; bottleneck absent — verified bitwise)."""
    n = x.shape[0]
    mean = np.sum(x) / n
    d = x - mean
    ssq = np.sum(d * d)
    return np.sqrt(ssq / (n - 1)) == 0.0


def _valid_mask_numpy(vals: np.ndarray, filter_zero_var: bool) -> np.ndarray:
    """No-.so fallback: same semantics/order as the Rust kernel."""
    keep = ~np.isnan(vals).any(axis=1)
    keep &= np.isfinite(vals).all(axis=1)
    if filter_zero_var and vals.shape[1] > 1:
        for i in np.flatnonzero(keep):
            if _std_zero_two_pass(vals[i]):
                keep[i] = False
    return keep


def _fast_path_ok(data: pd.DataFrame):
    """Guard: all-float64/int64 numpy dtypes, unique index and columns.
    Anything else must take the ORIGINAL path (object/bool dtypes change the
    is_numeric_dtype filter; duplicate gene labels change .loc semantics)."""
    if not isinstance(data, pd.DataFrame):
        return False
    if data.index.has_duplicates or data.columns.has_duplicates:
        return False
    for dt in data.dtypes:
        if not isinstance(dt, np.dtype) or dt not in (np.dtype("float64"),
                                                      np.dtype("int64")):
            return False
    return True


def feature_manipulation(data: pd.DataFrame, n_threads: int = 8,
                         print_result: bool = False):
    """Bit-exact fast port of the ORIGINAL feature_manipulation(data,
    is_matrix=True, print_result=...). Returns (valid_genes, vals, mask) —
    vals the float64 C-order (genes, samples) values matrix, mask the
    per-gene keep flags — or None when the fast-path guards fail."""
    if not _fast_path_ok(data):
        return None
    vals = data.to_numpy(dtype=np.float64)
    if not vals.flags["C_CONTIGUOUS"]:
        vals = np.ascontiguousarray(vals)
    g, s = vals.shape
    filter_zero_var = s > 1          # ORIGINAL: df.shape[0] > 1 AFTER .T
    if _RUST_READY:
        mask = _ir.lr_gene_valid_mask(vals, filter_zero_var, n_threads) != 0
    else:
        mask = _valid_mask_numpy(vals, filter_zero_var)
    if print_result:
        # Same lines/counts as the ORIGINAL (per-step funnel). The numeric
        # filter cannot remove anything on the fast path (guarded dtypes).
        # ORIGINAL counts: complete = dropna survivors; finite is counted
        # WITHIN complete (a NaN row is already gone before the isfinite
        # step), so the infinite-filter numerator must exclude NaN rows.
        has_nan = np.isnan(vals).any(axis=1)
        n_complete = int((~has_nan).sum())
        n_finite = int(np.isfinite(vals).all(axis=1).sum())  # subset of complete
        n_valid = int(mask.sum())
        print(f"After NA filter: {n_complete}/{g}")
        print(f"After non-numeric filter: {n_complete}/{n_complete}")
        print(f"After infinite filter: {n_finite}/{n_complete}")
        print(f"After zero-variance filter: {n_valid}/{n_finite}")
    valid_genes = [gene for gene, k in zip(data.index, mask) if k]
    return valid_genes, vals, mask


# ---------------------------------------------------------------------------
# compute_LR_pairs — bit-exact fast replacement
# ---------------------------------------------------------------------------
def compute_LR_pairs(RNA_tpm: pd.DataFrame,
                     cancer_type: str,
                     intercell_networks: dict,
                     group_lrpairs: list,
                     verbose: bool = False,
                     n_threads: int = 8) -> pd.DataFrame:
    fast = feature_manipulation(RNA_tpm, n_threads=n_threads,
                                print_result=verbose)
    if fast is None:
        # Guards failed — the untouched ORIGINAL core (exact by construction).
        from iobrpy.workflow.LR_cal import compute_LR_pairs as _orig
        return _orig(RNA_tpm, cancer_type, intercell_networks, group_lrpairs,
                     verbose)
    valid_genes, vals, valid_mask = fast
    sub = np.ascontiguousarray(vals[np.flatnonzero(valid_mask)])

    # 1. Log2 transform TPM (bitwise == np.log2(RNA_tpm.loc[valid] + 1))
    expr = np.log2(sub + 1)
    samples = RNA_tpm.columns.tolist()
    genes = valid_genes
    if verbose:
        print(f"[LOG] After feature_manipulation, genes: {len(valid_genes)}")
        print(f"[LOG] gene_expr shape: ({len(genes)}, {len(samples)})")

    # 2. Build raw pairs list in order (ORIGINAL loop semantics)
    net = intercell_networks[cancer_type]
    lr_raw = []
    for i in range(len(net['ligands'])):
        a = str(net['ligands'][i])
        b = str(net['receptors'][i])
        lr_raw.append(f"{a}_{b}")
    pairs = list(dict.fromkeys(lr_raw))   # unique preserving order
    if verbose:
        print(f"[LOG] unique pairs: {len(pairs)}")

    # 3. Per-pair skipna min via np.fmin gather (bitwise == the ORIGINAL
    #    gene_expr.loc[[a, b]].min(axis=0); absent genes -> all-NaN row).
    gpos = {g: i for i, g in enumerate(genes)}     # unique index guarded
    n_s = len(samples)
    ia = np.full(len(pairs), -1, dtype=np.intp)
    ib = np.full(len(pairs), -1, dtype=np.intp)
    for k, p in enumerate(pairs):
        a, b = p.split('_')      # ORIGINAL unpack (ValueError on >2 parts)
        ia[k] = gpos.get(a, -1)
        ib[k] = gpos.get(b, -1)
    both = (ia >= 0) & (ib >= 0)
    mat = np.full((len(pairs), n_s), np.nan)
    if both.any():
        mat[both] = np.fmin(expr[ia[both]], expr[ib[both]])
    mat = mat.T                  # samples x pairs (ORIGINAL np.vstack(data).T)

    # 4. Grouping by positions — ORIGINAL list surgery, with the np.delete
    #    chain replaced by original-column tracking + one final gather.
    cols = pairs.copy()
    orig_idx = list(range(len(pairs)))
    for grp in group_lrpairs:
        main = grp['main']
        raw_remove = grp.get('involved_pairs', [])
        if isinstance(raw_remove, str) and raw_remove:
            remove = [raw_remove]
        elif isinstance(raw_remove, list):
            remove = raw_remove
        else:
            remove = []
        combo = grp.get('combo_name', main)
        for idx, c in enumerate(cols):
            if c == main:
                cols[idx] = combo
        for ip in remove:
            while ip in cols:
                j = cols.index(ip)
                cols.pop(j)
                orig_idx.pop(j)
    # remove duplicate columns created by renaming (keep first)
    unique_cols = []
    unique_idx = []
    seen = set()
    for idx, c in enumerate(cols):
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
            unique_idx.append(idx)
    orig_idx = [orig_idx[i] for i in unique_idx]
    mat = mat[:, orig_idx]
    cols = unique_cols
    if verbose:
        print(f"[LOG] after grouping columns: {len(cols)}")

    # 5. Drop columns where any sample is NaN (ORIGINAL mask)
    mask = ~np.isnan(mat).any(axis=0)
    final_cols = [c for c, m in zip(cols, mask) if m]
    mat = mat[:, mask]
    if verbose:
        print(f"[LOG] after drop any NA: {len(final_cols)}")

    return pd.DataFrame(mat, index=samples, columns=final_cols)


# ---------------------------------------------------------------------------
# LR_cal — shell around compute_LR_pairs (ORIGINAL statement order)
# ---------------------------------------------------------------------------
def LR_cal(input_file,
           output_file=None,
           data_type: str = 'count',
           id_type: str = 'ensembl',
           cancer_type: str = 'pancan',
           verbose: bool = False,
           n_threads: int = 8,
           engine: str = 'fast') -> pd.DataFrame:
    """Accelerated LR_cal. ``input_file`` accepts the ORIGINAL's path (str)
    or a DataFrame directly (iobrx extension). ``engine='original'`` runs the
    ORIGINAL compute_LR_pairs core inside the same shell. Returns the frame
    that is written to ``output_file`` (ORIGINAL returns None)."""
    if isinstance(input_file, pd.DataFrame):
        df = input_file
    else:
        sep_in = detect_sep(input_file)
        df = pd.read_csv(input_file, sep=sep_in, index_col=0)
    if verbose:
        print(f"Loaded {df.shape}")
    if data_type == 'count':
        # BUG-COMPATIBLE with iobrpy 0.2.0: the ORIGINAL call omits the two
        # required positional arguments anno_grch38/anno_gc_vm32, so this
        # raises the identical TypeError. It invokes the REAL count2tpm, so
        # an upstream signature fix propagates automatically.
        from iobrpy.workflow.count2tpm import count2tpm
        df = count2tpm(df, idType=id_type, org='hsa', source='local')
        if verbose:
            print(f"TPM {df.shape}")
    data = _lr_data()
    intercell_networks = data['intercell_networks']
    group_lrpairs = data['group_lrpairs']
    if engine == 'original':
        from iobrpy.workflow.LR_cal import compute_LR_pairs as _orig
        result = _orig(df, cancer_type, intercell_networks, group_lrpairs,
                       verbose)
    else:
        result = compute_LR_pairs(df, cancer_type, intercell_networks,
                                  group_lrpairs, verbose, n_threads)
    result.insert(0, 'ID', result.index)
    if output_file is not None:
        sep_out = detect_sep(output_file)
        result.to_csv(output_file, sep=sep_out, index=False)
        if verbose:
            print(f"Saved {result.shape}")
    return result
