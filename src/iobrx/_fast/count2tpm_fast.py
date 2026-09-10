# -*- coding: utf-8 -*-
"""Vectorized drop-in replacement for IOBRpy's ``count2tpm``.

Semantics are byte-for-byte those of
``iobrpy/workflow/count2tpm.py`` (feature_manipulation + remove_duplicate_genes
+ count2tpm), including the quirk that when either ``anno_grch38`` or
``anno_gc_vm32`` is not a DataFrame the *packaged* tables are reloaded from
``iobrpy.resources/count2tpm_data.pkl`` and the passed ``anno_grch38`` is
IGNORED. The packaged tables (and their length/symbol refinement) are cached at
module level after the first load, since they are process constants.

Cold-process cost note: the packaged pickle stores BOTH annotation tables as
raw Python lists of row-dicts (~18 MB); a first packaged-branch call used to
pay ~0.18-0.25 s for read_pickle + ``pd.DataFrame(list)`` of *both* tables +
the refine sort — even though e.g. the ``hsa/Ensembl`` path only ever reads a
3-column refined view of ``anno_grch38``. This module therefore

  * constructs each packaged table DataFrame lazily (a human-only call no
    longer materializes the mouse table), and
  * persists each branch's *refined* reference table
    (``anno[cols] -> sort_values('eff_length', desc) -> drop_duplicates('id')
    -> set_index('id')``) to a parquet disk cache next to this file
    (``_pkgcache/``), keyed on the source pickle's (path, size, mtime_ns) and
    the exact refine parameters. On a cache hit the packaged pickle is never
    opened. Any disk-cache problem (missing dir, stale key, corrupt file,
    read error) silently falls back to computing from the pickle, so results
    can never diverge from the upstream-defined refinement.

No per-column ``apply`` and no row-wise numeric loops: every numeric filter is a
single vectorized numpy/pandas operation. String index handling uses the
vectorized pandas ``.str`` accessor.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_numeric_dtype
from iobrx._resources import resource_path

__all__ = ["count2tpm", "feature_manipulation", "remove_duplicate_genes"]

# ---------------------------------------------------------------------------
# Packaged annotation cache (constant resources, loaded once per process)
# ---------------------------------------------------------------------------

_PACKAGED_RAW: dict | None = None          # raw pickled dict (lists of row-dicts)
_PACKAGED_CACHE: dict = {}                 # table name -> DataFrame (lazy per table)
_PACKAGED_SRC_IDENT: tuple | None = None   # (path, size, mtime_ns) of the source pickle
_REFINED_CACHE: dict = {}                  # ('__packaged__', branch) or (id(anno), branch) -> refined df_ref

_DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pkgcache")


def _as_df(obj, name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        return pd.DataFrame(obj)
    except Exception as e:  # pragma: no cover - mirrors upstream error contract
        raise TypeError(f"{name} must be a DataFrame; got {type(obj).__name__}") from e


def _packaged_source_ident() -> tuple:
    """(realpath, size, mtime_ns) of the packaged count2tpm pickle (resolved once)."""
    global _PACKAGED_SRC_IDENT
    if _PACKAGED_SRC_IDENT is None:
        from importlib.resources import files
        path = os.path.realpath(resource_path('count2tpm_data.pkl'))
        st = os.stat(path)
        _PACKAGED_SRC_IDENT = (path, int(st.st_size), int(st.st_mtime_ns))
    return _PACKAGED_SRC_IDENT


def _load_packaged_raw() -> dict:
    global _PACKAGED_RAW
    if _PACKAGED_RAW is None:
        with open(_packaged_source_ident()[0], 'rb') as f:
            _PACKAGED_RAW = pd.read_pickle(f, compression=None)
    return _PACKAGED_RAW


def _packaged_table(name: str) -> pd.DataFrame:
    """Construct (once) the packaged table ``name`` exactly as upstream does
    (``pd.DataFrame`` over the pickled list of row-dicts)."""
    if name not in _PACKAGED_CACHE:
        _PACKAGED_CACHE[name] = _as_df(_load_packaged_raw()[name], name)
    return _PACKAGED_CACHE[name]


def _load_packaged() -> dict:
    """Load (once, lazily per table) and cache the packaged anno tables."""
    return {'anno_grch38': _packaged_table('anno_grch38'),
            'anno_gc_vm32': _packaged_table('anno_gc_vm32')}


def _string_index(index: pd.Index) -> pd.Index:
    """``index.astype(str)`` semantics, skipping the conversion when every
    element is already a Python str (infer_dtype is C-fast)."""
    if index.dtype == object and infer_dtype(index) in ("string", "empty"):
        return index
    return index.astype(str)


# ---------------------------------------------------------------------------
# Refined reference tables: pure refine + process/disk caches
# ---------------------------------------------------------------------------

def _refine(anno: pd.DataFrame, columns: list, new_names: list) -> pd.DataFrame:
    """The upstream refinement, verbatim:
    ``anno[columns] -> columns=new_names -> sort_values('eff_length', desc)
    -> drop_duplicates('id') -> set_index('id')``."""
    df_ref = anno[columns].copy()
    df_ref.columns = new_names
    df_ref = df_ref.sort_values("eff_length", ascending=False).drop_duplicates(subset="id")
    return df_ref.set_index("id")


def _disk_cache_paths(branch: str):
    base = os.path.join(_DISK_CACHE_DIR, branch)
    return base + ".parquet", base + ".meta.json"


def _load_refined_disk(branch: str, columns: list, new_names: list):
    """Load a refined packaged table from the disk cache, or None.

    A cache entry is honored only when the meta matches the source pickle's
    current (path, size, mtime_ns) and the exact refine parameters; anything
    else is treated as a miss. Any error -> None (caller recomputes)."""
    try:
        pq, meta_path = _disk_cache_paths(branch)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if (meta.get("source") != list(_packaged_source_ident())
                or meta.get("columns") != list(columns)
                or meta.get("new_names") != list(new_names)):
            return None
        df = pd.read_parquet(pq)
        if list(df.columns) != list(new_names[1:]):  # index restored separately
            return None
        if df.index.name != "id":
            return None
        return df
    except Exception:
        return None


def _write_refined_disk(branch: str, columns: list, new_names: list, df_ref: pd.DataFrame) -> None:
    """Best-effort persist of a refined packaged table (atomic tmp+rename)."""
    try:
        os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
        pq, meta_path = _disk_cache_paths(branch)
        tmp_pq, tmp_meta = pq + ".tmp", meta_path + ".tmp"
        df_ref.to_parquet(tmp_pq)
        with open(tmp_meta, "w") as f:
            json.dump({"source": list(_packaged_source_ident()),
                       "branch": branch, "columns": list(columns),
                       "new_names": list(new_names),
                       "shape": list(df_ref.shape)}, f)
        os.replace(tmp_pq, pq)
        os.replace(tmp_meta, meta_path)
    except Exception:
        pass  # cache is an optimization only; never fail the computation


def _packaged_refined(branch: str, columns: list, new_names: list) -> pd.DataFrame:
    """Refined reference table for a *packaged*-table branch: process cache ->
    disk cache -> compute from the packaged pickle (and persist)."""
    key = ("__packaged__", branch)
    hit = _REFINED_CACHE.get(key)
    if hit is not None:
        return hit
    df_ref = _load_refined_disk(branch, columns, new_names)
    if df_ref is None:
        table = _packaged_table('anno_grch38' if branch.startswith("hsa") else 'anno_gc_vm32')
        df_ref = _refine(table, columns, new_names)
        _write_refined_disk(branch, columns, new_names, df_ref)
    _REFINED_CACHE[key] = df_ref
    return df_ref


def _refined_ref(anno: pd.DataFrame, branch: str,
                 columns: list, rename: bool) -> pd.DataFrame:
    """``anno[columns] -> sort_values('eff_length', desc) -> drop_duplicates('id')
    -> set_index('id')`` — cached when ``anno`` is the process-constant packaged
    table, computed per call otherwise."""
    new_names = ["id", "eff_length", "symbol"]
    key = (id(anno), branch)
    hit = _REFINED_CACHE.get(key)
    if hit is not None:
        return hit
    df_ref = _refine(anno, columns, new_names)
    if anno is _PACKAGED_CACHE_GET(branch):
        _REFINED_CACHE[key] = df_ref
    return df_ref


def _PACKAGED_CACHE_GET(branch: str):  # identity check against already-built packaged tables
    name = "anno_grch38" if branch.startswith("hsa") else "anno_gc_vm32"
    return _PACKAGED_CACHE.get(name)


# ---------------------------------------------------------------------------
# Vectorized feature_manipulation
# ---------------------------------------------------------------------------

def feature_manipulation(data: pd.DataFrame,
                         feature: list | None = None,
                         is_matrix: bool = False,
                         print_result: bool = False) -> list:
    """Vectorized equivalent of upstream ``feature_manipulation``.

    Filters features on: any NA -> non-numeric dtype -> any +/-inf -> zero
    variance (pandas std, ddof=1). Per-column statistics are independent of
    which *other* columns are examined, so the upstream progressive column
    lists collapse to an AND of vectorized masks with identical output.
    """
    df = data
    if is_matrix:
        feature = list(df.index)
        work = df.T                       # columns == features
    else:
        if feature is None:
            feature = list(df.columns)
        work = df[feature]

    na_bad = work.isna().any(axis=0)

    num_ok = work.dtypes.map(is_numeric_dtype)
    num_ok.name = None

    keep = (~na_bad) & num_ok
    cand = [f for f, k in zip(feature, keep.values) if k] if not bool(keep.all()) else feature

    # inf / zero-variance only over NA-free numeric columns (as upstream)
    if len(cand) < len(feature):
        sub = work[cand]
    else:
        sub = work
    inf_bad = np.isinf(sub.to_numpy(dtype=np.float64)).any(axis=0)
    inf_bad = pd.Series(inf_bad, index=sub.columns)

    zero_var = sub.std(axis=0) == 0

    ok = keep.reindex(sub.columns).fillna(False).values & (~inf_bad.values) & (~zero_var.values)
    out = [f for f, k in zip(sub.columns, ok) if k]
    # preserve upstream ordering: features appear in their original order
    order = {f: i for i, f in enumerate(feature)}
    out.sort(key=order.__getitem__)
    return out


def _filter_matrix_features(countMat: pd.DataFrame) -> pd.DataFrame:
    """Fast path of feature_manipulation(is_matrix=True) for an all-numeric
    float matrix (the only shape count2tpm produces: ``.astype(float)``)."""
    v = countMat.to_numpy(dtype=np.float64, copy=False)
    na_bad = np.isnan(v).any(axis=1)
    if na_bad.any():
        pre_ok = ~na_bad
        v = countMat[pre_ok].to_numpy(dtype=np.float64, copy=False)
    else:
        pre_ok = None
    inf_bad = np.isinf(v).any(axis=1)
    ok = ~(inf_bad)
    if pre_ok is not None:
        # recompose on the full frame: NA rows must stay EXCLUDED (the
        # original feature_manipulation drops any row with an NA before the
        # inf check), so the rebuilt mask starts all-False and only the
        # NA-free rows inherit their inf status.
        full_ok = np.zeros(countMat.shape[0], dtype=bool)
        idx_pre = np.flatnonzero(pre_ok)
        full_ok[idx_pre] = ok
        ok = full_ok
    # zero-variance: identical per-gene pandas nanstd (ddof=1); layout of the
    # transposed reduction matches upstream df[feature].std(axis=0) bit-for-bit
    # (both accumulate the 10 sample rows top-down, sequentially).
    countMat = countMat.loc[ok]
    zv = countMat.T.std(axis=0).to_numpy() == 0
    if zv.any():
        countMat = countMat.loc[~zv]
    return countMat


# ---------------------------------------------------------------------------
# Vectorized remove_duplicate_genes
# ---------------------------------------------------------------------------

def remove_duplicate_genes(df: pd.DataFrame,
                           column_of_symbol: str = "symbol",
                           method: str = "mean",
                           show_progress: bool = True) -> pd.DataFrame:
    """Vectorized equivalent of upstream ``remove_duplicate_genes``.

    Rows are scored (mean/sd/sum across numeric columns), sorted by score
    descending (pandas default quicksort, as upstream) and the first occurrence
    per symbol is kept; result is indexed by symbol with expression columns only.
    """
    if column_of_symbol not in df.columns:
        raise ValueError(f"Column '{column_of_symbol}' not found in DataFrame.")

    sym_col = column_of_symbol
    value_cols = [c for c in df.columns if c != sym_col]
    numeric_cols = [c for c in value_cols if is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found for duplicate resolution.")

    if method == "mean":
        score = df[numeric_cols].mean(axis=1, skipna=True)
    elif method == "sd":
        score = df[numeric_cols].std(axis=1, skipna=True)
    elif method == "sum":
        score = df[numeric_cols].sum(axis=1, skipna=True)
    else:
        raise ValueError("method must be 'mean', 'sd', or 'sum'")

    out = df.copy()
    out["_score"] = score
    out = out.sort_values("_score", ascending=False)
    out = out.drop(columns=["_score"])
    out = out.drop_duplicates(subset=[sym_col], keep="first")
    out = out.set_index(sym_col)
    return out[value_cols]


# ---------------------------------------------------------------------------
# Vectorized count2tpm
# ---------------------------------------------------------------------------

_ENS_VERSION_RE = r"^(ENS[A-Z]*\d{3,})\.\d+$"


def _strip_versions(index: pd.Index) -> tuple[pd.Index, int]:
    """Vectorized equivalent of ``strip_versions_in_index`` (Ensembl-only)."""
    idx = _string_index(index)
    s = pd.Series(idx.to_numpy(dtype=object))
    dotted = s.str.contains(".", regex=False)
    n = 0
    if bool(dotted.any()):
        new = s.copy()
        cand = s[dotted]
        replaced = cand.str.replace(_ENS_VERSION_RE, r"\1", regex=True)
        changed = replaced != cand
        n = int(changed.sum())
        if n:
            new[dotted] = replaced
        return pd.Index(new.to_numpy(dtype=object), name=index.name), n
    return idx, 0


def count2tpm(count_mat: pd.DataFrame,
              anno_grch38: pd.DataFrame,
              anno_gc_vm32: pd.DataFrame = None,
              idType: str = "Ensembl",
              org: str = "hsa",
              source: str = "local",
              effLength_df: pd.DataFrame = None,
              id_col: str = "id",
              gene_symbol_col: str = "symbol",
              length_col: str = "eff_length",
              check_data: bool = False,
              remove_version: bool = False) -> pd.DataFrame:
    """Vectorized drop-in for ``iobrpy.workflow.count2tpm.count2tpm``.

    Produces a DataFrame identical to the upstream implementation (same index,
    same column order, same row order, identical float64 values).
    """
    if org not in ("hsa", "mmus"):
        raise ValueError("`org` must be 'hsa' or 'mmus'.")

    countMat = count_mat.astype(float)  # astype already copies, as upstream's copy().astype()

    # ---- optional Ensembl version stripping (vectorized) --------------------
    if remove_version:
        idx = _string_index(countMat.index)
        looks_ensembl = bool((idx.str.startswith("ENSG") | idx.str.startswith("ENSMUSG")).any())
        if looks_ensembl:
            new_index, n_stripped = _strip_versions(countMat.index)
            if n_stripped:
                countMat.index = new_index

    # ---- NA / check_data filtering (vectorized masks) -----------------------
    has_na = bool(countMat.isna().to_numpy().any())
    if has_na or check_data:
        if not all(is_numeric_dtype(t) for t in countMat.dtypes):
            features = feature_manipulation(countMat, is_matrix=True)
            countMat = countMat.loc[features]
        else:
            countMat = _filter_matrix_features(countMat)

    # ---- annotation tables (upstream quirk preserved) ------------------------
    packaged_mode = not (isinstance(anno_grch38, pd.DataFrame) and isinstance(anno_gc_vm32, pd.DataFrame))
    if not packaged_mode:
        anno_grch38 = _as_df(anno_grch38, "anno_grch38")
        anno_gc_vm32 = _as_df(anno_gc_vm32, "anno_gc_vm32")

    len_series = None
    symbol_map = None

    if effLength_df is not None:
        eff = effLength_df.rename(columns={id_col: "id", length_col: "eff_length",
                                           gene_symbol_col: "gene_symbol"})
        eff = eff.drop_duplicates(subset="id").set_index("id")
        common = countMat.index.intersection(eff.index)
        if common.empty:
            raise ValueError("Identifier of matrix is not match to references.")
        countMat = countMat.loc[common]
        eff = eff.loc[common]
        len_series = eff["eff_length"]
        symbol_map = eff["gene_symbol"]
    else:
        if source != "local":
            raise NotImplementedError("source='biomart' not implemented.")

        symbol_col = "symbol"
        idl = idType.lower()
        if org == "hsa" and idl == "ensembl":
            countMat.index = countMat.index.str[:15]
            df_ref = (_packaged_refined("hsa_ensembl", ["id", "eff_length", "symbol"],
                                        ["id", "eff_length", "symbol"])
                      if packaged_mode else
                      _refined_ref(anno_grch38, "hsa_ensembl", ["id", "eff_length", "symbol"], False))
        elif org == "hsa" and idl == "entrez":
            df_ref = (_packaged_refined("hsa_entrez", ["entrez", "eff_length", "symbol"],
                                        ["id", "eff_length", "symbol"])
                      if packaged_mode else
                      _refined_ref(anno_grch38, "hsa_entrez", ["entrez", "eff_length", "symbol"], True))
        elif org == "hsa" and idl == "symbol":
            if packaged_mode:
                df_ref = _packaged_refined("hsa_symbol", ["symbol", "eff_length", "gc"],
                                           ["id", "eff_length", "gc"])
            else:
                df_ref = anno_grch38[["symbol", "eff_length", "gc"]].copy()
                df_ref.columns = ["id", "eff_length", "gc"]
                df_ref = df_ref.sort_values("eff_length", ascending=False).drop_duplicates(subset="id").set_index("id")
            symbol_col = "id"
        elif org == "mmus" and idl == "ensembl":
            df_ref = (_packaged_refined("mmus_ensembl", ["id", "eff_length", "symbol"],
                                        ["id", "eff_length", "symbol"])
                      if packaged_mode else
                      _refined_ref(anno_gc_vm32, "mmus_ensembl", ["id", "eff_length", "symbol"], False))
        elif org == "mmus" and idl == "mgi":
            df_ref = (_packaged_refined("mmus_mgi", ["mgi_id", "eff_length", "symbol"],
                                        ["id", "eff_length", "symbol"])
                      if packaged_mode else
                      _refined_ref(anno_gc_vm32, "mmus_mgi", ["mgi_id", "eff_length", "symbol"], True))
        elif org == "mmus" and idl == "symbol":
            if packaged_mode:
                df_ref = _packaged_refined("mmus_symbol", ["symbol", "eff_length", "gc"],
                                           ["id", "eff_length", "gc"])
            else:
                df_ref = anno_gc_vm32[["symbol", "eff_length", "gc"]].copy()
                df_ref.columns = ["id", "eff_length", "gc"]
                df_ref = df_ref.sort_values("eff_length", ascending=False).drop_duplicates(subset="id").set_index("id")
            symbol_col = "id"
        else:
            raise ValueError(f"Unsupported idType for {org}: {idType}")

        common = countMat.index.intersection(df_ref.index)
        if common.empty:
            raise ValueError("Identifier of matrix is not match to references.")
        df_ref = df_ref.loc[common]
        countMat = countMat.loc[common]
        len_series = df_ref["eff_length"]
        if symbol_col == "id":
            symbol_map = pd.Series(df_ref.index, index=df_ref.index)
        else:
            symbol_map = df_ref[symbol_col]

    # ---- duplicate handling (no-op for unique indexes, as upstream) ----------
    if len_series.index.duplicated().any():
        mask = ~len_series.index.duplicated(keep="first")
        len_series = len_series[mask]
        symbol_map = symbol_map[mask]

    len_series = len_series.reindex(countMat.index)
    symbol_map = pd.Series(symbol_map).reindex(countMat.index)

    # ---- NA lengths ----------------------------------------------------------
    valid = ~len_series.isna()
    if not bool(valid.all()):
        warnings.warn(f">>>--- Omit {int((~valid).sum())} genes of which length is not available !")
        countMat = countMat.loc[valid]
        len_series = len_series[valid]
        symbol_map = symbol_map[valid]

    # ---- NA / blank symbols ---------------------------------------------------
    symbol_valid = symbol_map.notna() & (symbol_map.astype(str).str.strip() != "")
    if not bool(symbol_valid.all()):
        countMat = countMat.loc[symbol_valid]
        len_series = len_series[symbol_valid]
        symbol_map = symbol_map[symbol_valid]

    # ---- TPM math (pandas calls verbatim as upstream — the internal block of
    # div(axis=0) is F-ordered and sum(axis=0) then uses pairwise summation down
    # each column; any other summation layout changes the last bits) ------------
    divisor = len_series / 1000.0
    rpk = countMat.div(divisor, axis=0)
    col_sums = rpk.sum(axis=0)
    tpm = rpk.div(col_sums, axis=1) * 1e6
    tpm = tpm.replace([np.inf, -np.inf], 0).fillna(0.0)

    tpm_df = tpm
    tpm_df.insert(0, "symbol", symbol_map.to_numpy())

    # ---- collapse duplicate symbols (highest mean TPM row wins) ----------------
    tpm_df = remove_duplicate_genes(tpm_df, "symbol")
    tpm_df.index.name = None
    return tpm_df
