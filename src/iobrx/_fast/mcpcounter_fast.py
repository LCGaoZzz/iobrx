"""mcpcounter_fast: cached + vectorized drop-in for
iobrpy.workflow.mcpcounter.MCPcounter_estimate().

Same signature: MCPcounter_estimate(expression, features_type)

Bit-exactness strategy (gated against iobrx/refs/mcpcounter.parquet on the
official eset_stad_symbol x HUGO_symbols call):

  1. mcp_data.pkl loading and the .str.upper().str.strip() normalization are
     constant per features_type; memoized at module level keyed by
     (resource path, features_type). The package resource is immutable.
  2. `sig_df[sig_df[id_col].isin(expression.index)]` is replaced by
     Index.get_indexer on the marker ids (same rows kept, same order);
     marker lists per population and the first-appearance population order
     (valid[pop_col].unique()) are unchanged.
  3. The per-population gene selection keeps the EXACT upstream call
     `expression.index.intersection(genes)` — pandas returns a
     non-sorted, non-marker order here, so the call itself is preserved
     verbatim and only the downstream `.loc[common].mean(axis=0)` is
     replaced by a positional gather + ndarray col-sum / count
     (verified bit-identical to DataFrame.mean(axis=0, skipna=True) for
     float64 input without NaN; with NaN or non-float64 dtypes the module
     falls back to the identical pandas statements).
  4. Output frame: rows in population order (empty-match populations are
     skipped with the same warning), columns = expression.columns —
     identical to the upstream transpose construction.

Cache invalidation: mcpcounter_fast.invalidate_cache().
"""
import numpy as np
import pandas as pd
from importlib.resources import files

__all__ = ["MCPcounter_estimate", "invalidate_cache"]

_marker_cache = {}


def invalidate_cache():
    _marker_cache.clear()


def _import_original():
    from iobrpy.workflow import mcpcounter  # noqa
    return mcpcounter


def _load_markers(features_type: str):
    key = ("mcp_data.pkl", features_type)
    ent = _marker_cache.get(key)
    if ent is not None:
        return ent

    resource_path = files('iobrpy.resources').joinpath('mcp_data.pkl')
    data = pd.read_pickle(resource_path)
    probesets = data.get('probesets')
    genes_df = data.get('genes')
    if probesets is None or genes_df is None:
        raise FileNotFoundError(
            f"Pickle file must contain 'probesets' and 'genes' DataFrames: {resource_path}")

    if features_type == 'affy133P2_probesets':
        sig_df = probesets.copy()
        id_col = sig_df.columns[0]
        pop_col = sig_df.columns[1]
    else:
        sig_df = genes_df.copy()
        mapping = {
            'HUGO_symbols': ('HUGO symbols', 'Cell population'),
            'ENTREZ_ID': ('ENTREZID', 'Cell population'),
            'ENSEMBL_ID': ('ENSEMBL ID', 'Cell population')
        }
        if features_type not in mapping:
            raise KeyError(f"Unknown features_type: {features_type}")
        id_col, pop_col = mapping[features_type]

    if id_col.lower() != 'probeid':
        sig_df[id_col] = sig_df[id_col].str.upper().str.strip()

    ent = (sig_df, id_col, pop_col)
    _marker_cache[key] = ent
    return ent


def MCPcounter_estimate(expression, features_type):
    """Estimate cell population abundance from expression data."""
    sig_df, id_col, pop_col = _load_markers(features_type)

    use_numpy = (
        isinstance(expression, pd.DataFrame)
        and all(dt == np.dtype("float64") for dt in expression.dtypes)
    )
    if not use_numpy:
        return _import_original().MCPcounter_estimate(
            expression=expression, features_type=features_type)

    vals = expression.to_numpy(dtype=np.float64, copy=False)
    has_nan = bool(np.isnan(vals).any())

    # valid marker rows (upstream: sig_df[sig_df[id_col].isin(expression.index)])
    marker_pos = expression.index.get_indexer(sig_df[id_col].to_numpy())
    ok = marker_pos >= 0
    sig_ids = sig_df[id_col].to_numpy()[ok]
    pops_arr = sig_df[pop_col].to_numpy()[ok]
    if len(sig_ids) == 0:
        raise ValueError("No common genes found. Check feature type and identifiers.")

    pop_list = list(pd.unique(pops_arr))

    rows = []
    row_names = []
    for pop in pop_list:
        genes = sig_ids[pops_arr == pop].tolist()
        # exact upstream call: defines both membership and row ORDER
        common = expression.index.intersection(genes)
        print(f"Population '{pop}': {len(common)} marker genes matched.")
        if not len(common):
            continue
        gpos = expression.index.get_indexer(common)
        block = vals[gpos]
        if has_nan:
            # identical pandas statements on an F-order copy (pandas' blocks
            # are column-major; a C-order input would flip the reduction order)
            block_mean = pd.DataFrame(np.asfortranarray(block)).mean(axis=0, skipna=True).to_numpy()
        else:
            # pandas' blocks are (ncols, nrows) C-contiguous, i.e. column-major:
            # its sum(axis=0) reduces the contiguous axis pairwise. Summing an
            # F-order copy reproduces that order bit-for-bit (verified).
            block_mean = np.asfortranarray(block).sum(axis=0) / block.shape[0]
        rows.append(block_mean)
        row_names.append(pop)

    out = pd.DataFrame(np.asarray(rows), index=pd.Index(row_names),
                       columns=expression.columns)
    return out
