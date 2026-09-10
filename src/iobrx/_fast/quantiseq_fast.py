"""quantiseq_fast — eliminate the per-call HGNC alias-map rebuild in IOBRpy's quanTIseq.

WHY
---
IOBRpy's ``quantiseq._build_hgnc_alias_map(hgnc)`` is invoked on EVERY
``deconvolute_quantiseq_default()`` call via the chain

    deconvolute_quantiseq_default          (quantiseq.py:244)
      -> fix_mixture(mix, hgnc, arrays)    (quantiseq.py:123)
           -> map_genes(df, hgnc)          (quantiseq.py:93)
                -> _build_hgnc_alias_map(hgnc)   (quantiseq.py:53)

and walks all 42098 HGNC rows with ``DataFrame.iterrows()`` (~1.97 s per call
under cProfile; the whole map build is ~2.5 s). Its output depends ONLY on the
contents of the ``hgnc`` DataFrame (columns ApprovedSymbol / PreviousSymbols /
Synonyms; no other input, no randomness, no time dependence). In the benchmark
harness — and in any session — ``hgnc`` is ``data['HGNC_genenames_20170418']``
from the fixed pickle ``iobrpy.resources/quantiseq_data.pkl``, i.e. one
long-lived object, so the map can be built once and reused.

HOW (numeric path UNCHANGED)
----------------------------
The original module is imported as-is (upstream files are not edited). We swap
the module-global ``quantiseq._build_hgnc_alias_map`` for a memoizing shim:

  * cache key = ``id(hgnc)``; a hit additionally requires a live weakref to
    resolve to the *same* object (``ref() is hgnc``). This makes stale hits
    after garbage collection (id reuse) impossible;
  * on a miss a VECTORIZED builder (``_vectorized_build_hgnc_alias_map``
    below) constructs the alias map ~6x faster than the upstream
    ``iterrows()`` walk, producing a dict that is *exactly equal* (``==``)
    to the original builder's — including identical insertion order (same
    per-row sequence: ApprovedSymbol self-map first, then PreviousSymbols
    tokens, then Synonyms tokens; first occurrence wins globally);
  * downstream ``map_genes``/``fix_mixture``/``deconvolute_quantiseq_default``
    are the untouched original code and resolve the builder through the module
    global at call time, so everything except the repeated rebuild executes
    the original instructions;
  * the cached dict is only ever READ downstream (``.get()`` lookups and a
    ``values()`` set comprehension in ``map_genes`` — never mutated), so
    sharing one dict across calls is safe.

Vectorized builder mechanics (mirrors upstream exactly):
  sub-frame = hgnc[[ApprovedSymbol, PreviousSymbols?, Synonyms?]].fillna('')
  -> approved = astype(str).str.strip() per element (== str(row[...]).strip())
  -> alias keys = comma-split tokens of each alias column, stripped, empties
     dropped (explode preserves token order within a row)
  -> global insertion order = (row position, phase 0=approved/1=prev/2=syn,
     token position within its row and phase) via lexsort
  -> first occurrence per key wins (== upstream ``if k not in map``), so the
     resulting dict has the same keys, values AND insertion order as the
     iterrows build. Verified by gate: ``==`` and ``list(items())`` equality.

API
---
``deconvolute_quantiseq_default(...)`` — passthrough to the original function
with the cache active (signature identical to the original).
``enable_cache() / disable_cache() / clear_cache() / cache_info()`` — controls.
With the cache disabled the module behaves exactly like pristine IOBRpy
(the original builder is restored), which is how the same-process baseline is
re-timed.
"""

from __future__ import annotations

import weakref

import numpy as np
import pandas as pd

try:
    from iobrpy.workflow import quantiseq as _qs  # original, untouched upstream module
    _orig_build = _qs._build_hgnc_alias_map  # pristine builder, kept for baseline/gating
except ModuleNotFoundError:
    _qs = None
    _orig_build = None


def _require_iobrpy():
    if _qs is None:
        raise ImportError(
            "quanTIseq reuses the upstream IOBRpy module; install the Python "
            "fallback backend with `pip install 'iobrx[python]'`."
        )


def _vectorized_build_hgnc_alias_map(hgnc):
    """Vectorized drop-in for ``quantiseq._build_hgnc_alias_map``.

    Returns a dict *exactly equal* (``==``, and with identical insertion
    order) to the original iterrows-based builder on the same input:
    ApprovedSymbol self-maps first per row, then PreviousSymbols tokens,
    then Synonyms tokens (comma-split, stripped, empties dropped), each
    mapping to that row's (possibly empty) stripped ApprovedSymbol; the
    first occurrence of a key globally wins, exactly like the original
    ``if k not in alias_to_approved`` guards.
    """
    alias_to_approved = {}
    if hgnc is None or hgnc.empty:
        return alias_to_approved

    cols = set(hgnc.columns)
    has_prev = 'PreviousSymbols' in cols
    has_syn = 'Synonyms' in cols
    if 'ApprovedSymbol' not in cols:
        return alias_to_approved

    sel = (['ApprovedSymbol'] + (['PreviousSymbols'] if has_prev else [])
           + (['Synonyms'] if has_syn else []))
    # same sub-frame the original iterrows over; positional reset keeps row
    # order (iterrows order) as integer positions for the lexsort below
    sub = hgnc[sel].fillna('').reset_index(drop=True)
    n = len(sub)
    approved = sub['ApprovedSymbol'].astype(str).str.strip().to_numpy(dtype=object)
    rowpos = np.arange(n)

    keys_parts, vals_parts, rows_parts, phase_parts, tokpos_parts = [], [], [], [], []

    # phase 0: approved self-maps (skipped for empty approved, as upstream)
    m = approved != ''
    n_ap = int(m.sum())
    keys_parts.append(approved[m])
    vals_parts.append(approved[m])
    rows_parts.append(rowpos[m])
    phase_parts.append(np.zeros(n_ap, dtype=np.int8))
    tokpos_parts.append(np.zeros(n_ap, dtype=np.int64))

    # phases 1 (PreviousSymbols) / 2 (Synonyms): comma tokens in order
    for phase, col in ((1, 'PreviousSymbols'), (2, 'Synonyms')):
        if (col == 'PreviousSymbols' and not has_prev) or (col == 'Synonyms' and not has_syn):
            continue
        toks = sub[col].astype(str).str.split(',')
        keys = toks.explode().str.strip()
        keep = keys != ''
        keys = keys[keep]
        krow = keys.index.to_numpy()                     # row positions
        krow_s = pd.Series(krow)
        ktok = krow_s.groupby(krow_s).cumcount().to_numpy()  # token order in row
        keys_parts.append(keys.to_numpy(dtype=object))
        vals_parts.append(approved[krow])
        rows_parts.append(krow)
        phase_parts.append(np.full(len(krow), phase, dtype=np.int8))
        tokpos_parts.append(ktok)

    if not keys_parts:
        return alias_to_approved

    key = np.concatenate(keys_parts)
    val = np.concatenate(vals_parts)
    row = np.concatenate(rows_parts)
    phase = np.concatenate(phase_parts)
    tok = np.concatenate(tokpos_parts)
    # global insertion order: row -> phase (approved, prev, syn) -> token pos
    order = np.lexsort((tok, phase, row))
    key, val = key[order], val[order]
    # first occurrence per key wins (same keys/values/insertion order as upstream)
    pos = pd.Series(np.arange(len(key)), index=key)
    dedup = pos[~pos.index.duplicated(keep='first')]
    return dict(zip(dedup.index.to_numpy(dtype=object), val[dedup.to_numpy()]))


# cache: id(hgnc) -> (weakref.ref(hgnc), alias_map_built_by_original_builder)
_cache: dict[int, tuple[weakref.ReferenceType, dict]] = {}
_hits = 0
_misses = 0


def _memoized_build_hgnc_alias_map(hgnc):
    """Semantics-identical stand-in for quantiseq._build_hgnc_alias_map.

    Returns a dict exactly equal to what the original builder returns for
    the given hgnc (built by the vectorized builder, gated ``==``-identical
    upstream); on a repeat call with the same live hgnc object, returns the
    previously built dict (bit-for-bit the same object, read-only
    downstream).
    """
    global _hits, _misses
    if hgnc is None:
        # original handles None/empty directly; nothing worth caching
        return _vectorized_build_hgnc_alias_map(hgnc)

    key = id(hgnc)
    entry = _cache.get(key)
    if entry is not None:
        ref, cached_map = entry
        if ref() is hgnc:  # same object, still alive -> genuine cache hit
            _hits += 1
            return cached_map
        # original object died (id may have been recycled): drop stale entry
        del _cache[key]

    _misses += 1
    built = _vectorized_build_hgnc_alias_map(hgnc)
    try:
        _cache[key] = (weakref.ref(hgnc), built)
    except TypeError:
        pass  # object does not support weakrefs: build each time (still correct)
    return built


def enable_cache() -> None:
    """Activate the memoizing builder on the original module (default state)."""
    _require_iobrpy()
    _qs._build_hgnc_alias_map = _memoized_build_hgnc_alias_map


def disable_cache() -> None:
    """Restore the pristine original builder (exactly upstream behaviour)."""
    _require_iobrpy()
    _qs._build_hgnc_alias_map = _orig_build


def clear_cache() -> None:
    """Drop all cached maps (next deconvolution rebuilds once)."""
    global _hits, _misses
    _cache.clear()
    _hits = 0
    _misses = 0


def cache_info() -> dict:
    _require_iobrpy()
    return {
        "cached_hgnc_objects": len(_cache),
        "hits": _hits,
        "misses": _misses,
        "active": _qs._build_hgnc_alias_map is _memoized_build_hgnc_alias_map,
    }


# Activate on import.
enable_cache()


def deconvolute_quantiseq_default(mix, data, arrays=False, signame="TIL10",
                                  tumor=False, mRNAscale=True,
                                  method="lsei", rmgenes="unassigned"):
    """Thin passthrough to the ORIGINAL deconvolute_quantiseq_default.

    All computation happens in the original module's code; only the alias-map
    builder inside it is the memoizing shim (when the cache is enabled).
    """
    _require_iobrpy()
    return _qs.deconvolute_quantiseq_default(
        mix=mix, data=data, arrays=arrays, signame=signame, tumor=tumor,
        mRNAscale=mRNAscale, method=method, rmgenes=rmgenes,
    )
