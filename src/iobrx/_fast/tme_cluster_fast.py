"""tme_cluster_fast: bit-exact Rust-accelerated port of the
iobrpy.workflow.tme_cluster CLI pipeline (self-implemented Hartigan-Wong-style
k-means with NbClust-like multiple starts + Krzanowski-Lai best-k selection).

Bit-exactness strategy (gated byte-for-byte against
``python -m iobrpy.main tme_cluster`` on the frozen campaign inputs:
375 samples x 235 TME genes, sha256 8684ad4e…, and 10 samples x 22
CIBERSORT weights, sha256 079ef917…):

  1. Preprocessing (ID-column split, ``--features``/``--pattern`` column
     selection, ``pd.to_numeric(errors='coerce')``, z-score with
     ``std(ddof=1)``, ``count2tpm.feature_manipulation`` filtering) are the
     ORIGINAL pandas statements run verbatim on the in-memory frame, so the
     float bits, the surviving column set and their order are identical.
  2. ALL randomness stays in Python: the original draws its initial centres
     with ``np.random.RandomState(seed).choice(n, k, replace=False)`` from a
     stream shared by the whole main-k loop (neighbor k's get their own
     ``RandomState(seed + 9973*k)``). Here the ``nstart`` centre matrices of
     one k are drawn up front in the exact original call order — stream
     identical, because the draws are the only RNG calls the original makes.
  3. The deterministic core runs in Rust (``_rust.tme_kmeans_best``): initial
     nearest-centre assignment, the point-by-point Hartigan move loop
     (original traversal order, strict ``delta > tol``, first-best-j tie
     rule, incremental sums/counts incl. the stale-residual centres of
     emptied clusters), empty-cluster repair reassigning the farthest point
     — iterating the missing clusters in CPython ``set(range(k)) -
     set(unique)`` slot order, reproduced and verified against the live
     interpreter — and ``compute_withinss``. numpy's reduction orders are
     reproduced exactly (pairwise summation with 8192-element nditer
     chunking for flat/inner sums, sequential row accumulation for
     ``.mean(axis=0)``, first-extremum argmin/argmax with numpy's NaN
     rules), all empirically pinned on numpy 2.2.6.
  4. nstart selection keeps the FIRST strict minimum of withinss
     (``w < best_w``), evaluated in start order (parallel start evaluation
     is result-independent; selection stays sequential).
  5. KL index: ``_rust.tme_kl_index`` evaluates the original expression
     (libm ``powf`` == CPython ``**``, verified bit-equal on the gate W
     values); ``best_k = max(kl_scores, key=kl_scores.get)`` (first maximum
     in ascending-k insertion order) with the ``min_nc`` fallback when no KL
     is computable — both verbatim.
  6. Output frame construction (``ID``, ``cluster`` = ``TME{label+1}``, then
     the scaled feature columns via ``pd.concat([out,
     data.reset_index(drop=True)], axis=1)``) is the original code, so
     ``out.to_csv(path, sep=sep_out, index=False)`` reproduces the CLI's
     output file byte-for-byte. The CLI's file writing, tqdm bar and the
     IOBRpy banner are stdout/decoration and stay out of the API.

``tme_cluster_reference`` is the backend='python' path AND the dev-time
oracle: it runs the ORIGINAL module-level functions
(``iobrpy.workflow.tme_cluster.best_hartigan_run`` and friends) over the same
in-memory preprocessing.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = ["tme_cluster_fast", "tme_cluster_reference"]


def _preprocess(df, id, features, pattern, scale, print_result):
    """The original main()'s parsing/scaling block, verbatim on a frame.

    Returns (ids, data) with ``data`` the (possibly z-scored and filtered)
    feature frame whose ``.values`` is the clustering matrix X.
    """
    from iobrpy.workflow.count2tpm import feature_manipulation

    if id and id in df.columns:
        ids = df[id].astype(str)
        df = df.drop(columns=[id])
    else:
        ids = df.iloc[:, 0].astype(str)
        df = df.drop(df.columns[0], axis=1)

    if features:
        m = re.match(r'^(\d+):(\d+)$', features)
        cols = df.columns[int(m.group(1)) - 1:int(m.group(2))]
    elif pattern:
        cols = [c for c in df.columns if re.search(pattern, c)]
    else:
        cols = df.columns

    data = df[cols].apply(pd.to_numeric, errors='coerce')
    if scale:
        data = (data - data.mean()) / data.std(ddof=1)
        valid_features = feature_manipulation(
            data, feature=list(data.columns), print_result=print_result)
        data = data[valid_features]
    if data.empty:
        raise ValueError(
            "No valid features remain after filtering. Check input data and parameters.")
    return ids, data


def _finalize(ids, data, kl_scores, labels_by_k, min_nc, print_result):
    """The original main()'s tail: best-k selection, labelling and the
    output-frame concat, verbatim."""
    if print_result and kl_scores:
        for k in sorted(kl_scores):
            print(f"k={k} KL={kl_scores[k]:.4f}")

    best_k = max(kl_scores, key=kl_scores.get) if kl_scores else min_nc
    if print_result:
        print(f"Best k by KL: {best_k}")

    labels = labels_by_k[best_k]
    clusters = [f"TME{l+1}" for l in labels]

    out = pd.DataFrame({'ID': ids, 'cluster': clusters})
    out = pd.concat([out, data.reset_index(drop=True)], axis=1)
    if print_result:
        counts = out['cluster'].value_counts().reset_index()
        counts.columns = ['cluster', 'count']
        print(counts.to_string(index=False))
    return out


def tme_cluster_reference(df, id=None, features=None, pattern=None, scale=True,
                          min_nc=2, max_nc=6, nstart=10, max_iter=10, tol=1e-4,
                          seed=123, print_result=False):
    """backend='python' path: the ORIGINAL iobrpy functions, in memory.

    Same statements and same call order as ``tme_cluster.main()``, with the
    file I/O, tqdm and banner removed; the numeric path is the untouched
    upstream code (imported, not copied).
    """
    from iobrpy.workflow.tme_cluster import best_hartigan_run

    ids, data = _preprocess(df, id, features, pattern, scale, print_result)
    X = data.values
    n, p = X.shape

    rng = np.random.RandomState(seed)
    kl_scores = {}
    labels_by_k = {}
    withinss_by_k = {}

    main_k_values = list(range(min_nc, max_nc + 1))
    for k in main_k_values:
        labels_k, _, w_k = best_hartigan_run(X, k, nstart, max_iter, tol, rng)
        labels_by_k[k] = labels_k
        withinss_by_k[k] = w_k

    needed_neighbors = set()
    for k in main_k_values:
        if k - 1 >= 1:
            needed_neighbors.add(k - 1)
        needed_neighbors.add(k + 1)
    neighbor_ks = sorted(needed_neighbors - withinss_by_k.keys())
    for k in neighbor_ks:
        rng_neighbor = np.random.RandomState(seed + 9973 * k)
        _, _, w_k = best_hartigan_run(X, k, nstart, max_iter, tol, rng_neighbor)
        withinss_by_k[k] = w_k

    for k in main_k_values:
        prev_k, next_k = k - 1, k + 1
        if prev_k < 1 or next_k not in withinss_by_k:
            continue
        W_prev = withinss_by_k[prev_k]
        W_k = withinss_by_k[k]
        W_next = withinss_by_k[next_k]
        num = abs((k - 1) ** (2 / p) * W_prev - k ** (2 / p) * W_k)
        denom = abs(k ** (2 / p) * W_k - (k + 1) ** (2 / p) * W_next)
        kl_scores[k] = num / (denom if denom > 0 else 1e-8)

    return _finalize(ids, data, kl_scores, labels_by_k, min_nc, print_result)


def tme_cluster_fast(df, id=None, features=None, pattern=None, scale=True,
                     min_nc=2, max_nc=6, nstart=10, max_iter=10, tol=1e-4,
                     seed=123, print_result=False, n_threads=1):
    """Rust-core path; see the module docstring for the bit-exactness
    contract. ``df`` is the parsed input table (first column or ``id``
    column = sample IDs), exactly what the CLI reads from its input file.
    """
    import iobrx._rust as _ir

    ids, data = _preprocess(df, id, features, pattern, scale, print_result)
    X = data.values
    n, p = X.shape
    # int64 (scale=False on integer input) casts exactly; float64 passes
    # through; .values of a consolidated frame is already C-contiguous.
    Xc = np.ascontiguousarray(X, dtype=np.float64)

    rng = np.random.RandomState(seed)
    kl_scores = {}
    labels_by_k = {}
    withinss_by_k = {}

    def best_run(k, k_rng):
        # nstart centre draws from the SAME RandomState stream in the SAME
        # order as the original's interleaved draw/run loop (the draws are
        # the only RNG calls, so batching them per k is stream-identical).
        starts = max(int(nstart), 1)
        inits = np.empty((starts, k, p), dtype=np.float64)
        for t in range(starts):
            indices = k_rng.choice(n, size=k, replace=False)
            inits[t] = Xc[indices]
        labels, w = _ir.tme_kmeans_best(
            Xc, inits, max_iter=int(max_iter), tol=float(tol),
            n_threads=int(n_threads))
        return labels, w

    main_k_values = list(range(min_nc, max_nc + 1))
    for k in main_k_values:
        labels_k, w_k = best_run(k, rng)
        labels_by_k[k] = labels_k
        withinss_by_k[k] = w_k

    needed_neighbors = set()
    for k in main_k_values:
        if k - 1 >= 1:
            needed_neighbors.add(k - 1)
        needed_neighbors.add(k + 1)
    neighbor_ks = sorted(needed_neighbors - withinss_by_k.keys())
    for k in neighbor_ks:
        rng_neighbor = np.random.RandomState(seed + 9973 * k)
        _, w_k = best_run(k, rng_neighbor)
        withinss_by_k[k] = w_k

    for k in main_k_values:
        prev_k, next_k = k - 1, k + 1
        if prev_k < 1 or next_k not in withinss_by_k:
            continue
        kl_scores[k] = _ir.tme_kl_index(
            int(k), int(p),
            float(withinss_by_k[prev_k]), float(withinss_by_k[k]),
            float(withinss_by_k[next_k]))

    return _finalize(ids, data, kl_scores, labels_by_k, min_nc, print_result)
