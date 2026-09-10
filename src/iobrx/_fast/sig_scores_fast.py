"""sig_scores_fast: bit-exact, vectorized drop-ins for IOBRpy's
iobrpy.workflow.calculate_sig_score.sig_score_pca / sig_score_zscore.

Identical signatures, identical output DataFrames (column order, values,
dtypes), verified bitwise against the original pipeline and against
iobrx/refs/sig_pca.parquet / sig_zscore.parquet (max abs diff 0.0).

Bit-exactness strategy (all verified against the original on the official
imvigor210_eset x signature_collection data, 58/58 signatures):

  - preprocess_eset / filter_signatures are reused from the original module
    unchanged (run ONCE per call, exactly like the original structure).

  - The per-signature z-score chain
        mat = eset2.loc[valid].T
        mat = mat.sub(mat.mean(axis=0), axis=1)
        mat = mat.div(mat.std(axis=0, ddof=1).replace(0, nan), axis=1).fillna(0)
    is hoisted to ONE full-matrix pass. Per-gene mean/std over samples are
    independent of which signature contains the gene, and pandas column
    reductions depend only on that column's values and memory layout:
      * the per-gene MEAN in the original runs on the transposed .loc view
        (samples-x-genes); reproducing it as (eset2.values.T).mean(axis=0)
        on the same view is bitwise identical (numpy strided axis-0
        accumulation == pandas nanmean for float64, verified 58/58);
      * the per-gene STD in the original runs on the *post-subtraction*
        frame, whose values block is Fortran-ordered (pairwise summation);
        reproducing it with the exact same pandas ops
        (mat_full.sub(mu).std(axis=0, ddof=1)) on the full matrix is
        bitwise identical (verified 58/58), including the
        .replace(0, nan) -> fillna(0) semantics.
    sub/div/fillna are elementwise, hence layout- and width-independent.

  - sorted(set(genes) & set(eset2.index)) gene ordering is preserved
    exactly: per-signature column indices are precomputed from one
    gene->row dict; pandas .loc label indexing is replaced by numpy fancy
    indexing (eset2.index is verified unique; for a non-unique index the
    original .loc would pull duplicate rows, so we transparently fall
    back to the original implementation in that case).

  - PCA(n_components=1, svd_solver='full', random_state=0) is kept
    PER SIGNATURE (no batching - bit-equality of a batched path is NOT
    proven, so it is not used). The sklearn estimator overhead is removed
    by replaying sklearn 1.7.2 PCA._fit_full's exact numeric sequence:
        mean = np.mean(X, axis=0)
        Xc = np.array(X, copy=True, order='K'); Xc -= mean
        U, S, Vt = scipy.linalg.svd(Xc, full_matrices=False)
        svd_flip(U, Vt, u_based_decision=False)   # Vt-row-sign fix
        score = U[:, 0] * S[0]                    # fit_transform truncation
    The PCA input is materialized in the SAME memory layout the original
    mat.values has (probed once per call); scipy.linalg.svd results differ
    in last bits between C- and F-ordered inputs, so layout is matched,
    not assumed.

  - mean_expr (per-sample mean over the signature's raw genes) is computed
    as np.asfortranarray(X[rows]).mean(axis=0), reproducing pandas'
    pairwise summation over the contiguous gene axis (verified 58/58;
    layout probed once per call, C-layout fallback kept symmetric).

  - Threading: measured SLOWER than serial for these workloads (58 SVDs of
    <= 348x166; verified bit-identical at 8/32 threads but wall time worse
    due to pool dispatch + BLAS contention), so execution stays serial.
    parallel_size is accepted for signature compatibility; the original's
    joblib 'processes' path is intentionally not replicated (it cannot
    change results, only scheduling).
"""
import numpy as np
import pandas as pd
import scipy.linalg as _sla

try:
    from iobrpy.workflow.calculate_sig_score import (
        preprocess_eset,
        filter_signatures,
        sig_score_pca as _sig_score_pca_original,
        sig_score_zscore as _sig_score_zscore_original,
    )
except ModuleNotFoundError:
    preprocess_eset = filter_signatures = None
    _sig_score_pca_original = _sig_score_zscore_original = None


def _require_upstream():
    if preprocess_eset is None:
        raise ImportError(
            "signature scoring reuses upstream IOBRpy helpers; install the Python "
            "fallback backend with `pip install 'iobrx[python]'`."
        )


def _pc1_full_lean(X):
    """Bit-exact replay of
    PCA(n_components=1, svd_solver='full', random_state=0).fit_transform(X)[:, 0]
    (sklearn 1.7.2 PCA._fit_full + fit_transform + svd_flip, numeric ops only).
    X: (n_samples, n_features) float64; memory layout is preserved."""
    mean = np.mean(X, axis=0)                                   # self.mean_ = xp.mean(X, axis=0)
    Xc = np.array(X, dtype=np.float64, copy=True, order="K")    # xp.asarray(X, copy=True)
    Xc -= mean                                                  # X_centered -= self.mean_
    U, S, Vt = _sla.svd(Xc, full_matrices=False)                # scipy.linalg.svd
    # svd_flip(U, Vt, u_based_decision=False)
    max_abs_v_rows = np.argmax(np.abs(Vt), axis=1)
    shift = np.arange(Vt.shape[0])
    indices = max_abs_v_rows + shift * Vt.shape[1]
    signs = np.sign(np.take(np.reshape(Vt, (-1,)), indices, axis=0))
    U *= signs[np.newaxis, :]
    # fit_transform: U[:, :1] *= S[:1]  ->  element-wise U[:, 0] * S[0]
    return U[:, 0] * S[0]


def _standardize_full(eset2):
    """One full-matrix pass of the per-signature z-score chain (bitwise equal
    per gene; see module docstring). Returns the samples-x-genes frame."""
    mat_full = eset2.T                                          # samples x genes
    mu = mat_full.mean(axis=0)                                  # per-gene mean over samples
    centered = mat_full.sub(mu, axis=1)
    sd = centered.std(axis=0, ddof=1).replace(0, np.nan)        # per-gene sd, ddof=1
    return centered.div(sd, axis=1).fillna(0.0)


def _probe_layouts(eset2, sigs, gene_pos):
    """Run the original per-signature chain ONCE on the first signature to
    observe the memory layouts the original feeds to np.mean / scipy.linalg.svd
    (they depend on pandas' internal block layout decisions)."""
    name, genes = next(iter(sigs.items()))
    valid = sorted(set(genes) & set(eset2.index))
    tmp = eset2.loc[valid]                                      # genes x samples
    mat = tmp.T
    mat = mat.sub(mat.mean(axis=0), axis=1)
    mat = mat.div(mat.std(axis=0, ddof=1).replace(0, np.nan), axis=1).fillna(0.0)
    cols = np.array([gene_pos[g] for g in valid], dtype=np.intp)
    return (
        bool(tmp.values.flags["F_CONTIGUOUS"]),   # raw slice layout (mean_expr)
        bool(mat.values.flags["F_CONTIGUOUS"]),   # standardized layout (PCA input)
        cols,
    )


def _column_indices(sigs, gene_pos):
    """Precompute per-signature sorted(set(genes) & index) column positions."""
    out = {}
    for name, genes in sigs.items():
        valid = sorted(g for g in set(genes) if g in gene_pos)
        out[name] = (np.array([gene_pos[g] for g in valid], dtype=np.intp), len(valid))
    return out


def _attach_tme_contrasts(pdata, sigs):
    """Same TMEscore derived columns as the original."""
    if {"TMEscoreA_CIR", "TMEscoreB_CIR"}.issubset(sigs):
        pdata["TMEscore_CIR"] = pdata["TMEscoreA_CIR"] - pdata["TMEscoreB_CIR"]
    if {"TMEscoreA_plus", "TMEscoreB_plus"}.issubset(sigs):
        pdata["TMEscore_plus"] = pdata["TMEscoreA_plus"] - pdata["TMEscoreB_plus"]
    return pdata


def sig_score_pca_fast(eset, sig_dict, mini_gene_count, adjust_eset, parallel_size=1):
    """Bit-exact drop-in for calculate_sig_score.sig_score_pca (parallel_size
    accepted for signature compatibility; execution is serial - see module
    docstring)."""
    _require_upstream()
    pdata = pd.DataFrame({"ID": eset.columns})
    eset2 = preprocess_eset(eset, adjust_eset)

    min_size = max(mini_gene_count, 2)
    sigs = filter_signatures(sig_dict, eset2, min_size)

    if not sigs:
        return pdata

    # .loc label semantics with duplicate index rows cannot be reproduced by a
    # position map; defer to the original for guaranteed identity.
    if not eset2.index.is_unique:
        return _sig_score_pca_original(eset, sig_dict, mini_gene_count, adjust_eset,
                                       parallel_size)

    gene_pos = {g: i for i, g in enumerate(eset2.index)}
    n_samples = eset2.shape[1]

    raw_f, z_f, _ = _probe_layouts(eset2, sigs, gene_pos)

    # one full-matrix standardization shared by all signatures
    Zv = _standardize_full(eset2).values                        # samples x genes
    Xv = eset2.values                                           # genes x samples

    for name, genes in sigs.items():
        valid = sorted(g for g in set(genes) if g in gene_pos)
        if len(valid) < 2:                                      # original fallback
            pdata[name] = np.zeros(n_samples, dtype=float)
            continue
        cols = np.array([gene_pos[g] for g in valid], dtype=np.intp)
        sub = Zv[:, cols]
        sub = np.asfortranarray(sub) if z_f else np.ascontiguousarray(sub)
        pc1 = _pc1_full_lean(sub)
        rows = Xv[cols]
        mean_expr = np.asfortranarray(rows).mean(axis=0) if raw_f else rows.mean(axis=0)
        corr = np.corrcoef(pc1, mean_expr)[0, 1]
        direction = np.sign(corr) if not np.isnan(corr) else 1.0
        pdata[name] = pc1 * direction

    return _attach_tme_contrasts(pdata, sigs)


def sig_score_zscore_fast(eset, sig_dict, mini_gene_count, adjust_eset, parallel_size=1):
    """Bit-exact drop-in for calculate_sig_score.sig_score_zscore."""
    _require_upstream()
    pdata = pd.DataFrame({"ID": eset.columns})
    eset2 = preprocess_eset(eset, adjust_eset)

    min_size = max(mini_gene_count, 2)
    sigs = filter_signatures(sig_dict, eset2, min_size)

    if not sigs:
        return pdata

    if not eset2.index.is_unique:
        return _sig_score_zscore_original(eset, sig_dict, mini_gene_count, adjust_eset,
                                          parallel_size)

    gene_pos = {g: i for i, g in enumerate(eset2.index)}
    n_samples = eset2.shape[1]
    Xv = eset2.values                                           # genes x samples

    # raw-slice layout probe (same as pca path; one cheap chain on first sig)
    raw_f, _, _ = _probe_layouts(eset2, sigs, gene_pos)

    for name, genes in sigs.items():
        valid = sorted(g for g in set(genes) if g in gene_pos)
        if len(valid) == 0:                                     # original fallback
            pdata[name] = np.zeros(n_samples, dtype=float)
            continue
        rows = Xv[np.array([gene_pos[g] for g in valid], dtype=np.intp)]
        pdata[name] = np.asfortranarray(rows).mean(axis=0) if raw_f else rows.mean(axis=0)

    return _attach_tme_contrasts(pdata, sigs)
