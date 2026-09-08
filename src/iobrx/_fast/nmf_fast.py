"""nmf_fast: lazy-import, API-shaped drop-in for iobrpy.workflow.nmf (CLI step).

Upstream ``nmf.py`` is CLI-only (``main()`` + argparse): read a samples x
features matrix, select a 1-based inclusive column range, optionally apply
log1p / L1-row normalization / a non-negativity shift, scan k in
[kmin, kmax] fitting ``sklearn.decomposition.NMF(n_components=k,
init='nndsvda', random_state=..., max_iter=...)`` with every other sklearn
default (solver='cd', tol=1e-4, shuffle=False), label samples by
``W.argmax(axis=1)``, pick the k with the highest ``silhouette_score(W,
labels)`` (ties -> smaller k), refit once with best_k, and write
``clusters.csv`` / ``top_features_per_cluster.csv`` / ``pca_plot.png``.

Bit-exactness strategy (gate: frozen cibersort_stad10.csv x the official
docs command ``--kmin 2 --kmax 10 --features 1:22 --max-iter 10000
--skip_k_2 --random-state 42``; all three output files byte-identical,
sha256 equal to the recorded golden):

  1. Every numeric and I/O statement is copied VERBATIM from upstream (the
     functions below are ast-extracted from nmf.py 0.2.0): the same sklearn
     NMF / PCA / silhouette_score / normalize calls with the same arguments
     (same library versions -> identical floating-point bits), the same
     preprocessing order, the same argmax / tie-break expressions, the same
     pandas ``to_csv`` writers and the same plotting helpers. Only
     documented, non-numeric deltas exist: matplotlib imports moved inside
     the two plotting helpers (module top-level in the original), print
     statements in the k-scan gated behind ``verbose``, and ``main()``'s
     argparse/sys.exit shell replaced by :func:`nmf_cluster`.
  2. Threading is deliberately left AMBIENT (no threadpoolctl, no env
     changes): OpenMP/BLAS thread counts can change reduction order inside
     the CD solver, so pinning them could break bit-exactness against the
     original CLI (which also runs ambient). ``n_threads`` is validated for
     API uniformity but does not steer the NMF fit.
  3. Acceleration is import-side: upstream pays the full ``iobrpy.main``
     import chain (~1.4-2.2 s cold start) plus an unconditional
     ``matplotlib.pyplot`` import even when nobody looks at the plot; this
     module is imported lazily by ``iobrx.nmf`` and pulls matplotlib in
     only when ``plot=True``.
  4. BUG-COMPAT (upstream quirks, reproduced on purpose):
     - ``top_features_per_cluster.csv`` is written BEFORE the output
       directory is created and the failure is swallowed -> with a
       non-existent ``outdir`` the file is silently missing while
       clusters.csv / pca_plot.png are written normally.
     - ``read_matrix`` falls back to a literal four-space ``sep`` on any
       parse error, so real TAB-separated files do not parse.
     - ``shift`` is ignored unless ``min(X) < 0``; negative data without a
       shift raises ValueError (same message as upstream).
     - ``random_state`` does not influence results (nndsva init is
       deterministic and shuffle=False), verified seed-invariant upstream.
     - ``kmax >= n_samples`` is silently clamped to ``n_samples - 1``.

API deviations from the CLI (control flow only, zero numeric effect):
  * ``sys.exit(1)`` paths raise instead: ``features`` parse errors and
    "no valid k in range" become ValueError / RuntimeError (same message
    text); ``ensure_nonneg``'s ValueError propagates unchanged.
  * Plotting is opt-in (``plot=False`` default); the CLI always draws.
  * Progress prints and the IOBRpy banner are gated behind ``verbose``
    (the CLI always prints them).
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize as _sk_normalize

from iobrx._threads import resolve_threads

__all__ = ["nmf_cluster", "read_matrix", "parse_features_arg", "ensure_nonneg",
           "fit_nmf", "try_k_range_nmf_argmax", "save_outputs"]


def read_matrix(path):
    # Try comma first, then tab
    try:
        df = pd.read_csv(path, index_col=0)
        return df
    except Exception:
        df = pd.read_csv(path, sep='    ', index_col=0)
        return df

def _sqrt_chi2_quantile(prob: float) -> float:
    """
    Return sqrt(chi2.ppf(prob, df=2)) without SciPy.
    Values are accurate enough for plotting; we snap to nearest key.
    """
    table = {0.80: 1.794, 0.85: 1.948, 0.90: 2.146, 0.95: 2.447, 0.99: 3.034}
    # clamp and pick nearest
    prob = min(max(prob, 0.80), 0.99)
    key = min(table.keys(), key=lambda k: abs(k - prob))
    return table[key]

def _confidence_ellipse(points_2d, ax, q=0.90, method='mcd', trim=0.10, **kwargs):
    """
    Draw a covariance-based confidence ellipse for a 2D cluster.

    Parameters
    ----------
    points_2d : (n,2) ndarray
        Cluster points in PCA space.
    ax : matplotlib Axes
        Target axes.
    q : float
        Coverage probability (e.g., 0.90 -> ~90% ellipse). Smaller -> tighter.
    method : {'mcd','trim','classic'}
        'mcd'   : robust covariance via Minimum Covariance Determinant (if available);
        'trim'  : drop top 'trim' fraction of farthest points (Euclidean) then classic cov;
        'classic': use all points with classic covariance.
    trim : float
        Fraction (0~0.4) of outermost points to drop when method='trim' or MCD fallback.

    Notes
    -----
    Ellipse axes = 2 * sqrt(chi2_quantile(q, df=2)) * sqrt(eigenvalues_of_cov).
    This is robust to outliers when 'mcd' succeeds or after trimming.
    """
    from matplotlib.patches import Ellipse  # lazy (module top-level in the original)
    n = points_2d.shape[0]
    if n < 3:
        return None

    # --- estimate center and covariance robustly ---
    mu, cov = None, None
    used_pts = points_2d

    if method == 'mcd':
        try:
            # robust covariance (will ignore a subset of outliers automatically)
            from sklearn.covariance import MinCovDet
            support_fraction = None if trim <= 0 else max(0.5, 1.0 - float(trim))
            mcd = MinCovDet(support_fraction=support_fraction, random_state=0).fit(points_2d)
            mu = mcd.location_
            cov = mcd.covariance_
        except Exception:
            method = 'trim'  # fallback if sklearn not installed or fails

    if method == 'trim':
        # drop outer 'trim' fraction by Euclidean distance to mean, then recompute
        mu0 = points_2d.mean(axis=0)
        d2 = ((points_2d - mu0) ** 2).sum(axis=1)
        if trim > 0:
            keep = d2 <= np.quantile(d2, 1.0 - float(trim))
            used_pts = points_2d[keep]
        mu = used_pts.mean(axis=0)
        cov = np.cov(used_pts, rowvar=False)

    if method == 'classic':
        mu = points_2d.mean(axis=0)
        cov = np.cov(points_2d, rowvar=False)

    # guard
    if cov is None or not np.all(np.isfinite(cov)):
        return None

    # --- build ellipse from eigen-decomposition ---
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    scale = _sqrt_chi2_quantile(q)
    width, height = 2 * scale * np.sqrt(np.maximum(vals, 1e-12))
    ell = Ellipse(xy=mu, width=width, height=height, angle=theta,
                  fill=False, lw=1.2, **kwargs)
    ax.add_patch(ell)
    return ell

def _plot_pca_scatter(pca_coords, cluster_labels, sample_index, out_path,
                      annotate_mode='centroids',  # 'none' | 'centroids' | 'few' | 'all'
                      topk_per_cluster=3,         # used when annotate_mode='few'
                      draw_ellipses=True):
    import matplotlib.pyplot as plt  # lazy: matplotlib loads only when plotting
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    clusters = {}
    for i, c in enumerate(cluster_labels):
        clusters.setdefault(c, []).append(i)
    for cname, idxs in clusters.items():
        pts = pca_coords[np.array(idxs)]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.55, label=cname)
    centroids = {}
    for cname, idxs in clusters.items():
        pts = pca_coords[np.array(idxs)]
        ctr = pts.mean(axis=0)
        centroids[cname] = ctr
        if annotate_mode in ('centroids', 'all', 'few'):
            ax.text(ctr[0], ctr[1], f' {cname}', fontsize=9, weight='bold',
                    va='center', ha='left', zorder=6)
    if draw_ellipses:
        # draw tighter, robust ellipses (less spillover at boundaries)
        for cname, idxs in clusters.items():
            pts = pca_coords[np.array(idxs)]
            _confidence_ellipse(pts, ax, q=0.90, method='mcd', trim=0.10)
    if annotate_mode in ('all', 'few'):
        for cname, idxs in clusters.items():
            pts = pca_coords[np.array(idxs)]
            labels = [sample_index[i] for i in idxs]
            if annotate_mode == 'all':
                choose = range(len(labels))
            else:
                ctr = centroids[cname]
                d2 = ((pts - ctr) ** 2).sum(axis=1)
                choose = np.argsort(d2)[-min(topk_per_cluster, len(labels)):]
            for k in choose:
                ax.text(pts[k, 0], pts[k, 1], f' {labels[k]}', fontsize=7, alpha=0.8)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('Samples in NMF-component space (PCA of W)')
    ax.legend(title='cluster', frameon=False, fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def parse_features_arg(feat_str, n_cols):
    """Parse --features argument. Support 'm-n' or 'm:n' or single 'm' (1-based, inclusive).
       Returns start_idx (inclusive, 0-based), end_idx (exclusive)
    """
    if feat_str is None:
        return 0, n_cols
    feat_str = feat_str.strip()
    sep = None
    if '-' in feat_str:
        sep = '-'
    elif ':' in feat_str:
        sep = ':'

    try:
        if sep:
            parts = feat_str.split(sep)
            if len(parts) != 2:
                raise ValueError
            start = int(parts[0])
            end = int(parts[1])
        else:
            start = int(feat_str)
            end = start
    except Exception:
        raise argparse.ArgumentTypeError("--features format error: use 'm-n' or 'm:n' or single number (1-based)")

    if start < 1 or end < 1 or start > end:
        raise argparse.ArgumentTypeError(f"--features invalid range: {feat_str}")
    start_idx = start - 1
    end_idx = end

    if start_idx >= n_cols:
        raise argparse.ArgumentTypeError(f"--features start column ({start}) exceeds number of columns ({n_cols})")
    if end_idx > n_cols:
        end_idx = n_cols
    return start_idx, end_idx

def ensure_nonneg(X, shift=None):
    minv = X.min()
    if minv < 0:
        if shift is None:
            raise ValueError(f"Input contains negative values (min={minv:.4g}); NMF requires non-negative data. Use --shift to add a constant or preprocess the data.")
        else:
            X = X + abs(minv) + shift
    return X

def fit_nmf(X, n_components, random_state=42, max_iter=1000):
    model = NMF(n_components=n_components, init='nndsvda', random_state=random_state, max_iter=max_iter)
    W = model.fit_transform(X)
    H = model.components_
    rec_err = getattr(model, 'reconstruction_err_', None)
    return model, W, H, rec_err

def try_k_range_nmf_argmax(X, kmin, kmax, random_state=42, max_iter=1000, skip_k_2=False, verbose=True):
    """For each k (optionally skipping k==2 when skip_k_2=True): fit NMF(n_components=k),
    assign labels = argmax(W), compute silhouette score on W.
    Return dict: k -> (silhouette, reconstruction_err)
    """
    n_samples = X.shape[0]
    results = {}
    for k in range(kmin, kmax + 1):
        if k == 2 and skip_k_2:
            if verbose:
                print(f"skip k={k} (skip_k_2=True)")
            continue
        if k < 2:
            if verbose:
                print(f"skip k={k} (k must be >=2)")
            continue
        if k >= n_samples:
            if verbose:
                print(f"skip k={k} (k must be less than number of samples {n_samples})")
            continue
        try:
            model, W, _, rec_err = fit_nmf(X, n_components=k, random_state=random_state, max_iter=max_iter)
            labels = W.argmax(axis=1)
            unique_count = len(np.unique(labels))
            if unique_count <= 1:
                sil = float('-inf')
                if verbose:
                    print(f"k={k}: only {unique_count} cluster(s) found by argmax -> silhouette not defined")
            else:
                sil = silhouette_score(W, labels)
                if rec_err is None:
                    if verbose:
                        print(f"k={k}: silhouette={sil:.4f}")
                else:
                    if verbose:
                        print(f"k={k}: silhouette={sil:.4f}, rec_err={rec_err:.4f}")
            results[k] = (sil, rec_err)
        except Exception as e:
            if verbose:
                print(f"k={k} training failed: {e}")
    return results

def save_outputs(outdir, sample_index, cluster_labels, features_df=None, pca_coords=None):
    """
    Save clustering results and optional PCA plot.

    When `features_df` is provided, write clusters.csv with:
      - column 1: 'sample' (sample IDs)
      - column 2: 'cluster' (cluster labels like 'cluster1', 'cluster2', ...)
      - columns 3+: feature values actually used for clustering
                   (after preprocessing), keeping original feature names.
    Otherwise, keep the old behavior (sample as index).
    """
    os.makedirs(outdir, exist_ok=True)

    # Build the base (sample, cluster) table in the given sample order
    labels_df = pd.DataFrame({'sample': sample_index, 'cluster': cluster_labels})

    if features_df is not None:
        # Reorder features_df to match the provided sample order
        # and drop its index so that 'sample' remains the only ID column.
        feats_ordered = features_df.loc[sample_index]
        out_df = pd.concat([labels_df, feats_ordered.reset_index(drop=True)], axis=1)
        out_df.to_csv(os.path.join(outdir, 'clusters.csv'), index=False)
    else:
        # Fallback to previous format (sample as index)
        labels_df.set_index('sample', inplace=True)
        labels_df.to_csv(os.path.join(outdir, 'clusters.csv'))

    # Optional PCA scatter in W-space (unchanged)
    if pca_coords is not None:
        plot_path = os.path.join(outdir, 'pca_plot.png')
        _plot_pca_scatter(
            pca_coords=pca_coords,
            cluster_labels=cluster_labels,
            sample_index=sample_index,
            out_path=plot_path,
            annotate_mode='none',   # 'none' | 'centroids' | 'few' | 'all'
            topk_per_cluster=3,
            draw_ellipses=True
        )

def _print_banner():
    """Upstream main()'s closing IOBRpy banner, verbatim (lazy import)."""
    from iobrpy.utils.print_colorful_message import print_colorful_message
    print("   ")
    print_colorful_message("#########################################################", "blue")
    print_colorful_message(" IOBRpy: Immuno-Oncology Biological Research using Python ", "cyan")
    print_colorful_message(" If you encounter any issues, please report them at ", "cyan")
    print_colorful_message(" https://github.com/IOBR/IOBRpy/issues ", "cyan")
    print_colorful_message("#########################################################", "blue")
    print(" Author: Haonan Huang, Dongqiang Zeng")
    print(" Email: interlaken@smu.edu.cn ")
    print_colorful_message("#########################################################", "blue")
    print("   ")


def nmf_cluster(data, kmin=2, kmax=8, features=None, log1p=False, normalize=False,
                shift=None, random_state=42, max_iter=1000, skip_k_2=False,
                plot=False, outdir=None, verbose=False, n_threads=None,
                backend="auto"):
    """API reproduction of ``iobrpy.workflow.nmf.main()`` (see module docstring).

    Parameters mirror the CLI flags (``kmin``/``kmax``/``features``/``log1p``/
    ``normalize``/``shift``/``random_state``/``max_iter``/``skip_k_2``) plus
    ``plot`` (opt-in pca_plot.png; the CLI always draws), ``outdir`` (write
    the CLI's output files; None -> in-memory only), ``verbose`` (reproduce
    the CLI's stdout incl. banner), ``n_threads`` (validated, unused — the
    fit stays ambient-threaded for bit-exactness) and ``backend`` (no native
    kernel exists for nmf; 'rust' raises).

    ``data`` is a samples x features DataFrame (index = sample IDs, the CLI
    input file's content) or a path read with the original ``read_matrix``.

    Returns a dict: ``clusters`` / ``top_features`` (DataFrames exactly
    equal to the written CSVs), ``best_k``, ``best_silhouette``,
    ``best_rec_err``, ``k_results`` ({k: (silhouette, rec_err)} incl.
    -inf entries), ``labels`` (0-based argmax), ``W`` / ``H`` / ``model``
    (best-k refit), ``pca_coords`` (2-D PCA of W or None), ``used_features``
    (preprocessed matrix with original index/columns), ``outdir`` (abspath
    or None). CLI ``sys.exit(1)`` paths raise ValueError / RuntimeError.
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError("backend must be 'auto', 'rust', or 'python'")
    if backend == "rust":
        raise RuntimeError(
            "iobrx.nmf is a pure-Python bit-exact port with no native "
            "kernel; use backend='auto' or backend='python'")
    resolve_threads(n_threads)  # validate only; see module docstring item 2
    if plot and outdir is None:
        raise ValueError(
            "plot=True requires outdir=<path> (pca_plot.png is written there)")

    # 1) Read input
    if isinstance(data, (str, os.PathLike)):
        if verbose:
            print('Reading matrix:', data)
        df = read_matrix(data)
    else:
        df = data
    if verbose:
        print('Input shape (n_samples x total_columns):', df.shape)

    # Parse features
    n_cols = df.shape[1]
    try:
        start_idx, end_idx = parse_features_arg(features, n_cols)
    except Exception as e:
        # CLI: print('ERROR parsing --features:', e); sys.exit(1)
        raise ValueError(f"ERROR parsing --features: {e}") from None

    if not (start_idx == 0 and end_idx == n_cols):
        selected_df = df.iloc[:, start_idx:end_idx]
        if verbose:
            print(f"Using column range: {start_idx+1} to {end_idx} (total {selected_df.shape[1]} columns)")
    else:
        selected_df = df
        if verbose:
            print('Using all columns from input')

    # 2) Preprocessing (fixed order, identical operations)
    X = selected_df.values.astype(float)
    if log1p:
        if verbose:
            print('Applying log1p transform')
        X = np.log1p(X)
    if normalize:
        if verbose:
            print('Applying L1 row normalization')
        X = _sk_normalize(X, norm='l1', axis=1)

    # Ensure non-negative (original CLI catches, prints 'ERROR:' and exits 1;
    # the API lets the identically-messaged ValueError propagate)
    X = ensure_nonneg(X, shift=shift)

    # Keep a DataFrame of the actual feature values used for clustering
    # (after all preprocessing such as log1p/normalization/shift).
    used_features_df = pd.DataFrame(
        X,
        index=selected_df.index,          # keep the original sample order/index
        columns=selected_df.columns       # keep the original feature names
    )

    n_samples = X.shape[0]
    if kmax >= n_samples:
        if verbose:
            print(f"Warning: kmax ({kmax}) >= number of samples ({n_samples}). Setting kmax to {n_samples-1}.")
        kmax = n_samples - 1

    # 3) Try k range (NMF argmax clustering)
    if verbose:
        print('Trying k range:', kmin, '->', kmax)
    results = try_k_range_nmf_argmax(
        X, kmin, kmax, random_state=random_state, max_iter=max_iter,
        skip_k_2=skip_k_2, verbose=verbose)
    # Filter out ks that failed or gave -inf silhouette
    valid_results = {k: v for k, v in results.items() if v[0] is not None and v[0] != float('-inf')}
    if not valid_results:
        msg = 'No successful results in the provided k range{}.'.format(
            ' (excluding k=2)' if skip_k_2 else ''
        )
        if verbose:
            print(msg)
        # CLI: sys.exit(1)
        raise RuntimeError(msg)

    # 4) Choose best k (max silhouette). If tie, prefer smaller k.
    best_k = max(sorted(valid_results.keys()), key=lambda kk: (valid_results[kk][0], -kk))
    best_sil, best_rec_err = valid_results[best_k]
    if verbose:
        print(f"Selected best k = {best_k}, silhouette={best_sil:.4f}, rec_err={best_rec_err:.4f}")

    # 5) Retrain with best_k and produce clusters using argmax on W
    model, W, H, rec_err = fit_nmf(X, n_components=best_k, random_state=random_state, max_iter=max_iter)
    numeric_labels = W.argmax(axis=1)
    cluster_labels = [f"cluster{int(l)+1}" for l in numeric_labels]

    # 5.1) Top features per cluster (use H matrix)
    feature_names = list(selected_df.columns)
    n_feats = len(feature_names)
    top_n = min(15, n_feats)
    rows = []
    for comp_idx in range(H.shape[0]):
        top_idx = np.argsort(-H[comp_idx])[:top_n]
        top_names = [feature_names[i] for i in top_idx]
        # pad if needed (not necessary here since top_n <= n_feats)
        rows.append(top_names)
    col_names = [f'top_{i+1}' for i in range(top_n)]
    top_df = pd.DataFrame(rows, index=[f'cluster{i+1}' for i in range(H.shape[0])], columns=col_names)
    top_df.index.name = 'cluster'

    # BUG-COMPAT (upstream main() step 5.1): this file is written BEFORE
    # save_outputs() creates outdir, inside a swallowing try/except — with a
    # non-existent outdir it is silently missing while the other outputs are
    # written normally. Reproduced verbatim on the outdir path.
    if outdir is not None:
        try:
            top_df.to_csv(os.path.join(outdir, 'top_features_per_cluster.csv'))
        except Exception as e:
            if verbose:
                print('Failed to save top features per cluster:', e)

    # 6) PCA for visualization (use W space)
    try:
        pca = PCA(n_components=2, random_state=random_state)
        W2 = pca.fit_transform(W)
    except Exception:
        W2 = None

    # 7) Save outputs (clusters.csv [+ pca_plot.png when plot=True]);
    #    plot gating happens through pca_coords (save_outputs is verbatim).
    if outdir is not None:
        save_outputs(outdir, selected_df.index.tolist(), cluster_labels,
                     features_df=used_features_df,
                     pca_coords=W2 if plot else None)
        if verbose:
            print('Outputs saved to:', os.path.abspath(outdir))
            print('Files: clusters.csv, pca_plot.png , top_features_per_cluster.csv')

    # Build the clusters.csv frame for the return value with the identical
    # pandas statements save_outputs uses (sample column + preprocessed
    # features reordered to the sample order, index dropped).
    sample_index = selected_df.index.tolist()
    labels_df = pd.DataFrame({'sample': sample_index, 'cluster': cluster_labels})
    feats_ordered = used_features_df.loc[sample_index]
    clusters_df = pd.concat([labels_df, feats_ordered.reset_index(drop=True)], axis=1)

    if verbose:
        _print_banner()

    return {
        "clusters": clusters_df,
        "top_features": top_df,
        "best_k": int(best_k),
        "best_silhouette": float(best_sil),
        "best_rec_err": best_rec_err,
        "k_results": results,
        "labels": numeric_labels,
        "W": W,
        "H": H,
        "model": model,
        "pca_coords": W2,
        "used_features": used_features_df,
        "outdir": os.path.abspath(outdir) if outdir is not None else None,
    }
