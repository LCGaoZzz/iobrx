"""sig_ssgsea_fast: Rust-backed drop-in for IOBRpy's ``sig_score_ssgsea``
stage (``calculate_sig_score(method='ssgsea')``).

The original stage is::

    eset2 = preprocess_eset(eset, adjust_eset)            # log2 heuristic + feature filtering
    sigs  = filter_signatures(sig_dict, eset2, mini_gene_count)
    min_size = max(mini_gene_count, 5)
    sigs  = filter_signatures(sig_dict, eset2, min_size)
    ss = gp.ssgsea(data=eset2, gene_sets=sigs, outdir=None,
                   sample_norm_method='rank', permutation_num=0,
                   no_plot=True, threads=parallel_size,
                   min_size=min_size, ssgsea_norm=True)
    nes = ss.res2d.pivot(index='Term', columns='Name', values='NES').T.reset_index()
    nes.rename(columns={'Name': 'ID'}, inplace=True)
    + TMEscoreA_CIR/B_CIR and TMEscoreA_plus/B_plus contrasts

This module keeps every surrounding step (preprocess, signature filtering,
pivot, contrasts) from the ORIGINAL upstream module and replaces ONLY the
``gp.ssgsea`` call with the bit-exact Rust core ``iobrx_rust.ssgsea_core``
(a port of gseapy 1.3.0's compiled ``ssgsea_rs``: permutation_num=0,
sample_norm_method='rank', correl_norm_type='rank' -> CorrelType::Rank,
weight=0.25, global min-max NES normalization ``nes = es / (max_es - min_es)``).
The gseapy Python-side data path (``SingleSampleGSEA.load_data`` =
``_load_data``/``_check_data`` and ``GSEAbase.load_gmt`` dict filtering) is
replicated literally by the sibling module ``ssgsea_fast`` (bugs included),
so the Rust core receives exactly the matrix gseapy's core would receive.

Gate: reproduces refs/sig_ssgsea.parquet (872x348 imvigor210 eset x
signature_collection, mini_gene_count=3, adjust_eset=True) to <= 1e-6
relative error (achieved: bit-exact / last-ulp; see
iobrx/bench/rust_ssgsea_results.json for the measured value).
"""
# surrounding semantics come from the ORIGINAL module (read-only import)
from iobrpy.workflow.calculate_sig_score import (  # noqa: E402,F401
    preprocess_eset,
    filter_signatures,
)

# the gseapy->Rust core replacement (gseapy load_data/load_gmt literal ports
# + iobrx_rust.ssgsea_core + res2d reconstruction)
from iobrx._fast.ssgsea_fast import sig_score_ssgsea_fast as _sig_score_ssgsea_fast  # noqa: E402


def sig_ssgsea_fast(eset_df, sig_dict, mini_gene_count=3, adjust_eset=True, n_threads=224):
    """Drop-in replacement for ``sig_score_ssgsea`` on the official data.

    Parameters
    ----------
    eset_df : pd.DataFrame
        Expression matrix, genes (index) x samples (columns) — e.g.
        ``testdata/imvigor210_eset.parquet`` (872 x 348).
    sig_dict : dict
        Signature name -> gene list (e.g. the merged ``signature_collection``
        from ``iobrpy.resources.calculate_data.pkl``).
    mini_gene_count : int
        Original ``mini_gene_count`` (official protocol: 3).
    adjust_eset : bool
        Original ``adjust_eset`` (official protocol: True).
    n_threads : int
        Rayon threads for the Rust core (machine: 224 cores).

    Returns
    -------
    pd.DataFrame shaped exactly like the original's output:
    348 rows (one per sample, ``ID`` column first) x NES columns for every
    surviving signature (alphabetical, from the pivot) plus the derived
    ``TMEscore_CIR`` / ``TMEscore_plus`` contrasts when both constituents
    are present.
    """
    return _sig_score_ssgsea_fast(
        eset_df, sig_dict, mini_gene_count, adjust_eset, n_threads
    )
