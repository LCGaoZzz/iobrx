"""In-process port of IOBRpy's runall command routing and assay defaults.

Analysis calls preserve upstream data transformations, output formats and
parameter defaults. CIBERSORT uses the original file-based solver, as in
`tme_profile`, to avoid solver/input-rounding divergence at this boundary.

Reliability changes deliberately differ from upstream: QC errors propagate;
dry runs only print a plan; content hashes bind resumes to their inputs,
references, tool binaries, parameters and calculation products. Unrelated
notes and figures do not affect resume. Runtime sidecars are additional outputs.

External programs still perform alignment/reconstruction. The Python wrapper
removes repeated CLI imports; it does not promise to accelerate those tools.
`backend="python"` remains the explicit untouched-upstream escape hatch and
does not inherit these orchestration reliability changes.
"""

import argparse
import fnmatch
import json
import os
import shlex
import sys
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from iobrx._run_state import artifact_records, file_hash, signature, write_json
from iobrx._resources import resource_path

_RUN_LOCK = threading.RLock()

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is a hard iobrx dependency
    pd = None  # pandas is optional; used only for merging deconvolution results

# Known step names (kept for backward-compatible "sectioned" command style)
METHOD_SECTIONS = {
    "fastq_qc", "batch_salmon", "merge_salmon", "batch_star_count", "merge_star_count",
    "prepare_salmon", "count2tpm",
    "calculate_sig_score", "sig_score",
    "cibersort", "IPS", "estimate", "mcpcounter", "quantiseq", "epic",
    "LR_cal",
    "trust4",
}

# --------------------- Utilities ---------------------

def _ensure_dir(p: Path) -> None:
    """Create a directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def _nonempty(p: Path) -> bool:
    """Return True if the path exists and is a non-empty file or directory."""
    return p.exists() and ((p.is_file() and p.stat().st_size > 0) or (p.is_dir() and any(p.iterdir())))

# ------------------ In-process step executor ------------------
#
# Replaces the upstream ``subprocess.run(["iobrpy", <step>, ...])`` child.
# The command list built by the (verbatim) orchestration is parsed with a
# mirror of the ``iobrpy.main`` subparser for that step and executed with
# the ported ``iobrx`` functions + the ``iobrpy.main`` dispatch-layer I/O
# transformations.

_SUBSTEP_VERBOSE = True


def _print_dispatch_banner() -> None:
    """The IOBRpy banner exactly as the ``iobrpy.main`` dispatch prints it."""
    print("   ")
    from iobrpy.utils.print_colorful_message import print_colorful_message
    print_colorful_message("#########################################################", "blue")
    print_colorful_message(" IOBRpy: Immuno-Oncology Biological Research using Python ", "cyan")
    print_colorful_message(" If you encounter any issues, please report them at ", "cyan")
    print_colorful_message(" https://github.com/IOBR/IOBRpy/issues ", "cyan")
    print_colorful_message("#########################################################", "blue")
    print_colorful_message(" Author: Haonan Huang, Dongqiang Zeng")
    print_colorful_message(" Email: interlaken@smu.edu.cn ")
    print_colorful_message("#########################################################", "blue")
    print("   ")


def _infer_sep_cli(path):
    """VERBATIM copy of ``iobrpy.workflow.quantiseq.infer_sep`` (re-implemented
    locally so the executor does not pull quantiseq's statsmodels import)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return ','
    elif ext in ('.tsv', '.txt'):
        return '\t'
    else:
        return None


def _step_parser(step: str) -> argparse.ArgumentParser:
    """Mirror of the ``iobrpy.main`` subparser for *step*.

    Definitions copied flag-for-flag (names, dests, types, choices, defaults)
    from ``iobrpy/main.py`` so that tokens the orchestrator builds — and any
    sectioned passthrough — parse with IDENTICAL defaults and validation as
    the child ``iobrpy <step>`` CLI. Unknown tokens go to the ignored
    ``parse_known_args`` remainder, exactly like the top-level
    ``parse_known_args`` of ``iobrpy.main``.
    """
    parser = argparse.ArgumentParser(prog=f"iobrpy {step}")
    if step == "fastq_qc":
        parser.add_argument('--path1_fastq', required=True, help='Directory containing raw FASTQ files')
        parser.add_argument('--path2_fastp', required=True, help='Output directory for cleaned FASTQ files')
        parser.add_argument('--num_threads', type=int, default=8, help='Threads per fastp process')
        parser.add_argument('--suffix1', default='_1.fastq.gz', help="R1 suffix; R2 inferred by replacing '1' with '2'")
        parser.add_argument('--batch_size', type=int, default=1, help='Number of concurrent samples (processes)')
        parser.add_argument('--se', action='store_true', help='Single-end sequencing; omit for paired-end')
        parser.add_argument('--length_required', type=int, default=50, help='Minimum read length to keep')
    elif step == "batch_salmon":
        parser.add_argument('--index', required=True, help='Path to Salmon index')
        parser.add_argument('--path_fq', required=True, help='Directory containing FASTQ files')
        parser.add_argument('--path_out', required=True, help='Output directory for per-sample results')
        parser.add_argument('--suffix1', default='_1.fastq.gz', help="R1 suffix; R2 inferred by replacing '1' with '2'")
        parser.add_argument('--batch_size', type=int, default=1, help='Number of concurrent samples (processes)')
        parser.add_argument('--num_threads', type=int, default=8, help='Threads per Salmon process')
        parser.add_argument('--gtf', default=None, help='Optional GTF file path for Salmon (-g)')
    elif step == "merge_salmon":
        parser.add_argument('--path_salmon', required=True, help='Root folder searched recursively for quant.sf')
        parser.add_argument('--project', required=True, help='Output file prefix')
        parser.add_argument('--num_processes', type=int, default=None, help='Threads for loading quant.sf (I/O bound)')
    elif step == "batch_star_count":
        parser.add_argument('--index', required=True, help='STAR genome index directory')
        parser.add_argument('--path_fq', required=True, help='Folder containing FASTQs (R1 endswith suffix1)')
        parser.add_argument('--path_out', required=True, help='Output folder for STAR results')
        parser.add_argument('--suffix1', default='_1.fastq.gz', help='R1 suffix; R2 is inferred by 1→2')
        parser.add_argument('--batch_size', type=int, default=1, help='#samples per batch (sequential batches)')
        parser.add_argument('--num_threads', type=int, default=8, help='Threads for STAR and BAM sorting')
    elif step == "merge_star_count":
        parser.add_argument('--path', required=True, help='Folder containing STAR outputs')
        parser.add_argument('--project', required=True, help='Output name prefix')
    elif step == "prepare_salmon":
        parser.add_argument('-i', '--input', dest='eset_path', required=True,
                            help='Path to input Salmon file (TSV or TSV.GZ)')
        parser.add_argument('-o', '--output', dest='output_matrix', required=True,
                            help='Path to save cleaned TPM matrix')
        parser.add_argument('-r', '--return_feature', dest='return_feature', choices=['ENST', 'ENSG', 'symbol'],
                            default='symbol', help='Which gene feature to retain')
        parser.add_argument('--remove_version', action='store_true',
                            help='Remove version suffix from gene IDs')
    elif step == "count2tpm":
        parser.add_argument('-i', '--input', dest='count_mat', required=True,
                            help='Path to input count matrix (CSV/TSV, genes×samples)')
        parser.add_argument('--effLength_csv', type=str,
                            help='Optional CSV with id, eff_length, and gene_symbol columns')
        parser.add_argument('--idtype', choices=["ensembl", "entrez", "symbol", "mgi"], default="ensembl",
                            help='Gene ID type')
        parser.add_argument('--org', choices=["hsa", "mmus"], default="hsa",
                            help='Organism: hsa or mmus')
        parser.add_argument('--source', choices=["local", "biomart"], default="local",
                            help='Source of gene lengths')
        parser.add_argument('--id', dest='id_col', default="id",
                            help='Column name for gene ID in effLength CSV')
        parser.add_argument('--length', dest='length_col', default="eff_length",
                            help='Column name for gene length in effLength CSV')
        parser.add_argument('--gene_symbol', dest='gene_symbol_col', default="symbol",
                            help='Column name for gene symbol in effLength CSV')
        parser.add_argument('--check_data', action='store_true',
                            help='Check and remove missing values in count matrix')
        parser.add_argument('-o', '--output', dest='output_path', required=True,
                            help='Path to save TPM matrix')
        parser.add_argument('--remove_version', action='store_true',
                            help='Remove version suffix from gene IDs before processing')
    elif step == "log2_eset":
        parser.add_argument('-i', '--input', required=True,
                            help='Path to input matrix (CSV/TSV/TXT, optionally .gz). Rows=genes, cols=samples.')
        parser.add_argument('-o', '--output', required=True,
                            help="Path to save the log2(x+1) matrix. Extension selects delimiter (.csv/.tsv or mirror input).")
    elif step == "calculate_sig_score":
        parser.add_argument('-i', '--input', dest='input_path', required=True,
                            help='Path to input expression matrix')
        parser.add_argument('-o', '--output', dest='output_path', required=True,
                            help='Path to save signature scores')
        parser.add_argument('--signature', required=True, nargs='+',
                            help='One or more signature GROUP names to use.')
        parser.add_argument('--method', dest='score_method', choices=['pca', 'zscore', 'ssgsea', 'integration'],
                            default='pca', help='Scoring method to apply')
        parser.add_argument('--mini_gene_count', type=int, default=3,
                            help='Minimum genes per signature')
        parser.add_argument('--adjust_eset', action='store_true',
                            help='Apply additional filtering after log2 transform')
        parser.add_argument('--parallel_size', type=int, default=1,
                            help='Threads for scoring (PCA/zscore/ssGSEA)')
    elif step == "cibersort":
        parser.add_argument('-i', '--input', dest='input_path', required=True,
                            help='Path to mixture file (CSV or TSV)')
        parser.add_argument('--perm', type=int, default=100,
                            help='Number of permutations')
        parser.add_argument('--QN', type=lambda x: x.lower() == 'true', default=True,
                            help='Quantile normalization (True/False)')
        parser.add_argument('--absolute', type=lambda x: x.lower() == 'true', default=False,
                            help='Absolute mode (True/False)')
        parser.add_argument('--abs_method', default='sig.score',
                            choices=['sig.score', 'no.sumto1'],
                            help='Absolute scoring method')
        parser.add_argument('--threads', type=int, dest='threads', default=1,
                            help='Number of parallel threads')
        parser.add_argument('-o', '--output', dest='output_path', required=True,
                            help='Path to save CIBERSORT results (CSV or TSV)')
    elif step == "IPS":
        parser.add_argument('-i', '--input', dest='input_path', required=True,
                            help='Path to expression matrix file (e.g., EXPR.txt)')
        parser.add_argument('-o', '--output', dest='output_path', required=True,
                            help='Path to save IPS results (e.g., IPS_results.txt)')
    elif step == "estimate":
        parser.add_argument('-i', '--input', dest='input_path', required=True,
                            help='Path to input matrix file (genes x samples)')
        parser.add_argument('-p', '--platform', choices=['affymetrix', 'agilent', 'illumina'],
                            default='affymetrix',
                            help='Specify the platform type for the input data')
        parser.add_argument('-o', '--output', dest='output_path', required=True,
                            help='Path to save estimate results')
    elif step == "mcpcounter":
        parser.add_argument('-i', '--input', dest='input_path', required=True,
                            help='Path to input expression matrix (TSV, genes×samples)')
        parser.add_argument('-f', '--features', required=True,
                            choices=['affy133P2_probesets', 'HUGO_symbols', 'ENTREZ_ID', 'ENSEMBL_ID'],
                            help='Type of gene identifiers')
        parser.add_argument('-o', '--output', dest='output_path', required=True,
                            help='Path to save MCPcounter results (TSV)')
    elif step == "quantiseq":
        parser.add_argument('-i', '--input', dest='input', required=True,
                            help='Path to the input mixture matrix TSV/CSV file (genes x samples)')
        parser.add_argument('-o', '--output', dest='output', required=True,
                            help='Path to save the deconvolution results TSV')
        parser.add_argument('--arrays', action='store_true',
                            help='Perform quantile normalization on array data before deconvolution')
        parser.add_argument('--signame', default='TIL10',
                            help='Name of the signature set to use (default: TIL10)')
        parser.add_argument('--tumor', action='store_true',
                            help='Remove genes with high expression in tumor samples')
        parser.add_argument('--scale_mrna', dest='mRNAscale', action='store_true',
                            help='Enable mRNA scaling; use raw signature proportions otherwise')
        parser.add_argument('--method', choices=['lsei', 'hampel', 'huber', 'bisquare'], default='lsei',
                            help='Deconvolution method: lsei (least squares) or robust norms')
        parser.add_argument('--rmgenes', default='unassigned',
                            help="Genes to remove: 'default', 'none', or comma-separated list")
    elif step == "epic":
        parser.add_argument('-i', '--input', dest='input', required=True,
                            help='Path to the bulk expression matrix (genes×samples)')
        parser.add_argument('-o', '--output', dest='output', required=True,
                            help='Path to save EPIC cell fractions (CSV/TSV)')
        parser.add_argument('--reference', choices=['TRef', 'BRef', 'both'], default='TRef',
                            help='Which reference to use for deconvolution')
    elif step == "LR_cal":
        parser.add_argument('-i', '--input', required=True,
                            help='Path to input expression matrix (genes x samples)')
        parser.add_argument('-o', '--output', required=True, help='Path to save LR scores')
        parser.add_argument('--data_type', choices=['count', 'tpm'], default='tpm',
                            help='Type of input data: count or tpm')
        parser.add_argument('--id_type', default='ensembl',
                            help='Gene ID type.Choices: ensembl, entrez, symbol, mgi.')
        parser.add_argument('--cancer_type', default='pancan', help='Cancer type network')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    else:
        raise ValueError(f"runall executor: unknown step {step!r}")
    return parser


# ---------------- per-step executors (iobrpy.main dispatch, verbatim I/O) ----

def _exec_fastq_qc(args, verbose):
    import iobrx
    result = iobrx.fastq_qc(
        path1_fastq=args.path1_fastq, path2_fastp=args.path2_fastp,
        num_threads=args.num_threads, suffix1=args.suffix1,
        batch_size=args.batch_size, se=args.se,
        length_required=args.length_required, verbose=verbose,
    )
    return int(result["rc"])


def _exec_batch_salmon(args, verbose):
    import iobrx
    res = iobrx.batch_salmon(
        index=args.index, path_fq=args.path_fq, path_out=args.path_out,
        suffix1=args.suffix1, batch_size=args.batch_size,
        num_threads=args.num_threads, gtf=args.gtf, verbose=verbose,
    )
    return int(res.get("rc", 0)) if isinstance(res, dict) else 0


def _exec_merge_salmon(args, verbose):
    import iobrx
    iobrx.merge_salmon(
        path_salmon=args.path_salmon, project=args.project,
        num_processes=args.num_processes, verbose=verbose,
    )
    return 0


def _exec_batch_star_count(args, verbose):
    import iobrx
    res = iobrx.batch_star_count(
        index=args.index, path_fq=args.path_fq, path_out=args.path_out,
        suffix1=args.suffix1, batch_size=args.batch_size,
        num_threads=args.num_threads, verbose=verbose,
    )
    return int(res.get("rc", 0)) if isinstance(res, dict) else 0


def _exec_merge_star_count(args, verbose):
    import iobrx
    iobrx.merge_star_count(path=args.path, project=args.project,
                           verbose=verbose)
    return 0


def _exec_prepare_salmon(args, verbose):
    import iobrx
    iobrx.prepare_salmon(
        input=args.eset_path, output=args.output_matrix,
        return_feature=args.return_feature,
        remove_version=args.remove_version, verbose=verbose,
    )
    return 0


def _exec_count2tpm(args, verbose):
    # iobrpy.main dispatch for count2tpm, verbatim (load + call + write).
    import iobrx
    if args.count_mat.endswith('.gz'):
        count_mat = pd.read_csv(args.count_mat, sep='\t', index_col=0, compression='gzip')
    else:
        sep = '\t' if args.count_mat.endswith(('.tsv', '.tsv.gz')) else ','
        count_mat = pd.read_csv(args.count_mat, sep=sep, index_col=0)
    eff_df = pd.read_csv(args.effLength_csv) if args.effLength_csv else None
    tpm_df = iobrx.count2tpm(
        count_mat=count_mat,
        anno_grch38=None,
        anno_gc_vm32=None,
        idType=args.idtype,
        org=args.org,
        source=args.source,
        remove_version=args.remove_version,
        effLength_df=eff_df,
        id_col=args.id_col,
        length_col=args.length_col,
        gene_symbol_col=args.gene_symbol_col,
        check_data=args.check_data,
    )
    tpm_df.to_csv(args.output_path)
    if verbose:
        print(f"Saved TPM matrix to {args.output_path}")
        _print_dispatch_banner()
    return 0


def _exec_log2_eset(args, verbose):
    import iobrx
    iobrx.log2_eset(input=args.input, output=args.output, verbose=verbose)
    return 0


def _exec_calculate_sig_score(args, verbose):
    # iobrpy.main dispatch for calculate_sig_score, verbatim.
    import iobrx
    ext = Path(args.input_path).suffix.lower()
    if ext == '.csv':
        eset_df = pd.read_csv(args.input_path, sep=',', index_col=0)
    elif ext == '.txt':
        eset_df = pd.read_csv(args.input_path, sep='\t', index_col=0)
    else:
        eset_df = pd.read_csv(
            args.input_path,
            sep=None,
            engine='python',
            index_col=0
        )
    scores_df = iobrx.calculate_sig_score(
        eset_df,
        args.signature,
        args.score_method,
        args.mini_gene_count,
        args.adjust_eset,
        n_threads=args.parallel_size,
    )
    scores_df.to_csv(args.output_path, index=False)
    if verbose:
        print(f"Signature scores saved to {args.output_path}")
        _print_dispatch_banner()
    return 0


def _exec_cibersort(args, verbose):
    # iobrpy.main dispatch for cibersort, verbatim (incl. the mixture read
    # semantics of the original function: sep=None + engine='python').
    from iobrpy.workflow.cibersort import cibersort
    result_df = cibersort(
        str(args.input_path),
        perm=args.perm,
        QN=args.QN,
        absolute=args.absolute,
        abs_method=args.abs_method,
        n_jobs=args.threads,
    )
    result_df.columns = [col + '_CIBERSORT' for col in result_df.columns]
    result_df.index.name = 'ID'
    delim = ',' if args.output_path.lower().endswith('.csv') else '\t'
    result_df.to_csv(args.output_path, sep=delim, index=True)
    if verbose:
        print(f"CIBERSORT results saved to {args.output_path}")
        _print_dispatch_banner()
    return 0


def _exec_ips(args, verbose):
    import iobrx
    iobrx.ips(args.input_path, output_file=args.output_path, verbose=verbose)
    return 0


def _exec_estimate(args, verbose):
    # iobrpy.main dispatch for estimate, verbatim.
    import iobrx
    sep = '\t' if args.input_path.lower().endswith(('.tsv', '.txt')) else ','
    in_df = pd.read_csv(args.input_path, sep=sep, index_col=0)
    score_df = iobrx.estimate_score(in_df, args.platform)
    score_df = score_df.T
    score_df.columns = [col + '_estimate' for col in score_df.columns]
    out_sep = '\t' if args.output_path.lower().endswith(('.tsv', '.txt')) else ','
    score_df.to_csv(args.output_path, sep=out_sep, index_label='ID')
    if verbose:
        print(f"Estimate scores saved to {args.output_path}")
        _print_dispatch_banner()
    return 0


def _exec_mcpcounter(args, verbose):
    # iobrpy.main dispatch for mcpcounter, verbatim (preprocess_input is the
    # UNTOUCHED original helper; the numeric core is the ported one).
    import iobrx
    from iobrpy.workflow.mcpcounter import preprocess_input
    expr_df = preprocess_input(args.input_path)
    scores_df = iobrx.mcpcounter(expr_df, args.features)
    out_df = scores_df.T
    out_df.columns = [col.replace(' ', '_') + '_MCPcounter' for col in out_df.columns]
    out_ext = Path(args.output_path).suffix.lower()
    out_sep = ',' if out_ext == '.csv' else '\t'
    out_df.to_csv(args.output_path, sep=out_sep, index_label='ID', float_format='%.7f')
    if verbose:
        print(f"MCPcounter results saved to {args.output_path}")
        _print_dispatch_banner()
    return 0


def _exec_quantiseq(args, verbose):
    # iobrpy.main rewrites argv into workflow/quantiseq.py main(); this
    # reproduces that main()'s post-parse body verbatim, with the ported
    # numeric core and its memoized resource bundle (same pickle).
    import iobrx
    in_sep = _infer_sep_cli(args.input)
    mix = pd.read_csv(args.input, sep=in_sep, index_col=0)
    res = iobrx.quantiseq(
        mix,
        data=None,
        arrays=args.arrays,
        signame=args.signame,
        tumor=args.tumor,
        mRNAscale=args.mRNAscale,
        method=args.method,
        rmgenes=args.rmgenes,
    )
    out_sep = _infer_sep_cli(args.output) or '\t'
    res.columns = [
        'ID' if col == 'Sample'
        else f"{col.replace('.', '_')}_quantiseq"
        for col in res.columns
    ]
    eps = 1e-8
    num_cols = res.columns.drop('ID')
    res[num_cols] = res[num_cols].mask(res[num_cols].abs() < eps, 0)
    res.to_csv(args.output, sep=out_sep, index=False)
    if verbose:
        print(f"Results saved to {args.output}")
        _print_dispatch_banner()
    return 0


def _exec_epic(args, verbose):
    # iobrpy.main rewrites argv into workflow/epic.py main(); this reproduces
    # that main()'s post-parse body verbatim. Flags not exposed by the
    # iobrpy.main 'epic' subparser take workflow/epic.py's own defaults:
    # solver='trust-constr', seed=None (no np.random.seed), mRNA_cell_sub={},
    # sigGenes=None, scale_exprs/with_other_cells/constrained_sum=True,
    # range_based_optim=False, jitter=0.0, unlog=False.
    import pickle
    import warnings
    from importlib.resources import files

    import iobrx
    from iobrpy.workflow.epic import (
        infer_sep, _to_df, merge_duplicates, mRNA_cell_default,
    )

    sep_in = infer_sep(args.input)
    bulk = pd.read_csv(args.input, sep=sep_in, index_col=0)

    ref_pkg = resource_path('epic_TRef_BRef.pkl')
    with ref_pkg.open('rb') as f:
        ref_data = pickle.load(f)

    refs = []
    if args.reference in ('TRef', 'both'): refs.append('TRef')
    if args.reference in ('BRef', 'both'): refs.append('BRef')

    profs, vars_, flags, sgs = [], [], [], []
    for key in refs:
        dd = ref_data[key]
        profs.append(_to_df(dd['refProfiles'], dd))
        varr = dd.get('refProfiles.var')
        if varr is not None:
            flags.append(True)
            vars_.append(_to_df(varr, dd))
        else:
            flags.append(False)
        sgs.extend(dd.get('sigGenes', []))

    ref_profiles = pd.concat(profs, axis=1)
    ref_profiles = merge_duplicates(ref_profiles, "reference profiles").loc[:, ~ref_profiles.columns.duplicated()]
    if any(flags):
        full_vars = []
        for present, vdf, prof in zip(flags, vars_, profs):
            full_vars.append(vdf if present else pd.DataFrame(0, index=prof.index, columns=prof.columns))
        ref_vars = pd.concat(full_vars, axis=1).loc[:, ref_profiles.columns]
        var_present = True
    else:
        ref_vars = None
        var_present = False

    sig_ref = [g for g in dict.fromkeys(sgs) if g in ref_profiles.index]
    reference = {
        'refProfiles': ref_profiles,
        'refProfiles.var': ref_vars,
        'sigGenes': sig_ref,
        'mRNA_cell': mRNA_cell_default,
        'var_present': var_present
    }

    # Heuristic: if no overlap, likely genes are in columns -> transpose
    if not set(bulk.index).intersection(sig_ref):
        warnings.warn("Detected genes in columns; transposing bulk.")
        bulk = bulk.T

    res = iobrx.epic(
        bulk, reference,
        mRNA_cell=None,
        mRNA_cell_sub={},
        sig_genes=None,
        scale_exprs=True,
        with_other_cells=True,
        constrained_sum=True,
        range_based_optim=False,
        solver='trust-constr',
        init_jitter=0.0,
        unlog_bulk=False,
    )

    # Save cell fractions
    ext = os.path.splitext(args.output)[1].lower()
    sep_out = ',' if ext == '.csv' else '\t'
    out_df = res['cellFractions'].copy()
    out_df.columns = [f"{col}_EPIC" for col in out_df.columns]
    out_df.to_csv(args.output, sep=sep_out, index=True)
    if verbose:
        print(f"Saved cellFractions ➜ {args.output}")
        _print_dispatch_banner()
    return 0


def _exec_lr_cal(args, verbose):
    import iobrx
    iobrx.lr_cal(
        args.input,
        output_file=args.output,
        data_type=args.data_type,
        id_type=args.id_type,
        cancer_type=args.cancer_type,
        verbose=args.verbose,
    )
    return 0


def _exec_trust4(tokens, verbose):
    # Raw tokens go through the ported trust4 CLI (same implementation
    # iobrx.trust4 wraps) so arbitrary passthrough keeps upstream
    # parse_known_args semantics; banner mirrors the iobrpy.main dispatch
    # (printed after the stage's SystemExit, for ANY exit code).
    from iobrx._fast.trust4_fast import main as _t4_main, print_iobrpy_banner
    try:
        _t4_main(list(tokens), engine="fast")
        return 0
    except SystemExit as e:
        if verbose:
            print_iobrpy_banner()
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 1)


_EXECUTORS = {
    "fastq_qc": _exec_fastq_qc,
    "batch_salmon": _exec_batch_salmon,
    "merge_salmon": _exec_merge_salmon,
    "batch_star_count": _exec_batch_star_count,
    "merge_star_count": _exec_merge_star_count,
    "prepare_salmon": _exec_prepare_salmon,
    "count2tpm": _exec_count2tpm,
    "log2_eset": _exec_log2_eset,
    "calculate_sig_score": _exec_calculate_sig_score,
    "cibersort": _exec_cibersort,
    "IPS": _exec_ips,
    "estimate": _exec_estimate,
    "mcpcounter": _exec_mcpcounter,
    "quantiseq": _exec_quantiseq,
    "epic": _exec_epic,
    "LR_cal": _exec_lr_cal,
}


def _dispatch_inprocess(cmd: List[str], cwd: Optional[Path] = None,
                        verbose: bool = True) -> int:
    """Execute one constructed ``["iobrpy", <step>, ...]`` command in-process.

    Returns the exit code the child CLI would have produced: 0 on success,
    the step's own rc (batch_salmon/batch_star_count/trust4), 2 for argparse
    usage errors, 1 for an uncaught exception (traceback printed to stderr,
    like a crashing child).
    """
    step = cmd[1]
    tokens = [str(x) for x in cmd[2:]]
    prev_cwd = None
    try:
        if cwd is not None:
            prev_cwd = os.getcwd()
            os.chdir(str(cwd))
        if step == "trust4":
            return _exec_trust4(tokens, verbose)
        parser = _step_parser(step)
        args, _ignored_unknown = parser.parse_known_args(tokens)
        return _EXECUTORS[step](args, verbose)
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        if prev_cwd is not None:
            os.chdir(prev_cwd)


def _run(cmd: List[str], cwd: Optional[Path] = None, dry: bool = False) -> int:
    """Run one pipeline step, streaming output to console.

    Upstream runs ``subprocess.run(cmd, stdout=sys.stdout,
    stderr=sys.stderr, cwd=...)``; this port keeps the console protocol
    byte-identical and executes the SAME command list in-process via
    :func:`_dispatch_inprocess` (the port's single structural change).
    """
    line = " ".join(shlex.quote(x) for x in cmd)
    header = f"[run] {line}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    if dry:
        print("[dry-run] skipped execution")
        return 0
    rc = _dispatch_inprocess(cmd, cwd=cwd, verbose=_SUBSTEP_VERBOSE)
    if rc != 0:
        print(f"[ERROR] {cmd[1]} failed (rc={rc}).")
    else:
        print(f"[ok] {cmd[1]}")
    return rc

def _normalize_flag_token(tok: str) -> str:
    """Normalize a long flag token by replacing '-' with '_' in its name."""
    if tok.startswith("--") and len(tok) > 2:
        return f"--{tok[2:].replace('-', '_')}"
    return tok

def _flag_name(tok: str) -> Optional[str]:
    """Return normalized flag name (lowercase, '_' instead of '-') or None if not a long flag."""
    if tok.startswith("--") and len(tok) > 2:
        return tok[2:].replace("-", "_").lower()
    return None

def _parse_passthrough_blocks(tokens: List[str]) -> Dict[str, List[str]]:
    """
    Parse the legacy "sectioned" style:
      fastq_qc --num_threads 8 ... batch_salmon --index ...
    If the user did not provide module name sections, this returns {}.
    """
    blocks: Dict[str, List[str]] = {}
    cur: Optional[str] = None
    expect_value = False

    for tok in tokens:
        name = tok[2:] if tok.startswith("--") else tok
        # Switch section when encountering a bare step name
        if not tok.startswith("--") and name in METHOD_SECTIONS and not expect_value:
            cur = name
            blocks.setdefault(cur, [])
            continue
        # Collect tokens into the current section
        if cur is not None:
            ntok = _normalize_flag_token(tok)
            blocks[cur].append(ntok)
            expect_value = ntok.startswith("--")
        else:
            expect_value = False
    return blocks

def _find_latest(dirpath: Path, globs: List[str]) -> Optional[Path]:
    """Return the most recently modified file in dirpath matching any pattern, or None."""
    cand: List[Path] = []
    for pat in globs:
        cand += list(dirpath.glob(pat))
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]

def _append_passthrough(cmd: List[str], buckets: Dict[str, List[str]], key: str, alt_key: Optional[str] = None) -> None:
    """Append auto/sectioned-routed args for a given step into the subcommand list."""
    for k in filter(None, [key, alt_key]):
        if k in buckets and buckets[k]:
            cmd += buckets[k]

# ------------------ Auto-routing of flags ------------------

# Per-step known long flags (normalized names without '--')
FLAG_BUCKETS: Dict[str, set] = {
    # fastq_qc: threads and batch_size will be injected at top-level
    "fastq_qc": {"se", "length_required", "suffix1"},
    # salmon
    "batch_salmon": {"index", "suffix1", "gtf"},
    "merge_salmon": {"project"},  # num_processes injected from --threads
    # star
    "batch_star_count": {"index", "suffix1"},
    "merge_star_count": {"project"},
    # TPM conversion
    "prepare_salmon": {"return_feature", "remove_version"},
    "count2tpm": {"idtype", "org", "source", "id", "length", "gene_symbol", "check_data", "efflength_csv", "remove_version"},
    # signature scores
    "calculate_sig_score": {"signature", "method", "mini_gene_count", "adjust_eset"},
    # deconvolution
    "cibersort": {"perm", "qn", "absolute", "abs_method"},  # cibersort.threads comes from --threads
    "IPS": set(),
    "estimate": {"platform"},
    "mcpcounter": {"features"},
    "quantiseq": {"arrays", "signame", "tumor", "scale_mrna", "method"},
    "epic": {"reference"},
    # ligand-receptor
    "LR_cal": {"data_type", "id_type", "cancer_type", "verbose"},
    # TCR/BCR repertoire (TRUST4 wrapper)
    "trust4": {"fqdir", "ref"},  # -t threads comes from top-level --threads
}

def _consume_top_level_scalars(unknown: List[str]) -> Tuple[List[str], Optional[int], Optional[int]]:
    """
    Pull legacy concurrency flags out of 'unknown' and promote them to top-level:
      threads: threads / num_threads / parallel_size / num_processes
      batch:   batch_size
    Returns (remaining_unknown, threads_val, batch_val).
    """
    tokens = list(unknown)
    threads_val: Optional[int] = None
    batch_val: Optional[int] = None

    def pop_int(names: List[str]) -> Optional[int]:
        nonlocal tokens
        got: Optional[int] = None
        i = 0
        names_set = {n.lower() for n in names}
        while i < len(tokens):
            name = _flag_name(tokens[i]) or ""
            if name in names_set:
                # If next token is a value, capture and remove both
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    try:
                        got = int(tokens[i + 1])
                    except Exception:
                        pass
                    del tokens[i:i + 2]
                    continue
                else:
                    # Flag without value -> remove flag token and continue
                    del tokens[i:i + 1]
                    continue
            i += 1
        return got

    # The last occurrence wins
    v_threads = pop_int(["threads", "num_threads", "parallel_size", "num_processes"])
    if v_threads is not None:
        threads_val = v_threads
    v_batch = pop_int(["batch_size"])
    if v_batch is not None:
        batch_val = v_batch

    return tokens, threads_val, batch_val

def _autobucket(tokens: List[str], mode: str) -> Dict[str, List[str]]:
    """
    Route long flags to steps when the user does NOT provide section names.
    Special-cases:
      - --index targets batch_salmon (salmon) or batch_star_count (star)
      - --project targets merge_salmon (salmon) or merge_star_count (star)
      - --remove_version is routed by mode:
          salmon -> prepare_salmon
          star   -> count2tpm
      - --method is disambiguated between calculate_sig_score and quantiseq by value
    """
    buckets: Dict[str, List[str]] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        name = _flag_name(tok)
        if not name:
            i += 1
            continue

        val: Optional[str] = None
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            val = tokens[i + 1]

        targets: List[str] = []

        # Mode-aware routing
        if name == "index":
            targets = ["batch_salmon" if mode == "salmon" else "batch_star_count"]
        elif name == "project":
            targets = ["merge_salmon" if mode == "salmon" else "merge_star_count"]
        elif name == "remove_version":
            targets = ["prepare_salmon" if mode == "salmon" else "count2tpm"]
        elif name == "suffix1":
            targets = ["fastq_qc", "batch_salmon" if mode == "salmon" else "batch_star_count"]
        else:
            # Generic mapping via FLAG_BUCKETS
            for mod, flags in FLAG_BUCKETS.items():
                if name in flags:
                    # Disambiguate 'method' between calc_sig_score and quantiseq
                    if name == "method" and mod in ("calculate_sig_score", "quantiseq") and val:
                        v = str(val).lower()
                        if v in {"integration", "pca", "zscore", "ssgsea"}:
                            targets = ["calculate_sig_score"]
                        elif v in {"lsei", "hampel", "huber", "bisquare"}:
                            targets = ["quantiseq"]
                        else:
                            targets = ["calculate_sig_score"]
                        break
                    targets = [mod]
                    break

        if targets:
            for target in targets:
                buckets.setdefault(target, []).append(_normalize_flag_token(tok))
                if val is not None:
                    buckets[target].append(val)
        else:
            print(f"[warn] Unrecognized flag (ignored by router): {tok}{(' ' + val) if val else ''}")

        i += 1 if val is None else 2
    return buckets

# --------------------- Main pipeline ---------------------

def _main_impl(argv: Optional[List[str]] = None, *, completed_steps=None) -> int:
    """Route upstream commands, recording successful table-producing steps.

    File presence and an unchanged hash do not prove that the writer succeeded.
    Resume requires a completion record from a zero-exit invocation as well.
    """
    parser = argparse.ArgumentParser(prog="iobrpy runall", description="End-to-end orchestrator (salmon/star) with auto routing.")
    parser.add_argument("--mode", choices=["salmon", "star"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fastq", required=True, help="Path to raw FASTQ directory (used as fastq_qc --path1_fastq)")
    parser.add_argument("--threads", type=int, default=None, help="Unified concurrency for multiple steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Unified batch size for fastq_qc/salmon/star")
    parser.add_argument("--resume", action="store_true", help="Reuse verified successful steps")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    ns, unknown = parser.parse_known_args(argv)

    completed_steps = {} if completed_steps is None else completed_steps

    def reusable(step, output):
        if not ns.resume:
            return False
        try:
            return completed_steps.get(step) == artifact_records([output])
        except (OSError, RuntimeError):
            return False

    def record_completed(step, output):
        if not ns.dry_run:
            completed_steps[step] = artifact_records([output])

    def run_table(cmd, output):
        if not ns.dry_run:
            completed_steps.pop(cmd[1], None)
        rc = _run(cmd, dry=ns.dry_run)
        if rc == 0:
            record_completed(cmd[1], output)
        return rc

    outdir = Path(ns.outdir).resolve()

    # Numbered directories (tme_cluster removed; LR_cal renamed to 06)
    d_fastp    = outdir / "01-qc"
    d_tpm      = outdir / "03-tpm"
    d_sigscore = outdir / "04-signatures"
    d_deconv   = outdir / "05-tme"
    d_lrcal    = outdir / "06-LR_cal"
    d_tcrbcr   = outdir / "07-TCRBCR"
    d_salmon   = outdir / "02-salmon"
    d_star     = outdir / "02-star"
    # Create common directories (shared across modes)
    for d in [d_fastp, d_tpm, d_sigscore, d_deconv, d_lrcal, d_tcrbcr]:
        if not ns.dry_run:
            _ensure_dir(d)

    # Create the mode-specific directory only
    if not ns.dry_run:
        _ensure_dir(d_salmon if ns.mode == "salmon" else d_star)

    # Legacy "sectioned" style (optional)
    blocks_named = _parse_passthrough_blocks([_normalize_flag_token(t) for t in unknown])
    using_named = any(blocks_named.values())

    # Consume legacy concurrency and batch flags
    unknown, legacy_threads, legacy_batch = _consume_top_level_scalars(unknown)

    # Auto-route long flags if no sections were provided
    blocks_auto = {} if using_named else _autobucket([_normalize_flag_token(t) for t in unknown], ns.mode)

    # Merge routes from both sources
    blocks: Dict[str, List[str]] = {}
    for k in set(list(blocks_named.keys()) + list(blocks_auto.keys())):
        blocks[k] = (blocks_named.get(k) or []) + (blocks_auto.get(k) or [])

    # Guarantee that any provided --suffix1 also reaches the quantification step
    # (salmon/star), even if the user only attached it to fastq_qc in sectioned mode.
    suffix_tokens: Optional[List[str]] = None
    for name in ("fastq_qc", "fastq"):
        toks = blocks.get(name) or []
        for idx, tok in enumerate(toks):
            if tok == "--suffix1":
                if idx + 1 < len(toks) and not toks[idx + 1].startswith("--"):
                    suffix_tokens = [tok, toks[idx + 1]]
                else:
                    suffix_tokens = [tok]
                break
        if suffix_tokens:
            break

    if suffix_tokens:
        quant_key = "batch_salmon" if ns.mode == "salmon" else "batch_star_count"
        quant_alias = "salmon" if ns.mode == "salmon" else "star"

        def _ensure_suffix(target: str) -> None:
            tokens = blocks.setdefault(target, [])
            if not any(t == "--suffix1" for t in tokens):
                tokens += suffix_tokens

        _ensure_suffix(quant_key)
        _ensure_suffix(quant_alias)

    # Final unified values (explicit top-level overrides legacy)
    threads = ns.threads if ns.threads is not None else (legacy_threads if legacy_threads is not None else 8)
    batch_size = ns.batch_size if ns.batch_size is not None else (legacy_batch if legacy_batch is not None else 1)

    # 1) fastq_qc -> 01-qc/
    fastp_done_flag = d_fastp / ".fastq_qc.done"
    if ns.resume and fastp_done_flag.exists() and _nonempty(d_fastp):
        print("[resume] fastq_qc skipped (01-qc/ already has outputs).")
    else:
        cmd = ["iobrpy", "fastq_qc",
               "--path1_fastq", str(Path(ns.fastq).resolve()),
               "--path2_fastp", str(d_fastp),
               "--num_threads", str(threads),
               "--batch_size", str(batch_size)]
        _append_passthrough(cmd, blocks, "fastq_qc", "fastq")
        rc = _run(cmd, dry=ns.dry_run)
        if rc != 0:
            return rc
        if not ns.dry_run:
            fastp_done_flag.write_text("done\n", encoding="utf-8")

    # 2/3/4) Quantify & merge -> 02-*/ and 03-tpm/
    if ns.mode == "salmon":
        # 2a) batch_salmon -> 02-salmon/
        if not (ns.resume and _nonempty(d_salmon / ".batch_salmon.done") and _nonempty(d_salmon)):
            cmd = ["iobrpy", "batch_salmon",
                   "--path_fq", str(d_fastp),
                   "--path_out", str(d_salmon),
                   "--num_threads", str(threads),
                   "--batch_size", str(batch_size)]
            _append_passthrough(cmd, blocks, "batch_salmon", "salmon")
            rc = _run(cmd, dry=ns.dry_run)
            if rc != 0:
                return rc
            if not ns.dry_run:
                (d_salmon / ".batch_salmon.done").write_text("done\n", encoding="utf-8")
        else:
            print("[resume] batch_salmon skipped.")

        # 3a) merge_salmon (cwd=02-salmon/)
        if not (ns.resume and _nonempty(d_salmon / ".merge_salmon.done")):
            merge_args = (blocks.get("merge_salmon") or []) + (blocks.get("salmon") or [])
            has_project = any(tok == "--project" for tok in merge_args)

            cmd = ["iobrpy", "merge_salmon",
                   "--path_salmon", str(d_salmon)]

            if not has_project:
                cmd += ["--project", "runall"]

            cmd += ["--num_processes", str(threads)]
            _append_passthrough(cmd, blocks, "merge_salmon", "salmon")
            rc = _run(cmd, cwd=d_salmon, dry=ns.dry_run)
            if rc != 0:
                return rc
            if not ns.dry_run:
                (d_salmon / ".merge_salmon.done").write_text("done\n", encoding="utf-8")
        else:
            print("[resume] merge_salmon skipped.")

        merged_salmon_tpm = _find_latest(d_salmon, ["*_salmon_tpm.tsv", "*_salmon_tpm.tsv.gz"])
        if ns.dry_run and merged_salmon_tpm is None:
            merged_salmon_tpm = d_salmon / "runall_salmon_tpm.tsv.gz"
        if merged_salmon_tpm is None:
            print("[ERROR] Cannot find merged Salmon TPM in '02-salmon/' (pattern '*_salmon_tpm.tsv*').")
            return 2

        # 4a) prepare_salmon -> 03-tpm/prepare_salmon.csv, then log2_eset -> 03-tpm/tpm_matrix.csv
        prep_csv   = d_tpm / "prepare_salmon.csv"
        tpm_matrix = d_tpm / "tpm_matrix.csv"

        if reusable("prepare_salmon", prep_csv) and reusable("log2_eset", tpm_matrix):
            # If final log2'ed matrix exists, skip both steps.
            print("[resume] prepare_salmon + log2_eset skipped.")
        else:
            # Run prepare_salmon only if the intermediate is missing (resume-friendly).
            if not reusable("prepare_salmon", prep_csv):
                cmd = ["iobrpy", "prepare_salmon",
                       "--input", str(merged_salmon_tpm),
                       "--output", str(prep_csv)]
                # Apply default return_feature only if user didn't provide one
                ps_args = blocks.get("prepare_salmon") or []
                if not any(a.startswith("--return_feature") for a in ps_args):
                    cmd += ["--return_feature", "symbol"]
                _append_passthrough(cmd, blocks, "prepare_salmon")
                # Default: invoke --remove_version unless user already set it
                if not any(a.startswith("--remove_version") for a in ps_args):
                    cmd.append("--remove_version")
                rc = run_table(cmd, prep_csv)
                if rc != 0:
                    return rc

            # Always ensure final matrix is log2(x+1) from the intermediate.
            cmd = ["iobrpy", "log2_eset",
                   "-i", str(prep_csv),
                   "-o", str(tpm_matrix)]
            rc = run_table(cmd, tpm_matrix)
            if rc != 0:
                return rc

    else:
        # 2b) batch_star_count -> 02-star/
        if not (ns.resume and _nonempty(d_star / ".batch_star_count.done") and _nonempty(d_star)):
            cmd = ["iobrpy", "batch_star_count",
                   "--path_fq", str(d_fastp),
                   "--path_out", str(d_star),
                   "--num_threads", str(threads),
                   "--batch_size", str(batch_size)]
            _append_passthrough(cmd, blocks, "batch_star_count", "star")
            rc = _run(cmd, dry=ns.dry_run)
            if rc != 0:
                return rc
            if not ns.dry_run:
                (d_star / ".batch_star_count.done").write_text("done\n", encoding="utf-8")
        else:
            print("[resume] batch_star_count skipped.")

        # 3b) merge_star_count (cwd=02-star/)
        if not (ns.resume and _nonempty(d_star / ".merge_star_count.done")):
            # Collect passthrough args that may contain --project
            merge_args = (blocks.get("merge_star_count") or []) + (blocks.get("star") or [])
            has_project = any(tok == "--project" for tok in merge_args)

            cmd = ["iobrpy", "merge_star_count",
                   "--path", str(d_star)]

            # Only use default project name when user did NOT provide one
            if not has_project:
                cmd += ["--project", "runall"]

            _append_passthrough(cmd, blocks, "merge_star_count", "star")
            rc = _run(cmd, cwd=d_star, dry=ns.dry_run)
            if rc != 0:
                return rc
            if not ns.dry_run:
                (d_star / ".merge_star_count.done").write_text("done\n", encoding="utf-8")
        else:
            print("[resume] merge_star_count skipped.")

        merged_star_counts = _find_latest(d_star, ["*_star_ReadsPerGene.tsv", "*_star_ReadsPerGene.tsv.gz", "*.STAR.count*.gz"])
        if ns.dry_run and merged_star_counts is None:
            merged_star_counts = d_star / "runall_star_ReadsPerGene.tsv"
        if merged_star_counts is None:
            print("[ERROR] Cannot find merged STAR ReadsPerGene in '02-star/' (pattern '*_star_ReadsPerGene.tsv*').")
            return 2

        # 4b) count2tpm -> 03-tpm/count2tpm.csv, then log2_eset -> 03-tpm/tpm_matrix.csv
        prep_csv   = d_tpm / "count2tpm.csv"
        tpm_matrix = d_tpm / "tpm_matrix.csv"

        if reusable("count2tpm", prep_csv) and reusable("log2_eset", tpm_matrix):
            print("[resume] count2tpm + log2_eset skipped.")
        else:
            # Run count2tpm only if the intermediate is missing (resume-friendly).
            if not reusable("count2tpm", prep_csv):
                cmd = ["iobrpy", "count2tpm",
                       "--input", str(merged_star_counts),
                       "--output", str(prep_csv),
                       "--idtype", "ensembl",
                       "--org", "hsa",
                       "--source", "local"]
                # Default: invoke --remove_version unless user already set it
                c2_args = blocks.get("count2tpm") or []
                if not any(a.startswith("--remove_version") for a in c2_args):
                    cmd.append("--remove_version")
                _append_passthrough(cmd, blocks, "count2tpm")
                rc = run_table(cmd, prep_csv)
                if rc != 0:
                    return rc

            # Always ensure final matrix is log2(x+1) from the intermediate.
            cmd = ["iobrpy", "log2_eset",
                   "-i", str(prep_csv),
                   "-o", str(tpm_matrix)]
            rc = run_table(cmd, tpm_matrix)
            if rc != 0:
                return rc

    # 5) calculate_sig_score -> 04-signatures/
    sig_out = d_sigscore / "calculate_sig_score.csv"
    if reusable("calculate_sig_score", sig_out):
        print("[resume] calculate_sig_score skipped.")
    else:
        cmd = ["iobrpy", "calculate_sig_score",
               "--input", str(tpm_matrix),
               "--output", str(sig_out),
               "--parallel_size", str(threads)]
        # Defaults for calculate_sig_score
        cs_args = (blocks.get("calculate_sig_score") or []) + (blocks.get("sig_score") or [])
        if not any(a.startswith("--signature") for a in cs_args):
            cmd += ["--signature", "all"]
        if not any(a.startswith("--method") for a in cs_args):
            cmd += ["--method", "integration"]
        if not any(a.startswith("--mini_gene_count") for a in cs_args):
            cmd += ["--mini_gene_count", "2"]
        if not any(a.startswith("--adjust_eset") for a in cs_args):
            cmd += ["--adjust_eset"]
        _append_passthrough(cmd, blocks, "calculate_sig_score", "sig_score")
        rc = run_table(cmd, sig_out)
        if rc != 0:
            return rc

    # 6) Deconvolution (6 methods) -> 05-tme/
    if not ns.dry_run and not _nonempty(tpm_matrix):
        print("[ERROR] TPM matrix missing. Abort before deconvolution.")
        return 2

    produced: List[Path] = []
    for m in ["cibersort", "IPS", "estimate", "mcpcounter", "quantiseq", "epic"]:
        out_file = d_deconv / f"{m}_results.csv"
        if reusable(m, out_file):
            print(f"[resume] {m} skipped.")
            produced.append(out_file)
            continue

        if m == "cibersort":
            cmd = ["iobrpy", "cibersort", "--input", str(tpm_matrix), "--output", str(out_file),
                   "--threads", str(threads)]
        elif m == "IPS":
            cmd = ["iobrpy", "IPS", "--input", str(tpm_matrix), "--output", str(out_file)]
        elif m == "estimate":
            cmd = ["iobrpy", "estimate", "--input", str(tpm_matrix), "--platform", "affymetrix", "--output", str(out_file)]
        elif m == "mcpcounter":
            cmd = ["iobrpy", "mcpcounter", "--input", str(tpm_matrix), "--features", "HUGO_symbols", "--output", str(out_file)]
        elif m == "quantiseq":
            cmd = ["iobrpy", "quantiseq", "--input", str(tpm_matrix), "--output", str(out_file)]
            # Defaults: enable arrays/tumor/scale_mrna unless user set them explicitly
            q_args = blocks.get("quantiseq") or []
            if not any(a == "--arrays" for a in q_args):
                cmd.append("--arrays")
            if not any(a == "--tumor" for a in q_args):
                cmd.append("--tumor")
            if not any(a.startswith("--scale_mrna") for a in q_args) and not any(a.startswith("--mRNAscale") for a in q_args):
                cmd.append("--scale_mrna")
        else:
            cmd = ["iobrpy", "epic", "--input", str(tpm_matrix), "--reference", "TRef", "--output", str(out_file)]

        _append_passthrough(cmd, blocks, m)
        rc = run_table(cmd, out_file)
        if rc != 0:
            return rc
        produced.append(out_file)

    # 7) Merge deconvolution results -> 05-tme/deconvo_merged.csv
    merged_wide_path = d_deconv / "deconvo_merged.csv"

    if ns.dry_run:
        print(f"[dry-run] merge deconvolution -> {merged_wide_path}")
    elif reusable("merge_deconvolution", merged_wide_path):
        print("[resume] merge deconvolution skipped.")
    else:
        if pd is None:
            print("[WARN] pandas is not available; skip merged deconvolution table.")
        else:
            completed_steps.pop("merge_deconvolution", None)
            def _read_csv_any(p: Path):
                """Read a CSV with a safe fallback parser."""
                try:
                    return pd.read_csv(p)
                except Exception:
                    return pd.read_csv(p, engine="python")

            def _normalize_id(df: pd.DataFrame) -> pd.DataFrame:
                """
                Standardize the sample identifier column to 'ID' and cleanup.
                - Accept 'ID' or 'Unnamed: 0' as the sample column; otherwise use the first column.
                - Drop redundant 'Unnamed:*' columns.
                - Ensure 'ID' is string type and appears as the first column.
                """
                if "ID" in df.columns:
                    pass
                elif "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "ID"})
                else:
                    df = df.rename(columns={df.columns[0]: "ID"})

                drop_cols = [c for c in df.columns if c.startswith("Unnamed:") and c != "ID"]
                if drop_cols:
                    df = df.drop(columns=drop_cols)

                df["ID"] = df["ID"].astype(str)
                cols = ["ID"] + [c for c in df.columns if c != "ID"]
                return df[cols]

            # Read all produced method outputs (paths are assumed in `produced`)
            frames = []
            for f in produced:
                df = _read_csv_any(f)
                df = _normalize_id(df)
                # Drop columns that are entirely NaN
                df = df.dropna(axis=1, how="all")
                frames.append(df)

            # Outer-join on 'ID' to build a single wide matrix per sample
            wide = frames[0]
            for df in frames[1:]:
                # If different methods accidentally share identical column names,
                # suffix the incoming duplicates to avoid collisions.
                overlap = (set(wide.columns) & set(df.columns)) - {"ID"}
                if overlap:
                    df = df.rename(columns={c: f"{c}_dup" for c in overlap})
                wide = pd.merge(wide, df, on="ID", how="outer")

            # Sort by ID and write the final wide table
            wide = wide.sort_values("ID").reset_index(drop=True)
            wide.to_csv(merged_wide_path, index=False)
            record_completed("merge_deconvolution", merged_wide_path)
            print(f"[ok] merged deconvolution -> {merged_wide_path}")

    # 8) LR_cal -> 06-LR_cal/
    lr_out = d_lrcal / "lr_cal.csv"
    if reusable("LR_cal", lr_out):
        print("[resume] LR_cal skipped.")
    else:
        cmd = ["iobrpy", "LR_cal",
               "--input", str(tpm_matrix),
               "--output", str(lr_out),
               "--data_type", "tpm",
               "--id_type", "symbol",
               "--cancer_type", "pancan",
               "--verbose"]
        _append_passthrough(cmd, blocks, "LR_cal")
        rc = run_table(cmd, lr_out)
        if rc != 0:
            return rc
    # 9) TRUST4 TCR/BCR repertoire -> 07-TCRBCR/
    tcrbcr_done_flag = d_tcrbcr / ".trust4.done"
    if ns.mode == "star":
        trust4_input_root = d_star
        trust4_cli = ["-b", str(trust4_input_root)]
    else:
        trust4_input_root = d_fastp
        trust4_cli = ["--fqdir", str(trust4_input_root)]

    if ns.resume and tcrbcr_done_flag.exists() and _nonempty(d_tcrbcr):
        print("[resume] trust4 skipped (07-TCRBCR/ already has outputs).")
    else:
        cmd = ["iobrpy", "trust4"] + trust4_cli + [
            "-o", str(d_tcrbcr),
            "-t", str(threads),
        ]
        _append_passthrough(cmd, blocks, "trust4")
        rc = _run(cmd, dry=ns.dry_run)
        if rc != 0:
            return rc
        if not ns.dry_run:
            tcrbcr_done_flag.write_text("done\n", encoding="utf-8")

    print("\n[done] runall finished.")
    return 0


# External products consumed by later stages or used to establish completion.
# Table outputs are taken from completed_steps, not guessed by extension.
_EXTERNAL_PRODUCTS = {
    "01-qc": (".fastq_qc.done", "*.task.complete", "task.complete", "*.iobrx.json",
              "*.fastq", "*.fastq.gz", "*.fq", "*.fq.gz", "*_fastp.json"),
    "02-salmon": (".batch_salmon.done", ".merge_salmon.done", "task.complete", "*.iobrx.json",
                  "quant.sf", "*_salmon_tpm.tsv", "*_salmon_tpm.tsv.gz",
                  "*_salmon_count.tsv", "*_salmon_count.tsv.gz"),
    "02-star": (".batch_star_count.done", ".merge_star_count.done", "*.task.complete", "*.iobrx.json",
                "*.bam", "*.bai", "*ReadsPerGene.out.tab", "*_star_ReadsPerGene.tsv",
                "*_star_ReadsPerGene.tsv.gz", "*.STAR.count*.gz"),
    "07-TCRBCR": (".trust4.done", "*.TRUST4.done", "*_report.tsv", "*_annot.fa", "*_cdr3.out",
                  "trust4_immdata.csv", "trust4_immune_indices.csv"),
}


def _external_product(relative):
    path = Path(relative)
    return len(path.parts) > 1 and any(
        fnmatch.fnmatchcase(path.name, pattern)
        for pattern in _EXTERNAL_PRODUCTS.get(path.parts[0], ()))


def _pipeline_product_paths(root, completed_steps):
    """Known calculation products, including custom filenames in stage records."""
    paths = set()

    def include(records):
        for record in records:
            path = Path(record["path"])
            # Stage sidecars use absolute paths. Do not inventory other runs.
            if path.is_relative_to(root):
                paths.add(str(path.relative_to(root)))

    for records in completed_steps.values():
        include(records)
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if not _external_product(relative):
            continue
        paths.add(relative)
        if path.name.endswith(".iobrx.json") and path.is_file():
            try:
                include(json.loads(path.read_text())["outputs"])
            except (OSError, ValueError, KeyError, TypeError):
                # The sidecar itself remains checked; malformed state cannot
                # authorize a skip in the per-sample execution helpers.
                pass
    return paths


def _pipeline_products(root, paths):
    return {relative: file_hash(root / relative) for relative in sorted(paths)
            if (root / relative).is_file()}


def _guarded_runall(argv):
    # Reserve a run directory for one input/configuration. Legacy outputs cannot
    # be trusted merely because they contain flags with familiar filenames.
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outdir")
    parser.add_argument("--fastq")
    parser.add_argument("--mode")
    parser.add_argument("--dry_run", action="store_true")
    ns, _ = parser.parse_known_args(args)
    if ns.dry_run or not ns.outdir or not ns.fastq or ns.mode not in {"salmon", "star"}:
        return _main_impl(args)
    root = Path(ns.outdir).resolve()
    state_path = root / ".iobrx-run-state.json"
    inputs = {Path(ns.fastq).resolve()}
    for token in args:
        candidate = Path(token.partition("=")[2] if token.startswith("--") and "=" in token else token)
        if not str(candidate).startswith("-") and candidate.exists():
            candidate = candidate.resolve()
            if candidate != root and not candidate.is_relative_to(root):
                if root.is_relative_to(candidate):
                    raise ValueError("outdir must be outside input and reference directories")
                inputs.add(candidate)
    import shutil
    tools = [x for x in ("fastp", "multiqc", "salmon", "STAR", "run-trust4", "samtools") if shutil.which(x)]
    current = signature(sorted(inputs), {"argv": [a for a in args if a != "--resume"]}, tools)
    completed_steps = {}
    if root.exists() and any(root.iterdir()):
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("Existing run has no verifiable state; choose a new outdir") from None
        if previous.get("schema_version") != 2 or not isinstance(previous.get("completed_steps"), dict):
            raise ValueError("Existing run has no per-step completion records; choose a new outdir")
        # Schema 2 formerly recorded the entire tree. Project those records
        # onto calculation products so existing runs need no forced migration.
        paths = _pipeline_product_paths(root, previous["completed_steps"])
        expected = {path: digest for path, digest in previous.get("outputs", {}).items()
                    if path in paths or _external_product(path)}
        if previous.get("signature") != current or expected != _pipeline_products(root, paths):
            raise ValueError("Run inputs, references, parameters or outputs changed; choose a new outdir")
        if "--resume" in args:
            completed_steps = previous["completed_steps"]
    root.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 2, "signature": current, "status": "running",
             "outputs": _pipeline_products(root, _pipeline_product_paths(root, completed_steps)),
             "completed_steps": completed_steps}
    write_json(state_path, state)
    rc = 1
    try:
        rc = _main_impl(args, completed_steps=completed_steps)
        return rc
    finally:
        state.update(status="completed" if rc == 0 else "failed", exit_code=rc,
                     outputs=_pipeline_products(root, _pipeline_product_paths(root, completed_steps)))
        write_json(state_path, state)


def runall_argv(argv: Optional[List[str]] = None, verbose: bool = True) -> int:
    """Library entry: run the orchestrator over an argv-style token list.

    Returns the process exit code the upstream CLI produces (0 on success;
    failing step's rc; 2 for usage errors / missing merged matrix) instead
    of calling ``sys.exit``.
    """
    global _SUBSTEP_VERBOSE
    try:
        with _RUN_LOCK:
            _SUBSTEP_VERBOSE = bool(verbose)
            return _guarded_runall(argv)
    except SystemExit as e:  # argparse usage errors keep their exit code
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 1)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI-compatible entry, mirroring ``iobrpy.workflow.runall.main``
    (including its ``sys.exit(rc)`` failure behaviour)."""
    rc = runall_argv(argv)
    if rc:
        sys.exit(rc)


def runall_original(argv: Optional[List[str]] = None) -> int:
    """backend='python' escape hatch: the UNTOUCHED upstream orchestrator
    (``iobrpy.workflow.runall.main``), which spawns the ``iobrpy`` child CLIs
    through ``subprocess`` (requires the ``iobrpy`` console script and the
    external tools on PATH). Returns the upstream exit code."""
    from iobrpy.workflow import runall as runall_mod
    try:
        runall_mod.main(list(argv) if argv is not None else None)
        return 0
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 1)


if __name__ == "__main__":
    main()
