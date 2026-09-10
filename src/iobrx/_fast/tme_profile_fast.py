"""tme_profile_fast: single-process re-orchestration of IOBRpy's tme_profile CLI.

The original ``iobrpy tme_profile`` (``iobrpy/workflow/tme_profile.py``) shells
out to serial ``subprocess.run(["iobrpy", <subcmd>, ...])`` calls — one cold
interpreter + import start (~1.4-2.2 s each) per sub-step.  This module runs
the SAME chain in ONE process, replacing sub-steps with the iobrx bit-exact
kernels where they exist and calling the ORIGINAL code in-process where they
do not:

  step  sub-command            replacement here
  ----  ---------------------  ------------------------------------------------
  1     calculate_sig_score    ``iobrx.calculate_sig_score`` (bit-exact fast)
  2     cibersort              ORIGINAL ``cibersort(input_path, ...)`` called
                               in-process on the ORIGINAL file path by default
                               (``cibersort_backend='original'``) — see the
                               knife-edge + parser-roundtrip notes below
  3     IPS                    ORIGINAL ``iobrpy.workflow.IPS.main()`` via the
                               same ``sys.argv`` bridge ``iobrpy.main`` uses
                               (no function entry exists upstream)
  4     estimate               ``iobrx.estimate_score`` (bit-exact fast)
  5     mcpcounter             ORIGINAL ``preprocess_input`` (file cleaning is
                               part of the contract) + ``iobrx.mcpcounter``
  6     quantiseq              ``iobrx.quantiseq`` (original numerics with the
                               memoized HGNC alias map)
  7     epic                   ``iobrx.epic`` on the CLI-COMPOSED reference
                               (see ``_compose_epic_reference``: the raw pkl
                               TRef dict lacks ``var_present``/``mRNA_cell``,
                               so composing exactly like ``epic.main()`` does
                               is REQUIRED for byte parity)
  8     (in-process merge)     ORIGINAL ``_read_csv_any`` / ``_normalize_id``
                               helpers reading the six deconvolution CSVs BACK
                               from disk, verbatim merge loop
  9     LR_cal                 ``iobrx.lr_cal`` (R6/v3: Rust-accelerated
                               drop-in, output sha256-equal to the ORIGINAL
                               ``iobrpy.workflow.LR_cal.LR_cal()``)

CIBERSORT knife-edge note (measured on the frozen TPM_stad10 input)
-------------------------------------------------------------------
Sample TCGA-BR-A4IV's mixture column is tie-heavy (11,030 exact zeros plus
large duplicate clusters), and its nu-SVR solve sits at the solver tolerance
knife edge.  TWO independent perturbations flip it to a different (valid,
deterministic) solution basin — chosen-nu RMSE 1.041405 vs the gold's
1.041700; 13 cells differ, max abs diff 2.5e-3:

1. the native (rust/libsvm) backend of ``iobrx.cibersort``, run on the exact
   frame the original parses from this file (thread-invariant: identical
   deviation at n_threads 1/4/16; the ORIGINAL python solver is bit-stable
   across n_jobs 1/4/16 and reproduces the gold bytes).  The official gate
   frames do not exhibit the divergence (gate: max abs diff 0.0), so this is
   an input-conditional divergence OUTSIDE the gate's coverage, not a gate
   regression.
2. the ``to_csv -> read_csv`` round trip of the ``backend='python'``
   fallback: pandas' float parser is not correctly rounded on the 17-digit
   text of this file (``float('1860154.9999834974')`` = ...974 while BOTH
   pandas engines parse ...972; and re-parsing pandas' own ``to_csv`` repr
   output shifts some cells another ulp), so the frame the fallback's
   original-solver call receives differs from the direct-read frame in
   2,830/501,810 cells — enough to flip the same knife-edge sample.

Because the tme_profile parity contract is byte equality with the original
CLI on the frozen input, the cibersort step defaults to
``cibersort_backend='original'``: the ORIGINAL ``cibersort()`` function
called in-process on the ORIGINAL input path — byte-identical to the CLI
branch by construction (same function, same file, unseeded P-values exactly
like upstream).  Pass ``cibersort_backend='auto'``/``'rust'`` for the native
solver (faster; may deviate on knife-edge samples; seeded P-values) or
``'python'`` for the iobrx fallback (inherits the round-trip perturbation).

Why file semantics are preserved instead of frame passing
---------------------------------------------------------
* Each CLI sub-step reads the input file with its OWN parser settings
  (``cibersort`` uses ``sep=None, engine='python'``; ``IPS`` defaults unknown
  extensions to ``,``, never sniffing; ``mcpcounter.preprocess_input`` strips
  version suffixes / uppercases / collapses duplicates by max; ``LR_cal`` and
  ``estimate`` use main.py's own extension rules).  A single shared in-memory
  frame could differ in the last ulp (C vs python float parsing) or in gene
  names, so every step here replicates its step's exact read call.
* The merge step consumes the WRITTEN CSVs: mcpcounter's ``float_format
  ='%.7f'``, IPS ``round(6)``, quantiseq's ``1e-8`` masking and cibersort's
  float32 weights are part of ``deconvo_merged.csv``'s bytes.  The six
  deconvolution CSVs are therefore written first and read back with the
  original helpers — a to_csv→read_csv round trip is NOT the identity and is
  deliberately reproduced rather than shortcut.
* Post-write transformations from ``iobrpy.main`` (``_CIBERSORT`` / ``_IPS`` /
  ``_estimate`` / ``_MCPcounter`` / ``_quantiseq`` / ``_EPIC`` suffixing,
  transposes, ``index_label='ID'``, per-branch separator choices) are
  replicated verbatim per step, including each branch's own quirks.

Determinism and the ``parallel`` extra
--------------------------------------
Sequential mode (the contract path) reproduces the original step order and
every output file byte-identically on the frozen input, EXCEPT the
``P-value_CIBERSORT`` column of ``cibersort_results.csv`` /
``deconvo_merged.csv`` (unseeded upstream by design).  ``parallel=True``
(off by default) additionally splits the calculate_sig_score integration
legs across worker PROCESSES: the per-signature pandas glue (``set(index)``
+ ``.loc`` per signature, ~120 s of GIL-bound work over ~12.6k signatures on
the frozen input — the actual bottleneck; the rust ssGSEA leg is ~0.3 s)
runs chunk-wise in separate interpreters, and the independent light deconv
steps overlap in a thread pool.  Per-signature scoring is a pure,
order-independent function (the fast modules' own contract), and the parent
reassembly uses the verbatim ``_sig_score_integration_fast`` statements, so
the output bytes are unchanged — verified byte-identical to the gold CSV.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from iobrx._resources import resource_path

__all__ = ["tme_profile_fast"]


# ---------------------------------------------------------------------------
# per-step input readers — each replicates ONE CLI branch verbatim
# ---------------------------------------------------------------------------
def _read_sig_score(path):
    """iobrpy.main calculate_sig_score branch: ext-based sep, index_col=0."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, sep=",", index_col=0)
    if ext == ".txt":
        return pd.read_csv(path, sep="\t", index_col=0)
    return pd.read_csv(path, sep=None, engine="python", index_col=0)


def _read_cibersort(path):
    """iobrpy.workflow.cibersort.cibersort: sniffing python-engine read."""
    return pd.read_csv(path, sep=None, engine="python", index_col=0)


def _read_estimate(path):
    """iobrpy.main estimate branch: tab for .tsv/.txt else comma."""
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep, index_col=0)


def _infer_sep(path):
    """iobrpy.workflow.quantiseq.infer_sep == iobrpy.workflow.epic.infer_sep."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".csv":
        return ","
    if ext in (".tsv", ".txt"):
        return "\t"
    return None


# ---------------------------------------------------------------------------
# EPIC reference composition — verbatim replication of epic.main()'s block.
# The raw packaged TRef/BRef dicts carry only refProfiles / refProfiles.var /
# sigGenes; the CLI adds var_present=True, mRNA_cell_default and the
# sigGenes∩refProfiles filter BEFORE calling EPIC() — and EPIC's variance
# weighting hinges on var_present, so passing the raw dict would silently
# produce different (unweighted) cell fractions.
# ---------------------------------------------------------------------------
_EPIC_REF_CACHE: dict[str, dict] = {}


def _compose_epic_reference(reference: str) -> dict:
    ent = _EPIC_REF_CACHE.get(reference)
    if ent is not None:
        return ent
    import pickle
    from importlib.resources import files

    from iobrpy.workflow.epic import _to_df, merge_duplicates, mRNA_cell_default

    ref_pkg = resource_path("epic_TRef_BRef.pkl")
    with ref_pkg.open("rb") as f:
        ref_data = pickle.load(f)

    refs = []
    if reference in ("TRef", "both"):
        refs.append("TRef")
    if reference in ("BRef", "both"):
        refs.append("BRef")

    profs, vars_, flags, sgs = [], [], [], []
    for key in refs:
        dd = ref_data[key]
        profs.append(_to_df(dd["refProfiles"], dd))
        varr = dd.get("refProfiles.var")
        if varr is not None:
            flags.append(True)
            vars_.append(_to_df(varr, dd))
        else:
            flags.append(False)
        sgs.extend(dd.get("sigGenes", []))

    ref_profiles = pd.concat(profs, axis=1)
    ref_profiles = merge_duplicates(ref_profiles, "reference profiles").loc[
        :, ~ref_profiles.columns.duplicated()
    ]
    if any(flags):
        full_vars = []
        for present, vdf, prof in zip(flags, vars_, profs):
            full_vars.append(
                vdf if present
                else pd.DataFrame(0, index=prof.index, columns=prof.columns)
            )
        ref_vars = pd.concat(full_vars, axis=1).loc[:, ref_profiles.columns]
        var_present = True
    else:
        ref_vars = None
        var_present = False
    sig_ref = [g for g in dict.fromkeys(sgs) if g in ref_profiles.index]
    composed = {
        "refProfiles": ref_profiles,
        "refProfiles.var": ref_vars,
        "sigGenes": sig_ref,
        "mRNA_cell": mRNA_cell_default,
        "var_present": var_present,
    }
    # Strong module-level ref: epic_fast memoizes on id(reference).
    _EPIC_REF_CACHE[reference] = composed
    return composed


# ---------------------------------------------------------------------------
# sub-steps (each mirrors one CLI branch: read -> compute -> write)
# ---------------------------------------------------------------------------
def _step_sig_score(in_path, out_file, threads, signature, method,
                    mini_gene_count, adjust_eset, backend):
    import iobrx

    eset = _read_sig_score(str(in_path))
    names = [signature] if isinstance(signature, str) else list(signature)
    scores_df = iobrx.calculate_sig_score(
        eset, names, method,
        mini_gene_count=mini_gene_count, adjust_eset=adjust_eset,
        n_threads=threads, backend=backend,
    )
    scores_df.to_csv(out_file, index=False)


def _step_cibersort(in_path, out_file, threads, perm, QN, absolute,
                    abs_method, backend):
    if backend == "original":
        # Byte-exact contract route: the ORIGINAL solver on the ORIGINAL file
        # path — exactly what the CLI branch does (main.py passes
        # args.input_path straight through to cibersort()).  NO frame round
        # trip: pandas' read_csv float parser is not correctly rounded on
        # 17-digit text (float('1860154.9999834974') = ...974 while BOTH
        # pandas engines parse ...972), so to_csv -> read_csv is NOT
        # value-preserving (2,830/501,810 cells of the frozen TPM_stad10
        # input shift ~1 ulp), which alone flips the knife-edge sample
        # TCGA-BR-A4IV to the other solver basin.  iobrx.cibersort's
        # python-fallback (DataFrame -> temp CSV -> original) inherits that
        # flip; the native backend deviates on the same sample directly
        # (deterministic, thread-invariant; see module docstring).
        from iobrpy.workflow.cibersort import cibersort as original

        result_df = original(str(in_path), perm=perm, QN=QN, absolute=absolute,
                             abs_method=abs_method, n_jobs=threads)
    else:
        import iobrx

        eset = _read_cibersort(str(in_path))
        result_df = iobrx.cibersort(
            eset, perm=perm, QN=QN, absolute=absolute, abs_method=abs_method,
            n_threads=threads, backend=backend,
        )
    # iobrpy.main cibersort branch post-processing
    result_df.columns = [col + "_CIBERSORT" for col in result_df.columns]
    result_df.index.name = "ID"
    delim = "," if str(out_file).lower().endswith(".csv") else "\t"
    result_df.to_csv(out_file, sep=delim, index=True)


def _step_ips(in_path, out_file):
    """ORIGINAL IPS CLI main() in-process (sys.argv bridge, as iobrpy.main does).

    Must run on the calling thread when other threads are active (sys.argv is
    process-global); the sequential contract path trivially satisfies this and
    ``parallel=True`` keeps this step on the calling thread by construction.
    """
    from iobrpy.workflow.IPS import main as IPS_main

    argv_orig = sys.argv[:]
    try:
        sys.argv = [argv_orig[0] if argv_orig else "iobrpy",
                    "--input", str(in_path), "--output", str(out_file)]
        IPS_main()
    finally:
        sys.argv = argv_orig


def _step_estimate(in_path, out_file, platform):
    import iobrx

    in_df = _read_estimate(str(in_path))
    score_df = iobrx.estimate_score(in_df, platform=platform)
    # iobrpy.main estimate branch post-processing
    score_df = score_df.T
    score_df.columns = [col + "_estimate" for col in score_df.columns]
    out_sep = "\t" if str(out_file).lower().endswith((".tsv", ".txt")) else ","
    score_df.to_csv(out_file, sep=out_sep, index_label="ID")


def _step_mcpcounter(in_path, out_file, features):
    import iobrx
    from iobrpy.workflow.mcpcounter import preprocess_input

    expr_df = preprocess_input(str(in_path))
    scores_df = iobrx.mcpcounter(expr_df, features_type=features)
    # iobrpy.main mcpcounter branch post-processing
    out_df = scores_df.T
    out_df.columns = [
        col.replace(" ", "_") + "_MCPcounter" for col in out_df.columns
    ]
    out_ext = Path(out_file).suffix.lower()
    out_sep = "," if out_ext == ".csv" else "\t"
    out_df.to_csv(out_file, sep=out_sep, index_label="ID", float_format="%.7f")


def _step_quantiseq(in_path, out_file, arrays, signame, tumor, mRNAscale,
                    method, rmgenes):
    import iobrx

    in_sep = _infer_sep(in_path)
    mix = pd.read_csv(in_path, sep=in_sep, index_col=0)
    res = iobrx.quantiseq(
        mix, arrays=arrays, signame=signame, tumor=tumor,
        mRNAscale=mRNAscale, method=method, rmgenes=rmgenes,
    )
    # iobrpy.workflow.quantiseq.main() post-processing
    out_sep = _infer_sep(out_file) or "\t"
    res.columns = [
        "ID" if col == "Sample"
        else f"{col.replace('.', '_')}_quantiseq"
        for col in res.columns
    ]
    eps = 1e-8
    num_cols = res.columns.drop("ID")
    res[num_cols] = res[num_cols].mask(res[num_cols].abs() < eps, 0)
    res.to_csv(out_file, sep=out_sep, index=False)


def _step_epic(in_path, out_file, reference):
    import iobrx

    sep_in = _infer_sep(in_path)
    bulk = pd.read_csv(in_path, sep=sep_in, index_col=0)
    ref = _compose_epic_reference(reference)
    # epic.main() heuristic: no overlap with sigGenes -> genes are in columns
    if not set(bulk.index).intersection(ref["sigGenes"]):
        warnings.warn("Detected genes in columns; transposing bulk.")
        bulk = bulk.T
    res = iobrx.epic(
        bulk, reference=ref,
        mRNA_cell=None, mRNA_cell_sub={}, sig_genes=None,
        scale_exprs=True, with_other_cells=True, constrained_sum=True,
        range_based_optim=False, solver="trust-constr",
        init_jitter=0.0, unlog_bulk=False,
    )
    # epic.main() output: cellFractions, '_EPIC' suffix, index written
    ext = os.path.splitext(str(out_file))[1].lower()
    sep_out = "," if ext == ".csv" else "\t"
    out_df = res["cellFractions"].copy()
    out_df.columns = [f"{col}_EPIC" for col in out_df.columns]
    out_df.to_csv(out_file, sep=sep_out, index=True)


def _merge_deconv(produced, merged_path):
    """Original tme_profile.main() merge block, verbatim (helpers imported)."""
    from iobrpy.workflow.tme_profile import _normalize_id, _read_csv_any

    frames = []
    for f in produced:
        df = _read_csv_any(f)
        df = _normalize_id(df)
        df = df.dropna(axis=1, how="all")
        frames.append(df)
    if frames:
        wide = frames[0]
        for df in frames[1:]:
            overlap = (set(wide.columns) & set(df.columns)) - {"ID"}
            if overlap:
                df = df.rename(columns={c: f"{c}_dup" for c in overlap})
            wide = pd.merge(wide, df, on="ID", how="outer")
        wide = wide.sort_values("ID").reset_index(drop=True)
        wide.to_csv(merged_path, index=False)


def _step_lr_cal(in_path, out_file, data_type, id_type, cancer_type, verbose):
    """iobrx.lr_cal — accelerated drop-in for the ORIGINAL LR_cal (R6 / v3).

    Was: ``iobrpy.workflow.LR_cal.LR_cal()`` called in-process (~4.1 s of the
    frozen TPM_stad10 chain, 81% of it four per-gene pandas filter passes).
    Now: ``iobrx.lr_cal`` (backend='auto' → the Rust ``lr_gene_valid_mask``
    gene filter + ``np.fmin`` pair gather), whose output CSV is sha256-equal
    to the ORIGINAL's on the frozen official inputs (R3 verification: stad
    TPM 54,658x10 → 10x773 ``a49ddff5…``, eset_stad_symbol 50,181x10,
    imvigor210 872x348, ENSG degenerate 10x0), including the ORIGINAL's
    ``detect_sep`` path-read semantics and the ``insert(0,'ID',…)``+``to_csv``
    tail.  ``n_threads`` is left at the wrapper default (None → min(8, cpu);
    the per-row computation is pure, so results are thread-count invariant).

    NOTE: the tme_profile CLI injects ``--data_type tpm``; the FUNCTION's own
    default is 'count' (which would prepend count2tpm) — the injected value is
    what must be passed, exactly like the CLI path does.
    """
    import iobrx

    iobrx.lr_cal(str(in_path), str(out_file), data_type=data_type,
                 id_type=id_type, cancer_type=cancer_type, verbose=verbose)


# ---------------------------------------------------------------------------
# parallel extra: process-split calculate_sig_score('integration')
#
# The integration legs are pure per-signature functions over ONE shared
# preprocessed frame (sig_score_fast's own contract: "pure functions ->
# thread-safe, order-stable").  Splitting the filtered signature dict into
# ordered chunks and running _pca_one/_zscore_one per chunk in worker
# PROCESSES bypasses the GIL that serializes the ~120 s of per-signature
# pandas glue; the parent reassembles with the verbatim
# _sig_score_integration_fast statements (column order = chunk order = dict
# order; TME contrasts appended per leg on the FULL filtered key set, exactly
# where the unsplit legs put them).  Verified byte-identical to the gold
# calculate_sig_score.csv on the frozen TPM_stad10 input.
# ---------------------------------------------------------------------------
def _chunk_items(items, n_chunks):
    """Ordered contiguous chunks of list(items) (empty chunks dropped)."""
    n = len(items)
    n_chunks = max(1, min(n_chunks, n))
    size = -(-n // n_chunks)  # ceil
    return [items[i:i + size] for i in range(0, n, size)]


def _worker_pca_chunk(eset2, items):
    from iobrx._fast import sig_score_fast as F

    return [(name, F._pca_one(eset2, name, genes)) for name, genes in items]


def _worker_zscore_chunk(eset2, items):
    from iobrx._fast import sig_score_fast as F

    return [(name, F._zscore_one(eset2, name, genes)) for name, genes in items]


def _worker_ssgsea(eset2, sig_dict, mini_gene_count, threads):
    from iobrx._fast import sig_score_fast as F

    res2d = F._ssgsea_res2d_shared(eset2, sig_dict, mini_gene_count, threads)
    return F._nes_from_res2d(res2d)


def _step_sig_score_split(in_path, out_file, threads, signature, method,
                          mini_gene_count, adjust_eset, n_procs):
    """Process-split integration; falls back to the sequential step for
    non-integration methods or when gseapy is missing (upstream degrade)."""
    from joblib import Parallel, delayed

    from iobrpy.workflow.calculate_sig_score import (
        _merge_signature_groups,
        filter_signatures,
        preprocess_eset,
    )
    from iobrx._fast import sig_score_fast as F

    if str(method).lower() != "integration":
        # legs differ per method; keep the public single-call path
        return _step_sig_score(in_path, out_file, threads, signature, method,
                               mini_gene_count, adjust_eset, "auto")

    from importlib.resources import files

    eset = _read_sig_score(str(in_path))
    names = [signature] if isinstance(signature, str) else list(signature)
    all_sigs = pd.read_pickle(resource_path("calculate_data.pkl"))
    sig_dict = _merge_signature_groups(all_sigs, names)
    if not isinstance(sig_dict, dict) or len(sig_dict) == 0:
        raise KeyError(f"No valid signatures found from groups: {names}")

    # verbatim _sig_score_integration_fast head: RAW-index filter, ONE preprocess
    filtered_sigs = {
        name: [g for g in genes if g in eset.index]
        for name, genes in sig_dict.items()
        if len([g for g in genes if g in eset.index]) >= mini_gene_count
    }
    eset2 = preprocess_eset(eset, adjust_eset)
    min_size = max(mini_gene_count, 2)
    sigs = filter_signatures(filtered_sigs, eset2, min_size)
    items = list(sigs.items())

    p_chunks = _chunk_items(items, n_procs)
    z_chunks = _chunk_items(items, n_procs)

    gp_missing = F.gp is None
    if gp_missing or not filtered_sigs:
        # upstream degrade path: PCA + zscore only (same RuntimeWarning text
        # is emitted by the original; keep behavior, skip the ssgsea worker)
        ssgsea_job = None
    else:
        ssgsea_job = delayed(_worker_ssgsea)(eset2, filtered_sigs,
                                             mini_gene_count, threads)

    tasks = ([delayed(_worker_pca_chunk)(eset2, c) for c in p_chunks]
             + [delayed(_worker_zscore_chunk)(eset2, c) for c in z_chunks]
             + ([ssgsea_job] if ssgsea_job is not None else []))
    results = Parallel(n_jobs=int(n_procs), prefer="processes")(tasks)
    n_p = len(p_chunks)
    n_z = len(z_chunks)
    p_res, z_res = results[:n_p], results[n_p:n_p + n_z]
    nes = results[n_p + n_z] if ssgsea_job is not None else None

    if gp_missing or not filtered_sigs:
        reason = (
            "gseapy is not available"
            if gp_missing
            else f"no signatures passed the mini_gene_count={mini_gene_count} filter"
        )
        warnings.warn(
            "Skipping ssGSEA in integration scoring because "
            f"{reason}; returning PCA and z-score results only.",
            RuntimeWarning,
            stacklevel=2,
        )

    # verbatim reassembly (same statements/order as the unsplit legs)
    pdata = pd.DataFrame({"ID": eset.columns})
    for chunk in p_res:
        for (_name, vec) in chunk:
            name, values = vec
            pdata[name] = values
    pdata = F._attach_tme_contrasts(pdata, sigs)
    p = pdata.set_index("ID").add_suffix("_PCA")

    zdata = pd.DataFrame({"ID": eset.columns})
    for chunk in z_res:
        for (_name, vec) in chunk:
            name, values = vec
            zdata[name] = values
    zdata = F._attach_tme_contrasts(zdata, sigs)
    z = zdata.set_index("ID").add_suffix("_zscore")

    if nes is None:
        out = pd.concat([p, z], axis=1).reset_index()
    else:
        s = nes.set_index("ID").add_suffix("_ssGSEA")
        out = pd.concat([p, z, s], axis=1).reset_index()
    out.to_csv(out_file, index=False)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def tme_profile_fast(
    input,
    output,
    threads: int = 1,
    *,
    signature="all",
    sig_method: str = "integration",
    mini_gene_count: int = 2,
    adjust_eset: bool = True,
    perm: int = 100,
    QN: bool = True,
    absolute: bool = False,
    abs_method: str = "sig.score",
    platform: str = "affymetrix",
    features: str = "HUGO_symbols",
    arrays: bool = True,
    signame: str = "TIL10",
    tumor: bool = True,
    mRNAscale: bool = True,
    quantiseq_method: str = "lsei",
    rmgenes: str = "unassigned",
    epic_reference: str = "TRef",
    data_type: str = "tpm",
    id_type: str = "symbol",
    cancer_type: str = "pancan",
    lr_verbose: bool = True,
    backend: str = "auto",
    cibersort_backend: str = "original",
    parallel: bool = False,
    sig_procs: int | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """Run the full tme_profile chain in one process; returns output paths.

    Parameter defaults are the values the ORIGINAL orchestrator injects into
    each sub-command's CLI when the user does not override them (verified
    against ``iobrpy.main`` branch defaults, which differ from the workflow
    functions' own defaults — e.g. LR_cal CLI data_type='tpm' vs function
    'count').  ``threads`` mirrors ``--threads`` (default 1) and feeds
    calculate_sig_score's ``parallel_size`` and cibersort's ``n_jobs``, as
    upstream.  ``backend`` selects the native kernels for calculate_sig_score;
    ``cibersort_backend`` defaults to 'original' (the ORIGINAL solver called
    in-process on the original file path) because BOTH the native backend and
    the python-fallback's CSV round trip deviate on a knife-edge sample of the
    frozen tme_profile input — see the module docstring.  Any step failure
    raises (the original hard-stops the chain on the first non-zero rc; there
    is no resume, and neither is there here).
    """
    in_path = Path(input).resolve()
    outdir = Path(output).resolve()
    d_sig = outdir / "01-signatures"
    d_tme = outdir / "02-tme"
    d_lr = outdir / "03-LR_cal"
    for d in (d_sig, d_tme, d_lr):
        d.mkdir(parents=True, exist_ok=True)

    sig_out = d_sig / "calculate_sig_score.csv"
    cs_out = d_tme / "cibersort_results.csv"
    ips_out = d_tme / "IPS_results.csv"
    est_out = d_tme / "estimate_results.csv"
    mcp_out = d_tme / "mcpcounter_results.csv"
    qs_out = d_tme / "quantiseq_results.csv"
    epic_out = d_tme / "epic_results.csv"
    merged_out = d_tme / "deconvo_merged.csv"
    lr_out = d_lr / "lr_cal.csv"
    # merge input order == the original's `produced` order
    produced = [cs_out, ips_out, est_out, mcp_out, qs_out, epic_out]

    def _run(name, fn, *a, **k):
        t = time.perf_counter()
        fn(*a, **k)
        if verbose:
            print(f"[iobrx tme_profile] [ok] {name} "
                  f"({time.perf_counter() - t:.2f}s)", flush=True)

    t0 = time.perf_counter()
    if not parallel:
        # ---- contract path: original step order, one process ----
        _run("calculate_sig_score", _step_sig_score, in_path, sig_out, threads,
             signature, sig_method, mini_gene_count, adjust_eset, backend)
        _run("cibersort", _step_cibersort, in_path, cs_out, threads, perm, QN,
             absolute, abs_method, cibersort_backend)
        _run("IPS", _step_ips, in_path, ips_out)
        _run("estimate", _step_estimate, in_path, est_out, platform)
        _run("mcpcounter", _step_mcpcounter, in_path, mcp_out, features)
        _run("quantiseq", _step_quantiseq, in_path, qs_out, arrays, signame,
             tumor, mRNAscale, quantiseq_method, rmgenes)
        _run("epic", _step_epic, in_path, epic_out, epic_reference)
        _run("merge", _merge_deconv, produced, merged_out)
        _run("LR_cal", _step_lr_cal, in_path, lr_out, data_type, id_type,
             cancer_type, lr_verbose)
    else:
        # ---- extra: overlap independent work; output bytes unchanged ----
        # sig_score first (dominant): its integration legs are split across
        # `sig_procs` worker processes (GIL-bound per-signature glue becomes
        # truly parallel; rust ssGSEA leg ~0.3 s rides along).  Then the six
        # light steps: five in a thread pool while IPS runs on the calling
        # thread (its sys.argv bridge is process-global), merge last.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_procs = int(sig_procs) if sig_procs else max(2, min(int(threads), 12))
        _run("calculate_sig_score(split)", _step_sig_score_split, in_path,
             sig_out, threads, signature, sig_method, mini_gene_count,
             adjust_eset, n_procs)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = [
                ex.submit(_step_cibersort, in_path, cs_out, threads, perm, QN,
                          absolute, abs_method, cibersort_backend),
                ex.submit(_step_estimate, in_path, est_out, platform),
                ex.submit(_step_mcpcounter, in_path, mcp_out, features),
                ex.submit(_step_quantiseq, in_path, qs_out, arrays, signame,
                          tumor, mRNAscale, quantiseq_method, rmgenes),
                ex.submit(_step_epic, in_path, epic_out, epic_reference),
                ex.submit(_step_lr_cal, in_path, lr_out, data_type, id_type,
                          cancer_type, lr_verbose),
            ]
            _run("IPS", _step_ips, in_path, ips_out)
            for fut in as_completed(futs):
                fut.result()  # re-raise: hard-stop like the original
        _run("merge", _merge_deconv, produced, merged_out)
    if verbose:
        print(f"[iobrx tme_profile] [done] total {time.perf_counter() - t0:.2f}s",
              flush=True)

    return {
        "calculate_sig_score": str(sig_out),
        "cibersort": str(cs_out),
        "IPS": str(ips_out),
        "estimate": str(est_out),
        "mcpcounter": str(mcp_out),
        "quantiseq": str(qs_out),
        "epic": str(epic_out),
        "deconvo_merged": str(merged_out),
        "LR_cal": str(lr_out),
    }
