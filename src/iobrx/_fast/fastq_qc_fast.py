"""fastq_qc_fast: accelerated drop-in for the ``iobrpy.workflow.fastq_qc``
CLI stage (``python -m iobrpy.main fastq_qc --path1_fastq ... --path2_fastp ...``).

Same behaviour as upstream (fastq_qc.py, IOBRpy 0.2.0): discover
``*<suffix1>`` FASTQs under *path1_fastq* (``os.listdir`` + ``random.shuffle``,
as upstream), run one ``fastp`` child process per sample through a
``multiprocessing.Pool(processes=batch_size)`` + ``imap_unordered`` (the
EXACT upstream scheduling primitives), write the cleaned reads next to
``<sample>_fastp.html`` / ``<sample>_fastp.json`` reports into *path2_fastp*,
drop a ``<sample>.task.complete`` marker per finished sample (resume: a
sample whose outputs + marker already exist is skipped), then aggregate all
``*_fastp.json`` into a MultiQC report under ``<path2_fastp>/multiqc_report``
and print the IOBRpy banner.

The fastp / MultiQC command lines are built token-for-token like upstream
(``--disable_length_filtering --disable_quality_filtering --n_base_limit 6
--compression 6`` etc.), so with the same fastp binary on PATH the cleaned
FASTQ / JSON / HTML products are byte-identical to the original's. Both
binaries are resolved exactly as upstream resolves them — bare command names
through the child process PATH; the optional ``fastp_bin`` / ``multiqc_bin``
arguments only replace that first argv token (default ``None`` = upstream
behaviour) so deployments can pin a binary without touching PATH.

BUG-COMPATIBLE details preserved from upstream:

* ``length_required`` is accepted, threaded through the task tuples and
  NEVER used — the upstream fastp command disables length filtering and does
  not pass the value; the port keeps the dead parameter (API surface parity).
* ``suffix2`` is derived with the upstream's blunt ``suffix1.replace("1",
  "2")`` (every '1' in the suffix flips, not just the read-number token).
* The sample dispatch order is ``random.shuffle``d (upstream); with
  ``imap_unordered`` the completion order — and therefore the order of the
  ``[Start]/[Done]`` console lines — is nondeterministic in BOTH the
  original and the port. On-disk outputs are order-invariant.
* A failing fastp run is caught per sample (``CalledProcessError`` -> status
  "error", stderr decoded into the message); the stage still exits 0.

Documented API additions (no behavioural change at defaults): ``fastp_bin``
/ ``multiqc_bin`` binary overrides, ``verbose=False`` to silence the
orchestration protocol (error messages are always printed), and a returned
summary dict (upstream returns None).
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
from multiprocessing import Pool

__all__ = ["fastq_qc", "fastq_qc_original", "process_sample"]


def _banner():
    """Print the IOBRpy banner (verbatim upstream behaviour)."""
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


def process_sample(file, path1_fastq, path2_fastp, num_threads, suffix1, se,
                   length_required, fastp_bin="fastp", verbose=True):
    """Process a single FASTQ file using fastp (upstream ``process_sample``).

    Parameters & fastp options are unchanged from the original; ``fastp_bin``
    only replaces the argv[0] command token (default ``"fastp"``, resolved
    through PATH exactly as upstream). Returns a dict with sample id,
    status, and output file paths.
    """
    suffix2 = suffix1.replace("1", "2")
    outputs = []

    if not file.endswith(suffix1):
        return {"sample": file, "status": "ignored", "outputs": []}

    forward_file = os.path.join(path1_fastq, file)
    sample_id = file[:-len(suffix1)]
    output_forward = os.path.join(path2_fastp, file)
    task_file = os.path.join(path2_fastp, f"{sample_id}.task.complete")

    # anticipate paired file path for printing, even if se is True
    reverse_file = forward_file[:-len(suffix1)] + suffix2
    output_reverse = output_forward[:-len(suffix1)] + suffix2

    if os.path.exists(output_forward) and os.path.exists(task_file) and (se or os.path.exists(output_reverse)):
        # Already processed; report outputs for summary
        outputs.append(output_forward)
        if not se:
            outputs.append(output_reverse)
        if verbose:
            print(f"[Skip] {sample_id} already finished; skipping.")
        return {"sample": sample_id, "status": "skipped", "outputs": outputs}

    if verbose:
        print(f"[Start] {sample_id} is running...")

    try:
        if se:
            command = [
                fastp_bin, "-i", forward_file, "-o", output_forward,
                "--thread", str(num_threads),
                "--disable_length_filtering",
                "--disable_quality_filtering",
                "--n_base_limit", "6", "--compression", "6",
                "--html", f"{path2_fastp}/{sample_id}_fastp.html",
                "--json", f"{path2_fastp}/{sample_id}_fastp.json"
            ]
        else:
            command = [
                fastp_bin, "-i", forward_file, "-o", output_forward,
                "-I", reverse_file, "-O", output_reverse,
                "--thread", str(num_threads),
                "--disable_length_filtering",
                "--disable_quality_filtering",
                "--n_base_limit", "6", "--compression", "6",
                "--html", f"{path2_fastp}/{sample_id}_fastp.html",
                "--json", f"{path2_fastp}/{sample_id}_fastp.json"
            ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(task_file, 'w') as f:
            f.write("Processing complete for " + sample_id)
        if verbose:
            print(f"[Done] {sample_id} finished successfully.")

        outputs.append(output_forward)
        if not se:
            outputs.append(output_reverse)
        return {"sample": sample_id, "status": "processed", "outputs": outputs}

    except subprocess.CalledProcessError as e:
        print(f"Error processing {sample_id}: {e.stderr.decode()}")
        return {"sample": sample_id, "status": "error", "outputs": []}


def _worker(args_tuple):
    """Unpack arguments for Pool.imap_unordered -> process_sample."""
    return process_sample(*args_tuple)


def _run_multiqc(path2_fastp: str, multiqc_bin: str = "multiqc", verbose: bool = True):
    """Generate the MultiQC report over all fastp JSON results (upstream).

    Output directory: ``{path2_fastp}/multiqc_report``
    Output file:      ``multiqc_fastp_report.html`` (+ data dir)
    """
    out_dir = os.path.join(path2_fastp, "multiqc_report")
    report_html = os.path.join(out_dir, "multiqc_fastp_report.html")
    if os.path.isfile(report_html) and os.path.getsize(report_html) > 0:
        if verbose:
            print("MultiQC report already exists; skipping MultiQC.")
            print(report_html)
        return report_html

    # collect fastp JSONs; if none, skip quietly
    json_reports = [f for f in os.listdir(path2_fastp) if f.endswith("_fastp.json")]
    if not json_reports:
        if verbose:
            print("No fastp JSON files found; skipping MultiQC.")
        return None

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        multiqc_bin,
        "--module", "fastp",          # restrict to fastp module
        "--force",                    # overwrite existing outputs
        "--outdir", out_dir,
        "--filename", "multiqc_fastp_report",
        path2_fastp                   # scan the fastp output directory
    ]
    try:
        completed = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        report_html = os.path.join(out_dir, "multiqc_fastp_report.html")
        data_dir = os.path.join(out_dir, "multiqc_data")
        if verbose:
            print("\nMultiQC report saved to:")
            print(report_html)
        return report_html
    except FileNotFoundError:
        print("MultiQC not found. Please install it, e.g.: conda install -c bioconda multiqc")
    except subprocess.CalledProcessError as e:
        print(f"MultiQC failed: {e.stderr.decode()}")
    return None


def fastq_qc(path1_fastq, path2_fastp, num_threads=8, suffix1="_1.fastq.gz",
             batch_size=1, se=False, length_required=50,
             fastp_bin=None, multiqc_bin=None, verbose=True):
    """Preprocess FASTQ files using fastp in parallel (upstream
    ``step1_fastq_qc`` scheduling, verbatim).

    Parameters
    ----------
    path1_fastq : str
        Path to raw FASTQ files (non-recursive ``os.listdir``, as upstream).
    path2_fastp : str
        Output directory for cleaned FASTQs, fastp reports, task markers and
        the MultiQC report (created if missing).
    num_threads : int, default 8
        Threads per fastp process (upstream ``--num_threads``).
    suffix1 : str, default '_1.fastq.gz'
        R1 suffix; R2 inferred by the upstream ``suffix1.replace("1", "2")``.
    batch_size : int, default 1
        ``multiprocessing.Pool`` size = concurrent samples (upstream CLI
        default; the upstream PYTHON function default is 5 — this port
        mirrors the CLI).
    se : bool, default False
        Single-end mode (fastp ``-i/-o`` only).
    length_required : int, default 50
        BUG-COMPATIBLE dead parameter: accepted and ignored, exactly as
        upstream (the fastp command disables length filtering).
    fastp_bin, multiqc_bin : str, optional
        Binary path/name overrides; ``None`` = bare ``"fastp"`` /
        ``"multiqc"`` resolved through PATH (upstream behaviour).
    verbose : bool, default True
        Reproduce the upstream console protocol; error messages print
        regardless.

    Returns
    -------
    dict
        ``{"results": [<per-sample dict>, ...], "outputs": [<sorted unique
        output paths>], "multiqc_report": <path or None>}`` — upstream
        returns None.
    """
    fastp_cmd = fastp_bin if fastp_bin is not None else "fastp"
    multiqc_cmd = multiqc_bin if multiqc_bin is not None else "multiqc"

    # Keep your original banner at the beginning (unchanged)
    if verbose:
        print("### FASTQ files quality control using fastp ###")

    os.makedirs(path2_fastp, exist_ok=True)
    fastq_files = [f for f in os.listdir(path1_fastq) if f.endswith(suffix1)]
    random.shuffle(fastq_files)

    tasks = [(file, path1_fastq, path2_fastp, num_threads, suffix1, se,
              length_required, fastp_cmd, verbose)
             for file in fastq_files]

    results = []
    if tasks:
        with Pool(processes=batch_size) as pool:
            for res in pool.imap_unordered(_worker, tasks):
                results.append(res)

    # ------- FINAL: print output file paths first -------
    all_outputs = []
    for r in results:
        all_outputs.extend(r.get("outputs", []))

    # Unique & sorted for readability
    unique_outputs = sorted(set(all_outputs))

    if verbose:
        print("\nSaved/Existing output files:")
        if unique_outputs:
            for p in unique_outputs:
                print(p)
        else:
            print("(No outputs produced or files already present.)")

    report = _run_multiqc(path2_fastp, multiqc_cmd, verbose=verbose)

    # ------- Then print the IOBRpy banner specified by user -------
    if verbose:
        _banner()

    return {"results": results, "outputs": unique_outputs,
            "multiqc_report": report}


def fastq_qc_original(path1_fastq, path2_fastp, num_threads=8,
                      suffix1="_1.fastq.gz", batch_size=1, se=False,
                      length_required=50):
    """Run the untouched upstream CLI ``main()`` (argv rewrite, exactly how
    ``iobrpy.main`` dispatches this subcommand). Escape hatch for
    ``backend='python'``; returns None since upstream returns nothing."""
    from iobrpy.workflow import fastq_qc as _orig

    argv = ["fastq_qc",
            "--path1_fastq", str(path1_fastq),
            "--path2_fastp", str(path2_fastp),
            "--num_threads", str(int(num_threads)),
            "--suffix1", str(suffix1),
            "--batch_size", str(int(batch_size)),
            *(["--se"] if se else []),
            "--length_required", str(int(length_required))]
    saved = sys.argv
    try:
        sys.argv = argv
        _orig.main()
    finally:
        sys.argv = saved
    return None
