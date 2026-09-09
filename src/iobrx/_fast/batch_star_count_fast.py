"""batch_star_count_fast: accelerated drop-in for the
``iobrpy.workflow.batch_star_count`` CLI stage (``python -m iobrpy.main
batch_star_count --index ... --path_fq ... --path_out ...``).

Same behaviour as upstream (batch_star_count.py, IOBRpy 0.2.0): discover
``*<suffix1>`` FASTQs under *path_fq* (``os.listdir`` + ``sorted`` +
``random.shuffle``, as upstream), raise ``RLIMIT_NOFILE`` to 65535 before
each sample (upstream ``set_ulimit``), and run STAR two-pass per sample with
the upstream command line token-for-token

    STAR --genomeDir <index> --readFilesIn <f1> <f2>
         --outFileNamePrefix <path_out>/<sample_id>_ --twopassMode Basic
         --runThreadN <N> --outBAMsortingThreadN <N>
         --outTmpDir <path_out>/<sample_id>_STARtmp
         --readFilesCommand zcat --outSAMtype BAM SortedByCoordinate
         --quantMode GeneCounts --limitBAMsortRAM 137438953472

with STAR's stdout/stderr passed through to ``sys.stderr`` (upstream), a
``<sample_id>.task.complete`` marker + an (empty) ``<sample_id>/`` directory
on success, resume-skip when marker + non-empty sorted BAM already exist, and
the final per-sample BAM/GeneCounts location summary.

BUG-COMPATIBLE details preserved from upstream:

* Despite the docstrings ("Number of samples to process concurrently"), the
  upstream batch loop is STRICTLY SEQUENTIAL — ``batch_size`` only slices the
  shuffled file list into chunks that are then processed one-by-one with a
  plain ``for`` loop (no Pool). The port keeps the identical loop; changing
  it would alter STAR's resource footprint and the interleaving of its log
  output.
* ``suffix2`` / ``f2`` use the blunt ``suffix1.replace("1", "2")`` and
  ``f1.replace(suffix1, suffix2)`` (the latter replaces EVERY occurrence of
  suffix1 inside the full path, not just the tail).
* ``sample_id`` strips suffix1 via ``basename(f1).replace(suffix1, "")``
  (again every occurrence).
* A failing STAR run raises ``RuntimeError(f"STAR failed for sample {sid}
  with exit code {rc}")`` which aborts the whole stage (no per-sample error
  capture, unlike the fastq_qc/salmon stages) — propagated verbatim.

The STAR binary is resolved exactly as upstream resolves it — the bare name
``"STAR"`` through the child process PATH (in the official toolchain that is
the conda bash wrapper which dispatches to the best SIMD build, e.g.
``STAR-avx2``); ``star_bin`` only replaces that argv token (default ``None``
= upstream behaviour). ``zcat`` for ``--readFilesCommand`` is likewise
PATH-resolved by STAR itself, as upstream.

Known nondeterminism (identical in original and port): ``random.shuffle``
sample order -> console line order and per-sample completion order vary
run-to-run; on-disk outputs are order-invariant. STAR's own Log.* files
embed timestamps (and the thread count in Log.out) — content-compare those,
byte-compare BAM / ReadsPerGene.out.tab / SJ.out.tab.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from iobrx._run_state import signature, completed, complete

try:
    import resource
except ImportError:
    resource = None

__all__ = ["batch_star_count", "batch_star_count_original", "process_sample"]


def _print_iobrpy_banner():
    """Print the IOBRpy banner (the upstream ``iobrpy.main`` dispatch prints
    it after the stage function returns without raising)."""
    print("   ")
    try:
        from iobrpy.utils.print_colorful_message import print_colorful_message
        print_colorful_message("#########################################################", "blue")
        print_colorful_message(" IOBRpy: Immuno-Oncology Biological Research using Python ", "cyan")
        print_colorful_message(" If you encounter any issues, please report them at ", "cyan")
        print_colorful_message(" https://github.com/IOBR/IOBRpy/issues ", "cyan")
        print_colorful_message("#########################################################", "blue")
        print(" Author: Haonan Huang, Dongqiang Zeng")
        print(" Email: interlaken@smu.edu.cn ")
        print_colorful_message("#########################################################", "blue")
    except Exception:
        # Fallback without colors
        print("#########################################################")
        print(" IOBRpy: Immuno-Oncology Biological Research using Python ")
        print(" If you encounter any issues, please report them at ")
        print(" https://github.com/IOBR/IOBRpy/issues ")
        print("#########################################################")
        print(" Author: Haonan Huang, Dongqiang Zeng")
        print(" Email: interlaken@smu.edu.cn ")
        print("#########################################################")
    print("   ")


def set_ulimit(n=65535):
    """Raise RLIMIT_NOFILE before each STAR run (verbatim upstream)."""
    if resource is None:
        return

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(n, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception as e:
        print(f"Warning: ulimit raise failed: {e} (soft={soft}, hard={hard})")


def process_sample(f1, path_out, index, suffix1, num_threads, star_bin="STAR",
                   verbose=True):
    """Process a single sample using STAR for alignment (verbatim upstream)."""
    # Set ulimit before processing
    set_ulimit()

    # Build corresponding second FASTQ filename by replacing '1' with '2' in the suffix
    suffix2 = suffix1.replace("1", "2")
    f2 = f1.replace(suffix1, suffix2)

    # Extract sample ID assuming no other underscores in the filename
    sample_id = os.path.basename(f1).replace(suffix1, "")

    marker = Path(path_out) / f"{sample_id}.task.complete"
    outputs = [Path(path_out) / f"{sample_id}_Aligned.sortedByCoord.out.bam",
               Path(path_out) / f"{sample_id}_ReadsPerGene.out.tab"]
    run_signature = signature([f1, f2, index],
                              {"threads": num_threads, "suffix1": suffix1,
                               "twopassMode": "Basic", "limitBAMsortRAM": 137438953472},
                              [star_bin])
    if completed(marker, run_signature, outputs):
        if verbose:
            print(f"[Skip] {sample_id} already finished; skipping.")
    else:
        marker.unlink(missing_ok=True)
        if verbose:
            print(f"[Start] {sample_id} is running...")

        command = [
            star_bin,
            "--genomeDir", index,
            "--readFilesIn", f1, f2,
            "--outFileNamePrefix", os.path.join(path_out, sample_id) + "_",
            "--twopassMode", "Basic",
            "--runThreadN", str(num_threads),
            "--outBAMsortingThreadN", str(num_threads),
            "--outTmpDir", os.path.join(path_out, sample_id + "_STARtmp"),
            "--readFilesCommand", "zcat",
            "--outSAMtype", "BAM", "SortedByCoordinate",
            "--quantMode", "GeneCounts",
            "--limitBAMsortRAM", "137438953472",
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"STAR failed for sample {sample_id} with exit code {e.returncode}"
            ) from e

        # Create task.complete file
        os.makedirs(os.path.join(path_out, sample_id), exist_ok=True)
        complete(marker, run_signature, outputs, "")
        if verbose:
            print(f"[Done] {sample_id} finished successfully.")
    return sample_id


def _print_outputs_summary(files_1, path_out, suffix1, verbose=True):
    """Print the final output file locations for each sample (verbatim).

    This does not change computation, only reports paths.
    """
    if not files_1 or not verbose:
        return
    print("\n===== Output files summary =====")
    for f1 in sorted(files_1):
        sample_id = os.path.basename(f1).replace(suffix1, "")
        bam = os.path.join(path_out, f"{sample_id}_Aligned.sortedByCoord.out.bam")
        gene = os.path.join(path_out, f"{sample_id}_ReadsPerGene.out.tab")
        # Print whether file exists
        bam_status = "OK" if os.path.exists(bam) and os.path.getsize(bam) > 0 else "missing"
        gene_status = "OK" if os.path.exists(gene) and os.path.getsize(gene) > 0 else "missing"
        print(f"{sample_id}\n  BAM:        {bam} [{bam_status}]\n  GeneCounts: {gene} [{gene_status}]")


def batch_star_count(index, path_fq, path_out, suffix1="_1.fastq.gz",
                     batch_size=1, num_threads=8, star_bin=None, verbose=True):
    """Process FASTQ files using STAR two-pass alignment (upstream
    ``process_samples`` scheduling, verbatim — including its strictly
    sequential batch loop).

    Parameters
    ----------
    index : str
        STAR genome index directory (upstream ``--index`` -> ``--genomeDir``).
    path_fq : str
        Directory containing the FASTQs; R1 = ``*<suffix1>`` (upstream
        ``--path_fq``; non-recursive ``os.listdir``, sorted then shuffled).
    path_out : str
        Output folder for STAR results (upstream ``--path_out``; created if
        missing).
    suffix1 : str, default '_1.fastq.gz'
        R1 suffix; R2 inferred by the upstream ``suffix1.replace("1", "2")``.
    batch_size : int, default 1
        Upstream ``--batch_size``: chunk size of the SEQUENTIAL loop
        (bug-compatible — samples never actually run concurrently).
    num_threads : int, default 8
        Threads for STAR and BAM sorting (``--runThreadN`` /
        ``--outBAMsortingThreadN``).
    star_bin : str, optional
        Binary path/name override; ``None`` = bare ``"STAR"`` resolved
        through PATH (upstream behaviour).
    verbose : bool, default True
        Reproduce the upstream console protocol. STAR's own output always
        streams to stderr (upstream ``stdout=sys.stderr``).

    Returns
    -------
    dict
        ``{"rc": 0, "samples": [<sample_id>, ... in dispatch order]}``.
        A failing STAR run raises ``RuntimeError`` (upstream behaviour);
        upstream itself returns None.
    """
    star_cmd = star_bin if star_bin is not None else "STAR"

    # Create output directory if it does not exist
    os.makedirs(path_out, exist_ok=True)

    # Get all _1.fastq.gz files and shuffle them
    files_1 = sorted([os.path.join(path_fq, file) for file in os.listdir(path_fq) if file.endswith(suffix1)])
    random.shuffle(files_1)

    total = len(files_1)
    if total == 0:
        if verbose:
            print(f"No files with suffix '{suffix1}' found under: {path_fq}")
        return {"rc": 0, "samples": []}

    if verbose:
        print(
            f"[Plan] Processing {total} samples (Order: RANDOM SHUFFLED).\n "
            f"      Batch size: {batch_size}\n       Threads per job: {num_threads}"
        )

    # Initialize batch index
    samples = []
    batch_index = 0
    while batch_index < len(files_1):
        # Get batch of files
        batch_files = files_1[batch_index:batch_index + batch_size]

        # Process batch of files
        for f1 in batch_files:
            samples.append(
                process_sample(f1, path_out, index, suffix1, num_threads,
                               star_cmd, verbose))

        # Increment batch index
        batch_index += batch_size

    # Final summary of output files
    _print_outputs_summary(files_1, path_out, suffix1, verbose=verbose)
    if verbose:
        _print_iobrpy_banner()
    return {"rc": 0, "samples": samples}


def batch_star_count_original(index, path_fq, path_out, suffix1="_1.fastq.gz",
                              batch_size=1, num_threads=8):
    """Run the untouched upstream CLI ``main()`` (argv rewrite, exactly how
    ``iobrpy.main`` dispatches this subcommand). Escape hatch for
    ``backend='python'``; returns ``{"rc": 0, "samples": None}`` (upstream
    returns nothing; a STAR failure propagates the upstream RuntimeError)."""
    from iobrpy.workflow import batch_star_count as _orig

    argv = ["batch_star_count",
            "--index", str(index),
            "--path_fq", str(path_fq),
            "--path_out", str(path_out),
            "--suffix1", str(suffix1),
            "--batch_size", str(int(batch_size)),
            "--num_threads", str(int(num_threads))]
    saved = sys.argv
    try:
        sys.argv = argv
        _orig.main()
    finally:
        sys.argv = saved
    return {"rc": 0, "samples": None}
