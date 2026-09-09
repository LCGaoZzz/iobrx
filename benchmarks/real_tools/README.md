# Real FASTQ, BAM and HLA comparison

This recipe compares **iobrx 0.3.0 with unmodified PyPI IOBRpy 0.2.1**, using
actual external tools and public reads. It preserves failures and variability;
it does not imply every workflow is faster or every output is byte-identical.

## Results and protocol

<!-- REAL_RESULTS_START -->
| Workflow | Threads | IOBRpy median | iobrx median | Original / iobrx | Declared products |
| --- | ---: | ---: | ---: | ---: | --- |
| `fastq_qc` | 4 | 3.781 s | 2.456 s | 1.54× | 3/3 equal |
| `batch_salmon` | 4 | 24.086 s | 24.426 s | 0.99× | 3/3 differ; original repeats also differ |
| `batch_star_count` | 4 | 8.692 s | 7.876 s | 1.10× | 3/3 equal |
| `extract_hla_read` | defaults | 3.804 s | 4.822 s | 0.79× | 3/3 equal |
| `spechla` | 4 | 93.335 s | 86.324 s | 1.08× | 3/3 equal |
| `batch_salmon` | 1 | 69.094 s | 65.474 s | 1.06× | 3/3 differ; original repeats also differ |

Measured 2026-09-09 on Intel Core i9-13900KF, WSL2 Ubuntu, Python 3.11.16, no AVX-512. **36 successful timed invocations**; the failed micromamba setup attempt is separately retained. Ratios below 1 mean iobrx was slower.

For 4-thread Salmon, maximum across-arm TPM difference was **1,962.060302**, versus **1,941.317656** across original-only repetitions (lowest cross-arm Pearson r **0.998672**). Single-thread maxima were **951.350945** across arms and **1,068.496998** within the original (lowest cross-arm r **0.999149**). Single threading did not produce byte equality. High correlation is not a substitute for equality, and these observations do not isolate the cause of nondeterminism. Both protocols retain their mismatches.

[Machine-readable results](results/2026-09-09/summary.json) · [4-thread numerical comparisons](results/2026-09-09/runs-001/batch_salmon/numeric_variation.json) · [1-thread numerical comparisons](results/2026-09-09/runs-single-thread/batch_salmon/numeric_variation.json)
<!-- REAL_RESULTS_END -->

Primary cases have **three repetitions per arm**, alternating IOBRpy→iobrx,
iobrx→IOBRpy, IOBRpy→iobrx. Both request four threads where the API exposes
threads (extraction retains upstream tool defaults), batch size one, and use
identical tools, inputs, references and active output paths. Every invocation
starts a fresh process and output directory. IOBRpy runs its official CLI;
iobrx runs its public Python API. Times include process/import overhead,
iobrx's real provenance/output checks and external computation. They are
**not Rust-kernel microbenchmarks**.

Downloads, installation, compilation, reference indexing, comparison and plots
are outside the timers. OS caches are not flushed; wrappers may shuffle sample
order. No other analysis or benchmark runs are launched concurrently. GNU time
records each process tree's maximum RSS, not the sum of children's memory.
Three runs describe this workstation, not a confidence interval or a promise
for all hardware/data sizes. All individual times are retained.

## Inputs and limits

| Case | Real reads and reference | Comparison |
| --- | --- | --- |
| fastp + MultiQC | GEUVADIS ERR188044 / ERR188104; first 100,000 pairs each | Cleaned FASTQ records and fastp measurements; HTML reports exist |
| Salmon | Same reads; **complete GENCODE v44 transcript FASTA**, k=31 | Quantification plus within/between-arm numerical variability |
| STAR | Yeast DRR392086 / DRR392094; first 30,000 pairs each; complete R64 GCF_000146045.2 genome | Actual BAM records and GeneCounts |
| HLA extraction | SpecHLA NA06985 exon example, 52,771 pairs; BWA alignment to GRCh38 chromosome 6 | Paired and singleton FASTQ products from a real sorted/indexed BAM |
| SpecHLA | Same NA06985 enriched FASTQs; exon mode; upstream package's IPD-IMGT/HLA 3.38.0 assets | Eight-locus/16-allele report and 16 allele FASTAs |

NA06985 is **already HLA-enriched**, and the BAM uses primary chr6 only.
Missing ALT-region warnings remain in the logs. This tests wrapper agreement,
**not independent genotype accuracy**. Yeast is a complete small-genome
alignment fixture, not a human TME demonstration. Prefix samples are bounded
examples, not representative large cohorts. The Salmon transcriptome index
has no genome decoys and defines this experiment, not a universal production
reference recommendation. TRUST4, CRAM, all HLA reference layouts and full-size
human cohorts remain outside this new evidence.

NA06985 is pinned to [SpecHLA commit c965423](https://github.com/deepomicslab/SpecHLA/tree/c965423bb263eb4ead599d7cca8d103e616cf1c5/example/exon).
All source URLs, accessions and digests are in
`results/2026-09-09/preparation/sources.json`. Gzip headers may differ after
prefix extraction; decompressed digests identify the actual records. These
hashes document this experiment and add no normal iobrx/harness requirement.

## Reproduce on Linux x86-64 / WSL2

Use a fresh work directory, micromamba, a system C/C++ compiler, Make, Perl,
gzip and `/usr/bin/time`. Allow approximately 15 GB for references, tools and
repetitions. The shared Omicos environment need not be modified.

```bash
# From the iobrx checkout.
export IOBRX_REAL_WORK="$PWD/real-tools-work"
micromamba create -y -p "$IOBRX_REAL_WORK/toolchain" \
  -f benchmarks/real_tools/conda-linux-64.lock
micromamba create -y -p "$IOBRX_REAL_WORK/conda-cli" \
  -f benchmarks/real_tools/conda-cli-linux-64.lock
export PATH="$IOBRX_REAL_WORK/toolchain/bin:$IOBRX_REAL_WORK/bin:$PATH"
export CONDA_PREFIX="$IOBRX_REAL_WORK/toolchain"
export CONDA_EXE="$IOBRX_REAL_WORK/conda-cli/bin/conda"
python -m pip install --only-binary=:all: \
  -c benchmarks/real_tools/python-constraints.txt \
  iobrx==0.3.0 multiqc nbformat nbclient ipykernel
python benchmarks/real_tools/install_static_tools.py --workdir "$IOBRX_REAL_WORK"
ln -s "$CONDA_EXE" "$IOBRX_REAL_WORK/bin/conda"
python benchmarks/real_tools/prepare.py --workdir "$IOBRX_REAL_WORK" --threads 4
python benchmarks/real_tools/prepare_hla.py --workdir "$IOBRX_REAL_WORK" --threads 4 \
  > "$IOBRX_REAL_WORK/hla-preflight.log" 2>&1
python benchmarks/real_tools/run.py --config "$IOBRX_REAL_WORK/cases.json" \
  --output "$IOBRX_REAL_WORK/repetitions" --repeats 3
python benchmarks/real_tools/salmon_variation.py \
  --case-dir "$IOBRX_REAL_WORK/repetitions/batch_salmon"
```

Before PyPI publication, substitute the downloaded release wheel's local
filename for `iobrx==0.3.0`. Static fastp's URL is mutable: the installer reports
a changed build rather than silently calling it the measured binary. STAR
2.7.11b and Salmon 1.10.0 use official versioned releases; remaining tools use
the explicit environment locks. SpecHLA setup may install dependencies and
compile its bundled components. Complete that untimed preflight first.
The ordinary core-analysis release container does **not** include this external
alignment/HLA toolchain.

**Conda CLI matters:** the original extraction helper assumes `conda list
--json` returns a list; current micromamba's different structure caused an
`AttributeError` in our first attempt. A real, separately installed Conda CLI
then inspected the same tool environment successfully. No upstream calculation
or dependency-check code was patched. The failed attempt is preserved separately
and excluded from successful-run medians.

## Equality and failure semantics

- FASTQ: decompressed record bytes; fastp JSON excludes only the command string.
- STAR: raw counts and sorted complete SAM records; headers and arbitrary
  alignment-record order are excluded.
- Salmon: raw `quant.sf` bytes plus per-column numerical differences. No
  post-hoc tolerance changes a mismatch into a pass.
- HLA: final allele report and sixteen FASTAs. Intermediate phasing products
  and random SAM header IDs are outside this specific check.

Missing/empty products, header-only typing reports and all-N allele sequences
fail inspection. The runner preserves existing outputs and refuses to overwrite
a prior case. HTML layout, timestamps and UUIDs are not equality targets.
It exits **1** for any scientific mismatch, including the observed four-thread
Salmon variability, while preserving completed results. Inspect `summary.json`
and `numeric_variation.json`; exit status alone is not a scientific conclusion.
A failed tool invocation stops the run and preserves its logs.

For supplemental single-thread Salmon, copy `cases.json` and set top-level
`threads`, Salmon's `num_threads` and its CLI thread argument to 1. Run
`--case batch_salmon` in a new directory. Keep references, tools and methods
unchanged. The committed result labels this separate protocol explicitly.

## Executed notebooks

Tutorials **24–28** each run one of these APIs again on the prepared fixtures.
Their demo times are separate from benchmark medians. Figures show all timing
points, medians, units and actual scientific products. Two revisions are
retained, with final PNG/PDF/SVG exports and review notes. The notebooks use
`IOBRX_REAL_WORK` and the same prepared environment.

These scripts, explicit environments and compact logs make the **new**
measurements rerunnable. They do not supply the missing original R3–R6 logs.


Release-branch local checks: **247 default tests passed**, 15 full tests left for release CI; 28 executed notebooks / 84 figure exports / four data digests validated; source/installed release checks and `pip check` passed. [Log](results/2026-09-09/verification.log).
