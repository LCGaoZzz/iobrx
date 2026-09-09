# iobrx

**Fast, faithful tumor microenvironment analysis with a pandas API.**

[![CI](https://github.com/LCGaoZzz/iobrx/actions/workflows/ci.yml/badge.svg)](https://github.com/LCGaoZzz/iobrx/actions/workflows/ci.yml)
[![version](https://img.shields.io/badge/version-0.3.0-blue)](https://github.com/LCGaoZzz/iobrx/blob/main/CHANGELOG.md)
[![tutorials](https://img.shields.io/badge/executed_notebooks-23-teal)](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/README.md)
[![license](https://img.shields.io/badge/code-MIT-black)](https://github.com/LCGaoZzz/iobrx/blob/main/LICENSE)

English · [中文说明](https://github.com/LCGaoZzz/iobrx/blob/main/README.zh-CN.md) · [Tutorial gallery](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/README.md) · [CPU compatibility](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md) · [Benchmarks](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)

**iobrx is an independently maintained acceleration and integration layer built
on the original [IOBRpy](https://github.com/IOBR/IOBRpy) toolkit.** It depends on
IOBRpy and reuses its reference resources, gene signatures and analysis semantics.
iobrx adds Rust kernels, parallel/vectorized execution, a pandas API, tested
tutorials and Omicos agent interfaces. Inputs and outputs are ordinary
DataFrames and files. Numerical parity is tested per method and fixture;
it is not a universal bit-exact guarantee for every new workflow, parameter or
input. See the explicit exceptions and [validation scope](docs/VALIDATION_0.3.md).

[Original IOBRpy repository](https://github.com/IOBR/IOBRpy) · [Official IOBRpy documentation](https://iobr.github.io/IOBRpy/)

**28 public APIs** — 26 workflow functions (plus the `deconvolute_quantiseq`
alias and the `load_official` data helper) covering the following IOBRpy workflow families:
immune deconvolution (CIBERSORT, BayesPrism, EPIC, quanTIseq, MCP-counter,
ESTIMATE), signature scoring (PCA / z-score / ssGSEA / integration), TPM
conversion and annotation, Immunophenoscore, ligand–receptor pairing, NMF and
TME clustering, the full `tme_profile` chain, RNA-seq file merging, and the
FASTQ→TME orchestration stages (fastp / salmon / STAR / TRUST4 / SpecHLA).

**Omicos:** the harness now exposes 27 typed analysis identifiers (the four signature modes are separate identifiers), including 16 new adapters. HLA and custom-reference BayesPrism remain direct-API capabilities. [Input contracts and limits](agent-harness/omicos/skills/iobrx/references/extended-workflows.md).

**Evidence scope:** R3–R6 timings below are imported campaign measurements. Some raw campaign scripts/logs are not archived here, so these numbers are not independently reproducible from this PR alone. New local validation separates numerical tests, stub-tool contracts and actual tutorial timings.

### What's new in 0.3.0

- **18 additional workflow APIs** (campaign rounds R3–R6): `nmf`,
  `merge_salmon`, `merge_star_count`, `prepare_salmon`, `log2_eset`, `ips`,
  `mouse2human`, `lr_cal`, `tme_cluster`, `bayesprism`, `tme_profile`,
  `fastq_qc`, `batch_salmon`, `batch_star_count`, `trust4`, `runall`,
  `spechla`, `hla_typing`, plus a 30.3× glue optimization inside
  `calculate_sig_score`.
- **4 new Rust kernels**: `lr_gene_valid_mask` (LR_cal gene filter),
  `tme_kmeans` (k-means + KL index), `merge_salmon_parse` (quant.sf reader),
  `bp_gibbs` (BayesPrism Gibbs sampler reproducing numpy's full RNG chain
  bit-for-bit).
- **`tme_profile` end-to-end 10.59×** vs the original CLI on the frozen STAD
  fixture — the endpoint of a measured bottleneck-shift chain
  1.12× → 8.44× → 10.59× ([BENCHMARKS.md Part II](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)).
- **Campaign-reported benchmark (R6)**: 14 core candidates re-measured under one
  unified cold-start protocol against the unmodified original CLI; every
  parity contract reported PASS. The original submission recorded 188 default tests; current checks and limitations are tracked in [validation](docs/VALIDATION_0.3.md).

![Example: CIBERSORT composition and all 22 LM22 populations](https://raw.githubusercontent.com/LCGaoZzz/iobrx/main/tutorials/figures/07_cibersort.png)

## Additional executed tutorials

Eleven new notebooks cover IPS, LR scores, NMF, TME clustering, conditional
log transforms, mouse mapping, Salmon/STAR merging, Salmon preparation,
`tme_profile` and BayesPrism. Each includes actual outputs and an embedded
figure with two reviewed revisions. The table reports one final notebook API
call on the local WSL/Omicos interpreter, two requested threads, excluding
input preparation. These are small tutorial observations, not speedup claims.
NMF's BLAS parallelism is not governed solely by the requested thread count.
The BayesPrism example uses a short demo chain; file-merging examples are synthetic.

| Tutorial | Input shape | API call time |
| --- | --- | --- |
| [13_ips](tutorials/13_ips.ipynb) | 48058 × 4 | 0.016 s |
| [14_lr_cal](tutorials/14_lr_cal.ipynb) | 48058 × 4 | 0.143 s |
| [15_nmf](tutorials/15_nmf.ipynb) | 10 × 22 | 0.465 s |
| [16_tme_cluster](tutorials/16_tme_cluster.ipynb) | 10 × 22 | 0.044 s |
| [17_log2_eset](tutorials/17_log2_eset.ipynb) | 48058 × 4 | 0.235 s |
| [18_mouse2human](tutorials/18_mouse2human.ipynb) | 4 × 3 | 0.020 s |
| [19_merge_salmon](tutorials/19_merge_salmon.ipynb) | 3 × 3 | 0.076 s |
| [20_prepare_salmon](tutorials/20_prepare_salmon.ipynb) | 3 × 4 | 0.014 s |
| [21_merge_star_count](tutorials/21_merge_star_count.ipynb) | 3 × 3 | 0.073 s |
| [22_tme_profile](tutorials/22_tme_profile.ipynb) | 48058 × 2 | 4.828 s |
| [23_bayesprism](tutorials/23_bayesprism.ipynb) | 128 × 3 | 0.351 s |

## Install

**0.3.0 is a development release under review in PR #5.** On 2026-09-09,
GitHub's latest release was v0.1.0 without binary attachments. The repository
contains a wheel/PyPI/container release workflow; that is not evidence that
those distributions have been published. Do not rely on a PyPI or Tsinghua
mirror install until a tested version appears on the public release page.

Validated source-build target: **Python 3.11, Linux x86-64 / WSL2**. Install
Cargo and a C++17 compiler, then use an isolated Python environment:

```bash
git clone https://github.com/LCGaoZzz/iobrx.git
cd iobrx
# While 0.3.0 is under review, select the PR's source:
git fetch origin pull/5/head
git switch --detach FETCH_HEAD
python -m pip install -c tests/constraints-validated.txt ".[tutorials,test]"
python -c "import iobrx; print(iobrx.backend_info())"
python -m jupyterlab tutorials
```

Use the same interpreter as the Jupyter kernel. In an existing Omicos
environment, first check dependency compatibility; use an isolated environment
when the required numerical versions conflict. `pip install` from this source
builds the extension; it is not an installation without compilation.

The release workflow builds and tests a Linux CPython 3.11 wheel, source
archive and versioned container. Publication and mirror synchronization are
separate steps. See [release notes](docs/releases/0.3.0.md) and
[current releases](https://github.com/LCGaoZzz/iobrx/releases).

**Why the dependency pins** (details and measurements in
[BENCHMARKS.md §I.7](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)):

| Pin | Reason |
| --- | --- |
| `iobrpy>=0.2.0` | Provides the bundled reference data (LM22, EPIC TRef, quanTIseq TIL10, TRUST4/SpecHLA assets) and the original-workflow fallback lane. All parity contracts are verified against iobrpy 0.2.0 (PyPI 0.2.1 is numerically identical on every ported path). |
| `numpy>=1.22,<2.3` | numpy 2.4 changed reduction rounding by 1 ulp (a signature-wide `X.mean()`), which flips NuSVR support sets in CIBERSORT (nSV 417 vs 420) — bit-parity would break *from the original's side*. |
| `scikit-learn>=1.2,<1.8` | The Rust core vendors scikit-learn 1.7.2's `svm.cpp` byte-identical. sklearn 1.9.0 moved the ORIGINAL's NuSVR numerics (up to 5.7e-3 on 5/10 official samples), so a ≥1.8 original no longer matches its own frozen references. |

**CPU compatibility and operating-system packaging are separate.** Upstream
IOBRpy's binary distribution limits straightforward installation on other
Python/platform combinations. Windows users can use WSL2; macOS, ARM and
native Windows are not claimed as validated full-stack targets. See the
[compatibility notes](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md) for fallback behavior and limits.

## Quick start

```python
import numpy as np
import pandas as pd
import iobrx

iobrx.set_threads(8)
counts = pd.read_parquet("tutorials/data/eset_stad.parquet")  # genes × samples
tpm = iobrx.count2tpm(counts, check_data=True, remove_version=True)
log_expression = np.log2(tpm + 1)

# Deconvolution and scores (linear TPM for the RNA-seq deconvolution examples)
cib = iobrx.cibersort(tpm, perm=100, QN=False)          # Rust NuSVR core
epic = iobrx.epic(tpm)["cellFractions"]
qnt = iobrx.quantiseq(tpm, tumor=True, rmgenes="default")
scores = iobrx.calculate_sig_score(
    log_expression, "signature_collection", method="integration"
)

# 0.3.0 ports
lr = iobrx.lr_cal(eset="tpm_symbol.csv", output_file="lr.csv",
                  data_type="tpm", id_type="symbol", cancer_type="pancan")
ips = iobrx.ips(eset="tpm_symbol.csv", output_file="ips.csv")
clusters = iobrx.tme_cluster(df=pd.read_csv("tme_transposed.csv"), id="sample")
iobrx.bayesprism(bulk="bulk_counts.csv", out_dir="bp_out", n_threads=8)
#   add backend="rust" for the bit-exact Rust Gibbs kernel

# The whole TME profiling chain in one process (10.59× vs the original CLI)
iobrx.tme_profile(input="TPM.csv", output="tme_out", threads=16)

# FASTQ → TME orchestration (external tools must be on PATH — see below)
iobrx.runall(mode="salmon", outdir="run_out", fastq="raw_fastq_dir",
             threads=16, resume=True)
```

Open [the complete workflow notebook](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/12_complete_workflow.ipynb)
for input checks, interpretation, plots, and per-stage timers on the public
example data (no download needed after installation). The four standalone
signature tutorials use the public IMvigor210 demonstration panel:
872 features × 348 samples.

## The 28 APIs at a glance

| API | One-liner |
| --- | --- |
| `cibersort` | CIBERSORT LM22 immune deconvolution (NuSVR) — Rust core, seeded permutations |
| `calculate_sig_score` | Per-sample signature scores: `pca` / `zscore` / `ssgsea` / `integration` |
| `count2tpm` | Raw count matrix → TPM (vectorized, bit-identical) |
| `quantiseq` | quanTIseq TIL10 deconvolution with a memoized HGNC alias map |
| `deconvolute_quantiseq` | Alias kept for the original function name |
| `epic` | EPIC cell fractions and mRNA proportions |
| `mcpcounter` | MCP-counter population abundance scores |
| `estimate_score` | ESTIMATE stromal / immune scores and tumor purity |
| `anno_eset` | Aggregate probes / Ensembl ids to gene symbols |
| `bayesprism` | BayesPrism deconvolution — python-fast default, opt-in Rust Gibbs, determinism-fixed |
| `tme_profile` | The whole TME profiling chain (sig scores + 6 deconvolutions + LR_cal) in one process |
| `nmf` | NMF clustering with silhouette-based k selection |
| `tme_cluster` | TME k-means clustering with KL-index best-k selection (Rust core) |
| `lr_cal` | Ligand–receptor pair expression matrix, min of log2 TPM (Rust gene-filter core) |
| `ips` | Immunophenoscore (Charoentong 2017 four-block design) |
| `merge_salmon` | Merge Salmon `quant.sf` dirs → TPM / count matrices (Rust parse engine) |
| `merge_star_count` | Merge STAR `ReadsPerGene.out.tab` files → one count matrix |
| `prepare_salmon` | Salmon TPM → deduplicated symbol / ENSG / ENST matrix |
| `log2_eset` | `log2(x+1)` on a genes × samples matrix |
| `mouse2human` | Mouse → human gene-symbol conversion |
| `fastq_qc` | FASTQ QC with fastp + MultiQC (external-tool stage) |
| `batch_salmon` | Batch Salmon quantification over paired-end FASTQs |
| `batch_star_count` | Batch STAR two-pass alignment + GeneCounts |
| `trust4` | TRUST4 TCR/BCR reconstruction + accelerated immune post-processing |
| `spechla` | SpecHLA full-resolution HLA typing for one sample |
| `hla_typing` | Batch HLA typing from a directory of BAM files |
| `runall` | End-to-end FASTQ → TME orchestrator (salmon / star chains) |
| `load_official` | Resolve / download the public IOBR example data |

## Three routes, one choice — how every API is accelerated

Each ported module was evaluated along three routes: keep calling the
**original iobrpy CLI**, write a **pure-Python fast path**, or build a
**Rust kernel**. The final choice is whatever passed the bit-exact contract
at the best measured wall time. Speedups for the 19 ported APIs are the
**R6 formal blind benchmark** (unified cold-start protocol, fresh subprocess
per run, interleaved arms, medians — `research/bench_r6/results.json`,
[BENCHMARKS.md §II.6](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)); orchestration-stage ratios are the
R5 real-data runs ([§II.8](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)); pre-port modules carry their
official-gate numbers ([Part I](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)).

| Module | Original route | Pure-Python route | Rust route | Final choice | Speedup | Parity contract |
| --- | --- | --- | --- | --- | --- | --- |
| cibersort | iobrpy CLI | python fallback | Rust NuSVR core (vendored sklearn-1.7.2 libsvm) | **rust (auto)** | official gates: 34.1× e2e, up to **53.6×** held-out — see BENCHMARKS Part I | bit-exact except the unseeded P-value column |
| calculate_sig_score | iobrpy CLI | python | Rust ssGSEA/PCA cores + glue optimization | **rust (auto)** | glue step **30.3×** (R5); stage gates 6.6–11.5× (Part I) | bit-exact |
| count2tpm / epic / estimate / mcpcounter / quantiseq / anno_eset | iobrpy CLI | python | rust / vectorized | **rust-vectorized (auto)** | see BENCHMARKS Part I (count2tpm up to 48.8×, quantiseq up to 16.9×, mcpcounter 4.2–5.0×, anno_eset 3.5–3.8×, estimate 2.5–3.4×, epic 1.6–1.8×) | bit-exact |
| nmf | iobrpy CLI | **python (chosen)** | no Rust core (both arms share sklearn's NMF) | **python** | **1.21×** (R6) | bit-exact: 3-file sha256 (clusters / top_features / pca_plot.png) |
| merge_salmon | iobrpy CLI | python sequential parse (1.86× R6) | Rust read/parse engine | **rust (auto)** | **2.41×** (R6) | column-aligned token-exact (upstream `as_completed` column order is nondeterministic; port order is deterministic) |
| merge_star_count | iobrpy CLI | **python (chosen)** | none | **python** | **1.75×** (R6) | column-aligned token-exact + stat-row **bug-compat** (upstream's leading-4-stat-rows defect preserved) |
| prepare_salmon | iobrpy CLI | **python (chosen)** | none | **python** | **2.37×** (R6) | bit-exact (sha256) |
| log2_eset | iobrpy CLI | **python (chosen)** | none | **python** | **2.29×** (R6) | bit-exact (sha256) |
| ips | iobrpy CLI | **python (chosen)** | none | **python** | **3.25×** (R6) | bit-exact (sha256) |
| mouse2human | iobrpy CLI | **python (chosen)** | none | **python** | **3.76×** (R6) | bit-exact (sha256) |
| lr_cal | iobrpy CLI | python fallback | Rust gene-filter core | **rust (auto)** | **8.48×** (R6) | bit-exact (sha256) |
| tme_cluster | iobrpy CLI | python RNG / k-means loop | Rust k-means core | **rust (auto)** | **7.17×** (R6) | bit-exact (sha256) |
| bayesprism | iobrpy CLI | **python-fast (auto default)** | Rust Gibbs core (opt-in) | **python auto; `backend="rust"` optional** | python **2.62×** / rust **3.74×** (R6) | bit-exact: hs0 3-file sha256; the Rust kernel reproduces numpy's full RNG chain bit-for-bit |
| tme_profile | iobrpy CLI (9 serial sub-steps) | — | reuse_fast v3 (sig glue + Rust LR_cal) | **reuse_fast (`cibersort_backend="original"`)** | **10.59×** (R6) | 9 outputs: 7 raw-byte identical + 2 identical after stripping the unseeded cibersort P-value column |
| fastq_qc | iobrpy CLI | **python (chosen)** | none (the tool itself does the work) | **python** | ≈1.0× real data (R5; launch layer 14.9× lighter) | bit-exact except fastp-internal HTML jitter |
| batch_salmon | iobrpy CLI | **python (chosen)** | none | **python** | ≈1.0× per-sample real data (R5; launch layer 14.8× lighter) | bit-exact except run-metadata timestamps |
| batch_star_count | iobrpy CLI | **python (chosen)** | none | **python** | ≈1× real data (R5, under the declared 16-vs-32-thread deviation; launch layer 10.6× lighter) | BAM record stream + count tables bit-exact; header @PG/@CO carry the declared thread count |
| trust4 | iobrpy CLI | **python (chosen, accelerated post-processing)** | none | **python** | ≈1.0× real data (R5; stub launch 3.3× lighter) | bit-exact: 12/12 files byte-identical incl. post-processing outputs |
| runall | iobrpy CLI | **python (chosen)** | none | **python** | **1.02×** real data (R5) | bit-exact except documented tool jitter classes |
| spechla | iobrpy CLI | **python (chosen)** | none | **python** | **1.01×** real data (R5) | bit-exact except samtools @PG random-ID jitter |
| hla_typing | iobrpy CLI | **python (chosen)** | none | **python** | **1.05×** real data (R5) | bit-exact except samtools @PG random-ID jitter |

## Why the speedups differ: one floor model

Every number in the table above obeys **speedup ≈ min(1/(1−p), floor)** —
`p` is the share of the original wall time that is a compressible hotspot,
and each module class has its own incompressible floor. This was the
campaign's measured conclusion across rounds R1–R6
([BENCHMARKS.md Part II](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)), not a post-hoc story:

1. **A Rust kernel pays off (≥5×) only when p ≥ 0.6.** `lr_cal`'s per-gene
   pandas filter passes were p=0.81 of the original wall → **8.48×**;
   `tme_cluster`'s pure-Python k-means dominates → **7.17×**. BayesPrism's
   Gibbs sampler is only p≈0.24 → the Rust kernel lands at **3.74×** (1.56×
   over the python lane), below 5× despite being bit-exact by construction.
2. **Bottleneck shift (multi-stage Amdahl).** `tme_profile`:
   **1.12×** (one process, original code — the un-optimized sig-score glue
   was ~84% of the wall) → **8.44×** (the 30.3× glue fix moves the floor to
   cibersort + LR_cal ≈76%) → **10.59×** (LR_cal→Rust, sub-step
   4.10 s → 0.22 s ≈18.6×; the floor is now the contract-bound original
   CIBERSORT solver at 58.6% + the sig Rust chain at 31%). The next floor is
   ~9.3 s (~16.6×) — **≥12× is unreachable without breaking the cibersort
   bit-exact contract.**
3. **Write-out ceiling (~2.4×) for the io-merge modules.** `merge_salmon`'s
   Rust parser is **10.32×** faster in isolation, but the byte-level output
   contract (pandas `to_csv` serialization + gzip) caps end-to-end at ~2.5×:
   measured 2.41–2.48× ≈ 96% of the same-window theoretical ceiling. Rust
   cannot buy its way past the write floor.
4. **Import + pandas-IO floor (~2–3.8×) for the small modules.** Every
   original CLI call pays a ~1.3–1.4 s `iobrpy.main` import on 1.3–1.9 s
   total walls. Removing it exposes pandas `read_csv`/`to_csv` as the new
   floor: pure-IO modules land at 2.29–2.37× (`log2_eset`,
   `prepare_salmon`), modules with real net compute at 3.25–3.76× (`ips`,
   `mouse2human`), and `nmf` converges to 1.21× because both arms run the
   same sklearn solver (shared-kernel floor).
5. **Orchestration ceiling (speedup ≈ 1).** For `fastq_qc` / `batch_salmon`
   / `batch_star_count` / `trust4` / `runall` / `spechla` / `hla_typing` the
   external binaries *are* the wall. Real-data ratios: 0.99–1.25× (the 1.25×
   STAR residual is fully attributable to a declared 32→16 thread deviation,
   visible in STAR's own mapping-speed log). The ported value is API
   consistency, in-process composition (no console-script/PATH requirement),
   resume/parallel scheduling and byte-identical products — plus a Python
   launch layer **3.3–14.9× lighter**, measurable when the tools are
   stubbed.

## Bit-exact contracts and known nondeterminism

**Bit-exact** means: identical index/columns/dtypes/NaN mask with
`max_abs_diff == 0.0` on every numeric cell, or sha256-equal output files,
versus the ORIGINAL iobrpy 0.2.0 executed in the same environment. The
contracts are enforced by `tests/test_parity_*.py` (188 tests pass by
default; the official-data gates run in CI with the frozen fixtures).

Every known nondeterministic item is an **upstream** property, proven by
original-vs-original controls, and handled by an explicit contract clause:

- **CIBERSORT P-value column** — the original seeds its permutations from OS
  entropy (`SeedSequence()` unseeded), so *the original itself* is not
  reproducible run-to-run on that column. iobrx seeds them: stable across
  runs and thread counts, identical formula and `1/perm` granularity. The
  column is excluded from every byte contract that contains it (cibersort
  output; tme_profile's `cibersort_results.csv` / `deconvo_merged.csv`).
- **BayesPrism state order** — the original's cell-state iteration order
  follows the per-process `PYTHONHASHSEED` and can flip discrete Gibbs draws.
  iobrx's default `state_order="sorted"` removes the dependency (theta /
  theta_cv 100% bit-exact vs a `PYTHONHASHSEED=0` original; Z_tumor ULP-only,
  max abs 1.42e-14, zero discrete flips); `state_order="legacy"` with
  `PYTHONHASHSEED=0` reproduces the frozen gold shas exactly (the R6 parity
  configuration).
- **`as_completed` column order** — the original `merge_salmon` /
  `merge_star_count` column order varies run-to-run (three different orders
  observed in three R6 reps). iobrx emits a deterministic sorted order; the
  parity contract aligns by column name and compares token-for-token and
  frame-bit-for-bit.
- **samtools @PG / @RG header IDs** — spechla / hla_typing
  `<sample>.realign.sort.bam` inherit random `samtools merge` header ids
  (e.g. `bwa-7A10F178`); an original-vs-original rerun shows the same
  jitter. Alignment record streams and every other product are byte-compared.
- **fastp HTML jitter** — the duplication rate's 7th significant digit in
  `fastp.html` varies per run *inside the fastp binary* (an original CLI
  rerun reproduced the port's digit and differed from its own frozen
  baseline); cleaned FASTQs are byte-identical.
- **Run metadata** — log timestamps (salmon/STAR/TRUST4 logs, multiqc
  uuid/creation dates) are declared run-metadata; **every data artifact is
  sha256-compared without normalization**.
- **Bug-compat preserved** — upstream defects that are visible in output
  bytes are kept deliberately, e.g. `merge_star_count`'s leading four global
  stat rows that the original never purges, and `lr_cal`'s count-branch
  behavior.

Exact parity is an **observed result on specified inputs and pinned
dependency versions** (numpy<2.3, scikit-learn<1.8 — see Install), not a
cross-platform floating-point guarantee ([details](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md)).

## External tools (bring your own)

iobrx orchestrates the heavy binaries but does not bundle them. Install them
yourself and put them on `PATH` (or pass the per-stage `*_bin` overrides);
iobrx reproduces the upstream command lines token-for-token, so with the
same binary and inputs the products are byte-identical to the original's.

| Stage | External tools |
| --- | --- |
| `fastq_qc` | fastp, MultiQC |
| `batch_salmon` | salmon |
| `batch_star_count` | STAR (+ samtools) |
| `trust4` | TRUST4 (`run-trust4`) |
| `spechla` / `hla_typing` | SpecHLA toolchain: samtools, bwa/bowtie2, bcftools, freebayes, vcflib, blastn, bamUtil (`bam`) |
| `runall` | everything in the chosen salmon/star chain |

All pure-compute APIs (deconvolution, signature scores, TPM, annotation,
IPS, LR_cal, NMF/TME clustering, merges) need **no external tools** — only
iobrpy (references + fallback) and the pinned scientific stack.

## Omicos and agent workflows

The [agent harness](https://github.com/LCGaoZzz/iobrx/tree/main/agent-harness) adds a JSON CLI, an optional
stdio MCP server and a portable Omicos Agent/Skill pair for all 11 analyses.
It validates matrix orientation, declared scale and gene IDs, then records
parameters, input/output hashes, versions, backend and elapsed time in a result
manifest. Analyses use the existing iobrx API and preserve its result layouts.

After installing iobrx, from this checkout:

```bash
python -m pip install ./agent-harness
iobrx-agent doctor
iobrx-agent run --request agent-harness/examples/signature_pca.json
```

See [Omicos setup](https://github.com/LCGaoZzz/iobrx/blob/main/agent-harness/omicos/README.md) for workspace/catalog
installation and MCP configuration, and [harness validation](https://github.com/LCGaoZzz/iobrx/blob/main/agent-harness/VALIDATION.md)
for the actual test record. The companion harness is installed from this
repository; it is not yet a separately published PyPI package.

## How long does each analysis take?

Measured on **Intel Core i9-13900KF, WSL2 Ubuntu, Python 3.11, 8 requested
threads, no AVX-512**, using the existing Omicos environment. Each row runs
in a fresh Python process: first call, followed by three repeat calls.
The main time is the **median of those three repeats**. Data loading, input
preparation and plotting are excluded; the complete workflow includes its
own normalization and analysis stages. OS file caches may already be warm.

<!-- BENCHMARK_TABLE_START -->
| Analysis / notebook | Input: features × samples | Median | First call | Parity vs IOBRpy |
| --- | --- | ---: | ---: | --- |
| [Gene annotation and duplicate resolution](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/01_gene_annotation.ipynb) | 60,483 × 10 | **16.7 ms** | 36.0 ms | bit-identical |
| [Count-to-TPM normalization](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/02_counts_to_tpm.ipynb) | 60,483 × 10 | **67.5 ms** | 111.6 ms | bit-identical |
| [PCA signature scoring](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/03_signature_pca.ipynb) | 872 × 348 | **244.1 ms** | 600.4 ms | bit-identical |
| [Mean-based signature scoring](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/04_signature_zscore.ipynb) | 872 × 348 | **78.6 ms** | 487.8 ms | bit-identical |
| [ssGSEA signature enrichment](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/05_signature_ssgsea.ipynb) | 872 × 348 | **86.7 ms** | 495.9 ms | bit-identical |
| [Integrated signature scoring](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/06_signature_integration.ipynb) | 872 × 348 | **336.0 ms** | 817.9 ms | bit-identical |
| [CIBERSORT immune composition](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/07_cibersort.ipynb) | 48,058 × 10 | **26.37 s** | 26.99 s | bit-identical, P-value excepted |
| [EPIC cell fractions and mRNA proportions](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/08_epic.ipynb) | 48,058 × 10 | **6.8 ms** | 301.9 ms | bit-identical |
| [quanTIseq immune deconvolution](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/09_quantiseq.ipynb) | 48,058 × 10 | **40.9 ms** | 475.2 ms | bit-identical |
| [MCP-counter population scores](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/10_mcpcounter.ipynb) | 48,058 × 10 | **1.1 ms** | 10.0 ms | bit-identical |
| [ESTIMATE stromal and immune scores](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/11_estimate.ipynb) | 48,058 × 10 | **15.8 ms** | 31.0 ms | bit-identical |
| [A complete, inspectable TME workflow](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/12_complete_workflow.ipynb) | 60,483 × 10 | **28.29 s** | 29.59 s | per stage, as rows above |
<!-- BENCHMARK_TABLE_END -->

These are local wall-clock measurements, not a promise for other hardware or
cohorts. CIBERSORT uses **100 permutations and `QN=False`** on the full STAD
TPM matrix. The restricted IMvigor210 signature panel is a different
workload. The complete workflow runs integration scoring on the full STAD
expression matrix, so its total is not the sum of the standalone rows.

The **Parity** column is not a timing: it states what the official gates
([`tests/test_parity_official.py`](https://github.com/LCGaoZzz/iobrx/blob/main/tests/test_parity_official.py)) assert for
that analysis on the same fixtures — equal index, labels and column names, and
exact equality of every numeric cell (`max_abs_diff == 0.0`) against the
ORIGINAL `iobrpy` implementations executed in the same environment
([validation record](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/results/validation.json)). The single
exception is CIBERSORT's P-value: the original seeds its permutations from OS
entropy and is not reproducible run-to-run even by itself, whereas iobrx's
P-values are seeded — stable across runs and thread counts, with the
identical formula and `1/perm` granularity. All 11 gates are re-run by CI on
every push and pull request under the
[validated constraints](https://github.com/LCGaoZzz/iobrx/blob/main/tests/constraints-validated.txt), and the same
contract held on the historical 224-thread Xeon campaign
([BENCHMARKS.md](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)). Exact parity is an observed result on those
environments, not a cross-platform floating-point guarantee
([details](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md)).

[Raw repeats, ranges and environment](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/results/benchmark.json) ·
[Benchmark method and reproduction](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/BENCHMARKS.md) ·
[Historical Xeon performance campaign](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)
[Full performance record: acceleration stack + port campaign](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)

The historical 224-thread Xeon speedups (Part I) and the R6 blind port
benchmark (Part II) are retained as separate experiments; they are not used
to advertise desktop performance.

## Choose the right output

| Analysis | Input used in the tutorials | Output and interpretation |
| --- | --- | --- |
| `anno_eset` | Ensembl expression + annotation | Gene-symbol matrix; duplicate candidates are ranked and one row retained |
| `count2tpm` | Raw nonnegative counts | TPM matrix; filtering/deduplication can leave sums below one million |
| `calculate_sig_score` | Suitable preprocessed expression | `pca`, `zscore`, `ssgsea`, or `integration`; method-specific signature scores |
| `cibersort` | Linear TPM, `QN=False` | Relative LM22 immune fractions + fit statistics |
| `epic` | Linear TPM | Cell fractions, mRNA proportions, fit diagnostics |
| `quantiseq` | Linear TPM, tumor settings explicit | TIL10 fractions and an uncharacterized remainder |
| `mcpcounter` | `log2(TPM + 1)` | Population abundance scores, not percentages |
| `estimate_score` | `log2(TPM + 1)`, `platform="rnaseq"` | Stromal, immune and combined enrichment scores |
| `bayesprism` | Pseudo-bulk counts + scRNA reference | Cell-type/state deconvolution with credible intervals |
| `lr_cal` | TPM (symbol or Ensembl) | Ligand–receptor pair matrix (min of the two log2 TPMs) |
| `ips` | TPM/FPKM expression set | Immunophenoscore 0–1 across four blocks (Charoentong 2017) |

The upstream method named **`zscore`** averages preprocessed signature
expression; it does not guarantee standardized output. `integration`
concatenates three methods rather than averaging them. Only the exact
`platform="affymetrix"` string requests IOBRpy's calibrated ESTIMATE purity
transformation; the historical `"affy"` default does not. These RNA-seq
tutorials therefore show scores without that purity transformation.

## Compatibility and numerical fidelity

```python
iobrx.backend_info()  # native availability, sorting dispatch, bundled BLAS

# Optional: explicitly use the original IOBRpy workflow.
cib = iobrx.cibersort(tpm, perm=100, QN=False, backend="python")
# backend="rust" requires native support and reports an error if unavailable.
```

`backend="auto"` is the default wherever an accelerated lane exists (see the
route table for which lane each module ships). Set `IOBRX_DISABLE_RUST=1`
**before importing** to disable native acceleration process-wide. Missing
native support does not prevent using the public analysis API when IOBRpy
and its dependencies are installed. Missing compatible OpenBLAS triggers
fallback for CIBERSORT/PCA.

All parity gates run in CI on every push and pull request under the
[validated constraints](https://github.com/LCGaoZzz/iobrx/blob/main/tests/constraints-validated.txt); their assertion is
exact equality — labels and every numeric cell, or file sha256 — against the
ORIGINAL executed on the same fixtures, with the documented nondeterminism
clauses above. Exact parity is an **observed result on specified inputs and
dependency versions**, not a cross-platform floating-point guarantee. Native
CIBERSORT uses seeded permutations; the original Python implementation uses
unseeded permutations, so its P-value column is excluded from
exact-equality assertions. See
[precision and fallback details](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md).

## Reproduce, test, contribute

```bash
python -m pytest -q                      # 188 smoke + parity + portability tests
IOBRX_TESTDATA=tutorials/data python -m pytest -q -m full   # official-data gates
python scripts/execute_tutorials.py      # all 12, fresh kernels, embedded plots
python scripts/benchmark_tutorials.py    # first call + 3 repeats per analysis
python scripts/validate_tutorials.py     # outputs, exports and data checksums
```

Tutorials can be read directly on GitHub. To run them interactively:
`python -m jupyterlab tutorials`. Plotting helpers are in
[`tutorials/_common.py`](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/_common.py); every analysis call remains
visible in its notebook. [The figure-review record](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/FIGURE_REVIEW.md)
documents both refinements, with draft and final overview images.

Contributions should preserve the parity gates and add a focused regression
test for changed numerical behavior. Files in `bench/` are frozen historical
artifacts. Report package versions, backend information, input shape and
expression scale when reporting a problem.

## Upstream credit, citation and license

We thank the [IOBRpy developers](https://github.com/IOBR/IOBRpy), the
[IOBR team](https://github.com/IOBR/IOBR), and the original method authors for
the workflows and reference resources on which this package builds.

When publishing results, cite IOBR/IOBRpy and the original methods actually
used — including CIBERSORT (Newman et al.), BayesPrism, quanTIseq, EPIC,
MCP-counter, ESTIMATE, IPS (Charoentong et al. 2017), TRUST4, SpecHLA, and
the tool papers for fastp/salmon/STAR when you run the orchestration stages.
See the [IOBRpy preprint](https://doi.org/10.64898/2026.07.17.739055) and the
[upstream citation guide](https://iobr.github.io/IOBRpy/Citation.html). The
preprint reports CIBERSORT *thread scaling* inside the original
implementation — a different measurement axis from the port speedups here
(see [BENCHMARKS.md §II.9](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)).

iobrx provides acceleration and tutorials; it does not replace those methods
or validate clinical conclusions from an unlabeled example dataset.

iobrx code: [MIT](https://github.com/LCGaoZzz/iobrx/blob/main/LICENSE). Vendored sources retain their
[third-party notices](https://github.com/LCGaoZzz/iobrx/blob/main/rust/vendor/THIRD_PARTY_NOTICES.md). Public example data
retains its [upstream attribution and GPL-3 terms](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/data/README.md).
