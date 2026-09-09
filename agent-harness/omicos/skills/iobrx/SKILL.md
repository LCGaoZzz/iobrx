---
id: iobrx
name: iobrx
description: Run iobrx expression scoring, deconvolution, IPS, ligand-receptor scores, NMF/TME clustering, bundled-reference BayesPrism, matrix utilities, FASTQ QC, Salmon/STAR, TRUST4 or runall through typed requests and JSON result manifests; inspect existing results without rerunning.
tier: community
category: general_omics_analysis
summary: iobrx bulk tumor microenvironment analysis with portable execution and traceable results.
execution_mode: packaged_python
runtime_entrypoint: scripts/run_iobrx.py
---

# iobrx analysis

Use this Skill for the named bulk-expression analyses or an explicit iobrx
request. It uses the public iobrx API, its acceleration/fallback behavior and
the reference resources installed with IOBRpy. The catalog exposes 27 analysis
identifiers, including 16 additional adapters in harness 0.2.0. Read
`references/extended-workflows.md` for their input types and limitations.
HLA/SpecHLA and custom BayesPrism references remain Python API capabilities;
this harness does not expose them or differential-expression testing.

## Resolve and prepare

Resolve `scripts/run_iobrx.py` using `skill_resource` with
`include_runtime_path: true`. Invoke that canonical script with the Python
interpreter in the user's prepared iobrx/Omicos environment. Its `SKILL_ID` is
`iobrx`; its sibling `scripts/iobrx_harness` contains the same runtime as the
installable CLI. Keep the delivered directory intact. No guessed cache path,
repository checkout, shell working directory or per-skill environment variable
is required. Install dependencies outside the materialized Skill directory.

```text
<python> <resolved scripts/run_iobrx.py> doctor
<python> <resolved scripts/run_iobrx.py> capabilities
```

`doctor` checks imports and bundled reference files; it is not a numerical
parity test. A missing native extension can still allow the documented Python
fallback. An explicit `backend=rust` request must fail if native execution is
unavailable. See `references/runtime.md` for installation and compatibility.

## Execute the user's analysis

Read `references/request-and-results.md` when creating the request. Determine
the file, matrix orientation, scale, gene-ID type and species from the user's
data/provenance. Resolve missing material information with the user; do not
ask again for choices already supplied. Do not infer TPM from a filename or
automatically transform a matrix to make a solver accept it.

Use `capabilities` for supported parameters. Write the request JSON in the
user's workspace, with a new output directory and the needed thread/backend
settings. Relative paths resolve beside the request file. Then:

```text
<python> <resolved scripts/run_iobrx.py> validate --request <request.json>
<python> <resolved scripts/run_iobrx.py> run --request <request.json>
<python> <resolved scripts/run_iobrx.py> status <run-directory>
```

Validation reads the declared files without analysis or output writes. It checks
supported scale/ID/species declarations, uniqueness and finite numeric values;
it cannot establish that the declared biological scale is true. Inspect the
result and error before making any corrective change. A failed analysis is
not permission to silently switch methods, change normalization or replace
existing outputs. Once the request is valid and within the user's authorized
task, proceed without an extra approval ceremony.

## Read and explain results

Require process exit code zero and manifest `status=completed` before reporting
success. Inspect warnings and artifact integrity. `running` is a recorded
state, not proof that a live worker exists. The manifest contains effective
parameters, input SHA-256, versions, requested threads, selected backend,
analysis time, total time, artifact shapes/dtypes and hashes. `status` checks
output hashes without rerunning analysis; it does not revalidate the source
matrix or recreate a killed worker.

Read `references/interpretation.md` for method-specific meaning. Report
outputs and timings with input size and settings. Use the Parquet artifacts
for subsequent numerical work and the CSV files for inspection. Describe
MCP-counter/ESTIMATE/signature outputs as scores, not fractions. CIBERSORT's
stochastic P-values are excluded from the package's bit-exact parity claim.

For figures, start from the repository's existing analysis notebooks rather
than inventing another numerical pipeline. Their URLs and figure conventions
are in `references/interpretation.md`.
