# Additional workflows (harness 0.3.0 / iobrx 0.3.0+)

The original 11 requests retain schema version `1.0`. Sixteen new identifiers
use the same direct `run` interface, with optional validation and result inspection.
`capabilities` describes the JSON adapter's
parameter names, defaults and allowed fields; arbitrary CLI tokens and
executable overrides are not accepted.

| Input | Analyses | Contract |
| --- | --- | --- |
| Expression matrix | `ips`, `lr_cal`, `log2_eset`, `mouse2human`, `bayesprism`, `tme_profile` | Original orientation/scale/ID/species fields; mouse mapping requires `mmus`; BayesPrism requires integer counts and uses the bundled reference |
| `feature_table` | `nmf`, `tme_cluster` | `orientation: samples_by_features`, `scale: linear` or `preprocessed`; indexed numeric table; NMF requires nonnegative features |
| `salmon_matrix` | `prepare_salmon` | TSV or gzipped TSV with a `Name` column containing eight GENCODE pipe-delimited fields |
| `salmon_directory` | `merge_salmon` | Nonempty per-sample `quant.sf` files; only these files are copied to the output before merging |
| `star_directory` | `merge_star_count` | `*ReadsPerGene.out.tab` files in the input directory; copied before merging |
| `fastq_directory` | `fastq_qc`, `batch_salmon`, `batch_star_count`, `trust4`, `runall` | Nonempty paired reads; QC also supports `se: true`; installed tools are required |

Each non-matrix input has `kind` and `path`. Batch quantification additionally
requires `input.index`; Salmon accepts `input.gtf`. TRUST4 requires `input.f`
and `input.ref` (reference FASTA files). `runall` requires all three reference
fields plus `parameters.mode: salmon` or `star`. TRUST4 uses `_1/_2.fastq.gz`
pairing. Keep raw reads and indices outside the new output directory.

```json
{
  "schema_version": "1.0",
  "analysis": "batch_salmon",
  "input": {"kind": "fastq_directory", "path": "reads", "index": "references/salmon"},
  "parameters": {"suffix1": "_1.fastq.gz", "batch_size": 1},
  "threads": 4,
  "output_dir": "runs/salmon-001"
}
```

Validation checks pairs, paths (including symlinks), files and tools. It cannot
establish that an index matches the organism, reference version or assay.
Establish that provenance from the task and reference documentation. Default
metadata recording avoids reading entire indices just to hash them; optional
`provenance: sha256` adds full input/reference hashing and its I/O cost.

Native outputs live under `analysis/`; tabular results also receive CSV and
Parquet exports. Manifests include effective parameters, tool paths, input/output
metadata and elapsed time; hashes are opt-in. Nonzero tool returns, missing
outputs and `SystemExit` become failed manifests. Existing directories are
usable when there are no run-file collisions. Use the host's job management;
the harness does not add a scheduler or automatic retry engine. The public
`runall` workflow retains its own checkpoint checks for safe resume; changing
harness provenance does not disable those checks.

`tme_profile` defaults to RNA-seq settings here: `QN=false`, `arrays=false`,
`platform=rnaseq`, original CIBERSORT solver. The direct API and `runall` retain
upstream assay defaults; do not assume they are interchangeable. File-only
APIs require CSV staging for Parquet or transposed inputs, recorded in warnings.
NMF may differ at floating-point roundoff after memory-layout conversion;
the adapter test uses `rtol=1e-12, atol=1e-14` and exact feature rankings.
`backend_used: iobrx-api` deliberately does not claim every kernel used Rust.

HLA/SpecHLA and custom-reference BayesPrism are direct Python API capabilities.
`iobrx.extract_hla_read(sample_id, bam_path, ref, outdir)` performs standalone
extraction without typing and defaults to `auto_install=False`. The existing
`spechla` and `hla_typing` wrappers retain upstream auto-install behavior;
use a prepared isolated environment for those calls.
External-tool contract tests use stubs; they do not validate biological
alignment, reconstruction or HLA typing accuracy.
