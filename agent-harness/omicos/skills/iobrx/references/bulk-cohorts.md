# Bulk h5ad and multi-cohort recipes

## h5ad input without a handwritten transpose/export pipeline

The matrix adapters accept `.h5ad` with the optional `anndata>=0.11` dependency
(or the installable companion's `[h5ad]` extra). Each observation must already
be a bulk or deliberately prepared pseudobulk sample. This reader does not
aggregate single cells, select Primary tumors, normalize counts, infer scale,
convert natural-log data to log2, strip TCGA barcodes or deduplicate genes.
Preserve the user's cohort and preprocessing contract.

```json
{
  "schema_version": "1.0",
  "analysis": "count2tpm",
  "input": {
    "path": "cohort-A.h5ad",
    "orientation": "samples_by_genes",
    "scale": "counts",
    "gene_id": "ensembl",
    "organism": "hsa",
    "h5ad": {"matrix": "layers/counts"}
  },
  "parameters": {"check_data": true, "remove_version": true},
  "threads": 8,
  "output_dir": "runs/cohort-A/count2tpm"
}
```

Choose `input.h5ad.matrix` explicitly: `X`, `layers/<name>` or `raw.X`.
`raw.X` uses `raw.var`, which can contain a different gene set from `var`.
The default labels are `obs_names` and the selected `var_names`; optional
`sample_column` and `gene_column` select metadata columns without changing
labels. Duplicates, missing labels and surrounding whitespace are rejected by
the same checks as table inputs. A column called `case_id` is not necessarily
a unique sample key: retain a separate case-to-sample mapping when needed.

Only the selected expression, obs and corresponding var are read. Dense,
CSR and CSC expression are supported. Other layers, embeddings and uns are
not loaded. The reader opens the source read-only and rejects linked/virtual
external storage. It checks the size of the final float64 matrix before
reading expression: the default `max_dense_bytes` is 536870912 (512 MiB).
This is **not a peak-RAM limit**; metadata, sparse buffers, conversions and
solver workspaces need additional memory. Raise the budget explicitly when
justified; never silently drop samples/genes to make a request fit.

Count-to-TPM remains an explicit analysis. Reuse its saved gene-by-sample
Parquet with `orientation=genes_by_samples`, the returned IDs, and `scale=tpm`
for EPIC/CIBERSORT. Prepare log2(TPM+1) explicitly when that is the selected
ESTIMATE protocol; a `log1p` layer name does not establish a log2 scale.

## Many cohorts, existing run contracts

Use a manifest of **literal cohort names and request paths**, not inferred
folder prefixes. Preserve all user-fixed parameters, sources, seeds and sample
filters in the normal per-run requests. A plain loop reuses the existing
`run` interface and in-process resource caches; Omicos remains responsible
for jobs, sessions and recovery. There is no additional workflow engine.

The following recipe expects already prepared requests and a frozen sample
ID list per cohort. `request_files` deliberately lists paths explicitly.
Use one interpreter so library imports and caches can be reused. The separate
runs still retain their own request, diagnostics and result manifest.

```python
from pathlib import Path
import pandas as pd
from iobrx_harness.runtime import read_json, run
from iobrx_harness.cohort_tables import align_samples, combine_cohorts

request_files = {
    "cohort-A": [Path("requests/A-epic.json"), Path("requests/A-cibersort.json"),
                 Path("requests/A-estimate.json")],
    "cohort-B": [Path("requests/B-epic.json"), Path("requests/B-cibersort.json"),
                 Path("requests/B-estimate.json")],
}
# Read these from the frozen cohort metadata in a real project.
sample_ids = {"cohort-A": ["001", "NA"], "cohort-B": ["S1", "S2"]}
exports = {"epic": "cellFractions", "cibersort": "result", "estimate_score": "result"}
by_method = {method: {} for method in exports}
for cohort, paths in request_files.items():
    for request_path in paths:
        request_path = request_path.resolve()
        request = read_json(request_path)
        method = request["analysis"]
        table_name = exports[method]
        if cohort in by_method[method]:
            raise ValueError(f"Duplicate cohort/method request: {cohort}/{method}")
        manifest = run(request, request_path.parent)
        if manifest["status"] != "completed":
            raise RuntimeError(manifest.get("error", manifest["status"]))
        artifact = next(a for a in manifest["artifacts"]
                        if a["format"] == "parquet" and a["table"] == table_name)
        table = pd.read_parquet(Path(manifest["manifest_path"]).parent / artifact["path"])
        if method == "estimate_score":
            table = table.T  # This API returns results x samples, not fractions.
        by_method[method][cohort] = align_samples(table, sample_ids[cohort])

# Keep method meanings and cohort IDs separate; no positional/inner joins.
combined = {method: combine_cohorts(tables) for method, tables in by_method.items()}
```

`align_samples` rejects missing, extra and repeated samples; it does not take
an intersection or fill missing results. `combine_cohorts` requires matching
result columns and retains a `(cohort, sample_id)` index, so IDs shared across
cohorts cannot collide. Keep EPIC's mRNA proportions and fit diagnostics in
their original artifacts; the illustrative export above selects cell fractions.
Report any explicit exclusions and the case/sample mapping separately.

Do not concatenate cohort inputs merely to avoid initialization. Quantile
normalization and cohort-dependent preprocessing/statistics may change with
batch composition. Combining results after independent calls is different
from combining inputs before a call. Benchmark cold and warm timings on the
same inputs/parameters; do not promise a fixed speedup or assert all methods
use Rust. Inspect each run's `backend_used` and warnings where applicable.

For 1000+ samples, choose an explicit `threads` budget from the CPU allocation
actually granted to the job. A host's physical-core count may exceed its
container/cgroup/scheduler entitlement. Small 4/8/16-thread pilots can reveal
where scaling saturates; retain the same scientific parameters. Avoid nesting
many cohort processes each with that full budget, or multiplying Rayon and
BLAS thread pools. The library's conservative default remains at most 8;
affinity-aware detection does not measure cgroup CPU-time quotas.
