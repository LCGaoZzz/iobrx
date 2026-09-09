# Request and result contract v1

`schema` emits the full JSON Schema, including analysis-specific allowed
parameters and defaults. `capabilities --analysis <name>` returns just that
adapter's contract and output meaning (MCP: `iobrx_capabilities(analysis=...)`).
Omit the analysis to discover the full catalog.
Unknown fields and unsupported choices are errors in this JSON adapter. Use
the public Python API for custom calls outside its schema; this catalog is
not a capability limit on the library or the agent.

```json
{
  "schema_version": "1.0",
  "analysis": "cibersort",
  "input": {
    "path": "tpm.parquet",
    "orientation": "genes_by_samples",
    "scale": "tpm",
    "gene_id": "symbol",
    "organism": "hsa"
  },
  "parameters": {"perm": 100, "QN": false, "backend": "auto"},
  "threads": 4,
  "output_dir": "runs/cibersort-001"
}
```

- Files: CSV/TSV with identifiers in the first column; Parquet with a stored
  index (not an unlabeled RangeIndex). No pickle or executable request format.
- Orientation: `genes_by_samples` or `samples_by_genes`; only an explicitly
  requested transpose is performed. Sample and gene IDs must be unique,
  nonempty, and have no surrounding whitespace. Literal text IDs are retained.
- Scale: `counts`, `tpm`, `linear`, `log2p1` (log2(x+1), not natural log), or
  `preprocessed` (signature scoring only; signed continuous expression, as in
  the IMvigor210 panel). Matrices must be finite. The first four scales must
  be nonnegative with positive sample totals; preprocessed samples cannot be
  all zero. The preprocessed declaration requires known normalization provenance;
  it is not a fallback for an unidentified matrix.
  No imputation or unrequested biological normalization is performed by the
  adapter. File-only APIs can require CSV serialization; see extended workflows. Upstream method preprocessing remains controlled by its parameters.
- IDs/species: the capability catalog lists valid combinations. Packaged
  references are human except count-to-TPM, which also supports `mmus`.
- `threads`: positive integer, default min(8, CPU count). It is passed to
  iobrx's threading API, not a hard memory/CPU limit for BLAS or the OS.
- `provenance`: `metadata` by default (paths, size/mtime, input shape, parameters
  and versions). Set `sha256` only when a content audit is useful; it reads
  input/reference contents and tool binaries, checks input stability through
  execution, and records output hashes. It can add substantial I/O on large data.
- Output directories may already exist, including with unrelated files. Existing
  run records, `analysis/` or colliding result exports are protected from overwrite.
  Relative request paths
  resolve beside the request file. With stdin (`--request -`) they resolve
  against the cwd, or `--workspace` when supplied. MCP paths always resolve
  inside its explicitly configured workspace, including existing symlinks.

Each run records normalized `request.json` and an atomically replaced
`results_manifest.json`. On success, the manifest has `status=completed`, input
metadata/shape, installed package versions, backend used,
`analysis_seconds`, `elapsed_seconds`, warnings and artifact metadata.
Analysis time includes API resource/solver initialization; total time also
includes imports, input checks/loading, optional hashing and output serialization, but not
process startup. It is not interchangeable with warm notebook benchmark time.

Outputs preserve the Python API's index/columns/dtypes in Parquet. CSV is a
convenience export and may lose dtype or floating-point round-trip fidelity.
For EPIC, all three result tables are retained separately. Scores are never
silently transposed to look like fraction tables. Non-finite output values
are counted and warned about (e.g. disabled permutation inference or absent
marker sets). Empty or entirely non-finite numerical tables fail.

Call `run` directly for a known request: it performs its own input checks.
`validate` is a separate, optional preflight, and `doctor` is optional environment
diagnosis. There is no required sequence of tool calls.

Exit codes: 0 success/validation/status read; 2 bad CLI/request/input; 3
environment/execution/result or artifact-inspection failure; 4 output collision; 130 caught
interrupt. After a run directory is created, failures are recorded there.
Before that point, an error is returned only on stdout. Logs use stderr.
An uncatchable kill may leave `running` and partial outputs: do not reuse them
as completed results. Make a new directory for a fresh run.

`status` reads recorded execution state and checks whether listed files exist.
It does not hash them or classify a later edit as a failed analysis. Inspect
actual tables when judging whether existing results suit the current task.

For an explicit content audit, use `status <run-directory> --verify-hashes`
(MCP: `iobrx_status(path, verify_hashes=true)`). This also works on older manifests
that already contain hashes. `integrity.mode` distinguishes existence checking
from hashing; `missing`, `mismatches` and `unhashed` report inspection findings.
Missing files, hash mismatches or absent hashes for a requested audit return
exit 3 / MCP `isError=true`, while preserving the historical `status` and adding
`inspection_error`. An existence check says nothing about content equality.
Neither mode modifies the saved manifest, checks process liveness or reruns
analysis. Hashes are optional provenance, not biological validation or signatures
against edits of both data and manifest. Metadata alone does not prove inputs
were unchanged during a run; preserve or snapshot sources when that matters.

For the 16 new adapters, see [extended-workflows.md](extended-workflows.md). Their typed file/directory inputs replace the expression matrix fields when applicable.
