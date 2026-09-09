# iobrx agent harness

An Omicos-first interface for **27 typed analysis identifiers**: a JSON CLI, an optional
stdio MCP server, and a portable Omicos Agent/Skill pair. Every analysis calls
the existing iobrx API. The harness adds input checks and auditable result
files, without changing the numerical algorithms. Agents can run known requests
directly, use the public Python API for custom work, and reuse Omicos's tools,
session context and existing results. Detailed method guidance is loaded on demand.

| Interface | Entry point | Purpose |
| --- | --- | --- |
| CLI | `iobrx-agent` / `python -m iobrx_harness` | JSON capability discovery, validation, execution and result checks |
| Omicos Skill | `omicos/skills/iobrx/scripts/run_iobrx.py` | Canonical script plus the same runtime, portable through catalog delivery |
| MCP (optional) | `iobrx-agent-mcp --workspace PATH` | Five workspace-scoped tools over stdio |
| Omicos Agent | `omicos/agents/iobrx_analyst.md` | Routing, scientific input choices and result interpretation |

## Install in your analysis environment

Use the **same Python interpreter as iobrx** (the Omicos WSL environment is a
validated choice). First install iobrx following the [main installation
instructions](https://github.com/LCGaoZzz/iobrx#install), then, from the repository root:

```bash
python -m pip install ./agent-harness
```

The companion package is pure Python and does not compile another Rust
extension. It requires iobrx 0.3.x; it does not remove iobrx's own installation
requirements. It has **not been separately published to PyPI**. This source
install is independent of the wheel/PyPI release work. The main package does
not acquire an MCP dependency.

For the optional MCP interface:

```bash
python -m pip install './agent-harness[mcp]'
iobrx-agent-mcp --workspace "$PWD"
```

MCP uses the maintained v1 SDK line (`mcp>=1.28,<2`), with subprocess isolation
per tool call. Long analyses are synchronous calls; set the host's tool timeout
to cover the requested permutations/sample count, or use the CLI through the
host's shell job runner. No background scheduler or resumable workflow engine
is provided by this harness.

## Run a real example

From the repository root, the bundled IMvigor210 expression panel needs no
download. The example paths are relative to the **request file's directory**,
not the current shell directory:

```bash
iobrx-agent run --request agent-harness/examples/signature_pca.json
```

The request explicitly declares orientation, scale, gene identifiers and
species. See [examples/signature_pca.json](examples/signature_pca.json) and
[the complete request contract](omicos/skills/iobrx/references/request-and-results.md).
Existing output directories are usable; existing run records and colliding
exports are protected. `run` includes input checks. Use `capabilities --analysis <name>` when
discovering parameters, `validate` for a useful preflight, `doctor` for environment
diagnosis, and `status` for saved result inspection. No fixed sequence is required.
Input checks cannot infer whether a biological scale declaration is correct.

Stdout contains one JSON document (except `--help`); Python/native diagnostics
go to stderr. Each successful run writes:

```text
results_manifest.json       # status, timings, environment, backend, metadata, warnings
request.json                # normalized request including effective defaults
result.parquet              # index, labels and dtypes preserved
result.csv                  # convenient human-readable export
```

EPIC instead writes `cellFractions`, `mRNAProportions` and `fit_gof` in both
formats. Default provenance records metadata without hashing input matrices,
reference indices, tools or outputs. `status` checks artifact availability;
it does not treat subsequent editing as execution failure. For content auditing,
set `"provenance": "sha256"` in the request and use
`iobrx-agent status <run-directory> --verify-hashes` (MCP: `verify_hashes=true`).
Inspection findings remain separate from the recorded execution state.
See the [contract](omicos/skills/iobrx/references/request-and-results.md) for
missing files, absent hashes and exit codes. After an uncatchable kill, a `running` manifest may
remain. Never treat that state as successful completion.

## Analysis coverage

The original 11 matrix-analysis identifiers below retain their request contracts.
Harness 0.2.0 adds 16 identifiers, for 27 in total; see the
[extended workflow contracts](omicos/skills/iobrx/references/extended-workflows.md)
for their inputs, parameters and output artifacts.

| Analysis name | Input scale | Gene IDs | Output meaning |
| --- | --- | --- | --- |
| `anno_eset` | counts / TPM / linear / log2(x+1) | Ensembl or platform probes | Symbol-indexed expression, input scale preserved |
| `count2tpm` | counts | Ensembl / Entrez / symbol / mouse MGI | TPM with bundled gene lengths |
| `signature_pca` | TPM / linear / log2(x+1) / preprocessed | human symbols | PCA signature scores |
| `signature_zscore` | TPM / linear / log2(x+1) / preprocessed | human symbols | Original IOBRpy mean-based scores |
| `signature_ssgsea` | TPM / linear / log2(x+1) / preprocessed | human symbols | ssGSEA enrichment scores |
| `signature_integration` | TPM / linear / log2(x+1) / preprocessed | human symbols | PCA + mean-based + ssGSEA columns |
| `cibersort` | TPM / linear | human symbols | LM22 weights and fit/permutation diagnostics |
| `epic` | TPM | human symbols | Cell fractions, mRNA proportions, fit diagnostics |
| `quantiseq` | TPM | human symbols | TIL10 cell fractions |
| `mcpcounter` | linear / log2(x+1) | human symbols / Entrez / affy133P2 probes | Marker abundance scores |
| `estimate_score` | linear / log2(x+1) | human symbols | Stromal/immune/ESTIMATE scores |

`preprocessed` explicitly accepts signed, finite continuous expression for
signature scoring, as in the bundled IMvigor210 panel. It does not assert that
the values are TPM or log2(x+1), or automatically reconstruct their original scale.

Mouse data is supported by `count2tpm` and the explicit `mouse2human` adapter.
The extended adapters include FASTQ QC, Salmon/STAR orchestration, TRUST4,
BayesPrism with its bundled reference, and ligand–receptor analysis. These
adapters do not install external tools; prepare the required environment first.
HLA/SpecHLA and advanced custom-reference analyses, including custom-reference
BayesPrism, remain direct Python API capabilities outside this harness.
The typed adapters are conveniences, not an instruction to avoid those APIs.

Defaults follow iobrx except that **CIBERSORT QN defaults to false** for the
illustrated RNA-seq use case and ESTIMATE defaults to `rnaseq`. Explicitly
choose the settings required by your experiment. The CLI does not silently
transpose, log-transform, map IDs or fill missing values. The selected upstream
algorithm's own preprocessing (e.g. `adjust_eset`) still applies. No AVX-512
requirement is added; `doctor` and each manifest report native availability.

## Connect Omicos

See [Omicos integration](omicos/README.md) for local discovery, the exact
catalog layout and stdio configuration. A local installer copies the Agent
and Skill without editing existing files or deploying anything:

```bash
python agent-harness/install_omicos.py --destination /path/to/workspace --dry-run
python agent-harness/install_omicos.py --destination /path/to/workspace
```

Workspace overlays depend on the Omicos account's extension entitlement.
For the shared cloud catalog, use `--layout catalog` on a clean **omicos-admin
feature checkout**, then its validator, PR review/merge and normal catalog
deployment procedure. Copying files alone does not publish an Omicos skill.

## Tests and development

```bash
python -m pip install './agent-harness[mcp,test]'
python -m pytest agent-harness/tests -q
python -m build agent-harness
```

The tests exercise the original 11 analyses and the additional adapters with public repository fixtures, preserve
numerical results through Parquet, check malformed inputs and collisions,
check metadata-only execution and opt-in audits, relocate the complete Skill,
and perform real MCP stdio calls, including direct execution without preflight.
The separate harness CI builds its wheel/sdist, installs the wheel and runs
these tests alongside iobrx's existing numerical CI. See
[VALIDATION.md](VALIDATION.md) for the observed local validation record.

## Relationship to upstream

Inspired by [IOBRpy's agent-harness](https://github.com/IOBR/IOBRpy/tree/main/agent-harness)
(reviewed at `e3808dbfce7735164a762a6222dc4e03ec81a093`). This is an independent
adapter for iobrx's API and Omicos's current Agent/Skill format, not a copy of
the upstream CLI or a claim of full upstream command coverage. Existing
[notebooks and figures](https://github.com/LCGaoZzz/iobrx/tree/main/tutorials)
remain the detailed teaching and visualization material.

New adapters and their precise scope: [extended workflows](omicos/skills/iobrx/references/extended-workflows.md). HLA/SpecHLA and custom-reference BayesPrism remain outside this harness.
