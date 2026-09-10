---
id: iobrx
name: iobrx
description: Run or interpret iobrx bulk expression, tumor microenvironment and prepared FASTQ analyses in Omicos; use the public Python API or portable JSON adapters and reuse existing results.
tier: community
category: general_omics_analysis
summary: iobrx bulk tumor microenvironment analysis with portable execution and traceable results.
execution_mode: packaged_python
runtime_entrypoint: scripts/run_iobrx.py
---

# iobrx analysis

Use the existing iobrx/Omicos Python environment and available results.
The public Python API supports exploratory and custom calls; the portable
CLI/MCP adapters provide 27 typed analyses with consistent files and timings.
Choose either interface, consult its API when needed, and complete authorized
preparation, execution and debugging using the host's existing tools.

Preserve user-fixed samples, methods, seeds, sources and permission boundaries.
Establish expression scale, orientation, species and IDs from data provenance;
a filename or successful schema check cannot establish those semantics.
Record meaningful transformations. Ordinary implementation choices and small
checks within the task do not need another approval.

If `import iobrx` fails in the kernel, install the pinned version (with the
mirror fallback from the Runtime reference) — environment preparation is
ordinary task work. Multi-cohort runs can use `batch` to execute several
requests and receive sample-id-aligned merged tables.

## Portable entrypoint

For the bundled CLI, resolve `scripts/run_iobrx.py` with `skill_resource`,
`include_runtime_path: true`, and keep its sibling runtime directory intact.
Use the prepared interpreter; no repository checkout or guessed cache path is needed.

```text
<python> <resolved scripts/run_iobrx.py> run --request <request.json>
```

Known requests can run directly. Use `capabilities --analysis <name>` for an unfamiliar adapter,
`validate` for a useful preflight, `doctor` for environment diagnosis, and
`status` to inspect saved results. These are independent tools, not mandatory
stages. Default provenance records metadata without full file hashing; explicit
SHA-256 auditing is available when the task calls for it.

Judge completion from actual outputs, warnings and scientific diagnostics.
Reuse tables and figures when suitable; a manifest alone is not evidence of
biological correctness. Use Omicos's job/session tools for long runs and recovery.

## References on demand

- [Request and results](references/request-and-results.md): JSON fields, paths, outputs and optional auditing.
- [Additional workflows](references/extended-workflows.md): feature tables, FASTQ/tools, custom-reference and file-only API details.
- [Runtime](references/runtime.md): installation, environment failures and backend compatibility.
- [Interpretation and figures](references/interpretation.md): score/fraction meaning, uncertainty and existing notebooks.
