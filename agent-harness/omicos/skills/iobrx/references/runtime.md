# Runtime and installation

Use Python 3.11 on Linux x86-64 or WSL2, the validated iobrx target. Install
iobrx following https://github.com/LCGaoZzz/iobrx#install, in the user's analysis
environment. The catalog wrapper additionally needs `jsonschema>=4.23,<5`.
The installable companion provides that requirement automatically. Install
packages outside the Skill directory: Omicos can replace materialized files.

The canonical command is the resolved `scripts/run_iobrx.py` with that Python.
Its sibling runtime is self-contained and needs no install of iobrx-harness.
The optional installable CLI uses precisely the same source files. Neither
entrypoint compiles another extension or introduces an AVX-512 requirement.

Use `doctor` when installation diagnosis is useful; it is not a prerequisite
for an analysis. Normal runs record environment versions and backend information
without checking every bundled reference or hashing harness sources.
`doctor` reports interpreter, import path, package versions, harness source
hash, native availability and bundled OpenBLAS detection. Missing iobrx,
IOBRpy, required Python dependencies or reference resources yields nonzero
exit and a JSON failure. It checks imports/resources, not every solver's
numerical correctness. `auto` may use the upstream Python implementation for
signature/CIBERSORT; manifests record the actual selected backend. Explicit
`rust` does not silently fall back.

Use the pinned numerical environment from the repository when numerical
parity matters. The existing official parity gates concern matched versions
and fixtures; an arbitrary package upgrade does not inherit that guarantee.
Upstream binary availability still limits native Windows, macOS, ARM and
other Python versions. Use the released 0.3.0 numerical baseline for this
integration, or the explicitly selected source revision for development.
Do not claim a later source revision is a published wheel.


## Prepare once, not inside every run

Resolve the analysis interpreter through Omicos first (not whichever `pip`
happens to be on PATH). An available Skill is not evidence that its runtime
dependencies are installed. Use the host's package-install tool when available;
the equivalent shell commands for the released baseline are:

```bash
"$ANALYSIS_PYTHON" -m pip install --only-binary=:all: "iobrx==0.3.0" "jsonschema>=4.23,<5"
# Only for h5ad inputs:
"$ANALYSIS_PYTHON" -m pip install "anndata>=0.11"
"$ANALYSIS_PYTHON" -m pip check
```

`ANALYSIS_PYTHON` denotes the resolved existing interpreter. Do not substitute
a host path from another machine. Honor the host's configured, trusted index;
retry a network failure using its approved mirror, not an arbitrary index.
`--only-binary` prevents an accidental Rust source build on unsupported hosts.
A failed installation/check is not permission to change the numerical pins.
Resolve conflicts against the existing environment lock, or use a host-managed
isolated environment. Never use `--no-deps` as a conflict workaround.

For deployment images, provision this baseline during image/environment build
and reuse the existing package/wheel cache across sessions. This avoids paying
for IOBRpy's resource dependency in each fresh analysis kernel. The dependency
is retained here: resources and Python fallbacks still use it. An extra cannot
relax an exact requirement already declared by a base package. Splitting that
dependency needs its own resource-license/fallback and parity validation.
Do not edit the shared environment lock from an analysis Skill.
