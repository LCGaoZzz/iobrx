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
other Python versions. Follow the main repository's current installation
status; adding a harness does not imply PyPI/wheel publication has happened.
