# Desktop tutorial benchmark

This is a new measurement of the portable implementation on the existing
WSL2 Omicos environment. It is separate from the historical Xeon campaign in
the repository-root `BENCHMARKS.md`.

## Protocol

1. Use the exact input preparation and analysis calls in
   [`scripts/build_tutorials.py`](../scripts/build_tutorials.py).
2. Start a fresh Python process for each of the 12 analyses.
3. Load/prepare its input outside the measured interval.
4. Record its first analysis call, then three subsequent calls sequentially.
5. Report the median of the three repeats, retaining all observations and
   their min/max in [`results/benchmark.json`](results/benchmark.json).

The timer covers the Python API call and resource initialization that occurs
inside it. It excludes Python process startup, imports, data loading, graphics,
and output serialization. In contrast, each notebook's `*_timing.json` is a
single measured call, and `notebook_execution.json` includes the kernel and
plotting time. These are different quantities and should not be mixed.

No benchmark workers run concurrently. The OS file cache can be warm. Repeated
calls benefit from reference-resource caching and reused thread pools; the
first-call column makes that distinction visible. The benchmark is a practical
desktop example, not a controlled hardware comparison or a statistically
precise performance claim based on only three repeats.

## Workloads

- Annotation and TPM: the 60,483 × 10 public STAD count fixture.
- Four signature methods: the 872 × 348 restricted IMvigor210 panel.
- Deconvolution, MCP-counter and ESTIMATE: the 48,058 × 10 returned STAD TPM
  matrix, with the method-specific transformations visible in the notebooks.
- CIBERSORT: LM22, `perm=100`, `QN=False`, relative mode, 8 requested threads.
- Complete workflow: counts → TPM plus integration signatures, CIBERSORT,
  EPIC, quanTIseq, MCP-counter and ESTIMATE; annotation is a separate branch.
  Its signature scoring uses the full STAD matrix, not the IMvigor210 panel.

Thus the complete-workflow time is **not** the sum of the standalone rows.
None of these times is an acceleration ratio versus an upstream baseline.

## Parity versus IOBRpy

Each benchmark row corresponds to an official parity gate in
[`tests/test_parity_official.py`](../tests/test_parity_official.py). A gate
runs the ORIGINAL `iobrpy` implementation on the same fixtures in the same
environment and asserts equal index, labels and column names plus exact
equality of every numeric cell (`max_abs_diff == 0.0`,
`cells_bit_identical == numeric_cells`). Eleven analyses are bit-identical;
CIBERSORT is bit-identical in every column except the P-value, whose
upstream OS-entropy seeding makes it unreproducible even for the original
itself — iobrx uses a seeded port with the identical formula and `1/perm`
granularity. The complete workflow composes stages that each carry their own
gate.

The gates are part of CI (every push and pull request, under the validated
constraints) and were additionally executed in the release validation
([results/validation.json](results/validation.json)): 34 tests passed on the
desktop environment above — 13 smoke, 10 portability, 11 parity. Exact
equality is observed on the stated environments and dependency versions; it
is not a guarantee across CPU dispatch paths, BLAS builds or platforms —
see [PORTABILITY.md](../docs/PORTABILITY.md).

## Environment

Intel Core i9-13900KF; WSL2 Ubuntu; Python 3.11.15; 32 visible logical CPUs;
8 requested iobrx threads; AVX2 present, AVX-512 absent. Numerical dependencies
are recorded in the benchmark JSON and in
[`tests/constraints-validated.txt`](../tests/constraints-validated.txt).

The package was tested through the existing Omicos Python environment. Added
IOBRpy/build dependencies were isolated in an overlay during this audit;
the numerical stack was the existing environment's NumPy/SciPy/sklearn.
No personal filesystem paths are needed to reproduce the run.

## Reproduce

From an installed checkout with the tutorial dependencies:

```bash
python scripts/benchmark_tutorials.py
python scripts/update_tutorial_index.py
```

The second command updates the English/Chinese README tables and tutorial
gallery from the measured JSON. Keep hardware and data constant before
comparing two software versions. Do not compare this desktop table against
the historical 224-thread, different-input server campaign as a speedup.
