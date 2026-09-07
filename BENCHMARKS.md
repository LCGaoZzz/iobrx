# Benchmarks

**Status: placeholder — the full benchmark write-up lands next round.**

This release (0.1.0) packages the finished, frozen fast stack; the curated
benchmark tables, plots and reproduction scripts are the deliverable of the
next round and will live in this file.

## What was measured (summary of the frozen provenance)

The packaged stack is the one measured in the development workspace's
benchmark harness (`iobrx/bench/`, read-only for this repository):

- `FINAL_SUMMARY.json` — final freeze validation:
  - 11/11 official gates pass; 10/11 stages max abs diff **0.0**
    (every numeric cell bit-identical, index and columns equal);
    CIBERSORT is 0.0 on all 240 non-P-value cells (P-value column excepted
    by design, see README Caveats).
  - Held-out BLCA control (60,483 x 5): total wall time 55.99 s → 4.90 s
    (**11.4x** end-to-end); cibersort 45.8 s → 0.85 s (**53.6x**),
    count2tpm 4.07 s → 0.08 s (**48.8x**), quantiseq 0.88 s → 0.05 s
    (**16.5x**), all max abs diff 0.0.
  - cibersort thread scaling (fresh subprocess per point): 95.2 s @ 1 thread
    → 3.3–3.6 s @ 64–224 threads, bit-identical vs the frozen reference at
    every thread count.
  - Warm full 11-stage loop: **3.80 s** median total (224-core node).
- `refs/*.parquet` / `refs/epic.pkl` — the frozen bit-exact reference outputs
  for the 11 official stages.
- `refs/baseline_timing.json` — the original-baseline wall times.

## Reproducing locally

The `full` pytest marker and `examples/full_workflow.py --verify-parity`
re-derive the parity gates against the original iobrpy on the fly
(data via `IOBRX_TESTDATA` or the IOBR `data-v1.0` GitHub release).
Timing-focused reproduction scripts (fresh-subprocess protocol, thread
sweeps) ship with the benchmark round.
