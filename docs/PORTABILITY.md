# CPU portability and precision

## What changed in 0.2.0

The previous native build linked a sorting object compiled unconditionally
for AVX-512. On unsupported CPUs that instruction path could terminate the
process with `SIGILL`; an import-time exception handler cannot recover from
an illegal CPU instruction.

The new build removes that object and all AVX-specific compiler flags. The
vendored libsvm remains compiled with conservative floating-point options.
Quantile normalization calls the installed NumPy's `argsort`, gather, mean,
and inverse-permutation operations through `iobrx._sorting`. NumPy owns CPU
dispatch and the ordering of equal-valued elements. Neither a forced scalar
sort nor a forced AVX-512 sort can match every local NumPy installation.

Consequently, equality is checked against the **reference running in the same
environment**. Different CPU-dispatched NumPy/BLAS paths may differ across
machines, even when both implementations agree on each machine.

## Runtime behavior

| Environment | CIBERSORT | Signature scoring | Other public analyses |
| --- | --- | --- | --- |
| Native extension + compatible Linux wheel OpenBLAS | Native solver; NumPy-dispatched normalization | Native kernels; shared preprocessing | Vectorized Python/NumPy/SciPy |
| Native extension, other BLAS layout | Original IOBRpy fallback | PCA reference fallback; remaining kernels available | Same public API |
| Extension missing or `IOBRX_DISABLE_RUST=1` | Original IOBRpy | Original IOBRpy | Vectorized Python/NumPy/SciPy |
| Explicit `backend="rust"` without extension | Clear error | Clear error | Backend selector not applicable |

For CIBERSORT, an explicit Rust request also requires compatible BLAS.
For signature scoring the selector requires the extension, but its PCA leg
may use sklearn when matching bundled LAPACK symbols are unavailable.
Calculation errors are not swallowed to pretend that a successful analysis
occurred. The original file-based CIBERSORT path uses a private temporary
directory, which is cleaned up on return or failure.

`backend_info()` reports availability, not a promise that every operation
will use native acceleration. `bundled_openblas_found` indicates matching
library files; individual symbol compatibility is checked when needed.

## Tested and untested platforms

The release candidate was built and exercised in an existing Omicos Python
3.11 environment on **WSL2 Ubuntu / Intel Core i9-13900KF**. The CPU exposes
AVX2 and does not expose AVX-512. Standard smoke tests, official-data parity,
fallback tests, sorting tests with advanced NumPy CPU features disabled, and
the full tutorial collection are the validation targets.

The AVX-512 requirement is removed from iobrx's build. This does **not** mean
every transitive dependency is available on every operating system. IOBRpy's
published wheel tags currently constrain easy installation; the validated
route is Linux x86-64 / Python 3.11, including Windows through WSL2. Native
Windows, macOS, ARM, and other Python versions require separate packaging
and numerical validation and are not advertised as supported here.

Runtime fallback is distinct from installation: maturin still builds a native
wheel during `pip install .`, so source installation needs Rust and C++.
The 0.2.0 release supplies a CPython 3.11 manylinux2014 x86-64 wheel for
binary-only pip installation. Release checks install all dependencies without
compilation on Ubuntu 24.04 and test a Debian 12 container; the wheel's glibc
baseline alone does not certify the entire dependency stack on older Linux.
For a pre-existing environment with all dependencies installed, a source-tree
fallback can be used without compiling by placing `src/` on `PYTHONPATH` and
setting `IOBRX_DISABLE_RUST=1`. This does not solve missing upstream packages.

## Numerical contract

- CI uses [the validated versions](../tests/constraints-validated.txt).
  Package metadata pins IOBRpy, NumPy, SciPy, scikit-learn and GSEApy to the
  exact validated versions. Other bounded ranges are not a guarantee of
  equality for every combination. The release workflow also tests ordinary
  pip dependency resolution; the container locks the complete dependency set.
- Official tests compare against IOBRpy executed on the same fixtures, not
  against a plot or a correlation threshold. They assert matching labels and
  exact numerical values.
- CIBERSORT reproduces the reference's CSV round trip, solver settings, and
  reductions. Its non-P-value columns are tested exactly. Native permutations
  use seed 0; the Python reference/fallback retains upstream OS-entropy seeding.
  A fixed permutation formula does not imply identical P-values between
  implementations with different random draws.
- Ties, NaNs and C/F/strided array layouts are covered in normalization tests.
  Sorting tests also run in a fresh process with advanced NumPy CPU features
  disabled. This checks another dispatch path; it is not a substitute for
  executing the full stack on every CPU architecture.
- Tutorial heatmap scaling, grouped bar categories and positive-only density
  displays do not modify analysis outputs. Small reference-gene overlaps can
  produce weakly constrained estimates; use the full transcriptome and
  inspect overlap before drawing biological conclusions.

## Historical performance

[BENCHMARKS.md](../BENCHMARKS.md) and `bench/` record the original server campaign.
They are retained unchanged. The current tutorial benchmark uses a different
CPU, sample matrix and RNA-seq settings; no speedup is inferred by comparing
those unrelated timings.
