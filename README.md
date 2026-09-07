# iobrx

**iobrx** is a bit-exact, multithreaded acceleration layer for
[IOBRpy](https://pypi.org/project/iobrpy/) — the Python port of the IOBR
tumor-microenvironment workflow suite. It wraps CIBERSORT, signature scoring
(PCA / z-score / ssGSEA / integration), count→TPM conversion, EPIC, quanTIseq,
MCP-counter, ESTIMATE and probe annotation in implementations whose outputs
are **bit-identical** to the originals on the official validation data (all 11
official gates: max abs diff 0.0; the only excepted column is CIBERSORT's
P-value, which the ORIGINAL cannot reproduce run-to-run by design). The heavy
solvers run in a Rust extension (rayon threads, a vendored libsvm byte-identical
to scikit-learn 1.7.2, and the very OpenBLAS entry points the installed
numpy/scipy ship), so a 10-sample CIBERSORT that takes ~46 s single-threaded
in IOBRpy completes in well under a second — with every weight bit equal.
iobrx is a layer, not a fork: it imports the unmodified `iobrpy` package for
resources and fallback code paths, and adds speed without changing numbers.

## Install

iobrx is a hybrid Python + Rust (maturin) package. Source builds need a Rust
toolchain (`cargo` >= 1.7x) and a C++17 compiler; `iobrpy` (>= 0.2.0) is
installed automatically from PyPI as a dependency:

```bash
pip install iobrpy          # resources/fallbacks; pulled in automatically anyway
pip install .               # source build (compiles the Rust extension)
```

Prebuilt wheels are planned for a future release; until then a Rust toolchain
is required. Supported Python: >= 3.11 on Linux x86-64 (the bit-exactness
contract is validated on AVX-512 hardware; see Caveats).

## Quickstart

```python
import iobrx, pandas as pd

# any genes x samples DataFrame (symbols, or see anno_eset to convert IDs)
eset = pd.read_parquet("imvigor210_eset.parquet")          # 872 x 348

# CIBERSORT against LM22 (Rust NuSVR, rayon threads)
cib = iobrx.cibersort(eset, perm=100, QN=True, n_threads=None)
print(cib.shape)          # (348, 25) — 22 cell types + P-value/Corr/RMSE

# per-sample signature scores — 323 signatures in one call
pca = iobrx.calculate_sig_score(eset, "signature_collection", method="pca")
ss  = iobrx.calculate_sig_score(eset, "signature_collection", method="ssgsea")
all3 = iobrx.calculate_sig_score(eset, "signature_collection",
                                 method="integration")

# Ensembl probe ids -> gene symbols (bit-exact, ~3x faster)
# anno = pd.read_parquet("anno_grch38.parquet")
# symbols = iobrx.anno_eset(raw_eset, anno, symbol="symbol", probe="id")

# deconvolution & scoring helpers
epi = iobrx.epic(eset)                    # TRef reference by default
til = iobrx.quantiseq(eset)               # TIL10, lsei
mcp = iobrx.mcpcounter(eset)              # MCP-counter, HUGO symbols
est = iobrx.estimate_score(eset, platform="affy")

# raw counts -> TPM (packaged annotation tables by default)
# tpm = iobrx.count2tpm(counts, idType="Ensembl", org="hsa")

# thread policy: n_threads=None -> min(8, os.cpu_count()); change globally:
iobrx.set_threads(64)
```

Runnable scripts: [`examples/quickstart.py`](examples/quickstart.py) (downloads
the official cohort if no local copy) and
[`examples/full_workflow.py`](examples/full_workflow.py) (the 11-stage official
workflow with a per-stage timing table and optional `--verify-parity`).

## API

| Function | One-liner |
| --- | --- |
| `iobrx.cibersort(eset, perm=100, QN=True, absolute=False, abs_method='sig.score', n_threads=None)` | LM22 NuSVR deconvolution; weights/Corr/RMSE bit-exact, seeded P-value |
| `iobrx.calculate_sig_score(eset, signature, method, mini_gene_count=3, adjust_eset=True, n_threads=None)` | PCA / z-score / ssGSEA / integration signature scoring with shared preprocessing |
| `iobrx.count2tpm(count_mat, anno_grch38=None, ...)` | raw counts → TPM; `anno=None` uses the packaged tables |
| `iobrx.quantiseq(mix, data=None, ...)` / `iobrx.deconvolute_quantiseq(...)` | quanTIseq TIL10 with a memoized HGNC alias map |
| `iobrx.epic(bulk, reference=None, ...)` | EPIC deconvolution; `reference=None` uses the packaged TRef |
| `iobrx.mcpcounter(eset, features_type='HUGO_symbols')` | MCP-counter cell-population scores |
| `iobrx.estimate_score(eset, platform='affy')` | ESTIMATE stromal/immune scores + tumor purity |
| `iobrx.anno_eset(eset, annotation='anno_grch38', symbol='symbol', probe='id', method='mean')` | probe → symbol aggregation with upstream tie semantics |
| `iobrx.set_threads(n)` | default thread count for `n_threads=None` calls |

Full numpydoc docstrings are on every public function (`help(iobrx)`).

## The Rust extension module

The compiled extension installs as the submodule **`iobrx._rust`** (maturin's
hybrid layout: `module-name = "iobrx._rust"` in `pyproject.toml`, matching the
`#[pymodule] fn _rust` in `rust/src/lib.rs`, as pyo3 requires). Importing
`iobrx` also registers the historical alias `sys.modules["iobrx_rust"]`, so
`import iobrx_rust` keeps working once iobrx has been imported. The extension
is an implementation detail: everything it exposes is exercised through the
`iobrx` API above. The crate vendors three third-party C/C++ sources (see
[`rust/vendor/THIRD_PARTY_NOTICES.md`](rust/vendor/THIRD_PARTY_NOTICES.md)):
scikit-learn 1.7.2's `svm.cpp`/`svm.h` (dense libsvm, compiled with sklearn's
baseline x86-64 flags so SMO rounding is identical), the two small sklearn
headers it includes, and Intel's `x86-simd-sort` argsort (the exact AVX-512
kernel numpy dispatches to, so tie-ordering matches `np.argsort` bit-for-bit).

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) (next round).

## Caveats

Read these before comparing iobrx output against IOBRpy output yourself.

1. **CIBERSORT P-value nondeterminism is upstream, not ours.** The ORIGINAL
   `cibersort` seeds its permutations from OS entropy (`SeedSequence()` with
   no seed), so *its own* P-values change run-to-run and cannot be reproduced
   by anyone — including IOBRpy itself. iobrx uses a seeded port of numpy's
   PCG64 (default `seed=0`): the formula `p = count(null_r >= r)/perm` and its
   `1/perm` granularity are identical, and iobrx P-values are stable
   run-to-run and thread-count-invariant. Weights, Correlation and RMSE —
   everything the solver computes — are bit-exact.
2. **NaN input contract.** A mixture containing NaN raises
   `ValueError: Input y contains NaN.` — the same exception the original
   raises from sklearn, on the same inputs (verified in the adversarial
   battery). iobrx does not impute or silently drop NaN rows here; neither
   does IOBRpy.
3. **Degenerate signature overlap.** Signatures whose gene overlap with the
   eset is below `mini_gene_count` are dropped silently (upstream semantics);
   if nothing survives you get the upstream `KeyError`/`ValueError`
   (`No valid signatures found ...` / `No overlapping genes found ...`). EPIC
   raises `ValueError` when fewer signature genes than cell types intersect
   the bulk. Very small overlaps (e.g. imvigor210's 16 LM22 genes) still run,
   but like the original the solution is then weakly determined — compare
   like-for-like, not against intuition.
4. **csv-parse contract note.** The official IOBRpy `cibersort` is file-based:
   it round-trips your DataFrame through `to_csv` and re-parses with pandas'
   python engine, whose float parse is not the identity (1 ulp drift on ~12%
   of cells). iobrx reproduces that exact parse in memory (that is *why* it is
   bit-exact against the official pipeline on DataFrame inputs). Consequence,
   inherited from the contract: if you feed matrices that were already parsed
   through a *different* precision route (e.g. a `%.17g` CSV read beforehand),
   those 1-ulp input differences can flip NuSVR support sets under
   `QN=False`. Under `QN=True` — the default and the only officially
   validated mode — outputs collapse back to bit-identical.
5. **Parity pins: `numpy<2.3` and `scikit-learn<1.8` (validated line:
   numpy 2.2.x, scikit-learn 1.7.x).** iobrx's CIBERSORT solver vendors
   scikit-learn **1.7.2**'s `svm.cpp` byte-identical and ports numpy 2.2.x's
   exact reduction orders. Both upstream stacks drifted after the freeze:
   scikit-learn 1.9.0 changed NuSVR/libsvm numerics, and numpy 2.4 changed
   reduction rounding (a signature-wide `X.mean()` differs by 1 ulp from
   2.2.x — enough to flip NuSVR support sets). In each case iobrx keeps
   producing the frozen, officially validated numbers (verified against the
   frozen references: 240/240 cells) — it is the *original* that moves; the
   pins keep both sides of the parity contract on the same solver. The other
   ten gates pass bit-exact on numpy 2.2–2.4 and scipy 1.16–1.17 (both sides
   resolve the same runtime BLAS/LAPACK), so only these two pins are needed.

## License

MIT — see [LICENSE](LICENSE). Vendored third-party sources keep their own
notices: [`rust/vendor/THIRD_PARTY_NOTICES.md`](rust/vendor/THIRD_PARTY_NOTICES.md).
