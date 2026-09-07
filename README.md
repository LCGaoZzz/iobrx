# iobrx

[![PyPI](https://img.shields.io/badge/PyPI-0.1.0-blue)](https://pypi.org/project/iobrx/)
[![build](https://img.shields.io/badge/build-maturin-orange)](./rust/Cargo.toml)
[![tests](https://img.shields.io/badge/tests-13%2F13%20smoke%20%2B%2011%2F11%20full-brightgreen)](./tests)
[![license](https://img.shields.io/badge/license-MIT-black)](./LICENSE)

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
numpy/scipy ship), so a 10-sample CIBERSORT that takes ~95 s single-threaded
in IOBRpy completes in ~3 s — with every weight bit equal.
iobrx is a layer, not a fork: it imports the unmodified `iobrpy` package for
resources and fallback code paths, and adds speed without changing numbers.

The full measurement record — every acceleration route the campaign explored,
per-stage tables, thread scaling, the 83-case adversarial precision audit and
the verdict — is in [BENCHMARKS.md](BENCHMARKS.md), with every number traced
to a frozen artifact in [`bench/`](bench/). The short version:

| Route (measured) | Official 11-stage workflow | Precision (BENCHMARKS.md §2, §4) |
| --- | --- | --- |
| R0 — ORIGINAL IOBRpy 0.2.0 | 97.13 s = 1.0x | reference |
| R1 — joblib threads only | 6.42x best | bit-identical weights |
| R2 — vectorized stages only | count2tpm 38.7x, quantiseq 16.9x (stage) | bit-exact |
| R3 — Rust bit-faithful CIBERSORT | 34.3x stage @224T | bit-identical incl. csv-parse |
| **R5 — shipped: Rust + vectorized full stack** | **3.75–4.13 s fresh = 23.5–25.9x**; held-out BLCA 11.43x | 10/11 gates max abs diff 0.0; CIBERSORT 240/240 non-P cells |

## Install

iobrx is a hybrid Python + Rust (maturin) package; there are no prebuilt
wheels yet, so a source build needs a Rust toolchain (`cargo` ≥ 1.7x) and a
C++17 compiler:

```bash
pip install .          # compiles the Rust extension (see BENCHMARKS.md §6 for build times)
```

- **Python ≥ 3.11** on Linux x86-64 (iobrpy itself ships cp311 wheels; the
  bit-exactness contract is validated on AVX-512 hardware, see Caveats).
- `iobrpy >= 0.2.0` is installed automatically from PyPI and used for
  resources and fallback paths.
- **Version pins: `numpy<2.3` and `scikit-learn<1.8`** — these keep both
  sides of the parity contract on the same numerics, because upstream
  numeric drift (numpy 2.4 reduction rounding; scikit-learn 1.9 NuSVR
  changes) moves the *ORIGINAL's* outputs off the frozen bit-exact line
  (details in BENCHMARKS.md §7 "Parity pins").

## Quickstart

Block 1 is **self-sufficient**: seeded synthetic matrices (~200 genes × 20
samples each, numpy-generated) built on the reference data packaged with
iobrpy — no download of any kind. Verified verbatim in a fresh venv with
`IOBRX_TESTDATA` unset and the download mirrors unreachable: rc=0, ~3 s wall.
Block 2 then runs the **official IMvigor210 cohort** through the documented
data resolution (`$IOBRX_TESTDATA` → local cache → mirror download) and skips
with a clear message when neither is available:

```python
import pickle

import numpy as np
import pandas as pd
import iobrx
from importlib.resources import files

rng = np.random.default_rng(20260907)          # everything below is seeded


def synth(index, n=20):
    """A small synthetic cohort: len(index) genes x n samples."""
    return pd.DataFrame(rng.lognormal(0.0, 1.0, (len(index), n)),
                        index=list(index), columns=[f"S{i}" for i in range(n)])


res = files("iobrpy.resources")   # LM22 / signatures / annotation packaged with iobrpy

# --- part 1: symbol-indexed cohort (200 LM22 genes x 20 samples) ---
lm22 = pd.read_csv(res.joinpath("lm22.txt"), sep=r"\s+", engine="python", index_col=0)
eset = synth(lm22.index[:200])

cib = iobrx.cibersort(eset.copy(), perm=10, QN=True)      # Rust NuSVR, rayon threads
pca = iobrx.calculate_sig_score(eset.copy(), "signature_collection", method="pca")
ssg = iobrx.calculate_sig_score(eset.copy(), "signature_collection", method="ssgsea")
est = iobrx.estimate_score(eset.copy(), platform="affy")
mcp = iobrx.mcpcounter(eset.copy(), features_type="HUGO_symbols")
print("cibersort/PCA/ssGSEA/ESTIMATE/MCP:",
      cib.shape, pca.shape, ssg.shape, est.shape, mcp.shape)

# --- part 2: Ensembl counts cohort -> symbols + TPM (200 probes x 20) ---
with open(str(res.joinpath("anno_eset.pkl")), "rb") as f:
    anno = pd.DataFrame(pickle.load(f)["anno_grch38"])
counts = synth(anno["id"].dropna().sample(200, random_state=7).tolist())
sym = iobrx.anno_eset(counts.copy(), "anno_grch38", symbol="symbol", probe="id")
tpm = iobrx.count2tpm(counts.copy(), idType="Ensembl", org="hsa")
print("anno_eset/count2tpm:", sym.shape, tpm.shape)

# --- part 3: deconvolution against the packaged references ---
til = pd.read_pickle(str(res.joinpath("quantiseq_data.pkl")))["TIL10_signature"]
tref = pd.read_pickle(str(res.joinpath("epic_TRef_BRef.pkl")))["TRef"]
epi_genes = list(dict.fromkeys(list(tref["sigGenes"])
                               + list(tref["refProfiles"].index[:150])))[:200]
til_out = iobrx.quantiseq(synth(til.index.astype(str)[:200]))          # TIL10, lsei
epi_out = iobrx.epic(synth(epi_genes))                                 # TRef
print("quanTIseq/EPIC:", til_out.shape, epi_out["cellFractions"].shape)

# threads: n_threads=None -> min(8, os.cpu_count()); per call or globally
iobrx.set_threads(64)
```

```python
# --- official cohorts (optional): the documented data resolution ---------
# iobrx.load_official(name) resolves $IOBRX_TESTDATA/<name>.parquet|.rda,
# then a local cache, then the IOBR data-v1.0 release via mirrors
# (github.com direct, gh-proxy.com, ghproxy.net). Offline it raises
# iobrx.OfficialDataUnavailable with the remediation spelled out.
import iobrx

try:
    eset = iobrx.load_official("imvigor210_eset")       # 872 genes x 348 samples
except iobrx.OfficialDataUnavailable as exc:
    print(f"[official-data demo skipped] {exc}")
else:
    cib = iobrx.cibersort(eset.copy(), perm=100, QN=True)
    print(cib.shape)                                    # (348, 25)
    both = iobrx.calculate_sig_score(eset.copy(), "signature_collection",
                                     method="integration")
    est = iobrx.estimate_score(eset.copy(), platform="affy")
```

The official STAD cohort calls of the older quickstart (anno_eset, EPIC,
quanTIseq, MCP-counter, count2TPM on `eset_stad` × `anno_grch38`) are the 11
stages of [`examples/full_workflow.py`](examples/full_workflow.py), which
also prints the per-stage timing table. Runnable scripts:
[`examples/quickstart.py`](examples/quickstart.py) (official IMvigor210
cohort, per-step timings) and
[`examples/full_workflow.py`](examples/full_workflow.py) (optional
`--verify-parity`).

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

Everything measured — six routes, per-stage tables, cibersort thread scaling
1–224T, cold start & memory, the adversarial precision audit and the shipped
verdict — is documented in **[BENCHMARKS.md](BENCHMARKS.md)**, and every
number there traces to a verbatim artifact in [`bench/`](bench/README.md).

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
   the bulk (e.g. the curated 872-gene IMvigor210 matrix has **zero** EPIC
   TRef overlap — run EPIC on a full transcriptome, not on a curated panel).
   Very small overlaps (e.g. imvigor210's 16 LM22 genes) still run,
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
6. **Thread guidance.** `n_threads=None` (the default) resolves to
   `min(8, os.cpu_count())` — deliberately conservative: each extra rayon
   thread costs ~1.24 MiB resident during CIBERSORT (+277 MiB at 224T vs
   +9.5 MiB at 1T), and beyond ~64 threads the official 10-sample stage
   barely improves (128T 2.91 s vs 224T 3.06 s; HT logical cores are slightly
   *slower*). For large cohorts raise `n_threads` per call or via
   `iobrx.set_threads(n)`; the full scaling table is BENCHMARKS.md §5.

## FAQ

- **Why bit-exact? Why not just "close enough"?** Because "close" is not
  small here: varying only the NuSVR stopping tolerance moves reported cell
  fractions by up to ~8 percentage points on the official problem
  (tolerance experiment, BENCHMARKS.md §7). Any faster approximate solver
  would change results at scientifically meaningful magnitude, so iobrx
  vendors the original solver semantics (byte-identical libsvm, exact numpy
  reduction orders, even pandas' exact csv float parse).
- **Can I relax bit-exactness for more speed?** Not via this package: the
  shipped route is the fastest route that passed the bit-exact gate
  (BENCHMARKS.md §9). You can, of course, call the underlying `iobrpy`
  originals yourself with different solver settings.
- **P-values differ run-to-run — is that a bug?** No. The ORIGINAL seeds
  permutations from OS entropy, so its P-values are unreproducible *by
  design*; iobrx's P-values are seeded (stable, thread-invariant) with the
  identical formula. Everything else is bit-exact (Caveat 1).
- **How do I cite this?** Cite IOBRpy and the underlying method papers
  (CIBERSORT: Newman 2015, doi:10.1038/nmeth.3337; EPIC, quanTIseq,
  MCP-counter, ESTIMATE, ssGSEA as applicable), and link this repository +
  BENCHMARKS.md for the acceleration claims. Related Python→Rust precedent:
  GSEApy (doi:10.1093/bioinformatics/btac757); see BENCHMARKS.md §8.

## Contributing

Bug reports and PRs are welcome. The parity contract is the contract: any
change to `src/` or `rust/` must keep the 11 official gates bit-exact
(`pytest -m full`, data via `IOBRX_TESTDATA` or the IOBR `data-v1.0`
release) and pass the 13 synthetic smoke tests. Do not edit anything under
`bench/` — those are frozen measurement artifacts.

## License

MIT — see [LICENSE](LICENSE). Vendored third-party sources keep their own
notices: [`rust/vendor/THIRD_PARTY_NOTICES.md`](rust/vendor/THIRD_PARTY_NOTICES.md).
