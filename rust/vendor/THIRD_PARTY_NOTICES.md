# Third-party notices for vendored sources

The Rust crate vendors three third-party C/C++ sources under this directory.
The scikit-learn/libsvm sources are compiled, unmodified, into the native
extension. The historical x86-simd-sort snapshot is retained for provenance
but is **not compiled or linked in version 0.2.0**; sorting now calls NumPy.

## sklearn/svm/src/libsvm/svm.cpp, svm.h, _svm_cython_blas_helpers.h
## sklearn/svm/src/newrand/newrand.h

- Source: scikit-learn 1.7.2 (https://scikit-learn.org), files
  `sklearn/svm/src/libsvm/{svm.cpp,svm.h,_svm_cython_blas_helpers.h}` and
  `sklearn/svm/src/newrand/newrand.h`. `svm.cpp`/`svm.h` originate from
  LIBSVM (Chih-Chung Chang and Chih-Jen Lin, National Taiwan University).
- License: BSD 3-Clause — Copyright (c) 2007–2025 The scikit-learn developers.
  https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
- Why vendored: byte-identical libsvm sources are required for bit-exact
  reproduction of sklearn's NuSVR solves; they are compiled with sklearn's
  baseline x86-64 target flags (no FMA contraction) to preserve rounding.

## x86-simd-sort/

- Source: Intel's x86-simd-sort (https://github.com/intel/x86-simd-sort),
  as vendored by NumPy at NumPy's pinned submodule commit (9a1b616d),
  `src/*.{h,hpp}` only.
- License: Apache License 2.0 — Copyright (c) 2022 Intel Corporation.
  http://www.apache.org/licenses/LICENSE-2.0
- Why vendored: the AVX-512 SKX argsort kernel is the exact permutation
  kernel numpy dispatches to on AVX-512F+DQ CPUs; compiling it here makes
  quantile-normalization tie ordering bit-identical to `np.argsort`.
