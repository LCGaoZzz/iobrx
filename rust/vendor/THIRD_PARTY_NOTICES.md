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
- License: BSD 3-Clause — Copyright (c) 2007–2024 The scikit-learn developers.
  Full terms: `LICENSE.scikit-learn` (from scikit-learn 1.7.2's `COPYING`)
  and `LICENSE.libsvm` (copied from the vendored `svm.cpp` header, including
  Copyright (c) 2000–2009 Chih-Chung Chang and Chih-Jen Lin).
- Why vendored: byte-identical libsvm sources are required for bit-exact
  reproduction of sklearn's NuSVR solves; they are compiled with sklearn's
  baseline x86-64 target flags (no FMA contraction) to preserve rounding.

## x86-simd-sort/

- Source: Intel's x86-simd-sort (https://github.com/intel/x86-simd-sort),
  as vendored by NumPy at NumPy's pinned submodule commit (9a1b616d),
  `src/*.{h,hpp}` only.
- License: Apache License 2.0 — Copyright (c) 2022 Intel Corporation.
  Full terms: `LICENSE.x86-simd-sort` (Apache License 2.0).
- Why retained: this snapshot supplied the original AVX-512 SKX argsort
  kernel. Version 0.2.0 uses local NumPy dispatch instead and retains these
  uncompiled sources solely for provenance.
