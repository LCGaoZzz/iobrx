use std::path::PathBuf;

// Self-contained build: every vendored C/C++ source lives under
// <this crate>/vendor. Nothing outside the repo is referenced.
//
// IMPORTANT (bit-exactness): sklearn wheels are compiled for the baseline
// x86-64 target (SSE2, no FMA/BMI). Compiling with -march=native would let
// gcc contract a*b+c into FMA (default -ffp-contract=fast), changing
// rounding inside the SMO solver. We stay on baseline and additionally
// disable contraction explicitly. -O2/-O3 does not alter IEEE semantics.
fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("vendor");
    let wrapper = manifest.join("wrapper.cpp");
    let argsort_wrapper = manifest.join("argsort_wrapper.cpp");
    let sklearn_libsvm = vendor.join("sklearn/svm/src/libsvm");
    let sklearn_src = vendor.join("sklearn/svm/src");
    let xss = vendor.join("x86-simd-sort");

    cc::Build::new()
        .cpp(true)
        .opt_level(3)
        .flag("-std=c++17")
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .flag("-fno-unsafe-math-optimizations")
        .include(&vendor)
        .include(&sklearn_libsvm)
        .include(&sklearn_src)
        .file(&wrapper)
        .compile("iobrx_svm");

    // numpy's AVX512_SKX dispatch variant of x86-simd-sort argsort: compiled
    // with AVX512F+VL+DQ (+BW, matching SKX feature detection). Pure
    // comparison/integer SIMD - no FP arithmetic - so the permutation is
    // compiler-version independent.
    cc::Build::new()
        .cpp(true)
        .opt_level(3)
        .flag("-std=c++17")
        .flag("-mavx512f")
        .flag("-mavx512vl")
        .flag("-mavx512dq")
        .flag("-mavx512bw")
        .flag("-fno-fast-math")
        .include(&vendor)
        .include(&xss)
        .file(&argsort_wrapper)
        .compile("iobrx_argsort");

    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={}", wrapper.display());
    println!("cargo:rerun-if-changed={}", argsort_wrapper.display());
    println!("cargo:rerun-if-changed={}", vendor.join("svm.cpp").display());
    println!("cargo:rerun-if-changed={}", vendor.join("svm.h").display());
}
