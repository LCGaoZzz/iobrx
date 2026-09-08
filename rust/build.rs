use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("vendor");
    let wrapper = manifest.join("wrapper.cpp");
    let mut build = cc::Build::new();
    build.cpp(true).opt_level(3).std("c++17");
    // Keep original libsvm arithmetic; no host-specific ISA or contraction.
    if build.get_compiler().is_like_msvc() {
        build.flag("/fp:strict");
    } else {
        build.flag("-ffp-contract=off")
            .flag("-fno-fast-math")
            .flag("-fno-unsafe-math-optimizations");
    }
    build.include(&vendor)
        .include(vendor.join("sklearn/svm/src/libsvm"))
        .include(vendor.join("sklearn/svm/src"))
        .file(&wrapper)
        .compile("iobrx_svm");
    // NumPy dispatches sorting, including tie order. No AVX-512 object is linked.
    println!("cargo:rerun-if-changed={}", wrapper.display());
    println!("cargo:rerun-if-changed={}", vendor.display());
}
