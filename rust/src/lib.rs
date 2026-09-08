use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// ------------------------------------------------------------------
// process-global rayon pool registry (lazily initialized, keyed by n_threads)
// ------------------------------------------------------------------
// cibersort_core / ssgsea_core used to build a fresh
// rayon::ThreadPoolBuilder().num_threads(n).build() inside EVERY call, so
// repeated calls kept paying pool (thread spawn + join) construction. Pools
// are now built ONCE per distinct thread count and reused for the process
// lifetime. Scheduling semantics are unchanged (same work-stealing pool
// type, same thread count), so float results are unaffected (verified:
// outputs stay bit-identical to refs and across n_threads 1/64/224).
fn global_pool(n_threads: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    static REGISTRY: OnceLock<Mutex<HashMap<u32, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let key = n_threads as u32;
    let mut map = registry
        .lock()
        .map_err(|e| format!("rayon pool registry lock poisoned: {}", e))?;
    if let Some(p) = map.get(&key) {
        return Ok(Arc::clone(p));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| format!("rayon pool build failed: {}", e))?,
    );
    map.insert(key, Arc::clone(&pool));
    Ok(pool)
}

// ------------------------------------------------------------------
// extern C surface (baseline libsvm wrapper.cpp)
// ------------------------------------------------------------------
extern "C" {
    fn iobrx_svm_init();
    fn iobrx_set_dot(f: *const std::ffi::c_void);
    fn iobrx_svm_ready() -> i32;
    fn iobrx_nusvr_linear_fit(
        x: *const f64,
        l: i32,
        dim: i32,
        y: *const f64,
        nu: f64,
        c: f64,
        tol: f64,
        cache_mb: f64,
        shrinking: i32,
        out_dual_coef: *mut f64,
        out_sv_ind: *mut i32,
        out_n_iter: *mut i32,
        out_fit_status: *mut i32,
    ) -> i32;
}

// ------------------------------------------------------------------
// BLAS function pointers (resolved from Python via ctypes addresses)
// ------------------------------------------------------------------
// scipy LP64 openblas ddot (Fortran): double ddot_(int*, double*, int*, double*, int*)
type FortranDdot = unsafe extern "C" fn(
    *const i32,
    *const f64,
    *const i32,
    *const f64,
    *const i32,
) -> f64;
// numpy ILP64 openblas cblas_dgemv (C interface, int64 args):
// void cblas_dgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
type Gemv64 =
    unsafe extern "C" fn(i64, i64, i64, i64, f64, *const f64, i64, *const f64, i64, f64, *mut f64, i64);

static G_DDOT: AtomicU64 = AtomicU64::new(0);
static G_GEMV64: AtomicU64 = AtomicU64::new(0);
static G_DGESDD: AtomicU64 = AtomicU64::new(0);
static G_SVM_INITED: AtomicBool = AtomicBool::new(false);

// scipy LP64 openblas LAPACK dgesdd (Fortran):
// subroutine dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info)
type FortranDgesdd = unsafe extern "C" fn(
    *const u8,        // jobz ('S' for compute_uv with full_matrices=.false.)
    *const i32,       // m
    *const i32,       // n
    *mut f64,         // a (column-major m x n, overwritten)
    *const i32,       // lda
    *mut f64,         // s (min(m,n))
    *mut f64,         // u (m x min(m,n) column-major, ldu = m)
    *const i32,       // ldu
    *mut f64,         // vt (min(m,n) x n column-major, ldvt = min(m,n))
    *const i32,       // ldvt
    *mut f64,         // work
    *const i32,       // lwork
    *mut i32,         // iwork (8 * min(m,n))
    *mut i32,         // info
);

#[inline]
fn get_dgesdd() -> Option<FortranDgesdd> {
    let a = G_DGESDD.load(Ordering::Relaxed);
    if a == 0 {
        None
    } else {
        Some(unsafe { std::mem::transmute::<usize, FortranDgesdd>(a as usize) })
    }
}

const CBLAS_ROWMAJOR: i64 = 101;
const CBLAS_COLMAJOR: i64 = 102;
const CBLAS_NOTRANS: i64 = 111;
const CBLAS_TRANS: i64 = 112;

#[inline]
fn get_ddot() -> Option<FortranDdot> {
    let a = G_DDOT.load(Ordering::Relaxed);
    if a == 0 {
        None
    } else {
        Some(unsafe { std::mem::transmute::<usize, FortranDdot>(a as usize) })
    }
}

#[inline]
fn gemv(
    layout: i64,
    trans: i64,
    m: i64,
    n: i64,
    a: *const f64,
    lda: i64,
    x: *const f64,
    y: *mut f64,
) -> bool {
    let f = G_GEMV64.load(Ordering::Relaxed);
    if f == 0 {
        return false;
    }
    let f: Gemv64 = unsafe { std::mem::transmute::<usize, Gemv64>(f as usize) };
    unsafe {
        f(
            layout,
            trans,
            m,
            n,
            1.0,
            a,
            lda,
            x,
            1,
            0.0,
            y,
            1,
        )
    };
    true
}

#[pyfunction]
#[pyo3(name = "init_blas", signature = (ddot_addr, gemv64_addr, dgesdd_addr=0))]
fn init_blas_py(_py: Python, ddot_addr: u64, gemv64_addr: u64, dgesdd_addr: u64) -> PyResult<(bool, bool)> {
    // Addresses of 0 leave the corresponding pointer untouched, so an
    // independent late initializer (e.g. the sig-score PCA leg resolving only
    // scipy_dgesdd_) cannot clobber pointers a prior init_blas call set.
    unsafe {
        if ddot_addr != 0 {
            iobrx_set_dot(ddot_addr as *const std::ffi::c_void);
            G_DDOT.store(ddot_addr, Ordering::Relaxed);
        }
    }
    if gemv64_addr != 0 {
        G_GEMV64.store(gemv64_addr, Ordering::Relaxed);
    }
    if dgesdd_addr != 0 {
        G_DGESDD.store(dgesdd_addr, Ordering::Relaxed);
    }
    if !G_SVM_INITED.swap(true, Ordering::SeqCst) {
        unsafe { iobrx_svm_init() };
    }
    Ok((ddot_addr != 0, gemv64_addr != 0))
}

#[pyfunction]
fn blas_ready() -> (bool, bool, bool, bool) {
    (
        unsafe { iobrx_svm_ready() != 0 },
        G_GEMV64.load(Ordering::Relaxed) != 0,
        G_DGESDD.load(Ordering::Relaxed) != 0,
        G_SVM_INITED.load(Ordering::Relaxed),
    )
}

// ------------------------------------------------------------------
// numpy-exact reductions
// ------------------------------------------------------------------
// numpy pairwise summation (numpy/_core/src/umath/loops_utils.h.src, PW_BLOCKSIZE=128),
// generic over f64/f32. n<8 starts at -0.0 to preserve signed zeros.
#[inline]
fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for i in 0..n {
            res += a[i];
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let n8 = n - (n % 8);
        let mut i = 8;
        while i < n8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
    }
}

#[inline]
fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f32;
        for i in 0..n {
            res += a[i];
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let n8 = n - (n % 8);
        let mut i = 8;
        while i < n8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

// numpy's nditer reduction buffer chunks the pairwise sum at 8192 elements
// and accumulates the block sums sequentially (empirically verified vs np.sum).
const NP_REDUCE_CHUNK: usize = 8192;

#[inline]
fn np_sum_f64(a: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for chunk in a.chunks(NP_REDUCE_CHUNK) {
        acc += pairwise_sum_f64(chunk);
    }
    acc
}

#[inline]
fn np_sum_f32(a: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for chunk in a.chunks(NP_REDUCE_CHUNK) {
        acc += pairwise_sum_f32(chunk);
    }
    acc
}

// axis-0 reduction of a C-contiguous (g, n) matrix: per-column sequential row
// accumulation (numpy nditer: out[j] += a[i, j] in row order).
fn np_sum_axis0(y: &[f64], g: usize, n: usize, out: &mut [f64]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for i in 0..g {
        let row = &y[i * n..(i + 1) * n];
        for j in 0..n {
            out[j] += row[j];
        }
    }
}



// zscore1d from cibersort.py (ddof=1, zeros on zero variance / size<2)
fn zscore1d(a: &[f64], scratch: &mut Vec<f64>) -> Vec<f64> {
    let n = a.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let m = np_sum_f64(a) / (n as f64);
    scratch.clear();
    scratch.extend(a.iter().map(|&v| v - m));
    let mut d2 = Vec::with_capacity(n);
    for &d in scratch.iter() {
        d2.push(d * d);
    }
    let v = np_sum_f64(&d2) / ((n - 1) as f64);
    if v == 0.0 {
        return vec![0.0; n];
    }
    let sv = v.sqrt();
    scratch.iter().map(|&d| d / sv).collect()
}

// corr_pearson_fast from cibersort.py (numpy op order preserved)
fn corr_pearson_fast(a: &[f64], b: &[f64], buf: &mut Vec<f64>) -> f64 {
    let n = a.len();
    let am = np_sum_f64(a) / (n as f64);
    let bm = np_sum_f64(b) / (n as f64);
    buf.clear();
    buf.extend(a.iter().map(|&v| v - am));
    let av: Vec<f64> = buf.clone();
    buf.clear();
    buf.extend(b.iter().map(|&v| v - bm));
    let bv: Vec<f64> = buf.clone();
    let mut saa = Vec::with_capacity(n);
    let mut sbb = Vec::with_capacity(n);
    let mut sab = Vec::with_capacity(n);
    for i in 0..n {
        saa.push(av[i] * av[i]);
        sbb.push(bv[i] * bv[i]);
        sab.push(av[i] * bv[i]);
    }
    let saa = np_sum_f64(&saa);
    let sbb = np_sum_f64(&sbb);
    let sab = np_sum_f64(&sab);
    let denom = (saa * sbb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        sab / denom
    }
}

// ------------------------------------------------------------------
// quantile normalization (numpy-exact incl. argsort tie semantics)
// ------------------------------------------------------------------
// Y is row-major (g, n). Replicates quantile_normalize_fast:
//   order = np.argsort(Y, axis=0)                      -> local NumPy CPU dispatch
//   sorted_Y = take_along_axis(Y, order, axis=0)
//   mean_sorted = sorted_Y.mean(axis=1)                -> pairwise over row of n
//   inv_order[order[r,j], j] = r
//   out[i,j] = mean_sorted[inv_order[i,j]]
fn quantile_normalize(py: Python, y: &mut [f64], g: usize, n: usize) -> PyResult<()> {
    let input = y.to_vec().into_pyarray_bound(py).reshape([g, n])?;
    let result = py.import_bound("iobrx._sorting")?
        .getattr("quantile_normalize")?.call1((input,))?;
    let result = result.extract::<PyReadonlyArray2<f64>>()?;
    y.copy_from_slice(result.as_slice().map_err(|_| PyRuntimeError::new_err("NumPy sort result must be contiguous"))?);
    Ok(())
}

// ------------------------------------------------------------------
// ORIGINAL input semantics: pandas read_csv float parse (in memory)
// ------------------------------------------------------------------
// The reference cibersort() is file-based: the official protocol feeds it
// df.to_csv(temp.csv) and it runs pd.read_csv(path, sep=None, engine='python'),
// whose numeric conversion is pandas' precise_xstrtod
// (pandas/_libs/src/parser/tokenizer.c; bitwise identical to the C engine's
// float_precision=None/'high' - verified on the full BLCA matrix). That parse
// is NOT the identity on arbitrary float64 values: accumulating up to 17
// significant digits in double arithmetic and scaling once by a table power
// of ten is off by 1-2 ulp on ~12% of values (BLCA held-out symbol matrix:
// 23930/205050 cells, max 2 ulp). On the degenerate mixture TCGA-2F-A9KR
// (|Correlation| ~ 0.002) those last bits flip the NuSVR support set and move
// 15/22 weights by up to 1.79e-3. The shim therefore maps the incoming
// mixture values through the exact same parse, computed in memory:
// token = python repr(v) (what to_csv writes for a float64 cell - verified
// equal to repr on >700k dataset cells + a 10k synthetic battery),
// value = precise_xstrtod(token). Validated bitwise against the real
// to_csv->read_csv(engine='python') round trip on the full BLCA (205050) and
// STAD (501810) symbol matrices: 0 differing cells.

static E10: [f64; 309] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5,
    1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17,
    1e18, 1e19, 1e20, 1e21, 1e22, 1e23,
    1e24, 1e25, 1e26, 1e27, 1e28, 1e29,
    1e30, 1e31, 1e32, 1e33, 1e34, 1e35,
    1e36, 1e37, 1e38, 1e39, 1e40, 1e41,
    1e42, 1e43, 1e44, 1e45, 1e46, 1e47,
    1e48, 1e49, 1e50, 1e51, 1e52, 1e53,
    1e54, 1e55, 1e56, 1e57, 1e58, 1e59,
    1e60, 1e61, 1e62, 1e63, 1e64, 1e65,
    1e66, 1e67, 1e68, 1e69, 1e70, 1e71,
    1e72, 1e73, 1e74, 1e75, 1e76, 1e77,
    1e78, 1e79, 1e80, 1e81, 1e82, 1e83,
    1e84, 1e85, 1e86, 1e87, 1e88, 1e89,
    1e90, 1e91, 1e92, 1e93, 1e94, 1e95,
    1e96, 1e97, 1e98, 1e99, 1e100, 1e101,
    1e102, 1e103, 1e104, 1e105, 1e106, 1e107,
    1e108, 1e109, 1e110, 1e111, 1e112, 1e113,
    1e114, 1e115, 1e116, 1e117, 1e118, 1e119,
    1e120, 1e121, 1e122, 1e123, 1e124, 1e125,
    1e126, 1e127, 1e128, 1e129, 1e130, 1e131,
    1e132, 1e133, 1e134, 1e135, 1e136, 1e137,
    1e138, 1e139, 1e140, 1e141, 1e142, 1e143,
    1e144, 1e145, 1e146, 1e147, 1e148, 1e149,
    1e150, 1e151, 1e152, 1e153, 1e154, 1e155,
    1e156, 1e157, 1e158, 1e159, 1e160, 1e161,
    1e162, 1e163, 1e164, 1e165, 1e166, 1e167,
    1e168, 1e169, 1e170, 1e171, 1e172, 1e173,
    1e174, 1e175, 1e176, 1e177, 1e178, 1e179,
    1e180, 1e181, 1e182, 1e183, 1e184, 1e185,
    1e186, 1e187, 1e188, 1e189, 1e190, 1e191,
    1e192, 1e193, 1e194, 1e195, 1e196, 1e197,
    1e198, 1e199, 1e200, 1e201, 1e202, 1e203,
    1e204, 1e205, 1e206, 1e207, 1e208, 1e209,
    1e210, 1e211, 1e212, 1e213, 1e214, 1e215,
    1e216, 1e217, 1e218, 1e219, 1e220, 1e221,
    1e222, 1e223, 1e224, 1e225, 1e226, 1e227,
    1e228, 1e229, 1e230, 1e231, 1e232, 1e233,
    1e234, 1e235, 1e236, 1e237, 1e238, 1e239,
    1e240, 1e241, 1e242, 1e243, 1e244, 1e245,
    1e246, 1e247, 1e248, 1e249, 1e250, 1e251,
    1e252, 1e253, 1e254, 1e255, 1e256, 1e257,
    1e258, 1e259, 1e260, 1e261, 1e262, 1e263,
    1e264, 1e265, 1e266, 1e267, 1e268, 1e269,
    1e270, 1e271, 1e272, 1e273, 1e274, 1e275,
    1e276, 1e277, 1e278, 1e279, 1e280, 1e281,
    1e282, 1e283, 1e284, 1e285, 1e286, 1e287,
    1e288, 1e289, 1e290, 1e291, 1e292, 1e293,
    1e294, 1e295, 1e296, 1e297, 1e298, 1e299,
    1e300, 1e301, 1e302, 1e303, 1e304, 1e305,
    1e306, 1e307, 1e308,
];

// Exact-tie detection lives in tie_python_takes_lower_bytes (below, next to
// build_repr_token): true iff 2*v == (2D-1) * 10^k, i.e. v is exactly midway
// between the two adjacent shortest-decimal candidates (D-1)*10^k and D*10^k.
// CPython repr (Gay dtoa mode 0) then takes the EVEN last digit; rust {:e}
// takes the upper one, so an odd D must be stepped down.

// Build the python-repr token for a finite f64 into `tok` (capacity >= 32),
// returning its length. Zero heap: the {:e} shortest digits go through a
// stack fmt::Write, digits/tie-fix/token all live in stack buffers.
fn build_repr_token(v: f64, tok: &mut [u8]) -> usize {
    if v == 0.0 {
        let s: &[u8] = if v.is_sign_negative() { b"-0.0" } else { b"0.0" };
        tok[..s.len()].copy_from_slice(s);
        return s.len();
    }
    struct W {
        b: [u8; 48],
        n: usize,
    }
    impl core::fmt::Write for W {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let bytes = s.as_bytes();
            self.b[self.n..self.n + bytes.len()].copy_from_slice(bytes);
            self.n += bytes.len();
            Ok(())
        }
    }
    use core::fmt::Write as _;
    let mut w = W { b: [0u8; 48], n: 0 };
    let _ = write!(w, "{:e}", v); // shortest round-trip, e.g. "4.5934999998440803e5"
    let sci = &w.b[..w.n];
    // split mantissa / exponent at 'e'
    let e_pos = sci.iter().position(|&c| c == b'e').unwrap();
    let mant = &sci[..e_pos];
    let mut x: i32 = 0;
    for &c in &sci[e_pos + 1..] {
        if c.is_ascii_digit() {
            x = x * 10 + (c - b'0') as i32;
        }
    }
    if sci[e_pos + 1] == b'-' {
        x = -x;
    }
    let neg = mant[0] == b'-';
    // digits: mantissa without '-' and '.'
    let mut dg: [u8; 24] = [0; 24];
    let mut nd: usize = 0;
    for &c in mant {
        if c.is_ascii_digit() {
            dg[nd] = c;
            nd += 1;
        }
    }
    // python tie correction: exact halfway cases take the EVEN last digit
    // (Gay dtoa mode 0); rust {:e} takes the upper -> step down when odd.
    if tie_python_takes_lower_bytes(v, &dg[..nd], x) {
        let mut i = nd;
        loop {
            i -= 1;
            if dg[i] > b'0' {
                dg[i] -= 1;
                break;
            }
            dg[i] = b'9';
        }
        if dg[0] == b'0' {
            // D was 10...0 (defensive; odd D cannot decrement to a shorter
            // length, but keep the value correct regardless)
            dg.copy_within(1..nd, 0);
            nd -= 1;
            x -= 1;
        }
    }
    let n = nd as i32;
    let mut t = 0usize;
    if neg {
        tok[t] = b'-';
        t += 1;
    }
    if x >= 16 || x <= -5 {
        tok[t] = dg[0];
        t += 1;
        if n > 1 {
            tok[t] = b'.';
            t += 1;
            tok[t..t + nd - 1].copy_from_slice(&dg[1..nd]);
            t += nd - 1;
        }
        tok[t] = b'e';
        t += 1;
        tok[t] = if x >= 0 { b'+' } else { b'-' };
        t += 1;
        let a = x.unsigned_abs();
        if a < 10 {
            tok[t] = b'0';
            t += 1;
        }
        let mut abuf = [0u8; 4];
        let as_ = a.to_string();
        let ab = as_.as_bytes();
        abuf[..ab.len()].copy_from_slice(ab);
        tok[t..t + ab.len()].copy_from_slice(&abuf[..ab.len()]);
        t += ab.len();
    } else if x >= n - 1 {
        tok[t..t + nd].copy_from_slice(&dg[..nd]);
        t += nd;
        for _ in 0..(x - (n - 1)) {
            tok[t] = b'0';
            t += 1;
        }
        tok[t] = b'.';
        t += 1;
        tok[t] = b'0';
        t += 1;
    } else if x >= 0 {
        let ip = (x + 1) as usize;
        tok[t..t + ip].copy_from_slice(&dg[..ip]);
        t += ip;
        tok[t] = b'.';
        t += 1;
        tok[t..t + nd - ip].copy_from_slice(&dg[ip..nd]);
        t += nd - ip;
    } else {
        tok[t] = b'0';
        t += 1;
        tok[t] = b'.';
        t += 1;
        for _ in 0..(-x - 1) {
            tok[t] = b'0';
            t += 1;
        }
        tok[t..t + nd].copy_from_slice(&dg[..nd]);
        t += nd;
    }
    t
}

// byte-slice twin of tie_python_takes_lower
fn tie_python_takes_lower_bytes(v: f64, digits: &[u8], x: i32) -> bool {
    if digits.len() < 2 {
        return false;
    }
    let mut d: u128 = 0;
    for &c in digits {
        d = d * 10 + (c - b'0') as u128;
    }
    if d % 2 == 0 {
        return false;
    }
    let n = digits.len() as i32;
    let k = x - n + 1;
    let b = v.to_bits();
    let exp_bits = ((b >> 52) & 0x7FF) as i32;
    let (mut m, e2): (u64, i32) = if exp_bits == 0 {
        (b & 0x000F_FFFF_FFFF_FFFF, -1074)
    } else {
        (
            ((b & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000) as u64,
            exp_bits - 1075,
        )
    };
    if m == 0 {
        return false;
    }
    let mut u_l = e2 + 1;
    while m % 2 == 0 {
        m /= 2;
        u_l += 1;
    }
    let mut w_l: i32 = 0;
    while m % 5 == 0 {
        m /= 5;
        w_l += 1;
    }
    let mut q: u128 = 2 * d - 1;
    let mut w_r: i32 = 0;
    while q % 5 == 0 {
        q /= 5;
        w_r += 1;
    }
    (m as u128) == q && u_l == k && w_l == k + w_r
}

// python repr(float) token for a finite f64 = what pandas to_csv writes for a
// float64 cell. Positional iff -4 <= exp10 <= 15, else scientific with at
// least 2 exponent digits; integral positional values get a trailing ".0".
// Ties: CPython repr (Gay dtoa mode 0) breaks exact halfway cases
// round-half-EVEN on the last digit; Rust's {:e} breaks them upward, so an
// exact tie with an odd upper candidate is corrected down by one digit
// (verified: X.25-exact -> repr "...2" [even, lower] vs rust "...3";
// X.75-exact -> repr "...8" [even, upper] == rust "...8").
fn py_repr_f64(v: f64, buf: &mut String) {
    let mut tmp = [0u8; 40];
    let len = build_repr_token(v, &mut tmp);
    buf.push_str(std::str::from_utf8(&tmp[..len]).unwrap());
}

// verbatim port of pandas 2.3.3 precise_xstrtod (tokenizer.c) for the clean
// numeric tokens py_repr_f64 emits (no whitespace / thousands separators /
// signs beyond the leading one). Same accumulation order, same single-table
// power scaling, same subnormal double-division path.
fn precise_xstrtod(s: &[u8]) -> f64 {
    let max_digits = 17usize;
    let mut p = 0usize;
    let n = s.len();
    let mut negative = false;
    if p < n && (s[p] == b'+' || s[p] == b'-') {
        negative = s[p] == b'-';
        p += 1;
    }
    let mut number = 0.0f64;
    let mut exponent: i32 = 0;
    let mut num_digits = 0usize;
    let mut num_decimals = 0usize;
    // Process string of digits (integer part).
    while p < n && s[p].is_ascii_digit() {
        if num_digits < max_digits {
            number = number * 10.0 + (s[p] - b'0') as f64;
            num_digits += 1;
        } else {
            exponent += 1;
        }
        p += 1;
    }
    // Process decimal part.
    if p < n && s[p] == b'.' {
        p += 1;
        while num_digits < max_digits && p < n && s[p].is_ascii_digit() {
            number = number * 10.0 + (s[p] - b'0') as f64;
            num_digits += 1;
            num_decimals += 1;
            p += 1;
        }
        // Consume extra decimal digits (dropped).
        while p < n && s[p].is_ascii_digit() {
            p += 1;
        }
        exponent -= num_decimals as i32;
    }
    if num_digits == 0 {
        // C: *error = ERANGE; return 0.0
        return 0.0;
    }
    // Correct for sign.
    if negative {
        number = -number;
    }
    // Process an exponent string.
    if p < n && (s[p] == b'e' || s[p] == b'E') {
        p += 1;
        let mut neg = false;
        if p < n && (s[p] == b'+' || s[p] == b'-') {
            neg = s[p] == b'-';
            p += 1;
        }
        let mut nd = 0usize;
        let mut ev: i32 = 0;
        while nd < max_digits && p < n && s[p].is_ascii_digit() {
            ev = ev * 10 + (s[p] - b'0') as i32;
            nd += 1;
            p += 1;
        }
        exponent = if neg { exponent - ev } else { exponent + ev };
    }
    // Scale the result.
    if exponent > 308 {
        f64::INFINITY // C: *error = ERANGE; return HUGE_VAL
    } else if exponent > 0 {
        number * E10[exponent as usize]
    } else if exponent < -308 {
        if exponent < -616 {
            0.0
        } else {
            let mut r = number / E10[(-308 - exponent) as usize];
            r /= E10[308];
            r
        }
    } else {
        number / E10[(-exponent) as usize]
    }
}

#[inline]
fn csv_parse_one(v: f64) -> f64 {
    if !v.is_finite() {
        return v; // to_csv writes ''/inf; read_csv restores NaN/inf
    }
    let mut tok = [0u8; 40];
    let len = build_repr_token(v, &mut tok);
    precise_xstrtod(&tok[..len])
}

/// Debug/parity helper: the exact token py_repr_f64 builds for v (what the
/// parse consumes). Compare with python repr(v) / pandas to_csv output.
#[pyfunction]
#[pyo3(name = "csv_repr_token")]
fn csv_repr_token_py(v: f64) -> String {
    let mut s = String::new();
    py_repr_f64(v, &mut s);
    s
}

/// Elementwise v -> precise_xstrtod(repr(v)): the exact float64 the ORIGINAL
/// file-based pipeline computes with after df.to_csv() ->
/// pd.read_csv(sep=None, engine='python'). NaN/±inf pass through. Any input
/// layout; returns a C-order array of the same shape. n_threads=0 uses the
/// rayon default pool, otherwise the crate's pooled registry (the same pools
/// cibersort_core uses).
#[pyfunction]
#[pyo3(name = "csv_parse_roundtrip", signature = (x, n_threads=0u32))]
fn csv_parse_roundtrip_py(
    py: Python,
    x: PyReadonlyArray2<f64>,
    n_threads: u32,
) -> PyResult<Py<PyArray2<f64>>> {
    let (r, c) = (x.shape()[0], x.shape()[1]);
    let view = x.as_array();
    let mut data: Vec<f64> = Vec::with_capacity(r * c);
    if view.is_standard_layout() {
        data.extend_from_slice(
            view.as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("contiguity check failed"))?,
        );
    } else {
        for i in 0..r {
            for j in 0..c {
                data.push(view[[i, j]]);
            }
        }
    }
    let n_threads_eff = if n_threads == 0 {
        rayon::current_num_threads()
    } else {
        n_threads as usize
    };
    let pool = global_pool(n_threads_eff).map_err(PyRuntimeError::new_err)?;
    py.allow_threads(|| {
        pool.install(|| {
            data.par_iter_mut().for_each(|v| *v = csv_parse_one(*v));
        });
    });
    Ok(data
        .into_pyarray_bound(py)
        .reshape([r, c])
        .unwrap()
        .unbind())
}

// ------------------------------------------------------------------
// numpy SeedSequence + PCG64 + Lemire bounded integers (exact ports)
// ------------------------------------------------------------------
const SS_INIT_A: u32 = 0x43b0_d7e5;
const SS_MULT_A: u32 = 0x931e_8875;
const SS_INIT_B: u32 = 0x8b51_f9dd;
const SS_MULT_B: u32 = 0x58f3_8ded;
const SS_MIX_MULT_L: u32 = 0xca01_f9dd;
const SS_MIX_MULT_R: u32 = 0x4973_f715;

#[inline]
fn ss_hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(SS_MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^= v >> 16;
    v
}

#[inline]
fn ss_mix(x: u32, y: u32) -> u32 {
    let mut r = SS_MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(SS_MIX_MULT_R.wrapping_mul(y));
    r ^= r >> 16;
    r
}

#[derive(Clone)]
struct SeedSequence {
    entropy_words: Vec<u32>, // little-endian base-2^32 digits of the entropy int
    spawn_key_words: Vec<u32>,
    pool: Vec<u32>, // pool_size = 4
}

impl SeedSequence {
    fn new_root(entropy: u64) -> Self {
        let entropy_words = vec![(entropy & 0xFFFF_FFFF) as u32, (entropy >> 32) as u32];
        let mut s = SeedSequence {
            entropy_words,
            spawn_key_words: vec![],
            pool: vec![0; 4],
        };
        s.remix();
        s
    }

    fn child(&self, i: u64) -> Self {
        let mut spawn_key = self.spawn_key_words.clone();
        spawn_key.push((i & 0xFFFF_FFFF) as u32);
        if i >> 32 != 0 {
            spawn_key.push((i >> 32) as u32);
        }
        let mut s = SeedSequence {
            entropy_words: self.entropy_words.clone(),
            spawn_key_words: spawn_key,
            pool: vec![0; 4],
        };
        s.remix();
        s
    }

    fn remix(&mut self) {
        // get_assembled_entropy
        let mut run = self.entropy_words.clone();
        if !self.spawn_key_words.is_empty() && run.len() < 4 {
            run.resize(4, 0);
        }
        run.extend_from_slice(&self.spawn_key_words);
        // mix_entropy
        let mut hc = SS_INIT_A;
        let ent = &run[..];
        for i in 0..4 {
            self.pool[i] = ss_hashmix(if i < ent.len() { ent[i] } else { 0 }, &mut hc);
        }
        for i_src in 0..4 {
            for i_dst in 0..4 {
                if i_src != i_dst {
                    let h = ss_hashmix(self.pool[i_src], &mut hc);
                    self.pool[i_dst] = ss_mix(self.pool[i_dst], h);
                }
            }
        }
        for i_src in 4..ent.len() {
            for i_dst in 0..4 {
                let h = ss_hashmix(ent[i_src], &mut hc);
                self.pool[i_dst] = ss_mix(self.pool[i_dst], h);
            }
        }
    }

    fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hc = SS_INIT_B;
        let mut out = Vec::with_capacity(n_words);
        for i in 0..n_words {
            let mut data_val = self.pool[i % 4];
            data_val ^= hc;
            hc = hc.wrapping_mul(SS_MULT_B);
            data_val = data_val.wrapping_mul(hc);
            data_val ^= data_val >> 16;
            out.push(data_val);
        }
        out
    }

    fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let w = self.generate_state_u32(2 * n_words);
        w.chunks(2)
            .map(|c| (c[0] as u64) | ((c[1] as u64) << 32))
            .collect()
    }
}

const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    fn from_seed_sequence(ss: &SeedSequence) -> Self {
        let words = ss.generate_state_u64(4);
        // pcg64.c: s = (seed[0] << 64) | seed[1]; i = (inc[0] << 64) | inc[1]
        let initstate = ((words[0] as u128) << 64) | (words[1] as u128);
        let initseq = ((words[2] as u128) << 64) | (words[3] as u128);
        let mut r = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        r.step();
        r.state = r.state.wrapping_add(initstate);
        r.step();
        r
    }

    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULT)
            .wrapping_add(self.inc);
    }

    #[inline]
    fn next64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        let x = hi ^ lo;
        let rot = (hi >> 58) as u32;
        if rot == 0 {
            x
        } else {
            x.rotate_right(rot)
        }
    }

    #[inline]
    fn next32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            self.uinteger
        } else {
            let n = self.next64();
            self.has_uint32 = true;
            self.uinteger = (n >> 32) as u32;
            (n & 0xFFFF_FFFF) as u32
        }
    }

    // random_bounded_uint64(off=0, rng) for rng < 0xFFFFFFFF: numpy routes to
    // buffered_bounded_lemire_uint32 (32-bit Lemire on next_uint32 stream).
    fn bounded_below_2p32(&mut self, rng: u32) -> u32 {
        if rng == 0 {
            return 0;
        }
        if rng == 0xFFFF_FFFF {
            return self.next32();
        }
        let rng_excl = rng.wrapping_add(1);
        let mut m = (self.next32() as u64) * (rng_excl as u64);
        let mut leftover = (m & 0xFFFF_FFFF) as u32;
        if leftover < rng_excl {
            let threshold = (0xFFFF_FFFFu32 - rng) % rng_excl;
            while leftover < threshold {
                m = (self.next32() as u64) * (rng_excl as u64);
                leftover = (m & 0xFFFF_FFFF) as u32;
            }
        }
        (m >> 32) as u32
    }

    // Generator.integers(0, n_bound, size, dtype=int64) with 0 < n_bound < 2^31
    // (the cibersort permutation draw). Returns int64 values.
    fn integers_below(&mut self, n_bound: u64, size: usize, out: &mut Vec<i64>) {
        debug_assert!(n_bound > 0 && n_bound <= 0xFFFF_FFFF);
        let rng = (n_bound - 1) as u32; // closed upper bound
        out.clear();
        for _ in 0..size {
            out.push(self.bounded_below_2p32(rng) as i64);
        }
    }
}

// ------------------------------------------------------------------
// core CIBERSORT solve (core_alg from cibersort.py, bit-exact)
// ------------------------------------------------------------------
struct CoreFitScratch {
    dual: Vec<f64>,       // cap g
    sv_ind: Vec<i32>,     // cap g
    sv_mat: Vec<f64>,     // [nsv][c]
    w64: Vec<f64>,        // c
    w32: Vec<f32>,        // c
    w_use32: Vec<f32>,    // c
    w_use64: Vec<f64>,    // c
    k: Vec<f64>,          // g
    d: Vec<f64>,          // g
    d2: Vec<f64>,         // g
    corr_buf: Vec<f64>,
    y32z_buf: Vec<f64>,
}

impl CoreFitScratch {
    fn new(g: usize, c: usize) -> Self {
        CoreFitScratch {
            dual: vec![0.0; g],
            sv_ind: vec![0; g],
            sv_mat: vec![0.0; g * c],
            w64: vec![0.0; c],
            w32: vec![0.0; c],
            w_use32: vec![0.0; c],
            w_use64: vec![0.0; c],
            k: vec![0.0; g],
            d: vec![0.0; g],
            d2: vec![0.0; g],
            corr_buf: Vec::with_capacity(g),
            y32z_buf: Vec::new(),
        }
    }
}

struct CoreOut {
    w_use: Vec<f32>,  // c
    w_raw: Vec<f32>,  // c
    w_raw_sum: f32,
    rmse: f64,
    r: f64,
    n_iters: [i32; 3],
}

#[allow(clippy::too_many_arguments)]
fn core_alg(
    x_std: &[f64], // [g][c] row-major (what sklearn's check_array feeds libsvm)
    x_f: &[f64],   // same values, F-layout (what numpy's X @ w_use multiplies)
    y: &[f64],     // g
    g: usize,
    c: usize,
    s: &mut CoreFitScratch,
) -> Result<CoreOut, String> {
    let nu_values = [0.25f64, 0.5f64, 0.75f64];
    let mut best: Option<(Vec<f32>, Vec<f32>, f32, f64, f64)> = None;
    let mut best_rmse = f64::INFINITY;
    let mut best_corr = -2.0f64;
    let mut n_iters = [0i32; 3];

    for (i_nu, &nu) in nu_values.iter().enumerate() {
        let (nsv, n_iter) = unsafe {
            let mut n_iter_v: i32 = 0;
            let mut fit_status: i32 = 0;
            let nsv = iobrx_nusvr_linear_fit(
                x_std.as_ptr(),
                g as i32,
                c as i32,
                y.as_ptr(),
                nu,
                1.0,     // C
                1e-3,    // tol
                200.0,   // cache_size MB
                1,       // shrinking
                s.dual.as_mut_ptr(),
                s.sv_ind.as_mut_ptr(),
                &mut n_iter_v,
                &mut fit_status,
            );
            (nsv, n_iter_v)
        };
        if nsv < 0 {
            return Err(format!("libsvm fit failed with code {}", nsv));
        }
        n_iters[i_nu] = n_iter;

        // SV matrix (support vectors in libsvm order) and w = dual_coef @ SV
        for i in 0..nsv as usize {
            let src = s.sv_ind[i] as usize;
            for j in 0..c {
                s.sv_mat[i * c + j] = x_std[src * c + j];
            }
        }
        // numpy np.dot(coef, SV) == cblas_dgemv(RowMajor, Trans, nSV, C, SV, lda=C, coef)
        if !gemv(
            CBLAS_ROWMAJOR,
            CBLAS_TRANS,
            nsv as i64,
            c as i64,
            s.sv_mat.as_ptr(),
            c as i64,
            s.dual.as_ptr(),
            s.w64.as_mut_ptr(),
        ) {
            return Err("openblas gemv64 not initialized".into());
        }
        for j in 0..c {
            s.w32[j] = s.w64[j] as f32;
            if s.w32[j] < 0.0 {
                s.w32[j] = 0.0;
            }
        }
        let sum = np_sum_f32(&s.w32[..c]); // float32 pairwise sum, like w.sum()
        let (w_use, w_raw, s_val) = if sum <= 0.0 {
            // uniform fallback
            let uni = (1.0f64 / (c.max(1) as f64)) as f32;
            for j in 0..c {
                s.w_use32[j] = uni;
            }
            let w_use = s.w_use32[..c].to_vec();
            let s2 = np_sum_f32(&w_use);
            (w_use.clone(), w_use, s2)
        } else {
            for j in 0..c {
                s.w_use32[j] = s.w32[j] / sum;
            }
            (s.w_use32[..c].to_vec(), s.w32[..c].to_vec(), sum)
        };

        // k = X @ w_use: numpy multiplies the ORIGINAL F-order X, i.e.
        // cblas_dgemv(ColMajor, NoTrans, G, C, X_f, lda=G, w)
        for j in 0..c {
            s.w_use64[j] = w_use[j] as f64;
        }
        if !gemv(
            CBLAS_COLMAJOR,
            CBLAS_NOTRANS,
            g as i64,
            c as i64,
            x_f.as_ptr(),
            g as i64,
            s.w_use64.as_ptr(),
            s.k.as_mut_ptr(),
        ) {
            return Err("openblas gemv64 not initialized".into());
        }

        // rmse = sqrt(mean((k - y)**2))
        for i in 0..g {
            let dd = s.k[i] - y[i];
            s.d2[i] = dd * dd;
        }
        let rmse = (np_sum_f64(&s.d2[..g]) / (g as f64)).sqrt();

        // r = corr_pearson_fast(k, y)
        let r = corr_pearson_fast(&s.k[..g], y, &mut s.corr_buf);

        if (rmse < best_rmse) || (rmse == best_rmse && r > best_corr) {
            best_rmse = rmse;
            best_corr = r;
            best = Some((w_use, w_raw, s_val, rmse, r));
        }
    }

    match best {
        Some((w_use, w_raw, s_val, rmse, r)) => Ok(CoreOut {
            w_use,
            w_raw,
            w_raw_sum: s_val,
            rmse,
            r,
            n_iters,
        }),
        None => Err("no nu fit succeeded".into()),
    }
}

// median of an f64 slice (numpy np.median semantics: mean of two middle for even)
fn np_median(v: &mut Vec<f64>) -> f64 {
    let n = v.len();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        // np.median computes np.mean of the two middle values
        let a = v[n / 2 - 1];
        let b = v[n / 2];
        (a + b) / 2.0
    }
}

// ------------------------------------------------------------------
// Python-facing functions
// ------------------------------------------------------------------

/// Raw linear NuSVR fit mirroring sklearn's parameter setup exactly.
/// X: (G, C) float64 row-major; y: (G,) float64.
/// Returns (dual_coef [nSV] f64, support_indices [nSV] i64, n_iter i32).
#[pyfunction]
#[pyo3(name = "fit_nusvr_linear", signature = (x, y, nu, _c=1.0, tol=1e-3, cache_size_mb=200.0, shrinking=true))]
fn fit_nusvr_linear_py(
    py: Python,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    nu: f64,
    _c: f64,
    tol: f64,
    cache_size_mb: f64,
    shrinking: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<i64>>, i32)> {
    let xs = x.as_slice().map_err(|_| {
        PyRuntimeError::new_err("X must be C-contiguous float64 row-major")
    })?;
    // fail loud on F-contiguous/strided X: as_slice() can return the raw
    // memory of an F-order array which the row-major solver would silently
    // misinterpret (same hazard class as the fixed quantile_normalize_np)
    {
        let v = x.as_array();
        let degenerate = x.shape()[0] <= 1 || x.shape()[1] <= 1;
        if !v.is_standard_layout() && !degenerate {
            return Err(PyValueError::new_err(
                "X must be C-contiguous float64 row-major (got non-contiguous/F-order array)",
            ));
        }
    }
    let ys = y.as_slice().map_err(|_| PyRuntimeError::new_err("y must be contiguous"))?;
    let (g, c) = (x.shape()[0], x.shape()[1]);
    if ys.len() != g {
        return Err(PyRuntimeError::new_err("X and y length mismatch"));
    }
    if unsafe { iobrx_svm_ready() } == 0 {
        return Err(PyRuntimeError::new_err(
            "openblas ddot not initialized; call init_blas() first",
        ));
    }
    let mut dual: Vec<f64> = vec![0.0; g];
    let mut sv_ind: Vec<i32> = vec![0; g];
    let mut n_iter: i32 = 0;
    let mut fit_status: i32 = 0;
    let nsv = unsafe {
        iobrx_nusvr_linear_fit(
            xs.as_ptr(),
            g as i32,
            c as i32,
            ys.as_ptr(),
            nu,
            _c,
            tol,
            cache_size_mb,
            if shrinking { 1 } else { 0 },
            dual.as_mut_ptr(),
            sv_ind.as_mut_ptr(),
            &mut n_iter,
            &mut fit_status,
        )
    };
    if nsv < 0 {
        return Err(PyRuntimeError::new_err(format!(
            "libsvm fit failed (code {}, fit_status {})",
            nsv, fit_status
        )));
    }
    dual.truncate(nsv as usize);
    sv_ind.truncate(nsv as usize);
    let sv_i64: Vec<i64> = sv_ind.iter().map(|&v| v as i64).collect();
    Ok((
        dual.into_pyarray_bound(py).unbind(),
        sv_i64.into_pyarray_bound(py).unbind(),
        n_iter,
    ))
}

/// np.argsort(x) parity helper: returns the exact permutation numpy produces.
#[pyfunction]
#[pyo3(name = "argsort_f64_np")]
fn argsort_f64_np_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<u64>>> {
    let input = x.as_array().to_owned().into_pyarray_bound(py);
    py.import_bound("iobrx._sorting")?.getattr("argsort")?
        .call1((input,))?.extract::<Py<PyArray1<u64>>>()
}

/// quantile_normalize_fast parity helper. Layout-correct: the values are
/// gathered through the ndarray view honoring strides, so an F-contiguous
/// input (e.g. the direct result of DataFrame.to_numpy()) is interpreted
/// correctly instead of having its column-major memory silently misread as
/// row-major (the previous as_slice() behavior — values were garbage on
/// F-order input while C-order input was bit-exact).
#[pyfunction]
#[pyo3(name = "quantile_normalize_np")]
fn quantile_normalize_np_py(
    py: Python,
    y: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let (g, n) = (y.shape()[0], y.shape()[1]);
    let view = y.as_array();
    let mut data: Vec<f64> = Vec::with_capacity(g * n);
    if view.is_standard_layout() {
        // fast path: identical bytes to the previous as_slice() read
        data.extend_from_slice(
            view.as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("contiguity check failed"))?,
        );
    } else {
        // honor strides (F-order or any non-standard layout)
        for i in 0..g {
            for j in 0..n {
                data.push(view[[i, j]]);
            }
        }
    }
    quantile_normalize(py, &mut data, g, n)?;
    Ok(data
        .into_pyarray_bound(py)
        .reshape([g, n])
        .unwrap()
        .unbind())
}

/// default_rng(seed).integers(0, n_bound, size, dtype=int64) parity helper.
#[pyfunction]
#[pyo3(name = "rng_integers_np")]
fn rng_integers_np_py(
    py: Python,
    seed: u64,
    n_bound: u64,
    size: usize,
) -> PyResult<Py<PyArray1<i64>>> {
    let ss = SeedSequence::new_root(seed);
    let mut rng = Pcg64::from_seed_sequence(&ss);
    let mut out = Vec::with_capacity(size);
    rng.integers_below(n_bound, size, &mut out);
    Ok(out.into_pyarray_bound(py).unbind())
}

/// SeedSequence(seed).spawn(n)[i].generate_state(1)[0] parity helper.
#[pyfunction]
#[pyo3(name = "seedsequence_spawn_seeds")]
fn seedsequence_spawn_seeds_py(py: Python, seed: u64, n: usize) -> PyResult<Py<PyArray1<u32>>> {
    let root = SeedSequence::new_root(seed);
    let mut seeds = Vec::with_capacity(n);
    for i in 0..n {
        let child = root.child(i as u64);
        let st = child.generate_state_u32(1);
        seeds.push(st[0]);
    }
    Ok(seeds.into_pyarray_bound(py).unbind())
}

/// Full CIBERSORT core over an already-aligned problem.
///
/// mix_y: FULL mixture matrix values (G_total x N, row-major f64) exactly as
///        loaded from the sorted/unique-indexed DataFrame (before exp2/QN).
/// common_mask: bool [G_total] (mix index isin sig index).
/// sig_x: f64 [G_common x C] aligned to the masked rows (raw sig values).
///
/// Replicates cibersort() steps 3-10 exactly: exp2 heuristic on full Y,
/// optional quantile normalization on full Y, common-gene restriction,
/// scalar X standardization, per-sample z-scores, nu loop solves,
/// permutations (seeded PCG64 - the original uses unseeded OS entropy),
/// p = count(null_r >= r) / perm.
#[pyfunction]
#[pyo3(
    name = "cibersort_core",
    signature = (mix_y, common_mask, sig_x, perm=100u32, use_qn=true, n_threads=0u32, seed=0u64, absolute=false, exp2_done=false)
)]
#[allow(clippy::too_many_arguments)]
fn cibersort_core_py(
    py: Python,
    mix_y: PyReadonlyArray2<f64>,
    common_mask: PyReadonlyArray1<bool>,
    sig_x: PyReadonlyArray2<f64>,
    perm: u32,
    use_qn: bool,
    n_threads: u32,
    seed: u64,
    absolute: bool,
    exp2_done: bool,
) -> PyResult<Py<PyDict>> {
    if unsafe { iobrx_svm_ready() } == 0 {
        return Err(PyRuntimeError::new_err(
            "openblas ddot not initialized; call init_blas() first",
        ));
    }
    // Fail loud on non-C-contiguous input: PyReadonlyArray::as_slice can
    // hand back the raw memory of an F-contiguous array, which this
    // row-major code would silently misinterpret (garbage, not an error).
    // The Python shim always passes np.ascontiguousarray(...) buffers.
    for (name, arr) in [("mix_y", &mix_y), ("sig_x", &sig_x)] {
        let v = arr.as_array();
        let std_c = v.is_standard_layout();
        // degenerate dims (0 rows/cols or single row/col) are trivially fine
        let degenerate = arr.shape()[0] <= 1 || arr.shape()[1] <= 1;
        if !std_c && !degenerate {
            return Err(PyValueError::new_err(format!(
                "{} must be C-contiguous float64 row-major (got non-contiguous/F-order array)",
                name
            )));
        }
    }
    let mix = mix_y
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("mix_y must be C-contiguous f64"))?;
    let mask = common_mask
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("common_mask must be contiguous bool"))?;
    let sig = sig_x
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("sig_x must be C-contiguous f64"))?;
    let (gtot, n) = (mix_y.shape()[0], mix_y.shape()[1]);
    let gc: usize = mask.iter().filter(|&&b| b).count();
    let c = sig_x.shape()[1];
    if sig_x.shape()[0] != gc {
        return Err(PyRuntimeError::new_err(
            "sig_x rows must equal the number of True entries in common_mask",
        ));
    }
    if gc == 0 {
        return Err(PyRuntimeError::new_err("no overlapping genes"));
    }

    // NumPy sorting needs the GIL; all solver work below still releases it.
    // -- step 3: exp2 heuristic on FULL Y. When exp2_done=true the CALLER
    // already applied the original heuristic + numpy's vectorized exp2
    // (np.max(Y) < 50 -> np.exp2(Y, out=Y)); numpy's exp2 and Rust's
    // libm exp2 differ by 1 ulp on ~5% of inputs, and those last bits
    // are what the bit-exactness contract is built on, so the numpy
    // path is the only one the shim uses. The internal branch below is
    // kept only for direct cibersort_core callers that opt out.
    let mut y_full: Vec<f64> = mix.to_vec();
    let exp2_applied;
    if exp2_done {
        exp2_applied = true;
    } else {
        let mut ymax = f64::NEG_INFINITY;
        for &v in y_full.iter() {
            if v > ymax {
                ymax = v;
            }
        }
        exp2_applied = ymax < 50.0;
        if exp2_applied {
            for v in y_full.iter_mut() {
                *v = v.exp2();
            }
        }
    }
    // -- step 4: quantile normalization on FULL Y
    if use_qn {
        quantile_normalize(py, &mut y_full, gtot, n)?;
    }

    let out_dict = PyDict::new_bound(py);
    let r = py.allow_threads(move || -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f64>, Vec<f32>, Vec<i32>, bool, f64, f64, Vec<f64>, f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        // -- step 6: common restriction
        let mut yc: Vec<f64> = Vec::with_capacity(gc * n);
        let mut row_of_common: Vec<usize> = Vec::with_capacity(gc);
        for i in 0..gtot {
            if mask[i] {
                row_of_common.push(i);
                yc.extend_from_slice(&y_full[i * n..(i + 1) * n]);
            }
        }
        // NaN contract (mirrors sklearn's NuSVR.fit validation, which is where
        // the original raises): if any NaN survives into the common-restricted
        // mixture (a NaN gene row inside the signature intersection, or a NaN
        // created by QN when NaN counts differ across columns), the original
        // raises ValueError('Input y contains NaN.') on the first sample fit.
        // We raise the same ValueError here instead of panicking on the
        // NaN-poisoned nu selection downstream.
        if yc.iter().any(|v| v.is_nan()) {
            return Err("Input y contains NaN.".into());
        }
        // -- step 8: scalar standardization of X (mean/std over all entries, ddof=1).
        // The reference pipeline's X is F-contiguous (pandas .loc -> to_numpy), and
        // numpy reduces a full-array sum over FLAT MEMORY ORDER (column-stacked for
        // F-order) with the nditer reduce buffer: pairwise summation per 8192-element
        // chunk, chunk sums accumulated sequentially. Verified bitwise against
        // X.mean()/X.std(ddof=1) on both the official 531x22 and the degenerate
        // 12x22 LM22-restricted matrices (the previous per-column sequential
        // accumulation diverged by 1 ulp on the 12x22 case).
        let mut chunk: Vec<f64> = Vec::with_capacity(NP_REDUCE_CHUNK);
        let mut flat_sum = 0.0f64;
        for j in 0..c {
            for i in 0..gc {
                chunk.push(sig[i * c + j]);
                if chunk.len() == NP_REDUCE_CHUNK {
                    flat_sum += pairwise_sum_f64(&chunk);
                    chunk.clear();
                }
            }
        }
        if !chunk.is_empty() {
            flat_sum += pairwise_sum_f64(&chunk);
        }
        let x_mean = flat_sum / ((gc * c) as f64);
        let mut xd2_f: Vec<f64> = vec![0.0; gc * c]; // F-layout: [j*gc + i] = (i, j)
        chunk.clear();
        let mut flat_sq_sum = 0.0f64;
        for j in 0..c {
            for i in 0..gc {
                let d = sig[i * c + j] - x_mean;
                let dd = d * d;
                xd2_f[j * gc + i] = dd;
                chunk.push(dd);
                if chunk.len() == NP_REDUCE_CHUNK {
                    flat_sq_sum += pairwise_sum_f64(&chunk);
                    chunk.clear();
                }
            }
        }
        if !chunk.is_empty() {
            flat_sq_sum += pairwise_sum_f64(&chunk);
        }
        let x_std = (flat_sq_sum / ((gc * c - 1) as f64)).sqrt();
        if x_std == 0.0 {
            return Err("Signature matrix has zero variance.".into());
        }
        let x_std_vec: Vec<f64> = sig.iter().map(|&v| (v - x_mean) / x_std).collect();
        // F-layout copy of the standardized X for numpy-parity X @ w (dgemv ColMajor)
        let mut x_std_f: Vec<f64> = vec![0.0; gc * c];
        for j in 0..c {
            for i in 0..gc {
                x_std_f[j * gc + i] = x_std_vec[i * c + j];
            }
        }

        // -- step 9: per-sample z-scores (axis-0 sequential row accumulation)
        let mut col_mean = vec![0.0f64; n];
        np_sum_axis0(&yc, gc, n, &mut col_mean);
        for v in col_mean.iter_mut() {
            *v /= gc as f64;
        }
        let mut sq = vec![0.0f64; gc * n];
        for i in 0..gc {
            for j in 0..n {
                let d = yc[i * n + j] - col_mean[j];
                sq[i * n + j] = d * d;
            }
        }
        let mut col_var = vec![0.0f64; n];
        np_sum_axis0(&sq, gc, n, &mut col_var);
        let mut yz: Vec<f64> = vec![0.0; gc * n];
        for j in 0..n {
            let sd = (col_var[j] / ((gc - 1) as f64)).sqrt() + 1e-12;
            for i in 0..gc {
                yz[i * n + j] = (yc[i * n + j] - col_mean[j]) / sd;
            }
        }

        // -- Y_flat (float32 ravel of the COMMON y) for permutations
        let y_flat: Vec<f32> = yc.iter().map(|&v| v as f32).collect();

        // -- permutation seeds: SeedSequence(seed).spawn(perm)
        let perm_seeds: Vec<u32> = if perm > 0 {
            let root = SeedSequence::new_root(seed);
            (0..perm as u64).map(|i| root.child(i).generate_state_u32(1)[0]).collect()
        } else {
            vec![]
        };

        // -- run the 10 sample solves + perm solves at FIT-level granularity:
        // 110 tasks x 3 nu values = 330 independent libsvm fits (identical
        // computations and selection semantics as the sequential original;
        // scheduling does not affect any float result).
        // pass A (cheap): build every task's y vector
        let total = n + perm as usize;
        let x_ref = &x_std_vec;
        let x_f_ref = &x_std_f;
        let y_flat_ref = &y_flat;
        let perm_seeds_ref = &perm_seeds;

        let n_threads_eff = if n_threads == 0 {
            rayon::current_num_threads()
        } else {
            n_threads as usize
        };
        let pool = global_pool(n_threads_eff)?;

        let ys: Vec<Vec<f64>> = pool.install(|| {
            (0..total)
                .into_par_iter()
                .map(|task| {
                    if task < n {
                        (0..gc).map(|i| yz[i * n + task]).collect()
                    } else {
                        let seed = perm_seeds_ref[task - n] as u64;
                        let ss = SeedSequence::new_root(seed);
                        let mut rng = Pcg64::from_seed_sequence(&ss);
                        let mut idx: Vec<i64> = Vec::with_capacity(gc);
                        rng.integers_below((gc * n) as u64, gc, &mut idx);
                        let mut vals32: Vec<f64> = Vec::with_capacity(gc);
                        for &ix in idx.iter() {
                            vals32.push(y_flat_ref[ix as usize] as f64);
                        }
                        let mut scratch = Vec::new();
                        zscore1d(&vals32, &mut scratch)
                    }
                })
                .collect()
        });

        const NU_VALUES: [f64; 3] = [0.25, 0.5, 0.75];

        // pass B: 330 independent (task, nu) fits + downstream stats
        struct FitOut {
            w_use: Vec<f32>,
            w_raw: Vec<f32>,
            s: f32,
            rmse: f64,
            r: f64,
            n_iter: i32,
        }
        let fit_results: Result<Vec<Option<FitOut>>, String> = pool.install(|| {
            (0..total * 3)
                .into_par_iter()
                .map(|idx| {
                    let task = idx / 3;
                    let nu = NU_VALUES[idx % 3];
                    let y = &ys[task];
                    let mut s = CoreFitScratch::new(gc, c);
                    // ---- single nu fit (inner body of core_alg) ----
                    let mut n_iter_v: i32 = 0;
                    let nsv = unsafe {
                        let mut fit_status: i32 = 0;
                        let nsv = iobrx_nusvr_linear_fit(
                            x_ref.as_ptr(),
                            gc as i32,
                            c as i32,
                            y.as_ptr(),
                            nu,
                            1.0,
                            1e-3,
                            200.0,
                            1,
                            s.dual.as_mut_ptr(),
                            s.sv_ind.as_mut_ptr(),
                            &mut n_iter_v,
                            &mut fit_status,
                        );
                        if nsv < 0 {
                            return Err(format!("libsvm fit failed with code {}", nsv));
                        }
                        nsv
                    };
                    for i in 0..nsv as usize {
                        let src = s.sv_ind[i] as usize;
                        for j in 0..c {
                            s.sv_mat[i * c + j] = x_ref[src * c + j];
                        }
                    }
                    if !gemv(
                        CBLAS_ROWMAJOR,
                        CBLAS_TRANS,
                        nsv as i64,
                        c as i64,
                        s.sv_mat.as_ptr(),
                        c as i64,
                        s.dual.as_ptr(),
                        s.w64.as_mut_ptr(),
                    ) {
                        return Err("openblas gemv64 not initialized".into());
                    }
                    for j in 0..c {
                        s.w32[j] = s.w64[j] as f32;
                        if s.w32[j] < 0.0 {
                            s.w32[j] = 0.0;
                        }
                    }
                    let sum = np_sum_f32(&s.w32[..c]);
                    let (w_use, w_raw, s_val) = if sum <= 0.0 {
                        let uni = (1.0f64 / (c.max(1) as f64)) as f32;
                        for j in 0..c {
                            s.w_use32[j] = uni;
                        }
                        let w_use = s.w_use32[..c].to_vec();
                        let s2 = np_sum_f32(&w_use);
                        (w_use.clone(), w_use, s2)
                    } else {
                        for j in 0..c {
                            s.w_use32[j] = s.w32[j] / sum;
                        }
                        (s.w_use32[..c].to_vec(), s.w32[..c].to_vec(), sum)
                    };
                    for j in 0..c {
                        s.w_use64[j] = w_use[j] as f64;
                    }
                    if !gemv(
                        CBLAS_COLMAJOR,
                        CBLAS_NOTRANS,
                        gc as i64,
                        c as i64,
                        x_f_ref.as_ptr(),
                        gc as i64,
                        s.w_use64.as_ptr(),
                        s.k.as_mut_ptr(),
                    ) {
                        return Err("openblas gemv64 not initialized".into());
                    }
                    for i in 0..gc {
                        let dd = s.k[i] - y[i];
                        s.d2[i] = dd * dd;
                    }
                    let rmse = (np_sum_f64(&s.d2[..gc]) / (gc as f64)).sqrt();
                    let r = corr_pearson_fast(&s.k[..gc], y, &mut s.corr_buf);
                    Ok(Some(FitOut {
                        w_use,
                        w_raw,
                        s: s_val,
                        rmse,
                        r,
                        n_iter: n_iter_v,
                    }))
                })
                .collect()
        });
        let fit_outs = fit_results?;
        if fit_outs.len() != total * 3 {
            return Err("one or more libsvm fits failed".into());
        }

        // sequential per-task selection (identical comparison rule to core_alg)
        let mut weights: Vec<f32> = Vec::with_capacity(n * c);
        let mut w_raw_all: Vec<f32> = Vec::with_capacity(n * c);
        let mut w_raw_sums: Vec<f32> = Vec::with_capacity(n);
        let mut rs: Vec<f32> = Vec::with_capacity(n);
        let mut rmses: Vec<f32> = Vec::with_capacity(n);
        let mut null_r: Vec<f32> = Vec::with_capacity(perm as usize);
        for task in 0..total {
            let mut best: Option<(Vec<f32>, Vec<f32>, f32, f64, f64)> = None;
            let mut best_rmse = f64::INFINITY;
            let mut best_corr = -2.0f64;
            for k in 0..3 {
                let fo = fit_outs[task * 3 + k].as_ref().unwrap();
                if (fo.rmse < best_rmse) || (fo.rmse == best_rmse && fo.r > best_corr) {
                    best_rmse = fo.rmse;
                    best_corr = fo.r;
                    best = Some((
                        fo.w_use.clone(),
                        fo.w_raw.clone(),
                        fo.s,
                        fo.rmse,
                        fo.r,
                    ));
                }
            }
            let (w_use, w_raw, s_val, rmse, r) = best.ok_or_else(|| {
                // unreachable for finite inputs (first nu always wins against
                // best_rmse = +inf); NaN y is caught by the explicit check above
                "no nu candidate was selectable (non-finite mixture values)".to_string()
            })?;
            if task < n {
                weights.extend_from_slice(&w_use);
                w_raw_all.extend_from_slice(&w_raw);
                w_raw_sums.push(s_val);
                rs.push(r as f32);
                rmses.push(rmse as f32);
            } else {
                null_r.push(r as f32);
            }
        }
        let mut iters: Vec<i32> = Vec::with_capacity(total * 3);
        for fo in fit_outs.iter() {
            iters.push(fo.as_ref().unwrap().n_iter);
        }

        // p-value: count(nulldist >= r) / perm (formula identical to original)
        let mut pvals: Vec<f64> = vec![9999.0; n];
        if perm > 0 {
            null_r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for i in 0..n {
                let mut cnt = 0u32;
                for &nv in null_r.iter() {
                    if nv >= rs[i] {
                        cnt += 1;
                    }
                }
                pvals[i] = cnt as f64 / perm as f64;
            }
        }

        // medians for absolute mode (computed only when requested)
        let (y_col_medians, y_median_full) = if absolute {
            let mut meds = Vec::with_capacity(n);
            for j in 0..n {
                let mut col: Vec<f64> = (0..gc).map(|i| yc[i * n + j]).collect();
                meds.push(np_median(&mut col));
            }
            let mut all: Vec<f64> = y_full.clone();
            let m = np_median(&mut all);
            (meds, m.max(1.0))
        } else {
            (vec![], 0.0)
        };

        // stash into a tuple; converted to Python objects back on the GIL thread
        Ok((
            weights,
            w_raw_all,
            w_raw_sums,
            rs,
            rmses,
            pvals,
            null_r,
            iters,
            exp2_applied,
            x_mean,
            x_std,
            y_col_medians,
            y_median_full,
            x_std_vec,
            yz,
            yc,
            col_mean,
            col_var.clone(),
        ))
    });

    // wrap into dict (need py for array creation); the NaN contract error is a
    // ValueError (sklearn parity), everything else stays a RuntimeError
    let (weights, w_raw_all, w_raw_sums, rs, rmses, pvals, null_r, iters, exp2_applied, x_mean, x_std, y_col_medians, y_median_full, x_std_vec, yz, yc_dbg, col_mean_dbg, col_var_dbg) =
        match r {
            Err(msg) if msg == "Input y contains NaN." => {
                return Err(PyValueError::new_err(msg))
            }
            Err(msg) => return Err(PyRuntimeError::new_err(msg)),
            Ok(v) => v,
        };
    let weights_arr = weights.into_pyarray_bound(py);
    let weights_2d = weights_arr.reshape([n, c]).unwrap().clone();
    let w_raw_arr = w_raw_all.into_pyarray_bound(py);
    let w_raw_2d = w_raw_arr.reshape([n, c]).unwrap().clone();
    out_dict.set_item("weights", weights_2d)?;
    out_dict.set_item("w_raw", w_raw_2d)?;
    out_dict.set_item("w_raw_sums", w_raw_sums.into_pyarray_bound(py))?;
    out_dict.set_item("correlation", rs.into_pyarray_bound(py))?;
    out_dict.set_item("rmse", rmses.into_pyarray_bound(py))?;
    out_dict.set_item("pvalue", pvals.into_pyarray_bound(py))?;
    out_dict.set_item("null_r", null_r.into_pyarray_bound(py))?;
    out_dict.set_item("n_iter", iters.into_pyarray_bound(py))?;
    out_dict.set_item("exp2_applied", exp2_applied)?;
    out_dict.set_item("x_mean", x_mean)?;
    out_dict.set_item("x_std", x_std)?;
    out_dict.set_item("y_col_medians", y_col_medians.into_pyarray_bound(py))?;
    out_dict.set_item("y_median_full", y_median_full)?;
    out_dict.set_item("x_std_vec", x_std_vec.into_pyarray_bound(py))?;
    out_dict.set_item("yz", yz.into_pyarray_bound(py).reshape([gc, n]).unwrap().clone())?;
    out_dict.set_item("y_common", yc_dbg.into_pyarray_bound(py).reshape([gc, n]).unwrap().clone())?;
    out_dict.set_item("col_mean", col_mean_dbg)?;
    out_dict.set_item("col_var", col_var_dbg)?;
    Ok(out_dict.unbind())
}

// ------------------------------------------------------------------
// ssGSEA kernel — bit-exact port of gseapy 1.3.0 `ssgsea_rs` nperm=0 path
// (stats.rs ss_gsea): per-sample descending argsort (utils.rs argsort(false)
// = stable ascending sort then reverse), pandas rank(method="average") via
// sample_norm_method='rank' -> 10000*rank/G, |x|^weight weighting,
// algorithm.rs fast_random_walk_ss closed-form ES, and the global
// min-max NES normalization nes = es / (max_es - min_es).
// ------------------------------------------------------------------

// sample_norm_method enum: 0=None/'custom', 1='rank', 2='log_rank', 3='log'
const SSGSEA_NORM_CUSTOM: u8 = 0;
const SSGSEA_NORM_RANK: u8 = 1;
const SSGSEA_NORM_LOG_RANK: u8 = 2;
// correl_norm_type enum (gseapy CorrelType): 0=Rank, 1=SymRank, 2=ZScore

// gseapy utils.rs `Statistic::stat(0)` on the sorted descending values:
// mean and population sd, sequential f64 sums exactly like iter().sum().
#[inline]
fn ssgsea_stat0(v: &[f64]) -> (f64, f64) {
    let n = v.len();
    let sum: f64 = {
        let mut a = 0.0f64;
        for &x in v.iter() {
            a += x;
        }
        a
    };
    let mean = sum / (n as f64);
    let var: f64 = {
        let mut a = 0.0f64;
        for &x in v.iter() {
            let d = mean - x;
            a += d * d;
        }
        a
    } / (n as f64);
    (mean, var.sqrt())
}

// One sample: descending order (stable-asc-then-reverse == gseapy
// utils.rs argsort(false)), tie-averaged 1-based ranks, sample_norm_method
// transform, correl_norm_type transform on the sorted array, then |x|^weight.
// Returns (pos_of_gene, weights aligned to the descending walk).
fn ssgsea_prepare_sample(
    col: &[f64],
    sample_norm: u8,
    correl_type: u8,
    weight: f64,
) -> (Vec<i32>, Vec<f64>) {
    let g = col.len();
    let mut order: Vec<u32> = (0..g as u32).collect();
    order.sort_by(|&a, &b| col[a as usize].partial_cmp(&col[b as usize]).unwrap());
    order.reverse(); // gseapy: if !ascending { sidx.reverse(); sval.reverse(); }
    let mut pos_of_gene = vec![0i32; g];
    for (p, &gi) in order.iter().enumerate() {
        pos_of_gene[gi as usize] = p as i32;
    }
    let gf = g as f64;
    let mut vals = vec![0.0f64; g]; // aligned with the descending walk
    match sample_norm {
        SSGSEA_NORM_RANK | SSGSEA_NORM_LOG_RANK => {
            // pandas dat.rank(axis=0, method="average", na_option="bottom"):
            // rank 1 = smallest value. A tie group spanning descending 0-based
            // positions [i, j) holds descending positions i..j-1, i.e. ascending
            // 1-based ranks G-i .. G-j+1, whose mean is G - (i + j - 1)/2.
            let mut i = 0usize;
            while i < g {
                let v = col[order[i] as usize];
                let mut j = i + 1;
                while j < g && col[order[j] as usize] == v {
                    j += 1;
                }
                let avg_rank = gf - ((i + j - 1) as f64) / 2.0;
                let scaled = (10000.0 * avg_rank) / gf; // np: 10000 * data / shape[0]
                let t = if sample_norm == SSGSEA_NORM_RANK {
                    scaled
                } else {
                    (scaled + std::f64::consts::E).ln() // np.log(x + np.exp(1))
                };
                for p in i..j {
                    vals[p] = t;
                }
                i = j;
            }
        }
        3 => {
            // 'log': dat[dat < 1] = 1; np.log(dat + np.exp(1))
            for (p, &gi) in order.iter().enumerate() {
                let mut v = col[gi as usize];
                if v < 1.0 {
                    v = 1.0;
                }
                vals[p] = (v + std::f64::consts::E).ln();
            }
        }
        _ => {
            // None / 'custom': use input values as rank metric
            for (p, &gi) in order.iter().enumerate() {
                vals[p] = col[gi as usize];
            }
        }
    }
    if weight > 0.0 {
        match correl_type {
            2 => {
                // CorrelType::ZScore: z-standardize the sorted values (stat(0))
                let (m, sd) = ssgsea_stat0(&vals);
                for v in vals.iter_mut() {
                    *v = (*v - m) / sd;
                }
            }
            // CorrelType::Rank: do nothing.
            // CorrelType::SymRank: gseapy 1.3.0's branch writes its result
            // through a discarded closure (`if *x > mid { *x } else { ... };`)
            // — an observable no-op; replicated as a no-op on purpose.
            _ => {}
        }
    }
    // gseapy: tmp.1.iter_mut().for_each(|x| *x = x.abs().powf(weight));
    // (weight == 0 gives all-ones weights, matching the classic unweighted ES)
    for v in vals.iter_mut() {
        *v = v.abs().powf(weight);
    }
    (pos_of_gene, vals)
}

// algorithm.rs fast_random_walk_ss — closed-form ssGSEA ES. Sums accumulate
// sequentially in ascending hit-position order, exactly like
// iter().map(...).sum::<f64>() folds starting from 0.0.
#[inline]
fn ssgsea_es(
    weights_desc: &[f64],
    pos_of_gene: &[i32],
    gene_idx: &[i32],
    n_genes: usize,
    scratch: &mut Vec<i32>,
) -> f64 {
    let nf = n_genes as f64;
    scratch.clear();
    scratch.extend(gene_idx.iter().map(|&gi| pos_of_gene[gi as usize]));
    scratch.sort_unstable(); // flatnonzero order of tag_new
    let k = scratch.len();
    let kf = k as f64; // gseapy: k = tag_indicator.iter().sum() (exact)
    let mut num = 0.0f64; // Σ r_i * (n - idx_i)
    let mut den = 0.0f64; // Σ r_i
    let mut nidx = 0.0f64; // Σ (n - idx_i)
    for &ix in scratch.iter() {
        let ifl = ix as f64;
        let w = weights_desc[ix as usize];
        let nmi = nf - ifl;
        num += w * nmi;
        den += w;
        nidx += nmi;
    }
    let step_cdf_in = num / den;
    let step_cdf_out = (nf * (nf + 1.0) / 2.0 - nidx) / (nf - kf);
    step_cdf_in - step_cdf_out
}

/// ssGSEA core replicating gseapy 1.3.0 `ssgsea_rs` (permutation_num=0 path
/// used by IOBRpy's calculate_sig_score(method='ssgsea')).
///
/// Parameters
/// ----------
/// expr : f64 array [G x N] (genes x samples), C-contiguous, finite values
///     (the output of gseapy's _check_data).
/// gene_sets : list of (name, gene_indices) with unique row indices into expr.
/// weight : gseapy weight (default 0.25).
/// min_size / max_size : gene-set size window (default 5 / 500); sets outside
///     are dropped exactly like GSEAResult::ss_gsea's `hit` check.
/// sample_norm : 0 custom, 1 'rank' (default), 2 'log_rank', 3 'log'.
/// correl_type : 0 Rank (default), 1 SymRank, 2 ZScore.
/// n_threads : rayon worker count (scoped pool; output is thread-count
///     invariant because every reduction is sequential per sample/set).
///
/// Returns dict: names (surviving sets, input order), es [n_sigs x N],
/// nes [N x n_sigs], es_min, es_max, norm (= es_max - es_min).
#[pyfunction]
#[pyo3(signature = (expr, gene_sets, weight=0.25, min_size=5, max_size=500, sample_norm=1u8, correl_type=0u8, n_threads=1))]
fn ssgsea_core(
    py: Python,
    expr: PyReadonlyArray2<f64>,
    gene_sets: Vec<(String, Vec<i32>)>,
    weight: f64,
    min_size: usize,
    max_size: usize,
    sample_norm: u8,
    correl_type: u8,
    n_threads: usize,
) -> PyResult<Py<PyDict>> {
    let view = expr.as_array();
    let (g, n) = {
        let d = view.dim();
        (d.0, d.1)
    };
    let flat = view
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("expr must be C-contiguous f64 [G x N]"))?;
    debug_assert_eq!(flat.len(), g * n);
    let pool = global_pool(n_threads.max(1)).map_err(PyRuntimeError::new_err)?;

    // per-sample ranking (parallel over samples); cols[j] = (pos_of_gene, weights_desc)
    let cols: Vec<(Vec<i32>, Vec<f64>)> = pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|j| {
                let mut col = vec![0.0f64; g];
                for (i, slot) in col.iter_mut().enumerate() {
                    *slot = flat[i * n + j];
                }
                ssgsea_prepare_sample(&col, sample_norm, correl_type, weight)
            })
            .collect()
    });

    // per-signature ES (parallel over signatures, sequential over samples)
    let kept: Vec<Option<(String, Vec<f64>)>> = pool.install(|| {
        gene_sets
            .into_par_iter()
            .map(|(name, gidx)| {
                let k = gidx.len();
                if k > max_size || k < min_size {
                    return None; // ss_gsea: hit > max_size || hit < min_size
                }
                let mut es = Vec::with_capacity(n);
                let mut scratch: Vec<i32> = Vec::with_capacity(k);
                for (pos_of_gene, weights) in cols.iter() {
                    es.push(ssgsea_es(weights, pos_of_gene, &gidx, g, &mut scratch));
                }
                Some((name, es))
            })
            .collect()
    });

    let mut names: Vec<String> = Vec::new();
    let mut es_rows: Vec<Vec<f64>> = Vec::new();
    for item in kept.into_iter().flatten() {
        names.push(item.0);
        es_rows.push(item.1);
    }
    let n_sigs = names.len();

    // ss_gsea tail: global min/max over all summaries, nes = es / (max - min).
    // Fold semantics (f64::max/f64::min skip NaN) replicated by comparisons.
    let mut mx = f64::NEG_INFINITY;
    let mut mn = f64::INFINITY;
    for row in es_rows.iter() {
        for &v in row.iter() {
            if v > mx {
                mx = v;
            }
            if v < mn {
                mn = v;
            }
        }
    }
    let norm = mx - mn;

    // es buffer [n_sigs x N]; nes buffer [N x n_sigs]
    let mut es_flat = vec![0.0f64; n_sigs * n];
    let mut nes_flat = vec![0.0f64; n * n_sigs];
    for (s, row) in es_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            es_flat[s * n + j] = v;
            nes_flat[j * n_sigs + s] = v / norm;
        }
    }

    let out = PyDict::new_bound(py);
    out.set_item("names", names)?;
    out.set_item(
        "es",
        es_flat
            .into_pyarray_bound(py)
            .reshape([n_sigs, n])
            .map_err(PyRuntimeError::new_err)?
            .clone(),
    )?;
    out.set_item(
        "nes",
        nes_flat
            .into_pyarray_bound(py)
            .reshape([n, n_sigs])
            .map_err(PyRuntimeError::new_err)?
            .clone(),
    )?;
    out.set_item("es_min", mn)?;
    out.set_item("es_max", mx)?;
    out.set_item("norm", norm)?;
    Ok(out.unbind())
}

// ------------------------------------------------------------------
// Rust PCA leg for calculate_sig_score(method='pca') — bit-exact port of
// sklearn PCA(n_components=1, svd_solver='full') on the pandas z-scored
// frame, calling dgesdd from the SAME dlopened scipy openblas the crate
// uses for ddot (init_blas), so the LAPACK binary path is identical.
//
// sklearn 1.7.2 call chain replicated exactly:
//   * mat = eset2.loc[valid].T (348 x g F-order); pandas z-score chain:
//       mean(axis=0)     : nanmean on the F-order block -> values.sum(axis=1)
//                          = SEQUENTIAL per-gene accumulation over samples
//                          (numpy reduce over the strided axis of an F-order
//                          array iterates columns vector-wise);
//       std(axis=0,ddof=1): nanvar COPIES values to C-order (mask path),
//                          so avg AND sqr-sum are numpy PAIRWISE per gene
//                          over samples; sqr = (avg - x) ** 2 as an array
//                          power (x*x semantics); var = pw/(n-1); sqrt.
//       div/fillna       : std==0.0 (exact) -> 0.0, else v1/std elementwise;
//   * PCA._fit_full: mean_ = np.mean(X, axis=0) on the F-ORDER z (public
//     .values of the div result is F-order) -> PAIRWISE per gene over
//     samples; X_centered = copy - mean_; scipy.linalg.svd(X_centered,
//     full_matrices=False) -> f2py dgesdd with jobz='S', lda=m, lwork from
//     the dgesdd_lwork query (identical to the lwork=-1 query - verified
//     for every signature shape on the official data);
//   * svd_flip(u_based_decision=False): sign from Vt row 0
//     (argmax |Vt[0,:]| first-max, np.sign);
//   * fit_transform: pc1 = (U[:,0] * sign) * S[0].
// mean_expr (tmp.mean(axis=0), pandas on the C-order (g, 348) frame) is
// PAIRWISE per sample over genes — verified bitwise for all 58 official
// signatures. The shim keeps np.corrcoef(pc1, mean_expr) + np.sign in
// Python: identical inputs -> identical bits.
// ------------------------------------------------------------------

// np.sign for f64 (+-0.0 -> 0.0, NaN -> NaN)
#[inline]
fn np_sign_f64(v: f64) -> f64 {
    if v.is_nan() {
        f64::NAN
    } else if v == 0.0 {
        0.0
    } else if v > 0.0 {
        1.0
    } else {
        -1.0
    }
}

// dgesdd(jobz='S', m, n, a, lda=m, s, u, ldu=m, vt, ldvt=min(m,n), work,
// lwork, iwork, info) with the exact lwork scipy.linalg.svd uses (the
// workspace query result). Returns (u, s, vt) col-major buffers.
fn dgesdd_jobz_s(
    f: FortranDgesdd,
    a: &mut [f64], // col-major m x m? no: m rows x n cols, len m*n
    m: usize,
    n: usize,
    s: &mut [f64],
    u: &mut [f64],
    vt: &mut [f64],
) -> Result<(), String> {
    let minmn = m.min(n);
    debug_assert!(a.len() == m * n && s.len() == minmn);
    debug_assert!(u.len() == m * minmn && vt.len() == minmn * n);
    let mut iwork: Vec<i32> = vec![0; 8 * minmn];
    let mut info: i32 = 0;
    let m32 = m as i32;
    let n32 = n as i32;
    let lda = m as i32;
    let ldu = m as i32;
    let ldvt = minmn as i32;
    // workspace query (identical to scipy's dgesdd_lwork for every shape
    // used by the official signature set)
    let mut work_query: Vec<f64> = vec![0.0; 8 * minmn + 2];
    let lw_query: i32 = -1;
    unsafe {
        f(
            b"S".as_ptr(),
            &m32,
            &n32,
            a.as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            vt.as_mut_ptr(),
            &ldvt,
            work_query.as_mut_ptr(),
            &lw_query,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        return Err(format!("dgesdd lwork query failed, info={}", info));
    }
    let lwork = work_query[0] as usize;
    let mut work: Vec<f64> = vec![0.0; lwork.max(1)];
    let lw = lwork as i32;
    info = 0;
    unsafe {
        f(
            b"S".as_ptr(),
            &m32,
            &n32,
            a.as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            vt.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lw,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        return Err(format!("dgesdd failed, info={}", info));
    }
    Ok(())
}

// shared computation over one signature's selected rows
struct PcaPrepared {
    z_f: Vec<f64>,   // (g, n) gene-major: z_f[k*n + i] = z-scored[i, k]
    cmean: Vec<f64>, // sklearn mean_ (pairwise per gene over samples)
}

fn pca_prepare(x: &[f64], g: usize, n: usize) -> PcaPrepared {
    debug_assert!(x.len() == g * n);
    // pandas mat.mean(axis=0): sequential per gene over samples
    let mut mean_seq = vec![0.0f64; g];
    for k in 0..g {
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += x[k * n + i];
        }
        mean_seq[k] = acc / (n as f64);
    }
    // v1 (gene-major)
    let mut v1_f = vec![0.0f64; g * n];
    for k in 0..g {
        for i in 0..n {
            v1_f[k * n + i] = x[k * n + i] - mean_seq[k];
        }
    }
    // pandas std(axis=0, ddof=1): nanvar on the C-order copy -> pairwise
    let mut avg_pw = vec![0.0f64; g];
    for k in 0..g {
        avg_pw[k] = pairwise_sum_f64(&v1_f[k * n..(k + 1) * n]) / (n as f64);
    }
    let mut sqr_f = vec![0.0f64; g * n];
    let mut std_pd = vec![0.0f64; g];
    for k in 0..g {
        let base = k * n;
        for i in 0..n {
            let d = avg_pw[k] - v1_f[base + i];
            sqr_f[base + i] = d * d;
        }
        std_pd[k] = (pairwise_sum_f64(&sqr_f[base..base + n]) / ((n - 1) as f64)).sqrt();
    }
    // div + fillna(0.0)
    let mut z_f = vec![0.0f64; g * n];
    for k in 0..g {
        let base = k * n;
        let sd = std_pd[k];
        if sd == 0.0 {
            for i in 0..n {
                z_f[base + i] = 0.0;
            }
        } else {
            for i in 0..n {
                z_f[base + i] = v1_f[base + i] / sd;
            }
        }
    }
    // sklearn mean_: pairwise per gene over samples (z public frame is F-order)
    let mut cmean = vec![0.0f64; g];
    for k in 0..g {
        cmean[k] = pairwise_sum_f64(&z_f[k * n..(k + 1) * n]) / (n as f64);
    }
    PcaPrepared { z_f, cmean }
}

/// Rust PCA leg: PC1 scores + per-sample mean of the selected rows, for one
/// signature. x: (g, n) row-major f64 (genes x samples, i.e.
/// eset2.loc[sorted(valid)].to_numpy()). Returns (pc1[n], mean_expr[n]).
/// Bit-exact vs PCA(n_components=1, svd_solver='full').fit_transform on the
/// pandas z-scored frame (verified for all 58 official signatures), provided
/// init_blas has been called with the scipy_dgesdd_ address.
#[pyfunction]
#[pyo3(name = "pca_pc1")]
fn pca_pc1_py(
    py: Python,
    x: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let f = get_dgesdd().ok_or_else(|| {
        PyRuntimeError::new_err("scipy dgesdd not initialized; call init_blas with its address first")
    })?;
    let xs = x
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("x must be C-contiguous f64 row-major (g, n)"))?;
    let (g, n) = (x.shape()[0], x.shape()[1]);
    if g < 2 || n < 1 {
        return Err(PyRuntimeError::new_err("pca_pc1 requires >= 2 genes and >= 1 sample"));
    }
    let minmn = g.min(n);
    let (pc1, mean_expr) = py.allow_threads(move || -> Result<(Vec<f64>, Vec<f64>), String> {
        let prep = pca_prepare(xs, g, n);
        // centered matrix, col-major (n samples x g genes): af[k*n + i]
        let mut af = vec![0.0f64; n * g];
        for k in 0..g {
            for i in 0..n {
                af[k * n + i] = prep.z_f[k * n + i] - prep.cmean[k];
            }
        }
        let mut s = vec![0.0f64; minmn];
        let mut u = vec![0.0f64; n * minmn];
        let mut vt = vec![0.0f64; minmn * g];
        dgesdd_jobz_s(f, &mut af, n, g, &mut s, &mut u, &mut vt)?;
        // svd_flip(u_based_decision=False): sign from Vt row 0 (first max)
        let mut am = 0usize;
        let mut best = f64::abs(vt[0]);
        for c in 1..g {
            let a = f64::abs(vt[c * minmn]);
            if a > best {
                best = a;
                am = c;
            }
        }
        let sign = np_sign_f64(vt[am * minmn]);
        // pc1 = (U[:,0] * sign) * S[0]   (U col 0 = u[i], col-major ldu=n)
        let s0 = s[0];
        let mut pc1 = vec![0.0f64; n];
        for i in 0..n {
            pc1[i] = (u[i] * sign) * s0;
        }
        // mean_expr: pandas tmp.mean(axis=0) -> pairwise per sample over genes
        let mut scratch = vec![0.0f64; g];
        let mut mean_expr = vec![0.0f64; n];
        for i in 0..n {
            for k in 0..g {
                scratch[k] = xs[k * n + i];
            }
            mean_expr[i] = pairwise_sum_f64(&scratch) / (g as f64);
        }
        Ok((pc1, mean_expr))
    })
    .map_err(PyRuntimeError::new_err)?;
    Ok((
        pc1.into_pyarray_bound(py).unbind(),
        mean_expr.into_pyarray_bound(py).unbind(),
    ))
}

/// The z-scored intermediate (n x g, C-order: z[i, j] = z-scored sample i,
/// gene j) — for gate verification against the pandas chain.
#[pyfunction]
#[pyo3(name = "pca_zscore_intermediate")]
fn pca_zscore_intermediate_py(py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let xs = x
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("x must be C-contiguous f64 row-major (g, n)"))?;
    let (g, n) = (x.shape()[0], x.shape()[1]);
    if g == 0 || n == 0 {
        return Err(PyRuntimeError::new_err("empty input"));
    }
    let out: Vec<f64> = py.allow_threads(move || {
        let prep = pca_prepare(xs, g, n);
        let mut rm = vec![0.0f64; n * g]; // (n, g) C-order
        for k in 0..g {
            for i in 0..n {
                rm[i * g + k] = prep.z_f[k * n + i];
            }
        }
        rm
    });
    Ok(out
        .into_pyarray_bound(py)
        .reshape([n, g])
        .map_err(PyRuntimeError::new_err)?
        .unbind())
}

/// Per-signature mean over selected rows (the zscore leg): out[i] =
/// pairwise sum over the g genes of column i, divided by g — bit-exact vs
/// pandas eset2.loc[valid].mean(axis=0) (verified for all 58 signatures).
#[pyfunction]
#[pyo3(name = "rows_colmean_pairwise")]
fn rows_colmean_pairwise_py(py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let xs = x
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("x must be C-contiguous f64 row-major (g, n)"))?;
    let (g, n) = (x.shape()[0], x.shape()[1]);
    if g == 0 || n == 0 {
        return Err(PyRuntimeError::new_err("empty input"));
    }
    let out: Vec<f64> = py.allow_threads(move || {
        let gf = g as f64;
        let mut scratch = vec![0.0f64; g];
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            for k in 0..g {
                scratch[k] = xs[k * n + i];
            }
            out[i] = pairwise_sum_f64(&scratch) / gf;
        }
        out
    });
    Ok(out.into_pyarray_bound(py).unbind())
}

// ------------------------------------------------------------------
// LR_cal: pandas-exact per-gene validity mask (feature_manipulation)
// ------------------------------------------------------------------
// Replicates the ORIGINAL per-gene pandas chain of
// iobrpy.workflow.LR_cal.feature_manipulation(data, is_matrix=True), which
// works on df.T (samples x genes), i.e. one row of the caller's
// genes x samples matrix per gene:
//   1. dropna(how='any')        -> gene row has no NaN
//   2. is_numeric_dtype(df[f])  -> guarded Python-side (fast path only for
//                                  all-float64/int64 frames; the ORIGINAL
//                                  per-column dtype check cannot differ
//                                  there because df.T unifies numeric
//                                  dtypes to float64)
//   3. np.isfinite(df[f]).all() -> gene row all finite
//   4. df[f].std() != 0         -> pandas nanvar on a NaN-free float64
//      Series is a TWO-PASS reduction with numpy PAIRWISE sums (bottleneck
//      is absent in the validated env; verified bit-identical against
//      pd.Series(x).std() on 4000 adversarial random vectors covering the
//      n<8, 8<=n<=128 and n>128 recursion regimes):
//        mean = np.sum(x) / n
//        ssq  = np.sum((x - mean) ** 2)
//        std  = sqrt(ssq / (n - 1))
//      computed here with the same np_sum_f64 (pairwise, 8192-chunked)
//      kernel cibersort's zscore1d already uses. NOTE: a vectorized
//      vals.sum(axis=1) is NOT bit-exact (numpy's inner-axis reduce is
//      sequential, ~20% of real rows differ in the last ulp); per-row
//      1-D pairwise is the exact pandas order.
// `filter_zero_var` mirrors the ORIGINAL `if df.shape[0] > 1` branch, whose
// shape[0] is the SAMPLE count of the transposed frame (zero-variance
// filtering is skipped for <=1 samples).
// The per-row computation is pure, so the output is thread-count invariant.
// Returns one u8 per gene (1 = passed all filters), original gene order.
#[pyfunction]
#[pyo3(name = "lr_gene_valid_mask")]
fn lr_gene_valid_mask_py(
    py: Python,
    x: PyReadonlyArray2<f64>,
    filter_zero_var: bool,
    n_threads: usize,
) -> PyResult<Py<PyArray1<u8>>> {
    let xs = x
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("x must be C-contiguous f64 row-major (genes, samples)"))?;
    let (g, s) = (x.shape()[0], x.shape()[1]);
    let mask: Vec<u8> = py
        .allow_threads(move || -> Result<Vec<u8>, String> {
            let pool = global_pool(n_threads.max(1))?;
            Ok(pool.install(|| {
                (0..g)
                    .into_par_iter()
                    .map(|i| {
                        let row = &xs[i * s..(i + 1) * s];
                        // 1. dropna(how='any')
                        if row.iter().any(|v| v.is_nan()) {
                            return 0u8;
                        }
                        // 3. np.isfinite(df[f]).all()
                        if row.iter().any(|v| v.is_infinite()) {
                            return 0u8;
                        }
                        // 4. df[f].std() != 0 (only when samples > 1)
                        if filter_zero_var && s > 1 {
                            let mean = np_sum_f64(row) / (s as f64);
                            let d2: Vec<f64> = row
                                .iter()
                                .map(|&v| {
                                    let d = v - mean;
                                    d * d
                                })
                                .collect();
                            let var = np_sum_f64(&d2) / ((s - 1) as f64);
                            if var.sqrt() == 0.0 {
                                return 0u8;
                            }
                        }
                        1u8
                    })
                    .collect()
            }))
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(mask.into_pyarray_bound(py).unbind())
}


// ------------------------------------------------------------------
// tme_cluster: Hartigan-Wong-style k-means core, bit-exact port of
// iobrpy/workflow/tme_cluster.py (iobrpy 0.2.0)
// ------------------------------------------------------------------
// Division of labour (see src/iobrx/_fast/tme_cluster_fast.py):
//   * ALL randomness stays in Python: the original draws its initial
//     centres with `np.random.RandomState(seed).choice(n, k,
//     replace=False)` inside a shared per-k stream (neighbor k's use
//     RandomState(seed + 9973*k)). The glue draws the nstart centre
//     matrices per k in the exact original call order and hands them in
//     as `inits` (nstart, k, p).
//   * This core is fully deterministic: initial nearest-centre
//     assignment, the point-by-point Hartigan move loop (traversal
//     order, strict `delta > tol`, first-best-j tie rule, incremental
//     sums/counts incl. the stale-residual centres of emptied clusters),
//     empty-cluster repair (reassign the farthest point; missing
//     clusters iterated in CPython `set(range(k)) - set(unique)` slot
//     order), max_iter=10 cap, and the within-cluster sum of squares.
//   * nstart selection keeps the FIRST strict minimum of withinss
//     (`w < best_w`), evaluated in start order.
//
// Floating-point order fidelity (numpy 2.2.6, verified empirically on
// the exact shapes of the frozen gates):
//   * flat / inner-axis `.sum()`  == pairwise summation chunked at 8192
//     elements with sequential accumulation across chunks (np_sum_f64);
//   * `.mean(axis=0)`             == per-column sequential row
//     accumulation, then / m;
//   * np.argmin / np.argmax       == first strict extremum; a[0] NaN ->
//     index 0, otherwise the first NaN wins;
//   * all elementwise ops are plain IEEE f64 (rustc does not contract
//     mul+add into FMA), matching numpy ufunc semantics bit-for-bit.
//
// `tme_kl_index` is the Krzanowski-Lai formula of the original,
// `num / (denom if denom > 0 else 1e-8)`; powf goes through the same
// system libm as CPython's float `**` (verified bit-equal on the gate
// W values).

const TME_SET_EMPTY: i64 = -1;
const TME_SET_MINSIZE: usize = 8;
const TME_SET_LINEAR_PROBES: usize = 9;
const TME_SET_PERTURB_SHIFT: u32 = 5;

/// Iteration order of CPython's `set(range(k)) - set(present)` for small
/// non-negative ints, given the missing values in ascending order.
///
/// tme_cluster.py's fix_empty() loops over that set, and CPython set
/// iteration is hash-slot order — NOT ascending once cluster ids >= 8
/// collide modulo the table size (e.g. missing {3, 9} iterates as
/// [9, 3]). set(range(k)) itself always iterates ascending (slot ==
/// value; the table is pre-sized from the length hint), so the
/// difference set is a fresh PySet_MINSIZE(8) table filled with the
/// missing ids in ascending order via set_add_entry's probe/growth
/// rules, then read in slot order. This mirrors CPython 3.11's
/// setobject.c exactly (set_add_entry linear-probe window, perturb
/// shifts, fill*5 >= mask*3 growth to the next power of two above
/// used*4, set_insert_clean rehash in old-slot order); verified against
/// the live interpreter for 47,824 cases (k = 2..14 exhaustive over all
/// missing subsets, k = 15..64 randomized, k up to 70,000 spot checks,
/// 0 mismatches). For every missing id < 8 it degenerates to ascending
/// order (also verified), which covers all realistic k.
fn tme_set_insert_clean(table: &mut [i64], mask: usize, v: i64) {
    let mut i = (v as usize) & mask;
    let mut perturb = v;
    loop {
        if table[i] == TME_SET_EMPTY {
            table[i] = v;
            return;
        }
        if i + TME_SET_LINEAR_PROBES <= mask {
            let mut slot = i;
            let mut placed = false;
            for _ in 0..TME_SET_LINEAR_PROBES {
                slot += 1;
                if table[slot] == TME_SET_EMPTY {
                    table[slot] = v;
                    placed = true;
                    break;
                }
            }
            if placed {
                return;
            }
        }
        perturb >>= TME_SET_PERTURB_SHIFT;
        i = (((i as i64) * 5 + 1 + perturb) as usize) & mask;
    }
}

fn tme_set_order(missing_asc: &[usize]) -> Vec<usize> {
    let mut size = TME_SET_MINSIZE;
    let mut table = vec![TME_SET_EMPTY; size];
    let mut fill = 0usize;
    for &vu in missing_asc {
        let v = vu as i64;
        // set_add_entry: probe for an empty slot (values are distinct).
        loop {
            let mask = size - 1;
            let mut i = (v as usize) & mask;
            let mut perturb = v;
            let mut placed = false;
            loop {
                let probes = if i + TME_SET_LINEAR_PROBES <= mask {
                    TME_SET_LINEAR_PROBES
                } else {
                    0
                };
                let mut slot = i;
                let mut p = probes;
                loop {
                    if table[slot] == TME_SET_EMPTY {
                        table[slot] = v;
                        placed = true;
                        break;
                    }
                    debug_assert!(table[slot] != v);
                    if p == 0 {
                        break;
                    }
                    p -= 1;
                    slot += 1;
                }
                if placed {
                    break;
                }
                perturb >>= TME_SET_PERTURB_SHIFT;
                i = (((i as i64) * 5 + 1 + perturb) as usize) & mask;
            }
            if placed {
                break;
            }
        }
        fill += 1;
        if fill * 5 >= (size - 1) * 3 {
            let minused = if fill > 50000 { fill * 2 } else { fill * 4 };
            let mut newsize = TME_SET_MINSIZE;
            while newsize <= minused {
                newsize <<= 1;
            }
            let mut newtable = vec![TME_SET_EMPTY; newsize];
            for s in 0..size {
                if table[s] != TME_SET_EMPTY {
                    tme_set_insert_clean(&mut newtable, newsize - 1, table[s]);
                }
            }
            table = newtable;
            size = newsize;
        }
    }
    table
        .into_iter()
        .filter(|&x| x != TME_SET_EMPTY)
        .map(|x| x as usize)
        .collect()
}

/// np.argmin over a 1-D f64 slice: first strict minimum; NaN semantics
/// as numpy (a[0] NaN -> 0; otherwise the first NaN wins).
fn tme_argmin_first(a: &[f64]) -> usize {
    let mut min = a[0];
    if min.is_nan() {
        return 0;
    }
    let mut idx = 0usize;
    for i in 1..a.len() {
        let v = a[i];
        if v < min {
            min = v;
            idx = i;
        } else if v.is_nan() {
            return i;
        }
    }
    idx
}

/// np.argmax counterpart of tme_argmin_first.
fn tme_argmax_first(a: &[f64]) -> usize {
    let mut max = a[0];
    if max.is_nan() {
        return 0;
    }
    let mut idx = 0usize;
    for i in 1..a.len() {
        let v = a[i];
        if v > max {
            max = v;
            idx = i;
        } else if v.is_nan() {
            return i;
        }
    }
    idx
}

/// fix_empty() of tme_cluster.py: for every missing cluster (in CPython
/// set iteration order), recompute each point's squared distance to its
/// CURRENT centre (`((data - centers[labels])**2).sum(axis=1)`, per-row
/// pairwise over p) and reassign the farthest point (np.argmax, first
/// maximum) to that cluster. centres/sums/counts are left stale exactly
/// as in the original.
fn tme_fix_empty(
    labels: &mut [i64],
    centers: &[f64],
    data: &[f64],
    n: usize,
    p: usize,
    k: usize,
    dists: &mut [f64],
    scratch: &mut [f64],
    present: &mut [bool],
) {
    for q in present.iter_mut() {
        *q = false;
    }
    for i in 0..n {
        present[labels[i] as usize] = true;
    }
    let missing_asc: Vec<usize> = (0..k).filter(|&j| !present[j]).collect();
    if missing_asc.is_empty() {
        return;
    }
    for j in tme_set_order(&missing_asc) {
        for i in 0..n {
            let c = &centers[(labels[i] as usize) * p..(labels[i] as usize + 1) * p];
            let x = &data[i * p..(i + 1) * p];
            for t in 0..p {
                let d = x[t] - c[t];
                scratch[t] = d * d;
            }
            dists[i] = np_sum_f64(&scratch[..p]);
        }
        let idx = tme_argmax_first(&dists[..n]);
        labels[idx] = j as i64;
    }
}

/// One hartigan_wong() run: `centers` arrives pre-drawn by the Python
/// RNG (`data[rng.choice(n, k, replace=False)]`) and is updated in
/// place; `labels` receives the final assignment.
fn tme_hartigan_wong(
    data: &[f64],
    n: usize,
    p: usize,
    k: usize,
    centers: &mut [f64],
    max_iter: i64,
    tol: f64,
    labels: &mut [i64],
) {
    let mut scratch = vec![0.0f64; p];
    let mut dist_row = vec![0.0f64; k];
    let mut dists = vec![0.0f64; n];
    let mut present = vec![false; k];

    // labels = np.argmin(((data[:, None] - centers[None, :])**2).sum(axis=2), axis=1)
    for i in 0..n {
        let x = &data[i * p..(i + 1) * p];
        for j in 0..k {
            let c = &centers[j * p..(j + 1) * p];
            for t in 0..p {
                let d = x[t] - c[t];
                scratch[t] = d * d;
            }
            dist_row[j] = np_sum_f64(&scratch[..p]);
        }
        labels[i] = tme_argmin_first(&dist_row[..k]) as i64;
    }
    tme_fix_empty(
        labels, centers, data, n, p, k, &mut dists, &mut scratch, &mut present,
    );

    let mut sums = vec![0.0f64; k * p];
    let mut counts = vec![0i64; k];
    let mut new_center = vec![0.0f64; p];

    for _ in 0..max_iter {
        // sums/counts are recomputed from scratch every outer iteration
        for s in sums.iter_mut() {
            *s = 0.0;
        }
        for c in counts.iter_mut() {
            *c = 0;
        }
        for i in 0..n {
            let l = labels[i] as usize;
            counts[l] += 1;
            let srow = &mut sums[l * p..(l + 1) * p];
            let x = &data[i * p..(i + 1) * p];
            for t in 0..p {
                srow[t] += x[t];
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                let cf = counts[j] as f64;
                for t in 0..p {
                    centers[j * p + t] = sums[j * p + t] / cf;
                }
            }
        }
        let mut moved = false;
        for i in 0..n {
            let lbl = labels[i] as usize;
            let x = &data[i * p..(i + 1) * p];
            for t in 0..p {
                let d = x[t] - centers[lbl * p + t];
                scratch[t] = d * d;
            }
            let curr_cost = np_sum_f64(&scratch[..p]);
            let mut best_delta = tol;
            let mut best_lbl = lbl;
            for j in 0..k {
                if j == lbl {
                    continue;
                }
                let new_cost;
                if counts[j] == 0 {
                    new_cost = 0.0; // original: `new_cost = 0`
                } else {
                    let cj = counts[j] as f64;
                    let dj = (counts[j] + 1) as f64;
                    for t in 0..p {
                        new_center[t] = (centers[j * p + t] * cj + x[t]) / dj;
                    }
                    for t in 0..p {
                        let d = x[t] - new_center[t];
                        scratch[t] = d * d;
                    }
                    new_cost = np_sum_f64(&scratch[..p]);
                }
                let delta = curr_cost - new_cost;
                if delta > best_delta {
                    best_delta = delta;
                    best_lbl = j;
                }
            }
            if best_lbl != lbl {
                moved = true;
                counts[lbl] -= 1;
                {
                    let s = &mut sums[lbl * p..(lbl + 1) * p];
                    for t in 0..p {
                        s[t] -= x[t];
                    }
                }
                counts[best_lbl] += 1;
                {
                    let s = &mut sums[best_lbl * p..(best_lbl + 1) * p];
                    for t in 0..p {
                        s[t] += x[t];
                    }
                }
                labels[i] = best_lbl as i64;
                if counts[lbl] > 0 {
                    let cf = counts[lbl] as f64;
                    for t in 0..p {
                        centers[lbl * p + t] = sums[lbl * p + t] / cf;
                    }
                }
                {
                    let cf = counts[best_lbl] as f64;
                    for t in 0..p {
                        centers[best_lbl * p + t] = sums[best_lbl * p + t] / cf;
                    }
                }
            }
        }
        tme_fix_empty(
            labels, centers, data, n, p, k, &mut dists, &mut scratch, &mut present,
        );
        if !moved {
            break;
        }
    }
}

/// compute_withinss() of tme_cluster.py: per present cluster (ascending,
/// as np.unique), centre = per-column sequential row accumulation / m
/// (`.mean(axis=0)`), contribution = chunked-pairwise flat sum of the
/// squared deviations in row-major order; contributions accumulate
/// sequentially from 0.0.
fn tme_withinss(data: &[f64], labels: &[i64], n: usize, p: usize, k: usize) -> f64 {
    let mut present = vec![false; k];
    for i in 0..n {
        present[labels[i] as usize] = true;
    }
    let mut w = 0.0f64;
    let mut colsum = vec![0.0f64; p];
    let mut sq: Vec<f64> = Vec::new();
    for lbl in 0..k {
        if !present[lbl] {
            continue;
        }
        for j in 0..p {
            colsum[j] = 0.0;
        }
        let mut m = 0usize;
        for i in 0..n {
            if labels[i] as usize == lbl {
                m += 1;
                let row = &data[i * p..(i + 1) * p];
                for j in 0..p {
                    colsum[j] += row[j];
                }
            }
        }
        let mf = m as f64;
        for j in 0..p {
            colsum[j] /= mf;
        }
        sq.clear();
        sq.reserve(m * p);
        for i in 0..n {
            if labels[i] as usize == lbl {
                let row = &data[i * p..(i + 1) * p];
                for j in 0..p {
                    let d = row[j] - colsum[j];
                    sq.push(d * d);
                }
            }
        }
        w += np_sum_f64(&sq);
    }
    w
}

/// best_hartigan_run() of tme_cluster.py over pre-drawn starts: run each
/// start (parallel across starts is safe — every run is a pure function
/// of its own centre matrix), then keep the FIRST strict minimum of
/// withinss in start order.
#[pyfunction]
#[pyo3(name = "tme_kmeans_best", signature = (x, inits, max_iter, tol, n_threads=1))]
fn tme_kmeans_best_py(
    py: Python,
    x: PyReadonlyArray2<f64>,
    inits: numpy::PyReadonlyArray3<f64>,
    max_iter: i64,
    tol: f64,
    n_threads: usize,
) -> PyResult<(Py<PyArray1<i64>>, f64)> {
    let xs = x.as_slice().map_err(|_| {
        PyRuntimeError::new_err("x must be C-contiguous f64 row-major (n, p)")
    })?;
    let ini = inits.as_slice().map_err(|_| {
        PyRuntimeError::new_err("inits must be C-contiguous f64 row-major (nstart, k, p)")
    })?;
    let (n, p) = (x.shape()[0], x.shape()[1]);
    let (s, k, pi) = (inits.shape()[0], inits.shape()[1], inits.shape()[2]);
    if pi != p {
        return Err(PyValueError::new_err(format!(
            "inits feature dim {} != x feature dim {}",
            pi, p
        )));
    }
    if s == 0 {
        return Err(PyValueError::new_err("inits must hold at least one start"));
    }
    if n == 0 {
        return Err(PyValueError::new_err("x must hold at least one sample"));
    }
    if k == 0 {
        // numpy: np.argmin over an empty axis raises exactly this
        return Err(PyValueError::new_err(
            "attempt to get argmin of an empty sequence",
        ));
    }
    let pool = if n_threads > 1 && s > 1 {
        Some(global_pool(n_threads).map_err(PyRuntimeError::new_err)?)
    } else {
        None
    };
    let (best_labels, best_w) = py.allow_threads(move || {
        let run_one = |t: usize| -> (Vec<i64>, f64) {
            let mut centers = ini[t * k * p..(t + 1) * k * p].to_vec();
            let mut labels = vec![0i64; n];
            tme_hartigan_wong(xs, n, p, k, &mut centers, max_iter, tol, &mut labels);
            let w = tme_withinss(xs, &labels, n, p, k);
            (labels, w)
        };
        let results: Vec<(Vec<i64>, f64)> = match &pool {
            Some(pl) => pl.install(|| (0..s).into_par_iter().map(run_one).collect()),
            None => (0..s).map(run_one).collect(),
        };
        let mut best = 0usize;
        for t in 1..s {
            if results[t].1 < results[best].1 {
                best = t;
            }
        }
        let (labels, w) = &results[best];
        (labels.clone(), *w)
    });
    Ok((best_labels.into_pyarray_bound(py).unbind(), best_w))
}

/// KL(k) of tme_cluster.py:
/// |(k-1)^(2/p)*W_prev - k^(2/p)*W_k| / |k^(2/p)*W_k - (k+1)^(2/p)*W_next|,
/// denominator <= 0 padded with 1e-8 (upstream's exact expression and
/// guard; powf resolves to the same libm pow CPython's `**` uses).
#[pyfunction]
#[pyo3(name = "tme_kl_index")]
fn tme_kl_index_py(k: i64, p: i64, w_prev: f64, w_k: f64, w_next: f64) -> f64 {
    let e = 2.0 / p as f64;
    let t1 = ((k - 1) as f64).powf(e) * w_prev;
    let t2 = (k as f64).powf(e) * w_k;
    let t3 = ((k + 1) as f64).powf(e) * w_next;
    let num = (t1 - t2).abs();
    let denom = (t2 - t3).abs();
    num / (if denom > 0.0 { denom } else { 1e-8 })
}

/// Test hook: CPython iteration order of `set(range(k)) - set(present)`
/// (see tme_set_order). Used by tests/test_parity_tme_cluster.py to pin
/// the empty-cluster repair order against the live interpreter.
#[pyfunction]
#[pyo3(name = "tme_missing_order")]
fn tme_missing_order_py(k: usize, present: Vec<usize>) -> Vec<usize> {
    let mut seen = vec![false; k];
    for j in present {
        if j < k {
            seen[j] = true;
        }
    }
    let missing_asc: Vec<usize> = (0..k).filter(|j| !seen[*j]).collect();
    tme_set_order(&missing_asc)
}

// ------------------------------------------------------------------
// merge_salmon: parallel quant.sf read/parse backend (engine='rust')
// ------------------------------------------------------------------
// Rust read path for the iobrx merge_salmon fast parser (Python reference:
// src/iobrx/_fast/merge_salmon_fast._parse_quant_fast + its merge-phase
// guards). Contract:
//
//   * ACCEPTED domain == a strict subset of what the Python fast path
//     proved bit-equivalent to the gold pd.read_csv (R3): every
//     equivalence guard of the Python parser is replicated here 1:1 (CR
//     bytes, >=2GiB, missing-final-newline tolerance, strict per-line tab
//     count, header checks, Name in column 0, per-column float-form LUT,
//     numeric-form / duplicate / non-utf8 Name guards, byte-identical
//     Name lists across files). Float tokens are additionally constrained
//     to a strict clean-float grammar in the PROVABLY-UNAMBIGUOUS numeric
//     domain (see salmon_float_token_ok): no whitespace, no inf/nan
//     spellings, no junk tails, <=15 mantissa digits, decimal scale
//     exponent in [-22, 22].
//   * Float conversion is this crate's verbatim pandas precise_xstrtod
//     port. Why that is gold-exact on the accepted domain — pandas 2.3.3
//     C-engine float64 conversion was re-pinned empirically for this
//     kernel (research/build_merge_salmon_rust/diag*.py):
//       - the default fast path is the precise_xstrtod family (double
//         rounding: can differ by 1 ulp from correctly-rounded for tokens
//         with >=16 significant digits or |decimal scale exponent| > 22,
//         e.g. "0.63e56");
//       - ONE token overflowing to +-inf anywhere in the column (e.g.
//         "1e400", "1.8e308") flips the WHOLE column to the
//         correctly-rounded slow path (underflow does NOT flip; file
//         size / row count are irrelevant; float_precision=
//         None/'high'/'legacy' all share the fast path, 'round_trip' is
//         always correctly rounded);
//       - inside this kernel's accepted domain (M <= 10^15-1 < 2^53 exact,
//         E10[|E|] exact for |E| <= 22, ONE correctly-rounded IEEE
//         multiply/divide) precise_xstrtod, the pandas slow path,
//         round_trip, CPython float(), glibc strtod and numpy
//         np.fromstring (the R3 Python parser) are PROVABLY bit-identical,
//         so kernel == gold no matter which pandas path the gold run took,
//         and kernel == python-fast engine on every accepted input.
//         Accepted values are <= 1e37 in magnitude, so an accepted file
//         can never itself contain a path-flipping overflow token.
//     Re-validated per build: frozen 8x60k fixture (960,000 tokens, 0
//     bitwise mismatches vs pd.read_csv and vs np.fromstring) + the R3
//     85k-token adversarial corpus filtered through the grammar (0
//     mismatches three-way).
//   * Anything outside the domain returns ok=false with the first tripped
//     guard as `reason`; the Python glue then re-runs the UNTOUCHED
//     sequential fast path (which itself declines to the original pandas
//     statements), so engine='rust' produces byte-identical output to
//     engine='python' on every input, and bug-compatibility (outer-join
//     unions, int64 dtype quirks, '#' header ValueError) is inherited
//     from the existing fallback chain, never re-implemented here.
//
// Parallelism: rayon over files for read+parse+per-file guards, then a
// sequential guard phase replicating the Python merge-phase order
// (reference file = paths[0] = first in output column order), then rayon
// over row bands for the (n_rows x n_files) row-major assembly. Results
// are thread-count invariant: each column is a pure function of one file
// and assembly indices are fixed (validated n_threads=1/4/8/16).

/// Strict clean-float token grammar restricted to the domain where every
/// converter in play is provably bit-identical (see block comment):
///
/// ```text
///   [+-]? ( D{1,15} [ . D* ]? | . D{1,15} )  ( [eE] [+-]? D+ )?
///   mdig = total mantissa digits            <= 15   (M < 2^53, exact)
///   E    = exponent - num_decimals          in [-22, 22]  (E10[|E|] exact,
///                                                single IEEE rounding)
/// ```
///
/// Every rejection is conservative: it only routes the WHOLE run to the
/// Python parser (np.fromstring), which handles the wider R3-validated
/// domain and declines further to the original pandas statements.
#[inline]
fn salmon_float_token_ok(s: &[u8]) -> bool {
    const MAX_MANT_DIGITS: u32 = 15;
    const MAX_SCALE_EXP: i32 = 22;
    let n = s.len();
    if n == 0 {
        return false;
    }
    let mut p = if s[0] == b'+' || s[0] == b'-' { 1 } else { 0 };
    let mut mdig: u32 = 0;
    while p < n && s[p].is_ascii_digit() {
        mdig += 1;
        p += 1;
    }
    let mut dfrac: i32 = 0;
    if p < n && s[p] == b'.' {
        p += 1;
        while p < n && s[p].is_ascii_digit() {
            mdig += 1;
            dfrac += 1;
            p += 1;
        }
    }
    if mdig == 0 || mdig > MAX_MANT_DIGITS {
        return false;
    }
    let mut e: i32 = 0;
    if p < n && (s[p] == b'e' || s[p] == b'E') {
        p += 1;
        let mut neg = false;
        if p < n && (s[p] == b'+' || s[p] == b'-') {
            neg = s[p] == b'-';
            p += 1;
        }
        let mut edig = 0u32;
        let mut ev: i32 = 0;
        while p < n && s[p].is_ascii_digit() {
            edig += 1;
            ev = ev.saturating_mul(10).saturating_add((s[p] - b'0') as i32);
            p += 1;
        }
        if edig == 0 {
            return false;
        }
        e = if neg { -ev } else { ev };
    }
    if p != n {
        return false;
    }
    let scale = e.saturating_sub(dfrac);
    scale >= -MAX_SCALE_EXP && scale <= MAX_SCALE_EXP
}

/// Python `_NAME_NUMERIC_LUT`: bytes that would make pandas infer a
/// NUMERIC dtype for the Name column (b"0123456789+-.eE" plus the b'\n'
/// separators of the joined check buffer).
#[inline]
fn salmon_name_numeric_byte(b: u8) -> bool {
    b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.' | b'e' | b'E' | b'\n')
}

/// Python `_FLOATMARK_LUT` = b".eEinNaAiIfF": a value column without any
/// of these bytes would be inferred int64 by read_csv -> decline.
const SALMON_FLOATMARK: &[u8] = b".eEinNaAiIfF";

struct SalmonOne {
    names: Vec<u8>, // Name tokens joined by b'\n' (byte-identical to the Python check buffer)
    n: usize,
    tpm: Vec<f64>,
    cnt: Vec<f64>,
}

/// Rust twin of `_parse_quant_fast` (per-file guards) — same accept
/// decisions on every input the Python fast path was validated on, and a
/// deliberately narrower domain beyond it (strict float grammar).
fn salmon_parse_one(path: &str) -> Result<SalmonOne, String> {
    let mut data = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    if data.contains(&b'\r') {
        return Err("CR byte present".into());
    }
    if !data.ends_with(b"\n") {
        data.push(b'\n'); // pandas tolerates a missing final newline
    }
    if data.len() >= 1usize << 31 {
        return Err("file too large for the fast path".into());
    }
    let h_end = match data.iter().position(|&b| b == b'\n') {
        Some(i) => i,
        None => return Err("no data rows".into()), // unreachable: '\n' was ensured
    };
    let header = match std::str::from_utf8(&data[..h_end]) {
        Ok(h) => h,
        Err(_) => return Err("non-utf8 header".into()),
    };
    let cols: Vec<&str> = header.split('\t').collect();
    let ncols = cols.len();
    if ncols < 3 {
        return Err("degenerate or duplicated header".into());
    }
    {
        let mut seen = std::collections::HashSet::with_capacity(ncols);
        if !cols.iter().all(|c| seen.insert(*c)) {
            return Err("degenerate or duplicated header".into());
        }
    }
    let i_name = cols.iter().position(|c| *c == "Name");
    let i_tpm = cols.iter().position(|c| *c == "TPM");
    let i_cnt = cols.iter().position(|c| *c == "NumReads");
    if i_name.is_none() || i_tpm.is_none() || i_cnt.is_none() {
        // upstream dies here with ValueError(Usecols do not match
        // columns) — the Python-side pandas fallback reproduces it
        return Err("usecol missing from header".into());
    }
    let (i_tpm, i_cnt) = (i_tpm.unwrap(), i_cnt.unwrap());
    if i_name.unwrap() != 0 {
        return Err("Name is not the first column".into());
    }

    let body = &data[h_end + 1..];
    // data ends with '\n', so the newline count == data-row count and the
    // split yields exactly n lines plus one empty tail piece
    let n = body.iter().filter(|&&b| b == b'\n').count();
    if n == 0 {
        return Err("no data rows".into());
    }
    let mut one = SalmonOne {
        names: Vec::with_capacity(data.len() / 2),
        n,
        tpm: Vec::with_capacity(n),
        cnt: Vec::with_capacity(n),
    };
    let mut tabs: Vec<usize> = Vec::with_capacity(ncols);
    let mut tpm_mark = false;
    let mut cnt_mark = false;
    for (li, line) in body.split(|&b| b == b'\n').take(n).enumerate() {
        tabs.clear();
        for (k, &b) in line.iter().enumerate() {
            if b == b'\t' {
                tabs.push(k);
            }
        }
        // every data line must carry exactly ncols-1 tabs, which also
        // rejects blank lines (pandas skips those, the fixed column
        // layout must not)
        if tabs.len() != ncols - 1 {
            return Err("ragged tab layout".into());
        }
        let name = &line[0..tabs[0]];
        let ts = if i_tpm == 0 { 0 } else { tabs[i_tpm - 1] + 1 };
        let te = if i_tpm == ncols - 1 { line.len() } else { tabs[i_tpm] };
        let cs = if i_cnt == 0 { 0 } else { tabs[i_cnt - 1] + 1 };
        let ce = if i_cnt == ncols - 1 { line.len() } else { tabs[i_cnt] };
        let tt = &line[ts..te];
        let ct = &line[cs..ce];
        if !salmon_float_token_ok(tt) {
            return Err("non-canonical TPM token".into());
        }
        if !salmon_float_token_ok(ct) {
            return Err("non-canonical NumReads token".into());
        }
        if !tpm_mark && tt.iter().any(|b| SALMON_FLOATMARK.contains(b)) {
            tpm_mark = true;
        }
        if !cnt_mark && ct.iter().any(|b| SALMON_FLOATMARK.contains(b)) {
            cnt_mark = true;
        }
        one.tpm.push(precise_xstrtod(tt));
        one.cnt.push(precise_xstrtod(ct));
        if li > 0 {
            one.names.push(b'\n');
        }
        one.names.extend_from_slice(name);
    }
    if !tpm_mark {
        // all-integer-form column: read_csv would infer int64 and to_csv
        // would re-print tokens without ".0" -> Python/pandas path
        return Err("integer-form TPM column".into());
    }
    if !cnt_mark {
        return Err("integer-form NumReads column".into());
    }
    Ok(one)
}

/// Parallel quant.sf parse + matrix assembly for iobrx.merge_salmon
/// (engine='rust'). `paths` are quant.sf files in OUTPUT COLUMN order
/// (sorted (sample, path)); n_threads=0 uses the rayon default pool,
/// otherwise the crate's pooled registry (same pools cibersort_core uses).
///
/// Returns (ok, reason, names, tpm, cnt):
///   ok=false -> `reason` names the first tripped equivalence guard and
///     the caller MUST fall back to the sequential Python parser (the
///     kernel never approximates: decline == whole-run fallback);
///   ok=true -> names = the reference file's Name tokens joined by b'\n'
///     (utf-8-validated; every file's list is byte-identical) as a uint8
///     array, tpm/cnt = (n_rows, n_files) float64 row-major matrices whose
///     values are bitwise equal to the gold pd.read_csv parse.
#[pyfunction]
#[pyo3(name = "merge_salmon_parse", signature = (paths, n_threads=0u32))]
fn merge_salmon_parse_py(
    py: Python,
    paths: Vec<String>,
    n_threads: u32,
) -> PyResult<(
    bool,
    String,
    Option<Py<PyArray1<u8>>>,
    Option<Py<PyArray2<f64>>>,
    Option<Py<PyArray2<f64>>>,
)> {
    if paths.is_empty() {
        return Ok((false, "no quant.sf files".into(), None, None, None));
    }
    let n_threads_eff = if n_threads == 0 {
        rayon::current_num_threads()
    } else {
        n_threads as usize
    };
    let pool = global_pool(n_threads_eff).map_err(PyRuntimeError::new_err)?;

    let assembled = py.allow_threads(
        || -> Result<(Vec<u8>, Vec<f64>, Vec<f64>, usize), String> {
            let results: Vec<Result<SalmonOne, String>> = pool.install(|| {
                paths
                    .par_iter()
                    .map(|p| salmon_parse_one(p))
                    .collect()
            });
            let mut okres: Vec<SalmonOne> = Vec::with_capacity(results.len());
            for r in results {
                okres.push(r?);
            }
            // ---- merge-phase guards, replicating the Python order ----
            let r0 = &okres[0];
            let n = r0.n;
            if r0.names.is_empty() {
                return Err("empty Name column".into());
            }
            if r0.names.iter().all(|&b| salmon_name_numeric_byte(b)) {
                return Err("numeric-form Name column".into());
            }
            {
                let mut seen = std::collections::HashSet::with_capacity(n * 2);
                for nm in r0.names.split(|&b| b == b'\n') {
                    if !seen.insert(nm) {
                        return Err("duplicate Name entries".into());
                    }
                }
            }
            if std::str::from_utf8(&r0.names).is_err() {
                return Err("non-utf8 Name column".into());
            }
            for j in 1..okres.len() {
                if okres[j].n != n || okres[j].names != r0.names {
                    // differing row sets/order: upstream outer-joins via
                    // concat -> Python/pandas path
                    return Err("row sets differ across samples".into());
                }
            }
            // ---- assemble (n x nf) row-major, parallel over row bands ----
            let nf = okres.len();
            let mut tpm_mat = vec![0.0f64; n * nf];
            let mut cnt_mat = vec![0.0f64; n * nf];
            let band = (n / (n_threads_eff * 4)).max(256);
            let chunk = band * nf;
            pool.install(|| {
                tpm_mat
                    .as_mut_slice()
                    .par_chunks_mut(chunk)
                    .zip(cnt_mat.as_mut_slice().par_chunks_mut(chunk))
                    .enumerate()
                    .for_each(|(ci, (tc, cc))| {
                        let row0 = ci * band;
                        let rows = tc.len() / nf;
                        for r in 0..rows {
                            let i = row0 + r;
                            let trow = &mut tc[r * nf..(r + 1) * nf];
                            let crow = &mut cc[r * nf..(r + 1) * nf];
                            for j in 0..nf {
                                trow[j] = okres[j].tpm[i];
                                crow[j] = okres[j].cnt[i];
                            }
                        }
                    });
            });
            let names = std::mem::take(&mut okres[0].names);
            Ok((names, tpm_mat, cnt_mat, n))
        },
    );

    let (names, tpm_mat, cnt_mat, n) = match assembled {
        Ok(v) => v,
        Err(reason) => return Ok((false, reason, None, None, None)),
    };
    let nf = paths.len();
    let names_arr = names.into_pyarray_bound(py).unbind();
    let tpm_arr = tpm_mat
        .into_pyarray_bound(py)
        .reshape([n, nf])
        .unwrap()
        .unbind();
    let cnt_arr = cnt_mat
        .into_pyarray_bound(py)
        .reshape([n, nf])
        .unwrap()
        .unbind();
    Ok((true, String::new(), Some(names_arr), Some(tpm_arr), Some(cnt_arr)))
}



// ==========================================================================
// BayesPrism native Gibbs kernel (backend='rust' lane)
// --------------------------------------------------------------------------
// Bit-exact port of the numpy 2.2.6 RNG chain that drives iobrpy
// bayesprism's Gibbs sampler (gold: IOBRpy src/iobrpy/bayesprism/gibbs.py):
//
//   SeedSequence(123).spawn(n)[i]  -> per-sample child streams (phase 1),
//   the phase-3 shared spawn(1)[0] quirk, Generator(MT19937(ss)) seeding
//   (generate_state(624) direct fill, key[0]=0x80000000, pos=623 quirk),
//   next_double = ((w1>>5)*2^26 + (w2>>6)) / 2^53, next64 = hi<<32|lo,
//   rng.multinomial = sequential conditional-binomial chain
//   (distributions.c random_multinomial), rng.binomial = inversion + BTPE
//   (incl. every goto branch and the (k/(nrq)) Stirling bounds), rng.gamma =
//   Marsaglia-Tsang over the ziggurat standard normal + the ziggurat
//   exponential (shape == 1.0 exactly), rdirichlet normalization with
//   numpy's contiguous pairwise summation, and prob_mat.sum(axis=0) with
//   numpy's SEQUENTIAL strided column accumulation.
//
// Verified bit-exact against numpy 2.2.6 gold streams (12.5M elements,
// 122/122 streams) and against gold gibbs.py sample_Z_theta_n /
// sample_theta_n chains (Z_n / theta_n / theta.cv_n), see
// research/build_bayesprism_rust/spike_report.json.
//
// Reuses the crate's existing SeedSequence (exact numpy port already merged
// for cibersort/tme; its generate_state_u32 cycles the 4-word pool, which
// numpy's `cycle(self.pool)` does for ANY n_words incl. 624 — verified),
// np_sum_f64/pairwise_sum_f64 (numpy contiguous reduction order), and the
// global_pool rayon registry (thread-count-independent outputs).
//
// The numpy Generator-level persistent binomial_t cache is value-transparent
// (cached fields are deterministic functions of (n,p)), so this port
// recomputes the setup per call: identical draws, identical RNG consumption.
//
// Math functions (log/exp/log1p/sqrt/floor/pow) resolve to the same glibc
// libm symbols numpy's scalar C code calls; the release target is baseline
// x86-64 (SSE2), so no FMA contraction can move the last bit.
// ==========================================================================

// Ziggurat tables + constants: VERBATIM literals transcribed from
// numpy 2.2.6 numpy/random/src/distributions/ziggurat_constants.h
// (double tables only; float tables are unused by the f64 code path).
const BP_KI_DOUBLE: [u64; 256] = [
    0x000EF33D8025EF6Au64, 0x0000000000000000u64, 0x000C08BE98FBC6A8u64,
    0x000DA354FABD8142u64, 0x000E51F67EC1EEEAu64, 0x000EB255E9D3F77Eu64,
    0x000EEF4B817ECAB9u64, 0x000F19470AFA44AAu64, 0x000F37ED61FFCB18u64,
    0x000F4F469561255Cu64, 0x000F61A5E41BA396u64, 0x000F707A755396A4u64,
    0x000F7CB2EC28449Au64, 0x000F86F10C6357D3u64, 0x000F8FA6578325DEu64,
    0x000F9724C74DD0DAu64, 0x000F9DA907DBF509u64, 0x000FA360F581FA74u64,
    0x000FA86FDE5B4BF8u64, 0x000FACF160D354DCu64, 0x000FB0FB6718B90Fu64,
    0x000FB49F8D5374C6u64, 0x000FB7EC2366FE77u64, 0x000FBAECE9A1E50Eu64,
    0x000FBDAB9D040BEDu64, 0x000FC03060FF6C57u64, 0x000FC2821037A248u64,
    0x000FC4A67AE25BD1u64, 0x000FC6A2977AEE31u64, 0x000FC87AA92896A4u64,
    0x000FCA325E4BDE85u64, 0x000FCBCCE902231Au64, 0x000FCD4D12F839C4u64,
    0x000FCEB54D8FEC99u64, 0x000FD007BF1DC930u64, 0x000FD1464DD6C4E6u64,
    0x000FD272A8E2F450u64, 0x000FD38E4FF0C91Eu64, 0x000FD49A9990B478u64,
    0x000FD598B8920F53u64, 0x000FD689C08E99ECu64, 0x000FD76EA9C8E832u64,
    0x000FD848547B08E8u64, 0x000FD9178BAD2C8Cu64, 0x000FD9DD07A7ADD2u64,
    0x000FDA9970105E8Cu64, 0x000FDB4D5DC02E20u64, 0x000FDBF95C5BFCD0u64,
    0x000FDC9DEBB99A7Du64, 0x000FDD3B8118729Du64, 0x000FDDD288342F90u64,
    0x000FDE6364369F64u64, 0x000FDEEE708D514Eu64, 0x000FDF7401A6B42Eu64,
    0x000FDFF46599ED40u64, 0x000FE06FE4BC24F2u64, 0x000FE0E6C225A258u64,
    0x000FE1593C28B84Cu64, 0x000FE1C78CBC3F99u64, 0x000FE231E9DB1CAAu64,
    0x000FE29885DA1B91u64, 0x000FE2FB8FB54186u64, 0x000FE35B33558D4Au64,
    0x000FE3B799D0002Au64, 0x000FE410E99EAD7Fu64, 0x000FE46746D47734u64,
    0x000FE4BAD34C095Cu64, 0x000FE50BAED29524u64, 0x000FE559F74EBC78u64,
    0x000FE5A5C8E41212u64, 0x000FE5EF3E138689u64, 0x000FE6366FD91078u64,
    0x000FE67B75C6D578u64, 0x000FE6BE661E11AAu64, 0x000FE6FF55E5F4F2u64,
    0x000FE73E5900A702u64, 0x000FE77B823E9E39u64, 0x000FE7B6E37070A2u64,
    0x000FE7F08D774243u64, 0x000FE8289053F08Cu64, 0x000FE85EFB35173Au64,
    0x000FE893DC840864u64, 0x000FE8C741F0CEBCu64, 0x000FE8F9387D4EF6u64,
    0x000FE929CC879B1Du64, 0x000FE95909D388EAu64, 0x000FE986FB939AA2u64,
    0x000FE9B3AC714866u64, 0x000FE9DF2694B6D5u64, 0x000FEA0973ABE67Cu64,
    0x000FEA329CF166A4u64, 0x000FEA5AAB32952Cu64, 0x000FEA81A6D5741Au64,
    0x000FEAA797DE1CF0u64, 0x000FEACC85F3D920u64, 0x000FEAF07865E63Cu64,
    0x000FEB13762FEC13u64, 0x000FEB3585FE2A4Au64, 0x000FEB56AE3162B4u64,
    0x000FEB76F4E284FAu64, 0x000FEB965FE62014u64, 0x000FEBB4F4CF9D7Cu64,
    0x000FEBD2B8F449D0u64, 0x000FEBEFB16E2E3Eu64, 0x000FEC0BE31EBDE8u64,
    0x000FEC2752B15A15u64, 0x000FEC42049DAFD3u64, 0x000FEC5BFD29F196u64,
    0x000FEC75406CEEF4u64, 0x000FEC8DD2500CB4u64, 0x000FECA5B6911F12u64,
    0x000FECBCF0C427FEu64, 0x000FECD38454FB15u64, 0x000FECE97488C8B3u64,
    0x000FECFEC47F91B7u64, 0x000FED1377358528u64, 0x000FED278F844903u64,
    0x000FED3B10242F4Cu64, 0x000FED4DFBAD586Eu64, 0x000FED605498C3DDu64,
    0x000FED721D414FE8u64, 0x000FED8357E4A982u64, 0x000FED9406A42CC8u64,
    0x000FEDA42B85B704u64, 0x000FEDB3C8746AB4u64, 0x000FEDC2DF416652u64,
    0x000FEDD171A46E52u64, 0x000FEDDF813C8AD3u64, 0x000FEDED0F909980u64,
    0x000FEDFA1E0FD414u64, 0x000FEE06AE124BC4u64, 0x000FEE12C0D95A06u64,
    0x000FEE1E579006E0u64, 0x000FEE29734B6524u64, 0x000FEE34150AE4BCu64,
    0x000FEE3E3DB89B3Cu64, 0x000FEE47EE2982F4u64, 0x000FEE51271DB086u64,
    0x000FEE59E9407F41u64, 0x000FEE623528B42Eu64, 0x000FEE6A0B5897F1u64,
    0x000FEE716C3E077Au64, 0x000FEE7858327B82u64, 0x000FEE7ECF7B06BAu64,
    0x000FEE84D2484AB2u64, 0x000FEE8A60B66343u64, 0x000FEE8F7ACCC851u64,
    0x000FEE94207E25DAu64, 0x000FEE9851A829EAu64, 0x000FEE9C0E13485Cu64,
    0x000FEE9F557273F4u64, 0x000FEEA22762CCAEu64, 0x000FEEA4836B42ACu64,
    0x000FEEA668FC2D71u64, 0x000FEEA7D76ED6FAu64, 0x000FEEA8CE04FA0Au64,
    0x000FEEA94BE8333Bu64, 0x000FEEA950296410u64, 0x000FEEA8D9C0075Eu64,
    0x000FEEA7E7897654u64, 0x000FEEA678481D24u64, 0x000FEEA48AA29E83u64,
    0x000FEEA21D22E4DAu64, 0x000FEE9F2E352024u64, 0x000FEE9BBC26AF2Eu64,
    0x000FEE97C524F2E4u64, 0x000FEE93473C0A3Au64, 0x000FEE8E40557516u64,
    0x000FEE88AE369C7Au64, 0x000FEE828E7F3DFDu64, 0x000FEE7BDEA7B888u64,
    0x000FEE749BFF37FFu64, 0x000FEE6CC3A9BD5Eu64, 0x000FEE64529E007Eu64,
    0x000FEE5B45A32888u64, 0x000FEE51994E57B6u64, 0x000FEE474A0006CFu64,
    0x000FEE3C53E12C50u64, 0x000FEE30B2E02AD8u64, 0x000FEE2462AD8205u64,
    0x000FEE175EB83C5Au64, 0x000FEE09A22A1447u64, 0x000FEDFB27E349CCu64,
    0x000FEDEBEA76216Cu64, 0x000FEDDBE422047Eu64, 0x000FEDCB0ECE39D3u64,
    0x000FEDB964042CF4u64, 0x000FEDA6DCE938C9u64, 0x000FED937237E98Du64,
    0x000FED7F1C38A836u64, 0x000FED69D2B9C02Bu64, 0x000FED538D06AE00u64,
    0x000FED3C41DEA422u64, 0x000FED23E76A2FD8u64, 0x000FED0A732FE644u64,
    0x000FECEFDA07FE34u64, 0x000FECD4100EB7B8u64, 0x000FECB708956EB4u64,
    0x000FEC98B61230C1u64, 0x000FEC790A0DA978u64, 0x000FEC57F50F31FEu64,
    0x000FEC356686C962u64, 0x000FEC114CB4B335u64, 0x000FEBEB948E6FD0u64,
    0x000FEBC429A0B692u64, 0x000FEB9AF5EE0CDCu64, 0x000FEB6FE1C98542u64,
    0x000FEB42D3AD1F9Eu64, 0x000FEB13B00B2D4Bu64, 0x000FEAE2591A02E9u64,
    0x000FEAAEAE992257u64, 0x000FEA788D8EE326u64, 0x000FEA3FCFFD73E5u64,
    0x000FEA044C8DD9F6u64, 0x000FE9C5D62F563Bu64, 0x000FE9843BA947A4u64,
    0x000FE93F471D4728u64, 0x000FE8F6BD76C5D6u64, 0x000FE8AA5DC4E8E6u64,
    0x000FE859E07AB1EAu64, 0x000FE804F690A940u64, 0x000FE7AB488233C0u64,
    0x000FE74C751F6AA5u64, 0x000FE6E8102AA202u64, 0x000FE67DA0B6ABD8u64,
    0x000FE60C9F38307Eu64, 0x000FE5947338F742u64, 0x000FE51470977280u64,
    0x000FE48BD436F458u64, 0x000FE3F9BFFD1E37u64, 0x000FE35D35EEB19Cu64,
    0x000FE2B5122FE4FEu64, 0x000FE20003995557u64, 0x000FE13C82788314u64,
    0x000FE068C4EE67B0u64, 0x000FDF82B02B71AAu64, 0x000FDE87C57EFEAAu64,
    0x000FDD7509C63BFDu64, 0x000FDC46E529BF13u64, 0x000FDAF8F82E0282u64,
    0x000FD985E1B2BA75u64, 0x000FD7E6EF48CF04u64, 0x000FD613ADBD650Bu64,
    0x000FD40149E2F012u64, 0x000FD1A1A7B4C7ACu64, 0x000FCEE204761F9Eu64,
    0x000FCBA8D85E11B2u64, 0x000FC7D26ECD2D22u64, 0x000FC32B2F1E22EDu64,
    0x000FBD6581C0B83Au64, 0x000FB606C4005434u64, 0x000FAC40582A2874u64,
    0x000F9E971E014598u64, 0x000F89FA48A41DFCu64, 0x000F66C5F7F0302Cu64,
    0x000F1A5A4B331C4Au64,
];
const BP_WI_DOUBLE: [f64; 256] = [
    8.68362706080130616677e-16f64, 4.77933017572773682428e-17f64, 6.35435241740526230246e-17f64,
    7.45487048124769627714e-17f64, 8.32936681579309972857e-17f64, 9.06806040505948228243e-17f64,
    9.71486007656776183958e-17f64, 1.02947503142410192108e-16f64, 1.08234302884476839838e-16f64,
    1.13114701961090307945e-16f64, 1.17663594570229211411e-16f64, 1.21936172787143633280e-16f64,
    1.25974399146370927864e-16f64, 1.29810998862640315416e-16f64, 1.33472037368241227547e-16f64,
    1.36978648425712032797e-16f64, 1.40348230012423820659e-16f64, 1.43595294520569430270e-16f64,
    1.46732087423644219083e-16f64, 1.49769046683910367425e-16f64, 1.52715150035961979750e-16f64,
    1.55578181694607639484e-16f64, 1.58364940092908853989e-16f64, 1.61081401752749279325e-16f64,
    1.63732852039698532012e-16f64, 1.66323990584208352778e-16f64, 1.68859017086765964015e-16f64,
    1.71341701765596607184e-16f64, 1.73775443658648593310e-16f64, 1.76163319230009959832e-16f64,
    1.78508123169767272927e-16f64, 1.80812402857991522674e-16f64, 1.83078487648267501776e-16f64,
    1.85308513886180189386e-16f64, 1.87504446393738816849e-16f64, 1.89668097007747596212e-16f64,
    1.91801140648386198029e-16f64, 1.93905129306251037069e-16f64, 1.95981504266288244037e-16f64,
    1.98031606831281739736e-16f64, 2.00056687762733300198e-16f64, 2.02057915620716538808e-16f64,
    2.04036384154802118313e-16f64, 2.05993118874037063144e-16f64, 2.07929082904140197311e-16f64,
    2.09845182223703516690e-16f64, 2.11742270357603418769e-16f64, 2.13621152594498681022e-16f64,
    2.15482589785814580926e-16f64, 2.17327301775643674990e-16f64, 2.19155970504272708519e-16f64,
    2.20969242822353175995e-16f64, 2.22767733047895534948e-16f64, 2.24552025294143552381e-16f64,
    2.26322675592856786566e-16f64, 2.28080213834501706782e-16f64, 2.29825145544246839061e-16f64,
    2.31557953510408037008e-16f64, 2.33279099280043561128e-16f64, 2.34989024534709550938e-16f64,
    2.36688152357916037468e-16f64, 2.38376888404542434981e-16f64, 2.40055621981350627349e-16f64,
    2.41724727046750252175e-16f64, 2.43384563137110286400e-16f64, 2.45035476226149539878e-16f64,
    2.46677799523270498158e-16f64, 2.48311854216108767769e-16f64, 2.49937950162045242375e-16f64,
    2.51556386532965786439e-16f64, 2.53167452417135826983e-16f64, 2.54771427381694417303e-16f64,
    2.56368581998939683749e-16f64, 2.57959178339286723500e-16f64, 2.59543470433517070146e-16f64,
    2.61121704706701939097e-16f64, 2.62694120385972564623e-16f64, 2.64260949884118951286e-16f64,
    2.65822419160830680292e-16f64, 2.67378748063236329361e-16f64, 2.68930150647261591777e-16f64,
    2.70476835481199518794e-16f64, 2.72019005932773206655e-16f64, 2.73556860440867908686e-16f64,
    2.75090592773016664571e-16f64, 2.76620392269639032183e-16f64, 2.78146444075954410103e-16f64,
    2.79668929362423005309e-16f64, 2.81188025534502074329e-16f64, 2.82703906432447923059e-16f64,
    2.84216742521840606520e-16f64, 2.85726701075460149289e-16f64, 2.87233946347097994381e-16f64,
    2.88738639737848191815e-16f64, 2.90240939955384233230e-16f64, 2.91741003166694553259e-16f64,
    2.93238983144718163965e-16f64, 2.94735031409293489611e-16f64, 2.96229297362806647792e-16f64,
    2.97721928420902891115e-16f64, 2.99213070138601307081e-16f64, 3.00702866332133102993e-16f64,
    3.02191459196806151971e-16f64, 3.03678989421180184427e-16f64, 3.05165596297821922381e-16f64,
    3.06651417830895451744e-16f64, 3.08136590840829717032e-16f64, 3.09621251066292253306e-16f64,
    3.11105533263689296831e-16f64, 3.12589571304399892784e-16f64, 3.14073498269944617203e-16f64,
    3.15557446545280064031e-16f64, 3.17041547910402852545e-16f64, 3.18525933630440648871e-16f64,
    3.20010734544401137886e-16f64, 3.21496081152744704901e-16f64, 3.22982103703941557538e-16f64,
    3.24468932280169778077e-16f64, 3.25956696882307838340e-16f64, 3.27445527514370671802e-16f64,
    3.28935554267536967851e-16f64, 3.30426907403912838589e-16f64, 3.31919717440175233652e-16f64,
    3.33414115231237245918e-16f64, 3.34910232054077845412e-16f64, 3.36408199691876507948e-16f64,
    3.37908150518594979994e-16f64, 3.39410217584148914282e-16f64, 3.40914534700312603713e-16f64,
    3.42421236527501816058e-16f64, 3.43930458662583133920e-16f64, 3.45442337727858401604e-16f64,
    3.46957011461378353333e-16f64, 3.48474618808741370700e-16f64, 3.49995300016538099813e-16f64,
    3.51519196727607440975e-16f64, 3.53046452078274009054e-16f64, 3.54577210797743572160e-16f64,
    3.56111619309838843415e-16f64, 3.57649825837265051035e-16f64, 3.59191980508602994994e-16f64,
    3.60738235468235137839e-16f64, 3.62288744989419151904e-16f64, 3.63843665590734438546e-16f64,
    3.65403156156136995766e-16f64, 3.66967378058870090021e-16f64, 3.68536495289491401456e-16f64,
    3.70110674588289834952e-16f64, 3.71690085582382297792e-16f64, 3.73274900927794352614e-16f64,
    3.74865296456848868882e-16f64, 3.76461451331202869131e-16f64, 3.78063548200896037651e-16f64,
    3.79671773369794425924e-16f64, 3.81286316967837738238e-16f64, 3.82907373130524317507e-16f64,
    3.84535140186095955858e-16f64, 3.86169820850914927119e-16f64, 3.87811622433558721164e-16f64,
    3.89460757048192620674e-16f64, 3.91117441837820542060e-16f64, 3.92781899208054153270e-16f64,
    3.94454357072087711446e-16f64, 3.96135049107613542983e-16f64, 3.97824215026468259474e-16f64,
    3.99522100857856502444e-16f64, 4.01228959246062907451e-16f64, 4.02945049763632792393e-16f64,
    4.04670639241074995115e-16f64, 4.06406002114225038723e-16f64, 4.08151420790493873480e-16f64,
    4.09907186035326643447e-16f64, 4.11673597380302570170e-16f64, 4.13450963554423599878e-16f64,
    4.15239602940268833891e-16f64, 4.17039844056831587498e-16f64, 4.18852026071011229572e-16f64,
    4.20676499339901510978e-16f64, 4.22513625986204937320e-16f64, 4.24363780509307796137e-16f64,
    4.26227350434779809917e-16f64, 4.28104737005311666397e-16f64, 4.29996355916383230161e-16f64,
    4.31902638100262944617e-16f64, 4.33824030562279080411e-16f64, 4.35760997273684900553e-16f64,
    4.37714020125858747008e-16f64, 4.39683599951052137423e-16f64, 4.41670257615420348435e-16f64,
    4.43674535190656726604e-16f64, 4.45696997211204306674e-16f64, 4.47738232024753387312e-16f64,
    4.49798853244554968009e-16f64, 4.51879501313005876278e-16f64, 4.53980845187003400947e-16f64,
    4.56103584156742206384e-16f64, 4.58248449810956667052e-16f64, 4.60416208163115281428e-16f64,
    4.62607661954784567754e-16f64, 4.64823653154320737780e-16f64, 4.67065065671263059081e-16f64,
    4.69332828309332890697e-16f64, 4.71627917983835129766e-16f64, 4.73951363232586715165e-16f64,
    4.76304248053313737663e-16f64, 4.78687716104872284247e-16f64, 4.81102975314741720538e-16f64,
    4.83551302941152515162e-16f64, 4.86034051145081195402e-16f64, 4.88552653135360343280e-16f64,
    4.91108629959526955862e-16f64, 4.93703598024033454728e-16f64, 4.96339277440398725619e-16f64,
    4.99017501309182245754e-16f64, 5.01740226071808946011e-16f64, 5.04509543081872748637e-16f64,
    5.07327691573354207058e-16f64, 5.10197073234156184149e-16f64, 5.13120268630678373200e-16f64,
    5.16100055774322824569e-16f64, 5.19139431175769859873e-16f64, 5.22241633800023428760e-16f64,
    5.25410172417759732697e-16f64, 5.28648856950494511482e-16f64, 5.31961834533840037535e-16f64,
    5.35353631181649688145e-16f64, 5.38829200133405320160e-16f64, 5.42393978220171234073e-16f64,
    5.46053951907478041166e-16f64, 5.49815735089281410703e-16f64, 5.53686661246787600374e-16f64,
    5.57674893292657647836e-16f64, 5.61789555355541665830e-16f64, 5.66040892008242216739e-16f64,
    5.70440462129138908417e-16f64, 5.75001376891989523684e-16f64, 5.79738594572459365014e-16f64,
    5.84669289345547900201e-16f64, 5.89813317647789942685e-16f64, 5.95193814964144415532e-16f64,
    6.00837969627190832234e-16f64, 6.06778040933344851394e-16f64, 6.13052720872528159123e-16f64,
    6.19708989458162555387e-16f64, 6.26804696330128439415e-16f64, 6.34412240712750598627e-16f64,
    6.42623965954805540945e-16f64, 6.51560331734499356881e-16f64, 6.61382788509766415145e-16f64,
    6.72315046250558662913e-16f64, 6.84680341756425875856e-16f64, 6.98971833638761995415e-16f64,
    7.15999493483066421560e-16f64, 7.37242430179879890722e-16f64, 7.65893637080557275482e-16f64,
    8.11384933765648418565e-16f64,
];
const BP_FI_DOUBLE: [f64; 256] = [
    1.00000000000000000000e+00f64, 9.77101701267671596263e-01f64, 9.59879091800106665211e-01f64,
    9.45198953442299649730e-01f64, 9.32060075959230460718e-01f64, 9.19991505039347012840e-01f64,
    9.08726440052130879366e-01f64, 8.98095921898343418910e-01f64, 8.87984660755833377088e-01f64,
    8.78309655808917399966e-01f64, 8.69008688036857046555e-01f64, 8.60033621196331532488e-01f64,
    8.51346258458677951353e-01f64, 8.42915653112204177333e-01f64, 8.34716292986883434679e-01f64,
    8.26726833946221373317e-01f64, 8.18929191603702366642e-01f64, 8.11307874312656274185e-01f64,
    8.03849483170964274059e-01f64, 7.96542330422958966274e-01f64, 7.89376143566024590648e-01f64,
    7.82341832654802504798e-01f64, 7.75431304981187174974e-01f64, 7.68637315798486264740e-01f64,
    7.61953346836795386565e-01f64, 7.55373506507096115214e-01f64, 7.48892447219156820459e-01f64,
    7.42505296340151055290e-01f64, 7.36207598126862650112e-01f64, 7.29995264561476231435e-01f64,
    7.23864533468630222401e-01f64, 7.17811932630721960535e-01f64, 7.11834248878248421200e-01f64,
    7.05928501332754310127e-01f64, 7.00091918136511615067e-01f64, 6.94321916126116711609e-01f64,
    6.88616083004671808432e-01f64, 6.82972161644994857355e-01f64, 6.77388036218773526009e-01f64,
    6.71861719897082099173e-01f64, 6.66391343908750100056e-01f64, 6.60975147776663107813e-01f64,
    6.55611470579697264149e-01f64, 6.50298743110816701574e-01f64, 6.45035480820822293424e-01f64,
    6.39820277453056585060e-01f64, 6.34651799287623608059e-01f64, 6.29528779924836690007e-01f64,
    6.24450015547026504592e-01f64, 6.19414360605834324325e-01f64, 6.14420723888913888899e-01f64,
    6.09468064925773433949e-01f64, 6.04555390697467776029e-01f64, 5.99681752619125263415e-01f64,
    5.94846243767987448159e-01f64, 5.90047996332826008015e-01f64, 5.85286179263371453274e-01f64,
    5.80559996100790898232e-01f64, 5.75868682972353718164e-01f64, 5.71211506735253227163e-01f64,
    5.66587763256164445025e-01f64, 5.61996775814524340831e-01f64, 5.57437893618765945014e-01f64,
    5.52910490425832290562e-01f64, 5.48413963255265812791e-01f64, 5.43947731190026262382e-01f64,
    5.39511234256952132426e-01f64, 5.35103932380457614215e-01f64, 5.30725304403662057062e-01f64,
    5.26374847171684479008e-01f64, 5.22052074672321841931e-01f64, 5.17756517229756352272e-01f64,
    5.13487720747326958914e-01f64, 5.09245245995747941592e-01f64, 5.05028667943468123624e-01f64,
    5.00837575126148681903e-01f64, 4.96671569052489714213e-01f64, 4.92530263643868537748e-01f64,
    4.88413284705458028423e-01f64, 4.84320269426683325253e-01f64, 4.80250865909046753544e-01f64,
    4.76204732719505863248e-01f64, 4.72181538467730199660e-01f64, 4.68180961405693596422e-01f64,
    4.64202689048174355069e-01f64, 4.60246417812842867345e-01f64, 4.56311852678716434184e-01f64,
    4.52398706861848520777e-01f64, 4.48506701507203064949e-01f64, 4.44635565395739396077e-01f64,
    4.40785034665803987508e-01f64, 4.36954852547985550526e-01f64, 4.33144769112652261445e-01f64,
    4.29354541029441427735e-01f64, 4.25583931338021970170e-01f64, 4.21832709229495894654e-01f64,
    4.18100649837848226120e-01f64, 4.14387534040891125642e-01f64, 4.10693148270188157500e-01f64,
    4.07017284329473372217e-01f64, 4.03359739221114510510e-01f64, 3.99720314980197222177e-01f64,
    3.96098818515832451492e-01f64, 3.92495061459315619512e-01f64, 3.88908860018788715696e-01f64,
    3.85340034840077283462e-01f64, 3.81788410873393657674e-01f64, 3.78253817245619183840e-01f64,
    3.74736087137891138443e-01f64, 3.71235057668239498696e-01f64, 3.67750569779032587814e-01f64,
    3.64282468129004055601e-01f64, 3.60830600989648031529e-01f64, 3.57394820145780500731e-01f64,
    3.53974980800076777232e-01f64, 3.50570941481406106455e-01f64, 3.47182563956793643900e-01f64,
    3.43809713146850715049e-01f64, 3.40452257044521866547e-01f64, 3.37110066637006045021e-01f64,
    3.33783015830718454708e-01f64, 3.30470981379163586400e-01f64, 3.27173842813601400970e-01f64,
    3.23891482376391093290e-01f64, 3.20623784956905355514e-01f64, 3.17370638029913609834e-01f64,
    3.14131931596337177215e-01f64, 3.10907558126286509559e-01f64, 3.07697412504292056035e-01f64,
    3.04501391976649993243e-01f64, 3.01319396100803049698e-01f64, 2.98151326696685481377e-01f64,
    2.94997087799961810184e-01f64, 2.91856585617095209972e-01f64, 2.88729728482182923521e-01f64,
    2.85616426815501756042e-01f64, 2.82516593083707578948e-01f64, 2.79430141761637940157e-01f64,
    2.76356989295668320494e-01f64, 2.73297054068577072172e-01f64, 2.70250256365875463072e-01f64,
    2.67216518343561471038e-01f64, 2.64195763997261190426e-01f64, 2.61187919132721213522e-01f64,
    2.58192911337619235290e-01f64, 2.55210669954661961700e-01f64, 2.52241126055942177508e-01f64,
    2.49284212418528522415e-01f64, 2.46339863501263828249e-01f64, 2.43408015422750312329e-01f64,
    2.40488605940500588254e-01f64, 2.37581574431238090606e-01f64, 2.34686861872330010392e-01f64,
    2.31804410824338724684e-01f64, 2.28934165414680340644e-01f64, 2.26076071322380278694e-01f64,
    2.23230075763917484855e-01f64, 2.20396127480151998723e-01f64, 2.17574176724331130872e-01f64,
    2.14764175251173583536e-01f64, 2.11966076307030182324e-01f64, 2.09179834621125076977e-01f64,
    2.06405406397880797353e-01f64, 2.03642749310334908452e-01f64, 2.00891822494656591136e-01f64,
    1.98152586545775138971e-01f64, 1.95425003514134304483e-01f64, 1.92709036903589175926e-01f64,
    1.90004651670464985713e-01f64, 1.87311814223800304768e-01f64, 1.84630492426799269756e-01f64,
    1.81960655599522513892e-01f64, 1.79302274522847582272e-01f64, 1.76655321443734858455e-01f64,
    1.74019770081838553999e-01f64, 1.71395595637505754327e-01f64, 1.68782774801211288285e-01f64,
    1.66181285764481906364e-01f64, 1.63591108232365584074e-01f64, 1.61012223437511009516e-01f64,
    1.58444614155924284882e-01f64, 1.55888264724479197465e-01f64, 1.53343161060262855866e-01f64,
    1.50809290681845675763e-01f64, 1.48286642732574552861e-01f64, 1.45775208005994028060e-01f64,
    1.43274978973513461566e-01f64, 1.40785949814444699690e-01f64, 1.38308116448550733057e-01f64,
    1.35841476571253755301e-01f64, 1.33386029691669155683e-01f64, 1.30941777173644358090e-01f64,
    1.28508722279999570981e-01f64, 1.26086870220185887081e-01f64, 1.23676228201596571932e-01f64,
    1.21276805484790306533e-01f64, 1.18888613442910059947e-01f64, 1.16511665625610869035e-01f64,
    1.14145977827838487895e-01f64, 1.11791568163838089811e-01f64, 1.09448457146811797824e-01f64,
    1.07116667774683801961e-01f64, 1.04796225622487068629e-01f64, 1.02487158941935246892e-01f64,
    1.00189498768810017482e-01f64, 9.79032790388624646338e-02f64, 9.56285367130089991594e-02f64,
    9.33653119126910124859e-02f64, 9.11136480663737591268e-02f64, 8.88735920682758862021e-02f64,
    8.66451944505580717859e-02f64, 8.44285095703534715916e-02f64, 8.22235958132029043366e-02f64,
    8.00305158146630696292e-02f64, 7.78493367020961224423e-02f64, 7.56801303589271778804e-02f64,
    7.35229737139813238622e-02f64, 7.13779490588904025339e-02f64, 6.92451443970067553879e-02f64,
    6.71246538277884968737e-02f64, 6.50165779712428976156e-02f64, 6.29210244377581412456e-02f64,
    6.08381083495398780614e-02f64, 5.87679529209337372930e-02f64, 5.67106901062029017391e-02f64,
    5.46664613248889208474e-02f64, 5.26354182767921896513e-02f64, 5.06177238609477817000e-02f64,
    4.86135532158685421122e-02f64, 4.66230949019303814174e-02f64, 4.46465522512944634759e-02f64,
    4.26841449164744590750e-02f64, 4.07361106559409394401e-02f64, 3.88027074045261474722e-02f64,
    3.68842156885673053135e-02f64, 3.49809414617161251737e-02f64, 3.30932194585785779961e-02f64,
    3.12214171919203004046e-02f64, 2.93659397581333588001e-02f64, 2.75272356696031131329e-02f64,
    2.57058040085489103443e-02f64, 2.39022033057958785407e-02f64, 2.21170627073088502113e-02f64,
    2.03510962300445102935e-02f64, 1.86051212757246224594e-02f64, 1.68800831525431419000e-02f64,
    1.51770883079353092332e-02f64, 1.34974506017398673818e-02f64, 1.18427578579078790488e-02f64,
    1.02149714397014590439e-02f64, 8.61658276939872638800e-03f64, 7.05087547137322242369e-03f64,
    5.52240329925099155545e-03f64, 4.03797259336302356153e-03f64, 2.60907274610215926189e-03f64,
    1.26028593049859797236e-03f64,
];
const BP_KE_DOUBLE: [u64; 256] = [
    0x001C5214272497C6u64, 0x0000000000000000u64, 0x00137D5BD79C317Eu64,
    0x00186EF58E3F3C10u64, 0x001A9BB7320EB0AEu64, 0x001BD127F719447Cu64,
    0x001C951D0F88651Au64, 0x001D1BFE2D5C3972u64, 0x001D7E5BD56B18B2u64,
    0x001DC934DD172C70u64, 0x001E0409DFAC9DC8u64, 0x001E337B71D47836u64,
    0x001E5A8B177CB7A2u64, 0x001E7B42096F046Cu64, 0x001E970DAF08AE3Eu64,
    0x001EAEF5B14EF09Eu64, 0x001EC3BD07B46556u64, 0x001ED5F6F08799CEu64,
    0x001EE614AE6E5688u64, 0x001EF46ECA361CD0u64, 0x001F014B76DDD4A4u64,
    0x001F0CE313A796B6u64, 0x001F176369F1F77Au64, 0x001F20F20C452570u64,
    0x001F29AE1951A874u64, 0x001F31B18FB95532u64, 0x001F39125157C106u64,
    0x001F3FE2EB6E694Cu64, 0x001F463332D788FAu64, 0x001F4C10BF1D3A0Eu64,
    0x001F51874C5C3322u64, 0x001F56A109C3ECC0u64, 0x001F5B66D9099996u64,
    0x001F5FE08210D08Cu64, 0x001F6414DD445772u64, 0x001F6809F6859678u64,
    0x001F6BC52A2B02E6u64, 0x001F6F4B3D32E4F4u64, 0x001F72A07190F13Au64,
    0x001F75C8974D09D6u64, 0x001F78C71B045CC0u64, 0x001F7B9F12413FF4u64,
    0x001F7E5346079F8Au64, 0x001F80E63BE21138u64, 0x001F835A3DAD9162u64,
    0x001F85B16056B912u64, 0x001F87ED89B24262u64, 0x001F8A10759374FAu64,
    0x001F8C1BBA3D39ACu64, 0x001F8E10CC45D04Au64, 0x001F8FF102013E16u64,
    0x001F91BD968358E0u64, 0x001F9377AC47AFD8u64, 0x001F95204F8B64DAu64,
    0x001F96B878633892u64, 0x001F98410C968892u64, 0x001F99BAE146BA80u64,
    0x001F9B26BC697F00u64, 0x001F9C85561B717Au64, 0x001F9DD759CFD802u64,
    0x001F9F1D6761A1CEu64, 0x001FA058140936C0u64, 0x001FA187EB3A3338u64,
    0x001FA2AD6F6BC4FCu64, 0x001FA3C91ACE0682u64, 0x001FA4DB5FEE6AA2u64,
    0x001FA5E4AA4D097Cu64, 0x001FA6E55EE46782u64, 0x001FA7DDDCA51EC4u64,
    0x001FA8CE7CE6A874u64, 0x001FA9B793CE5FEEu64, 0x001FAA9970ADB858u64,
    0x001FAB745E588232u64, 0x001FAC48A3740584u64, 0x001FAD1682BF9FE8u64,
    0x001FADDE3B5782C0u64, 0x001FAEA008F21D6Cu64, 0x001FAF5C2418B07Eu64,
    0x001FB012C25B7A12u64, 0x001FB0C41681DFF4u64, 0x001FB17050B6F1FAu64,
    0x001FB2179EB2963Au64, 0x001FB2BA2BDFA84Au64, 0x001FB358217F4E18u64,
    0x001FB3F1A6C9BE0Cu64, 0x001FB486E10CACD6u64, 0x001FB517F3C793FCu64,
    0x001FB5A500C5FDAAu64, 0x001FB62E2837FE58u64, 0x001FB6B388C9010Au64,
    0x001FB7353FB50798u64, 0x001FB7B368DC7DA8u64, 0x001FB82E1ED6BA08u64,
    0x001FB8A57B0347F6u64, 0x001FB919959A0F74u64, 0x001FB98A85BA7204u64,
    0x001FB9F861796F26u64, 0x001FBA633DEEE286u64, 0x001FBACB2F41EC16u64,
    0x001FBB3048B49144u64, 0x001FBB929CAEA4E2u64, 0x001FBBF23CC8029Eu64,
    0x001FBC4F39D22994u64, 0x001FBCA9A3E140D4u64, 0x001FBD018A548F9Eu64,
    0x001FBD56FBDE729Cu64, 0x001FBDAA068BD66Au64, 0x001FBDFAB7CB3F40u64,
    0x001FBE491C7364DEu64, 0x001FBE9540C9695Eu64, 0x001FBEDF3086B128u64,
    0x001FBF26F6DE6174u64, 0x001FBF6C9E828AE2u64, 0x001FBFB031A904C4u64,
    0x001FBFF1BA0FFDB0u64, 0x001FC03141024588u64, 0x001FC06ECF5B54B2u64,
    0x001FC0AA6D8B1426u64, 0x001FC0E42399698Au64, 0x001FC11BF9298A64u64,
    0x001FC151F57D1942u64, 0x001FC1861F770F4Au64, 0x001FC1B87D9E74B4u64,
    0x001FC1E91620EA42u64, 0x001FC217EED505DEu64, 0x001FC2450D3C83FEu64,
    0x001FC27076864FC2u64, 0x001FC29A2F90630Eu64, 0x001FC2C23CE98046u64,
    0x001FC2E8A2D2C6B4u64, 0x001FC30D654122ECu64, 0x001FC33087DE9C0Eu64,
    0x001FC3520E0B7EC6u64, 0x001FC371FADF66F8u64, 0x001FC390512A2886u64,
    0x001FC3AD137497FAu64, 0x001FC3C844013348u64, 0x001FC3E1E4CCAB40u64,
    0x001FC3F9F78E4DA8u64, 0x001FC4107DB85060u64, 0x001FC4257877FD68u64,
    0x001FC438E8B5BFC6u64, 0x001FC44ACF15112Au64, 0x001FC45B2BF447E8u64,
    0x001FC469FF6C4504u64, 0x001FC477495001B2u64, 0x001FC483092BFBB8u64,
    0x001FC48D3E457FF6u64, 0x001FC495E799D21Au64, 0x001FC49D03DD30B0u64,
    0x001FC4A29179B432u64, 0x001FC4A68E8E07FCu64, 0x001FC4A8F8EBFB8Cu64,
    0x001FC4A9CE16EA9Eu64, 0x001FC4A90B41FA34u64, 0x001FC4A6AD4E28A0u64,
    0x001FC4A2B0C82E74u64, 0x001FC49D11E62DE2u64, 0x001FC495CC852DF4u64,
    0x001FC48CDC265EC0u64, 0x001FC4823BEC237Au64, 0x001FC475E696DEE6u64,
    0x001FC467D6817E82u64, 0x001FC458059DC036u64, 0x001FC4466D702E20u64,
    0x001FC433070BCB98u64, 0x001FC41DCB0D6E0Eu64, 0x001FC406B196BBF6u64,
    0x001FC3EDB248CB62u64, 0x001FC3D2C43E593Cu64, 0x001FC3B5DE0591B4u64,
    0x001FC396F599614Cu64, 0x001FC376005A4592u64, 0x001FC352F3069370u64,
    0x001FC32DC1B22818u64, 0x001FC3065FBD7888u64, 0x001FC2DCBFCBF262u64,
    0x001FC2B0D3B99F9Eu64, 0x001FC2828C8FFCF0u64, 0x001FC251DA79F164u64,
    0x001FC21EACB6D39Eu64, 0x001FC1E8F18C6756u64, 0x001FC1B09637BB3Cu64,
    0x001FC17586DCCD10u64, 0x001FC137AE74D6B6u64, 0x001FC0F6F6BB2414u64,
    0x001FC0B348184DA4u64, 0x001FC06C898BAFF0u64, 0x001FC022A092F364u64,
    0x001FBFD5710F72B8u64, 0x001FBF84DD29488Eu64, 0x001FBF30C52FC60Au64,
    0x001FBED907770CC6u64, 0x001FBE7D80327DDAu64, 0x001FBE1E094BA614u64,
    0x001FBDBA7A354408u64, 0x001FBD52A7B9F826u64, 0x001FBCE663C6201Au64,
    0x001FBC757D2C4DE4u64, 0x001FBBFFBF63B7AAu64, 0x001FBB84F23FE6A2u64,
    0x001FBB04D9A0D18Cu64, 0x001FBA7F351A70ACu64, 0x001FB9F3BF92B618u64,
    0x001FB9622ED4ABFCu64, 0x001FB8CA33174A16u64, 0x001FB82B76765B54u64,
    0x001FB7859C5B895Cu64, 0x001FB6D840D55594u64, 0x001FB622F7D96942u64,
    0x001FB5654C6F37E0u64, 0x001FB49EBFBF69D2u64, 0x001FB3CEC803E746u64,
    0x001FB2F4CF539C3Eu64, 0x001FB21032442852u64, 0x001FB1203E5A9604u64,
    0x001FB0243042E1C2u64, 0x001FAF1B31C479A6u64, 0x001FAE045767E104u64,
    0x001FACDE9DBF2D72u64, 0x001FABA8E640060Au64, 0x001FAA61F399FF28u64,
    0x001FA908656F66A2u64, 0x001FA79AB3508D3Cu64, 0x001FA61726D1F214u64,
    0x001FA47BD48BEA00u64, 0x001FA2C693C5C094u64, 0x001FA0F4F47DF314u64,
    0x001F9F04336BBE0Au64, 0x001F9CF12B79F9BCu64, 0x001F9AB84415ABC4u64,
    0x001F98555B782FB8u64, 0x001F95C3ABD03F78u64, 0x001F92FDA9CEF1F2u64,
    0x001F8FFCDA9AE41Cu64, 0x001F8CB99E7385F8u64, 0x001F892AEC479606u64,
    0x001F8545F904DB8Eu64, 0x001F80FDC336039Au64, 0x001F7C427839E926u64,
    0x001F7700A3582ACCu64, 0x001F71200F1A241Cu64, 0x001F6A8234B7352Au64,
    0x001F630000A8E266u64, 0x001F5A66904FE3C4u64, 0x001F50724ECE1172u64,
    0x001F44C7665C6FDAu64, 0x001F36E5A38A59A2u64, 0x001F26143450340Au64,
    0x001F113E047B0414u64, 0x001EF6AEFA57CBE6u64, 0x001ED38CA188151Eu64,
    0x001EA2A61E122DB0u64, 0x001E5961C78B267Cu64, 0x001DDDF62BAC0BB0u64,
    0x001CDB4DD9E4E8C0u64,
];
const BP_WE_DOUBLE: [f64; 256] = [
    9.655740063209182975e-16f64, 7.089014243955414331e-18f64, 1.163941249669122378e-17f64,
    1.524391512353216015e-17f64, 1.833284885723743916e-17f64, 2.108965109464486630e-17f64,
    2.361128077843138196e-17f64, 2.595595772310893952e-17f64, 2.816173554197752338e-17f64,
    3.025504130321382330e-17f64, 3.225508254836375280e-17f64, 3.417632340185027033e-17f64,
    3.602996978734452488e-17f64, 3.782490776869649048e-17f64, 3.956832198097553231e-17f64,
    4.126611778175946428e-17f64, 4.292321808442525631e-17f64, 4.454377743282371417e-17f64,
    4.613133981483185932e-17f64, 4.768895725264635940e-17f64, 4.921928043727962847e-17f64,
    5.072462904503147014e-17f64, 5.220704702792671737e-17f64, 5.366834661718192181e-17f64,
    5.511014372835094717e-17f64, 5.653388673239667134e-17f64, 5.794088004852766616e-17f64,
    5.933230365208943081e-17f64, 6.070922932847179572e-17f64, 6.207263431163193485e-17f64,
    6.342341280303076511e-17f64, 6.476238575956142121e-17f64, 6.609030925769405241e-17f64,
    6.740788167872722244e-17f64, 6.871574991183812442e-17f64, 7.001451473403929616e-17f64,
    7.130473549660643409e-17f64, 7.258693422414648352e-17f64, 7.386159921381791997e-17f64,
    7.512918820723728089e-17f64, 7.639013119550825792e-17f64, 7.764483290797848102e-17f64,
    7.889367502729790548e-17f64, 8.013701816675454434e-17f64, 8.137520364041762206e-17f64,
    8.260855505210038174e-17f64, 8.383737972539139383e-17f64, 8.506196999385323132e-17f64,
    8.628260436784112996e-17f64, 8.749954859216182511e-17f64, 8.871305660690252281e-17f64,
    8.992337142215357066e-17f64, 9.113072591597909173e-17f64, 9.233534356381788123e-17f64,
    9.353743910649128938e-17f64, 9.473721916312949566e-17f64, 9.593488279457997317e-17f64,
    9.713062202221521206e-17f64, 9.832462230649511362e-17f64, 9.951706298915071878e-17f64,
    1.007081177024294931e-16f64, 1.018979547484694078e-16f64, 1.030867374515421954e-16f64,
    1.042746244856188556e-16f64, 1.054617701794576406e-16f64, 1.066483248011914702e-16f64,
    1.078344348241948498e-16f64, 1.090202431758350473e-16f64, 1.102058894705578110e-16f64,
    1.113915102286197502e-16f64, 1.125772390816567488e-16f64, 1.137632069661684705e-16f64,
    1.149495423059009298e-16f64, 1.161363711840218308e-16f64, 1.173238175059045788e-16f64,
    1.185120031532669434e-16f64, 1.197010481303465158e-16f64, 1.208910707027385520e-16f64,
    1.220821875294706151e-16f64, 1.232745137888415193e-16f64, 1.244681632985112523e-16f64,
    1.256632486302898513e-16f64, 1.268598812200397542e-16f64, 1.280581714730749379e-16f64,
    1.292582288654119552e-16f64, 1.304601620412028847e-16f64, 1.316640789066572582e-16f64,
    1.328700867207380889e-16f64, 1.340782921828999433e-16f64, 1.352888015181175458e-16f64,
    1.365017205594397770e-16f64, 1.377171548282880964e-16f64, 1.389352096127063919e-16f64,
    1.401559900437571538e-16f64, 1.413796011702485188e-16f64, 1.426061480319665444e-16f64,
    1.438357357315790180e-16f64, 1.450684695053687684e-16f64, 1.463044547929475721e-16f64,
    1.475437973060951633e-16f64, 1.487866030968626066e-16f64, 1.500329786250736949e-16f64,
    1.512830308253539427e-16f64, 1.525368671738125550e-16f64, 1.537945957544996933e-16f64,
    1.550563253257577148e-16f64, 1.563221653865837505e-16f64, 1.575922262431176140e-16f64,
    1.588666190753684151e-16f64, 1.601454560042916733e-16f64, 1.614288501593278662e-16f64,
    1.627169157465130500e-16f64, 1.640097681172717950e-16f64, 1.653075238380036909e-16f64,
    1.666103007605742067e-16f64, 1.679182180938228863e-16f64, 1.692313964762022267e-16f64,
    1.705499580496629830e-16f64, 1.718740265349031656e-16f64, 1.732037273081008369e-16f64,
    1.745391874792533975e-16f64, 1.758805359722491379e-16f64, 1.772279036068006489e-16f64,
    1.785814231823732619e-16f64, 1.799412295642463721e-16f64, 1.813074597718501559e-16f64,
    1.826802530695252266e-16f64, 1.840597510598587828e-16f64, 1.854460977797569461e-16f64,
    1.868394397994192684e-16f64, 1.882399263243892051e-16f64, 1.896477093008616722e-16f64,
    1.910629435244376536e-16f64, 1.924857867525243818e-16f64, 1.939163998205899420e-16f64,
    1.953549467624909132e-16f64, 1.968015949351037382e-16f64, 1.982565151475019047e-16f64,
    1.997198817949342081e-16f64, 2.011918729978734671e-16f64, 2.026726707464198289e-16f64,
    2.041624610503588774e-16f64, 2.056614340951917875e-16f64, 2.071697844044737034e-16f64,
    2.086877110088159721e-16f64, 2.102154176219292789e-16f64, 2.117531128241075913e-16f64,
    2.133010102535779087e-16f64, 2.148593288061663316e-16f64, 2.164282928437604723e-16f64,
    2.180081324120784027e-16f64, 2.195990834682870728e-16f64, 2.212013881190495942e-16f64,
    2.228152948696180545e-16f64, 2.244410588846308588e-16f64, 2.260789422613173739e-16f64,
    2.277292143158621037e-16f64, 2.293921518837311354e-16f64, 2.310680396348213318e-16f64,
    2.327571704043534613e-16f64, 2.344598455404957859e-16f64, 2.361763752697773994e-16f64,
    2.379070790814276700e-16f64, 2.396522861318623520e-16f64, 2.414123356706293277e-16f64,
    2.431875774892255956e-16f64, 2.449783723943070217e-16f64, 2.467850927069288738e-16f64,
    2.486081227895851719e-16f64, 2.504478596029557040e-16f64, 2.523047132944217013e-16f64,
    2.541791078205812227e-16f64, 2.560714816061770759e-16f64, 2.579822882420530896e-16f64,
    2.599119972249746917e-16f64, 2.618610947423924219e-16f64, 2.638300845054942823e-16f64,
    2.658194886341845120e-16f64, 2.678298485979525166e-16f64, 2.698617262169488933e-16f64,
    2.719157047279818500e-16f64, 2.739923899205814823e-16f64, 2.760924113487617126e-16f64,
    2.782164236246436081e-16f64, 2.803651078006983464e-16f64, 2.825391728480253184e-16f64,
    2.847393572388174091e-16f64, 2.869664306419817679e-16f64, 2.892211957417995598e-16f64,
    2.915044901905293183e-16f64, 2.938171887070028633e-16f64, 2.961602053345465687e-16f64,
    2.985344958730045276e-16f64, 3.009410605012618141e-16f64, 3.033809466085003416e-16f64,
    3.058552518544860874e-16f64, 3.083651274815310004e-16f64, 3.109117819034266344e-16f64,
    3.134964845996663118e-16f64, 3.161205703467105734e-16f64, 3.187854438219713117e-16f64,
    3.214925846206797361e-16f64, 3.242435527309451638e-16f64, 3.270399945182240440e-16f64,
    3.298836492772283149e-16f64, 3.327763564171671408e-16f64, 3.357200633553244075e-16f64,
    3.387168342045505162e-16f64, 3.417688593525636996e-16f64, 3.448784660453423890e-16f64,
    3.480481301037442286e-16f64, 3.512804889222979418e-16f64, 3.545783559224791863e-16f64,
    3.579447366604276541e-16f64, 3.613828468219060593e-16f64, 3.648961323764542545e-16f64,
    3.684882922095621322e-16f64, 3.721633036080207290e-16f64, 3.759254510416256532e-16f64,
    3.797793587668874387e-16f64, 3.837300278789213687e-16f64, 3.877828785607895292e-16f64,
    3.919437984311428867e-16f64, 3.962191980786774996e-16f64, 4.006160751056541688e-16f64,
    4.051420882956573177e-16f64, 4.098056438903062509e-16f64, 4.146159964290904582e-16f64,
    4.195833672073398926e-16f64, 4.247190841824385048e-16f64, 4.300357481667470702e-16f64,
    4.355474314693952008e-16f64, 4.412699169036069903e-16f64, 4.472209874259932285e-16f64,
    4.534207798565834480e-16f64, 4.598922204905932469e-16f64, 4.666615664711475780e-16f64,
    4.737590853262492027e-16f64, 4.812199172829237933e-16f64, 4.890851827392209900e-16f64,
    4.974034236191939753e-16f64, 5.062325072144159699e-16f64, 5.156421828878082953e-16f64,
    5.257175802022274839e-16f64, 5.365640977112021618e-16f64, 5.483144034258703912e-16f64,
    5.611387454675159622e-16f64, 5.752606481503331688e-16f64, 5.909817641652102998e-16f64,
    6.087231416180907671e-16f64, 6.290979034877557049e-16f64, 6.530492053564040799e-16f64,
    6.821393079028928626e-16f64, 7.192444966089361564e-16f64, 7.706095350032096755e-16f64,
    8.545517038584027421e-16f64,
];
const BP_FE_DOUBLE: [f64; 256] = [
    1.000000000000000000e+00f64, 9.381436808621747003e-01f64, 9.004699299257464817e-01f64,
    8.717043323812035949e-01f64, 8.477855006239896074e-01f64, 8.269932966430503241e-01f64,
    8.084216515230083777e-01f64, 7.915276369724956185e-01f64, 7.759568520401155522e-01f64,
    7.614633888498962833e-01f64, 7.478686219851951034e-01f64, 7.350380924314234843e-01f64,
    7.228676595935720206e-01f64, 7.112747608050760117e-01f64, 7.001926550827881623e-01f64,
    6.895664961170779872e-01f64, 6.793505722647653622e-01f64, 6.695063167319247333e-01f64,
    6.600008410789997004e-01f64, 6.508058334145710999e-01f64, 6.418967164272660897e-01f64,
    6.332519942143660652e-01f64, 6.248527387036659775e-01f64, 6.166821809152076561e-01f64,
    6.087253820796220127e-01f64, 6.009689663652322267e-01f64, 5.934009016917334289e-01f64,
    5.860103184772680329e-01f64, 5.787873586028450257e-01f64, 5.717230486648258170e-01f64,
    5.648091929124001709e-01f64, 5.580382822625874484e-01f64, 5.514034165406412891e-01f64,
    5.448982376724396115e-01f64, 5.385168720028619127e-01f64, 5.322538802630433219e-01f64,
    5.261042139836197284e-01f64, 5.200631773682335979e-01f64, 5.141263938147485613e-01f64,
    5.082897764106428795e-01f64, 5.025495018413477233e-01f64, 4.969019872415495476e-01f64,
    4.913438695940325340e-01f64, 4.858719873418849144e-01f64, 4.804833639304542103e-01f64,
    4.751751930373773747e-01f64, 4.699448252839599771e-01f64, 4.647897562504261781e-01f64,
    4.597076156421376902e-01f64, 4.546961574746155033e-01f64, 4.497532511627549967e-01f64,
    4.448768734145485126e-01f64, 4.400651008423538957e-01f64, 4.353161032156365740e-01f64,
    4.306281372884588343e-01f64, 4.259995411430343437e-01f64, 4.214287289976165751e-01f64,
    4.169141864330028757e-01f64, 4.124544659971611793e-01f64, 4.080481831520323954e-01f64,
    4.036940125305302773e-01f64, 3.993906844752310725e-01f64, 3.951369818332901573e-01f64,
    3.909317369847971069e-01f64, 3.867738290841376547e-01f64, 3.826621814960098344e-01f64,
    3.785957594095807899e-01f64, 3.745735676159021588e-01f64, 3.705946484351460013e-01f64,
    3.666580797815141568e-01f64, 3.627629733548177748e-01f64, 3.589084729487497794e-01f64,
    3.550937528667874599e-01f64, 3.513180164374833381e-01f64, 3.475804946216369817e-01f64,
    3.438804447045024082e-01f64, 3.402171490667800224e-01f64, 3.365899140286776059e-01f64,
    3.329980687618089852e-01f64, 3.294409642641363267e-01f64, 3.259179723935561879e-01f64,
    3.224284849560891675e-01f64, 3.189719128449572394e-01f64, 3.155476852271289490e-01f64,
    3.121552487741795501e-01f64, 3.087940669345601852e-01f64, 3.054636192445902565e-01f64,
    3.021634006756935276e-01f64, 2.988929210155817917e-01f64, 2.956517042812611962e-01f64,
    2.924392881618925744e-01f64, 2.892552234896777485e-01f64, 2.860990737370768255e-01f64,
    2.829704145387807457e-01f64, 2.798688332369729248e-01f64, 2.767939284485173568e-01f64,
    2.737453096528029706e-01f64, 2.707225967990600224e-01f64, 2.677254199320447947e-01f64,
    2.647534188350622042e-01f64, 2.618062426893629779e-01f64, 2.588835497490162285e-01f64,
    2.559850070304153791e-01f64, 2.531102900156294577e-01f64, 2.502590823688622956e-01f64,
    2.474310756653276266e-01f64, 2.446259691318921070e-01f64, 2.418434693988772144e-01f64,
    2.390832902624491774e-01f64, 2.363451524570596429e-01f64, 2.336287834374333461e-01f64,
    2.309339171696274118e-01f64, 2.282602939307167011e-01f64, 2.256076601166840667e-01f64,
    2.229757680581201940e-01f64, 2.203643758433594946e-01f64, 2.177732471487005272e-01f64,
    2.152021510753786837e-01f64, 2.126508619929782795e-01f64, 2.101191593889882581e-01f64,
    2.076068277242220372e-01f64, 2.051136562938377095e-01f64, 2.026394390937090173e-01f64,
    2.001839746919112650e-01f64, 1.977470661050988732e-01f64, 1.953285206795632167e-01f64,
    1.929281499767713515e-01f64, 1.905457696631953912e-01f64, 1.881811994042543179e-01f64,
    1.858342627621971110e-01f64, 1.835047870977674633e-01f64, 1.811926034754962889e-01f64,
    1.788975465724783054e-01f64, 1.766194545904948843e-01f64, 1.743581691713534942e-01f64,
    1.721135353153200598e-01f64, 1.698854013025276610e-01f64, 1.676736186172501919e-01f64,
    1.654780418749360049e-01f64, 1.632985287519018169e-01f64, 1.611349399175920349e-01f64,
    1.589871389693142123e-01f64, 1.568549923693652315e-01f64, 1.547383693844680830e-01f64,
    1.526371420274428570e-01f64, 1.505511850010398944e-01f64, 1.484803756438667910e-01f64,
    1.464245938783449441e-01f64, 1.443837221606347754e-01f64, 1.423576454324722018e-01f64,
    1.403462510748624548e-01f64, 1.383494288635802039e-01f64, 1.363670709264288572e-01f64,
    1.343990717022136294e-01f64, 1.324453279013875218e-01f64, 1.305057384683307731e-01f64,
    1.285802045452281717e-01f64, 1.266686294375106714e-01f64, 1.247709185808309612e-01f64,
    1.228869795095451356e-01f64, 1.210167218266748335e-01f64, 1.191600571753276827e-01f64,
    1.173168992115555670e-01f64, 1.154871635786335338e-01f64, 1.136707678827443141e-01f64,
    1.118676316700562973e-01f64, 1.100776764051853845e-01f64, 1.083008254510337970e-01f64,
    1.065370040500016602e-01f64, 1.047861393065701724e-01f64, 1.030481601712577161e-01f64,
    1.013229974259536315e-01f64, 9.961058367063713170e-02f64, 9.791085331149219917e-02f64,
    9.622374255043279756e-02f64, 9.454918937605585882e-02f64, 9.288713355604354127e-02f64,
    9.123751663104015530e-02f64, 8.960028191003285847e-02f64, 8.797537446727021759e-02f64,
    8.636274114075691288e-02f64, 8.476233053236811865e-02f64, 8.317409300963238272e-02f64,
    8.159798070923741931e-02f64, 8.003394754231990538e-02f64, 7.848194920160642130e-02f64,
    7.694194317048050347e-02f64, 7.541388873405840965e-02f64, 7.389774699236474620e-02f64,
    7.239348087570873780e-02f64, 7.090105516237182881e-02f64, 6.942043649872875477e-02f64,
    6.795159342193660135e-02f64, 6.649449638533977414e-02f64, 6.504911778675374900e-02f64,
    6.361543199980733421e-02f64, 6.219341540854099459e-02f64, 6.078304644547963265e-02f64,
    5.938430563342026597e-02f64, 5.799717563120065922e-02f64, 5.662164128374287675e-02f64,
    5.525768967669703741e-02f64, 5.390531019604608703e-02f64, 5.256449459307169225e-02f64,
    5.123523705512628146e-02f64, 4.991753428270637172e-02f64, 4.861138557337949667e-02f64,
    4.731679291318154762e-02f64, 4.603376107617516977e-02f64, 4.476229773294328196e-02f64,
    4.350241356888818328e-02f64, 4.225412241331623353e-02f64, 4.101744138041481941e-02f64,
    3.979239102337412542e-02f64, 3.857899550307485742e-02f64, 3.737728277295936097e-02f64,
    3.618728478193142251e-02f64, 3.500903769739741045e-02f64, 3.384258215087432992e-02f64,
    3.268796350895953468e-02f64, 3.154523217289360859e-02f64, 3.041444391046660423e-02f64,
    2.929566022463739317e-02f64, 2.818894876397863569e-02f64, 2.709438378095579969e-02f64,
    2.601204664513421735e-02f64, 2.494202641973178314e-02f64, 2.388442051155817078e-02f64,
    2.283933540638524023e-02f64, 2.180688750428358066e-02f64, 2.078720407257811723e-02f64,
    1.978042433800974303e-02f64, 1.878670074469603046e-02f64, 1.780620041091136169e-02f64,
    1.683910682603994777e-02f64, 1.588562183997316302e-02f64, 1.494596801169114850e-02f64,
    1.402039140318193759e-02f64, 1.310916493125499106e-02f64, 1.221259242625538123e-02f64,
    1.133101359783459695e-02f64, 1.046481018102997894e-02f64, 9.614413642502209895e-03f64,
    8.780314985808975251e-03f64, 7.963077438017040002e-03f64, 7.163353183634983863e-03f64,
    6.381905937319179087e-03f64, 5.619642207205483020e-03f64, 4.877655983542392333e-03f64,
    4.157295120833795314e-03f64, 3.460264777836904049e-03f64, 2.788798793574076128e-03f64,
    2.145967743718906265e-03f64, 1.536299780301572356e-03f64, 9.672692823271745359e-04f64,
    4.541343538414967652e-04f64,
];
const BP_ZIG_NOR_R: f64 = 3.6541528853610087963519472518f64;
const BP_ZIG_NOR_INV_R: f64 = 0.27366123732975827203338247596f64;
const BP_ZIG_EXP_R: f64 = 7.6971174701310497140446280481f64;

// ============================================================================
// Bit-exact port of the numpy 2.2.6 RNG chain used by iobrpy bayesprism's
// Gibbs sampler (Generator(MT19937(SeedSequence(123).spawn(n)[i]))):
//
//   * SeedSequence          — numpy/random/bit_generator.pyx (mix_entropy /
//                             generate_state cyclic-pool, verified to 624 words)
//   * MT19937 seeding quirk — numpy/random/_mt19937.pyx: key =
//                             generate_state(624, uint32); key[0] = 0x80000000;
//                             pos = 623 (first draw consumes key[623] BEFORE
//                             the first twist)
//   * mt19937.h/c           — twist / tempering / next_double (two-word
//                             ((w1>>5)*2^26+(w2>>6))/2^53) / next64 (hi first)
//   * distributions.c       — random_standard_exponential (ziggurat),
//                             random_standard_normal (ziggurat),
//                             random_standard_gamma (Marsaglia-Tsang),
//                             random_binomial (inversion + BTPE),
//                             random_multinomial (sequential conditional
//                             binomial chain)
//   * ziggurat_constants.h  — tables transcribed verbatim (zig_tables.rs)
//   * numpy pairwise summation (contiguous f64, loops_arithm_fp pairwise_sum)
//     for rdirichlet's np.sum(x); the prob_mat.sum(axis=0) column sums of a
//     C-contiguous (K,G) array are STRIDED reductions, which numpy accumulates
//     SEQUENTIALLY in k order (verified bitwise on adversarial inputs).
//
// The Generator-level persistent binomial_t cache of numpy is value-
// transparent (its fields are deterministic functions of (n,p)), so this port
// recomputes the setup on every call: identical draws, identical RNG
// consumption.
//
// No external crates; all math via Rust std -> glibc (log/exp/log1p/sqrt/floor
// == the same libm symbols numpy's scalar C code calls; the release target is
// baseline x86-64/SSE2, so LLVM cannot emit FMA contractions that would move
// the last bit).
// ============================================================================


// ---------------------------------------------------------------- MT19937
const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER: u32 = 0x8000_0000;
const MT_LOWER: u32 = 0x7fff_ffff;

struct BpMt19937 {
    key: Box<[u32; MT_N]>,
    pos: usize,
}

impl BpMt19937 {
    /// numpy Generator seeding: `MT19937(SeedSequence)` fills the state with
    /// `generate_state(624, uint32)`, forces key[0] = 0x80000000 and leaves
    /// pos = 623 (_mt19937.pyx __init__ — the leaked loop-variable quirk).
    fn from_seed_sequence(ss: &SeedSequence) -> Self {
        let val = ss.generate_state_u32(MT_N);
        let mut key = [0u32; MT_N];
        key[0] = 0x8000_0000;
        for i in 1..MT_N {
            key[i] = val[i];
        }
        BpMt19937 {
            key: Box::new(key),
            pos: MT_N - 1,
        }
    }

    /// mt19937_gen: full twist; sets pos = 0.
    #[inline]
    fn twist(&mut self) {
        let key = &mut self.key;
        for i in 0..MT_N - MT_M {
            let y = (key[i] & MT_UPPER) | (key[i + 1] & MT_LOWER);
            key[i] = key[i + MT_M] ^ (y >> 1) ^ (if y & 1 != 0 { MT_MATRIX_A } else { 0 });
        }
        for i in MT_N - MT_M..MT_N - 1 {
            let y = (key[i] & MT_UPPER) | (key[i + 1] & MT_LOWER);
            key[i] = key[i + MT_M - MT_N] ^ (y >> 1) ^ (if y & 1 != 0 { MT_MATRIX_A } else { 0 });
        }
        let y = (key[MT_N - 1] & MT_UPPER) | (key[0] & MT_LOWER);
        key[MT_N - 1] = key[MT_M - 1] ^ (y >> 1) ^ (if y & 1 != 0 { MT_MATRIX_A } else { 0 });
        self.pos = 0;
    }

    /// mt19937_next: draw + tempering.
    #[inline]
    fn next32(&mut self) -> u32 {
        if self.pos == MT_N {
            self.twist();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// mt19937_next64: high word drawn first.
    #[inline]
    fn next64(&mut self) -> u64 {
        ((self.next32() as u64) << 32) | (self.next32() as u64)
    }

    /// mt19937_next_double: classic two-word formula.
    #[inline]
    fn next_double(&mut self) -> f64 {
        let a = (self.next32() >> 5) as f64;
        let b = (self.next32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }
}

// ---------------------------------------------------------------- distributions
/// random_standard_exponential + standard_exponential_unlikely (distributions.c)
fn bp_std_exponential(mt: &mut BpMt19937) -> f64 {
    loop {
        let mut ri = mt.next64();
        ri >>= 3;
        let idx = (ri & 0xFF) as usize;
        ri >>= 8;
        let x = (ri as f64) * BP_WE_DOUBLE[idx];
        if ri < BP_KE_DOUBLE[idx] {
            return x; /* 98.9% of the time we return here 1st try */
        }
        // standard_exponential_unlikely
        if idx == 0 {
            /* Switch to 1.0 - U to avoid log(0.0), see GH 13361 */
            return BP_ZIG_EXP_R - (-mt.next_double()).ln_1p();
        } else if (BP_FE_DOUBLE[idx - 1] - BP_FE_DOUBLE[idx]) * mt.next_double() + BP_FE_DOUBLE[idx]
            < (-x).exp()
        {
            return x;
        }
        // else: fall through, draw again (C tail recursion)
    }
}

/// random_standard_normal (distributions.c, ziggurat; r = e3n52sb8)
fn bp_std_normal(mt: &mut BpMt19937) -> f64 {
    loop {
        let r = mt.next64();
        let idx = (r & 0xff) as usize;
        let r = r >> 8;
        let sign = r & 0x1;
        let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
        let mut x = (rabs as f64) * BP_WI_DOUBLE[idx];
        if sign & 0x1 != 0 {
            x = -x;
        }
        if rabs < BP_KI_DOUBLE[idx] {
            return x; /* 99.3% of the time return here */
        }
        if idx == 0 {
            loop {
                /* Switch to 1.0 - U to avoid log(0.0), see GH 13361 */
                let xx = -BP_ZIG_NOR_INV_R * (-mt.next_double()).ln_1p();
                let yy = -(-mt.next_double()).ln_1p();
                if yy + yy > xx * xx {
                    return if ((rabs >> 8) & 0x1) != 0 {
                        -(BP_ZIG_NOR_R + xx)
                    } else {
                        BP_ZIG_NOR_R + xx
                    };
                }
            }
        } else if (BP_FI_DOUBLE[idx - 1] - BP_FI_DOUBLE[idx]) * mt.next_double() + BP_FI_DOUBLE[idx]
            < (-0.5 * x * x).exp()
        {
            return x;
        }
    }
}

/// random_standard_gamma (distributions.c). shape >= 0 (Generator check).
fn bp_std_gamma(mt: &mut BpMt19937, shape: f64) -> f64 {
    if shape == 1.0 {
        return bp_std_exponential(mt);
    } else if shape == 0.0 {
        return 0.0;
    } else if shape < 1.0 {
        loop {
            let u = mt.next_double();
            let v = bp_std_exponential(mt);
            if u <= 1.0 - shape {
                let x = u.powf(1.0 / shape);
                if x <= v {
                    return x;
                }
            } else {
                let y = -((1.0 - u) / shape).ln();
                let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                if x <= v + y {
                    return x;
                }
            }
        }
    } else {
        let b = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * b).sqrt();
        loop {
            let mut x;
            let mut v;
            loop {
                x = bp_std_normal(mt);
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }            let v = v * v * v;
            let u = mt.next_double();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return b * v;
            }
            /* log(0.0) ok here */
            if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
                return b * v;
            }
        }
    }
}

/// random_gamma: scale * random_standard_gamma (scale = 1.0 in rdirichlet).
#[inline]
fn bp_gamma_scaled(mt: &mut BpMt19937, shape: f64, scale: f64) -> f64 {
    scale * bp_std_gamma(mt, shape)
}

/// random_binomial_btpe (distributions.c) — cache-free (see file header).
fn bp_binomial_btpe(mt: &mut BpMt19937, n: i64, p: f64) -> i64 {
    // ---- setup (deterministic function of (n, p)) ----
    let r = if p < 1.0 - p { p } else { 1.0 - p }; // MIN(p, 1.0 - p)
    let q = 1.0 - r;
    let fm = n as f64 * r + r;
    let m = fm.floor() as i64;
    let p1 = (2.195 * (n as f64 * r * q).sqrt() - 4.6 * q).floor() + 0.5;
    let xm = m as f64 + 0.5;
    let xl = xm - p1;
    let xr = xm + p1;
    let c = 0.134 + 20.5 / (15.3 + m as f64);
    let mut a = (fm - xl) / (fm - xl * r);
    let laml = a * (1.0 + a / 2.0);
    a = (xr - fm) / (xr * q);
    let lamr = a * (1.0 + a / 2.0);
    let p2 = p1 * (1.0 + 2.0 * c);
    let p3 = p2 + c / laml;
    let p4 = p3 + c / lamr;

    // ---- generating loop (Step10..Step60) ----
    loop {
        // Step10
        let nrq = n as f64 * r * q;
        let u = mt.next_double() * p4;
        let mut v = mt.next_double();
        let y: i64;
        if u <= p1 {
            y = (xm - p1 * v + u).floor() as i64;
            // Step60
            return if p > 0.5 { n - y } else { y };
        }
        // Step20
        if u <= p2 {
            let x = xl + (u - p1) / c;
            v = v * c + 1.0 - (m as f64 - x + 0.5).abs() / p1;
            if v > 1.0 {
                continue; // goto Step10
            }
            y = x.floor() as i64;
        } else if u <= p3 {
            // Step30 (left tail)
            y = (xl + v.ln() / laml).floor() as i64;
            /* Reject if v==0.0 since previous cast is undefined */
            if (y < 0) || (v == 0.0) {
                continue; // goto Step10
            }
            v = v * (u - p2) * laml;
        } else {
            // Step40 (right tail)
            y = (xr - v.ln() / lamr).floor() as i64;
            /* Reject if v==0.0 since previous cast is undefined */
            if (y > n) || (v == 0.0) {
                continue; // goto Step10
            }
            v = v * (u - p3) * lamr;
        }
        // Step50: determine if y is a candidate
        let k = (y - m).abs();
        if (k > 20) && ((k as f64) < nrq / 2.0 - 1.0) {
            // Step52 (Stirling bounds)
            let rho =
                (k as f64 / nrq) * ((k as f64 * (k as f64 / 3.0 + 0.625) + 0.16666666666666666)
                    / nrq
                    + 0.5);
            let t = ((-k) * k) as f64 / (2.0 * nrq);
            /* log(0.0) ok here */
            let big_a = v.ln();
            if big_a < t - rho {
                // Step60
                return if p > 0.5 { n - y } else { y };
            }
            if big_a > t + rho {
                continue; // goto Step10
            }
            let x1 = (y + 1) as f64;
            let f1 = (m + 1) as f64;
            let z = (n + 1 - m) as f64;
            let w = (n - y + 1) as f64;
            let x2 = x1 * x1;
            let f2 = f1 * f1;
            let z2 = z * z;
            let w2 = w * w;
            if big_a
                > xm * (f1 / x1).ln()
                    + ((n - m) as f64 + 0.5) * (z / w).ln()
                    + ((y - m) as f64) * (w * r / (x1 * q)).ln()
                    + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2)
                        / f1
                        / 166320.0
                    + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2)
                        / z
                        / 166320.0
                    + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2)
                        / x1
                        / 166320.0
                    + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2)
                        / w
                        / 166320.0
            {
                continue; // goto Step10
            }
            // Step60
            return if p > 0.5 { n - y } else { y };
        }
        // Step50 direct evaluation (k <= 20 or k >= nrq/2 - 1)
        let s = r / q;
        let a = s * (n as f64 + 1.0);
        let mut f = 1.0f64;
        if m < y {
            let mut i = m + 1;
            while i <= y {
                f *= a / i as f64 - s;
                i += 1;
            }
        } else if m > y {
            let mut i = y + 1;
            while i <= m {
                f /= a / i as f64 - s;
                i += 1;
            }
        }
        if v > f {
            continue; // goto Step10
        }
        // Step60
        return if p > 0.5 { n - y } else { y };
    }
}

/// random_binomial_inversion (distributions.c) — cache-free.
fn bp_binomial_inversion(mt: &mut BpMt19937, n: i64, p: f64) -> i64 {
    let q = 1.0 - p;
    let qn = (n as f64 * q.ln()).exp();
    let np = n as f64 * p;
    let bound = {
        let expr = np + 10.0 * (np * q + 1.0).sqrt();
        let mm = if (n as f64) < expr { n as f64 } else { expr };
        mm as i64
    };
    let mut x = 0i64;
    let mut px = qn;
    let mut u = mt.next_double();
    while u > px {
        x += 1;
        if x > bound {
            x = 0;
            px = qn;
            u = mt.next_double();
        } else {
            u -= px;
            px = ((n - x + 1) as f64 * p * px) / (x as f64 * q);
        }
    }
    x
}

/// random_binomial (distributions.c dispatcher; argument order (p, n) as in C).
fn bp_random_binomial(mt: &mut BpMt19937, p: f64, n: i64) -> i64 {
    if (n == 0) || (p == 0.0) {
        return 0;
    }
    if p <= 0.5 {
        if p * n as f64 <= 30.0 {
            bp_binomial_inversion(mt, n, p)
        } else {
            bp_binomial_btpe(mt, n, p)
        }
    } else {
        let q = 1.0 - p;
        if q * n as f64 <= 30.0 {
            n - bp_binomial_inversion(mt, n, q)
        } else {
            n - bp_binomial_btpe(mt, n, q)
        }
    }
}

/// random_multinomial (distributions.c): sequential conditional binomial
/// chain; `mnix` must be pre-zeroed (the pyx wrapper allocates np.zeros).
fn bp_random_multinomial(mt: &mut BpMt19937, n: i64, pix: &[f64], mnix: &mut [i64]) {
    let d = pix.len();
    let mut remaining_p = 1.0f64;
    let mut dn = n;
    for j in 0..d - 1 {
        mnix[j] = bp_random_binomial(mt, pix[j] / remaining_p, dn);
        dn -= mnix[j];
        if dn <= 0 {
            break;
        }
        remaining_p -= pix[j];
    }
    if dn > 0 {
        mnix[d - 1] = dn;
    }
}

/// _generator.pyx multinomial wrapper validation (CONS_BOUNDED_0_1 +
/// bp_kahan_sum(pix, d-1) <= 1 + 1e-12). Returns Err(message) like numpy.
fn bp_multinomial_validate(pix: &[f64]) -> Result<(), String> {
    for &x in pix.iter() {
        if !(x >= 0.0 && x <= 1.0) {
            return Err("pvals < 0, pvals > 1 or pvals contains NaNs".to_string());
        }
    }
    let d = pix.len();
    if bp_kahan_sum(pix, d - 1) > 1.0 + 1e-12 {
        return Err("sum(pvals[:-1]) > 1.0".to_string());
    }
    Ok(())
}

/// _common.pyx kahan_sum (compensated summation used by the validation only).
fn bp_kahan_sum(arr: &[f64], n: usize) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for i in 0..n {
        let y = arr[i] - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

// ---------------------------------------------------------------- numpy sums
// ---------------------------------------------------------------- Gibbs chain
struct BpGibbsOut {
    pub z: Vec<f64>,      // (G,K) row-major; empty when !accum_z
    pub theta: Vec<f64>,  // (K,)
    pub theta_cv: Vec<f64>, // (K,)
}

/// One bulk sample's Gibbs chain — sample_Z_theta_n (accum_z = true) /
/// sample_theta_n (accum_z = false) of gibbs.py, bit-exact.
///
/// phi: (K,G) row-major C-contiguous float64 (`phi.to_numpy()`);
/// x_n: (G,) int64 counts (`int(n)` of the int32 row);
/// alpha: gibbs_control['alpha'] as f64 (shape = (double)count + alpha);
/// keep-set: iterations i with i >= burn_in_int && (i - burn_in_int) %
/// thinning == 0 (== `np.arange(chain)[int(burn)::thinning]` membership).
fn bp_gibbs_chain(
    mt: &mut BpMt19937,
    phi: &[f64],
    k: usize,
    g: usize,
    x_n: &[i64],
    alpha: f64,
    chain_length: i64,
    burn_in: i64,
    thinning: i64,
    accum_z: bool,
) -> Result<BpGibbsOut, String> {
    let burn_i = burn_in as usize; // int(burn_in): truncation of the float control value
    let mut theta = vec![1.0f64 / k as f64; k]; // np.repeat(1 / K, K)
    let mut p = vec![0.0f64; k * g];
    let mut colpix = vec![0.0f64; k];
    let mut z_row = vec![0i64; k];
    let mut z_nk = vec![0i64; k];
    let mut z_sum = vec![0.0f64; g * k];
    let mut th_sum = vec![0.0f64; k];
    let mut th2_sum = vec![0.0f64; k];
    let mut draws = vec![0.0f64; k];
    let mut n_keep = 0usize;

    for i in 0..chain_length {
        // prob_mat = phi * theta_n_i[:, np.newaxis]
        for kk in 0..k {
            let t = theta[kk];
            let base = kk * g;
            for gg in 0..g {
                p[base + gg] = phi[base + gg] * t;
            }
        }
        // prob_mat /= prob_mat.sum(axis=0, keepdims=True)
        // strided reduction over axis 0 of a C-contiguous (K,G) array:
        // numpy accumulates SEQUENTIALLY in k order (verified bitwise).
        for gg in 0..g {
            let mut s = 0.0f64;
            for kk in 0..k {
                s += p[kk * g + gg];
            }
            for kk in 0..k {
                p[kk * g + gg] /= s;
            }
        }
        // Z_n_i = per-gene rng.multinomial(X_n[g], prob_mat[:, g]) ("sequential")
        for kk in 0..k {
            z_nk[kk] = 0;
        }
        let keep = i as usize >= burn_i
            && thinning > 0
            && ((i as usize - burn_i) % thinning as usize == 0);
        for gg in 0..g {
            for kk in 0..k {
                colpix[kk] = p[kk * g + gg];
            }
            bp_multinomial_validate(&colpix[..k])?;
            for kk in 0..k {
                z_row[kk] = 0; // np.zeros((d,), int64) in the wrapper
            }
            bp_random_multinomial(mt, x_n[gg], &colpix[..k], &mut z_row[..k]);
            for kk in 0..k {
                z_nk[kk] += z_row[kk];
                if keep && accum_z {
                    z_sum[gg * k + kk] += z_row[kk] as f64;
                }
            }
        }
        // theta_n_i = rdirichlet(Z_nk + alpha): K gammas, then x / np.sum(x)
        for kk in 0..k {
            if !(z_nk[kk] as f64 + alpha >= 0.0) {
                return Err("shape < 0".to_string()); // CONS_NON_NEGATIVE
            }
            draws[kk] = bp_gamma_scaled(mt, z_nk[kk] as f64 + alpha, 1.0);
        }
        let s = np_sum_f64(&draws[..k]);
        for kk in 0..k {
            theta[kk] = draws[kk] / s;
        }
        // retention (`if i in gibbs_idx`)
        if keep {
            n_keep += 1;
            for kk in 0..k {
                th_sum[kk] += theta[kk];
                th2_sum[kk] += theta[kk] * theta[kk];
            }
        }
    }

    let samples = n_keep as f64; // len(gibbs_idx)
    let mut theta_out = vec![0.0f64; k];
    let mut theta_cv = vec![0.0f64; k];
    for kk in 0..k {
        let tn = th_sum[kk] / samples;
        theta_out[kk] = tn;
        // np.sqrt(theta_n2_sum / samples_size - (theta_n ** 2)) / theta_n
        theta_cv[kk] = (th2_sum[kk] / samples - tn * tn).sqrt() / tn;
    }
    let mut z_out = Vec::new();
    if accum_z {
        z_out = vec![0.0f64; g * k];
        for idx in 0..g * k {
            z_out[idx] = z_sum[idx] / samples;
        }
    }
    Ok(BpGibbsOut {
        z: z_out,
        theta: theta_out,
        theta_cv,
    })
}

// ------------------------------------------------------------------
// Python-facing BayesPrism Gibbs kernel
// ------------------------------------------------------------------

/// Batched BayesPrism Gibbs phase: runs the per-sample chains of gibbs.py
/// (sample_Z_theta_n when accum_z, else sample_theta_n) for ALL samples,
/// rayon-parallel and order-preserving (worker count cannot change bits:
/// seeds are assigned per sample index).
///
/// Seed semantics (upstream quirks preserved):
///   seed_mode 0: sample i uses SeedSequence(seed).spawn(n)[i]  (phase 1/2)
///   seed_mode 1: EVERY sample uses SeedSequence(seed).spawn(1)[0]
///                (the phase-3 re-spawn quirk, gibbs.py ~L393)
///   seed_mode 2: SeedSequence(seed) root itself (probe chains: seed used
///                directly, NOT spawned)
///
/// phi: shared (K,G) float64 C-contiguous (`phi.to_numpy()`); x: (n,G)
/// int64 C-contiguous counts. phi_per (phase 3): (n*K,G) float64 rows =
/// per-sample concat([psi_mal_i, psi_env]); trim: (n,G) bool per-sample
/// gene mask (`np.max(phi_n, axis=0) > 0`); when both are given each chain
/// runs on the mask-selected columns in ascending order — exactly
/// `phi_n.loc[:, nonzero_idx]` + `X[i, nonzero_idx]`.
/// burn_in/thinning reproduce `if i in gibbs_idx` with
/// gibbs_idx = arange(chain_length)[int(burn_in)::thinning].
///
/// Returns (z_flat, theta_flat, theta_cv_flat): z_flat is (n*G*K,) row-major
/// per-sample (G,K) blocks when accum_z (empty otherwise); theta/cv are
/// (n*K,) row-major per-sample (K,) rows. Errors carry the numpy wrapper
/// messages ("pvals < 0, pvals > 1 or pvals contains NaNs", ...).
#[pyfunction]
#[pyo3(name = "bp_gibbs_phase", signature = (phi, x, alpha, chain_length, burn_in, thinning, seed, seed_mode, accum_z, n_threads, phi_per=None, trim=None))]
#[allow(clippy::too_many_arguments)]
fn bp_gibbs_phase_py(
    py: Python,
    phi: PyReadonlyArray2<f64>,
    x: PyReadonlyArray2<i64>,
    alpha: f64,
    chain_length: i64,
    burn_in: i64,
    thinning: i64,
    seed: u64,
    seed_mode: i64,
    accum_z: bool,
    n_threads: u32,
    phi_per: Option<PyReadonlyArray2<f64>>,
    trim: Option<PyReadonlyArray2<bool>>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let phi_s = phi
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("phi must be C-contiguous float64"))?;
    let (k, g_all) = (phi.shape()[0], phi.shape()[1]);
    let x_s = x
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("x must be C-contiguous int64"))?;
    let (n, g) = (x.shape()[0], x.shape()[1]);
    if g != g_all {
        return Err(PyValueError::new_err("x and phi gene dimensions differ"));
    }
    if k == 0 || n == 0 || g_all == 0 {
        return Err(PyValueError::new_err("empty phi/x"));
    }
    if thinning < 1 {
        return Err(PyValueError::new_err("thinning must be >= 1"));
    }
    if chain_length < 1 {
        return Err(PyValueError::new_err("chain.length must be >= 1"));
    }
    let per_s: Option<&[f64]> = match &phi_per {
        Some(p) => {
            let s = p
                .as_slice()
                .map_err(|_| PyRuntimeError::new_err("phi_per must be C-contiguous float64"))?;
            if s.len() != n * k * g_all {
                return Err(PyValueError::new_err("phi_per must hold n*K*G values"));
            }
            Some(s)
        }
        None => None,
    };
    let trim_s: Option<&[bool]> = match &trim {
        Some(t) => {
            let s = t
                .as_slice()
                .map_err(|_| PyRuntimeError::new_err("trim must be C-contiguous bool"))?;
            if s.len() != n * g_all {
                return Err(PyValueError::new_err("trim must hold n*G values"));
            }
            Some(s)
        }
        None => None,
    };
    if per_s.is_some() != trim_s.is_some() {
        return Err(PyValueError::new_err("phi_per and trim must be given together"));
    }
    if accum_z && per_s.is_some() {
        return Err(PyValueError::new_err(
            "accum_z with per-sample phi is not used by the pipeline",
        ));
    }

    let n_threads_eff = if n_threads == 0 {
        rayon::current_num_threads()
    } else {
        n_threads as usize
    };
    let pool = if n_threads_eff > 1 && n > 1 {
        Some(global_pool(n_threads_eff).map_err(PyRuntimeError::new_err)?)
    } else {
        None
    };

    let results = py.allow_threads(move || {
        let run_one = |i: usize| -> Result<BpGibbsOut, String> {
            let root = SeedSequence::new_root(seed);
            let ss = match seed_mode {
                0 => root.child(i as u64),
                1 => root.child(0),
                _ => root,
            };
            let mut mt = BpMt19937::from_seed_sequence(&ss);
            match (per_s, trim_s) {
                (Some(per), Some(tr)) => {
                    let mut gi = 0usize;
                    for j in 0..g_all {
                        if tr[i * g_all + j] {
                            gi += 1;
                        }
                    }
                    let mut phis = Vec::with_capacity(k * gi);
                    for kk in 0..k {
                        let row = &per[(i * k + kk) * g_all..(i * k + kk + 1) * g_all];
                        for j in 0..g_all {
                            if tr[i * g_all + j] {
                                phis.push(row[j]);
                            }
                        }
                    }
                    let mut xs = Vec::with_capacity(gi);
                    for j in 0..g_all {
                        if tr[i * g_all + j] {
                            xs.push(x_s[i * g_all + j]);
                        }
                    }
                    bp_gibbs_chain(
                        &mut mt, &phis, k, gi, &xs, alpha, chain_length, burn_in, thinning,
                        accum_z,
                    )
                }
                _ => {
                    let xrow = &x_s[i * g_all..(i + 1) * g_all];
                    bp_gibbs_chain(
                        &mut mt, phi_s, k, g_all, xrow, alpha, chain_length, burn_in, thinning,
                        accum_z,
                    )
                }
            }
        };
        match &pool {
            Some(pl) => pl.install(|| (0..n).into_par_iter().map(run_one).collect::<Vec<_>>()),
            None => (0..n).map(run_one).collect::<Vec<_>>(),
        }
    });

    let mut z_flat: Vec<f64> = Vec::new();
    let mut theta_flat: Vec<f64> = Vec::with_capacity(n * k);
    let mut cv_flat: Vec<f64> = Vec::with_capacity(n * k);
    for r in results {
        let out = r.map_err(PyValueError::new_err)?;
        if accum_z {
            z_flat.extend_from_slice(&out.z);
        }
        theta_flat.extend_from_slice(&out.theta);
        cv_flat.extend_from_slice(&out.theta_cv);
    }
    Ok((
        z_flat.into_pyarray_bound(py).unbind(),
        theta_flat.into_pyarray_bound(py).unbind(),
        cv_flat.into_pyarray_bound(py).unbind(),
    ))
}

/// Generator(MT19937(SeedSequence(seed) or its spawn child)).random(n) raw
/// stream parity helper for tests (child < 0 -> root, no spawn).
#[pyfunction]
#[pyo3(name = "bp_rng_doubles")]
fn bp_rng_doubles_py(py: Python, seed: u64, child: i64, n: usize) -> Py<PyArray1<f64>> {
    let root = SeedSequence::new_root(seed);
    let ss = if child < 0 { root } else { root.child(child as u64) };
    let mut mt = BpMt19937::from_seed_sequence(&ss);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(mt.next_double());
    }
    v.into_pyarray_bound(py).unbind()
}

/// SeedSequence(seed)[.spawn(child+1)[child]].generate_state(624, uint32):
/// the MT19937 state words BEFORE the key[0] = 0x80000000 override
/// (test parity helper).
#[pyfunction]
#[pyo3(name = "bp_seedseq_state624")]
fn bp_seedseq_state624_py(py: Python, seed: u64, child: i64) -> Py<PyArray1<u32>> {
    let root = SeedSequence::new_root(seed);
    let ss = if child < 0 { root } else { root.child(child as u64) };
    ss.generate_state_u32(624).into_pyarray_bound(py).unbind()
}


#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_blas_py, m)?)?;
    m.add_function(wrap_pyfunction!(blas_ready, m)?)?;
    m.add_function(wrap_pyfunction!(fit_nusvr_linear_py, m)?)?;
    m.add_function(wrap_pyfunction!(argsort_f64_np_py, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_normalize_np_py, m)?)?;
    m.add_function(wrap_pyfunction!(rng_integers_np_py, m)?)?;
    m.add_function(wrap_pyfunction!(seedsequence_spawn_seeds_py, m)?)?;
    m.add_function(wrap_pyfunction!(cibersort_core_py, m)?)?;
    m.add_function(wrap_pyfunction!(csv_parse_roundtrip_py, m)?)?;
    m.add_function(wrap_pyfunction!(csv_repr_token_py, m)?)?;
    m.add_function(wrap_pyfunction!(ssgsea_core, m)?)?;
    m.add_function(wrap_pyfunction!(pca_pc1_py, m)?)?;
    m.add_function(wrap_pyfunction!(pca_zscore_intermediate_py, m)?)?;
    m.add_function(wrap_pyfunction!(rows_colmean_pairwise_py, m)?)?;
    m.add_function(wrap_pyfunction!(lr_gene_valid_mask_py, m)?)?;
    m.add_function(wrap_pyfunction!(tme_kmeans_best_py, m)?)?;
    m.add_function(wrap_pyfunction!(tme_kl_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(tme_missing_order_py, m)?)?;
    m.add_function(wrap_pyfunction!(merge_salmon_parse_py, m)?)?;
    m.add_function(wrap_pyfunction!(bp_gibbs_phase_py, m)?)?;
    m.add_function(wrap_pyfunction!(bp_rng_doubles_py, m)?)?;
    m.add_function(wrap_pyfunction!(bp_seedseq_state624_py, m)?)?;
    Ok(())
}
