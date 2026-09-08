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
    Ok(())
}
