// Dense-only libsvm build, mirroring sklearn's libsvm_template.cpp
// (which includes svm.cpp twice: dense first, then sparse). We only need
// the dense symbol set (svm_train etc. with struct svm_node{dim,ind,values}).
// svm.cpp / svm.h come from iobrx/vendor (byte-identical to sklearn 1.7.2's
// sklearn/svm/src/libsvm/svm.cpp|h). The two remaining headers it includes
// (_svm_cython_blas_helpers.h, ../newrand/newrand.h) are taken from the
// sklearn source tree via -I flags in build.rs.
//
// The extern "C" surface below exposes exactly what sklearn's
// sklearn/svm/_libsvm.pyx::fit() does for NuSVR(kernel='linear'):
//   - problem: y as given, W = ones(l) (sample_weight=None -> np.ones)
//   - parameter: svm_type=NU_SVR, kernel_type=LINEAR, degree=3, gamma=0
//     (unused for LINEAR), coef0=0, cache_size, eps=tol, C, nu, p=0.0
//     (NuSVR hardcodes epsilon=0.0), shrinking, probability=0, nr_weight=0,
//     weight_label=NULL, weight=NULL, max_iter=-1, random_seed=0
//     (RNG is never consumed on the NU_SVR solve path; bounded_rand_int is
//     only reached from probability calibration / cross-validation folds).
//   - kernel dot: sklearn passes BlasFunctions.dot = scipy's bundled
//     openblas ddot (LP64, symbol scipy_ddot_). We route the same function
//     pointer in via iobrx_set_dot; a naive sequential loop is used only
//     as an un-initialized fallback (NOT bit-exact vs sklearn).
#define _DENSE_REP
#include "svm.cpp"

#include <string.h>

extern "C" {

static void iobrx_print_noop(const char *) {}

typedef double (*fortran_ddot_t)(const int *, const double *, const int *,
                                 const double *, const int *);
static fortran_ddot_t g_fddot = NULL;

/* Adapter: libsvm's BlasFunctions.dot signature (by-value ints) ->
 * Fortran LP64 ddot_ (by-pointer ints). */
static double iobrx_dot_thunk(int n, const double *x, int incx,
                              const double *y, int incy) {
  if (g_fddot != NULL) {
    return g_fddot(&n, x, &incx, y, &incy);
  }
  /* fallback: naive sequential (deterministic but != openblas FMA kernels) */
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += x[(size_t)i * (size_t)incx] * y[(size_t)i * (size_t)incy];
  return s;
}

void iobrx_set_dot(void *fn) { g_fddot = (fortran_ddot_t)fn; }

int iobrx_svm_ready() { return g_fddot != NULL; }

/* Silence libsvm's stdout printing (sklearn routes it through print_null). */
void iobrx_svm_init() {
  svm_set_print_string_function(&iobrx_print_noop);
}

/*
 * Fit linear NuSVR. X is row-major l x dim, C-contiguous float64 (identical
 * to what check_array(dtype=float64, order='C') produces in sklearn).
 * out_dual_coef must have capacity l; out_sv_ind capacity l.
 * Returns nSV (>0), or -1 on allocation/training failure.
 * fit_status is returned through *out_fit_status (0 = converged).
 */
int iobrx_nusvr_linear_fit(const double *X, int l, int dim, const double *y,
                           double nu, double C, double tol, double cache_mb,
                           int shrinking, double *out_dual_coef,
                           int *out_sv_ind, int *out_n_iter,
                           int *out_fit_status) {
  struct svm_node *nodes = (svm_node *)malloc(sizeof(svm_node) * (size_t)l);
  double *W = (double *)malloc(sizeof(double) * (size_t)l);
  if (nodes == NULL || W == NULL) {
    free(nodes);
    free(W);
    return -1;
  }
  for (int i = 0; i < l; ++i) {
    nodes[i].dim = dim;
    nodes[i].ind = i;
    nodes[i].values = const_cast<double *>(X + (size_t)i * (size_t)dim);
    W[i] = 1.0;
  }

  struct svm_problem prob;
  memset(&prob, 0, sizeof(prob));
  prob.l = l;
  prob.y = const_cast<double *>(y);
  prob.x = nodes;
  prob.W = W;

  struct svm_parameter param;
  memset(&param, 0, sizeof(param));
  param.svm_type = NU_SVR;
  param.kernel_type = LINEAR;
  param.degree = 3;
  param.gamma = 0.0; /* unused for LINEAR kernel */
  param.coef0 = 0.0;
  param.cache_size = cache_mb;
  param.eps = tol;
  param.C = C;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
  param.nu = nu;
  param.p = 0.0; /* NuSVR hardcodes epsilon=0.0; unused by NU_SVR solver */
  param.shrinking = shrinking;
  param.probability = 0;
  param.max_iter = -1;
  param.random_seed = 0; /* never consumed on the NU_SVR path */

  const char *err = svm_check_parameter(&prob, &param);
  if (err != NULL) {
    free(nodes);
    free(W);
    return -2;
  }

  BlasFunctions bf;
  bf.dot = &iobrx_dot_thunk;

  int status = 0;
  struct svm_model *model = svm_train(&prob, &param, &status, &bf);
  free(nodes);
  free(W);
  if (model == NULL) {
    return -3;
  }
  int nsv = model->l;
  memcpy(out_dual_coef, model->sv_coef[0], sizeof(double) * (size_t)nsv);
  for (int i = 0; i < nsv; ++i)
    out_sv_ind[i] = model->sv_ind[i];
  *out_n_iter = model->n_iter ? model->n_iter[0] : -1;
  *out_fit_status = status;
  svm_free_and_destroy_model(&model);
  return nsv;
}
}
