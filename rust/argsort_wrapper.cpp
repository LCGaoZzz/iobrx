// numpy 2.x dispatches float64 argsort (kind='quicksort') to the vendored
// x86-simd-sort AVX512-SKX kernel whenever the CPU exposes AVX512F+DQ+VL
// (numpy/_core/meson.build: x86_simd_argsort.dispatch.cpp with
// dispatch [AVX512_SKX, AVX2]). This TU compiles the same vendored sources
// (pinned at numpy's submodule commit 9a1b616d) with the SKX feature set so
// the produced permutation is bit-identical to np.argsort on this machine.
// The argsort kernel sorts `arr` in place (numpy passes a scratch copy) and
// expects `arg` pre-filled with iota; we do both here.
#include "x86-simd-sort/src/x86simdsort-static-incl.h"
#include <stddef.h>

extern "C" {
void iobrx_argsort_f64(double *arr_scratch, size_t *arg, size_t n) {
  for (size_t i = 0; i < n; ++i)
    arg[i] = i;
  if (n > 1) {
    x86simdsortStatic::argsort(arr_scratch, arg, n, /*hasnan=*/true);
  }
}
}
