"""Native (Rust) Gibbs kernel lane for ``iobrx.bayesprism`` (backend='rust').

Replaces ONLY the two parallel Gibbs phases (``run_gibbs_refPhi`` /
``run_gibbs_refTumor``) of the bit-exact fast-Python lane with batched calls
into ``iobrx._rust.bp_gibbs_phase`` — a line-by-line port of the numpy 2.2.6
RNG chain that drives the ORIGINAL sampler (verified bit-exact on 12.5M gold
stream elements, incl. full 1000-iteration chains against gold gibbs.py):

  * ``SeedSequence(123).spawn(n_samples)[i]`` per-sample child streams
    (phase 1/2) and the phase-3 quirk that EVERY sample re-spawns
    ``spawn(1)[0]`` (same child seed for all samples);
  * ``Generator(MT19937(ss))`` seeding: ``generate_state(624, uint32)``
    direct fill, ``key[0] = 0x80000000``, ``pos = 623`` (the first draw
    consumes ``key[623]`` BEFORE the first twist);
  * ``next_double = ((w1>>5)*2^26 + (w2>>6)) / 2^53`` (classic two-word),
    ``next_uint64 = hi<<32 | lo`` (high word first);
  * per-gene sequential ``rng.multinomial`` = conditional-binomial chain
    (distributions.c ``random_multinomial``), ``rng.binomial`` = inversion +
    BTPE with every goto branch and Stirling bound, ``rng.gamma`` =
    Marsaglia-Tsang over the ziggurat standard normal (+ the ziggurat
    exponential for shape == 1.0 exactly), rdirichlet normalization with
    numpy's contiguous pairwise summation, ``prob_mat.sum(axis=0)`` with
    numpy's SEQUENTIAL strided column accumulation, and the
    ``if i in gibbs_idx`` retention set (as an O(1) index test).

Everything else stays on the verified fast-Python lane: preprocessing,
``PrismFast``, the ORIGINAL iobrpy ``optim`` MAP update (scipy CG — untouched,
its ULP-level iteration path feeds phase 3), ``JointPostFast.merge_K``,
``ThetaPost``, ``extract`` and the output contract. The ``estimate_gibbs_time``
probe chains also stay in Python (their result is discarded; ~20 ms).

Integration: ``iobrx.bayesprism(..., backend='rust')`` dispatches to
:func:`bayesprism_rust`, which temporarily rebinds
``bayesprism_fast.GibbsSamplerFast`` to :class:`GibbsSamplerRust` (every call
site in ``bayesprism_fast`` resolves the sampler through that module global,
including ``PrismFast.run``'s instantiation and ``run()``'s dispatch) and
restores it on exit. The rebind is process-global for the duration of the
call; concurrent ``bayesprism`` calls in the SAME process from multiple
threads should not mix lanes (subprocess/serial use is unaffected).

Non-default Gibbs configurations fall back to the fast-Python lane with
identical semantics (documented per guard in ``_kernel_or_none``):
``fast.multinomial=True`` (different RNG stream), ``rng.backend='randomstate'``
(legacy RandomState streams are not ported), ``compute_elbo=True`` (the ELBO
log-factorial accumulation), a non-int / None / out-of-range ``seed``, and
``chain.length=None``. The kernel ports the DEFAULT path only:
sequential multinomial + generator backend + integer seed — the path the CLI
and ``Prism.run`` defaults take.
"""
from __future__ import annotations

import numpy as np

from iobrx._fast import bayesprism_fast as _bpf

__all__ = ["GibbsSamplerRust", "bayesprism_rust", "kernel_available"]

_MAX_RUST_THREADS = 16  # campaign machine guard; outputs are thread-count invariant


def _native():
    from iobrx import _backend

    if not _backend.select_backend("rust"):
        raise RuntimeError(
            "Rust backend requested but the native extension is unavailable "
            f"({_backend._native_error})"
        )
    native = _backend._native
    if not hasattr(native, "bp_gibbs_phase"):
        raise RuntimeError(
            "iobrx._rust lacks bp_gibbs_phase — rebuild the extension with the "
            "BayesPrism Gibbs kernel block (see INTEGRATION.md)"
        )
    return native


def kernel_available() -> bool:
    """True when the native Gibbs kernel can be used in this process."""
    try:
        _native()
        return True
    except RuntimeError:
        return False


def _kernel_or_none(gibbs_control: dict, compute_elbo: bool):
    """Extract kernel parameters from gibbs_control, or None -> Python lane.

    Mirrors the ORIGINAL defaults (chain.length=1000, burn.in=500, thinning=2,
    seed=123, alpha=1, fast.multinomial=False, rng.backend='generator').
    """
    if compute_elbo:
        return None
    if gibbs_control.get("fast.multinomial", False):
        return None
    if str(gibbs_control.get("rng.backend", "generator")).lower() != "generator":
        return None
    seed = gibbs_control.get("seed", 123)
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None
    if not (0 <= seed < 2**64):
        return None
    chain_length = gibbs_control.get("chain.length", 1000)
    burn_in = gibbs_control.get("burn.in", 500)
    thinning = gibbs_control.get("thinning", 2)
    alpha = gibbs_control.get("alpha", 1)
    n_cores = gibbs_control.get("n.cores", 1)
    gibbs_idx = _PyGibbs.get_gibbs_idx(gibbs_control)
    if chain_length is None:
        chain_length = int(np.max(gibbs_idx) + 1)  # ORIGINAL fallback
    chain_length = int(chain_length)
    burn_in = int(burn_in)          # ORIGINAL: all_idx[int(burn_in):]
    thinning = int(thinning)
    if thinning < 1 or chain_length < 1 or burn_in < 0:
        return None
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        return None
    n_threads = int(n_cores)
    n_threads = max(1, min(_MAX_RUST_THREADS, n_threads))
    return {
        "chain_length": chain_length,
        "burn_in": burn_in,
        "thinning": thinning,
        "seed": seed,
        "alpha": alpha,
        "n_threads": n_threads,
        "n_keep": len(gibbs_idx),
    }


# The UNPATCHED fast-Python sampler class, captured at import time (the
# bayesprism_fast module global is rebound while a rust-lane call runs).
_PyGibbs = _bpf.GibbsSamplerFast


class GibbsSamplerRust(_PyGibbs):
    """GibbsSamplerFast with the two batch phases executed natively.

    Inherits EVERYTHING else verbatim (``_make_rng``, ``_spawn_seeds``,
    ``sample_Z_theta_n`` / ``sample_theta_n`` — used by the discarded
    ``estimate_gibbs_time`` probe chains — ``get_gibbs_idx``, the prints).
    """

    def run_gibbs_refPhi(self, final, compute_elbo):
        references = _bpf._bp_mod("references")
        assert isinstance(self.reference, references.RefPhi), \
            "Gibbs is not final but ref is not refPhi"
        kern = _kernel_or_none(self.gibbs_control, bool(compute_elbo))
        if kern is None:
            return _PyGibbs.run_gibbs_refPhi(self, final=final,
                                             compute_elbo=compute_elbo)
        phi_df = self.reference.phi
        phi = np.ascontiguousarray(phi_df.to_numpy(), dtype=np.float64)
        X = self.X.to_numpy()
        x = np.ascontiguousarray(X, dtype=np.int64)
        n = x.shape[0]
        print("Start run...")
        native = _native()
        z_flat, theta_flat, cv_flat = native.bp_gibbs_phase(
            phi, x, kern["alpha"], kern["chain_length"], kern["burn_in"],
            kern["thinning"], kern["seed"], 0, not final, kern["n_threads"],
        )
        K = phi.shape[0]
        G = phi.shape[1]
        theta = theta_flat.reshape(n, K)
        cv = cv_flat.reshape(n, K)
        gibbs_list = []
        for i in range(n):
            if final:
                gibbs_list.append({
                    "theta_n": theta[i],
                    "theta.cv_n": cv[i],
                })
            else:
                gibbs_list.append({
                    "Z_n": z_flat[i * G * K:(i + 1) * G * K].reshape(G, K),
                    "theta_n": theta[i],
                    "theta.cv_n": cv[i],
                    "gibbs.constant": 0.0,
                })
        if not final:
            return _bpf.JointPostFast.new(self.X.index, self.X.columns,
                                          phi_df.index, gibbs_list)
        theta_post = _bpf._bp_mod("theta_post")
        return theta_post.ThetaPost.new(self.X.index, self.X.columns,
                                        gibbs_list)

    def run_gibbs_refTumor(self):
        references = _bpf._bp_mod("references")
        assert isinstance(self.reference, references.RefTumor)
        kern = _kernel_or_none(self.gibbs_control, False)
        psi_mal = self.reference.psi_mal
        psi_env = self.reference.psi_env
        key = self.reference.key
        if (kern is None
                or list(psi_mal.columns) != list(psi_env.columns)):
            # column-label mismatch would make the ORIGINAL pd.concat align
            # differently from a positional stack — stay on the Python lane.
            return _PyGibbs.run_gibbs_refTumor(self)
        X = self.X.to_numpy()
        x = np.ascontiguousarray(X, dtype=np.int64)
        n = x.shape[0]
        pm = np.ascontiguousarray(psi_mal.to_numpy(), dtype=np.float64)
        pe = np.ascontiguousarray(psi_env.to_numpy(), dtype=np.float64)
        K = 1 + pe.shape[0]
        G = pm.shape[1]
        phi3 = np.empty((n, K, G), dtype=np.float64)
        phi3[:, 0, :] = pm
        phi3[:, 1:, :] = pe[None, :, :]
        # ORIGINAL per-sample: nonzero_idx = np.max(phi_n, axis=0) > 0
        trim = np.ascontiguousarray(np.max(phi3, axis=1) > 0)
        print("Start run...")
        native = _native()
        # seed_mode=1: EVERY sample uses SeedSequence(seed).spawn(1)[0]
        # (the upstream phase-3 re-spawn quirk, gibbs.py ~L393).
        _z, theta_flat, cv_flat = native.bp_gibbs_phase(
            np.ascontiguousarray(phi3[0]), x, kern["alpha"],
            kern["chain_length"], kern["burn_in"], kern["thinning"],
            kern["seed"], 1, False, kern["n_threads"],
            phi_per=np.ascontiguousarray(phi3.reshape(n * K, G)),
            trim=trim,
        )
        theta = theta_flat.reshape(n, K)
        cv = cv_flat.reshape(n, K)
        gibbs_list = [{"theta_n": theta[i], "theta.cv_n": cv[i]}
                      for i in range(n)]
        theta_post = _bpf._bp_mod("theta_post")
        return theta_post.ThetaPost.new(self.X.index,
                                        [key] + list(psi_env.index),
                                        gibbs_list)


def bayesprism_rust(
    bulk,
    sc_dat=None,
    cell_state_labels=None,
    cell_type_labels=None,
    key: str = "Malignant_cells",
    out_dir=None,
    n_threads=None,
    state_order: str = "sorted",
    outlier_cut: float = 0.01,
    outlier_fraction: float = 0.1,
    pseudo_min: float = 1e-8,
    gibbs_control=None,
    opt_control=None,
) -> dict:
    """``iobrx.bayesprism`` with the native Gibbs kernel (backend='rust').

    Same signature/contract as ``bayesprism_fast.bayesprism`` minus the
    ``backend`` argument (fixed to the rust lane, with automatic
    fast-Python fallback for non-default Gibbs configurations, see the
    module docstring). Bit-exactness: with ``state_order='legacy'`` under a
    pinned PYTHONHASHSEED the three output CSVs are byte-identical to the
    ORIGINAL gold run; with ``state_order='sorted'`` outputs are
    bit-identical to the fast-Python lane (the kernel replaces only the
    sampler, whose streams are bit-identical).
    """
    _native()  # fail loud before any work if the kernel is missing
    old = _bpf.GibbsSamplerFast
    _bpf.GibbsSamplerFast = GibbsSamplerRust
    try:
        return _bpf.bayesprism(
            bulk,
            sc_dat=sc_dat,
            cell_state_labels=cell_state_labels,
            cell_type_labels=cell_type_labels,
            key=key,
            out_dir=out_dir,
            n_threads=n_threads,
            backend="auto",
            state_order=state_order,
            outlier_cut=outlier_cut,
            outlier_fraction=outlier_fraction,
            pseudo_min=pseudo_min,
            gibbs_control=gibbs_control,
            opt_control=opt_control,
        )
    finally:
        _bpf.GibbsSamplerFast = old
