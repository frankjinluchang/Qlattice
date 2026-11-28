"""
Module ``qlat.cgfhmc``
========================\n
Canonical Gauge-Fixed HMC (CGFHMC) for pure-gauge theory, with a Fourier
mode-dependent partial momentum refresh.\n
The algorithm
=============\n
**Canonical Gauge-Fixed HMC (CGFHMC)** is the canonical-form HMC of a
gauge-fixed lattice.  One trajectory is a standard force-gradient HMC
trajectory with two extra ingredients:\n
1. **Landau gauge fixing after every complete MD step**, applied canonically —
   the same transform ``gt`` is applied to the gauge field AND to the gauge
   momentum, so the pair ``(gf, gm)`` stays a canonical pair and the kinetic
   energy is invariant under the fixing::\n
       (gt, info) = fix_landau_gauge(gf, ...)
       gf @= gt * gf        # field
       gm @= gt * gm        # momentum (same transform, adjoint action)\n
2. **Partial momentum refresh** after every complete MD step, with a
   mode-dependent Fourier coefficient ``c(k)``::\n
       P_k <- sqrt(1 - c(k)^2) * P_k + c(k) * xi_k\n
   where ``xi`` is a fresh Gaussian gauge momentum.  Here ``c`` is the
   **fresh-noise weight**, not the correlation: ``c = 0`` is a strict no-op
   that draws no random number, ``c = 1`` fully resamples the mode, and the
   retained-momentum autocorrelation is ``sqrt(1 - c(k)^2)``.  The older
   convention stored the correlation ``c_corr``; the conversion is
   ``c = sqrt(1 - c_corr^2)`` (e.g. ``c_corr = 0.9`` <-> ``c = 0.4358...``).\n
The refresh preserves the Gaussian momentum ensemble mode by mode: with
``E|P_k|^2 = sigma_m^2``,\n
    E|P'_k|^2 = (1 - c(k)^2) * sigma_m^2 + c(k)^2 * sigma_m^2 = sigma_m^2\n
so ``c(k)`` sets only the mode-dependent decorrelation rate.  Both ``P_k`` and
``xi_k`` carry qlat's native per-component variance, so no rescaling is
needed and a constant ``c`` reduces to the usual Ornstein-Uhlenbeck refresh.\n
3. **Metropolis on the corrected energy**.  The stochastic kinetic change
   injected by the refresh is removed from the Metropolis variable::\n
       delta_h        = H_final - H_initial                 (raw, returned)
       dK_refresh_sum = sum over steps of (K_after - K_before)
       delta_h_accept = delta_h - dK_refresh_sum            (Metropolis input)\n
``metropolis_accept`` is called with ``delta_h_accept``; the returned
``delta_h`` is the raw trajectory value, matching
:func:`qlat.hmc_utils.run_hmc_pure_gauge`.\n
Exactness caveat
================\n
For a **uniform** ``c`` this is the standard exact Ornstein-Uhlenbeck partial
momentum refresh and the algorithm is exact HMC.  For a **mode-dependent**
``c(k)`` the refresh is a global (Fourier-space) operation and is not a
standard partial refresh, so removing ``dK_refresh_sum`` from the Metropolis
variable is an adopted convention supported by empirical sampling checks, not a
formal generalized-HMC proof.  Do not present mode-dependent CGFHMC as an exact
HMC.\n
Reversibility caveat
====================\n
CGFHMC is **not time-reversible**: ``fix_landau_gauge`` is a projection onto one
representative per gauge orbit, which discards the phase-space information a
reverse test relies on.  ``run_hmc_cgfhmc_pure_gauge`` therefore accepts
``is_reverse_test`` for signature compatibility with
:func:`qlat.hmc_utils.run_hmc_pure_gauge` but raises ``NotImplementedError``
when it is ``True`` rather than returning a meaningless verdict.\n
API
===\n
- :func:`k_vectors`, :func:`k_squared`, :func:`resolve_k_cut` — the momentum
  grid in ``total_site`` axis order ``(Lx, Ly, Lz, Lt)``.
- :func:`regularize_c_field`, :func:`generate_c_field`,
  :func:`c_field_from_file` — build a usable ``c(k)`` field.
- :func:`partial_refresh_momentum_mode` — the refresh itself.
- :func:`run_hmc_cgfhmc_pure_gauge` — one complete CGFHMC trajectory.\n
The ``c(k)`` field is a plain 4-D ``float64`` numpy array shaped like the
lattice (``total_site`` order).  It is small (2 KiB at ``4^4``) and must be
identical on every node; it carries no profile logic, so any prescription —
analytic, tabulated or loaded from a file — enters through the same path.\n
Documentation: ``docs/qlat/qlat_cgfhmc.md``\n
.. note:: Update the documentation when updating this source file.
"""

import math
import os

import numpy as np

class q:
    from qlat_utils import (
        timer,
        displayln_info,
        get_fname,
    )
    from .mpi import (
        glb_sum_double,
    )
    from .geometry import (
        Geometry,
    )
    from .hmc import (
        GaugeMomentum,
        gm_hamilton_node,
        gf_hamilton_node,
        gf_evolve,
    )
    from .hmc_utils import (
        metropolis_accept,
        gm_evolve_fg_pure_gauge,
    )
    from .gauge_fixing import (
        fix_landau_gauge,
    )
    from .field_utils_utils import (
        mk_fft,
    )

###
### momentum grid
###

def _require_geometry(geo):
    """Accept ONLY a ``qlat.Geometry``; reject tuples/shapes with a clear error."""
    if not isinstance(geo, q.Geometry):
        raise TypeError(
            "expected a qlat.Geometry as the first argument (e.g. "
            "q.Geometry(q.Coordinate([16, 16, 16, 32]))); got "
            f"{type(geo).__name__}.  Pass a Geometry so the momentum grid "
            "matches the gauge field exactly - 4-tuples/shapes are not accepted"
        )
    return geo

def k_vectors(geo):
    """Return the 4 momentum components on the mode grid.\n
    Shape ``(4, Lx, Ly, Lz, Lt)`` in ``total_site`` axis order.  Component
    ``i`` is the physical momentum along lattice axis ``i``, varying along its
    own axis, and covers ``[-pi, pi)`` in FFT convention::\n
        k_i[n] = (2*pi/L_i) * ( n          if n <= L_i/2
                                n - L_i    otherwise ).\n
    The folded (signed) index is used because index ``n`` and ``n - L`` label
    the same physical momentum in a discrete Fourier transform; without
    folding a profile in ``k^2`` would be misapplied at the upper half of each
    axis.  The Nyquist index ``n = L_i/2`` is included.\n
    Use ``np.sum(k_vectors(geo)**2, axis=0)`` (or :func:`k_squared`) for
    ``k^2``.
    """
    geo = _require_geometry(geo)
    shape = tuple(int(x) for x in geo.total_site)
    k_vec = np.empty((4,) + shape, dtype=np.float64)
    for i, L in enumerate(shape):
        n = np.arange(L, dtype=np.float64)
        comp = 2.0 * math.pi * np.where(n <= L / 2.0, n, n - L) / float(L)
        bc = [1, 1, 1, 1]
        bc[i] = L
        k_vec[i] = comp.reshape(bc)
    return k_vec

def k_squared(geo):
    """``k^2 = sum_i k_i^2`` on the mode grid, shaped ``(Lx, Ly, Lz, Lt)``."""
    return np.sum(k_vectors(geo) ** 2, axis=0)

def resolve_k_cut(geo, units):
    """Convert a threshold in units of ``2*pi/Lx`` to an absolute momentum.\n
    ``resolve_k_cut(geo, 2)`` on an ``Lx = 16`` lattice gives ``2*pi/8``, i.e.
    "the second non-trivial momentum".  Convenience helper for writing
    ``c(k)`` prescriptions.
    """
    geo = _require_geometry(geo)
    return float(units) * 2.0 * math.pi / float(geo.total_site[0])

###
### c(k) field
###

def _get_central_inversion_partner(c):
    """Average ``c`` with its 4D central-inversion partner ``c[(-n) % L]``.\n
    ``(-n) % L`` is the index of ``-k``.  Returns a NEW array.
    """
    c = np.asarray(c)
    if c.ndim != 4:
        raise ValueError(f"c field must be 4-D, got shape {c.shape}")
    partner = c
    for ax in range(4):
        # Reversing an axis gives c[L-1-n]; rolling by +1 gives c[(L-n) % L]
        # = c[(-n) % L], so index 0 stays put and the rest reverse.
        partner = np.roll(np.flip(partner, axis=ax), 1, axis=ax)
    return 0.5 * (c + partner)

def regularize_c_field(c, *, in_place=False, allow_nan="raise"):
    """Make a ``c(k)`` field usable by the refresh: ``0 <= c <= 1`` and
    ``c(-k) = c(k)``.\n
    Both conditions are required.  Range: ``sqrt(1 - c^2)`` must be real, and
    values are CLIPPED (not rescaled) so an out-of-range prescription is
    visibly saturated rather than silently distorted in shape.  Evenness is
    enforced by AVERAGING with the 4D central-inversion partner ``c[(-n) % L]``
    (the least-squares projector onto even fields) — the su(3) component fields
    are real, so their spectra obey ``X_{-k} = conj(X_k)`` and only even
    coefficients keep the refreshed momentum real and anti-Hermitian.\n
    ``allow_nan`` is ``"raise"`` (default) to refuse NaN/inf with an actionable
    message, or ``"zero"`` to map non-finite entries to ``0.0`` (a full refresh
    of the affected modes).  Diagnostics from the last call are available in
    ``regularize_c_field.last_report``.\n
    Returns the regularized ``float64`` field (a new array unless
    ``in_place=True`` and the input is already a ``float64`` ndarray).
    """
    arr = np.asarray(c)
    if np.iscomplexobj(arr):
        if np.max(np.abs(arr.imag)) > 0.0:
            raise ValueError(
                "c(k) must be REAL: a complex coefficient would make the "
                "inverse transform complex and break anti-Hermiticity/"
                "tracelessness of the refreshed momentum"
            )
        arr = arr.real
    if arr.ndim != 4:
        raise ValueError(
            f"c(k) field must be 4-D shaped like total_site, got shape {arr.shape}"
        )
    work = arr.astype(np.float64, copy=not in_place)
    nonfinite = ~np.isfinite(work)
    n_nonfinite = int(np.count_nonzero(nonfinite))
    if n_nonfinite:
        if allow_nan == "raise":
            raise ValueError(
                f"c(k) field contains {n_nonfinite} non-finite entries "
                "(NaN/inf); fix the generator or pass allow_nan='zero' to map "
                "them to 0.0 explicitly"
            )
        if allow_nan != "zero":
            raise ValueError(f"allow_nan must be 'raise' or 'zero', got {allow_nan!r}")
        work[nonfinite] = 0.0
    n_low = int(np.count_nonzero(work < 0.0))
    n_high = int(np.count_nonzero(work > 1.0))
    np.clip(work, 0.0, 1.0, out=work)
    evened = _get_central_inversion_partner(work)
    max_asym = float(np.max(np.abs(work - evened))) if work.size else 0.0
    work[...] = evened
    regularize_c_field.last_report = {
        "n_nonfinite_zeroed": n_nonfinite,
        "n_clipped_low": n_low,
        "n_clipped_high": n_high,
        "max_asymmetry_removed": max_asym,
        "c_min": float(work.min()) if work.size else 0.0,
        "c_max": float(work.max()) if work.size else 0.0,
        "c_mean": float(work.mean()) if work.size else 0.0,
    }
    return work

def generate_c_field(geo, func):
    """Build a regularized ``c(k)`` field by evaluating a user callable.\n
    ``func`` is called ONCE as ``func(k_vec)`` where ``k_vec`` is a real array
    of shape ``(4, Lx, Ly, Lz, Lt)`` holding the four momentum components in
    ``total_site`` axis order, each covering ``[-pi, pi)`` (see
    :func:`k_vectors`).  The return value must be real and broadcastable to the
    lattice shape; complex or genuinely non-real output is rejected.\n
    Example — a two-zone profile (stiff low modes, hard high modes)::\n
        tc = resolve_k_cut(geo, 2)
        func = lambda kv: np.where(np.sum(kv**2, axis=0) <= tc**2, 0.6, 0.95)\n
    Example — a gaussian::\n
        sigma = resolve_k_cut(geo, 2)
        func = lambda kv: 0.95 - 0.35 * np.exp(
            -np.sum(kv**2, axis=0) / (2 * sigma**2))\n
    Regularization is ALWAYS applied (no opt-out): the raw output goes through
    :func:`regularize_c_field`, so the result is clipped into ``[0, 1]`` and
    exactly even under ``k -> -k``.  NaN/inf raises ``ValueError``; use
    ``regularize_c_field(raw, allow_nan="zero")`` explicitly if discarding them
    is intended.  Diagnostics: ``generate_c_field.last_report``.\n
    Returns ``float64``, C-contiguous, shaped ``(Lx, Ly, Lz, Lt)``.
    """
    geo = _require_geometry(geo)
    if not callable(func):
        raise TypeError(
            f"func must be callable, got {type(func).__name__}; use e.g. "
            "lambda kv: np.where(np.sum(kv**2, axis=0) <= tc**2, 0.6, 0.95)"
        )
    shape = tuple(int(x) for x in geo.total_site)
    k_vec = k_vectors(geo)
    arr = np.asarray(func(k_vec))
    if np.iscomplexobj(arr):
        if np.max(np.abs(arr.imag)) > 0.0:
            raise ValueError(
                "func returned a genuinely COMPLEX c(k) field; it must be "
                "real-valued (a complex coefficient would break "
                "anti-Hermiticity/tracelessness of the refreshed momentum)"
            )
        arr = arr.real
    arr = np.asarray(arr, dtype=np.float64)
    try:
        arr = np.broadcast_to(arr, shape).copy()
    except ValueError:
        raise ValueError(
            f"func returned an array of shape {arr.shape} that is not "
            f"broadcastable to the lattice shape {shape}"
        )
    c = regularize_c_field(arr)
    generate_c_field.last_report = dict(
        regularize_c_field.last_report, total_site=list(shape)
    )
    return np.ascontiguousarray(c)

def c_field_from_file(path):
    """Load a real ``c(k)`` field from a ``.npy`` file and regularize it.\n
    The loaded array must be 4-D and shaped like the lattice; it is passed
    through :func:`regularize_c_field` (clip to ``[0, 1]``, enforce
    ``c(-k) = c(k)``, reject NaN/inf), exactly like a callable's output.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"c-field file not found: {path}")
    return regularize_c_field(np.load(path))

###
### partial momentum refresh
###

def _get_c_field_local(c_field, geo):
    """Return the local ``c(k)`` values and the per-mode refresh weights.\n
    ``xg_arr()`` labels each local site with the coordinate whose index it
    occupies, and (verified under MPI) that labelling still holds in k-space
    after a forward FFT, so ``c_field[xg]`` gives each local mode its own
    coefficient.  Returns ``(c_loc, keep, is_uniform)`` where ``keep`` is
    ``sqrt(1 - c_loc**2)``.
    """
    shape = tuple(int(x) for x in geo.total_site)
    c = regularize_c_field(c_field)
    if c.shape != shape:
        raise ValueError(f"c_field shape {c.shape} does not match the lattice {shape}")
    xg_arr = geo.xg_arr()
    c_loc = c[xg_arr[:, 0], xg_arr[:, 1], xg_arr[:, 2], xg_arr[:, 3]]
    c_loc = np.ascontiguousarray(c_loc, dtype=np.float64)
    keep = np.sqrt(np.maximum(0.0, 1.0 - c_loc * c_loc))
    is_uniform = bool(float(c_loc.min()) == float(c_loc.max()))
    return c_loc, keep, is_uniform

@q.timer
def partial_refresh_momentum_mode(gm, rng, c_field):
    """Mode-dependent partial momentum refresh of a ``GaugeMomentum``.\n
    Applies, independently for every Fourier mode ``k``, Lorentz index and
    su(3) component::\n
        P_k <- sqrt(1 - c(k)^2) * P_k + c(k) * xi_k\n
    so each mode's variance is preserved and ``c(k)`` sets only the
    mode-dependent refresh strength.  ``c`` is the FRESH-NOISE weight:
    ``c(k) = 0`` freezes the mode (bit-identical, no random number drawn for
    it) and ``c(k) = 1`` fully resamples it; the autocorrelation of the
    retained part is ``sqrt(1 - c(k)^2)``.\n
    Parameters
    ----------
    gm : GaugeMomentum
        Refreshed **in place**.
    rng : RngState
        RNG stream for this call.
    c_field : numpy.ndarray
        Real array shaped like the lattice giving ``c(k)`` on the mode grid.
        It is passed through :func:`regularize_c_field`, so the ``[0, 1]`` and
        ``c(-k) = c(k)`` preconditions are guaranteed rather than silently
        repaired elsewhere.\n
    Returns
    -------
    (applied, dK, info)
        ``applied`` — ``False`` iff ``c(k) == 0`` everywhere: a strict no-op
        that draws **no** random number and leaves ``gm`` untouched, so runs
        with the refresh disabled are bit-identical to runs without it.
        ``dK`` — ``K_after - K_before`` (MPI-reduced); the caller EXCLUDES this
        from the Metropolis variable (see the module docstring's exactness
        caveat).  ``info`` — diagnostics: ``c_min``/``c_max``/``c_mean``,
        ``n_modes``, ``n_modes_refreshed``, ``n_modes_frozen``,
        ``is_uniform`` and ``path``.\n
    Notes
    -----
    A **constant** ``c`` is applied directly in real space: the unitary
    transform commutes with a scalar mixing, so the result is mathematically
    identical and much cheaper.  ``info["path"]`` records which route ran.
    The Fourier route uses qlat's distributed FFT with ``is_normalizing=True``
    and works on any node decomposition.
    """
    geo = gm.geo
    c_loc, keep, is_uniform = _get_c_field_local(c_field, geo)
    n_modes = int(c_loc.size)
    info = {
        "c_min": float(c_loc.min()) if n_modes else 0.0,
        "c_max": float(c_loc.max()) if n_modes else 0.0,
        "c_mean": float(c_loc.mean()) if n_modes else 0.0,
        "n_modes": n_modes,
        # under the noise-weight convention a mode is refreshed iff it
        # receives fresh noise, i.e. c(k) > 0; c(k) == 0 freezes it.
        "n_modes_refreshed": int(np.count_nonzero(c_loc > 0.0)),
        "n_modes_frozen": int(np.count_nonzero(c_loc <= 0.0)),
        "is_uniform": is_uniform,
    }
    # ---- strict no-op: c == 0 everywhere (no fresh noise) -> draw nothing ----
    if float(c_loc.max()) <= 0.0:
        info["no_op"] = True
        info["path"] = "no-op (c == 0)"
        return False, 0.0, info
    info["no_op"] = False
    k_before = float(q.gm_hamilton_node(gm))
    # ---- FAST PATH: a constant c(k) needs no Fourier transform at all ----
    if is_uniform:
        c_uni = float(c_loc.min())
        info["path"] = "real-space (uniform c)"
        xi = q.GaugeMomentum(geo)
        xi.set_rand(rng, 1.0)
        gm_arr = np.asarray(gm)
        gm_arr[...] = float(keep.flat[0]) * gm_arr + c_uni * np.asarray(xi)
    else:
        info["path"] = "fourier (mode-dependent c)"
        xi = q.GaugeMomentum(geo)
        xi.set_rand(rng, 1.0)
        fft_f = q.mk_fft(True, is_normalizing=True)
        fft_b = q.mk_fft(False, is_normalizing=True)
        p_k = fft_f * gm
        x_k = fft_f * xi
        p_k_arr = np.asarray(p_k)
        x_k_arr = np.asarray(x_k)
        # broadcast the per-mode scalars over the (mu, 3, 3) block of a site
        p_k_arr[...] = (
            keep.reshape((-1,) + (1,) * (p_k_arr.ndim - 1)) * p_k_arr
            + c_loc.reshape((-1,) + (1,) * (p_k_arr.ndim - 1)) * x_k_arr
        )
        gm @= fft_b * p_k
    k_after = float(q.gm_hamilton_node(gm))
    dk = float(q.glb_sum_double(k_after - k_before))
    return True, dk, info

###
### the CGFHMC driver
###

@q.timer
def _cgfhmc_evolve_pure_gauge(
    gm,
    gf,
    ga,
    n_step,
    md_time,
    c_field,
    refresh_rng_base,
    gauge_fix_max_iter,
    gauge_fix_eps,
    gauge_fix_alpha,
    fix_records,
    refresh_stats,
):
    """Run the CGFHMC MD evolution of ``(gm, gf)`` in place.\n
    Same uniform symmetric 4th-order force-gradient integrator as
    :func:`qlat.hmc_utils.run_hmc_evolve_pure_gauge`, with the CGFHMC insertion
    (Landau fixing of ``gf`` and ``gm``, then the partial momentum refresh)
    after every COMPLETE MD step.  Returns the raw energy violation
    ``delta_h`` (already MPI-summed).
    """
    energy = float(q.gm_hamilton_node(gm)) + float(q.gf_hamilton_node(gf, ga))
    dt = md_time / n_step
    lam = 0.5 * (1.0 - 1.0 / math.sqrt(3.0))
    theta = (2.0 - math.sqrt(3.0)) / 48.0
    ttheta = theta * dt * dt * dt
    drift_total = 0.0
    # Each step is the SELF-CONTAINED symmetric ``lam, (1 - 2*lam), lam`` drift
    # sandwich, exactly like :func:`qlat_ref.cgfhmc.run_hmc_cgfhmc_pure_gauge`.
    # Do NOT hoist the leading ``lam*dt`` out of the loop the way
    # :func:`qlat.hmc_utils.run_hmc_evolve_pure_gauge` does: there the hoist is
    # algebraically invisible, but here the fixing + refresh sit at the step
    # boundary, so a hoisted drift would place them at ``(i+1)*dt + lam*dt``
    # instead of ``(i+1)*dt`` and change the algorithm (this was a real bug -
    # the two drivers disagreed by O(10) in ``delta_h`` for ``n_step >= 2``).
    for i in range(n_step):
        # ---- one COMPLETE MD step: the 5 sub-updates below ----
        q.gf_evolve(gf, gm, lam * dt)
        drift_total += lam * dt
        q.gm_evolve_fg_pure_gauge(gm, gf, ga, 4.0 * ttheta / dt, 0.5 * dt)
        q.gf_evolve(gf, gm, (1.0 - 2.0 * lam) * dt)
        drift_total += (1.0 - 2.0 * lam) * dt
        q.gm_evolve_fg_pure_gauge(gm, gf, ga, 4.0 * ttheta / dt, 0.5 * dt)
        q.gf_evolve(gf, gm, lam * dt)
        drift_total += lam * dt
        # ---- end of complete MD step i; drift so far = (i+1)*dt ----
        # The insertion is strictly after the step's last integrator
        # sub-update and before the next step's first one: the same symmetric
        # phase point on every step.
        gt, info = q.fix_landau_gauge(
            gf, max_iter=gauge_fix_max_iter, eps=gauge_fix_eps, alpha=gauge_fix_alpha
        )
        # fix_landau_gauge does NOT mutate gf: it returns (gt, info) with
        # gt * gf the fixed field, copied into the existing buffer so the
        # caller's proposal object sees the update.
        gf @= gt * gf
        # Canonical: rotate the momentum with the SAME gt (adjoint action).
        gm @= gt * gm
        if fix_records is not None:
            fix_records.append(
                {
                    "step": i,
                    "n_iter": int(info["n_iter"]),
                    "converged": bool(info["converged"]),
                    "functional": float(info["functional"]),
                    "divergence": float(info["divergence"]),
                }
            )
        # ---- partial momentum refresh, same boundary as the gt insertion ----
        applied, dk_step, rinfo = partial_refresh_momentum_mode(
            gm, refresh_rng_base.split(f"partial_refresh_momentum.{i}"), c_field
        )
        if applied:
            refresh_stats["n_applied"] += 1
            refresh_stats["dk_sum"] += dk_step
            refresh_stats["last_info"] = rinfo
    gf.unitarize()
    drift_expected = n_step * dt
    drift_tol = max(
        8.0 * abs(drift_expected) * 2.3e-16,
        3.0 * n_step * 2.3e-16 * abs(drift_expected),
    )
    assert abs(drift_total - drift_expected) <= drift_tol, (
        f"total drift {drift_total!r} != md_time {drift_expected!r}"
        f" (tol={drift_tol!r}); the integrator budget is inconsistent"
    )
    delta_h = float(q.gm_hamilton_node(gm)) + float(q.gf_hamilton_node(gf, ga)) - energy
    delta_h = float(q.glb_sum_double(delta_h))
    return delta_h

@q.timer(is_timer_fork=True)
def run_hmc_cgfhmc_pure_gauge(
    gf,
    ga,
    traj,
    rs,
    *,
    c_field,
    is_reverse_test=False,
    n_step=6,
    md_time=1.0,
    is_always_accept=False,
    gauge_fix_max_iter=25,
    gauge_fix_eps=1.0e-10,
    gauge_fix_alpha=0.1,
    info_out=None,
):
    """Run a single CGFHMC trajectory for pure gauge theory.\n
    Generates a random momentum, runs the CGFHMC MD evolution (force-gradient
    integrator, Landau gauge fixing of ``gf`` and ``gm`` after every complete
    MD step, then a partial mode-dependent momentum refresh), and applies the
    Metropolis accept/reject step on ``delta_h - dK_refresh_sum``.  On accept
    the input ``gf`` is updated in-place.\n
    Parameters
    ----------
    gf : GaugeField
        Input/output gauge field.
    ga : GaugeAction
    traj : int
        Trajectory number (used to split the RNG).
    rs : RngState
    c_field : numpy.ndarray
        Real ``c(k)`` field shaped like the lattice (``total_site`` order),
        the fresh-noise weight of the partial momentum refresh.  Build it with
        :func:`generate_c_field` (or :func:`regularize_c_field`).
    is_reverse_test : bool
        Present for signature compatibility with
        :func:`qlat.hmc_utils.run_hmc_pure_gauge`, but **not supported**:
        CGFHMC is not time-reversible (the per-step gauge fixing is a
        projection), so ``True`` raises ``NotImplementedError``.
    n_step : int
        Number of MD steps per trajectory (default 6).
    md_time : float
        Total molecular dynamics time (default 1.0).
    is_always_accept : bool
        If True, skip the Metropolis test and always update.
    gauge_fix_max_iter, gauge_fix_eps, gauge_fix_alpha
        Passed to :func:`qlat.gauge_fixing.fix_landau_gauge` for the per-step
        Landau fixing.  ``max_iter`` is deliberately small (25): the field is
        re-fixed every MD step and starts already near Landau gauge, so only
        the first step after a cold start needs many sweeps.
    info_out : dict, optional
        Filled in place with diagnostics: ``dK_sum``,
        ``n_refresh_applied``, ``n_step``, ``n_fix``, ``fix_n_iter_total``,
        ``fix_n_iter_max``, ``fix_n_converged``, ``fix_divergence_max``,
        ``delta_h_raw`` and ``delta_h_accept``.  The return arity is unchanged.\n
    Returns
    -------
    (flag, delta_h)
        ``flag`` is True if the trajectory was accepted.  ``delta_h`` is the
        RAW trajectory energy violation ``H_final - H_initial`` (matching
        :func:`qlat.hmc_utils.run_hmc_pure_gauge`); the Metropolis test uses
        ``delta_h - dK_sum`` internally.
    """
    if is_reverse_test:
        raise NotImplementedError(
            "CGFHMC applies a gauge-fixing projection each MD step and is not "
            "time-reversible; is_reverse_test is not supported"
        )
    fname = q.get_fname()
    rs = rs.split(f"{traj}")
    geo = gf.geo
    gf0 = type(gf)(geo)
    gf0 @= gf
    gm = q.GaugeMomentum(geo)
    gm.set_rand(rs.split("set_rand_gauge_momentum"), 1.0)
    fix_records = []
    refresh_stats = {"n_applied": 0, "dk_sum": 0.0, "last_info": None}
    delta_h = _cgfhmc_evolve_pure_gauge(
        gm,
        gf0,
        ga,
        n_step,
        md_time,
        c_field,
        rs.split("partial_refresh_momentum_base"),
        gauge_fix_max_iter,
        gauge_fix_eps,
        gauge_fix_alpha,
        fix_records,
        refresh_stats,
    )
    dk_sum = float(refresh_stats["dk_sum"])
    delta_h_accept = delta_h - dk_sum
    flag, accept_prob = q.metropolis_accept(
        delta_h_accept, traj, rs.split("metropolis_accept")
    )
    if flag or is_always_accept:
        q.displayln_info(f"{fname}: update gf (traj={traj})")
        gf @= gf0
    if info_out is not None:
        n_iters = [r["n_iter"] for r in fix_records]
        divs = [r["divergence"] for r in fix_records]
        info_out.update(
            {
                "dK_sum": dk_sum,
                "n_refresh_applied": int(refresh_stats["n_applied"]),
                "n_step": int(n_step),
                "n_fix": len(fix_records),
                "fix_n_iter_total": int(sum(n_iters)) if n_iters else 0,
                "fix_n_iter_max": int(max(n_iters)) if n_iters else 0,
                "fix_n_converged": int(sum(1 for r in fix_records if r["converged"])),
                "fix_divergence_max": float(max(divs)) if divs else 0.0,
                "delta_h_raw": delta_h,
                "delta_h_accept": delta_h_accept,
            }
        )
    return flag, delta_h
