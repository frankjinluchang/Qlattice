"""
Module ``qlat_ref.cgfhmc``
==========================\n
Canonical Gauge-Fixed HMC (CGFHMC): canonical pure-gauge HMC with a Landau
gauge-fixing projection after every molecular-dynamics step and a
Fourier-mode-dependent partial momentum refresh.\n
This mirrors ``qlat.cgfhmc`` (same names, argument order, defaults and mutation
semantics).  The uniform refresh it generalises is the one of the validated
pure-gauge pilot; see :mod:`qlat_ref.hmc_utils` for the plain driver it wraps.\n
Algorithm
---------\n
One trajectory is :func:`qlat_ref.hmc_utils.run_hmc_pure_gauge` with two
insertions per MD step:\n
1. **Landau gauge fixing** of both ``gf`` and ``gm``, strictly *after* the
   step's last integrator sub-update and *before* the next step's first one --
   the same symmetric phase point on every step.  The fixing is a projection:
   ``gt, info = fix_landau_gauge(gf0, ...)`` does not mutate ``gf0``, and the
   canonical update is ``gf0 @= gt * gf0`` together with ``gm @= gt * gm``
   (the same-site adjoint action, no shift).
2. **Partial momentum refresh**, per Fourier mode,
       ``P_k <- sqrt(1 - c(k)^2) * P_k + c(k) * xi_k``,
   with ``c`` the fresh-NOISE weight, not the correlation: ``c(k) = 0`` freezes
   the mode, ``c(k) = 1`` fully resamples it, and the retained autocorrelation
   is ``sqrt(1 - c^2)``.  See :func:`partial_refresh_momentum_mode`.\n
The refresh deliberately changes the kinetic energy, so that change is **not**
charged to the Metropolis test: the driver accepts on
``delta_h_accept = delta_h - dK_sum``, the sum of the per-step kinetic changes
returned by :func:`partial_refresh_momentum_mode`.  The **returned** ``delta_h``
is nevertheless the raw trajectory value ``H(final) - H(initial)``, exactly as
:func:`qlat_ref.hmc_utils.run_hmc_pure_gauge` defines it, so the two drivers
stay comparable; ``delta_h_accept`` is algorithm-internal and only reported
through ``info_out``.\n
Why the c(k) field must be real, even and bounded
------------------------------------------------\n
``P_mu(x) = i sum_a phi_a(x) T_a`` with ``phi_a`` real, and the eight real
component fields are transformed independently.  A real spectrum obeys
``X_{-k} = conj(X_k)``, so the mix preserves that Hermitian symmetry only for
even coefficients ``c(-k) = c(k)``; otherwise the inverse transform is complex
and the momentum leaves the anti-Hermitian traceless subspace.  The range
``0 <= c <= 1`` keeps ``sqrt(1 - c^2)`` real for the same reason.  Both
conditions are enforced by :func:`regularize_c_field`, and every generator and
consumer re-checks them rather than silently repairing an input.\n
.. note:: Single process.  ``numpy.fft`` is a global transform, so this
   implementation requires the whole lattice on one node; qlat's ``mk_fft``
   is distributed and runs under MPI.  The names, signatures and defaults are
   qlat's regardless; only the parallel decomposition differs.\n
.. note:: ``is_reverse_test`` is **not supported**: the gauge-fixing projection
   destroys time reversibility (measured: a forward-then-backward trajectory
   returns to ``|dU| = O(0.4)`` with the fixing on versus ``1e-15`` with it
   off).  The keyword is kept so the signature matches
   :func:`qlat_ref.hmc_utils.run_hmc_pure_gauge`, and passing ``True`` raises
   ``NotImplementedError`` rather than returning a meaningless verdict.\n
Documentation: ``docs/qlat-ref/qlat_cgfhmc.md``\n
.. note:: Update the documentation when updating this source file.
"""

import math
import os

import numpy as np

from .base import Geometry

from .field import GaugeField
from .field import GaugeMomentum

from .gauge_fixing import fix_landau_gauge

from .hmc import _get_lattice
from .hmc import _set_lattice
from .hmc import gf_evolve
from .hmc import gf_hamilton_node
from .hmc import gm_hamilton_node

from .hmc_utils import _displayln_info
from .hmc_utils import gm_evolve_fg_pure_gauge
from .hmc_utils import metropolis_accept

### -------------------------------------------------------------------

# Tolerance of the input validation: what counts as "even" (and as
# anti-Hermitian in the component decomposition).  It is the experimental
# implementation's, 64 ulp of the coefficient, i.e. it refuses a field that
# still needs regularizing while accepting one that has only been round-tripped
# through arithmetic.  Every comparison that uses it is a *scaled* one (against
# the magnitude of the values involved), never an absolute epsilon on a
# near-zero quantity.
_EPS_EVEN = 64.0 * np.finfo(np.float64).eps

# ``P_mu(x) = i sum_a phi_a(x) T_a`` with ``T_a = lambda_a / 2`` the su(3)
# generators normalised by ``Tr(T_a T_b) = delta_ab / 2``.  The eight component
# fields are real, and the same basis is used for the decomposition and the
# reconstruction so that a round trip is exact to round-off.
def _get_gell_mann_half():
    """Return the eight Hermitian traceless generators ``T_a = lambda_a / 2`` (helper)."""
    lam = np.zeros((8, 3, 3), dtype=np.complex128)
    lam[0] = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    lam[1] = [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]
    lam[2] = [[1, 0, 0], [0, -1, 0], [0, 0, 0]]
    lam[3] = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
    lam[4] = [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]
    lam[5] = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    lam[6] = [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]
    lam[7] = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=np.complex128
    ) / math.sqrt(3.0)
    return lam / 2.0

_T_A = _get_gell_mann_half()

### -------------------------------------------------------------------

def _require_geometry(geo):
    """
    Accept ONLY a :class:`qlat_ref.base.Geometry`, with an actionable error (helper).\n
    qlat raises the same way: a bare shape or a 4-tuple is refused, so the
    momentum grid is always guaranteed to match the gauge field the ``c(k)``
    field is applied to.
    """
    if not isinstance(geo, Geometry):
        raise TypeError(
            "expected a Geometry as the first argument (e.g."
            " Geometry(Coordinate([16, 16, 16, 32]))); got"
            f" {type(geo).__name__}.  Pass a Geometry so the momentum grid"
            " matches the gauge field exactly - 4-tuples/shapes are not accepted"
        )
    #
    return geo

def _get_total_site(geo):
    """Return the lattice extents as a tuple of four ``int`` (helper)."""
    return tuple(int(x) for x in geo.total_site)

def _get_folded_pmomenta(total_site):
    """
    Return the folded physical momentum along each axis (helper).\n
    Axis ``i`` holds ``(2*pi/L_i) * (n_i`` if ``n_i <= L_i/2``, else
    ``n_i - L_i)`` for ``n_i = 0, 1, ..., L_i - 1``.  Index and index-minus-``L``
    label the same momentum in a discrete Fourier transform, so without folding a
    profile in ``k^2`` would be applied to the wrong modes on the upper half of
    each axis.  The Nyquist index ``L_i/2`` is included by the same formula.
    """
    k_axes = []
    for length in total_site:
        n = np.arange(length, dtype=np.float64)
        k = 2.0 * math.pi * np.where(n <= length / 2.0, n, n - length) / float(length)
        k_axes.append(k)
    #
    return k_axes

### -------------------------------------------------------------------

def k_vectors(geo):
    """
    Return the four momentum components on the mode grid (qlat's ``k_vectors``).\n
    Shape ``(4, Lx, Ly, Lz, Lt)``, ``dtype`` ``float64``.  Component ``i`` is the
    physical momentum along lattice axis ``i`` -- the axes are qlat's
    ``total_site`` order, ``(Lx, Ly, Lz, Lt)``, *not* the ``(Nt, Nx, Ny, Nz)``
    site order of the field buffer -- varying along its own axis and covering
    ``[-pi, pi)`` in the FFT convention,
        ``k_i(n) = (2*pi/L_i) * (n`` if ``n <= L_i/2``, else ``n - L_i)``.\n
    The folded (signed) index is used because index ``n`` and index ``n - L``
    denote the same physical momentum in a discrete Fourier transform; without
    folding a profile in ``k^2`` would be misapplied at the upper half of each
    axis.  The Nyquist index ``n = L_i/2`` is included by the same formula.
    Use ``np.sum(k_vectors(geo)**2, axis=0)``, or :func:`k_squared`, for ``k^2``.
    """
    shape = _get_total_site(_require_geometry(geo))
    k_axes = _get_folded_pmomenta(shape)
    k_vec = np.empty((4,) + shape, dtype=np.float64)
    for i in range(4):
        # Broadcast the 1-D component i along its own axis and a constant along
        # the other three: the result is a plane whose value is k_i on every mode.
        broadcast = [1, 1, 1, 1]
        broadcast[i] = shape[i]
        k_vec[i] = k_axes[i].reshape(broadcast)
    #
    return k_vec

def k_squared(geo):
    """Return ``k^2 = sum_i k_i^2`` on the mode grid, shaped ``(Lx, Ly, Lz, Lt)``."""
    return np.sum(k_vectors(geo) ** 2, axis=0)

def resolve_k_cut(geo, units):
    """
    Return a momentum threshold expressed in units of ``2*pi/Lx`` (qlat's ``resolve_k_cut``).\n
    ``resolve_k_cut(geo, 2)`` on an ``Lx = 16`` lattice gives ``2*pi/8``, i.e.
    "the second non-trivial momentum".  Convenience helper for writing ``c(k)``
    prescriptions.  The parameter is ``units``, qlat's name: it is part of the
    interface (keyword calls and error messages depend on it).
    """
    shape = _get_total_site(_require_geometry(geo))
    return float(units) * 2.0 * math.pi / float(shape[0])

### -------------------------------------------------------------------

def _get_central_inversion_partner(c):
    """
    Return the even projection ``(c + c[(-n) % L]) / 2``, a NEW array (helper).\n
    ``(-n) % L`` is the index of ``-k``.  For one axis the partner is
    ``c[L - 1 - n]`` reversed and then rolled by one, i.e. index ``0`` stays
    put and the rest reverse, so the order is **flip then roll**.\n
    Averaging with the partner is the least-squares projector onto the even
    fields.  Both the regularizer and the consumer's "is this field even?"
    check are expressed through this one function, so the two cannot disagree.
    """
    partner = c
    for axis in range(4):
        partner = np.roll(np.flip(partner, axis=axis), 1, axis=axis)
    #
    return 0.5 * (c + partner)

def regularize_c_field(c, *, in_place=False, allow_nan="raise"):
    """
    Make a ``c(k)`` field usable by the refresh (qlat's ``regularize_c_field``).\n
    Two conditions are enforced, both required by the physics of
    :func:`partial_refresh_momentum_mode`:\n
    * **Range** ``[0, 1]``, because ``sqrt(1 - c^2)`` must be real: a complex
      scale factor would mix the component fields and the refreshed momentum
      would leave the anti-Hermitian traceless subspace.  Values are **clipped**
      rather than rescaled, so an out-of-range prescription is visibly
      saturated instead of silently distorted in shape.
    * **Evenness** ``c(-k) = c(k)`` under the 4D central inversion
      ``c[(-n) % L]``, which is what keeps the inverse transform real.  It is
      enforced by **averaging** with the partner, the least-squares projector
      onto even fields.\n
    Parameters
    ----------
    c : array_like
        Real 4-D array shaped like ``total_site``.  Complex input is rejected
        (a complex coefficient has no meaning here) unless its imaginary part is
        exactly zero.
    in_place : bool
        Write into ``c`` when it is already a ``float64`` ndarray; otherwise a
        new array is returned.
    allow_nan : {"raise", "zero"}
        ``"raise"`` (default) refuses NaN/inf with an actionable message rather
        than passing them through.  ``"zero"`` maps non-finite entries to
        ``0.0`` -- a frozen mode, i.e. no fresh noise -- and records the count.\n
    Returns
    -------
    numpy.ndarray
        The regularized ``float64`` field (a new array unless ``in_place``).\n
    The diagnostics of the last call are available in
    ``regularize_c_field.last_report``: ``n_nonfinite_zeroed``,
    ``n_clipped_low``, ``n_clipped_high``, ``max_asymmetry_removed``,
    ``c_min``, ``c_max``, ``c_mean`` and ``is_even``.
    """
    arr = np.asarray(c)
    if np.iscomplexobj(arr):
        if np.max(np.abs(arr.imag)) > 0.0:
            raise ValueError(
                "qlat_ref: c(k) must be real - a complex coefficient would make"
                " the inverse transform complex and break the anti-Hermiticity"
                " and tracelessness of the refreshed momentum"
            )
        arr = arr.real
    #
    if arr.ndim != 4:
        raise ValueError(
            f"qlat_ref: the c(k) field must be 4-D shaped (Lx, Ly, Lz, Lt),"
            f" got shape {arr.shape}"
        )
    #
    work = arr.astype(np.float64, copy=not in_place)
    nonfinite = ~np.isfinite(work)
    n_nonfinite = int(np.count_nonzero(nonfinite))
    if n_nonfinite:
        if allow_nan == "raise":
            raise ValueError(
                f"qlat_ref: the c(k) field has {n_nonfinite} non-finite entries"
                " (NaN/inf); fix the generator or pass allow_nan='zero' to map"
                " them to 0.0 explicitly"
            )
        #
        if allow_nan != "zero":
            raise ValueError(
                f"qlat_ref: allow_nan must be 'raise' or 'zero', got {allow_nan!r}"
            )
        #
        work[nonfinite] = 0.0
    #
    n_clipped_low = int(np.count_nonzero(work < 0.0))
    n_clipped_high = int(np.count_nonzero(work > 1.0))
    np.clip(work, 0.0, 1.0, out=work)
    evened = _get_central_inversion_partner(work)
    if work.size:
        max_asymmetry_removed = float(np.max(np.abs(work - evened)))
    else:
        max_asymmetry_removed = 0.0
    #
    work[...] = evened
    regularize_c_field.last_report = {
        "n_nonfinite_zeroed": n_nonfinite,
        "n_clipped_low": n_clipped_low,
        "n_clipped_high": n_clipped_high,
        "max_asymmetry_removed": max_asymmetry_removed,
        "c_min": float(work.min()) if work.size else 0.0,
        "c_max": float(work.max()) if work.size else 0.0,
        "c_mean": float(work.mean()) if work.size else 0.0,
        # Always True by construction: the projection is applied unconditionally
        # just above.  Recorded so a caller can assert it rather than assume it.
        "is_even": True,
    }
    return work

def generate_c_field(geo, func):
    """
    Build a regularized ``c(k)`` field from a prescription (qlat's ``generate_c_field``).\n
    The ``c(k)`` field controls the mode-dependent partial momentum refresh of
    :func:`partial_refresh_momentum_mode`.  This is the supported way to build
    one from a prescription: pass a function of the lattice momentum vector and
    it returns a ready-to-use field.  Any prescription -- analytic, numerically
    tabulated, fitted, or loaded from a file -- enters through the same single
    code path.\n
    Parameters
    ----------
    geo : Geometry
        The lattice geometry.  It is REQUIRED to be a
        :class:`qlat_ref.base.Geometry` (a 4-tuple or a bare shape is rejected
        with ``TypeError``) so that the mode grid is guaranteed to match the
        gauge field the refresh is applied to.
    func : callable
        Called ONCE as ``func(k_vec)``, where ``k_vec`` is the real array of
        shape ``(4, Lx, Ly, Lz, Lt)`` returned by :func:`k_vectors` -- the four
        momentum components in ``total_site`` axis order, each covering
        ``[-pi, pi)``.  The return value must be real and broadcastable to
        ``(Lx, Ly, Lz, Lt)``; complex or non-real output is rejected.\n
        Example -- a two-zone profile (stiff low modes, hard high modes)::
            tc = resolve_k_cut(geo, 1)            # threshold momentum
            c = generate_c_field(geo, lambda kv: np.where(
                    np.sum(kv**2, axis=0) <= tc**2, 0.6, 0.95))\n
        Example -- a Gaussian::
            sigma = resolve_k_cut(geo, 1)
            c = generate_c_field(geo, lambda kv: 0.95 - 0.35 * np.exp(
                    -np.sum(kv**2, axis=0) / (2 * sigma**2)))\n
    Returns
    -------
    numpy.ndarray
        ``float64``, C-contiguous, shaped ``(Lx, Ly, Lz, Lt)``, every entry in
        ``[0, 1]`` and exactly symmetric under the central inversion
        ``c(-k) = c(k)``.\n
    Regularization is always applied (there is no opt-out): the raw callable
    output goes through :func:`regularize_c_field`, which clips it into
    ``[0, 1]`` and averages it with its central-inversion partner.  NaN/inf in
    the callable output raises ``ValueError``; call
    :func:`regularize_c_field` explicitly with ``allow_nan="zero"`` if
    discarding them is intended.  The diagnostics of the last call are in
    ``generate_c_field.last_report`` (the field's, see
    :func:`regularize_c_field`).
    """
    geo = _require_geometry(geo)
    if not callable(func):
        raise TypeError(
            f"qlat_ref: func must be callable, got {type(func).__name__}; use"
            " e.g. lambda kv: np.where(np.sum(kv**2, axis=0) <= tc**2, 0.6, 0.95)"
        )
    #
    shape = _get_total_site(geo)
    raw = np.asarray(func(k_vectors(geo)))
    if np.iscomplexobj(raw):
        if np.max(np.abs(raw.imag)) > 0.0:
            raise ValueError(
                "qlat_ref: func returned a complex c(k) field; it must be real"
            )
        #
        raw = raw.real
    #
    raw = np.asarray(raw, dtype=np.float64)
    try:
        raw = np.array(np.broadcast_to(raw, shape), dtype=np.float64)
    except ValueError as exc:
        raise ValueError(
            f"qlat_ref: func returned shape {raw.shape}, which does not"
            f" broadcast to the lattice shape {shape}"
        ) from exc
    #
    c = regularize_c_field(raw)
    # qlat records the geometry next to the regularization diagnostics, so the
    # report is self-contained; mirror that key for parity.
    generate_c_field.last_report = dict(
        regularize_c_field.last_report, total_site=list(shape)
    )
    # The consumer requires a C-contiguous array and a plain reshape, so the
    # layout is pinned here rather than left to whatever the callable built.
    return np.ascontiguousarray(c)

def c_field_from_file(path):
    """
    Load a ``c(k)`` field from a ``.npy`` file and regularize it (qlat's ``c_field_from_file``).\n
    The stored array must be 4-D and shape-compatible with the target lattice;
    shape checking against a specific :class:`qlat_ref.base.Geometry` is the
    caller's job, because a tabulated field is not tied to one geometry.  Use
    ``np.save`` to write one, and :func:`regularize_c_field` if the stored data
    needs clipping or evenness enforced -- this function applies exactly that
    regularization, so a saved field round-trips.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"c_field file not found: {path}")
    #
    c = np.load(path)
    return regularize_c_field(c)

### -------------------------------------------------------------------

def _get_components(gm):
    """
    Decompose a momentum into its eight real components (helper).\n
    Return ``phi`` of shape ``(8, 4, Lx, Ly, Lz, Lt)`` -- (component, direction,
    mode) -- from
        ``phi_a = -2i Tr(P_mu(x) T_a)``,\n
    real because ``P`` is anti-Hermitian and ``T_a`` Hermitian.  Works on the
    ``(4, Lx, Ly, Lz, Lt, NC, NC)`` array of
    :func:`qlat_ref.field.get_array_from_field`, whose site axes are already
    qlat's x0-fastest order, so no transpose is needed.
    """
    arr = _get_lattice(gm)
    # (4, S..., 3, 3) x (8, 3, 3) -> (8, 4, S...)
    tr = np.einsum("m...ij,aji->am...", arr, _T_A, optimize=True)
    phi = -2.0j * tr
    #
    if phi.size:
        residue = float(np.max(np.abs(phi.imag)))
        scale = float(np.max(np.abs(phi.real)))
        assert residue <= _EPS_EVEN * max(1.0, scale), (
            f"qlat_ref: the momentum is not anti-Hermitian (imaginary residue"
            f" {residue:.3e} against scale {scale:.3e})"
        )
    #
    return np.ascontiguousarray(phi.real)

def _set_components(gm, phi):
    """
    Rebuild a momentum from its eight real components, in place (helper).\n
    Inverse of :func:`_get_components`: writes
    ``P_mu(x) = i sum_a phi_a T_a``, which is anti-Hermitian and traceless by
    construction, so the reconstruction imposes both conditions exactly.
    """
    arr = 1.0j * np.einsum(
        "am...,aij->m...ij", phi.astype(np.complex128), _T_A, optimize=True
    )
    _set_lattice(gm, arr)

### -------------------------------------------------------------------

def partial_refresh_momentum_mode(gm, rng, c_field):
    """
    Apply the mode-dependent partial momentum refresh, in place on ``gm``
    (qlat's ``partial_refresh_momentum_mode``).\n
    Per Fourier mode ``k``, direction ``mu`` and component ``a``,
        ``P_k <- sqrt(1 - c(k)^2) * P_k + c(k) * xi_k``,\n
    where ``xi`` is a Gaussian momentum of the same ensemble drawn from
    ``rng`` via :meth:`qlat_ref.field.GaugeMomentum.set_rand`.  ``c`` is the
    fresh-NOISE weight: ``c(k) = 0`` freezes the mode (no refresh at all) and
    ``c(k) = 1`` fully resamples it; the retained autocorrelation is
    ``sqrt(1 - c^2)``.  Because both terms carry the same per-mode variance the
    Gaussian momentum ensemble is invariant *mode by mode*, so the map is
    stationary and the decorrelation rate is the only thing ``c`` changes.\n
    The physical regime this generalises is the uniform (Ornstein-Uhlenbeck)
    refresh ``P <- sqrt(1 - c^2) P + c xi`` applied in real space; a constant
    ``c(k)`` reproduces it exactly.\n
    Parameters
    ----------
    gm : GaugeMomentum
        The momentum, refreshed **in place**.
    rng : RngState
        The stream for the fresh noise.  Drawn exactly once, and only when the
        refresh actually happens: with ``c`` identically zero **no draw is
        taken** and ``gm`` is left bit-identical, so a frozen mode costs
        nothing and does not disturb the stream.
    c_field : array_like
        Real 4-D ``float64`` array shaped like ``total_site``, in qlat's
        ``(Lx, Ly, Lz, Lt)`` axis order.  It must already satisfy
        ``0 <= c <= 1`` and ``c(-k) = c(k)``; violations raise ``ValueError``
        rather than being repaired -- run :func:`regularize_c_field` first.\n
    Returns
    -------
    (applied, dK, info)
        ``applied`` is whether a refresh happened (``False`` only for the
        frozen field); ``dK`` is ``K(after) - K(before)`` with
        ``K = gm_hamilton_node(gm)``, the kinetic-energy change the driver must
        **exclude** from the Metropolis test; ``info`` is a diagnostics
        dictionary with ``c_min``, ``c_max``, ``c_mean``, ``n_modes``,
        ``n_modes_refreshed``, ``n_modes_frozen``, ``is_even``,
        ``max_asymmetry``, ``n_site``, ``no_op`` and ``path``.\n
    Two code paths, mathematically identical.  For a **constant** ``c`` the
    mixing commutes with the (linear, orthonormal) Fourier transform, so
    applying it directly in real space gives exactly the same result and avoids
    the transform entirely; ``info["path"]`` records which route was taken
    (``"real-space (uniform c)"`` or ``"fourier (mode-dependent c)"``).  Every
    other field is mixed mode by mode in Fourier space.\n
    The transform uses ``norm="ortho"`` (unitary, ``1/sqrt(N)`` each way), the
    only convention under which multiplying per mode by ``sqrt(1 - c^2)`` and
    ``c`` is variance-preserving; with any other normalisation the noise and the
    momentum would carry different factors and the invariance would break.\n
    .. note:: Single process.  The transform is global over the lattice, so this
       requires the whole lattice on one node; qlat uses a distributed FFT.
    """
    geo = gm.geo
    total_site = _get_total_site(geo)
    c = np.asarray(c_field)
    if np.iscomplexobj(c):
        if np.max(np.abs(c.imag)) > 0.0:
            raise ValueError("qlat_ref: c_field must be real")
        #
        c = c.real
    #
    c = np.ascontiguousarray(c, dtype=np.float64)
    if c.size != int(np.prod(total_site)):
        raise ValueError(
            f"qlat_ref: c_field has {c.size} entries but the lattice has"
            f" {int(np.prod(total_site))}; it must be shaped (Lx, Ly, Lz, Lt)"
            f" = {total_site}"
        )
    #
    c = c.reshape(total_site)
    if not np.all(np.isfinite(c)):
        raise ValueError(
            "qlat_ref: c_field has non-finite entries; run regularize_c_field"
            " on it (with allow_nan='zero' if discarding them is intended)"
        )
    #
    if float(c.min()) < 0.0 or float(c.max()) > 1.0:
        raise ValueError(
            f"qlat_ref: c_field must satisfy 0 <= c <= 1 (got"
            f" [{float(c.min())}, {float(c.max())}]); run regularize_c_field on it"
        )
    #
    c_even = _get_central_inversion_partner(c)
    max_asymmetry = float(np.max(np.abs(c - c_even))) if c.size else 0.0
    if max_asymmetry > _EPS_EVEN:
        raise ValueError(
            f"qlat_ref: c_field is not even under k -> -k (max asymmetry"
            f" {max_asymmetry:.3e}); run regularize_c_field on it, which"
            " averages with c[(-n) % L]"
        )
    #
    n_modes = int(c.size)
    info = {
        "c_min": float(c.min()) if c.size else 0.0,
        "c_max": float(c.max()) if c.size else 0.0,
        "c_mean": float(c.mean()) if c.size else 0.0,
        "n_modes": n_modes,
        # Under the noise-weight convention a mode is "refreshed" iff it
        # receives fresh noise, i.e. iff c(k) > 0; c(k) == 0 freezes it.
        "n_modes_refreshed": int(np.count_nonzero(c > 0.0)),
        "n_modes_frozen": int(np.count_nonzero(c <= 0.0)),
        # qlat's key, computed the same way: the uniform fast path is taken iff
        # every mode carries the same coefficient.
        "is_uniform": bool(float(c.min()) == float(c.max())),
        "is_even": True,
        "max_asymmetry": max_asymmetry,
        "n_site": int(np.prod(total_site)),
    }
    if float(c.max()) <= 0.0:
        # Strict no-op: no fresh noise anywhere, so do not consume a draw and
        # leave the momentum bit-identical.  ``info["path"]`` names the route,
        # exactly as qlat does, so a caller can log one value unconditionally.
        info["no_op"] = True
        info["path"] = "no-op (c == 0)"
        return False, 0.0, info
    #
    info["no_op"] = False
    k_before = float(gm_hamilton_node(gm))
    if float(c.min()) == float(c.max()):
        c_uniform = float(c.min())
        info["path"] = "real-space (uniform c)"
        keep = math.sqrt(max(0.0, 1.0 - c_uniform * c_uniform))
        xi = GaugeMomentum(geo)
        xi.set_rand(rng, 1.0)
        _set_lattice(gm, keep * _get_lattice(gm) + c_uniform * _get_lattice(xi))
    else:
        info["path"] = "fourier (mode-dependent c)"
        keep = np.sqrt(np.maximum(0.0, 1.0 - c * c))
        phi = _get_components(gm)
        xi = GaugeMomentum(geo)
        xi.set_rand(rng, 1.0)
        xi_phi = _get_components(xi)
        # The last four axes of (8, 4, Lx, Ly, Lz, Lt) are the lattice axes.
        phi_k = np.fft.fftn(phi, axes=(-4, -3, -2, -1), norm="ortho")
        xi_k = np.fft.fftn(xi_phi, axes=(-4, -3, -2, -1), norm="ortho")
        mixed_k = keep * phi_k + c * xi_k
        phi_new = np.fft.ifftn(mixed_k, axes=(-4, -3, -2, -1), norm="ortho")
        _set_components(gm, np.ascontiguousarray(phi_new.real))
    #
    k_after = float(gm_hamilton_node(gm))
    return True, k_after - k_before, info

### -------------------------------------------------------------------

def _get_fix_stats(fix_records):
    """Summarize the per-step gauge-fixing records into ``info_out`` keys (helper)."""
    if not fix_records:
        return {
            "n_fix": 0,
            "fix_n_iter_total": 0,
            "fix_n_iter_max": 0,
            "fix_n_converged": 0,
            "fix_divergence_max": 0.0,
        }
    #
    n_iter_list = [int(rec["n_iter"]) for rec in fix_records]
    divergence_list = [float(rec["divergence"]) for rec in fix_records]
    n_converged = sum(1 for rec in fix_records if rec["converged"])
    return {
        "n_fix": len(fix_records),
        "fix_n_iter_total": int(sum(n_iter_list)),
        "fix_n_iter_max": int(max(n_iter_list)),
        "fix_n_converged": int(n_converged),
        "fix_divergence_max": float(max(divergence_list)),
    }

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
    """
    Run one Canonical Gauge-Fixed HMC trajectory (qlat's
    ``run_hmc_cgfhmc_pure_gauge``).\n
    Identical to :func:`qlat_ref.hmc_utils.run_hmc_pure_gauge` except that every
    MD step is followed by a Landau gauge-fixing projection of **both** ``gf``
    and ``gm`` and by a mode-dependent partial momentum refresh.  Draw a Gaussian
    momentum, run the force-gradient MD evolution with those two insertions, and
    apply the Metropolis step on ``delta_h_accept = delta_h - dK_sum``.  Return
    ``(flag, delta_h)``.  On accept ``gf`` is updated **in place** (qlat's
    convention ``gf @= gf0``); the trial field is an internal copy, so a
    rejected trajectory leaves ``gf`` bit-unchanged.  Take ``gf.copy()``
    **before** the call when the pre-trajectory configuration is needed.\n
    The argument order and defaults are those of
    :func:`qlat_ref.hmc_utils.run_hmc_pure_gauge` -- positional
    ``(gf, ga, traj, rs)``, then the keyword-only ``is_reverse_test``,
    ``n_step``, ``md_time`` and ``is_always_accept`` -- with the CGFHMC-specific
    keyword-only parameters after them, so the two drivers are interchangeable.\n
    Parameters
    ----------
    gf : GaugeField
        Input/output gauge field, mutated in place on accept.
    ga : GaugeAction
        The gauge action (``beta``, ``c1``).
    traj : int
        Trajectory number; the driver splits the RNG with ``str(traj)``.
    rs : RngState
        Driver-level seed on top of which the trajectory and consumer splits
        are taken.
    c_field : array_like
        The mode-dependent fresh-noise weight, real 4-D shaped like
        ``total_site`` in ``(Lx, Ly, Lz, Lt)`` order, with ``0 <= c <= 1`` and
        ``c(-k) = c(k)``.  Build it with :func:`generate_c_field` (or
        :func:`regularize_c_field` around a tabulated array).  Required: there
        is no prescription it could sensibly default to.
    is_reverse_test : bool
        **Not supported.**  The gauge-fixing projection discards the phase-space
        information a reverse test relies on, so a CGFHMC trajectory is not
        time reversible; passing ``True`` raises ``NotImplementedError``.  The
        keyword is kept so the signature matches ``run_hmc_pure_gauge``.
    n_step : int
        Number of MD steps per trajectory (qlat's default ``6``).
    md_time : float
        Total MD time (qlat's default ``1.0``); negative reverses the
        evolution.
    is_always_accept : bool
        Update ``gf`` even when the Metropolis test rejects.
    gauge_fix_max_iter : int
        Cap on the gauge-fixing sweeps **per MD step**.  Much smaller than the
        ``fix_landau_gauge`` default, because the field is re-fixed every step
        and starts already near Landau gauge; the first step of a cold chain is
        the expensive one.
    gauge_fix_eps : float
        Gauge-fixing convergence threshold on the RMS divergence.
    gauge_fix_alpha : float
        Gauge-fixing step size.
    info_out : dict, optional
        Filled in place with the diagnostics of the trajectory when given:
        ``dK_sum``, ``n_refresh_applied``, ``n_step``, ``n_fix``,
        ``fix_n_iter_total``, ``fix_n_iter_max``, ``fix_n_converged``,
        ``fix_divergence_max``, ``delta_h_raw`` and ``delta_h_accept``.\n
    Returns
    -------
    (flag, delta_h)
        ``flag`` is whether the Metropolis test accepted; ``delta_h`` is the
        **raw** energy violation of the trajectory
        (``gm_hamilton_node(gm) + gf_hamilton_node(gf, ga)`` at the end minus
        at the start), always the trajectory's own even when
        ``is_always_accept`` forces the update.  It is *not* the
        ``delta_h - dK_sum`` the Metropolis test used.\n
    .. note:: The RNG tree is qlat's, with one extra consumer:
       ``rs.split(str(traj))`` splits into ``"set_rand_gauge_momentum"`` for the
       momentum, ``"partial_refresh_momentum_base"`` for the refresh (which is
       itself split per MD step by ``f"partial_refresh_momentum.{i}"``) and
       ``"metropolis_accept"`` for the accept draw.  The streams are
       independent on purpose, so a rejected trajectory does not disturb the
       momentum or refresh streams.
    """
    if is_reverse_test:
        # Byte-identical to qlat's message: the wording is part of the parity
        # contract (a live-qlat test compares the two strings), so it carries no
        # ``qlat_ref:`` prefix even though the other errors here do.
        raise NotImplementedError(
            "CGFHMC applies a gauge-fixing projection each MD step and is not "
            "time-reversible; is_reverse_test is not supported"
        )
    #
    rs = rs.split(f"{traj}")
    geo = gf.geo
    gf0 = GaugeField(geo)
    gf0 @= gf
    gm = GaugeMomentum(geo)
    gm.set_rand(rs.split("set_rand_gauge_momentum"), 1.0)
    refresh_rs = rs.split("partial_refresh_momentum_base")
    # Resolve the c(k) field once: the same array is reused for every step, and
    # its validation (range, evenness) happens in the refresh on the first use.
    c = np.asarray(c_field)
    energy = gm_hamilton_node(gm) + gf_hamilton_node(gf0, ga)
    dt = md_time / n_step
    lam = 0.5 * (1.0 - 1.0 / math.sqrt(3.0))
    theta = (2.0 - math.sqrt(3.0)) / 48.0
    ttheta = theta * dt * dt * dt
    dk_sum = 0.0
    n_refresh_applied = 0
    fix_records = []
    # Each step is the symmetric ``lam, (1-2*lam), lam`` drift sandwich, so the
    # step boundaries fall exactly on multiples of ``dt`` and the fixing below
    # sits at the same phase point (``drift so far = (i+1)*dt``) on every step.
    for i in range(n_step):
        gf_evolve(gf0, gm, lam * dt)
        gm_evolve_fg_pure_gauge(gm, gf0, ga, 4.0 * ttheta / dt, 0.5 * dt)
        gf_evolve(gf0, gm, (1.0 - 2.0 * lam) * dt)
        gm_evolve_fg_pure_gauge(gm, gf0, ga, 4.0 * ttheta / dt, 0.5 * dt)
        gf_evolve(gf0, gm, lam * dt)
        #
        # End of MD step i, after its last sub-update and before the next
        # step's first: project both the links and the momentum onto Landau
        # gauge.  ``fix_landau_gauge`` does not mutate gf0; ``gt * gf0`` is the
        # fixed field and ``gt * gm`` the corresponding same-site rotation.
        gt, fix_info = fix_landau_gauge(
            gf0,
            max_iter=gauge_fix_max_iter,
            eps=gauge_fix_eps,
            alpha=gauge_fix_alpha,
        )
        gf0 @= gt * gf0
        gm @= gt * gm
        fix_records.append(fix_info)
        applied, dk_step, _refresh_info = partial_refresh_momentum_mode(
            gm, refresh_rs.split(f"partial_refresh_momentum.{i}"), c
        )
        if applied:
            n_refresh_applied += 1
            dk_sum += dk_step
    #
    gf0.unitarize()
    delta_h = gm_hamilton_node(gm) + gf_hamilton_node(gf0, ga) - energy
    # The refresh replaces momentum by design, so its kinetic change is not
    # charged to the Metropolis test.
    delta_h_accept = delta_h - dk_sum
    flag, accept_prob = metropolis_accept(
        delta_h_accept, traj, rs.split("metropolis_accept")
    )
    if flag or is_always_accept:
        _displayln_info(f"run_hmc_cgfhmc_pure_gauge: update gf (traj={traj})")
        gf @= gf0
    #
    if info_out is not None:
        info_out.update(
            {
                "dK_sum": float(dk_sum),
                "n_refresh_applied": int(n_refresh_applied),
                "n_step": int(n_step),
                **_get_fix_stats(fix_records),
                "delta_h_raw": float(delta_h),
                "delta_h_accept": float(delta_h_accept),
            }
        )
    #
    _displayln_info(
        f"run_hmc_cgfhmc_pure_gauge: traj={traj}"
        f" delta_h={delta_h:.16f} delta_h_accept={delta_h_accept:.16f}"
        f" dK_sum={dk_sum:.16f} n_refresh={n_refresh_applied:d}"
        f" accept_prob={accept_prob * 100.0:.1f}%"
    )
    return flag, delta_h
