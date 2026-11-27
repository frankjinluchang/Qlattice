"""
Module ``qlat_ref.gauge_fixing``
================================\n
Landau and Coulomb gauge fixing: the shared Jacobi steepest-descent core and
the public drivers that return the accumulated gauge transformation.\n
Documentation: ``docs/qlat-ref/qlat_gauge.md``\n
.. note:: Update the documentation when updating this source file.\n
Public interface
----------------\n
The free functions mirror ``qlat.gauge_fixing`` exactly -- same names, same
positional arguments, same defaults and the same ``(gt, info)`` return::\n
    fix_gauge(gf, dirs, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)
    fix_landau_gauge(gf, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)
    fix_coulomb_gauge(gf, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)\n
``gf`` may be a :class:`qlat_ref.field.GaugeField` **or** a plain NumPy link
field of shape ``(4, Nt, Nx, Ny, Nz, NC, NC)``: the two are equivalent
through the documented array bridge, and accepting both keeps the historical
``qlat_ref.gauge`` call style (plain arrays) working while giving the field
types the qlat API.  The returned ``gt`` is a
:class:`qlat_ref.field.GaugeTransform` when ``gf`` is a ``GaugeField`` and a
plain ``(Nt, Nx, Ny, Nz, NC, NC)`` array when ``gf`` is a plain array.\n
.. note:: **Intentional deviation from qlat: Coulomb gauge uses the spatial
   directions ``dirs = (0, 1, 2)``, not qlat's ``(1, 2, 3)``.**
   Component 3 is time throughout qlat: ``Geometry::total_site`` is
   ``(Lx, Ly, Lz, Lt)``, ``Geometry::spatial_volume`` divides out component 3,
   ``get_psel_tslice(total_site, *, t_dir=3)`` slices the time axis, and
   ``qlat_gpt.gauge_fix_coulomb`` separates the time axis with
   ``g.separate(u, 3)``.  The spatial directions are therefore ``0, 1, 2``.
   qlat's ``gf_fix_coulomb_gauge`` (``qlat/qlat/lib/gauge-fixing.cpp``) instead
   uses ``{1, 2, 3}``, i.e. it fixes the three directions that include time and
   excludes the spatial direction ``0``.  That is qlat-issue-8 in
   ``tmp/qlat-ref-reorg/qlat-issues.md``.  ``qlat_ref`` implements the correct
   set; ``q.fix_coulomb_gauge`` is *not* a reference for Coulomb gauge, and
   the cross-check compares against ``q.fix_gauge(gf, dirs=[0, 1, 2])`` (the
   same algorithm with explicit directions) instead.  The deviation is
   deliberate, documented and pinned by ``tests/test_gauge_fixing.py`` and
   ``tests/test_qlat_parity.py`` (listed in its ``DEVIATIONS`` allowlist).
"""

import numpy as np

from .base import NC
from .base import _ALL_DIRS
from .base import _SPATIAL_DIRS
from .base import get_dirs

from .field import GaugeField
from .field import GaugeTransform
from .field import apply_gauge_transform
from .field import get_local_gauge_gradient

from .mat import get_su3_exp

from .observables import get_divergence_from_gradient
from .observables import get_gauge_functional
from .observables import get_link_array

### -------------------------------------------------------------------

def _fix_gauge(U, *, dirs, gauge_name, max_iter, eps, alpha, verbose):
    """
    Shared iteration core of Landau and Coulomb gauge fixing (internal).\n
    The local gauge transformation of one sweep is\n
        ``g(x) = exp(alpha * D(x))``   (``D`` from
        :func:`qlat_ref.field.get_local_gauge_gradient`),\n
    applied to every site at once (Jacobi sweep):\n
        ``U_mu(x) <- g(x) U_mu(x) g(x + mu)^dagger``.\n
    The iteration stops when the RMS divergence is at most ``eps`` or after
    ``max_iter`` sweeps.\n
    Return ``(gt, U_fixed, info)``: ``gt`` is the accumulated gauge
    transformation of all sweeps (new steps multiplied on the left, so that
    ``apply_gauge_transform(U, gt)`` is exactly ``U_fixed``), ``U_fixed`` is
    the gauge-fixed field and ``info`` is the convergence history.  ``gt`` is
    only defined up to the transformations that leave the gauge condition
    invariant (see :func:`fix_landau_gauge`).\n
    ``U`` may be a plain array or a :class:`qlat_ref.field.GaugeField`; the
    returned ``gt`` and ``U_fixed`` follow the same convention.
    """
    input_field = U if isinstance(U, GaugeField) else None
    U_in = get_link_array(U)
    U = np.array(U_in, dtype=np.complex128, copy=True)
    if alpha <= 0.0:
        raise ValueError(f"qlat_ref: alpha must be positive: {alpha}")
    dirs = get_dirs(dirs)
    # Accumulated gauge transformation: gt(x) is the site field such that
    # apply_gauge_transform(U_input, gt) is the gauge-fixed field.
    gt = np.broadcast_to(np.eye(NC, dtype=np.complex128), U.shape[1:]).copy()
    # Evaluate the state before the first sweep and refresh it after each one.
    D = get_local_gauge_gradient(U, dirs=dirs)
    functional = get_gauge_functional(U, dirs)
    divergence = get_divergence_from_gradient(D)
    functional_list = []
    divergence_list = []
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        functional_list.append(functional)
        divergence_list.append(divergence)
        if verbose:
            print(
                f"qlat_ref: {gauge_name} sweep {n_iter}:"
                f" functional = {functional:.15e}"
                f" divergence = {divergence:.3e}"
            )
        if divergence <= eps:
            converged = True
            break
        g_step = get_su3_exp(alpha * D)
        U = apply_gauge_transform(U, g_step)
        # Sequential application: the new step acts first, so it multiplies
        # the accumulated transformation on the left.
        gt = g_step @ gt
        D = get_local_gauge_gradient(U, dirs=dirs)
        functional = get_gauge_functional(U, dirs)
        divergence = get_divergence_from_gradient(D)
    info = {
        "gauge": gauge_name,
        "dirs": dirs,
        "converged": converged,
        "n_iter": n_iter,
        "max_iter": max_iter,
        "eps": eps,
        "alpha": alpha,
        "functional": functional,
        "divergence": divergence,
        "functional_list": functional_list,
        "divergence_list": divergence_list,
    }
    if input_field is not None:
        geo = input_field.geo
        U_fixed = _make_field(GaugeField, geo, U, multiplicity=4)
        gt = _make_field(GaugeTransform, geo, gt, multiplicity=1)
    else:
        U_fixed = U
    return gt, U_fixed, info

### -------------------------------------------------------------------

def _make_field(cls, geo, arr, multiplicity):
    """Wrap a plain array into a field of ``cls`` on ``geo`` (helper)."""
    arr = np.asarray(arr, dtype=np.complex128)
    f = cls(geo)
    if arr.ndim == 7:
        # link field (4, Nt, Nx, Ny, Nz, NC, NC): direction axis to the middle
        flat = np.moveaxis(arr, 0, -3).reshape(
            (geo.local_volume, multiplicity, NC, NC), order="F"
        )
    else:
        # site field (Nt, Nx, Ny, Nz, NC, NC)
        flat = arr.reshape((geo.local_volume, multiplicity, NC, NC), order="F")
    f._arr[...] = flat
    return f

### -------------------------------------------------------------------

def fix_gauge(gf, dirs, max_iter=10000, eps=1.0e-10, alpha=0.1, verbose=False):
    """
    Plain local steepest-descent gauge fixing with the link directions
    ``dirs``.\n
    Does not modify ``gf``.  Returns ``(gt, info)`` where ``gt`` is such that
    ``gt * gf`` (equivalently ``apply_gauge_transform(gf, gt)``) is the
    gauge-fixed field, and ``info`` is a dictionary with the convergence
    history: ``gauge``, ``dirs``, ``converged``, ``n_iter``, ``max_iter``,
    ``eps``, ``alpha``, ``functional``, ``divergence``,
    ``functional_list`` and ``divergence_list``.\n
    ``gt`` is not unique: it depends on the convergence path and is defined
    only up to the residual stabiliser of the fixed field.\n
    Mirrors ``qlat.gauge_fixing.fix_gauge(gf, dirs, max_iter, eps, alpha,
    verbose)``; ``info["gauge"]`` is ``"custom"`` as in qlat.
    """
    gt, _, info = _fix_gauge(
        gf,
        dirs=dirs,
        gauge_name="custom",
        max_iter=max_iter,
        eps=eps,
        alpha=alpha,
        verbose=verbose,
    )
    return gt, info

def fix_landau_gauge(gf, max_iter=10000, eps=1.0e-10, alpha=0.1, verbose=False):
    """
    Return ``(gt, info)`` where ``gt * gf`` is in Landau gauge.\n
    Landau gauge maximizes the functional\n
        ``F = (1 / (V * 4 * NC)) sum_{x, mu} Re Tr U_mu(x)``\n
    over all gauge transformations; the plain local iteration of
    :func:`_fix_gauge` uses all four directions.  The iteration stops when the
    RMS divergence is at most ``eps`` (or after ``max_iter`` sweeps); the
    returned ``info`` dictionary records the convergence history.\n
    ``gt`` is not unique: the fixing path (``alpha``, ``max_iter``, rounding)
    selects one representative, and any transformation that leaves the gauge
    condition invariant (for Landau gauge a global center element) multiplies
    it without changing the fixed field.  Compare fixed *fields*, not the
    transformations.\n
    Mirrors ``qlat.gauge_fixing.fix_landau_gauge(gf, max_iter, eps, alpha,
    verbose)``.
    """
    gt, _, info = _fix_gauge(
        gf,
        dirs=_ALL_DIRS,
        gauge_name="landau",
        max_iter=max_iter,
        eps=eps,
        alpha=alpha,
        verbose=verbose,
    )
    return gt, info

def fix_coulomb_gauge(gf, max_iter=10000, eps=1.0e-10, alpha=0.1, verbose=False):
    """
    Return ``(gt, info)`` where ``gt * gf`` is in Coulomb gauge.\n
    Coulomb gauge maximizes the functional\n
        ``F_C = (1 / (V * 3 * NC)) sum_{x, i} Re Tr U_i(x)``\n
    over gauge transformations ``g(t, x)`` that act independently on every
    time slice; the iteration therefore uses the spatial directions only
    (``i = 0, 1, 2``, because component 3 is time) and is a separate 3D Landau
    fixing per time slice.  The
    gauge transformation is still applied to all four link directions, so the
    result is a complete 4D gauge transformation.  The iteration stops when
    the RMS divergence is at most ``eps`` (or after ``max_iter`` sweeps).\n
    ``gt`` is not unique for the same reasons as in :func:`fix_landau_gauge`;
    compare fixed *fields*, not the transformations.\n
    The name, signature and ``(gt, info)`` return mirror
    ``qlat.gauge_fixing.fix_coulomb_gauge``, but the spatial directions are the
    **correct** ``(0, 1, 2)`` rather than qlat's ``(1, 2, 3)``; this is the one
    intentional deviation of this function and it fixes qlat-issue-8.  See the
    module docstring.
    """
    gt, _, info = _fix_gauge(
        gf,
        dirs=_SPATIAL_DIRS,
        gauge_name="coulomb",
        max_iter=max_iter,
        eps=eps,
        alpha=alpha,
        verbose=verbose,
    )
    return gt, info
