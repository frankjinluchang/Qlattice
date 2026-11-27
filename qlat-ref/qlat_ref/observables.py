"""
Module ``qlat_ref.observables``
===============================\n
Gauge functionals, per-time-slice Coulomb functionals and the RMS
divergences that monitor the gauge condition; the unitarity checks live in
:mod:`qlat_ref.mat`.\n
Documentation: ``docs/qlat-ref/qlat_observables.md``\n
.. note:: Update the documentation when updating this source file.\n
Input types
-----------\n
Every function here accepts **either** a :class:`qlat_ref.field.GaugeField`
**or** a plain NumPy link field of shape::\n
    U.shape == (4, Nt, Nx, Ny, Nz, NC, NC)\n
because the two are equivalent through the documented array bridge
(``np.asarray(gf)``).  The plain-array path is what
``examples-py/gauge-fixing-cross-check.py`` and the old ``qlat_ref.gauge``
docs use; the field path is what ``qlat``'s own API uses.
"""

import numpy as np

from .base import NC
from .base import _SPATIAL_DIRS
from .base import get_dirs

from .field import GaugeField
from .field import GaugeTransform
from .field import _TIME_AXIS
from .field import get_array_from_field
from .field import get_local_gauge_gradient
from .mat import get_matrix_adjoint
from .mat import get_unitarity_error

### -------------------------------------------------------------------

def get_link_array(gf):
    """
    Return the ``(4, Nt, Nx, Ny, Nz, NC, NC)`` array of a gauge field.\n
    ``gf`` may be a :class:`qlat_ref.field.GaugeField`, a
    :class:`qlat_ref.field.GaugeMomentum` or a plain array; this is the single
    place where the two input conventions are reconciled.
    """
    if isinstance(gf, GaugeField):
        return get_array_from_field(gf)
    arr = np.asarray(gf, dtype=np.complex128)
    if arr.ndim != 7 or arr.shape[0] != 4:
        raise ValueError(
            f"qlat_ref: expected a gauge field with shape (4, Nt, Nx, Ny, Nz,"
            f" NC, NC), got shape {arr.shape}"
        )
    return arr

def get_site_array(gf):
    """
    Return the ``(Nt, Nx, Ny, Nz, NC, NC)`` array of a gauge transformation.\n
    ``gf`` may be a :class:`qlat_ref.field.GaugeTransform` or a plain array.
    """
    from .field import GaugeTransform
    #
    if isinstance(gf, GaugeTransform):
        geo = gf.geo
        total_site = geo.total_site.to_tuple()
        return np.asarray(gf).reshape(tuple(total_site) + (NC, NC), order="F")
    arr = np.asarray(gf, dtype=np.complex128)
    if arr.ndim != 6 or arr.shape[-2:] != (NC, NC):
        raise ValueError(
            f"qlat_ref: expected a site field with shape (Nt, Nx, Ny, Nz, NC,"
            f" NC), got shape {arr.shape}"
        )
    return arr

### -------------------------------------------------------------------

def _get_trace_field(U, mu):
    """Return ``Re Tr U_mu(x)`` on every site (helper of the functionals)."""
    return np.trace(U[mu], axis1=-2, axis2=-1).real

def get_gauge_functional(gf, dirs):
    """
    Return the normalized gauge functional,\n
        ``F = sum_{x, mu in dirs} Re Tr U_mu(x) / (V * len(dirs) * NC)``,\n
    where ``V`` is the total number of sites.  Landau gauge uses
    ``dirs = (0, 1, 2, 3)``; the Coulomb functional uses the spatial
    directions ``dirs = (0, 1, 2)`` -- component 3 is time in qlat's
    convention (see :data:`qlat_ref.base._SPATIAL_DIRS`).\n
    The signature mirrors ``qlat.gauge_fixing.get_gauge_functional(gf, dirs)``
    (both arguments positional).
    """
    U = get_link_array(gf)
    dirs = get_dirs(dirs)
    volume = int(np.prod(U.shape[1:5]))
    total = 0.0
    for mu in dirs:
        total += float(np.sum(_get_trace_field(U, mu)))
    return total / (volume * len(dirs) * NC)

def get_coulomb_functional_list(U):
    """
    Return the Coulomb functional of each time slice.\n
    The Coulomb functional sums over the spatial links only, so it splits into
    one independent ``3``-dimensional Landau functional per time slice.\n
    Time is component ``3`` of ``total_site`` (the qlat convention), so after
    indexing ``U[mu]`` -- which drops the direction axis -- the lattice axis of
    direction ``mu`` is array axis ``mu`` and the time axis is ``3``.  The
    spatial directions are therefore ``(0, 1, 2)`` and ``Lt`` is
    ``U.shape[4]`` of the link array (``_TIME_AXIS``).
    """
    U = get_link_array(U)
    n_t = U.shape[_TIME_AXIS]
    n_spatial = int(np.prod([U.shape[1 + mu] for mu in _SPATIAL_DIRS]))
    total = np.zeros(n_t, dtype=np.float64)
    for mu in _SPATIAL_DIRS:
        total += _get_trace_field(U, mu).sum(axis=_SPATIAL_DIRS)
    return (total / (n_spatial * len(_SPATIAL_DIRS) * NC)).tolist()

### -------------------------------------------------------------------

def _get_gradient_quad(D):
    """Return ``Tr[D(x) D(x)^dagger]`` on every site (helper of the divergences)."""
    return np.trace(D @ get_matrix_adjoint(D), axis1=-2, axis2=-1).real

def get_divergence_from_gradient(D):
    """Return the RMS divergence ``theta`` for a local gradient field ``D``."""
    D = np.asarray(D, dtype=np.complex128)
    volume = int(np.prod(D.shape[:4]))
    return float(np.sqrt(np.sum(_get_gradient_quad(D)) / (volume * NC)))

def get_gauge_divergence(gf, dirs):
    """
    Return the RMS lattice divergence ``theta`` of the gauge field.\n
    ``theta`` is ``sqrt(sum_x Tr[D(x) D(x)^dagger] / (V * NC))`` with ``D`` the
    local gradient of the functional; it vanishes exactly when the gauge
    condition ``D(x) = 0`` holds.  This is the convergence measure of
    :func:`qlat_ref.gauge_fixing.fix_landau_gauge` and
    :func:`qlat_ref.gauge_fixing.fix_coulomb_gauge`.\n
    The signature mirrors ``qlat.gauge_fixing.get_gauge_divergence(gf, dirs)``
    (both arguments positional).
    """
    return get_divergence_from_gradient(get_local_gauge_gradient(gf, dirs=dirs))

def get_coulomb_divergence_list(U):
    """
    Return the RMS divergence of each time slice for the Coulomb gauge.\n
    Each time slice is an independent ``3``-dimensional Landau problem, so the
    per-slice values are the natural convergence monitor of Coulomb gauge
    fixing.  The time axis is ``_TIME_DIR == 3`` (component 3 of
    ``total_site``), the same convention as :func:`get_coulomb_functional_list`.
    """
    D = get_local_gauge_gradient(U, dirs=_SPATIAL_DIRS)
    n_spatial = int(np.prod([D.shape[mu] for mu in _SPATIAL_DIRS]))
    per_slice = _get_gradient_quad(D).sum(axis=_SPATIAL_DIRS)
    return np.sqrt(per_slice / (n_spatial * NC)).tolist()

### -------------------------------------------------------------------

def get_unitarity_error_typed(gf):
    """
    Return the maximal ``|U U^dagger - 1|`` matrix element of the field.\n
    This is ``qlat.gauge_fixing.get_unitarity_error(gf)``: the typed entry
    point that accepts a :class:`qlat_ref.field.GaugeField` (over every site
    and every link direction) **or** a
    :class:`qlat_ref.field.GaugeTransform` (over every site).\n
    Any other type -- a :class:`qlat_ref.field.GaugeMomentum` (whose
    anti-Hermitian elements are not unitarity defects), a raw array, a plain
    list -- is rejected with ``TypeError``, exactly like ``qlat``'s version.\n
    The array-level helper :func:`qlat_ref.mat.get_unitarity_error` still takes
    any array; it is the generic matrix routine this entry point is built on.
    """
    if isinstance(gf, GaugeField):
        return get_unitarity_error(get_array_from_field(gf))
    if isinstance(gf, GaugeTransform):
        return get_unitarity_error(get_site_array(gf))
    raise TypeError(
        "get_unitarity_error: expected a GaugeField or a GaugeTransform, got "
        + type(gf).__name__
    )
