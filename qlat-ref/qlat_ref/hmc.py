"""
Module ``qlat_ref.hmc``
=======================\n
The Hamiltonian-Monte-Carlo building blocks of a pure-gauge lattice: the
Gaussian initialisation of a :class:`qlat_ref.field.GaugeMomentum`, the gauge
Hamiltonian (Wilson plaquette plus optional ``1 x 2`` rectangle improvement),
the force, the matrix-exponential evolution of the links, and the adjoint
action of a gauge transformation on a momentum.\n
The names, argument order, defaults and mutation semantics are those of
``qlat.hmc`` (a *subset* of it: see ``docs/qlat-ref/qlat_ref_overview.md``),
so a reader can diff this module against ``qlat/qlat/hmc.pyx`` and
``qlat/qlat/lib/hmc.cpp`` one function at a time.  The trajectory driver layer
is the sibling :mod:`qlat_ref.hmc_utils`, mirroring ``qlat/qlat/hmc_utils.py``.
The gauge action value itself is
:func:`qlat_ref.gauge_action.get_gauge_action` (the one implementation of the
plaquette-plus-rectangle sum); :func:`gf_hamilton_node` is the qlat-named entry
point that returns it.\n
Conventions
-----------\n
* A link field ``U`` (and a momentum field ``P``) is worked on in the
  ``(4, Nt, Nx, Ny, Nz, NC, NC)`` layout of
  :func:`qlat_ref.field.get_array_from_field`; the field classes keep qlat's
  ``(local_volume, 4, NC, NC)`` buffer.  The two are converted at the field
  boundary only, so every formula below is plain ``numpy``.
* A hop by one site in direction ``mu`` is ``numpy.roll(a, -1, axis=mu)`` on a
  site array, matching the periodic lattice of
  :class:`qlat_ref.base.Geometry`.
* ``P_mu(x)`` is anti-Hermitian and traceless, and the kinetic term is
  ``K = sum_{x, mu} (-1/2) Tr(P_mu(x)^2)``.\n
Mutation semantics (qlat's, see the reference design):
:func:`set_gm_force`, :func:`set_gm_force_dual`, :func:`gf_evolve`,
:func:`gf_evolve_dual` and :func:`set_rand_gauge_momentum` fill their first
argument **in place** and return ``None``;
:func:`gm_apply_gauge_transformation` returns a **new** momentum and leaves its
inputs untouched; :func:`gm_hamilton_node` and :func:`gf_hamilton_node`
return a ``float``.\n
.. note:: Single process.  qlat distributes the lattice over MPI nodes, so its
   ``*_node`` quantities are node-local and the drivers sum them with
   ``glb_sum_double``; here node-local is global, so the sums are the global
   ones.  The names keep qlat's ``_node`` suffix because they are qlat's names
   (``gm_hamilton``/``gf_hamilton`` do not exist in qlat's Python API).\n
.. note:: No communication: :func:`qlat_ref.hmc.set_gm_force` evaluates the
   staples by shifting the single-process link array, which is exactly what
   qlat's ``refresh_expanded`` boundary exchange makes available to its
   ``_no_comm`` kernels.\n
Documentation: ``docs/qlat-ref/qlat_hmc.md``\n
.. note:: Update the documentation when updating this source file.
"""

import numpy as np

from .base import NC

from .field import get_array_from_field
from .field import gm_apply_gauge_transformation as _gm_apply_gauge_transformation

from .gauge_action import get_gauge_action

from .mat import get_matrix_adjoint
from .mat import get_su3_exp
from .mat import get_traceless_antihermitian_part
from .mat import get_unitarized

### -------------------------------------------------------------------
### Private array bridge between the field classes and the (4, ...) layout
### -------------------------------------------------------------------

def _get_lattice(f):
    """
    Return the ``(4, Nt, Nx, Ny, Nz, NC, NC)`` working array of a field (helper).\n
    This is :func:`qlat_ref.field.get_array_from_field`, the one documented
    conversion between qlat's field buffer
    ``(local_volume, multiplicity, NC, NC)`` and the ``(4, ...)`` layout used by
    the kernels of this module.  The returned array is a fresh copy, so writing
    into it leaves the field unchanged;
    :func:`_set_lattice` writes it back.
    """
    if f.geo is None:
        raise ValueError("qlat_ref: the field was created without a Geometry")
    #
    if f.multiplicity != 4:
        raise ValueError(
            f"qlat_ref: expected a link field of multiplicity 4, got {f.multiplicity}"
        )
    #
    return get_array_from_field(f)

def _check_output_geo(f, geo):
    """
    Make ``f`` a multiplicity-4 field on ``geo``, matching qlat's ``init(geo)`` (helper).\n
    The force output arguments of qlat (``set_gm_force``, ``set_gm_force_dual``)
    are initialised by the callee with ``gm_force.init(geo)``, so qlat accepts a
    field created without a geometry, or one carrying another geometry, and
    silently replaces its content and geometry.  This reproduces that: the field
    is re-initialised only when it is not already a multiplicity-4 field on
    ``geo``, so the common case keeps the caller's buffer (and any live
    ``np.asarray`` view of it) intact.
    """
    if f.multiplicity != 4:
        raise ValueError(
            f"qlat_ref: expected a link field of multiplicity 4, got {f.multiplicity}"
        )
    #
    if f.geo is None or f.geo != geo:
        f.init_from_geo(geo, 4)

def _set_lattice(f, arr):
    """Write a ``(4, ...)`` working array into ``f``, in place (helper)."""
    geo = f.geo
    if geo is None:
        raise ValueError("qlat_ref: the field was created without a Geometry")
    #
    n = geo.local_volume
    mult = f.multiplicity
    f[:] = np.moveaxis(np.asarray(arr, dtype=np.complex128), 0, -3).reshape(
        (n, mult, NC, NC), order="F"
    )

### -------------------------------------------------------------------
### Lattice paths and staples (qlat's qcd-acc.h / lib/hmc.cpp)
### -------------------------------------------------------------------

def _get_shifted(a, offset):
    """
    Return the site field ``a`` translated by ``offset`` (helper).\n
    ``a`` has shape ``(Nt, Nx, Ny, Nz, NC, NC)`` and ``offset`` is four signed
    hops in that coordinate order, so ``offset[mu] = +1`` means the field at
    ``x + mu`` (the periodic lattice is a ``numpy.roll`` by ``-offset``).
    """
    ret = a
    for axis, n in enumerate(offset):
        if n != 0:
            ret = np.roll(ret, -n, axis=axis)
    #
    return ret

def _get_wilson_line(U, path):
    """
    Return the product of links along a signed-direction ``path`` (helper).\n
    Mirrors qlat's ``gf_wilson_line_no_comm``: start from the unit matrix at
    ``x``, multiply the link in direction ``dir`` at the current site, and step
    one site forward (``dir >= 0``) or one site backward before multiplying the
    adjoint link (``dir < 0``, direction ``-dir - 1``).
    """
    ret = np.broadcast_to(np.eye(NC, dtype=np.complex128), U.shape[1:]).copy()
    offset = [0, 0, 0, 0]
    for d in path:
        if d >= 0:
            ret = ret @ _get_shifted(U[d], offset)
            offset[d] += 1
        else:
            k = -d - 1
            offset[k] -= 1
            ret = ret @ get_matrix_adjoint(_get_shifted(U[k], offset))
    #
    return ret

def _get_plaq_staple(U, mu):
    """
    Return the plaquette staple of direction ``mu`` (helper).\n
    This is qlat's ``gf_plaq_staple_no_comm``: the sum of the six signed
    directions ``nu`` (``-4 .. 3``) other than ``mu`` and ``-mu - 1`` of the
    Wilson line ``(nu, mu, -nu - 1)``, i.e. the links that close a plaquette
    with ``U_mu(x)`` from both orientations.
    """
    acc = np.zeros_like(U[mu])
    for nu in range(-4, 4):
        if nu == mu or -nu - 1 == mu:
            continue
        #
        acc = acc + _get_wilson_line(U, (nu, mu, -nu - 1))
    #
    return acc

def _get_rect_staple(U, mu):
    """
    Return the ``1 x 2`` rectangle staple of direction ``mu`` (helper).\n
    This is qlat's ``gf_rect_staple_no_comm``: the three rectangle paths
    ``(nu, nu, mu, -nu - 1, -nu - 1)``, ``(nu, mu, mu, -nu - 1, -mu - 1)`` and
    ``(-mu - 1, nu, mu, mu, -nu - 1)`` summed over the six signed ``nu``.
    """
    acc = np.zeros_like(U[mu])
    for nu in range(-4, 4):
        if nu == mu or -nu - 1 == mu:
            continue
        #
        acc = acc + _get_wilson_line(U, (nu, nu, mu, -nu - 1, -nu - 1))
        acc = acc + _get_wilson_line(U, (nu, mu, mu, -nu - 1, -mu - 1))
        acc = acc + _get_wilson_line(U, (-mu - 1, nu, mu, mu, -nu - 1))
    #
    return acc

def _get_all_staple(U, ga, mu):
    """
    Return the ``c1``-weighted staple sum of direction ``mu`` (helper).\n
    This is qlat's ``gf_all_staple_no_comm``,
    ``(1 - 8 c1) * plaquette_staple + c1 * rectangle_staple``, where the
    rectangle term is skipped entirely when ``c1 == 0`` (the Wilson case).
    """
    c1 = ga.c1()
    acc = (1.0 - 8.0 * c1) * _get_plaq_staple(U, mu)
    if c1 != 0.0:
        acc = acc + c1 * _get_rect_staple(U, mu)
    #
    return acc

### -------------------------------------------------------------------
### Momentum initialisation
### -------------------------------------------------------------------

def set_rand_gauge_momentum(gm, sigma, rng):
    """
    Fill ``gm`` with Gaussian anti-Hermitian traceless matrices, in place.\n
    This is qlat's ``set_rand_gauge_momentum(gm, sigma, rng)`` -- note the
    argument order ``(gm, sigma, rng)``, which is qlat's; ``sigma`` defaults to
    ``1.0`` in the method form :meth:`qlat_ref.field.GaugeMomentum.set_rand`.\n
    The field is distributed as ``exp[-Tr(M^2) / (2 sigma^2)]``: for every
    global site (flat index ``gindex``, coordinate 0 fastest) the per-site
    stream is ``rng.split(str(gindex))`` (a split of the ASCII form of the integer,
    *not* an index bump) and every direction ``mu = 0 .. 3`` draws eight
    Gaussian coefficients ``basis[a] ~ exp(-basis[a]^2 / (2 (sigma/sqrt(2))^2))``
    that are assembled into ``M = basis[a] T_a`` with
    ``T_a = i * lambda_a`` (Gell-Mann), ``Tr[T_a T_b] = -2 delta_ab``.\n
    There is exactly one implementation: this function delegates to
    :meth:`qlat_ref.field.GaugeMomentum.set_rand`, verified to reproduce the
    C++ ``GaugeMomentum.set_rand`` values to floating-point precision.
    """
    gm.set_rand(rng, sigma)

### -------------------------------------------------------------------
### Hamiltonian
### -------------------------------------------------------------------

def gm_hamilton_node(gm):
    """
    Return the kinetic term of the Hamiltonian, ``K = sum_{x, mu}
    (-1/2) Tr(P_mu(x)^2)`` (qlat's ``gm_hamilton_node``).\n
    Single process, so this node-local quantity is the global one.  On every
    site the value is the sum of the squares of the eight Lie-algebra
    coefficients of the momentum matrix, which is what
    ``-1/2 Tr(M^2)`` evaluates to for the basis of this module
    (``Tr[T_a T_b] = -2 delta_ab``).
    """
    P = _get_lattice(gm)
    return float(np.sum(-0.5 * np.trace(P @ P, axis1=-2, axis2=-1).real))

def gf_hamilton_node(gf, ga):
    """
    Return the gauge part of the Hamiltonian,
    ``S = (beta/3) [ (1 - 8 c1) sum_P (3 - Re Tr U_P)
    + c1 sum_R (3 - Re Tr U_R) ]`` (qlat's ``gf_hamilton_node``).\n
    The sums run over the ``6`` oriented plaquettes and the ``12`` oriented
    ``1 x 2`` rectangles per site, and the free-field value ``3`` per
    plaquette is subtracted, so the unit gauge field has ``S = 0`` exactly.
    The action itself is :func:`qlat_ref.gauge_action.get_gauge_action`; this
    is the qlat-named entry point.
    """
    return float(get_gauge_action(gf, ga))

### -------------------------------------------------------------------
### Force
### -------------------------------------------------------------------

def set_gm_force(gm_force, gf, ga):
    """
    Set the molecular-dynamics force of the gauge action, in place on
    ``gm_force`` (qlat's ``set_gm_force``).\n
    For every site ``x`` and direction ``mu``, with ``S_mu(x)`` the
    ``c1``-weighted staple sum of the link ``U_mu(x)`` and ``S_gauge`` the
    gauge action of :func:`gf_hamilton_node`,
    :func:`qlat_ref.hmc.set_gm_force` fills
    ``F_mu(x) = trless_antiherm(-beta/3 * U_mu(x) S_mu(x)^dagger)``.
    That is the gradient of the action under left multiplication, in the
    sense that for every generator ``T_a``
    ``d/deps S(e^{eps T_a} U_mu(x))|_0 = -2 basis_a(F_mu(x))``
    (the factor ``-2`` comes from ``Tr[T_a T_b] = -2 delta_ab``).\n
    Like qlat, the output field is initialised by the callee
    (``gm_force.init(geo)``): a momentum created without a geometry, or one
    carrying another geometry, is accepted and re-initialised.\n
    .. note:: The order ``U * S^dagger`` (not ``S^dagger * U``) and the
       ``-beta/3`` normalisation are pinned by a finite-difference test; see
       ``tests/test_hmc_force.py``, which also asserts that the wrong forms do
       not pass.
    """
    U = _get_lattice(gf)
    _check_output_geo(gm_force, gf.geo)
    beta = ga.beta()
    force = np.empty_like(U)
    for mu in range(4):
        ad_staple = get_matrix_adjoint(_get_all_staple(U, ga, mu))
        force[mu] = get_traceless_antihermitian_part(
            (-beta / 3.0) * (U[mu] @ ad_staple)
        )
    #
    _set_lattice(gm_force, force)

def set_gm_force_dual(gm_force_dual, gf, gm_force):
    """
    Set the dual momentum force, in place on ``gm_force_dual`` (qlat's
    ``set_gm_force_dual``).\n
    The dual momentum of :func:`gf_evolve_dual` lives at ``x + mu`` and is
    related to the ordinary force by
    ``F_dual_mu(x) = -U_mu(x)^dagger F_mu(x) U_mu(x)``.  As in
    :func:`set_gm_force`, the output field is initialised by the callee, so it
    may be created without a geometry.
    """
    U = _get_lattice(gf)
    _check_output_geo(gm_force_dual, gf.geo)
    F = _get_lattice(gm_force)
    dual = np.empty_like(U)
    for mu in range(4):
        dual[mu] = -get_matrix_adjoint(U[mu]) @ F[mu] @ U[mu]
    #
    _set_lattice(gm_force_dual, dual)

### -------------------------------------------------------------------
### Evolution
### -------------------------------------------------------------------

def gf_evolve(gf, gm, step_size):
    """
    Evolve the links in place, ``U_mu(x) <- exp(step_size * P_mu(x)) U_mu(x)``
    (qlat's ``gf_evolve``).\n
    The matrix exponential is :func:`qlat_ref.mat.get_su3_exp`, the same
    truncated Taylor (Horner) series of order ``20`` as qlat's
    ``make_matrix_exp``, and the product is re-unitarised with the
    Gram-Schmidt step of :func:`qlat_ref.mat.get_unitarized` -- exactly the
    arithmetic of qlat's ``matrix_evolve``.
    """
    U = _get_lattice(gf)
    P = _get_lattice(gm)
    evolved = np.empty_like(U)
    for mu in range(4):
        evolved[mu] = get_unitarized(get_su3_exp(step_size * P[mu]) @ U[mu])
    #
    _set_lattice(gf, evolved)

def gf_evolve_dual(gf, gm_dual, step_size):
    """
    Evolve the links in place,
    ``U_mu(x) <- U_mu(x) exp(-step_size * P_dual_mu(x))`` (qlat's
    ``gf_evolve_dual``).\n
    The dual momentum is right-multiplied and lives at ``x + mu``; this is
    qlat's ``matrix_evolve_dual``.
    """
    U = _get_lattice(gf)
    P = _get_lattice(gm_dual)
    evolved = np.empty_like(U)
    for mu in range(4):
        evolved[mu] = get_unitarized(U[mu] @ get_su3_exp(-step_size * P[mu]))
    #
    _set_lattice(gf, evolved)

### -------------------------------------------------------------------
### Gauge transformation of a momentum
### -------------------------------------------------------------------

# One implementation, two entries: qlat exposes this through ``qlat.hmc`` and
# the reference implements it in :mod:`qlat_ref.field` (where the ``gt * gm``
# sugar of :class:`qlat_ref.field.GaugeTransform` also needs it), so the name is
# re-exported here as the *same* function object rather than re-wrapped.  That
# keeps ``qlat_ref.hmc.gm_apply_gauge_transformation`` and
# ``qlat_ref.field.gm_apply_gauge_transformation`` identical in signature and
# behaviour, and leaves a single place to change.
gm_apply_gauge_transformation = _gm_apply_gauge_transformation
