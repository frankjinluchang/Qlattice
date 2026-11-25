# cython: binding=True, embedsignature=True

"""
Module ``qlat.hmc``
=====================\n
Hybrid Monte Carlo (HMC) machinery for pure-gauge simulations.\n
Provides:\n
- ``GaugeMomentum`` — conjugate momentum field (``FieldColorMatrix`` with
  multiplicity 4, one anti-Hermitian matrix per lattice direction).
- Force computation (``set_gm_force``, ``set_gm_force_dual``).
- Molecular-dynamics evolution (``gf_evolve``, ``gf_evolve_dual``, and
  ``FieldRealD``-weighted variants).
- Hamiltonian helpers (``gm_hamilton_node``, ``gf_hamilton_node``).
- Metropolis accept/reject (``metropolis_accept``).
- A complete pure-gauge HMC driver (``run_hmc_pure_gauge``).
- Utility functions for anti-Hermitian matrix basis conversions and
  gauge-transformation momentum projection.\n
Documentation: ``docs/qlat/qlat_hmc.md``\n
.. note:: Update the documentation when updating this source file.
"""

from qlat_utils.all cimport *
from . cimport everything as cc
from .geometry cimport Geometry
from .gauge_action cimport GaugeAction
from .field_types cimport (
        FieldColorMatrix,
        FieldRealD,
        )
from .qcd cimport GaugeField, GaugeTransform

import numpy as np

cdef class GaugeMomentum(FieldColorMatrix):

    def __init__(self, Geometry geo=None):
        super().__init__(geo, 4)

    cdef cc.Handle[cc.GaugeMomentum] xxx(self):
        assert self.xx.multiplicity == 4 or self.xx.multiplicity == 0
        return cc.Handle[cc.GaugeMomentum](<cc.GaugeMomentum&>self.xx)

    def set_rand(self, RngState rng, cc.RealD sigma=1.0):
        set_rand_gauge_momentum(self, sigma, rng)

    def set_rand_fa(self, FieldRealD mf, RngState rng):
        set_rand_gauge_momentum_fa(self, mf, rng)

    def __rmul__(self, other):
        """
        ``gt * gm`` sugar for ``gm_apply_gauge_transformation(gm, gt)``.
        Only a ``GaugeTransform`` is handled here; the inverse direction needs
        the explicit ``q.gm_apply_gauge_transformation(gm, gt, is_dagger=True)``.
        """
        if isinstance(other, GaugeTransform):
            return gm_apply_gauge_transformation(self, other)
        return NotImplemented

###

def set_rand_gauge_momentum(GaugeMomentum gm, cc.RealD sigma, RngState rng):
    return cc.set_rand_gauge_momentum(gm.xxx().val(), sigma, rng.xx)

def set_rand_gauge_momentum_fa(GaugeMomentum gm, FieldRealD mf, RngState rng):
    return cc.set_rand_gauge_momentum(gm.xxx().val(), mf.xx, rng.xx)

def gm_hamilton_node(GaugeMomentum gm):
    return cc.gm_hamilton_node(gm.xxx().val())

def gm_hamilton_node_fa(GaugeMomentum gm, FieldRealD mf):
    return cc.gm_hamilton_node(gm.xxx().val(), mf.xx)

def gf_hamilton_node(GaugeField gf, GaugeAction ga):
    return cc.gf_hamilton_node(gf.xxx().val(), ga.xx)

def gf_evolve(GaugeField gf, GaugeMomentum gm, cc.RealD step_size):
    return cc.gf_evolve(gf.xxx().val(), gm.xxx().val(), step_size)

def gf_evolve_dual(GaugeField gf, GaugeMomentum gm_dual, cc.RealD step_size):
    return cc.gf_evolve_dual(gf.xxx().val(), gm_dual.xxx().val(), step_size)

def gf_evolve_fa(GaugeField gf, GaugeMomentum gm, FieldRealD mf, cc.RealD step_size):
    return cc.gf_evolve(gf.xxx().val(), gm.xxx().val(), mf.xx, step_size)

def gf_evolve_fa_dual(GaugeField gf, GaugeMomentum gm_dual, FieldRealD mf_dual, cc.RealD step_size):
    return cc.gf_evolve_dual(gf.xxx().val(), gm_dual.xxx().val(), mf_dual.xx, step_size)

def field_color_matrix_exp(FieldColorMatrix fc1, cc.PyComplexD coef):
    cdef FieldColorMatrix fc = FieldColorMatrix()
    cdef cc.ComplexD cc_coef = cc.ccpy_d(coef)
    cc.field_color_matrix_exp(fc.xx, fc1.xx, cc_coef)
    return fc

def field_color_matrix_mul(FieldColorMatrix fc1, FieldColorMatrix fc2):
    cdef FieldColorMatrix fc = FieldColorMatrix()
    cc.field_color_matrix_mul(fc.xx, fc1.xx, fc2.xx)
    return fc

def set_gm_force(GaugeMomentum gm_force, GaugeField gf, GaugeAction ga):
    return cc.set_gm_force(gm_force.xxx().val(), gf.xxx().val(), ga.xx)

def gm_apply_gauge_transformation(GaugeMomentum gm, GaugeTransform gt, cc.Bool is_dagger=False):
    """
    Return a new ``GaugeMomentum`` equal to ``gm`` transformed by ``gt``; the
    input ``gm`` is not modified.\n
    The adjoint action is at the same site for all four directions::\n
        gm'_mu(x) = gt(x) * gm_mu(x) * gt(x)^dagger      (is_dagger=False)
        gm'_mu(x) = gt(x)^dagger * gm_mu(x) * gt(x)      (is_dagger=True)\n
    ``is_dagger=True`` is the inverse of the default action (``gt`` is
    unitary), not a second independent transform; it conjugates the left
    factor and leaves the right one alone.\n
    There is no ``x + mu`` shift and no communication: ``gf_evolve``
    left-multiplies ``exp(step_size * gm)`` at the same site, so gauge
    covariance requires the momentum to conjugate at ``x``.  The result keeps
    the anti-Hermitian and traceless properties of ``gm`` for unitary ``gt``.\n
    Requires an initialised ``gm``.  Do NOT use this on a ``_dual`` momentum
    (``gf_evolve_dual`` right-multiplies and the dual momentum lives at
    ``x + mu``, so its covariant law is different).\n
    Sugar: ``gt * gm`` calls this with ``is_dagger=False``.
    """
    cdef GaugeMomentum gm1 = GaugeMomentum(gm.geo)
    cc.gm_apply_gauge_transformation(gm1.xxx().val(), gm.xxx().val(), gt.xxx().val(), is_dagger)
    return gm1

def set_gm_force_dual(GaugeMomentum gm_force_dual, GaugeField gf, GaugeMomentum gm_force):
    return cc.set_gm_force_dual(gm_force_dual.xxx().val(), gf.xxx().val(), gm_force.xxx().val())

def project_gauge_transform(GaugeMomentum gm, GaugeMomentum gm_dual, FieldRealD mf, FieldRealD mf_dual):
    """
    Project out the pure gauge transformation movement.
    """
    return cc.project_gauge_transform(gm.xxx().val(), gm_dual.xxx().val(), mf.xx, mf_dual.xx)

def set_gauge_transform_momentum(GaugeMomentum gm, GaugeMomentum gm_dual, FieldColorMatrix gtm):
    """
    Set `gm` and `gm_dual` with `gtm`.
    The overall effects of `gm` and `gm_dual` with `gf_evolve` and `gf_evolve_dual` is a pure gauge transformation from `gtm`.
    The name `gtm` stands for gauge tranformation momentum.
    """
    return cc.set_gauge_transform_momentum(gm.xxx().val(), gm_dual.xxx().val(), gtm.xx)

def dot_gauge_momentum(GaugeMomentum gm1, GaugeMomentum gm2):
    """
    return RealD dot field with multiplicity = 4
    """
    cdef FieldRealD f = FieldRealD()
    cc.dot_gauge_momentum(f.xx, gm1.xxx().val(), gm2.xxx().val())
    return f

def set_anti_hermitian_matrix_from_basis(FieldColorMatrix fc, FieldRealD basis):
    r"""
    fc[x, m] = \sum_a T_a * basis[x, m * 8 + a]
    T_a^dagger = -T_a
    Tr[T_a T_b] = -2 \delta_{a,b}
    """
    cc.set_anti_hermitian_matrix_from_basis(fc.xx, basis.xx)

def set_basis_from_anti_hermitian_matrix(FieldRealD basis, FieldColorMatrix fc):
    r"""
    fc[x, m] = \sum_a T_a * basis[x, m * 8 + a]
    T_a^dagger = -T_a
    Tr[T_a T_b] = -2 \delta_{a,b}
    """
    cc.set_basis_from_anti_hermitian_matrix(basis.xx, fc.xx)
