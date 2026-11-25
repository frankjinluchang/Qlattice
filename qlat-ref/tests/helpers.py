"""
Helpers shared by the :mod:`qlat_ref` ``unittest`` test modules.\n
The lattice is deliberately tiny so that the pure-NumPy reference
implementation runs in a few seconds, and every random field is derived
deterministically from a single seed constant.  The module is not a test
module (its name does not start with ``test_``), so discovery imports it for
the tests only.
"""

import unittest

import numpy as np

from qlat_ref import gauge as g

TOTAL_SITE = (2, 2, 2, 2)
SEED = 20240517
EPS = 1e-10
ALPHA = 0.1

### -------------------------------------------------------------------

def get_random_field(seed=SEED, total_site=TOTAL_SITE):
    """Return the deterministic random unitary gauge field used by the tests."""
    return g.get_random_gauge_field(total_site, seed=seed, scale=1.0)

def get_random_momentum(seed=SEED + 5, total_site=TOTAL_SITE):
    """Return a deterministic random anti-hermitian traceless momentum field."""
    rng = np.random.default_rng(seed)
    shape = (4,) + tuple(total_site) + (g.NC, g.NC)
    h = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return g.get_traceless_antihermitian_part(h)

def get_momentum_invariant(P):
    """Return the global translation and gauge invariant number ``sum_x Tr P^2``."""
    return float(np.sum(np.trace(P @ P, axis1=-2, axis2=-1).real))

def get_covariant_momentum(U):
    """
    Return a momentum built from the links of ``U`` that is adjoint covariant.\n
    Each entry is the anti-hermitian traceless part of the sum of the
    plaquettes in the three planes that contain ``mu`` and that are attached at
    the site, which is exactly a site-local object transforming as
    ``g(x) P_mu(x) g(x)^dagger``.
    """
    U = np.asarray(U, dtype=np.complex128)
    P = np.zeros_like(U)
    for mu in range(4):
        K = np.zeros_like(U[mu])
        for nu in range(4):
            if nu == mu:
                continue
            plaq = (
                U[mu]
                @ np.roll(U[nu], -1, axis=mu)
                @ g.get_matrix_adjoint(np.roll(U[mu], -1, axis=nu))
                @ g.get_matrix_adjoint(U[nu])
            )
            K = K + plaq
        P[mu] = g.get_traceless_antihermitian_part(K)
    return P

def get_max_difference(a, b):
    """Return ``max |a - b|`` over all entries of two equally shaped arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

### -------------------------------------------------------------------

class GaugeTestCase(unittest.TestCase):
    """Base class with the post-conditions shared by every successful fixing."""

    def check_fix(self, fix, U, **kwargs):
        """
        Run the fixing ``fix`` on ``U`` and check its post-conditions.\n
        Return ``(gt, U_fixed, info)``, with ``U_fixed`` rebuilt from the
        returned transform.  ``kwargs`` are forwarded to ``fix`` with the
        convergence tolerances of the test suite as defaults.
        """
        kwargs.setdefault("max_iter", 10000)
        kwargs.setdefault("eps", EPS)
        gt, info = fix(U, **kwargs)
        self.assertTrue(info["converged"], "the fixing did not converge")
        self.assertLessEqual(info["divergence"], EPS)
        self.assertGreaterEqual(info["n_iter"], 1)
        self.assertEqual(len(info["functional_list"]), info["n_iter"])
        U_fixed = g.apply_gauge_transform(U, gt)
        self.assertLessEqual(g.get_unitarity_error(U_fixed), 1e-10)
        self.assertTrue(g.check_unitarity(gt, eps=1e-10))
        return gt, U_fixed, info
