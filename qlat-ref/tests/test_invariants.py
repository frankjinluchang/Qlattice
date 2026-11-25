"""
Tests of the basic invariants of the gauge field: unitarity, the trivial
configurations where the gauge fixing has nothing to do, and the array
conventions of the module.\n
These are the cheapest tests in the suite and they pin the conventions the
other modules rely on.
"""

import numpy as np

from qlat_ref import gauge as g

from tests.helpers import EPS
from tests.helpers import SEED
from tests.helpers import TOTAL_SITE
from tests.helpers import GaugeTestCase
from tests.helpers import get_max_difference
from tests.helpers import get_random_field

class TestUnitarity(GaugeTestCase):
    def test_random_field_is_unitary(self):
        U = get_random_field()
        self.assertTrue(g.check_unitarity(U, eps=1e-12))
        self.assertLessEqual(g.get_unitarity_error(U), 1e-12)

    def test_random_transform_is_unitary(self):
        gt = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED, scale=0.5)
        self.assertTrue(g.check_unitarity(gt, eps=1e-12))
        self.assertEqual(gt.shape, TOTAL_SITE + (g.NC, g.NC))

    def test_deterministic_given_the_seed(self):
        self.assertTrue(np.array_equal(get_random_field(), get_random_field()))
        self.assertFalse(
            np.array_equal(get_random_field(), get_random_field(seed=SEED + 1))
        )

class TestTrivialConfigurations(GaugeTestCase):
    def test_unit_field_is_already_fixed(self):
        # The unit field is gauge fixed by construction.
        U = g.get_unit_gauge_field(TOTAL_SITE)
        self.assertEqual(g.get_gauge_functional(U), 1.0)
        self.assertLess(g.get_gauge_divergence(U), 1e-14)
        _, U_fixed, _ = self.check_fix(g.fix_landau_gauge, U, max_iter=10)
        self.assertEqual(get_max_difference(U_fixed, U), 0.0)

    def test_single_site_lattice_is_trivially_fixed(self):
        # On a 1^4 lattice a gauge transformation acts at a single site only
        # and cancels in Re Tr[g U g^dagger], so every field is trivially
        # "fixed" and the local gradient must vanish.
        U = g.get_random_gauge_field((1, 1, 1, 1), seed=SEED, scale=1.0)
        functional_before = g.get_gauge_functional(U)
        divergence_before = g.get_gauge_divergence(U)
        self.assertLess(divergence_before, 1e-14)
        _, U_fixed, info = self.check_fix(g.fix_landau_gauge, U, max_iter=10)
        self.assertEqual(info["n_iter"], 1)
        self.assertLess(abs(g.get_gauge_functional(U_fixed) - functional_before), 1e-14)

class TestArrayConventions(GaugeTestCase):
    def test_shapes_and_dtypes(self):
        U = get_random_field(total_site=(1, 2, 3, 4))
        self.assertEqual(U.shape, (4, 1, 2, 3, 4, g.NC, g.NC))
        self.assertEqual(U.dtype, np.complex128)
        gt = g.get_random_gauge_transform((1, 2, 3, 4), seed=SEED, scale=0.3)
        self.assertEqual(gt.shape, (1, 2, 3, 4, g.NC, g.NC))

    def test_link_transform_shifts_the_transformation(self):
        # The link convention is g(x) U_mu(x) g(x + mu)^dagger: a field that is
        # constant in one direction is unchanged by a transformation that is
        # also constant in that direction, while a rapidly varying one is not.
        U = g.get_unit_gauge_field(TOTAL_SITE)
        gt = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED, scale=0.5)
        U_transformed = g.apply_gauge_transform(U, gt)
        expected = np.stack(
            [gt @ g.get_matrix_adjoint(np.roll(gt, -1, axis=mu)) for mu in range(4)]
        )
        self.assertLess(get_max_difference(U_transformed, expected), 1e-14)
        # The link transformation is not a site-local operation: transforming
        # the unit field gives a field with weight on every site.
        self.assertGreater(np.max(np.abs(U_transformed - U[0])), 1e-2)

    def test_divergence_matches_the_gauge_condition(self):
        # The divergence vanishes exactly when the local gradient vanishes.
        U = get_random_field(seed=SEED + 18)
        D = g.get_local_gauge_gradient(U)
        self.assertLess(
            abs(g.get_divergence_from_gradient(D) - g.get_gauge_divergence(U)),
            1e-15,
        )
        self.assertGreater(g.get_gauge_divergence(U), 1e-2)
        _, U_fixed, _ = self.check_fix(g.fix_landau_gauge, U, alpha=0.1)
        self.assertLessEqual(g.get_gauge_divergence(U_fixed), EPS)
