"""
Tests of :func:`qlat_ref.gauge.apply_gauge_transform_to_momentum`.\n
A momentum is a Lie-algebra valued site field with the layout of a link field
but attached to a single site, so it is transformed by the adjoint action
``P_mu(x) -> g(x) P_mu(x) g(x)^dagger`` without the link shift.  There is no
HMC integrator in ``qlat_ref``, so the commutation with the gauge fixing is
checked as an algebraic statement about the field, not as a property of an
evolution.
"""

import numpy as np

from qlat_ref import gauge as g

from tests.helpers import GaugeTestCase
from tests.helpers import SEED
from tests.helpers import TOTAL_SITE
from tests.helpers import get_covariant_momentum
from tests.helpers import get_max_difference
from tests.helpers import get_momentum_invariant
from tests.helpers import get_random_field
from tests.helpers import get_random_momentum

class TestMomentumTransform(GaugeTestCase):
    def test_matches_adjoint_action(self):
        # The direct site-by-site expression is the definition.
        P = get_random_momentum()
        g_local = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 6, scale=0.5)
        P_transformed = g.apply_gauge_transform_to_momentum(P, g_local)
        g_dag = g.get_matrix_adjoint(g_local)
        direct = np.stack([g_local @ P[mu] @ g_dag for mu in range(4)])
        self.assertLess(get_max_difference(P_transformed, direct), 1e-14)

    def test_has_the_link_field_layout(self):
        P = get_random_momentum()
        g_local = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 6, scale=0.5)
        P_transformed = g.apply_gauge_transform_to_momentum(P, g_local)
        self.assertEqual(P_transformed.shape, P.shape)
        self.assertEqual(P_transformed.shape[0], 4)
        self.assertEqual(P_transformed.dtype, np.complex128)
        # No shift: the transformation is local in the site index, so a
        # momentum that vanishes everywhere except one site stays supported on
        # that site.  A link transformation would move weight to neighbours.
        P_local = np.zeros_like(P)
        P_local[:, 0, 0, 0, 0] = P[:, 0, 0, 0, 0]
        P_local_transformed = g.apply_gauge_transform_to_momentum(P_local, g_local)
        self.assertLess(
            get_max_difference(
                P_local_transformed[:, 1:], np.zeros_like(P_local[:, 1:])
            ),
            1e-15,
        )

    def test_does_not_modify_the_inputs(self):
        P = get_random_momentum()
        P_copy = P.copy()
        g_local = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 6, scale=0.5)
        g_copy = g_local.copy()
        g.apply_gauge_transform_to_momentum(P, g_local)
        self.assertTrue(np.array_equal(P, P_copy))
        self.assertTrue(np.array_equal(g_local, g_copy))

class TestMomentumInvariants(GaugeTestCase):
    def test_preserves_the_lie_algebra(self):
        # Anti-hermitian and traceless are preserved: the adjoint action is a
        # Lie algebra automorphism.
        P = get_random_momentum()
        g_local = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 6, scale=0.5)
        P_transformed = g.apply_gauge_transform_to_momentum(P, g_local)
        P_adjoint = g.get_matrix_adjoint(P_transformed)
        self.assertLess(float(np.max(np.abs(P_transformed + P_adjoint))), 1e-14)
        self.assertLess(
            float(np.max(np.abs(np.trace(P_transformed, axis1=-2, axis2=-1)))),
            1e-14,
        )

    def test_preserves_the_quadratic_invariant(self):
        P = get_random_momentum(seed=SEED + 9)
        g_local = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 10, scale=0.4)
        P_transformed = g.apply_gauge_transform_to_momentum(P, g_local)
        self.assertLess(
            abs(get_momentum_invariant(P_transformed) - get_momentum_invariant(P)),
            1e-12,
        )

class TestMomentumTransformInverse(GaugeTestCase):
    def test_is_dagger_is_the_inverse(self):
        # is_dagger is the inverse transformation: applying g and then g^dagger
        # restores the momentum.
        P = get_random_momentum(seed=SEED + 7)
        g_local = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 8, scale=0.5)
        P_round_trip = g.apply_gauge_transform_to_momentum(
            g.apply_gauge_transform_to_momentum(P, g_local),
            g_local,
            is_dagger=True,
        )
        self.assertLess(get_max_difference(P_round_trip, P), 1e-14)

    def test_is_dagger_uses_the_adjoint_transform(self):
        # is_dagger also equals the explicit expression with the adjoint
        # transform.
        P = get_random_momentum(seed=SEED + 7)
        g_local = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 8, scale=0.5)
        P_explicit = g.apply_gauge_transform_to_momentum(
            P, g.get_matrix_adjoint(g_local)
        )
        P_dagger = g.apply_gauge_transform_to_momentum(P, g_local, is_dagger=True)
        self.assertLess(get_max_difference(P_explicit, P_dagger), 1e-14)

class TestMomentumTransformComposition(GaugeTestCase):
    def test_composition_order(self):
        # The adjoint action is a representation: transforming by g2 g1 is the
        # same as transforming by g1 and then by g2.
        P = get_random_momentum(seed=SEED + 9)
        g1 = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 10, scale=0.4)
        g2 = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 11, scale=0.4)
        composed = g.apply_gauge_transform_to_momentum(P, g2 @ g1)
        sequential = g.apply_gauge_transform_to_momentum(
            g.apply_gauge_transform_to_momentum(P, g1), g2
        )
        self.assertLess(get_max_difference(composed, sequential), 1e-14)

    def test_link_field_has_the_same_order(self):
        # The same order holds for the link field (qlat convention), which is
        # what makes the accumulated transform of the fixing the left product
        # of the sweep transforms.
        U = get_random_field(seed=SEED + 14)
        g1 = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 10, scale=0.4)
        g2 = g.get_random_gauge_transform(TOTAL_SITE, seed=SEED + 11, scale=0.4)
        composed = g.apply_gauge_transform(U, g2 @ g1)
        sequential = g.apply_gauge_transform(g.apply_gauge_transform(U, g1), g2)
        self.assertLess(get_max_difference(composed, sequential), 1e-14)

class TestMomentumTransformCommutesWithFixing(GaugeTestCase):
    def test_transforming_the_field_transforms_the_momentum(self):
        # A field built from the links in the adjoint representation (a
        # momentum is such a site field) must transform the same way whether
        # the gauge transformation is applied to the momentum directly or
        # comes from fixing the field.  That is the commutation an HMC force
        # term relies on; the integrator itself does not exist in qlat_ref, so
        # only this algebraic statement is checked.
        U = get_random_field(seed=SEED + 12)
        P = get_covariant_momentum(U)
        for fix in (g.fix_landau_gauge, g.fix_coulomb_gauge):
            gt, U_fixed, info = self.check_fix(fix, U)
            self.assertTrue(info["converged"])
            P_transformed = g.apply_gauge_transform_to_momentum(P, gt)
            P_fixed = get_covariant_momentum(U_fixed)
            self.assertLess(get_max_difference(P_transformed, P_fixed), 1e-13)
            # gt is unitary, so the invariant content of P is untouched.
            self.assertLess(
                abs(get_momentum_invariant(P_transformed) - get_momentum_invariant(P)),
                1e-12,
            )
            # Undoing the fixing restores the original momentum.
            P_restored = g.apply_gauge_transform_to_momentum(
                P_transformed, gt, is_dagger=True
            )
            self.assertLess(get_max_difference(P_restored, P), 1e-13)
