"""
Tests of the gauge fixing itself: both gauges, their convergence measures, the
gauge covariance of the iteration, and the properties of the returned
accumulated transform.\n
The momentum transform has its own module, :mod:`tests.test_momentum`.
"""

import numpy as np

from qlat_ref import gauge as g

from tests.helpers import ALPHA
from tests.helpers import EPS
from tests.helpers import GaugeTestCase
from tests.helpers import SEED
from tests.helpers import TOTAL_SITE
from tests.helpers import get_max_difference
from tests.helpers import get_random_field

class TestLandauGaugeFixing(GaugeTestCase):
    def test_functional_increases(self):
        U = get_random_field()
        functional_before = g.get_gauge_functional(U)
        _, U_fixed, info = self.check_fix(g.fix_landau_gauge, U, alpha=ALPHA)
        self.assertGreater(g.get_gauge_functional(U_fixed), functional_before)
        self.assertGreater(info["functional"], functional_before)

    def test_divergence_below_eps(self):
        U = get_random_field()
        divergence_before = g.get_gauge_divergence(U)
        self.assertGreater(divergence_before, 1e-2)
        _, U_fixed, _ = self.check_fix(g.fix_landau_gauge, U, alpha=ALPHA)
        divergence_after = g.get_gauge_divergence(U_fixed)
        self.assertLessEqual(divergence_after, EPS)

    def test_functional_monotone(self):
        # The functional must not decrease during the plain steepest-descent
        # iteration (up to the rounding noise of the Jacobi sweep).
        U = get_random_field()
        _, _, info = self.check_fix(g.fix_landau_gauge, U, alpha=ALPHA)
        increments = np.diff(info["functional_list"])
        self.assertGreaterEqual(np.min(increments), -1e-12)

    def test_info_history_is_consistent(self):
        U = get_random_field()
        _, _, info = self.check_fix(g.fix_landau_gauge, U, alpha=ALPHA)
        self.assertEqual(len(info["functional_list"]), info["n_iter"])
        self.assertEqual(len(info["divergence_list"]), info["n_iter"])
        self.assertEqual(info["functional"], info["functional_list"][-1])
        self.assertLessEqual(info["divergence_list"][-1], EPS)
        self.assertEqual(info["dirs"], (0, 1, 2, 3))
        self.assertEqual(info["max_iter"], 10000)
        self.assertEqual(info["eps"], EPS)
        self.assertEqual(info["alpha"], ALPHA)

class TestCoulombGaugeFixing(GaugeTestCase):
    def test_functional_increases_per_time_slice(self):
        U = get_random_field(seed=SEED + 1)
        functional_before = g.get_coulomb_functional_list(U)
        _, U_fixed, _ = self.check_fix(g.fix_coulomb_gauge, U, alpha=ALPHA)
        functional_after = g.get_coulomb_functional_list(U_fixed)
        self.assertTrue(
            all(
                after > before
                for before, after in zip(functional_before, functional_after)
            )
        )

    def test_gauge_condition_per_time_slice(self):
        # Every time slice is an independent 3D Landau fixing and must satisfy
        # the Coulomb gauge condition on its own.  The convergence criterion
        # uses the RMS divergence over the whole lattice, so a single slice is
        # allowed a factor sqrt(Nt) more.
        U = get_random_field(seed=SEED + 1)
        _, U_fixed, _ = self.check_fix(g.fix_coulomb_gauge, U, alpha=ALPHA)
        divergence_list = g.get_coulomb_divergence_list(U_fixed)
        self.assertLessEqual(max(divergence_list), EPS * np.sqrt(len(divergence_list)))
        self.assertEqual(len(divergence_list), U.shape[1])

    def test_uses_the_correct_spatial_directions(self):
        # Component 3 is time (qlat's `total_site` is (Lx, Ly, Lz, Lt),
        # `Geometry::spatial_volume` divides out component 3 and
        # `get_psel_tslice` slices `t_dir=3`), so the spatial directions are
        # 0, 1, 2.  qlat's `gf_fix_coulomb_gauge` uses {1, 2, 3} instead, which
        # is qlat-issue-8; qlat_ref deliberately uses the correct set.
        U = get_random_field(seed=SEED + 1)
        _, _, info = self.check_fix(g.fix_coulomb_gauge, U, alpha=ALPHA)
        self.assertEqual(info["dirs"], (0, 1, 2))

    def test_correct_dirs_converge_and_qlat_dirs_do_not(self):
        # The distinguishing signature of the convention: the Coulomb gauge
        # condition must hold for {0, 1, 2} and must NOT hold for qlat's
        # {1, 2, 3}.  This is what makes the deviation observable rather than
        # cosmetic.
        U = get_random_field(seed=SEED + 1)
        _, U_fixed, _ = self.check_fix(g.fix_coulomb_gauge, U, alpha=ALPHA)
        self.assertLessEqual(g.get_gauge_divergence(U_fixed, (0, 1, 2)), EPS)
        self.assertGreater(g.get_gauge_divergence(U_fixed, (1, 2, 3)), 1e-2)

class TestGaugeCovariance(GaugeTestCase):
    def test_fixed_field_is_covariant(self):
        # A global gauge transformation g0 (the same SU(3) matrix on every
        # site) leaves the functional Re Tr U_mu(x) invariant, and the plain
        # local iteration is exactly covariant under it: fixing
        # g0 U g0^dagger must give g0 U_fixed g0^dagger and the same
        # functional value.
        U = get_random_field(seed=SEED + 2)
        g_transform = g.get_random_gauge_transform(
            (1, 1, 1, 1), seed=SEED + 3, scale=0.5
        )
        g_global = np.broadcast_to(g_transform, TOTAL_SITE + (g.NC, g.NC)).copy()
        U_transformed = g.apply_gauge_transform(U, g_global)
        _, U_fixed, info = self.check_fix(g.fix_landau_gauge, U)
        _, U_fixed_transformed, info_transformed = self.check_fix(
            g.fix_landau_gauge, U_transformed
        )
        expected = g.apply_gauge_transform(U_fixed, g_global)
        self.assertLess(get_max_difference(U_fixed_transformed, expected), 1e-10)
        self.assertLess(abs(info["functional"] - info_transformed["functional"]), 1e-12)

    def test_transform_is_covariant(self):
        # At the level of the transformations the same covariance says that
        # the accumulated transform of the transformed field is
        # g0 gt g0^dagger.
        U = get_random_field(seed=SEED + 2)
        g_transform = g.get_random_gauge_transform(
            (1, 1, 1, 1), seed=SEED + 3, scale=0.5
        )
        g_global = np.broadcast_to(g_transform, TOTAL_SITE + (g.NC, g.NC)).copy()
        U_transformed = g.apply_gauge_transform(U, g_global)
        gt, _, _ = self.check_fix(g.fix_landau_gauge, U)
        gt_transformed, _, _ = self.check_fix(g.fix_landau_gauge, U_transformed)
        expected_gt = g_global @ gt @ g.get_matrix_adjoint(g_global)
        self.assertLess(get_max_difference(gt_transformed, expected_gt), 1e-10)

class TestOverRelaxation(GaugeTestCase):
    def test_step_size_does_not_change_functional(self):
        # The converged functional must not depend on the step size (only the
        # number of sweeps does).
        U = get_random_field(seed=SEED + 4)
        _, _, info_a = self.check_fix(g.fix_landau_gauge, U, alpha=0.05)
        _, _, info_b = self.check_fix(g.fix_landau_gauge, U, alpha=0.1)
        self.assertLess(abs(info_a["functional"] - info_b["functional"]), 1e-10)

class TestTransformNotUnique(GaugeTestCase):
    def test_landau_transform_is_not_unique(self):
        # A global center element multiplies the accumulated transform without
        # changing the Landau-gauge fixed field.
        h = np.exp(2j * np.pi / 3) * np.eye(g.NC, dtype=np.complex128)
        U = get_random_field(seed=SEED + 16)
        gt, U_fixed, _ = self.check_fix(g.fix_landau_gauge, U)
        U_fixed_center = g.apply_gauge_transform(U, gt @ h)
        self.assertLess(get_max_difference(U_fixed, U_fixed_center), 1e-12)
        self.assertGreater(get_max_difference(gt, gt @ h), 1.0)

    def test_coulomb_transform_is_not_unique_per_slice(self):
        # For Coulomb gauge the same holds per time slice: the Coulomb
        # functional only involves the spatial links (0, 1, 2), which stay
        # inside one time slice, so an independent center element per slice
        # leaves the fixed field invariant.
        #
        # Component 3 is time (qlat's `total_site` is (Lx, Ly, Lz, Lt)), so in
        # the (4, L0, L1, L2, L3, NC, NC) link layout the time axis is axis 4
        # and the site-field layout is (L0, L1, L2, L3, NC, NC) with time on
        # axis 3.
        U = get_random_field(seed=SEED + 16)
        n_t = U.shape[4]
        h_slice = np.zeros(U.shape[1:], dtype=np.complex128)
        for t in range(n_t):
            z = np.exp(2j * np.pi / 3 * (t % 3))
            h_slice[..., t, :, :] = z * np.eye(g.NC)
        gt, U_fixed, _ = self.check_fix(g.fix_coulomb_gauge, U)
        U_fixed_center = g.apply_gauge_transform(U, gt @ h_slice)
        # Only the spatial links (directions 0, 1, 2) enter the Coulomb
        # functional, and a center element cancels in g U g^dagger.
        self.assertLess(get_max_difference(U_fixed[:3], U_fixed_center[:3]), 1e-12)
        # The transform really differs, so this is not a vacuous test.
        self.assertGreater(get_max_difference(gt, gt @ h_slice), 1.0)
        self.assertLessEqual(
            max(g.get_coulomb_divergence_list(U_fixed_center)),
            EPS * np.sqrt(n_t),
        )

class TestAccumulatedTransform(GaugeTestCase):
    def test_transform_reproduces_iterated_field(self):
        # The internal core also returns the field it iterated on; applying
        # the accumulated transform to the input must reproduce that field.
        # This is what pins down the accumulation order gt <- g_step gt.
        U = get_random_field(seed=SEED + 15)
        for dirs in ((0, 1, 2, 3), (1, 2, 3)):
            gt, U_fixed, info = g._fix_gauge(
                U,
                dirs=dirs,
                gauge_name="test",
                max_iter=10000,
                eps=EPS,
                alpha=ALPHA,
                verbose=False,
            )
            applied = g.apply_gauge_transform(U, gt)
            self.assertLess(get_max_difference(applied, U_fixed), 1e-13)
            self.assertTrue(info["converged"])

    def test_public_wrapper_returns_the_same_transform(self):
        U = get_random_field(seed=SEED + 15)
        for fix, dirs in (
            (g.fix_landau_gauge, (0, 1, 2, 3)),
            (g.fix_coulomb_gauge, (0, 1, 2)),
        ):
            gt_internal, _, info_internal = g._fix_gauge(
                U,
                dirs=dirs,
                gauge_name="test",
                max_iter=10000,
                eps=EPS,
                alpha=ALPHA,
                verbose=False,
            )
            gt_public, info_public = fix(U, max_iter=10000, eps=EPS, alpha=ALPHA)
            self.assertTrue(np.array_equal(gt_public, gt_internal))
            self.assertEqual(info_public["functional"], info_internal["functional"])

    def test_public_return_is_a_two_tuple(self):
        # The public API is ``(gt, info)``; the fixed field is not returned.
        U = get_random_field(seed=SEED + 15)
        result = g.fix_landau_gauge(U, max_iter=10000, eps=EPS, alpha=ALPHA)
        self.assertEqual(len(result), 2)
        gt, info = result
        self.assertEqual(gt.shape, U.shape[1:])
        self.assertIsInstance(info, dict)

class TestInputNotMutated(GaugeTestCase):
    def test_fixing_does_not_modify_the_input(self):
        U = get_random_field(seed=SEED + 17)
        U_copy = U.copy()
        for fix in (g.fix_landau_gauge, g.fix_coulomb_gauge):
            self.check_fix(fix, U)
            self.assertTrue(np.array_equal(U, U_copy))
