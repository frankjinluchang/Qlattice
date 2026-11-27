"""
Tests of the molecular-dynamics force, :func:`qlat_ref.hmc.set_gm_force`.\n
The force is the gradient of the gauge action under left multiplication of a
link, and the only honest way to test it is a finite difference of the action
itself.  For every generator ``T_a`` of the qlat basis
(``T_a = i lambda_a``, ``Tr[T_a T_b] = -2 delta_ab``) the reference must
satisfy
    d/deps S(exp(eps T_a) U_mu(x)) |_{eps=0} = -2 basis_a(F_mu(x))
and the two plausible-looking wrong forms -- ``-beta/3 S^dagger U`` and
``-beta/3 U S`` in place of ``-beta/3 U S^dagger`` -- must *fail* the same
check, so a later "simplification" is caught.\n
Two more properties pin the staple structure: the force must be exactly gauge
covariant (``F(gU) = g F(U) g^dagger``), and it must be anti-Hermitian
traceless.  A wrong staple orientation leaves a valid-looking anti-Hermitian
matrix that is not the gradient, which the finite difference catches and
covariance does not.
"""

import unittest

import numpy as np

import qlat_ref as qr
from qlat_ref import hmc
from qlat_ref.gauge_action import _get_plaq
from qlat_ref.gauge_action import _get_rect
from qlat_ref.gauge_action import get_gauge_action
from qlat_ref.mat import get_matrix_adjoint
from qlat_ref.mat import get_su3_exp
from qlat_ref.mat import get_traceless_antihermitian_part
from qlat_ref.field import get_array_from_field
from qlat_ref.observables import get_link_array

TOTAL_SITE = (2, 2, 2, 2)

# The generators of the qlat basis, T_a = i * lambda_a (Gell-Mann), normalized
# by Tr[T_a T_b] = -2 delta_ab.  Built here explicitly so the test does not
# depend on the basis assembly of the implementation it is checking.
_LAMBDA = np.asarray(
    [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3.0),
    ],
    dtype=np.complex128,
)
T_GENERATORS = (1j * _LAMBDA).astype(np.complex128)

# Sites and directions sampled by the finite-difference tests.  Two sites in
# different parity classes and all four directions is enough to catch a staple
# that mixes up the forward and backward paths.
SITES = ((0, 0, 0, 0), (1, 0, 1, 0))

# The negative assertions ("this wrong formula must fail") only need to show an
# O(1) disagreement, so one site keeps them cheap.
ONE_SITE = ((0, 0, 0, 0),)

# Central-difference step and its truncation allowance.  With eps = 1e-6 the
# O(eps^2) truncation of the cubic-plus action is well below 1e-6; the measured
# residual is 3e-8.
FD_EPS = 1e-6
FD_TOL = 1e-6

GAUGE_ACTIONS = (
    (5.5, 0.0),
    (2.13, -0.331),
)

### -------------------------------------------------------------------

def get_geometry(total_site=TOTAL_SITE):
    """Return the test geometry (helper)."""
    return qr.Geometry(qr.Coordinate(list(total_site)))

def get_random_field(seed="hmc-force", sigma=0.4, total_site=TOTAL_SITE):
    """Return a deterministic random gauge field (helper)."""
    gf = qr.GaugeField(get_geometry(total_site))
    gf.set_rand(qr.RngState(seed), sigma=sigma, n_step=1)
    return gf

def get_random_momentum(seed="hmc-force-gm", sigma=1.0, total_site=TOTAL_SITE):
    """Return a deterministic Gaussian momentum (helper)."""
    gm = qr.GaugeMomentum(get_geometry(total_site))
    gm.set_rand(qr.RngState(seed), sigma)
    return gm

def get_max_difference(a, b):
    """Return ``max |a - b|`` over two equally shaped arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

def get_basis_coefficient(F, a):
    """Return ``basis_a(F) = -Tr[T_a F] / 2`` for an anti-Hermitian ``F``."""
    return -0.5 * float(np.trace(T_GENERATORS[a] @ F).real)

def get_force_array(gf, ga):
    """Return the plain-array force of ``gf`` under ``ga`` (helper)."""
    gm_force = qr.GaugeMomentum(gf.geo)
    hmc.set_gm_force(gm_force, gf, ga)
    return get_array_from_field(gm_force)

### -------------------------------------------------------------------

class TestForceIsTheGradient(unittest.TestCase):
    """The golden finite-difference test, for both action branches."""

    def check_force(self, ga, formula, sites=SITES):
        """Compare ``dS/deps`` with ``-2 basis_a(F)`` for all sampled samples."""
        gf = get_random_field()
        U = get_link_array(gf)
        F = get_force_array(gf, ga)
        worst = 0.0
        for x in sites:
            for mu in range(4):
                for a in range(8):
                    U_plus = U.copy()
                    U_plus[mu][x] = get_su3_exp(FD_EPS * T_GENERATORS[a]) @ U[mu][x]
                    U_minus = U.copy()
                    U_minus[mu][x] = get_su3_exp(-FD_EPS * T_GENERATORS[a]) @ U[mu][x]
                    dS = (
                        get_gauge_action(U_plus, ga) - get_gauge_action(U_minus, ga)
                    ) / (2.0 * FD_EPS)
                    predicted = -2.0 * get_basis_coefficient(formula(U, F, mu, x, a), a)
                    worst = max(worst, abs(dS - predicted))
        self.assertLessEqual(worst, FD_TOL)
        return worst

    def test_force_matches_finite_difference_wilson(self):
        worst = self.check_force(
            qr.GaugeAction(*GAUGE_ACTIONS[0]),
            lambda U, F, mu, x, a: F[mu][x],
        )
        self.assertGreater(worst, 0.0)

    def test_force_matches_finite_difference_with_rectangles(self):
        self.check_force(
            qr.GaugeAction(*GAUGE_ACTIONS[1]),
            lambda U, F, mu, x, a: F[mu][x],
            sites=ONE_SITE,
        )

    def test_wrong_multiplication_order_fails(self):
        # -beta/3 * S^dagger * U  (instead of U * S^dagger): the left/right
        # factor is the whole content of "gradient of S under left
        # multiplication", and the finite difference rejects the alternative by
        # O(1) (measured residual ~ 6.9).
        gf = get_random_field()
        U = get_link_array(gf)
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        beta = ga.beta()
        wrong = np.empty_like(U)
        for mu in range(4):
            staple = hmc._get_all_staple(U, ga, mu)
            wrong[mu] = get_traceless_antihermitian_part(
                (-beta / 3.0) * (get_matrix_adjoint(staple) @ U[mu])
            )
        self.assertGreater(self.get_worst_residual(ga, wrong, sites=ONE_SITE), 1.0)

    def test_wrong_transpose_fails(self):
        # -beta/3 * U * S  (missing adjoint): also rejected by O(1) (measured
        # residual ~ 9.2).
        gf = get_random_field()
        U = get_link_array(gf)
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        beta = ga.beta()
        wrong = np.empty_like(U)
        for mu in range(4):
            staple = hmc._get_all_staple(U, ga, mu)
            wrong[mu] = get_traceless_antihermitian_part(
                (-beta / 3.0) * (U[mu] @ staple)
            )
        self.assertGreater(self.get_worst_residual(ga, wrong, sites=ONE_SITE), 1.0)

    def test_traceless_projection_is_required_for_a_lie_algebra_element(self):
        # The trace part of -beta/3 * U * S^dagger is *invisible* to the finite
        # difference (every T_a is traceless, so Tr[T_a (c 1)] = 0), yet the
        # projection is not optional: without it the result is not the
        # anti-Hermitian traceless element the momentum type requires, and
        # gf_evolve would evolve along a non-unitary direction.
        gf = get_random_field()
        U = get_link_array(gf)
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        beta = ga.beta()
        raw = np.empty_like(U)
        for mu in range(4):
            staple = hmc._get_all_staple(U, ga, mu)
            raw[mu] = (-beta / 3.0) * (U[mu] @ get_matrix_adjoint(staple))
        projected = np.empty_like(U)
        for mu in range(4):
            projected[mu] = get_traceless_antihermitian_part(raw[mu])
            self.assertLessEqual(
                float(np.max(np.abs(np.trace(projected[mu], axis1=-1, axis2=-2)))),
                1e-14,
            )
        # raw is not anti-Hermitian at all ...
        self.assertGreater(float(np.max(np.abs(raw + get_matrix_adjoint(raw)))), 1e-2)
        # ... while the projected form is, and it is what the finite difference
        # of the action agrees with.
        self.assertLessEqual(
            float(np.max(np.abs(projected + get_matrix_adjoint(projected)))), 1e-15
        )
        self.assertLessEqual(
            self.get_worst_residual(ga, projected, sites=ONE_SITE), FD_TOL
        )
        # the trace parts differ, so the projection did change the matrix
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        np.trace(raw, axis1=-2, axis2=-1)
                        - np.trace(projected, axis1=-2, axis2=-1)
                    )
                )
            ),
            1e-3,
        )

    def get_worst_residual(self, ga, F, gf=None, sites=SITES):
        """Return the worst ``|dS/deps + 2 basis_a(F)|`` for a candidate force."""
        if gf is None:
            gf = get_random_field()
        U = get_link_array(gf)
        worst = 0.0
        for x in sites:
            for mu in range(4):
                for a in range(8):
                    U_plus = U.copy()
                    U_plus[mu][x] = get_su3_exp(FD_EPS * T_GENERATORS[a]) @ U[mu][x]
                    U_minus = U.copy()
                    U_minus[mu][x] = get_su3_exp(-FD_EPS * T_GENERATORS[a]) @ U[mu][x]
                    dS = (
                        get_gauge_action(U_plus, ga) - get_gauge_action(U_minus, ga)
                    ) / (2.0 * FD_EPS)
                    worst = max(
                        worst, abs(dS - (-2.0 * get_basis_coefficient(F[mu][x], a)))
                    )
        return worst

class TestForceStructure(unittest.TestCase):
    """Anti-Hermiticity, tracelessness and the in-place contract."""

    def test_force_is_antihermitian_and_traceless(self):
        gf = get_random_field()
        for beta, c1 in GAUGE_ACTIONS:
            F = get_force_array(gf, qr.GaugeAction(beta, c1))
            self.assertLessEqual(get_max_difference(F, -get_matrix_adjoint(F)), 1e-15)
            self.assertLessEqual(
                float(np.max(np.abs(np.trace(F, axis1=-2, axis2=-1)))), 1e-14
            )

    def test_set_gm_force_fills_in_place_and_returns_none(self):
        gf = get_random_field()
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gm_force = qr.GaugeMomentum(gf.geo)
        gm_force.set_unit()
        result = hmc.set_gm_force(gm_force, gf, ga)
        self.assertIsNone(result)
        expected = get_force_array(gf, ga)
        self.assertLessEqual(
            get_max_difference(get_array_from_field(gm_force), expected), 0.0
        )

    def test_output_field_is_initialised_by_the_callee(self):
        # qlat calls ``gm_force.init(geo)`` inside set_gm_force, so a momentum
        # created without a geometry -- or carrying another geometry -- is
        # accepted and re-initialised instead of raising.  Same for the dual
        # setter.  The values must equal those of a pre-allocated output.
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gf = get_random_field()
        expected = get_force_array(gf, ga)
        gm_uninit = qr.GaugeMomentum()
        self.assertIsNone(hmc.set_gm_force(gm_uninit, gf, ga))
        self.assertEqual(gm_uninit.geo, gf.geo)
        self.assertLessEqual(
            get_max_difference(get_array_from_field(gm_uninit), expected), 0.0
        )
        # A momentum on an equal-but-distinct Geometry: geometry is compared by
        # value (qlat's Geometry, and qlat_ref's, define __eq__ that way and
        # qlat holds its own copy), so nothing has to be re-initialised and the
        # caller's buffer is written in place.
        other_geo = qr.Geometry(qr.Coordinate(list(TOTAL_SITE)))
        gm_other_geo = qr.GaugeMomentum(other_geo)
        self.assertIsNot(gm_other_geo.geo, gf.geo)
        hmc.set_gm_force(gm_other_geo, gf, ga)
        self.assertEqual(gm_other_geo.geo, gf.geo)
        self.assertLessEqual(
            get_max_difference(get_array_from_field(gm_other_geo), expected), 0.0
        )
        # A momentum on a *different* geometry is re-initialised onto gf.geo.
        small_geo = qr.Geometry(qr.Coordinate([2, 2, 2, 2]))
        gm_reinit = qr.GaugeMomentum(qr.Geometry(qr.Coordinate([4, 4, 4, 4])))
        self.assertNotEqual(gm_reinit.geo, gf.geo)
        self.assertGreater(gm_reinit.n_sites, gf.geo.local_volume)
        hmc.set_gm_force(gm_reinit, gf, ga)
        self.assertEqual(gm_reinit.geo, small_geo)
        self.assertLessEqual(
            get_max_difference(get_array_from_field(gm_reinit), expected), 0.0
        )
        gm_dual_uninit = qr.GaugeMomentum()
        self.assertIsNone(
            hmc.set_gm_force_dual(gm_dual_uninit, gf, qr.GaugeMomentum(gf.geo))
        )
        self.assertLessEqual(
            get_max_difference(get_array_from_field(gm_dual_uninit), 0.0),
            float(np.max(np.abs(expected))),
        )

    def test_preallocated_output_is_written_in_place(self):
        # the common case must not swap the caller's buffer, so a live
        # np.asarray view of the output keeps following the field
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gf = get_random_field()
        gm_force = qr.GaugeMomentum(gf.geo)
        view = np.asarray(gm_force)
        hmc.set_gm_force(gm_force, gf, ga)
        self.assertLessEqual(get_max_difference(view, np.asarray(gm_force)), 0.0)
        self.assertGreater(float(np.max(np.abs(view))), 1e-3)

    def test_force_is_not_zero_and_scales_with_beta(self):
        # S is linear in beta, so the force is too: this pins the -beta/3
        # normalisation from the other side of the finite difference.
        gf = get_random_field()
        f1 = get_force_array(gf, qr.GaugeAction(2.5, 0.0))
        f2 = get_force_array(gf, qr.GaugeAction(5.0, 0.0))
        self.assertGreater(float(np.max(np.abs(f1))), 1e-3)
        self.assertLessEqual(get_max_difference(f2, 2.0 * f1), 1e-13)

    def test_force_vanishes_on_the_unit_field(self):
        # Every plaquette equals the identity, so the staple is 6 * 1 and
        # -beta/3 U S^dagger = -2 beta * 1, whose traceless anti-Hermitian part
        # is zero (the c1 term is skipped when c1 == 0).
        gf = qr.GaugeField(get_geometry())
        gf.set_unit()
        F = get_force_array(gf, qr.GaugeAction(5.5, 0.0))
        self.assertLessEqual(float(np.max(np.abs(F))), 1e-15)

    def test_force_differs_between_the_two_action_branches(self):
        gf = get_random_field()
        f_plaq = get_force_array(gf, qr.GaugeAction(2.13, 0.0))
        f_rect = get_force_array(gf, qr.GaugeAction(2.13, -0.331))
        self.assertGreater(get_max_difference(f_plaq, f_rect), 1e-2)

class TestForceGaugeCovariance(unittest.TestCase):
    """``F(gU) = g F(U) g^dagger`` -- catches a wrong staple orientation."""

    def test_force_is_gauge_covariant(self):
        gf = get_random_field()
        gt = qr.GaugeTransform(gf.geo)
        gt.set_rand(qr.RngState("force-covariance"), sigma=0.5, n_step=1)
        gf_transformed = gt * gf
        for beta, c1 in GAUGE_ACTIONS:
            ga = qr.GaugeAction(beta, c1)
            F = qr.GaugeMomentum(gf.geo)
            hmc.set_gm_force(F, gf, ga)
            F_transformed = qr.GaugeMomentum(gf.geo)
            hmc.set_gm_force(F_transformed, gf_transformed, ga)
            # the force is a momentum: the adjoint action is at the same site
            F_expected = qr.gm_apply_gauge_transformation(F, gt)
            self.assertLessEqual(
                get_max_difference(
                    get_array_from_field(F_transformed),
                    get_array_from_field(F_expected),
                ),
                1e-12,
            )

    def test_force_average_is_not_shifted_by_the_transformation(self):
        # a gauge-invariant scalar of the force, as an orientation-independent
        # cross-check of the covariance assertion above
        gf = get_random_field()
        gt = qr.GaugeTransform(gf.geo)
        gt.set_rand(qr.RngState("force-invariant"), sigma=0.5, n_step=1)
        ga = qr.GaugeAction(*GAUGE_ACTIONS[1])
        f1 = get_force_array(gf, ga)
        f2 = get_force_array(gt * gf, ga)
        s1 = float(np.sum(np.trace(f1 @ f1, axis1=-2, axis2=-1).real))
        s2 = float(np.sum(np.trace(f2 @ f2, axis1=-2, axis2=-1).real))
        self.assertLessEqual(abs(s1 - s2), 1e-10 * max(1.0, abs(s1)))

class TestStaples(unittest.TestCase):
    """The staple sums are the link products the force is built from."""

    def test_plaquette_staple_closes_six_plaquettes(self):
        # The plaquette staple of direction mu is the sum of the 6 links that
        # close a plaquette with U_mu(x), i.e. six (nu, mu, -nu-1) paths over
        # nu != mu, -mu-1.
        U = get_link_array(get_random_field(sigma=0.5))
        for mu in range(4):
            staple = hmc._get_plaq_staple(U, mu)
            explicit = np.zeros_like(staple)
            for nu in range(-4, 4):
                if nu == mu or -nu - 1 == mu:
                    continue
                explicit = explicit + hmc._get_wilson_line(U, (nu, mu, -nu - 1))
            self.assertLessEqual(get_max_difference(staple, explicit), 0.0)

    def test_wilson_line_of_a_single_hop_is_the_link(self):
        U = get_link_array(get_random_field(sigma=0.5))
        for mu in range(4):
            self.assertLessEqual(
                get_max_difference(hmc._get_wilson_line(U, (mu,)), U[mu]), 0.0
            )
            # a hop forward and the same hop backwards is the identity
            line = hmc._get_wilson_line(U, (mu, -mu - 1))
            self.assertLessEqual(
                get_max_difference(line, np.broadcast_to(np.eye(3), line.shape)),
                1e-14,
            )

    def test_wilson_line_matches_the_plaquette_helper(self):
        # the same rectangle path as gauge_action._get_plaq, so the two
        # implementations of "the plaquette at a site" cannot drift apart
        U = get_link_array(get_random_field(sigma=0.5))
        for mu in range(3):
            for nu in range(mu + 1, 4):
                line = hmc._get_wilson_line(U, (mu, nu, -mu - 1, -nu - 1))
                self.assertLessEqual(
                    get_max_difference(line, _get_plaq(U, mu, nu)), 1e-14
                )

    def test_rectangle_staple_matches_the_rectangle_helper(self):
        # the (mu, mu, nu, ...) 1x2 path built by gauge_action._get_rect is the
        # first of the three rectangle paths of the staple
        U = get_link_array(get_random_field(sigma=0.5))
        for mu in range(3):
            for nu in range(mu + 1, 4):
                line = hmc._get_wilson_line(U, (mu, mu, nu, -mu - 1, -mu - 1, -nu - 1))
                self.assertLessEqual(
                    get_max_difference(line, _get_rect(U, mu, nu)), 1e-14
                )

### -------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
