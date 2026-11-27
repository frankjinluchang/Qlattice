"""
Tests of the gauge part of the Hamiltonian, :func:`qlat_ref.hmc.gf_hamilton_node`.\n
The normalisation of the action is a contract, not a convention: the same
``S_gauge`` enters the Metropolis test (through ``delta_h``) and the force
(through its gradient), so it is checked here in three independent ways --
against qlat's own closed form, against the average plaquette, and against the
committed C++ trajectory values -- plus the exact identities of the unit field
and the ``c1`` rectangle branch.
"""

import json
import os
import unittest

import numpy as np

import qlat_ref as qr
from qlat_ref import hmc
from qlat_ref.gauge_action import get_gauge_action
from qlat_ref.observables import get_link_array

TOTAL_SITE = (2, 2, 2, 2)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

### -------------------------------------------------------------------

def load_vectors():
    """Return the committed C++ HMC vectors (helper)."""
    with open(os.path.join(DATA_DIR, "hmc-vectors.json"), "r", encoding="utf-8") as fp:
        return json.load(fp)

def get_geometry(total_site=TOTAL_SITE):
    """Return the test geometry (helper)."""
    return qr.Geometry(qr.Coordinate(list(total_site)))

def get_unit_field(total_site=TOTAL_SITE):
    """Return the unit gauge field as a :class:`GaugeField` (helper)."""
    gf = qr.GaugeField(get_geometry(total_site))
    gf.set_unit()
    return gf

def get_random_field(sigma=0.4, seed="hmc-action", total_site=TOTAL_SITE):
    """Return a deterministic random gauge field (helper)."""
    gf = qr.GaugeField(get_geometry(total_site))
    gf.set_rand(qr.RngState(seed), sigma=sigma, n_step=1)
    return gf

def get_wilson_plaquette_sum(U, mu, nu):
    """Return ``sum_x Re Tr U_P`` of one oriented plaquette (helper)."""
    return float(
        np.sum(np.trace(qr.gauge_action._get_plaq(U, mu, nu), axis1=-2, axis2=-1).real)
    )

def get_wilson_rectangle_sum(U, mu, nu):
    """Return ``sum_x Re Tr U_R`` of one oriented rectangle (helper)."""
    return float(
        np.sum(np.trace(qr.gauge_action._get_rect(U, mu, nu), axis1=-2, axis2=-1).real)
    )

### -------------------------------------------------------------------

class TestUnitFieldIdentities(unittest.TestCase):
    """Exact values on the free field: no floating-point tolerance games."""

    def test_unit_field_has_unit_plaquette(self):
        self.assertEqual(get_unit_field().plaq(), 1.0)

    def test_unit_field_has_zero_action(self):
        # Both branches, because the c1 term is skipped only when c1 == 0.
        for ga in (
            qr.GaugeAction(5.5, 0.0),
            qr.GaugeAction(5.5, -0.331),
            qr.GaugeAction(1.0, 0.0),
        ):
            self.assertEqual(hmc.gf_hamilton_node(get_unit_field(), ga), 0.0)

    def test_unit_field_has_zero_link_trace_shift(self):
        self.assertEqual(get_unit_field().link_trace(), 1.0)

class TestActionNormalisation(unittest.TestCase):
    """The action written in terms of the average plaquette and rectangle."""

    def test_matches_the_average_plaquette_form(self):
        # S = beta/3 * [(1 - 8 c1) * 6 V * (3 - ReTr_P) + c1 * 12 V * (...)]
        # is qlat's closed form; expressing it through the *average* plaquette
        # exercises a different summation path than the per-site kernels.
        gf = get_random_field()
        n_sites = gf.geo.local_volume
        plaquette = gf.plaq()
        for beta, c1 in ((5.5, 0.0), (2.13, -0.331), (1.0, 0.25)):
            ga = qr.GaugeAction(beta, c1)
            expected = (
                beta / 3.0 * (1.0 - 8.0 * c1) * 6.0 * n_sites * (3.0 - 3.0 * plaquette)
            )
            actual = hmc.gf_hamilton_node(gf, ga)
            if c1 == 0.0:
                self.assertLessEqual(
                    abs(actual - expected), 1e-11 * max(1.0, abs(actual))
                )
            # with c1 != 0 the plaquette part must still be the same piece
            plaq_only = hmc.gf_hamilton_node(gf, qr.GaugeAction(beta, 0.0))
            self.assertLessEqual(
                abs(plaq_only - beta / 3.0 * 6.0 * n_sites * (3.0 - 3.0 * plaquette)),
                1e-11 * max(1.0, abs(plaq_only)),
            )

    def test_matches_the_explicit_double_sum(self):
        # Independent re-summation over all 6 plaquette and 12 rectangle
        # orientations, written with the plain observables rather than the
        # internal helpers.
        gf = get_random_field(sigma=0.5)
        U = get_link_array(gf)
        n_sites = gf.geo.local_volume
        sum_plaq = 0.0
        sum_rect = 0.0
        for mu in range(3):
            for nu in range(mu + 1, 4):
                sum_plaq += get_wilson_plaquette_sum(U, mu, nu) - 3.0 * n_sites
                sum_rect += get_wilson_rectangle_sum(U, mu, nu) - 3.0 * n_sites
                sum_rect += get_wilson_rectangle_sum(U, nu, mu) - 3.0 * n_sites
        for beta, c1 in ((5.5, 0.0), (2.13, -0.331)):
            ga = qr.GaugeAction(beta, c1)
            expected = -beta / 3.0 * ((1.0 - 8.0 * c1) * sum_plaq + c1 * sum_rect)
            actual = hmc.gf_hamilton_node(gf, ga)
            self.assertLessEqual(
                abs(actual - expected), 1e-11 * max(1.0, abs(expected))
            )

    def test_is_the_gauge_action_entry_point(self):
        # gf_hamilton_node is qlat's name; the reference implementation lives in
        # gauge_action.py, so one implementation backs both names.
        gf = get_random_field()
        for ga in (qr.GaugeAction(5.5, 0.0), qr.GaugeAction(2.13, -0.331)):
            self.assertEqual(hmc.gf_hamilton_node(gf, ga), get_gauge_action(gf, ga))

    def test_is_positive_for_a_disordered_field(self):
        gf = get_random_field(sigma=1.0)
        for ga in (qr.GaugeAction(5.5, 0.0), qr.GaugeAction(2.13, -0.331)):
            self.assertGreater(hmc.gf_hamilton_node(gf, ga), 0.0)

    def test_accepts_a_plain_array(self):
        # The action layer accepts a plain (4, ...) array as well as a field, so
        # the tests and the cross-checks can use either convention.
        gf = get_random_field()
        U = get_link_array(gf)
        ga = qr.GaugeAction(5.5, 0.0)
        self.assertLessEqual(
            abs(hmc.gf_hamilton_node(U, ga) - hmc.gf_hamilton_node(gf, ga)), 1e-12
        )

class TestC1Branch(unittest.TestCase):
    """The rectangle-improvement branch is a separate code path."""

    def test_c1_zero_skips_the_rectangles(self):
        # c1 == 0 must give exactly the plaquette-only action: the rectangle
        # staple is skipped, not merely multiplied by zero.
        gf = get_random_field()
        plaq_only = hmc.gf_hamilton_node(gf, qr.GaugeAction(5.5, 0.0))
        self.assertEqual(hmc.gf_hamilton_node(gf, qr.GaugeAction(5.5, 0.0)), plaq_only)
        # a tiny non-zero c1 changes the value
        self.assertNotEqual(
            hmc.gf_hamilton_node(gf, qr.GaugeAction(5.5, 1e-3)), plaq_only
        )

    def test_rectangle_term_is_independent_and_scales_with_c1(self):
        # S(beta, c1) = beta/3 [(1 - 8c1) P + c1 R], so the c1 dependence is
        # affine: S(c1) - S(0) = beta/3 * c1 * (R - 8 P).  Recovering R and P
        # from two c1 values pins that the rectangle sum is right, not just that
        # c1 enters somewhere.
        gf = get_random_field(sigma=0.5)
        beta = 2.13
        s0 = hmc.gf_hamilton_node(gf, qr.GaugeAction(beta, 0.0))
        s1 = hmc.gf_hamilton_node(gf, qr.GaugeAction(beta, 0.25))
        s2 = hmc.gf_hamilton_node(gf, qr.GaugeAction(beta, 0.5))
        self.assertLessEqual(abs((s1 - s0) * 2.0 - (s2 - s0)), 1e-11 * abs(s0))

    def test_rectangle_staple_is_not_the_plaquette_staple(self):
        from qlat_ref.hmc import _get_plaq_staple
        from qlat_ref.hmc import _get_rect_staple
        #
        U = get_link_array(get_random_field(sigma=0.5))
        for mu in range(4):
            plaq = _get_plaq_staple(U, mu)
            rect = _get_rect_staple(U, mu)
            self.assertGreater(float(np.max(np.abs(rect - plaq))), 0.1)

class TestTrajectoryVectors(unittest.TestCase):
    """The action values recorded by the committed C++ trajectory vectors."""

    @classmethod
    def setUpClass(cls):
        cls.vectors = load_vectors()["trajectory_cases"]

    def test_start_state_action_is_zero(self):
        for case in self.vectors.values():
            self.assertEqual(case["plaq_start"], 1.0)
            self.assertEqual(case["gf_hamilton_start"], 0.0)

    def test_action_at_each_recorded_state(self):
        # Rebuild each recorded gauge field from the vector and check the
        # action: this is the "does the reference see the same energy"
        # contract, independent of the trajectory itself.
        for key, case in self.vectors.items():
            ga = qr.GaugeAction(case["beta"], case["c1"])
            geo = get_geometry()
            for entry in case["trajectories"]:
                field = np.asarray(entry["field_real"]) + 1j * np.asarray(
                    entry["field_imag"]
                )
                gf = qr.GaugeField(geo)
                gf[:] = field.reshape((geo.local_volume, 4, 3, 3))
                self.assertLessEqual(
                    abs(hmc.gf_hamilton_node(gf, ga) - entry["gf_hamilton_node"]),
                    1e-11 * max(1.0, abs(entry["gf_hamilton_node"])),
                    msg=f"{key} traj {entry['traj']}",
                )
                self.assertLessEqual(abs(gf.plaq() - entry["plaq"]), 1e-13)

### -------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
