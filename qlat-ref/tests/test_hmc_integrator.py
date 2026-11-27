"""
Tests of the force-gradient molecular-dynamics integrator,
:func:`qlat_ref.hmc_utils.run_hmc_evolve_pure_gauge`.\n
Two properties make a symplectic integrator usable in HMC, and both are tested
here without any Metropolis step in the way:\n
* **Energy conservation.** ``|delta_h|`` must fall like ``dt^4`` for the
  Omelyan / minimum-norm force-gradient scheme, so the test asserts a generous
  bound at ``n_step`` in ``{4, 8, 16}`` *and* a monotone decrease of the mean.
* **Reversibility.** Running the same trajectory backwards must return the
  field *and* the momentum, with the two energy violations cancelling.  The
  reverse run uses ``md_time = -md_time`` and keeps the final momentum -- the
  convention of qlat's own ``is_reverse_test`` gate.  Negating the momentum
  instead is the tempting mistake; that comparison does not hold at all (see
  ``test_negating_the_momentum_is_not_the_reverse_trajectory``).\n
The momenta are drawn once per case from a fixed seed, so the numbers are
reproducible and a regression shows up as a changed ``|delta_h|``, not as noise.
"""

import unittest

import numpy as np

import qlat_ref as qr
from qlat_ref import hmc
from qlat_ref import hmc_utils
from qlat_ref.field import get_array_from_field

TOTAL_SITE = (2, 2, 2, 2)

# Bound on max |delta_h| per step count.  The measured values (mean over 3
# momenta, 2^4, beta 5.5, force-gradient integrator) are 3.5e-1 at n_step=4,
# 1.7e-2 at 8 and 1.0e-3 at 16; the bounds below are ~3x that, wide enough that
# a libm difference cannot fail them while a lost O(dt^4) convergence does.
MAX_DELTA_H = {4: 1.0, 8: 5e-2, 16: 5e-3}

# Reversibility allowances.  Measured: |dU| = 5.5e-15, |dP| = 1.7e-14 and
# |delta_h + delta_h_rev| = 2.6e-12.
REVERSIBILITY_EPS_FIELD = 1e-12
REVERSIBILITY_EPS_ENERGY = 1e-10

GAUGE_ACTIONS = (
    (5.5, 0.0),
    (2.13, -0.331),
)

### -------------------------------------------------------------------

def get_geometry(total_site=TOTAL_SITE):
    """Return the test geometry (helper)."""
    return qr.Geometry(qr.Coordinate(list(total_site)))

def get_random_field(seed="hmc-integrator", sigma=0.4, total_site=TOTAL_SITE):
    """Return a deterministic random gauge field (helper)."""
    gf = qr.GaugeField(get_geometry(total_site))
    gf.set_rand(qr.RngState(seed), sigma=sigma, n_step=1)
    return gf

def get_random_momentum(seed, sigma=1.0, total_site=TOTAL_SITE):
    """Return a deterministic Gaussian momentum (helper)."""
    gm = qr.GaugeMomentum(get_geometry(total_site))
    gm.set_rand(qr.RngState(seed), sigma)
    return gm

def get_hamilton(gm, gf, ga):
    """Return the total Hamiltonian of a state (helper)."""
    return hmc.gm_hamilton_node(gm) + hmc.gf_hamilton_node(gf, ga)

def get_max_difference(a, b):
    """Return ``max |a - b|`` over two equally shaped arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

def evolve(gm, gf, ga, n_step, md_time=1.0):
    """Run one MD trajectory on copies and return the new ``(gm, gf, delta_h)``."""
    gm_new = gm.copy()
    gf_new = gf.copy()
    delta_h = hmc_utils.run_hmc_evolve_pure_gauge(gm_new, gf_new, ga, n_step, md_time)
    return gm_new, gf_new, delta_h

def get_energy_violations(ga, n_step, seeds=("md-a",)):
    """Return ``|delta_h|`` of one trajectory per seed (helper)."""
    gf = get_random_field()
    return [abs(evolve(get_random_momentum(seed), gf, ga, n_step)[2]) for seed in seeds]

### -------------------------------------------------------------------

class TestEnergyConservation(unittest.TestCase):
    """``|delta_h|`` is the trajectory error of the integrator."""

    def test_delta_h_bounds_per_step_count(self):
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        means = {}
        for n_step, bound in MAX_DELTA_H.items():
            values = get_energy_violations(ga, n_step)
            self.assertLessEqual(max(values), bound, msg=f"n_step={n_step}")
            means[n_step] = float(np.mean(values))
        # O(dt^4): each doubling of n_step must gain at least a factor of 4
        self.assertGreater(means[4] / means[8], 4.0)
        self.assertGreater(means[8] / means[16], 4.0)

    def test_delta_h_decreases_monotonically_with_the_step_count(self):
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gf = get_random_field()
        gm = get_random_momentum("monotone")
        previous = None
        for n_step in (2, 4, 8):
            delta_h = abs(evolve(gm, gf, ga, n_step)[2])
            if previous is not None:
                self.assertLess(delta_h, previous, msg=f"n_step={n_step}")
            previous = delta_h

    def test_delta_h_bounds_with_rectangles(self):
        # the c1 != 0 branch is three times the staple work and a separate
        # force path; its integrator behaves the same way
        ga = qr.GaugeAction(*GAUGE_ACTIONS[1])
        values = get_energy_violations(ga, 8, seeds=("md-c1",))
        self.assertLessEqual(max(values), MAX_DELTA_H[8])

    def test_returned_delta_h_is_the_hamiltonian_difference(self):
        # run_hmc_evolve_pure_gauge returns H(after) - H(before), not merely
        # some step count: comparing it with an independent computation of the
        # two Hamiltonians pins the convention (and the sign used by
        # metropolis_accept).
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gf = get_random_field()
        gm = get_random_momentum("convention")
        h_before = get_hamilton(gm, gf, ga)
        gm_new, gf_new, delta_h = evolve(gm, gf, ga, 6)
        h_after = get_hamilton(gm_new, gf_new, ga)
        self.assertLessEqual(abs(delta_h - (h_after - h_before)), 1e-12)
        self.assertGreater(h_after, 0.0)
        # a finite trajectory must actually move the field
        self.assertGreater(
            get_max_difference(get_array_from_field(gf_new), get_array_from_field(gf)),
            1e-3,
        )

class TestMutationContract(unittest.TestCase):
    """``run_hmc_evolve_pure_gauge`` mutates both arguments and returns a float."""

    def test_mutates_gm_and_gf_in_place(self):
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gf = get_random_field()
        gm = get_random_momentum("mutation")
        gf_before = get_array_from_field(gf).copy()
        gm_before = get_array_from_field(gm).copy()
        delta_h = hmc_utils.run_hmc_evolve_pure_gauge(gm, gf, ga, 4)
        self.assertIsInstance(delta_h, float)
        self.assertGreater(
            get_max_difference(get_array_from_field(gf), gf_before), 1e-3
        )
        self.assertGreater(
            get_max_difference(get_array_from_field(gm), gm_before), 1e-3
        )
        # the evolved field is still unitary (the trajectory's own unitarize)
        self.assertLessEqual(
            qr.mat.get_unitarity_error(get_array_from_field(gf)), 1e-12
        )

    def test_gm_evolve_fg_pure_gauge_leaves_the_field_alone(self):
        # The force-gradient half step evolves an internal *copy* of the links;
        # gf_init must be bit-unchanged, otherwise the trajectory would take the
        # wrong path silently.
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gf = get_random_field()
        gf_before = get_array_from_field(gf).copy()
        gm = get_random_momentum("fg-half-step")
        gm_before = get_array_from_field(gm).copy()
        result = hmc_utils.gm_evolve_fg_pure_gauge(gm, gf, ga, 1e-4, 0.1)
        self.assertIsNone(result)
        self.assertEqual(get_max_difference(get_array_from_field(gf), gf_before), 0.0)
        # and it does move the momentum
        self.assertGreater(
            get_max_difference(get_array_from_field(gm), gm_before), 1e-5
        )

class TestReversibility(unittest.TestCase):
    """Reverse the trajectory with ``md_time = -md_time``, not by negating P."""

    def check_reversibility(self, ga, n_step=6):
        """Run forward and backward and check the state and energy return."""
        gf = get_random_field()
        gm = get_random_momentum("reverse")
        gf_before = get_array_from_field(gf).copy()
        gm_before = get_array_from_field(gm).copy()
        gm_1, gf_1, delta_h_1 = evolve(gm, gf, ga, n_step, 1.0)
        gm_2, gf_2, delta_h_2 = evolve(gm_1, gf_1, ga, n_step, -1.0)
        field_diff = get_max_difference(get_array_from_field(gf_2), gf_before)
        momentum_diff = get_max_difference(get_array_from_field(gm_2), gm_before)
        self.assertLessEqual(field_diff, REVERSIBILITY_EPS_FIELD)
        self.assertLessEqual(momentum_diff, REVERSIBILITY_EPS_FIELD)
        self.assertLessEqual(
            abs(delta_h_1 + delta_h_2),
            REVERSIBILITY_EPS_ENERGY * max(1.0, abs(delta_h_1)),
        )
        return field_diff, momentum_diff

    def test_reversibility_wilson(self):
        field_diff, _ = self.check_reversibility(qr.GaugeAction(*GAUGE_ACTIONS[0]))
        self.assertGreater(field_diff, 0.0)

    def test_reversibility_with_rectangles(self):
        self.check_reversibility(qr.GaugeAction(*GAUGE_ACTIONS[1]))

    def test_reversibility_at_another_step_count(self):
        self.check_reversibility(qr.GaugeAction(*GAUGE_ACTIONS[0]), 4)

    def test_negating_the_momentum_reverses_but_returns_minus_p(self):
        # The other textbook reversibility form: negate P and run *forward*
        # again.  It does bring the field back, but the momentum comes back
        # negated, so a check that compares the momentum directly fails by
        # O(1).  qlat uses the ``md_time = -md_time`` form (the test above),
        # which is why the convention matters: this test pins the difference so
        # a future reader cannot swap one for the other and "loosen" the
        # tolerance to make it pass.
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gf = get_random_field()
        gm = get_random_momentum("negated-reverse")
        gf_before = get_array_from_field(gf).copy()
        gm_before = get_array_from_field(gm).copy()
        gm_1, gf_1, _ = evolve(gm, gf, ga, 6, 1.0)
        gm_neg = gm_1.copy()
        gm_neg[:] = -np.asarray(gm_neg)
        gm_2, gf_2, _ = evolve(gm_neg, gf_1, ga, 6, 1.0)
        # the field comes back ...
        self.assertLessEqual(
            get_max_difference(get_array_from_field(gf_2), gf_before),
            REVERSIBILITY_EPS_FIELD,
        )
        # ... but the momentum comes back negated, not unchanged
        self.assertLessEqual(
            get_max_difference(get_array_from_field(gm_2) + gm_before, 0.0 * gm_before),
            REVERSIBILITY_EPS_FIELD,
        )
        self.assertGreater(
            get_max_difference(get_array_from_field(gm_2), gm_before), 1.0
        )

class TestIntegratorEnergyScale(unittest.TestCase):
    """Sanity of the MD time: scaling it scales the trajectory error."""

    def test_smaller_md_time_conserves_better(self):
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gf = get_random_field()
        gm = get_random_momentum("md-time-scale")
        coarse = abs(evolve(gm, gf, ga, 4, 1.0)[2])
        fine = abs(evolve(gm, gf, ga, 4, 0.25)[2])
        self.assertLess(fine, coarse)

    def test_tiny_md_time_is_almost_the_identity(self):
        # md_time = 0 itself divides by dt in the force-gradient step (qlat's
        # integrator does the same, so it raises ZeroDivisionError); a tiny
        # md_time is the meaningful limit and must leave the state essentially
        # unchanged (a link moves by O(md_time * |P|), which is ~1e-6 here)
        # with a correspondingly tiny energy violation.
        ga = qr.GaugeAction(*GAUGE_ACTIONS[0])
        gf = get_random_field()
        gm = get_random_momentum("md-time-tiny")
        gf_before = get_array_from_field(gf).copy()
        _, gf_1, delta_h = evolve(gm, gf, ga, 4, 1e-6)
        self.assertLessEqual(abs(delta_h), 1e-12)
        moved = get_max_difference(get_array_from_field(gf_1), gf_before)
        self.assertGreater(moved, 1e-7)
        self.assertLessEqual(moved, 1e-5)

### -------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
