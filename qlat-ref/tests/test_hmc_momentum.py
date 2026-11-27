"""
Tests of the momentum side of :mod:`qlat_ref.hmc` against live ``qlat``.\n
The momentum initialisation is the only part of HMC whose *exact* numbers are
observable: the C++ ``GaugeMomentum.set_rand`` fills each site from
``RngState(rs, gindex)`` (a **split** of the ASCII form of the flat global site
index, not an index bump) and draws eight Gaussian coefficients per direction
with width ``sigma / sqrt(2)``.  ``tests/data/hmc-vectors.json`` holds the values
the real C++ build produces, so the assertions here are against committed
reference numbers, not against a re-derivation of the same formula.\n
Exactness policy: the RNG **integer** stream is bit-exact (that is
:mod:`qlat_ref.rng`'s own contract) and the momentum **floats** are compared up
to floating-point precision (``1e-14`` absolute, while the entries are ``O(1)``
and the measured difference is ``<= 4.4e-16``).  The tolerance, rather than bit
equality, keeps a different libm (``log``/``sqrt`` in the polar Box-Muller) from
failing the build.
"""

import json
import os
import unittest
from unittest import mock

import numpy as np

import qlat_ref as qr
from qlat_ref import hmc
from qlat_ref import hmc_utils

from qlat_ref.field import _get_basis_antihermitian_matrix

TOTAL_SITE = (2, 2, 2, 2)

# Absolute tolerance for the momentum floats: the measured difference against
# the C++ vectors is <= 4.4e-16 on x86-64 glibc, i.e. a few ULP of an O(1)
# double; 1e-14 leaves room for a different libm without weakening the contract
# (a wrong scheme is off by O(0.1 .. 4), see test_split_not_index_bump).
EPS_MOMENTUM = 1e-14

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

### -------------------------------------------------------------------

def load_vectors():
    """Return the committed C++ HMC vectors (helper)."""
    with open(os.path.join(DATA_DIR, "hmc-vectors.json"), "r", encoding="utf-8") as fp:
        return json.load(fp)

def get_geometry(total_site=TOTAL_SITE):
    """Return the test geometry (helper)."""
    return qr.Geometry(qr.Coordinate(list(total_site)))

def get_momentum(rng, sigma=1.0, total_site=TOTAL_SITE):
    """Return a momentum filled through the qlat path (helper)."""
    gm = qr.GaugeMomentum(get_geometry(total_site))
    gm.set_rand(rng, sigma)
    return gm

def get_momentum_flat(gm):
    """Return the flat ``(local_volume * 4 * 9)`` complex array of a momentum."""
    return np.asarray(gm).reshape(-1)

def get_max_difference(a, b):
    """Return ``max |a - b|`` over two equally shaped arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

### -------------------------------------------------------------------

class TestMomentumVectors(unittest.TestCase):
    """The committed C++ ``GaugeMomentum.set_rand`` numbers."""

    @classmethod
    def setUpClass(cls):
        cls.vectors = load_vectors()["gauge_momentum_cases"]

    def get_case_values(self, case):
        """Return ``(real, imag)`` of a vector case as ``float64`` arrays."""
        return (
            np.asarray(case["values_real"], dtype=np.float64),
            np.asarray(case["values_imag"], dtype=np.float64),
        )

    def check_case(self, name, rng, sigma):
        """Compare one momentum against its committed C++ vector."""
        case = self.vectors[name]
        self.assertEqual(case["sigma"], sigma)
        gm = get_momentum(rng, sigma)
        flat = get_momentum_flat(gm)
        self.assertEqual(flat.shape[0], len(case["values_real"]))
        real, imag = self.get_case_values(case)
        self.assertLessEqual(get_max_difference(flat.real, real), EPS_MOMENTUM)
        self.assertLessEqual(get_max_difference(flat.imag, imag), EPS_MOMENTUM)
        # the kinetic term is part of the vector too: it is the HMC observable
        self.assertLessEqual(
            abs(hmc.gm_hamilton_node(gm) - case["gm_hamilton_node"]),
            EPS_MOMENTUM * max(1.0, abs(case["gm_hamilton_node"])),
        )
        return gm

    def test_root_state_sigma_one(self):
        self.check_case("gm_root_test_rng_sigma1", qr.RngState("test-rng"), 1.0)

    def test_root_state_sigma_half(self):
        self.check_case("gm_root_test_rng_sigma0p5", qr.RngState("test-rng"), 0.5)

    def test_split_state(self):
        self.check_case("gm_split_gm_sigma1", qr.RngState("test-gm").split("gm"), 1.0)

    def test_deep_split_chain(self):
        self.check_case(
            "gm_split_chain_sigma1",
            qr.RngState("test-gm").split("a").split("gm"),
            1.0,
        )

    def test_driver_seed_tree(self):
        # The exact tree run_hmc_pure_gauge uses: rs.split(traj) then
        # .split("set_rand_gauge_momentum").
        self.check_case(
            "gm_hmc_traj_chain",
            qr.RngState("hmc-seed").split("0").split("set_rand_gauge_momentum"),
            1.0,
        )

    def test_set_rand_does_not_mutate_the_rng(self):
        # qlat takes the state by const reference; the momentum stream must be
        # reusable, otherwise the driver's tree would drift between calls.
        rng = qr.RngState("test-rng")
        get_momentum(rng, 1.0)
        self.assertEqual(rng.rand_gen(), qr.RngState("test-rng").rand_gen())

class TestMomentumScheme(unittest.TestCase):
    """The RNG scheme itself, so a wrong derivation cannot look plausible."""

    def test_split_not_index_bump(self):
        # qlat derives the per-site stream with RngState(rs, gindex), i.e. a
        # split of the decimal form of gindex.  The plausible-looking
        # alternative -- setting rs.index = gindex and drawing from there --
        # gives a completely different field (measured max|dM| ~ 4), which is
        # exactly the bug this test pins: the split field must disagree with
        # the index-bumped field by O(1), while agreeing with the C++ vectors.
        rng = qr.RngState("test-rng")
        gm = get_momentum(rng, 1.0)
        sigma = 1.0
        from_bump = np.zeros_like(np.asarray(gm))
        for index in range(gm.geo.local_volume):
            rs = rng.copy()
            rs.index = index
            for mu in range(4):
                basis = np.asarray(
                    [
                        rs.g_rand_gen(center=0.0, sigma=sigma / np.sqrt(2.0))
                        for _ in range(8)
                    ]
                )
                from_bump[index, mu] = _get_basis_antihermitian_matrix(basis)
        self.assertGreater(get_max_difference(np.asarray(gm), from_bump), 0.1)

    def test_sigma_scales_the_coefficients(self):
        # sigma enters once, as the width of every Gaussian, so the whole field
        # (and the kinetic term) scales exactly: K ~ sigma^2.
        gm1 = get_momentum(qr.RngState("sigma-scaling"), 1.0)
        gm2 = get_momentum(qr.RngState("sigma-scaling"), 0.5)
        gm3 = get_momentum(qr.RngState("sigma-scaling"), 2.0)
        a1 = np.asarray(gm1)
        self.assertLessEqual(get_max_difference(np.asarray(gm2), 0.5 * a1), 1e-15)
        self.assertLessEqual(get_max_difference(np.asarray(gm3), 2.0 * a1), 1e-15)
        k1 = hmc.gm_hamilton_node(gm1)
        self.assertLessEqual(
            abs(hmc.gm_hamilton_node(gm2) - 0.25 * k1), 1e-13 * max(1.0, k1)
        )
        self.assertLessEqual(
            abs(hmc.gm_hamilton_node(gm3) - 4.0 * k1), 1e-13 * max(1.0, k1)
        )

    def test_momentum_is_antihermitian_and_traceless(self):
        gm = get_momentum(qr.RngState("traceless"), 1.0)
        arr = np.asarray(gm)
        adjoint = np.swapaxes(np.conjugate(arr), -1, -2)
        self.assertLessEqual(get_max_difference(arr, -adjoint), 1e-15)
        # the diagonal is set to zero by construction, so the trace vanishes to
        # the rounding of the (exactly zero) sum of three zeros
        self.assertLessEqual(
            float(np.max(np.abs(np.trace(arr, axis1=-2, axis2=-1)))), 1e-15
        )

    def test_default_sigma_is_one(self):
        # qlat's GaugeMomentum.set_rand(self, rng, sigma=1.0).
        a = get_momentum(qr.RngState("default-sigma"))
        b = get_momentum(qr.RngState("default-sigma"), 1.0)
        self.assertLessEqual(get_max_difference(np.asarray(a), np.asarray(b)), 0.0)

    def test_free_function_and_method_agree(self):
        # set_rand_gauge_momentum(gm, sigma, rng) and gm.set_rand(rng, sigma)
        # are two entries to one implementation; note qlat's argument order of
        # the free function: sigma BEFORE rng.
        gm1 = qr.GaugeMomentum(get_geometry())
        gm1.set_rand(qr.RngState("agree"), 0.7)
        gm2 = qr.GaugeMomentum(get_geometry())
        hmc.set_rand_gauge_momentum(gm2, 0.7, qr.RngState("agree"))
        self.assertLessEqual(get_max_difference(np.asarray(gm1), np.asarray(gm2)), 0.0)

    def test_gm_hamilton_node_matches_trace_form(self):
        # K = sum (-1/2) Tr(P^2); the same number must come out of the explicit
        # expression, so a change of normalisation cannot hide.
        gm = get_momentum(qr.RngState("kinetic"), 1.0)
        arr = np.asarray(gm)
        expected = float(np.sum(-0.5 * np.trace(arr @ arr, axis1=-2, axis2=-1).real))
        self.assertLessEqual(abs(hmc.gm_hamilton_node(gm) - expected), 1e-13)
        self.assertGreater(hmc.gm_hamilton_node(gm), 0.0)

class TestMomentumInTheDriver(unittest.TestCase):
    """Structural proof that the driver routes the momentum through qlat's tree."""

    def test_driver_uses_the_rng_momentum_path(self):
        # Spy on GaugeMomentum.set_rand: one trajectory must call it exactly
        # once, with sigma = 1.0 and the state rs.split(traj)
        # .split("set_rand_gauge_momentum").  Comparing the *state* rather than
        # the call site keeps the test valid under refactoring of the numbers.
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_unit()
        ga = qr.GaugeAction(5.5, 0.0)
        rs = qr.RngState("driver-spy")
        calls = []
        original = qr.GaugeMomentum.set_rand
        #
        def spy(self, rng, sigma=1.0):
            calls.append((sigma, [rng.rand_gen() for _ in range(4)]))
            return original(self, rng, sigma)
        #
        with mock.patch.object(qr.GaugeMomentum, "set_rand", spy):
            flag, delta_h = hmc_utils.run_hmc_pure_gauge(gf, ga, 3, rs, n_step=2)
        self.assertEqual(len(calls), 1)
        sigma, draws = calls[0]
        self.assertEqual(sigma, 1.0)
        expected = rs.split("3").split("set_rand_gauge_momentum")
        self.assertEqual(draws, [expected.rand_gen() for _ in range(4)])
        self.assertIsInstance(flag, bool)
        self.assertIsInstance(delta_h, float)

### -------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
