"""
Tests of :mod:`qlat_ref.cgfhmc`, the Canonical Gauge-Fixed HMC layer.\n
Three levels, in this order of importance:\n
1. **The c(k) field**, because everything downstream assumes its invariants:
   the mode grid and its folding, the ``(Lx, Ly, Lz, Lt)`` axis order (the
   lattice is the only place a transposition would survive unnoticed), and the
   regularizer's three jobs -- clip into ``[0, 1]``, enforce ``c(-k) = c(k)``
   and police NaN/inf without silently repairing them.
2. **The refresh**, whose contract is what makes the algorithm correct: the
   Gaussian ensemble must be preserved *mode by mode*, the momentum must stay
   anti-Hermitian and traceless, ``c = 0`` must be a strict no-op that consumes
   **no random number**, ``c = 1`` must be a complete resample, and the uniform
   fast path must be mathematically identical to mixing in Fourier space.
3. **The driver**, including the ``dK`` accounting that keeps the
   refresh-induced kinetic change out of the Metropolis test.\n
Everything is single process and needs only ``numpy``.  The tolerances are
absolute where a quantity can legitimately be zero (``dK``, an energy
violation, an anti-Hermiticity residual): a relative epsilon on such a value
would be meaningless, and comparing it bit-for-bit would be machine-sensitive.
Neither this module nor the implementation asserts on any hex string.\n
The multi-trajectory chains that only check statistics live in
:mod:`tests.test_cgfhmc_slow`, behind ``QLAT_REF_SLOW=1``.
"""

import math
import os
import tempfile
import unittest

import numpy as np

import qlat_ref as qr
from qlat_ref import cgfhmc
from qlat_ref.field import get_array_from_field
from qlat_ref.gauge_fixing import fix_landau_gauge
from qlat_ref.hmc import gf_evolve
from qlat_ref.hmc_utils import gm_evolve_fg_pure_gauge

TOTAL_SITE = (2, 2, 2, 2)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Tolerances.  ``EPS_FIELD`` pins a field against an independently computed one
# (a few ULP of the values involved); ``EPS_DK`` is absolute and generous
# because dK is a difference of two O(V) kinetic energies and is itself often
# near zero; ``EPS_HERM`` bounds a residual that must vanish by construction.
EPS_FIELD = 1e-13
EPS_DK = 1e-10
EPS_HERM = 1e-14

# Actions for the short chains: the endpoint of the coupling range the HMC
# vectors use, and the rectangle-improved path.
CHAIN_ACTIONS = ((5.5, 0.0), (2.13, -0.331))

### -------------------------------------------------------------------

def get_geometry(total_site=TOTAL_SITE):
    """Return a :class:`qlat_ref.Geometry` of ``total_site`` (helper)."""
    return qr.Geometry(qr.Coordinate(list(total_site)))

def get_c_field(geo, func):
    """Return the regularized ``c(k)`` field of ``func`` on ``geo`` (helper)."""
    return cgfhmc.generate_c_field(geo, func)

def get_random_momentum(geo, seed="momentum"):
    """Return a Gaussian momentum drawn from ``seed`` (helper)."""
    gm = qr.GaugeMomentum(geo)
    gm.set_rand(qr.RngState(seed), 1.0)
    return gm

def get_uniform_c(geo, c):
    """Return the constant ``c(k) = c`` field on ``geo`` (helper)."""
    return get_c_field(geo, lambda kv: c)

def get_two_zone_c(geo, c_low=0.35, c_high=0.9):
    """Return a two-zone ``c(k)``: stiff infrared, aggressive ultraviolet (helper)."""
    threshold = cgfhmc.resolve_k_cut(geo, 1)
    return get_c_field(
        geo, lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, c_low, c_high)
    )

def get_anticorrelation_residual(gm):
    """Return ``max|P + P^dagger|``, the anti-Hermiticity residual (helper)."""
    arr = get_array_from_field(gm)
    return float(np.max(np.abs(arr + np.conj(np.swapaxes(arr, -1, -2)))))

def get_traceless_residual(gm):
    """Return ``max|Tr P|``, the tracelessness residual (helper)."""
    arr = get_array_from_field(gm)
    return float(np.max(np.abs(np.trace(arr, axis1=-2, axis2=-1))))

### -------------------------------------------------------------------

class TestModeGrid(unittest.TestCase):
    """The momentum grid: shapes, dtype, folding, and the axis order."""

    def test_shapes_and_dtype(self):
        geo = get_geometry()
        k_vec = cgfhmc.k_vectors(geo)
        self.assertEqual(k_vec.shape, (4,) + TOTAL_SITE)
        self.assertEqual(k_vec.dtype, np.float64)
        k_sq = cgfhmc.k_squared(geo)
        self.assertEqual(k_sq.shape, TOTAL_SITE)
        self.assertEqual(k_sq.dtype, np.float64)

    def test_k_squared_is_the_component_sum(self):
        for total_site in (TOTAL_SITE, (2, 3, 4, 2), (2, 2, 4, 2)):
            with self.subTest(total_site=total_site):
                geo = get_geometry(total_site)
                k_vec = cgfhmc.k_vectors(geo)
                np.testing.assert_allclose(
                    cgfhmc.k_squared(geo),
                    np.sum(k_vec**2, axis=0),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_component_varies_only_along_its_own_axis(self):
        # A transposed axis -- the one bug the rest of the suite cannot see --
        # shows up here as a component that is not constant along its own axes.
        total_site = (2, 3, 4, 2)
        geo = get_geometry(total_site)
        k_vec = cgfhmc.k_vectors(geo)
        for i in range(4):
            with self.subTest(component=i):
                # Along every axis but i the component is constant.
                for axis in range(4):
                    if axis == i:
                        continue
                    first = np.take(k_vec[i], 0, axis=axis)
                    for n in range(total_site[axis]):
                        np.testing.assert_array_equal(
                            np.take(k_vec[i], n, axis=axis), first
                        )

    def test_folding_maps_the_upper_half_to_negative_momenta(self):
        # The partner of index n is (-n) % L, and k((-n) % L) == -k(n) -- which
        # is the whole point of folding.  The self-partner n == L/2 (the Nyquist
        # momentum) is excluded because there +pi and -pi are the same mode.
        geo = get_geometry((4, 4, 4, 4))
        k_vec = cgfhmc.k_vectors(geo)
        for i in range(4):
            with self.subTest(axis=i):
                axis_values = k_vec[i].max(axis=tuple(j for j in range(4) if j != i))
                for n in range(4):
                    partner = (-n) % 4
                    if partner == n:
                        self.assertIn(abs(axis_values[n]), (0.0, math.pi))
                        continue
                    np.testing.assert_allclose(
                        axis_values[partner], -axis_values[n], rtol=0.0, atol=EPS_FIELD
                    )

    def test_folded_range_and_nyquist(self):
        # The folded momentum covers [-pi, pi); an even axis reaches +pi at the
        # Nyquist index and an odd one never does.
        geo = get_geometry((4, 3, 4, 2))
        k_vec = cgfhmc.k_vectors(geo)
        self.assertGreaterEqual(float(k_vec.min()), -math.pi - EPS_FIELD)
        self.assertLessEqual(float(k_vec.max()), math.pi + EPS_FIELD)
        axis_x = k_vec[0].max(axis=(1, 2, 3))
        axis_y = k_vec[1].max(axis=(0, 2, 3))
        np.testing.assert_allclose(
            axis_x, [0.0, math.pi / 2, math.pi, -math.pi / 2], atol=EPS_FIELD
        )
        np.testing.assert_allclose(
            axis_y, [0.0, 2 * math.pi / 3, -2 * math.pi / 3], atol=EPS_FIELD
        )

    def test_resolve_k_cut(self):
        geo = get_geometry((8, 4, 4, 4))
        self.assertAlmostEqual(cgfhmc.resolve_k_cut(geo, 1), 2 * math.pi / 8, places=15)
        self.assertAlmostEqual(
            cgfhmc.resolve_k_cut(geo, 2), 2 * (2 * math.pi / 8), places=15
        )
        self.assertEqual(cgfhmc.resolve_k_cut(geo, 0), 0.0)

### -------------------------------------------------------------------

class TestRegularizeCField(unittest.TestCase):
    """``regularize_c_field``: range, evenness, NaN policy, layout."""

    def test_constant_field_survives_unchanged(self):
        get_geometry()
        for c in (0.0, 0.3, 0.5, 1.0):
            with self.subTest(c=c):
                arr = cgfhmc.regularize_c_field(np.full(TOTAL_SITE, c))
                np.testing.assert_array_equal(arr, np.full(TOTAL_SITE, c))

    def test_out_of_range_values_are_clipped_not_rescaled(self):
        get_geometry()
        arr = np.full(TOTAL_SITE, 0.5)
        arr[0, 0, 0, 0] = -0.5
        arr[1, 1, 1, 1] = 2.0
        out = cgfhmc.regularize_c_field(arr)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        # Clipping saturates the offending entries rather than shrinking the rest.
        self.assertEqual(cgfhmc.regularize_c_field.last_report["n_clipped_low"], 1)
        self.assertEqual(cgfhmc.regularize_c_field.last_report["n_clipped_high"], 1)

    def test_evenness_is_enforced_by_averaging(self):
        # (4, 2, 2, 2), not 4^4: what is checked is the flip-then-roll partner
        # formula along an even axis, which (4, 2, 2, 2) exercises -- including
        # the Nyquist self-partner -- for 1/16 of the elements.  The
        # (4, 4, 4, 4) variant lives in tests/test_cgfhmc_slow.py.
        geo = get_geometry((4, 2, 2, 2))
        # An explicitly odd prescription: c depends on the sign of k_x.
        odd = np.where(cgfhmc.k_vectors(geo)[0] >= 0.0, 0.0, 0.8)
        out = cgfhmc.regularize_c_field(odd)
        partner = out[(-np.arange(4)) % 4][:, :, :, :]
        np.testing.assert_allclose(out, partner, rtol=0.0, atol=0.0)
        # The odd input had a 0.8 swing.  Averaging with the partner halves it,
        # so the pair (n, L-n) meets in the middle, while the self-partners
        # n = 0 and n = L/2 keep their own value (0.0 here).
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 0.0, places=15)
        self.assertAlmostEqual(float(out[1, 0, 0, 0]), 0.4, places=15)
        self.assertAlmostEqual(float(out[2, 0, 0, 0]), 0.0, places=15)
        self.assertAlmostEqual(float(out[3, 0, 0, 0]), 0.4, places=15)

    def test_non_finite_raises_by_default(self):
        arr = np.full(TOTAL_SITE, 0.5)
        arr[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            cgfhmc.regularize_c_field(arr)

    def test_non_finite_is_zeroed_on_request(self):
        arr = np.full(TOTAL_SITE, 0.5)
        arr[0, 0, 0, 0] = np.nan
        arr[1, 1, 1, 1] = np.inf
        out = cgfhmc.regularize_c_field(arr, allow_nan="zero")
        self.assertEqual(cgfhmc.regularize_c_field.last_report["n_nonfinite_zeroed"], 2)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(float(out[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(out[1, 1, 1, 1]), 0.0)

    def test_bad_allow_nan_is_rejected(self):
        arr = np.full(TOTAL_SITE, np.nan)
        with self.assertRaises(ValueError):
            cgfhmc.regularize_c_field(arr, allow_nan="ignore")

    def test_complex_input_is_rejected(self):
        arr = np.full(TOTAL_SITE, 0.5 + 0.0j)
        np.testing.assert_array_equal(
            cgfhmc.regularize_c_field(arr), np.full(TOTAL_SITE, 0.5)
        )
        arr[0, 0, 0, 0] = 0.5 + 0.25j
        with self.assertRaises(ValueError):
            cgfhmc.regularize_c_field(arr)

    def test_three_dimensional_input_is_rejected(self):
        with self.assertRaises(ValueError):
            cgfhmc.regularize_c_field(np.full((2, 2, 2), 0.5))

    def test_in_place_writes_into_the_caller_array(self):
        arr = np.full(TOTAL_SITE, 0.5)
        out = cgfhmc.regularize_c_field(arr, in_place=True)
        self.assertIs(out, arr)
        out[0, 0, 0, 0] = 0.25
        self.assertEqual(float(arr[0, 0, 0, 0]), 0.25)

    def test_copy_by_default(self):
        arr = np.full(TOTAL_SITE, 0.5)
        out = cgfhmc.regularize_c_field(arr)
        self.assertIsNot(out, arr)
        out[0, 0, 0, 0] = 0.25
        self.assertEqual(float(arr[0, 0, 0, 0]), 0.5)

### -------------------------------------------------------------------

class TestGenerateCField(unittest.TestCase):
    """``generate_c_field``: the callable contract and the returned layout."""

    def test_returned_field_is_usable_as_is(self):
        geo = get_geometry((2, 3, 4, 2))
        c = get_two_zone_c(geo)
        self.assertEqual(c.shape, (2, 3, 4, 2))
        self.assertEqual(c.dtype, np.float64)
        self.assertTrue(c.flags["C_CONTIGUOUS"])
        self.assertGreaterEqual(float(c.min()), 0.0)
        self.assertLessEqual(float(c.max()), 1.0)

    def test_generator_outputs_are_already_even(self):
        geo = get_geometry((4, 4, 4, 4))
        cgfhmc.k_vectors(geo)
        threshold = cgfhmc.resolve_k_cut(geo, 1)
        gaussian_sigma = threshold
        for label, func in (
            ("uniform", lambda kv: 0.6),
            (
                "two-zone",
                lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, 0.3, 0.95),
            ),
            (
                "gaussian",
                lambda kv: (
                    0.95
                    - 0.35 * np.exp(-np.sum(kv**2, axis=0) / (2 * gaussian_sigma**2))
                ),
            ),
            ("k-squared ramp", lambda kv: np.sum(kv**2, axis=0) / (4 * math.pi**2)),
        ):
            with self.subTest(label=label):
                c = get_c_field(geo, func)
                self.assertTrue(cgfhmc.generate_c_field.last_report["is_even"])
                partner = c[(-np.arange(4)) % 4]
                np.testing.assert_allclose(c, partner, rtol=0.0, atol=1e-15)

    def test_odd_callable_is_symmetrized(self):
        geo = get_geometry((4, 4, 4, 4))
        c = get_c_field(geo, lambda kv: 0.5 + 0.2 * kv[0])
        np.testing.assert_allclose(c, c[(-np.arange(4)) % 4], rtol=0.0, atol=0.0)
        self.assertGreater(
            cgfhmc.generate_c_field.last_report["max_asymmetry_removed"], 0.0
        )

    def test_out_of_range_callable_is_clipped(self):
        geo = get_geometry()
        c = get_c_field(geo, lambda kv: -0.5 + 3.0 * np.sum(kv**2, axis=0))
        self.assertGreaterEqual(float(c.min()), 0.0)
        self.assertLessEqual(float(c.max()), 1.0)
        report = cgfhmc.generate_c_field.last_report
        self.assertGreater(report["n_clipped_low"] + report["n_clipped_high"], 0)

    def test_nan_in_the_callable_raises(self):
        geo = get_geometry()
        with self.assertRaises(ValueError):
            get_c_field(geo, lambda kv: np.where(kv[0] > 0.0, np.nan, 0.5))

    def test_broadcast_shapes_are_accepted(self):
        geo = get_geometry()
        # The subtest label must be a plain value: pytest-xdist cannot serialise
        # a lambda as a subtest parameter.
        for label, func in (
            ("scalar", lambda kv: 0.5),
            ("one element", lambda kv: np.array([0.5])),
            ("unit shape", lambda kv: np.full((1, 1, 1, 1), 0.5)),
            ("full shape", lambda kv: np.full(TOTAL_SITE, 0.5)),
        ):
            with self.subTest(label=label):
                np.testing.assert_array_equal(
                    get_c_field(geo, func), np.full(TOTAL_SITE, 0.5)
                )

    def test_non_broadcastable_output_is_rejected(self):
        geo = get_geometry()
        with self.assertRaises(ValueError):
            get_c_field(geo, lambda kv: np.zeros((3, 3, 3, 3)))

    def test_complex_output_is_rejected(self):
        geo = get_geometry()
        with self.assertRaises(ValueError):
            get_c_field(geo, lambda kv: np.full(TOTAL_SITE, 0.5 + 0.1j))

    def test_non_callable_is_rejected(self):
        geo = get_geometry()
        with self.assertRaises(TypeError):
            get_c_field(geo, np.full(TOTAL_SITE, 0.5))

    def test_geometry_is_required(self):
        # A bare shape or a tuple cannot describe the mode grid, so it is
        # refused rather than silently accepted and misapplied later.
        for bad in (TOTAL_SITE, list(TOTAL_SITE), None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    cgfhmc.generate_c_field(bad, lambda kv: 0.5)

### -------------------------------------------------------------------

class TestCFieldFromFile(unittest.TestCase):
    """``c_field_from_file`` round-trips a regularized field."""

    def test_round_trip(self):
        geo = get_geometry((2, 3, 4, 2))
        c = get_two_zone_c(geo)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "c_field.npy")
            np.save(path, c)
            loaded = cgfhmc.c_field_from_file(path)
        np.testing.assert_allclose(loaded, c, rtol=0.0, atol=EPS_FIELD)
        self.assertTrue(cgfhmc.regularize_c_field.last_report["is_even"])

    def test_loaded_field_is_regularized(self):
        # A stored field that violates the invariants is repaired on load, which
        # is the documented behaviour: the consumer refuses a bad field, so the
        # loader must produce a good one.
        raw = np.full((4, 4, 4, 4), 2.0)
        raw[0, 0, 0, 0] = -1.0
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "raw.npy")
            np.save(path, raw)
            loaded = cgfhmc.c_field_from_file(path)
        self.assertGreaterEqual(float(loaded.min()), 0.0)
        self.assertLessEqual(float(loaded.max()), 1.0)

### -------------------------------------------------------------------

class TestPartialRefreshMomentumMode(unittest.TestCase):
    """The mode-dependent refresh: the no-op, the resample, and the ensemble."""

    def test_frozen_field_is_a_strict_no_op(self):
        # The strongest contract of the module: c == 0 draws NO random number,
        # so a frozen mode is free and cannot disturb any stream.
        geo = get_geometry()
        gm = get_random_momentum(geo)
        before = np.asarray(gm).copy()
        frozen = get_uniform_c(geo, 0.0)
        rs_probe = qr.RngState("frozen-probe")
        rs_reference = qr.RngState("frozen-probe")
        applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(gm, rs_probe, frozen)
        self.assertFalse(applied)
        self.assertEqual(d_k, 0.0)
        self.assertTrue(info["no_op"])
        self.assertTrue(info["is_uniform"])
        self.assertTrue(np.array_equal(np.asarray(gm), before))
        self.assertEqual(rs_probe.get_state(), rs_reference.get_state())
        # qlat names the route even for the no-op, so a logger needs no branch.
        self.assertEqual(info["path"], "no-op (c == 0)")

    def test_complete_resample_matches_a_fresh_draw(self):
        geo = get_geometry()
        gm = get_random_momentum(geo, "start")
        seed = "resample"
        applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(
            gm, qr.RngState(seed), get_uniform_c(geo, 1.0)
        )
        self.assertTrue(applied)
        expected = get_random_momentum(geo, seed)
        np.testing.assert_allclose(
            np.asarray(gm), np.asarray(expected), rtol=0.0, atol=EPS_HERM
        )

    def test_uniform_path_equals_the_explicit_law(self):
        # The fast path is a *shortcut*, not an approximation: for a constant c
        # the transform commutes with the mixing, so it must reproduce
        # ``P <- sqrt(1-c^2) P + c xi`` exactly.
        geo = get_geometry()
        for c in (0.15, 0.5, 0.85):
            with self.subTest(c=c):
                gm = get_random_momentum(geo, "start")
                gm_before = np.asarray(gm).copy()
                seed = f"uniform-{c}"
                applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(
                    gm, qr.RngState(seed), get_uniform_c(geo, c)
                )
                self.assertTrue(applied)
                self.assertEqual(info["path"], "real-space (uniform c)")
                noise = np.asarray(get_random_momentum(geo, seed))
                keep = math.sqrt(1.0 - c * c)
                expected = keep * gm_before + c * noise
                np.testing.assert_allclose(
                    np.asarray(gm), expected, rtol=0.0, atol=EPS_FIELD
                )

    def test_mode_dependent_refresh_takes_the_fourier_path(self):
        geo = get_geometry()
        gm = get_random_momentum(geo)
        applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(
            gm, qr.RngState("fourier"), get_two_zone_c(geo)
        )
        self.assertTrue(applied)
        self.assertEqual(info["path"], "fourier (mode-dependent c)")
        # One bit of information the Fourier path must report: not every mode is
        # refreshed identically.
        self.assertGreater(info["c_max"], info["c_min"])
        self.assertEqual(
            info["n_modes_refreshed"] + info["n_modes_frozen"], info["n_modes"]
        )

    def test_partially_frozen_field_refreshes_only_the_rest(self):
        # (2, 2, 2, 4): the band split needs a threshold that separates modes,
        # which on 2^4 gives 7 frozen / 25 refreshed elements for 1/16 of the
        # transform cost.  The 4^4 version lives in tests/test_cgfhmc_slow.py.
        geo = get_geometry((2, 2, 2, 4))
        threshold = cgfhmc.resolve_k_cut(geo, 1)
        # Freeze the low modes entirely; the high ones are fully resampled.
        c = get_c_field(
            geo, lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, 0.0, 1.0)
        )
        info_expected_frozen = int(np.count_nonzero(c == 0.0))
        info_expected_refreshed = int(np.count_nonzero(c == 1.0))
        self.assertGreater(info_expected_frozen, 0)
        self.assertGreater(info_expected_refreshed, 0)
        gm = get_random_momentum(geo)
        applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(
            gm, qr.RngState("bands"), c
        )
        self.assertTrue(applied)
        self.assertEqual(info["n_modes_frozen"], info_expected_frozen)
        self.assertEqual(info["n_modes_refreshed"], info_expected_refreshed)

    def test_momentum_stays_anti_hermitian_and_traceless(self):
        # The physics reason c must be real, even and in [0, 1]: anything else
        # would rotate the momentum out of the su(3) Lie algebra.
        geo = get_geometry((2, 3, 4, 2))
        for label, c in (
            ("uniform 0.3", get_uniform_c(geo, 0.3)),
            ("uniform 0.9", get_uniform_c(geo, 0.9)),
            ("two-zone", get_two_zone_c(geo)),
        ):
            with self.subTest(label=label):
                gm = get_random_momentum(geo)
                cgfhmc.partial_refresh_momentum_mode(gm, qr.RngState(label), c)
                self.assertLess(get_anticorrelation_residual(gm), EPS_HERM)
                self.assertLess(get_traceless_residual(gm), EPS_HERM)

    def test_dk_is_the_kinetic_energy_change(self):
        from qlat_ref.hmc import gm_hamilton_node
        #
        geo = get_geometry()
        gm = get_random_momentum(geo)
        k_before = float(gm_hamilton_node(gm))
        applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(
            gm, qr.RngState("dk"), get_uniform_c(geo, 0.5)
        )
        self.assertTrue(applied)
        k_after = float(gm_hamilton_node(gm))
        self.assertAlmostEqual(d_k, k_after - k_before, delta=EPS_DK)

    def test_diagnostics_are_reported(self):
        geo = get_geometry()
        c = get_two_zone_c(geo)
        gm = get_random_momentum(geo)
        applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(
            gm, qr.RngState("info"), c
        )
        expected_keys = {
            "c_min",
            "c_max",
            "c_mean",
            "n_modes",
            "n_modes_refreshed",
            "n_modes_frozen",
            "is_uniform",
            "is_even",
            "max_asymmetry",
            "n_site",
            "no_op",
            "path",
        }
        self.assertEqual(set(info), expected_keys)
        self.assertEqual(info["n_modes"], c.size)
        self.assertEqual(info["n_site"], c.size)
        self.assertTrue(info["is_even"])
        self.assertFalse(info["is_uniform"])
        self.assertEqual(info["max_asymmetry"], 0.0)
        self.assertAlmostEqual(info["c_min"], float(c.min()), places=15)
        self.assertAlmostEqual(info["c_max"], float(c.max()), places=15)
        self.assertAlmostEqual(info["c_mean"], float(c.mean()), places=15)

    def test_out_of_range_field_is_refused(self):
        geo = get_geometry()
        gm = get_random_momentum(geo)
        bad = np.full(TOTAL_SITE, 0.5)
        bad[0, 0, 0, 0] = 1.5
        with self.assertRaises(ValueError):
            cgfhmc.partial_refresh_momentum_mode(gm, qr.RngState("bad"), bad)
        bad[0, 0, 0, 0] = -0.5
        with self.assertRaises(ValueError):
            cgfhmc.partial_refresh_momentum_mode(gm, qr.RngState("bad"), bad)

    def test_odd_field_is_refused(self):
        geo = get_geometry((4, 2, 2, 2))
        gm = get_random_momentum(geo)
        odd = np.where(cgfhmc.k_vectors(geo)[0] >= 0.0, 0.2, 0.8)
        with self.assertRaises(ValueError):
            cgfhmc.partial_refresh_momentum_mode(gm, qr.RngState("odd"), odd)

    def test_non_finite_field_is_refused(self):
        geo = get_geometry()
        gm = get_random_momentum(geo)
        bad = np.full(TOTAL_SITE, 0.5)
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            cgfhmc.partial_refresh_momentum_mode(gm, qr.RngState("nan"), bad)

    def test_wrong_size_field_is_refused(self):
        geo = get_geometry()
        gm = get_random_momentum(geo)
        with self.assertRaises(ValueError):
            cgfhmc.partial_refresh_momentum_mode(
                gm, qr.RngState("size"), np.full((2, 2, 2), 0.5)
            )

    def test_variance_is_preserved_mode_by_mode(self):
        # The stationarity claim of the module: refreshing with any c keeps the
        # Gaussian ensemble.  Checked on the total kinetic energy, which is
        # 1/4 of the sum of the squared components and therefore additive.
        from qlat_ref.hmc import gm_hamilton_node
        #
        # 2^4 with 16 draws: the claim is statistical in the number of
        # *samples*, not in the volume.  Measured 0.25% relative deviation here
        # against a broken-rescaling signature of 41% (sqrt(2)) or 100%, and one
        # draw costs 0.06 s here versus 0.98 s on 4^4.
        geo = get_geometry()
        c = get_two_zone_c(geo)
        energies_before = []
        energies_after = []
        for i in range(16):
            gm = get_random_momentum(geo, f"variance-{i}")
            energies_before.append(float(gm_hamilton_node(gm)))
            cgfhmc.partial_refresh_momentum_mode(gm, qr.RngState(f"mix-{i}"), c)
            energies_after.append(float(gm_hamilton_node(gm)))
        #
        mean_before = float(np.mean(energies_before))
        mean_after = float(np.mean(energies_after))
        # Relative band of 20% on 16 draws.  The measured standard error of the
        # mean kinetic energy is ~1% of its value, so this cannot fail by chance,
        # and a broken rescaling (which multiplies by sqrt(2), 2 or 1/2) is far
        # outside it.
        self.assertLess(abs(mean_after - mean_before) / mean_before, 0.2)

### -------------------------------------------------------------------

class TestCgfhmcDriverVectors(unittest.TestCase):
    """Parity with the committed golden trajectories.\n
    ``tests/data/cgfhmc-vectors.json`` holds short CGFHMC chains written by the
    experimental implementation this generator was promoted from, recorded on a
    live ``qlat`` build.  Those vectors pin the whole trajectory -- RNG tree, momentum
    draw, force-gradient integrator, per-step gauge fixing, partial refresh,
    ``dK`` accounting and Metropolis step -- so a change anywhere in the chain
    shows up here.  Tolerances are ABSOLUTE: every quantity below is a
    difference of large numbers, so a relative epsilon would be both weaker than
    intended and machine-sensitive.
    """

    def get_vectors(self):
        """Return the committed golden chains (helper)."""
        import json
        #
        with open(
            os.path.join(DATA_DIR, "cgfhmc-vectors.json"), "r", encoding="utf-8"
        ) as fp:
            return json.load(fp)

    def get_case_c_field(self, geo, case):
        """Return the ``c(k)`` field a golden case was generated with (helper)."""
        kind = case["c_field_kind"]
        if kind == "uniform":
            return get_uniform_c(geo, case["c_uniform"])
        if kind == "twozone":
            threshold = cgfhmc.resolve_k_cut(geo, 1)
            return get_c_field(
                geo,
                lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, 0.35, 0.9),
            )
        raise AssertionError(f"unknown c_field_kind {kind!r}")

    def test_chains_match_the_golden_vectors(self):
        vectors = self.get_vectors()
        self.assertEqual(
            sorted(vectors["chain_cases"]),
            ["beta2p13_rect_uniform", "beta5p5_twozone", "beta5p5_uniform"],
        )
        for name, case in sorted(vectors["chain_cases"].items()):
            with self.subTest(case=name):
                geo = get_geometry(tuple(case["total_site"]))
                gf = qr.GaugeField(geo)
                gf.set_unit()
                c = self.get_case_c_field(geo, case)
                rs = qr.RngState(case["seed"])
                for expected in case["trajectories"]:
                    info_out = {}
                    flag, delta_h = cgfhmc.run_hmc_cgfhmc_pure_gauge(
                        gf,
                        qr.GaugeAction(case["beta"], case["c1"]),
                        expected["traj"],
                        rs.split("run_hmc_cgfhmc_pure_gauge"),
                        c_field=c,
                        n_step=case["n_step"],
                        md_time=case["md_time"],
                        info_out=info_out,
                    )
                    self.assertEqual(int(flag), expected["flag"])
                    # Measured agreement with the reference is <= 1.8e-13 for
                    # delta_h and <= 1.2e-16 for plaq; one order above that
                    # leaves room for a different libm.
                    self.assertAlmostEqual(delta_h, expected["delta_h"], delta=1e-12)
                    self.assertAlmostEqual(
                        info_out["delta_h_accept"],
                        expected["delta_h_accept"],
                        delta=1e-12,
                    )
                    self.assertAlmostEqual(
                        info_out["dK_sum"], expected["dK_sum"], delta=1e-10
                    )
                    self.assertEqual(
                        info_out["n_refresh_applied"], expected["n_refresh_applied"]
                    )
                    self.assertEqual(
                        info_out["fix_n_iter_total"], expected["fix_n_iter_total"]
                    )
                    self.assertEqual(
                        info_out["fix_n_iter_max"], expected["fix_n_iter_max"]
                    )
                    self.assertAlmostEqual(gf.plaq(), expected["plaq"], delta=1e-12)
                    # The field AFTER the trajectory: the accepted proposal, or
                    # the previous field when the Metropolis test rejected.
                    # This is the assertion that pins the INSERTION POINT of the
                    # per-step fixing + refresh -- a driver that hoists one
                    # integrator sub-update out of the step loop matches the
                    # scalars of a single step and diverges from the second one
                    # on, which no delta_h-only check at n_step=1 would catch.
                    expected_field = np.array(
                        [
                            [complex(re, im) for re, im in row]
                            for row in expected["field"]
                        ]
                    ).reshape(-1, qr.NC * qr.NC)
                    got_field = np.asarray(gf).reshape(-1, qr.NC * qr.NC)
                    self.assertLess(
                        float(np.max(np.abs(got_field - expected_field))),
                        EPS_FIELD,
                        f"{name} traj {expected['traj']}: field differs",
                    )

    def test_the_insertion_point_is_the_step_boundary(self):
        # The per-step fixing + refresh must sit exactly at the step boundary, so
        # that the drift accumulated when they run is (i+1)*dt and every MD step
        # is the self-contained ``lam, (1 - 2*lam), lam`` sandwich of
        # ``run_hmc_evolve_pure_gauge``.  Hoisting one ``lam*dt`` drift out of the
        # loop -- or shrinking the last step's trailing drift to compensate -- is
        # invisible to a total-drift check, but changes the trajectory from step
        # 2 on (the two schedules agreed at n_step=1 only).  So replay the
        # documented schedule by hand on the reference integrator and require the
        # driver to land on the same (gf, gm) for n_step = 2 and 3.
        #
        # ``c = 0`` makes the refresh a strict no-op that draws no random number,
        # so this isolates the insertion point of the FIXING; the full
        # refresh-carrying trajectory is pinned by the golden vectors above.
        geo = get_geometry(TOTAL_SITE)
        ga = qr.GaugeAction(5.5, 0.0)
        lam = 0.5 * (1.0 - 1.0 / math.sqrt(3.0))
        theta = (2.0 - math.sqrt(3.0)) / 48.0
        seed = "insertion-point"
        for n_step, md_time in ((2, 0.25), (3, 0.375)):
            with self.subTest(n_step=n_step):
                dt = md_time / n_step
                ttheta = theta * dt * dt * dt
                gf_init = qr.GaugeField(geo)
                gf_init.set_rand(qr.RngState(seed), 0.3, 2)
                gf_init.unitarize()
                # The driver's RNG tree, so the hand replay gets the same draw.
                gm = qr.GaugeMomentum(geo)
                gm.set_rand(
                    qr.RngState(seed).split("1").split("set_rand_gauge_momentum"), 1.0
                )
                gf_hand = qr.GaugeField(geo)
                gf_hand @= gf_init
                gm_hand = qr.GaugeMomentum(geo)
                gm_hand @= gm
                for _ in range(n_step):
                    gf_evolve(gf_hand, gm_hand, lam * dt)
                    gm_evolve_fg_pure_gauge(
                        gm_hand, gf_hand, ga, 4.0 * ttheta / dt, 0.5 * dt
                    )
                    gf_evolve(gf_hand, gm_hand, (1.0 - 2.0 * lam) * dt)
                    gm_evolve_fg_pure_gauge(
                        gm_hand, gf_hand, ga, 4.0 * ttheta / dt, 0.5 * dt
                    )
                    gf_evolve(gf_hand, gm_hand, lam * dt)
                    gt, _ = fix_landau_gauge(
                        gf_hand, max_iter=25, eps=1.0e-10, alpha=0.1
                    )
                    gf_hand @= gt * gf_hand
                    gm_hand @= gt * gm_hand
                gf_hand.unitarize()
                gf_got = qr.GaugeField(geo)
                gf_got @= gf_init
                cgfhmc.run_hmc_cgfhmc_pure_gauge(
                    gf_got,
                    ga,
                    1,
                    qr.RngState(seed),
                    c_field=get_uniform_c(geo, 0.0),
                    n_step=n_step,
                    md_time=md_time,
                    is_always_accept=True,
                )
                self.assertLess(
                    float(np.max(np.abs(np.asarray(gf_got) - np.asarray(gf_hand)))),
                    EPS_FIELD,
                )

    def test_a_field_dependent_of_the_c_field(self):
        # The vectors above use c = 0.5 and a two-zone band; a frozen field must
        # produce a *different* chain, otherwise the c_field argument is ignored.
        vectors = self.get_vectors()
        case = vectors["chain_cases"]["beta5p5_uniform"]
        geo = get_geometry(tuple(case["total_site"]))
        deltas = []
        for kind in ("uniform", "twozone", "frozen"):
            gf = qr.GaugeField(geo)
            gf.set_unit()
            if kind == "frozen":
                c = get_uniform_c(geo, 0.0)
            else:
                c = self.get_case_c_field(geo, {"c_field_kind": kind, "c_uniform": 0.5})
            info_out = {}
            cgfhmc.run_hmc_cgfhmc_pure_gauge(
                gf,
                qr.GaugeAction(case["beta"], case["c1"]),
                1,
                qr.RngState(case["seed"]).split("run_hmc_cgfhmc_pure_gauge"),
                c_field=c,
                n_step=case["n_step"],
                md_time=case["md_time"],
                info_out=info_out,
            )
            deltas.append(info_out["delta_h_accept"])
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                self.assertGreater(abs(deltas[i] - deltas[j]), 1e-6)

class TestCgfhmcDriver(unittest.TestCase):
    """``run_hmc_cgfhmc_pure_gauge``: the trajectory and its accounting."""

    def test_is_reverse_test_is_refused(self):
        # The gauge-fixing projection makes the trajectory non-reversible, so the
        # keyword exists only for signature parity with run_hmc_pure_gauge.
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_unit()
        with self.assertRaises(NotImplementedError) as context:
            cgfhmc.run_hmc_cgfhmc_pure_gauge(
                gf,
                qr.GaugeAction(5.5, 0.0),
                1,
                qr.RngState("reverse"),
                c_field=get_uniform_c(geo, 0.5),
                is_reverse_test=True,
            )
        message = str(context.exception)
        self.assertIn("not time-reversible", message)
        self.assertIn("is_reverse_test is not supported", message)

    def test_short_chain_stays_healthy(self):
        # Two trajectories of two steps per action: enough for "the chain runs,
        # stays unitary and returns a finite violation".  The physics of *long*
        # chains (drift, plaq band) is in tests/test_cgfhmc_slow.py, which is
        # opt-in -- it is 40 trajectories and belongs there, not here.
        geo = get_geometry()
        c = get_two_zone_c(geo)
        for beta, c1 in CHAIN_ACTIONS:
            with self.subTest(beta=beta, c1=c1):
                gf = qr.GaugeField(geo)
                gf.set_unit()
                ga = qr.GaugeAction(beta, c1)
                rs = qr.RngState("cgfhmc-short-chain")
                info_out = {}
                flags = []
                for traj in range(1, 3):
                    flag, delta_h = cgfhmc.run_hmc_cgfhmc_pure_gauge(
                        gf,
                        ga,
                        traj,
                        rs.split("run_hmc_cgfhmc_pure_gauge"),
                        c_field=c,
                        n_step=2,
                        md_time=1.0,
                        info_out=info_out,
                    )
                    flags.append(flag)
                    self.assertIsInstance(flag, bool)
                    self.assertTrue(math.isfinite(delta_h))
                self.assertTrue(all(isinstance(f, bool) for f in flags))
                plaq = gf.plaq()
                self.assertGreater(plaq, 0.0)
                self.assertLessEqual(plaq, 1.0 + 1e-12)
                self.assertLess(float(qr.get_unitarity_error(gf)), 1e-12)

    def test_constant_kinetic_frozen_field_accounting(self):
        # With c == 0 the refresh is a no-op, so the accept variable must be the
        # raw violation: the two must agree exactly.
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_unit()
        info_out = {}
        cgfhmc.run_hmc_cgfhmc_pure_gauge(
            gf,
            qr.GaugeAction(5.5, 0.0),
            1,
            qr.RngState("frozen-accounting"),
            c_field=get_uniform_c(geo, 0.0),
            n_step=4,
            md_time=1.0,
            info_out=info_out,
        )
        self.assertEqual(info_out["dK_sum"], 0.0)
        self.assertEqual(info_out["n_refresh_applied"], 0)
        self.assertEqual(info_out["delta_h_accept"], info_out["delta_h_raw"])

    def test_dk_accounting(self):
        # With c > 0 the refresh changes the kinetic energy, and the Metropolis
        # test must receive delta_h - dK_sum, not delta_h.
        geo = get_geometry()
        for label, c in (
            ("uniform 0.5", get_uniform_c(geo, 0.5)),
            ("two-zone", get_two_zone_c(geo)),
        ):
            with self.subTest(label=label):
                gf = qr.GaugeField(geo)
                gf.set_unit()
                info_out = {}
                flag, delta_h = cgfhmc.run_hmc_cgfhmc_pure_gauge(
                    gf,
                    qr.GaugeAction(5.5, 0.0),
                    1,
                    qr.RngState(f"dk-accounting-{label}"),
                    c_field=c,
                    n_step=4,
                    md_time=1.0,
                    info_out=info_out,
                )
                self.assertTrue(info_out["n_refresh_applied"] > 0)
                self.assertAlmostEqual(
                    info_out["delta_h_accept"],
                    info_out["delta_h_raw"] - info_out["dK_sum"],
                    delta=EPS_DK,
                )
                # The returned value is the RAW one, by design, so it must equal
                # delta_h_raw and (in general) differ from delta_h_accept.
                self.assertAlmostEqual(delta_h, info_out["delta_h_raw"], delta=EPS_DK)
                self.assertNotAlmostEqual(
                    info_out["delta_h_raw"], info_out["delta_h_accept"], places=6
                )

    def test_info_out_keys(self):
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_unit()
        info_out = {"sentinel": 1}
        cgfhmc.run_hmc_cgfhmc_pure_gauge(
            gf,
            qr.GaugeAction(5.5, 0.0),
            1,
            qr.RngState("info-keys"),
            c_field=get_uniform_c(geo, 0.5),
            n_step=2,
            md_time=1.0,
            info_out=info_out,
        )
        expected = {
            "dK_sum",
            "n_refresh_applied",
            "n_step",
            "n_fix",
            "fix_n_iter_total",
            "fix_n_iter_max",
            "fix_n_converged",
            "fix_divergence_max",
            "delta_h_raw",
            "delta_h_accept",
        }
        self.assertTrue(expected.issubset(set(info_out)))
        self.assertEqual(info_out["sentinel"], 1)
        self.assertEqual(info_out["n_step"], 2)
        self.assertEqual(info_out["n_fix"], 2)
        self.assertEqual(info_out["n_refresh_applied"], 2)
        self.assertEqual(info_out["fix_n_converged"], 0)
        self.assertGreater(info_out["fix_n_iter_total"], 0)
        self.assertLessEqual(info_out["fix_n_iter_max"], 25)
        self.assertGreaterEqual(info_out["fix_divergence_max"], 0.0)

    def test_info_out_is_optional(self):
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_unit()
        flag, delta_h = cgfhmc.run_hmc_cgfhmc_pure_gauge(
            gf,
            qr.GaugeAction(5.5, 0.0),
            1,
            qr.RngState("no-info"),
            c_field=get_uniform_c(geo, 0.5),
            n_step=2,
        )
        self.assertIsInstance(flag, bool)
        self.assertTrue(math.isfinite(delta_h))

    def test_determinism(self):
        geo = get_geometry()
        c = get_two_zone_c(geo)
        results = []
        for _ in range(2):
            gf = qr.GaugeField(geo)
            gf.set_unit()
            rs = qr.RngState("determinism")
            chain = []
            for traj in range(1, 3):
                flag, delta_h = cgfhmc.run_hmc_cgfhmc_pure_gauge(
                    gf,
                    qr.GaugeAction(5.5, 0.0),
                    traj,
                    rs.split("run_hmc_cgfhmc_pure_gauge"),
                    c_field=c,
                    n_step=2,
                    md_time=1.0,
                )
                chain.append((flag, delta_h))
            results.append((chain, np.array(gf)))
        self.assertEqual(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])

    def test_in_place_mutation_on_accept_only(self):
        geo = get_geometry()
        c = get_uniform_c(geo, 0.5)
        gf = qr.GaugeField(geo)
        gf.set_unit()
        view = np.asarray(gf)
        pre_trajectory = gf.copy()
        info_out = {}
        flag, delta_h = cgfhmc.run_hmc_cgfhmc_pure_gauge(
            gf,
            qr.GaugeAction(5.5, 0.0),
            1,
            qr.RngState("mutation"),
            c_field=c,
            n_step=2,
            md_time=1.0,
            is_always_accept=False,
            info_out=info_out,
        )
        if info_out["delta_h_accept"] <= 0.0:
            # Accepted: the caller's buffer follows the update (it is the same
            # object), while a copy taken before the call does not.
            self.assertGreater(
                float(np.max(np.abs(view - np.asarray(pre_trajectory)))), 0.0
            )
            np.testing.assert_array_equal(np.asarray(gf), view)
        else:
            np.testing.assert_array_equal(np.asarray(gf), np.asarray(pre_trajectory))

    def test_always_accept_updates_even_on_reject(self):
        geo = get_geometry()
        c = get_two_zone_c(geo)
        gf = qr.GaugeField(geo)
        gf.set_unit()
        pre_trajectory = gf.copy()
        info_out = {}
        cgfhmc.run_hmc_cgfhmc_pure_gauge(
            gf,
            qr.GaugeAction(5.5, 0.0),
            1,
            qr.RngState("always-accept"),
            c_field=c,
            n_step=2,
            md_time=1.0,
            is_always_accept=True,
            info_out=info_out,
        )
        self.assertGreater(
            float(np.max(np.abs(np.asarray(gf) - np.asarray(pre_trajectory)))), 0.0
        )

    def test_gauge_fixing_actually_runs(self):
        # The field must be drawn towards Landau gauge: the functional cannot
        # decrease, and the fixing must report a real iteration count.
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_rand(qr.RngState("not-fixed"), 0.4, 2)
        functional_before = float(qr.get_gauge_functional(gf, list(range(4))))
        info_out = {}
        cgfhmc.run_hmc_cgfhmc_pure_gauge(
            gf,
            qr.GaugeAction(5.5, 0.0),
            1,
            qr.RngState("fixing"),
            c_field=get_uniform_c(geo, 0.5),
            n_step=2,
            md_time=1.0,
            is_always_accept=True,
            gauge_fix_max_iter=25,
            info_out=info_out,
        )
        functional_after = float(qr.get_gauge_functional(gf, list(range(4))))
        self.assertEqual(info_out["n_fix"], 2)
        self.assertGreater(info_out["fix_n_iter_total"], 0)
        self.assertGreaterEqual(functional_after, functional_before - 1e-6)

    def test_gauge_fix_parameters_are_honoured(self):
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_unit()
        low = {}
        high = {}
        cgfhmc.run_hmc_cgfhmc_pure_gauge(
            gf,
            qr.GaugeAction(5.5, 0.0),
            1,
            qr.RngState("fix-budget"),
            c_field=get_uniform_c(geo, 0.5),
            n_step=2,
            gauge_fix_max_iter=3,
            info_out=low,
        )
        cgfhmc.run_hmc_cgfhmc_pure_gauge(
            gf,
            qr.GaugeAction(5.5, 0.0),
            1,
            qr.RngState("fix-budget"),
            c_field=get_uniform_c(geo, 0.5),
            n_step=2,
            gauge_fix_max_iter=25,
            info_out=high,
        )
        self.assertLessEqual(low["fix_n_iter_max"], 3)
        self.assertGreater(high["fix_n_iter_max"], low["fix_n_iter_max"])

    def test_signature_matches_run_hmc_pure_gauge(self):
        # The parity contract: the shared parameters must keep qlat's names,
        # order and defaults, and the CGFHMC ones must be keyword-only after them.
        import inspect
        #
        from qlat_ref.hmc_utils import run_hmc_pure_gauge
        #
        cgfhmc_params = list(
            inspect.signature(cgfhmc.run_hmc_cgfhmc_pure_gauge).parameters.values()
        )
        pure_params = list(inspect.signature(run_hmc_pure_gauge).parameters.values())
        self.assertEqual(
            [p.name for p in cgfhmc_params[:4]], [p.name for p in pure_params[:4]]
        )
        for ref_param in pure_params[4:]:
            twin = next(p for p in cgfhmc_params if p.name == ref_param.name)
            self.assertEqual(twin.kind, ref_param.kind)
            self.assertEqual(twin.default, ref_param.default)
        # Everything after the shared block is keyword-only and CGFHMC-specific.
        shared_names = {p.name for p in pure_params}
        extras = [p for p in cgfhmc_params if p.name not in shared_names]
        self.assertEqual(
            [p.name for p in extras],
            [
                "c_field",
                "gauge_fix_max_iter",
                "gauge_fix_eps",
                "gauge_fix_alpha",
                "info_out",
            ],
        )
        for param in extras:
            self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(extras[0].default, inspect.Parameter.empty)

### -------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
