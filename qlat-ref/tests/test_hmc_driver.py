"""
Tests of the HMC driver layer, :mod:`qlat_ref.hmc_utils`.\n
Three things are tested here, in this order of importance:\n
1. **Parity with the C++ trajectory.** ``tests/data/hmc-vectors.json`` holds the
   plaq, ``delta_h`` and full gauge field of the first three trajectories of a
   pure-gauge chain (unit start, ``n_step=6``, ``md_time=1``) at ``beta = 5.5``,
   ``beta = 1.0`` and the rectangle-improved ``(2.13, -0.331)``.  Those vectors
   pin the whole chain -- RNG tree, momentum, force, matrix exponential,
   integrator and Metropolis step -- against the real ``qlat`` build, so a
   change anywhere in the module shows up as a mismatch here.
2. **The mutation contract.** ``run_hmc_pure_gauge`` mutates ``gf`` in place on
   accept and leaves it bit-unchanged on reject; the trial field is internal, so
   an array *view* taken before the call observes the update while an array
   *copy* does not.
3. **The Metropolis step itself**, including its RNG stream: it must draw from
   ``rs.split(traj).split("metropolis_accept")`` and *not* from the momentum
   stream, and ``delta_h <= 0`` must accept without consuming a draw.\n
The chains here are deliberately short (the trimmed CI set): the long scans
that measure ``<plaq>`` are in :mod:`tests.test_hmc_slow`, behind
``QLAT_REF_SLOW=1``.
"""

import json
import math
import os
import unittest

import numpy as np

import qlat_ref as qr
from qlat_ref import hmc
from qlat_ref import hmc_utils
from qlat_ref.field import get_array_from_field

TOTAL_SITE = (2, 2, 2, 2)
N_STEP = 6
MD_TIME = 1.0

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Tolerances against the committed C++ trajectory vectors: measured
# |dplaq| <= 1.2e-14, |ddelta_h| <= 1.2e-13 and |dfield| <= 1.8e-15, i.e. a few
# ULP; three orders above that leaves room for a different libm without
# weakening the parity claim.
EPS_PLAQ = 1e-12
EPS_DELTA_H = 1e-9
EPS_FIELD = 1e-12

# beta / c1 of the short chains: the two endpoints of the coupling range and the
# rectangle-improved action (a separate, three-times slower force path).
CHAIN_ACTIONS = ((5.5, 0.0), (1.0, 0.0), (2.13, -0.331))

### -------------------------------------------------------------------

def load_vectors():
    """Return the committed C++ HMC vectors (helper)."""
    with open(os.path.join(DATA_DIR, "hmc-vectors.json"), "r", encoding="utf-8") as fp:
        return json.load(fp)

def get_geometry(total_site=TOTAL_SITE):
    """Return the test geometry (helper)."""
    return qr.Geometry(qr.Coordinate(list(total_site)))

def get_unit_field(total_site=TOTAL_SITE):
    """Return the unit gauge field (helper)."""
    gf = qr.GaugeField(get_geometry(total_site))
    gf.set_unit()
    return gf

def get_max_difference(a, b):
    """Return ``max |a - b|`` over two equally shaped arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

def get_accepted_seed():
    """Return the driver seed whose first trajectory is accepted (helper).\n
    With ``beta = 5.5`` from the unit field the first trajectory of this seed
    tree has ``delta_h = 0.171`` and is accepted (that is the committed
    ``beta_5p5_c1_0p0`` vector), so the accept-path assertions do not depend on
    a lucky draw: the value is in ``tests/data/hmc-vectors.json``.
    """
    return qr.RngState("qlat-ref-hmc-traj").split("run_hmc_pure_gauge")

def get_momentum_for(geo):
    """Return a non-trivial momentum on ``geo`` (helper)."""
    gm = qr.GaugeMomentum(geo)
    gm.set_rand(qr.RngState("copy-semantics"), 1.0)
    return gm

def run_chain(ga, n_traj, seed="hmc-driver", **kwargs):
    """Run ``n_traj`` trajectories from the unit field, returning the log (helper)."""
    gf = get_unit_field()
    rs = qr.RngState(seed)
    log = []
    for traj in range(1, n_traj + 1):
        flag, delta_h = hmc_utils.run_hmc_pure_gauge(
            gf, ga, traj, rs.split("run_hmc_pure_gauge"), **kwargs
        )
        log.append((traj, flag, delta_h, gf.plaq()))
    return gf, log

### -------------------------------------------------------------------

class TestTrajectoryVectors(unittest.TestCase):
    """The committed C++ trajectory: full parity of one, two, three trajectories."""

    @classmethod
    def setUpClass(cls):
        cls.vectors = load_vectors()["trajectory_cases"]

    def check_chain(self, key):
        """Re-run a recorded chain and compare every recorded trajectory."""
        case = self.vectors[key]
        ga = qr.GaugeAction(case["beta"], case["c1"])
        gf = get_unit_field()
        rs = qr.RngState(case["seed"])
        self.assertEqual(gf.plaq(), case["plaq_start"])
        self.assertEqual(hmc.gf_hamilton_node(gf, ga), case["gf_hamilton_start"])
        for expected in case["trajectories"]:
            flag, delta_h = hmc_utils.run_hmc_pure_gauge(
                gf,
                ga,
                expected["traj"],
                rs.split("run_hmc_pure_gauge"),
                n_step=case["n_step"],
                md_time=case["md_time"],
            )
            msg = f"{key} traj {expected['traj']}"
            self.assertEqual(int(flag), expected["flag"], msg=msg)
            self.assertLessEqual(
                abs(delta_h - expected["delta_h"]), EPS_DELTA_H, msg=msg
            )
            self.assertLessEqual(abs(gf.plaq() - expected["plaq"]), EPS_PLAQ, msg=msg)
            self.assertLessEqual(
                abs(hmc.gf_hamilton_node(gf, ga) - expected["gf_hamilton_node"]),
                EPS_FIELD * max(1.0, abs(expected["gf_hamilton_node"])),
                msg=msg,
            )
            # The vectors store qlat's own ``np.array(gf)`` layout
            # (local_volume, 4, NC, NC) with the flat site index on axis 0 --
            # NOT the (4, Nt, Nx, Ny, Nz, NC, NC) working layout, whose site
            # axes are Fortran-ordered.  Comparing the wrong one shows an O(1)
            # difference while plaq still matches, which is a misleading
            # failure mode; this comment is the map.
            field = np.asarray(expected["field_real"]) + 1j * np.asarray(
                expected["field_imag"]
            )
            self.assertLessEqual(
                get_max_difference(np.asarray(gf).reshape(-1), field),
                EPS_FIELD,
                msg=msg,
            )

    def test_wilson_beta_5p5(self):
        self.check_chain("beta_5p5_c1_0p0")

    def test_wilson_beta_1p0(self):
        self.check_chain("beta_1p0_c1_0p0")

    def test_rectangle_improved(self):
        # the c1 != 0 chain: a different force path, same integrator and driver
        self.check_chain("beta_2p13_c1_m0p331")

class TestMutationContract(unittest.TestCase):
    """``gf`` is mutated in place on accept and untouched on reject."""

    def test_returns_the_qlat_tuple_and_mutates_the_same_object(self):
        ga = qr.GaugeAction(5.5, 0.0)
        gf = get_unit_field()
        gf_id = id(gf)
        result = hmc_utils.run_hmc_pure_gauge(
            gf, ga, 1, get_accepted_seed(), n_step=N_STEP
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        flag, delta_h = result
        self.assertIsInstance(flag, bool)
        self.assertIsInstance(delta_h, float)
        self.assertTrue(flag)
        self.assertEqual(id(gf), gf_id)
        # the accepted trajectory changed the field's data in place
        self.assertNotEqual(gf.plaq(), 1.0)

    def test_rejected_trajectory_leaves_the_field_bit_unchanged(self):
        # A large md_time makes the energy violation huge, so the trajectory is
        # rejected; the internal trial copy means gf must not move at all.
        ga = qr.GaugeAction(5.5, 0.0)
        gf = get_unit_field()
        before = get_array_from_field(gf).copy()
        flag, delta_h = hmc_utils.run_hmc_pure_gauge(
            gf, ga, 1, qr.RngState("mutation-reject"), n_step=1, md_time=8.0
        )
        self.assertFalse(flag)
        self.assertGreater(delta_h, 1.0)
        self.assertEqual(get_max_difference(get_array_from_field(gf), before), 0.0)

    def test_a_view_sees_the_update_and_a_copy_does_not(self):
        # np.asarray(gf) is a live view of the field buffer, so it follows the
        # in-place update; np.array(gf) is a real copy and does not.  This is the
        # documented aliasing consequence of D2 (see TestArrayCopySemantics for
        # the copy semantics themselves).
        ga = qr.GaugeAction(5.5, 0.0)
        gf = get_unit_field()
        view = np.asarray(gf)
        copied = np.array(gf)
        field_copy = gf.copy()
        flag, _ = hmc_utils.run_hmc_pure_gauge(
            gf, ga, 1, get_accepted_seed(), n_step=N_STEP
        )
        self.assertTrue(flag)
        # the live view follows the field ...
        self.assertLessEqual(get_max_difference(view, np.asarray(gf)), 0.0)
        self.assertGreater(get_max_difference(view, copied), 1e-2)
        # ... while np.array(gf) and gf.copy() keep the pre-trajectory value
        self.assertLessEqual(get_max_difference(copied, np.eye(3)), 0.0)
        self.assertLessEqual(get_max_difference(np.asarray(field_copy), np.eye(3)), 0.0)
        self.assertGreater(get_max_difference(np.asarray(gf), copied), 1e-2)

    def test_is_always_accept_updates_a_rejected_trajectory(self):
        # is_always_accept forces the update while still reporting the real
        # (rejecting) Metropolis flag -- that combination is what the
        # always-accept chains rely on, so both halves are asserted.
        ga = qr.GaugeAction(5.5, 0.0)
        gf = get_unit_field()
        before = get_array_from_field(gf).copy()
        flag, delta_h = hmc_utils.run_hmc_pure_gauge(
            gf,
            ga,
            1,
            qr.RngState("mutation-always-accept"),
            n_step=1,
            md_time=8.0,
            is_always_accept=True,
        )
        self.assertFalse(flag)
        self.assertGreater(delta_h, 1.0)
        self.assertGreater(get_max_difference(get_array_from_field(gf), before), 1e-2)

    def test_is_always_accept_matches_a_normal_accept(self):
        # when the trajectory is accepted anyway, is_always_accept must not
        # change the result at all
        ga = qr.GaugeAction(5.5, 0.0)
        gf_a = get_unit_field()
        gf_b = get_unit_field()
        flag_a, dh_a = hmc_utils.run_hmc_pure_gauge(
            gf_a, ga, 1, get_accepted_seed(), n_step=N_STEP
        )
        flag_b, dh_b = hmc_utils.run_hmc_pure_gauge(
            gf_b,
            ga,
            1,
            get_accepted_seed(),
            n_step=N_STEP,
            is_always_accept=True,
        )
        self.assertTrue(flag_a)
        self.assertEqual(flag_a, flag_b)
        self.assertEqual(dh_a, dh_b)
        self.assertLessEqual(
            get_max_difference(get_array_from_field(gf_a), get_array_from_field(gf_b)),
            0.0,
        )

    def test_reverse_test_gate_passes(self):
        # qlat's is_reverse_test gate: run the trajectory, then run it backwards
        # and assert the field returns; it must not raise for a healthy chain.
        ga = qr.GaugeAction(5.5, 0.0)
        gf = get_unit_field()
        flag, delta_h = hmc_utils.run_hmc_pure_gauge(
            gf, ga, 1, qr.RngState("reverse-gate"), n_step=4, is_reverse_test=True
        )
        self.assertIsInstance(flag, bool)
        self.assertIsInstance(delta_h, float)

class TestArrayCopySemantics(unittest.TestCase):
    """
    ``np.asarray(f)`` is a view, ``np.array(f)`` is a copy -- for every field type.\n
    NumPy asks ``__array__`` for a copy with ``copy=True`` when ``np.array`` is
    used and for a view with ``copy=None`` when ``np.asarray`` is used; the field
    and matrix types honour both, matching live qlat's buffer behaviour.  This is
    what makes the D2 mutation contract sound: a snapshot taken with
    ``np.array(gf)`` before ``run_hmc_pure_gauge`` cannot be disturbed by the
    in-place update, while a view is expected to follow it.
    """

    def check_copy_semantics(self, f, buffer_name):
        """Assert view/copy semantics for one field or matrix object."""
        view = np.asarray(f)
        copied = np.array(f)
        self.assertIs(view, getattr(f, buffer_name))
        self.assertLessEqual(get_max_difference(view, copied), 0.0)
        self.assertFalse(
            np.shares_memory(copied, view), "np.array must not alias the buffer"
        )
        self.assertTrue(copied.flags["OWNDATA"], "np.array must own its data")

    def test_field_types(self):
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_unit()
        self.check_copy_semantics(gf, "_arr")
        gm = get_momentum_for(geo)
        self.check_copy_semantics(gm, "_arr")
        gt = qr.GaugeTransform(geo)
        gt.set_unit()
        self.check_copy_semantics(gt, "_arr")
        fcm = qr.FieldColorMatrix(geo, 4)
        fcm.set_unit()
        self.check_copy_semantics(fcm, "_arr")

    def test_color_matrix(self):
        m = qr.ColorMatrix()
        self.check_copy_semantics(m, "_x")

    def test_the_dtype_request_still_works(self):
        gf = get_unit_field()
        self.assertEqual(np.asarray(gf, dtype=np.complex64).dtype, np.complex64)
        self.assertEqual(np.array(gf, dtype=np.complex64).dtype, np.complex64)

    def test_np_array_snapshot_survives_an_in_place_update(self):
        # the D2 consequence: the pre-trajectory configuration survives the
        # driver's in-place accept when it was taken with np.array(gf)
        ga = qr.GaugeAction(5.5, 0.0)
        gf = get_unit_field()
        snapshot = np.array(gf)
        flag, _ = hmc_utils.run_hmc_pure_gauge(
            gf, ga, 1, get_accepted_seed(), n_step=N_STEP
        )
        self.assertTrue(flag)
        self.assertLessEqual(get_max_difference(snapshot, np.eye(3)), 0.0)
        self.assertGreater(get_max_difference(np.asarray(gf), snapshot), 1e-2)

class TestRngTree(unittest.TestCase):
    """The three streams of the driver are qlat's and they are independent."""

    def get_expected_flag(self, delta_h, rs_traj):
        """Return the flag ``metropolis_accept`` must produce (helper).\n
        Derived **independently of the code under test**: the raw 64-bit draw
        ``rand_gen()`` is taken from a fresh copy of the accept stream and
        mapped to a uniform number by hand, ``u = raw * 2**-64``.  This does not
        call ``u_rand_gen`` at all, so it cannot inherit a wrong ``(lower,
        upper)`` argument order from the implementation -- which is exactly the
        bug this check exists to catch.
        """
        if delta_h <= 0.0:
            return True
        accept_prob = min(1.0, math.exp(-delta_h))
        raw = rs_traj.split("metropolis_accept").rand_gen()
        u = float(raw) * 2.0**-64
        return u <= accept_prob

    def test_delta_h_comes_from_the_momentum_stream(self):
        # Reproduce the trajectory by hand from the documented seed tree: the
        # momentum must come from rs.split(traj).split("set_rand_gauge_momentum")
        # and nothing else.
        ga = qr.GaugeAction(5.5, 0.0)
        gf = get_unit_field()
        rs = get_accepted_seed()
        flag, delta_h = hmc_utils.run_hmc_pure_gauge(gf, ga, 1, rs, n_step=N_STEP)
        self.assertTrue(flag)
        # replay by hand
        rs_traj = rs.split("1")
        gm = qr.GaugeMomentum(gf.geo)
        gm.set_rand(rs_traj.split("set_rand_gauge_momentum"), 1.0)
        gf_trial = get_unit_field()
        delta_h_replay = hmc_utils.run_hmc_evolve_pure_gauge(gm, gf_trial, ga, N_STEP)
        self.assertEqual(delta_h, delta_h_replay)
        self.assertEqual(flag, self.get_expected_flag(delta_h, rs_traj))
        self.assertLessEqual(
            get_max_difference(
                get_array_from_field(gf), get_array_from_field(gf_trial)
            ),
            0.0,
        )

    def test_rejected_trajectory_does_not_disturb_later_momenta(self):
        # The momentum stream is split per trajectory from the driver seed, so
        # what happens in the Metropolis step (even a rejection) cannot shift
        # the next trajectory's momentum.  Two runs that differ only in whether
        # trajectory 1 is forced must agree from trajectory 2 on.
        ga = qr.GaugeAction(5.5, 0.0)
        rs_seed = "stream-independence"
        gf_a = get_unit_field()
        flag_a, dh_a = hmc_utils.run_hmc_pure_gauge(
            gf_a, ga, 1, qr.RngState(rs_seed), n_step=1, md_time=8.0
        )
        self.assertFalse(flag_a)
        # with the same seed, the momentum of trajectory 2 is identical...
        gm_2 = qr.GaugeMomentum(gf_a.geo)
        gm_2.set_rand(
            qr.RngState(rs_seed).split("2").split("set_rand_gauge_momentum"),
            1.0,
        )
        gm_2_other = qr.GaugeMomentum(gf_a.geo)
        gm_2_other.set_rand(
            qr.RngState("another-seed").split("2").split("set_rand_gauge_momentum"),
            1.0,
        )
        self.assertGreater(
            get_max_difference(
                get_array_from_field(gm_2), get_array_from_field(gm_2_other)
            ),
            1e-2,
        )
        self.assertNotEqual(dh_a, 0.0)

class TestMetropolisAccept(unittest.TestCase):
    """The accept/reject step in isolation."""

    @classmethod
    def setUpClass(cls):
        cls.vectors = load_vectors()

    def test_non_positive_delta_h_is_accepted_without_a_draw(self):
        # delta_h <= 0 must short-circuit *before* touching the RNG: the driver
        # splits the accept stream from the trajectory stream, and consuming a
        # draw here would shift the accepted-only stream.
        rs_probe = qr.RngState("metropolis")
        state_before = [rs_probe.rand_gen() for _ in range(4)]
        for delta_h in (0.0, -1e-9, -25.0):
            rs = qr.RngState("metropolis")
            flag, accept_prob = hmc_utils.metropolis_accept(
                delta_h, 1, rs.split("metropolis_accept")
            )
            self.assertTrue(flag)
            self.assertEqual(accept_prob, 1.0)
            # neither the split nor a non-positive delta_h advanced the state
            self.assertEqual([rs.rand_gen() for _ in range(4)], state_before)

    def test_accept_probability_is_exp_minus_delta_h(self):
        for delta_h in (1e-3, 0.5, 3.0):
            flag, accept_prob = hmc_utils.metropolis_accept(
                delta_h, 1, qr.RngState("metropolis")
            )
            self.assertLessEqual(abs(accept_prob - math.exp(-delta_h)), 1e-15)
            self.assertIsInstance(flag, bool)

    def test_decision_matches_the_recorded_qlat_vectors(self):
        # Golden vectors from live qlat: each entry records the raw 64-bit draw
        # that qlat consumed and the flag it produced.  This pins the argument
        # order of the u_rand_gen call, which no self-consistent test can do.
        for name, case in self.vectors["metropolis_decision_cases"].items():
            flag, accept_prob = hmc_utils.metropolis_accept(
                case["delta_h"], 1, qr.RngState(case["seed"])
            )
            self.assertEqual(int(flag), case["flag"], msg=name)
            self.assertLessEqual(
                abs(accept_prob - case["accept_prob"]), 1e-15, msg=name
            )
            # the recorded raw draw must re-derive the recorded flag by hand
            u = float(case["raw_draw"]) * 2.0**-64
            self.assertEqual(u, case["u"], msg=name)
            if case["delta_h"] <= 0.0:
                self.assertEqual(case["flag"], 1, msg=name)
            else:
                self.assertEqual(
                    case["flag"], int(u <= math.exp(-case["delta_h"])), msg=name
                )

    def test_conventions_disagree_so_the_check_has_teeth(self):
        # At accept_prob = 1/2 the wrong argument order (1.0, 0.0) draws 1 - u,
        # and u <= 0.5 <=> 1 - u >= 0.5, so it must disagree on essentially
        # every seed.  A guard that silently passed for both conventions would
        # be useless; this asserts the vector set is discriminating.
        cases = [
            case
            for case in self.vectors["metropolis_decision_cases"].values()
            if abs(math.exp(-case["delta_h"]) - 0.5) < 1e-9
        ]
        self.assertGreaterEqual(len(cases), 3)
        for case in cases:
            u = case["u"]
            self.assertNotEqual(
                u <= 0.5, (1.0 - u) <= 0.5, msg="the two orders must differ here"
            )
        # Count both conventions explicitly.  NOTE the parentheses: a bare
        # ``a <= b == c`` is a Python *chained* comparison and means
        # ``(a <= b) and (b == c)``, and ``float == bool`` is always False, so
        # it would make this whole backstop vacuously true (audit finding Y1).
        # ``int(...)`` makes each term an unambiguous 0/1.
        agree_right = 0
        agree_buggy = 0
        for case in cases:
            accept_prob = math.exp(-case["delta_h"])
            u = case["u"]
            flag = bool(case["flag"])
            agree_right += int((u <= accept_prob) == flag)
            agree_buggy += int(((1.0 - u) <= accept_prob) == flag)
        #
        # The vectors are recorded from the correct convention, so it must agree
        # with every one of them ...
        self.assertEqual(agree_right, len(cases))
        # ... and the reversed order must agree with none, which is what makes
        # the vector set (and this guard) able to catch the bug at all.
        self.assertEqual(agree_buggy, 0)

    def test_uses_a_uniform_draw_in_the_unit_interval(self):
        # The accept test must consume exactly one uniform in [0, 1), drawn from
        # the stream it is handed: map the raw draw by hand and compare.
        delta_h = 0.5
        accept_prob = math.exp(-delta_h)
        for seed in ("metro-a", "metro-b", "metro-c"):
            flag, _ = hmc_utils.metropolis_accept(delta_h, 7, qr.RngState(seed))
            raw = qr.RngState(seed).rand_gen()
            expected = float(raw) * 2.0**-64 <= accept_prob
            self.assertEqual(flag, expected, msg=seed)
        # a large delta_h always rejects, independent of the draw
        flag, _ = hmc_utils.metropolis_accept(1e6, 1, qr.RngState("metro"))
        self.assertFalse(flag)

    def test_delta_h_zero_is_always_accepted(self):
        self.assertTrue(hmc_utils.metropolis_accept(0.0, 1, qr.RngState("m"))[0])

class TestChainFlags(unittest.TestCase):
    """End-to-end accept/reject flags against live qlat.\n
    The single-trajectory chains above run at ``md_time = 1``, where essentially
    every trajectory is accepted -- a flipped Metropolis decision would barely
    show.  These chains deliberately use a coarser step so that both branches
    occur and the recorded flag sequence discriminates: a wrong ``u_rand_gen``
    argument order changes at least two flags per case (verified when the
    vectors were generated).
    """

    @classmethod
    def setUpClass(cls):
        cls.vectors = load_vectors()

    def check_chain(self, key):
        """Replay one recorded chain and compare every flag (helper)."""
        case = self.vectors["chain_flag_cases"][key]
        ga = qr.GaugeAction(case["beta"], case["c1"])
        gf = get_unit_field()
        rs = qr.RngState(case["seed"])
        flags = []
        for traj in range(1, case["n_traj"] + 1):
            flag, delta_h = hmc_utils.run_hmc_pure_gauge(
                gf,
                ga,
                traj,
                rs.split("run_hmc_pure_gauge"),
                n_step=case["n_step"],
                md_time=case["md_time"],
            )
            flags.append(int(flag))
            self.assertLessEqual(
                abs(delta_h - case["delta_h"][traj - 1]),
                EPS_DELTA_H,
                msg=f"{key} traj {traj}",
            )
        self.assertEqual(flags, case["flags"], msg=key)
        self.assertEqual(flags, case["flags_qlat_ref"], msg=key)
        self.assertGreater(case["n_rejected"], 0, msg=f"{key}: no rejections")
        # the chain ends where the C++ chain ended
        self.assertLessEqual(abs(gf.plaq() - case["plaq_end"]), EPS_PLAQ, msg=key)

    def test_beta_5p5_coarse_step(self):
        self.check_chain("beta_5p5_md_time_2p0")

    def test_beta_1p0_coarse_step(self):
        self.check_chain("beta_1p0_md_time_3p0")

    def test_rectangle_improved_coarse_step(self):
        self.check_chain("beta_2p13_md_time_1p5_rect")

    def test_the_recorded_chains_exercise_both_branches(self):
        # guard the vectors themselves: a chain that accepts (or rejects)
        # everything cannot detect a flipped decision
        for key, case in self.vectors["chain_flag_cases"].items():
            flags = case["flags"]
            self.assertIn(1, flags, msg=key)
            self.assertIn(0, flags, msg=key)

class TestShortChains(unittest.TestCase):
    """A short chain stays healthy: acceptance, field range and no drift."""

    def test_chains_stay_healthy(self):
        for beta, c1 in CHAIN_ACTIONS:
            ga = qr.GaugeAction(beta, c1)
            gf, log = run_chain(ga, 6, seed=f"chain-{beta}-{c1}", n_step=N_STEP)
            accepted = sum(1 for entry in log if entry[1])
            self.assertGreaterEqual(accepted, 4, msg=f"beta={beta} c1={c1}")
            for traj, flag, delta_h, plaq in log:
                self.assertGreater(plaq, 0.0, msg=f"beta={beta} traj={traj}")
                self.assertLessEqual(plaq, 1.0, msg=f"beta={beta} traj={traj}")
                self.assertFalse(math.isnan(delta_h), msg=f"beta={beta} traj={traj}")
            self.assertLessEqual(
                qr.mat.get_unitarity_error(get_array_from_field(gf)), 1e-12
            )

    def test_always_accept_chain_does_not_drift(self):
        # Without the Metropolis test the discretisation error accumulates; on
        # a short chain at beta=5.5 the plaquette must still be sane and the
        # field unitary, which is the cheap drift gate.
        ga = qr.GaugeAction(5.5, 0.0)
        gf, log = run_chain(
            ga, 8, seed="always-accept-chain", n_step=N_STEP, is_always_accept=True
        )
        mean_delta_h = float(np.mean([entry[2] for entry in log]))
        self.assertLess(abs(mean_delta_h), 1.0)
        self.assertGreater(gf.plaq(), 0.3)
        self.assertLessEqual(
            qr.mat.get_unitarity_error(get_array_from_field(gf)), 1e-12
        )

    def test_plaq_increases_with_beta(self):
        # Same chain length and start, different coupling: the order parameter
        # must be ordered, which is a physics sanity check on the action scale.
        plaquettes = {}
        for beta in (1.0, 5.5):
            gf, _ = run_chain(
                qr.GaugeAction(beta, 0.0), 5, seed="beta-ordering", n_step=N_STEP
            )
            plaquettes[beta] = gf.plaq()
        self.assertLess(plaquettes[1.0], plaquettes[5.5])

### -------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
