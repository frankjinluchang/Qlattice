"""
Optional parity tests of :mod:`qlat_ref` against a live ``qlat`` build.\n
These tests are **skipped** when ``qlat`` is not importable, so the pure-NumPy
package (and its nix derivation, which depends on ``numpy`` only) is unaffected.
They are the executable form of the parity claim: for every shared name the
signature and, where the operation is deterministic, the value must agree with
``qlat``.\n
Intentional deviations are listed in :data:`DEVIATIONS` with a rationale and are
checked as such -- never silently skipped.  Anything not listed there must
match.\n
Run with the qlat environment sourced::\n
    source result-py-local/bin/setenv-qlat.sh
    cd qlat-ref && python3 -m unittest tests.test_qlat_parity -v
"""

import inspect
import unittest

import numpy as np

import qlat_ref as qr
from qlat_ref import field as fld

# Every name the parity tests compare against.  A qlat build that lacks one of
# them (an older release, or a partially installed tree) cannot run these
# comparisons, so it must cause a *skip*, not an error.
_REQUIRED_QLAT_NAMES = (
    "GaugeField",
    "GaugeMomentum",
    "GaugeTransform",
    "Geometry",
    "GaugeAction",
    "RngState",
    "begin_with_mpi",
    "end_with_mpi",
    "fix_coulomb_gauge",
    "fix_gauge",
    "fix_landau_gauge",
    "get_gauge_divergence",
    "get_gauge_functional",
    "get_unitarity_error",
    "gf_hamilton_node",
)

try:
    import qlat as q

    _missing = [n for n in _REQUIRED_QLAT_NAMES if not hasattr(q, n)]
    HAS_QLAT = not _missing
    _NO_QLAT_REASON = (
        f"the installed qlat lacks {_missing}"
        if _missing
        else "a live qlat build is required for the parity tests"
    )
except Exception as exc:  # noqa: BLE001 - any import failure means "no qlat here"
    q = None
    HAS_QLAT = False
    _NO_QLAT_REASON = f"qlat is not importable ({exc})"

TOTAL_SITE = (2, 2, 2, 2)
SEED = "qlat-parity-seed"

DEVIATIONS = {
    "fix_gauge": "qlat_ref adds an extra 'dirs' key to info; every qlat key is present",
    "fix_coulomb_gauge": (
        "qlat_ref fixes the CORRECT spatial directions (0, 1, 2) instead of"
        " qlat's (1, 2, 3). Component 3 is time in qlat's own convention"
        " (total_site = (Lx, Ly, Lz, Lt); Geometry::spatial_volume divides out"
        " component 3; get_psel_tslice slices t_dir=3; qlat_gpt.gauge_fix_coulomb"
        " separates axis 3), so qlat's gf_fix_coulomb_gauge fixes the three"
        " directions that INCLUDE time and excludes spatial direction 0."
        " This is qlat-issue-8 in tmp/qlat-ref-reorg/qlat-issues.md: a qlat bug,"
        " decided by the user to be fixed in qlat_ref rather than reproduced."
        " The same algorithm is still bit-comparable via fix_gauge(dirs=[...])."
    ),
}

### -------------------------------------------------------------------

def _get_signature(fn):
    """Return the comparable ``(name, kind, default)`` view of a signature."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    return [(p.name, p.kind.name, p.default) for p in sig.parameters.values()]

def _get_ref_field(geo_ref, arr):
    """Return a qlat_ref GaugeField holding the ``(4, ...)`` array ``arr``."""
    return fld.get_field_from_array(arr, geo_ref)

_MPI_STARTED = False

def setUpModule():
    """Start MPI once for the whole module if a live qlat is available.\n
    ``q.begin_with_mpi`` may be called only once per process, so the per-class
    ``setUpClass`` hooks cannot own it.
    """
    global _MPI_STARTED
    if not HAS_QLAT:
        return
    q.begin_with_mpi([[1, 1, 1, 1]])
    _MPI_STARTED = True

def tearDownModule():
    """Close the MPI session opened by :func:`setUpModule`."""
    global _MPI_STARTED
    if _MPI_STARTED:
        q.end_with_mpi()
        _MPI_STARTED = False

### -------------------------------------------------------------------

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestSignatureParity(unittest.TestCase):
    """The shared names must expose qlat's exact positional signatures."""

    # (qlat name, qlat_ref callable) pairs sharing an interface.
    SHARED = [
        ("fix_gauge", "gauge_fixing"),
        ("fix_landau_gauge", "gauge_fixing"),
        ("fix_coulomb_gauge", "gauge_fixing"),
        ("get_gauge_functional", "observables"),
        ("get_gauge_divergence", "observables"),
    ]

    def test_shared_function_signatures(self):
        for name, module_name in self.SHARED:
            with self.subTest(name=name):
                q_sig = _get_signature(getattr(q, name))
                ref_mod = getattr(qr, module_name)
                ref_sig = _get_signature(getattr(ref_mod, name))
                self.assertIsNotNone(q_sig, f"qlat.{name} has no signature")
                self.assertIsNotNone(ref_sig, f"qlat_ref.{name} has no signature")
                self.assertEqual(
                    q_sig,
                    ref_sig,
                    f"{name}: qlat {q_sig} != qlat_ref {ref_sig}",
                )

    def test_unitarity_error_signature(self):
        # Same single positional argument on both sides; the accepted types are
        # the same too (GaugeField and GaugeTransform), so this is no longer a
        # deviation and must NOT be listed in DEVIATIONS.
        name = "get_unitarity_error"
        q_sig = _get_signature(getattr(q, name))
        ref_sig = _get_signature(qr.observables.get_unitarity_error_typed)
        self.assertEqual(q_sig, ref_sig)
        self.assertNotIn(name, DEVIATIONS)

    def test_gauge_action_methods_are_methods(self):
        # beta()/c1() must be METHODS on both sides, not properties.
        self.assertTrue(callable(q.GaugeAction(5.5, 0.1).beta))
        self.assertTrue(callable(qr.GaugeAction(5.5, 0.1).beta))
        self.assertEqual(
            _get_signature(q.GaugeAction.beta),
            _get_signature(qr.GaugeAction.beta),
        )
        self.assertEqual(
            _get_signature(q.GaugeAction.c1), _get_signature(qr.GaugeAction.c1)
        )

### -------------------------------------------------------------------

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestEmptyGeometryParity(unittest.TestCase):
    """
    The empty-but-safe ``Geometry()`` contract must be identical in both
    packages (qlat-issue-9).\n
    qlat guards its zero-extent divisions, so the bare ``Geometry()`` is an
    empty but safe object; ``qlat_ref`` must report exactly the same values and
    must not raise where qlat returns a value.
    """

    # Properties whose value must match qlat exactly for the empty geometry.
    SCALAR_PROPERTIES = [
        "num_node",
        "id_node",
        "eo",
        "is_only_local",
        "total_volume",
        "spatial_volume",
        "local_volume",
        "local_volume_expanded",
    ]
    COORDINATE_PROPERTIES = [
        "total_site",
        "local_site",
        "node_site",
        "size_node",
        "coor_node",
        "expansion_left",
        "expansion_right",
    ]

    def setUp(self):
        self.geo = q.Geometry()
        self.geo_ref = qr.Geometry()

    def test_scalar_properties_match(self):
        for name in self.SCALAR_PROPERTIES:
            with self.subTest(name=name):
                self.assertEqual(getattr(self.geo, name), getattr(self.geo_ref, name))

    def test_coordinate_properties_match(self):
        for name in self.COORDINATE_PROPERTIES:
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(self.geo, name).to_list(),
                    getattr(self.geo_ref, name).to_list(),
                )

    def test_volumes_are_zero_and_do_not_crash(self):
        # The regression itself: qlat used to die with SIGFPE here.
        values = [self.geo.spatial_volume, self.geo.total_volume]
        values += [self.geo.local_volume, self.geo.local_volume_expanded]
        self.assertEqual(values, [0, 0, 0, 0])
        self.assertEqual(
            [
                self.geo_ref.spatial_volume,
                self.geo_ref.total_volume,
                self.geo_ref.local_volume,
                self.geo_ref.local_volume_expanded,
            ],
            values,
        )

    def test_display_matches(self):
        self.assertEqual(repr(self.geo), repr(self.geo_ref))
        self.assertEqual(str(self.geo), str(self.geo_ref))

    def test_coordinate_index_maps_are_safe_in_both(self):
        x = q.Coordinate([1, 1, 1, 1])
        x_ref = qr.Coordinate([1, 1, 1, 1])
        self.assertEqual(
            self.geo.coordinate_from_index(0).to_list(),
            self.geo_ref.coordinate_from_index(0).to_list(),
        )
        self.assertEqual(
            self.geo.g_coordinate_from_index(0).to_list(),
            self.geo_ref.g_coordinate_from_index(0).to_list(),
        )
        self.assertEqual(
            self.geo.coordinate_l_from_g(x).to_list(),
            self.geo_ref.coordinate_l_from_g(x_ref).to_list(),
        )
        self.assertEqual(
            self.geo.coordinate_g_from_l(x).to_list(),
            self.geo_ref.coordinate_g_from_l(x_ref).to_list(),
        )
        self.assertEqual(self.geo.xg_arr().shape, self.geo_ref.xg_arr().shape)

### -------------------------------------------------------------------

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestFieldValueParity(unittest.TestCase):
    """Deterministic field operations must reproduce qlat bit-for-bit."""

    geo = None
    geo_ref = None

    @classmethod
    def setUpClass(cls):
        # MPI is started once by setUpModule(); this only builds the geometries.
        import qlat_utils as qu
        #
        cls.total_site = list(TOTAL_SITE)
        cls.geo = q.Geometry(qu.Coordinate(cls.total_site))
        cls.geo_ref = qr.Geometry(qr.Coordinate(cls.total_site))

    def get_fields(self, sigma=0.5, n_step=1):
        gf = q.GaugeField(self.geo)
        gf.set_rand(q.RngState(SEED), sigma, n_step)
        gf_ref = qr.GaugeField(self.geo_ref)
        gf_ref.set_rand(qr.RngState(SEED), sigma, n_step)
        return gf, gf_ref

    def test_set_rand(self):
        gf, gf_ref = self.get_fields()
        self.assertLess(
            float(np.max(np.abs(np.asarray(gf) - np.asarray(gf_ref)))), 1e-13
        )

    def test_set_rand_n_step(self):
        gf, gf_ref = self.get_fields(sigma=0.7, n_step=3)
        self.assertLess(
            float(np.max(np.abs(np.asarray(gf) - np.asarray(gf_ref)))), 1e-13
        )

    def test_unitarize(self):
        gf, gf_ref = self.get_fields(sigma=1.0)
        gf.unitarize()
        gf_ref.unitarize()
        self.assertLess(
            float(np.max(np.abs(np.asarray(gf) - np.asarray(gf_ref)))), 1e-13
        )

    def test_plaq_and_link_trace(self):
        gf, gf_ref = self.get_fields()
        self.assertAlmostEqual(gf.plaq(), gf_ref.plaq(), places=12)
        self.assertAlmostEqual(gf.link_trace(), gf_ref.link_trace(), places=12)

    def test_gauge_transform_set_rand(self):
        gt = q.GaugeTransform(self.geo)
        gt.set_rand(q.RngState(SEED), 0.5, 1)
        gt_ref = qr.GaugeTransform(self.geo_ref)
        gt_ref.set_rand(qr.RngState(SEED), 0.5, 1)
        self.assertLess(
            float(np.max(np.abs(np.asarray(gt) - np.asarray(gt_ref)))), 1e-13
        )

    def test_gt_times_gf(self):
        gf, gf_ref = self.get_fields()
        gt, gt_ref = self.get_fields()[0], None
        gt = q.GaugeTransform(self.geo)
        gt.set_rand(q.RngState(SEED + "-gt"), 0.4, 1)
        gt_ref = qr.GaugeTransform(self.geo_ref)
        gt_ref.set_rand(qr.RngState(SEED + "-gt"), 0.4, 1)
        self.assertLess(
            float(np.max(np.abs(np.asarray(gt * gf) - np.asarray(gt_ref * gf_ref)))),
            1e-13,
        )

    def test_gt_composition_and_inverse(self):
        gt1 = q.GaugeTransform(self.geo)
        gt1.set_rand(q.RngState(SEED + "-1"), 0.4, 1)
        gt2 = q.GaugeTransform(self.geo)
        gt2.set_rand(q.RngState(SEED + "-2"), 0.5, 1)
        gt1_ref = qr.GaugeTransform(self.geo_ref)
        gt1_ref.set_rand(qr.RngState(SEED + "-1"), 0.4, 1)
        gt2_ref = qr.GaugeTransform(self.geo_ref)
        gt2_ref.set_rand(qr.RngState(SEED + "-2"), 0.5, 1)
        self.assertLess(
            float(
                np.max(np.abs(np.asarray(gt1 * gt2) - np.asarray(gt1_ref * gt2_ref)))
            ),
            1e-13,
        )
        self.assertLess(
            float(np.max(np.abs(np.asarray(gt1.inv()) - np.asarray(gt1_ref.inv())))),
            1e-13,
        )

    def test_gauge_momentum(self):
        gm = q.GaugeMomentum(self.geo)
        gm.set_rand(q.RngState(SEED), 1.0)
        gm_ref = qr.GaugeMomentum(self.geo_ref)
        gm_ref.set_rand(qr.RngState(SEED), 1.0)
        self.assertLess(
            float(np.max(np.abs(np.asarray(gm) - np.asarray(gm_ref)))), 1e-13
        )
        gt = q.GaugeTransform(self.geo)
        gt.set_rand(q.RngState(SEED), 0.5, 1)
        gt_ref = qr.GaugeTransform(self.geo_ref)
        gt_ref.set_rand(qr.RngState(SEED), 0.5, 1)
        self.assertLess(
            float(np.max(np.abs(np.asarray(gt * gm) - np.asarray(gt_ref * gm_ref)))),
            1e-13,
        )

### -------------------------------------------------------------------

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestCgfhmcParity(unittest.TestCase):
    """The CGFHMC layer must be qlat's, value for value.\n
    Every comparison here is between two *independent* implementations of the
    same Fourier construction: qlat's distributed ``mk_fft`` and this package's
    ``numpy.fft``.  A wrong axis order, folding convention or FFT normalisation
    would show up immediately, so the values are compared numerically rather
    than only structurally.\n
    The golden vectors in ``tests/data/cgfhmc-vectors.json`` already pin the
    whole trajectory; what they cannot pin is the ``c(k)`` *grid*, which is why
    the helpers are compared directly as well.
    """

    geo = None
    geo_ref = None

    @classmethod
    def setUpClass(cls):
        import qlat_utils as qu
        #
        cls.total_site = list(TOTAL_SITE)
        cls.geo = q.Geometry(qu.Coordinate(cls.total_site))
        cls.geo_ref = qr.Geometry(qr.Coordinate(cls.total_site))

    def get_cgk(self, name):
        """Return the qlat and qlat_ref implementations of ``name``."""
        if not hasattr(q, name) or not hasattr(qr, name):
            self.skipTest(f"the installed qlat has no {name}: needs a rebuild")
        return getattr(q, name), getattr(qr, name)

    def test_k_vectors_match(self):
        q_fn, ref_fn = self.get_cgk("k_vectors")
        self.assertLess(
            float(np.max(np.abs(q_fn(self.geo) - ref_fn(self.geo_ref)))), 1e-15
        )

    def test_k_squared_match(self):
        q_fn, ref_fn = self.get_cgk("k_squared")
        self.assertLess(
            float(np.max(np.abs(q_fn(self.geo) - ref_fn(self.geo_ref)))), 1e-14
        )

    def test_resolve_k_cut_matches(self):
        q_fn, ref_fn = self.get_cgk("resolve_k_cut")
        for units in (0, 1, 2, 3.5):
            with self.subTest(units=units):
                self.assertEqual(
                    float(q_fn(self.geo, units)), float(ref_fn(self.geo_ref, units))
                )

    def test_regularize_c_field_matches(self):
        q_fn, ref_fn = self.get_cgk("regularize_c_field")
        rng = np.random.default_rng(12345)
        # In range, out of range and non-finite entries in one array: the clip,
        # the evenness projection and the NaN policy must all agree.
        raw = rng.uniform(-0.5, 1.5, size=TOTAL_SITE)
        raw[0, 0, 0, 0] = np.nan
        raw[1, 1, 1, 1] = 2.5
        raw[0, 1, 0, 1] = -0.25
        got = q_fn(raw.copy(), allow_nan="zero")
        expected = ref_fn(raw.copy(), allow_nan="zero")
        self.assertLess(float(np.max(np.abs(got - expected))), 1e-15)

    def test_generate_c_field_matches(self):
        q_fn, ref_fn = self.get_cgk("generate_c_field")
        q_cut, ref_cut = self.get_cgk("resolve_k_cut")
        threshold = float(q_cut(self.geo, 1))
        expected_threshold = float(ref_cut(self.geo_ref, 1))
        self.assertEqual(threshold, expected_threshold)
        #
        def func(kv):
            return np.where(np.sum(kv**2, axis=0) <= threshold**2, 0.3, 0.95)
        #
        got = q_fn(self.geo, func)
        expected = ref_fn(self.geo_ref, func)
        self.assertLess(float(np.max(np.abs(got - expected))), 1e-15)

    def test_partial_refresh_matches(self):
        q_refresh, ref_refresh = self.get_cgk("partial_refresh_momentum_mode")
        q_gen, ref_gen = self.get_cgk("generate_c_field")
        q_cut, ref_cut = self.get_cgk("resolve_k_cut")
        # Same c(k) on both sides, then the SAME momentum and the same RNG seed:
        # any difference is the transform, not the inputs.
        cut = float(q_cut(self.geo, 1))
        c_q = q_gen(
            self.geo, lambda kv: np.where(np.sum(kv**2, axis=0) <= cut**2, 0.4, 0.9)
        )
        c_ref = ref_gen(
            self.geo_ref, lambda kv: np.where(np.sum(kv**2, axis=0) <= cut**2, 0.4, 0.9)
        )
        gm = q.GaugeMomentum(self.geo)
        gm.set_rand(q.RngState(SEED + "-cgfhmc"), 1.0)
        gm_ref = qr.GaugeMomentum(self.geo_ref)
        gm_ref.set_rand(qr.RngState(SEED + "-cgfhmc"), 1.0)
        applied, dk, info = q_refresh(gm, q.RngState(SEED + "-refresh"), c_q)
        applied_ref, dk_ref, info_ref = ref_refresh(
            gm_ref, qr.RngState(SEED + "-refresh"), c_ref
        )
        self.assertEqual(bool(applied), bool(applied_ref))
        self.assertLess(
            float(np.max(np.abs(np.asarray(gm) - np.asarray(gm_ref)))), 1e-12
        )
        self.assertAlmostEqual(dk, dk_ref, delta=1e-10)

    def test_uniform_refresh_matches(self):
        q_refresh, ref_refresh = self.get_cgk("partial_refresh_momentum_mode")
        q_gen, ref_gen = self.get_cgk("generate_c_field")
        c_q = q_gen(self.geo, lambda kv: 0.55)
        c_ref = ref_gen(self.geo_ref, lambda kv: 0.55)
        gm = q.GaugeMomentum(self.geo)
        gm.set_rand(q.RngState(SEED + "-uniform"), 1.0)
        gm_ref = qr.GaugeMomentum(self.geo_ref)
        gm_ref.set_rand(qr.RngState(SEED + "-uniform"), 1.0)
        q_refresh(gm, q.RngState(SEED + "-uniform-refresh"), c_q)
        ref_refresh(gm_ref, qr.RngState(SEED + "-uniform-refresh"), c_ref)
        self.assertLess(
            float(np.max(np.abs(np.asarray(gm) - np.asarray(gm_ref)))), 1e-13
        )

    def test_reverse_test_message_matches(self):
        # The wording is part of the contract: the two must raise the identical
        # NotImplementedError text so a caller can match on it.
        if not hasattr(q, "run_hmc_cgfhmc_pure_gauge"):
            self.skipTest("the installed qlat has no run_hmc_cgfhmc_pure_gauge")
        gf0 = q.GaugeField(self.geo)
        gf0.set_unit()
        gf_ref = qr.GaugeField(self.geo_ref)
        gf_ref.set_unit()
        c_q = q.generate_c_field(self.geo, lambda kv: 0.5)
        c_ref = qr.generate_c_field(self.geo_ref, lambda kv: 0.5)
        with self.assertRaises(NotImplementedError) as q_context:
            q.run_hmc_cgfhmc_pure_gauge(
                gf0,
                q.GaugeAction(5.5, 0.0),
                1,
                q.RngState(SEED),
                c_field=c_q,
                is_reverse_test=True,
            )
        with self.assertRaises(NotImplementedError) as ref_context:
            qr.run_hmc_cgfhmc_pure_gauge(
                gf_ref,
                qr.GaugeAction(5.5, 0.0),
                1,
                qr.RngState(SEED),
                c_field=c_ref,
                is_reverse_test=True,
            )
        self.assertEqual(str(q_context.exception), str(ref_context.exception))

    def test_full_trajectory_matches_at_several_step_counts(self):
        # Compare whole CGFHMC trajectories, not just the helper functions: the
        # per-step gauge-fixing + refresh must sit at the SAME point inside the
        # integrator in both drivers, and only a multi-step trajectory shows a
        # difference in that schedule.
        #
        # This is not hypothetical: qlat's driver once hoisted one ``lam*dt``
        # drift out of the step loop (a copy of run_hmc_evolve_pure_gauge's
        # shape) while qlat_ref closed each step with the self-contained
        # ``lam, (1-2*lam), lam`` sandwich. The two agreed at n_step=1 and
        # diverged by O(10) in delta_h from n_step=2 on -- the helpers compared
        # by the tests above all matched. Hence n_step > 1 here.
        if not hasattr(q, "run_hmc_cgfhmc_pure_gauge"):
            self.skipTest("the installed qlat has no run_hmc_cgfhmc_pure_gauge")
        cut = float(q.resolve_k_cut(self.geo, 1))
        #
        # The two-zone profile exercises the Fourier refresh path; the uniform
        # value is checked by the golden vectors in test_cgfhmc.py.
        def func(kv):
            return np.where(np.sum(kv**2, axis=0) <= cut**2, 0.2, 0.9)
        #
        c_q = q.generate_c_field(self.geo, func)
        c_ref = qr.generate_c_field(self.geo_ref, func)
        for n_step, md_time in ((2, 0.25), (3, 0.375), (8, 1.0)):
            with self.subTest(n_step=n_step):
                gf = q.GaugeField(self.geo)
                gf.set_rand(q.RngState(SEED + "-cgfhmc-traj"), 0.3, 2)
                gf.unitarize()
                gf_ref = qr.GaugeField(self.geo_ref)
                n = int(np.prod(TOTAL_SITE))
                flat = np.asarray(gf).reshape((n, 4, qr.NC, qr.NC), order="F")
                site_major = flat.reshape(
                    tuple(TOTAL_SITE) + (4, qr.NC, qr.NC), order="F"
                )
                gf_ref = fld.get_field_from_array(
                    np.moveaxis(site_major, -3, 0), self.geo_ref
                )
                info = {}
                info_ref = {}
                flag, delta_h = q.run_hmc_cgfhmc_pure_gauge(
                    gf,
                    q.GaugeAction(5.5, 0.0),
                    1,
                    q.RngState(SEED + "-cgfhmc-traj-run"),
                    c_field=c_q,
                    n_step=n_step,
                    md_time=md_time,
                    is_always_accept=True,
                    info_out=info,
                )
                flag_ref, delta_h_ref = qr.run_hmc_cgfhmc_pure_gauge(
                    gf_ref,
                    qr.GaugeAction(5.5, 0.0),
                    1,
                    qr.RngState(SEED + "-cgfhmc-traj-run"),
                    c_field=c_ref,
                    n_step=n_step,
                    md_time=md_time,
                    is_always_accept=True,
                    info_out=info_ref,
                )
                self.assertEqual(int(flag), int(flag_ref))
                # Absolute tolerances: delta_h is O(V * md_time) here (hundreds
                # at 2^4), so 1e-9 is ~1e-12 relative and far below the O(10)
                # signature of a wrong insertion point.
                self.assertAlmostEqual(delta_h, delta_h_ref, delta=1e-9)
                self.assertAlmostEqual(info["dK_sum"], info_ref["dK_sum"], delta=1e-9)
                self.assertEqual(info["n_fix"], info_ref["n_fix"])
                self.assertLess(
                    float(np.max(np.abs(np.asarray(gf) - np.asarray(gf_ref)))),
                    1e-12,
                )
                self.assertAlmostEqual(
                    float(gf.plaq()), float(gf_ref.plaq()), delta=1e-12
                )

### -------------------------------------------------------------------

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestGaugeFixingParity(unittest.TestCase):
    """The fixing paths must converge to the same field as qlat's."""

    geo = None
    geo_ref = None

    @classmethod
    def setUpClass(cls):
        import qlat_utils as qu
        #
        cls.geo = q.Geometry(qu.Coordinate(list(TOTAL_SITE)))
        cls.geo_ref = qr.Geometry(qr.Coordinate(TOTAL_SITE))

    def get_pair(self):
        gf = q.GaugeField(self.geo)
        gf.set_rand(q.RngState(SEED), 0.5, 1)
        arr = np.asarray(gf).copy()
        # (local_volume, 4, NC, NC) -> (4, Nt, Nx, Ny, Nz, NC, NC)
        n = int(np.prod(TOTAL_SITE))
        flat = arr.reshape((n, 4, qr.NC, qr.NC), order="F")
        site_major = flat.reshape(tuple(TOTAL_SITE) + (4, qr.NC, qr.NC), order="F")
        link_arr = np.moveaxis(site_major, -3, 0)
        gf_ref = fld.get_field_from_array(link_arr, self.geo_ref)
        return gf, gf_ref

    def test_fix_landau_gauge(self):
        gf, gf_ref = self.get_pair()
        gt, info = q.fix_landau_gauge(gf, max_iter=300, eps=1e-10, alpha=0.1)
        gt_ref, info_ref = qr.gauge_fixing.fix_landau_gauge(
            gf_ref, max_iter=300, eps=1e-10, alpha=0.1
        )
        self.assertEqual(info["n_iter"], info_ref["n_iter"])
        self.assertAlmostEqual(info["functional"], info_ref["functional"], places=10)
        self.assertLess(
            float(np.max(np.abs(np.asarray(gt * gf) - np.asarray(gt_ref * gf_ref)))),
            1e-9,
        )

    def test_fix_coulomb_gauge_uses_the_correct_directions(self):
        """
        qlat_ref's Coulomb fixing is INTENTIONALLY different from qlat's.\n
        Component 3 is time in qlat's convention (`total_site` is
        (Lx, Ly, Lz, Lt), `Geometry::spatial_volume` divides out component 3,
        `get_psel_tslice` slices `t_dir=3`), so the spatial directions are
        (0, 1, 2).  qlat's `fix_coulomb_gauge` uses {1, 2, 3} -- qlat-issue-8.
        Instead of comparing against the buggy fixer, this test asserts the
        deviation is real in both directions: qlat_ref reproduces qlat's
        algorithm exactly when given the same `dirs` via `fix_gauge`, and its
        own Coulomb default matches the *correct* directions.
        """
        gf, gf_ref = self.get_pair()
        # 1) Same algorithm: fix_gauge with qlat's dirs must agree bit-for-bit.
        gt_q, info_q = q.fix_gauge(
            gf, dirs=[1, 2, 3], max_iter=300, eps=1e-10, alpha=0.1
        )
        gt_ref, info_ref = qr.gauge_fixing.fix_gauge(
            gf_ref, (1, 2, 3), max_iter=300, eps=1e-10, alpha=0.1
        )
        self.assertEqual(info_q["n_iter"], info_ref["n_iter"])
        self.assertLess(
            float(np.max(np.abs(np.asarray(gt_q * gf) - np.asarray(gt_ref * gf_ref)))),
            1e-9,
        )
        # qlat's own coulomb fixer IS fix_gauge(dirs=[1,2,3]) -- no third
        # convention is hiding here.
        gt_qc, info_qc = q.fix_coulomb_gauge(gf, max_iter=300, eps=1e-10)
        self.assertEqual(info_qc["n_iter"], info_q["n_iter"])
        self.assertLess(
            float(np.max(np.abs(np.asarray(gt_qc * gf) - np.asarray(gt_q * gf)))),
            1e-9,
        )
        # 2) qlat_ref's Coulomb default is the CORRECT {0,1,2}.
        gt_ours, info_ours = qr.gauge_fixing.fix_coulomb_gauge(
            gf_ref, max_iter=300, eps=1e-10, alpha=0.1
        )
        self.assertEqual(tuple(info_ours["dirs"]), (0, 1, 2))
        fixed_ours = gt_ours * gf_ref
        self.assertLessEqual(
            qr.observables.get_gauge_divergence(fixed_ours, (0, 1, 2)), 1e-9
        )
        self.assertGreater(
            qr.observables.get_gauge_divergence(fixed_ours, (1, 2, 3)), 1e-2
        )
        # 3) ... and qlat's fixed field shows the opposite signature, which is
        # what makes the deviation observable rather than cosmetic.
        fixed_qlat = gt_qc * gf
        self.assertGreater(q.get_gauge_divergence(fixed_qlat, [0, 1, 2]), 1e-2)
        self.assertLessEqual(q.get_gauge_divergence(fixed_qlat, [1, 2, 3]), 1e-9)
        self.assertIn("fix_coulomb_gauge", DEVIATIONS)

    def test_functional_and_divergence_values(self):
        gf, gf_ref = self.get_pair()
        dirs = (0, 1, 2, 3)
        self.assertAlmostEqual(
            q.get_gauge_functional(gf, list(dirs)),
            qr.observables.get_gauge_functional(gf_ref, dirs),
            places=12,
        )
        self.assertAlmostEqual(
            q.get_gauge_divergence(gf, list(dirs)),
            qr.observables.get_gauge_divergence(gf_ref, dirs),
            places=12,
        )

    def test_info_keys_are_a_superset_of_qlat(self):
        gf, gf_ref = self.get_pair()
        _, info = q.fix_landau_gauge(gf, max_iter=50, eps=1e-8)
        _, info_ref = qr.gauge_fixing.fix_landau_gauge(gf_ref, max_iter=50, eps=1e-8)
        missing = set(info.keys()) - set(info_ref.keys())
        extra = set(info_ref.keys()) - set(info.keys())
        self.assertEqual(missing, set(), f"qlat_ref lost the qlat keys {missing}")
        # 'dirs' is the one documented addition.
        self.assertEqual(extra, {"dirs"})
        self.assertIn("fix_gauge", DEVIATIONS)

    def test_gauge_action_value(self):
        gf, gf_ref = self.get_pair()
        for beta, c1 in ((5.5, 0.0), (5.5, -0.331), (6.0, 0.1)):
            with self.subTest(beta=beta, c1=c1):
                self.assertAlmostEqual(
                    q.gf_hamilton_node(gf, q.GaugeAction(beta, c1)),
                    qr.gauge_action.get_gauge_action(gf_ref, qr.GaugeAction(beta, c1)),
                    places=9,
                )

### -------------------------------------------------------------------

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestFileFormatParity(unittest.TestCase):
    """Files must be interchangeable in both directions."""

    geo = None
    geo_ref = None

    @classmethod
    def setUpClass(cls):
        import qlat_utils as qu
        #
        cls.geo = q.Geometry(qu.Coordinate(list(TOTAL_SITE)))
        cls.geo_ref = qr.Geometry(qr.Coordinate(TOTAL_SITE))

    def test_qlat_written_file_is_readable(self):
        import tempfile
        import os
        #
        gf = q.GaugeField(self.geo)
        gf.set_rand(q.RngState(SEED), 0.5, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "qlat.lat")
            gf.save(path)
            gf_ref = qr.GaugeField(self.geo_ref)
            gf_ref.load(path)
        self.assertLess(
            float(np.max(np.abs(np.asarray(gf) - np.asarray(gf_ref)))), 1e-13
        )

    def test_qlat_reads_our_file(self):
        import tempfile
        import os
        #
        gf_ref = qr.GaugeField(self.geo_ref)
        gf_ref.set_rand(qr.RngState(SEED), 0.5, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ref.lat")
            gf_ref.save(path)
            gf = q.GaugeField(self.geo)
            gf.load(path)
        self.assertLess(
            float(np.max(np.abs(np.asarray(gf) - np.asarray(gf_ref)))), 1e-13
        )
