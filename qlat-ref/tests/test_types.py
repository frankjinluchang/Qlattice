"""
Tests of the qlat-matching field types (:mod:`qlat_ref.base`,
:mod:`qlat_ref.mat`, :mod:`qlat_ref.field`, :mod:`qlat_ref.gauge_action`).\n
These are the cheap structural tests of the type layer: the constructors, the
NumPy buffer and its array bridge, the ``gt * gf`` / ``gt * gm`` conventions,
the NERSC ``.lat`` round trip and the ``GaugeAction`` accessors.  Numerical
parity against a live ``qlat`` build is a separate, optional module so this
one stays dependency-free.
"""

import os
import tempfile
import unittest

import numpy as np

import qlat_ref as qr
from qlat_ref import field as fld

from tests.helpers import SEED
from tests.helpers import TOTAL_SITE

### -------------------------------------------------------------------

class TestGeometryAndCoordinate(unittest.TestCase):
    def test_geometry_properties(self):
        geo = qr.Geometry(qr.Coordinate(TOTAL_SITE))
        self.assertEqual(geo.total_site.to_list(), list(TOTAL_SITE))
        self.assertEqual(geo.total_volume, np.prod(TOTAL_SITE))
        self.assertEqual(geo.local_volume, geo.total_volume)
        self.assertEqual(geo.local_site.to_list(), list(TOTAL_SITE))
        self.assertEqual(geo.num_node, 1)
        self.assertEqual(geo.id_node, 0)
        self.assertTrue(geo.is_only_local)

    def test_geometry_round_trips_coordinates(self):
        geo = qr.Geometry(qr.Coordinate(TOTAL_SITE))
        for index in range(geo.local_volume):
            x = geo.coordinate_from_index(index)
            self.assertEqual(geo.index_from_coordinate(x), index)
            self.assertEqual(geo.coordinate_g_from_l(x), x)

    def test_geometry_reproduces_from_its_show(self):
        geo = qr.Geometry(qr.Coordinate(TOTAL_SITE))
        self.assertEqual(geo.show(), f"Geometry({list(TOTAL_SITE)})")

    def test_coordinate_basics(self):
        x = qr.Coordinate([1, 2, 3, 4])
        self.assertEqual(x.to_list(), [1, 2, 3, 4])
        self.assertEqual(len(x), 4)
        self.assertEqual(x[0], 1)
        self.assertEqual(x[-1], 4)
        self.assertEqual((x + x).to_list(), [2, 4, 6, 8])
        self.assertEqual((2 * x).to_list(), [2, 4, 6, 8])
        self.assertEqual(x.to_numpy().dtype, np.int32)
        self.assertEqual(hash(x), hash(qr.Coordinate([1, 2, 3, 4])))

class TestEmptyGeometry(unittest.TestCase):
    """
    The empty-but-safe contract for a default-constructed ``Geometry()``
    (qlat-issue-9 in ``tmp/qlat-ref-reorg/qlat-issues.md``).\n
    qlat's bare ``Geometry()`` is an *empty but safe* object: an uninitialized
    geometry node (``size_node == [0, 0, 0, 0]``, ``num_node == 0``), zero
    volumes, and no property that divides by a zero extent.  ``qlat_ref``
    matches that contract exactly, so the same assertions hold for both
    packages.
    """

    def test_all_properties_are_empty_and_safe(self):
        geo = qr.Geometry()
        self.assertEqual(geo.total_site.to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.local_site.to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.node_site.to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.size_node.to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.coor_node.to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.expansion_left.to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.expansion_right.to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.num_node, 0)
        self.assertEqual(geo.id_node, 0)
        self.assertEqual(geo.eo, 0)
        self.assertEqual(geo.total_volume, 0)
        self.assertEqual(geo.spatial_volume, 0)
        self.assertEqual(geo.local_volume, 0)
        self.assertEqual(geo.local_volume_expanded, 0)
        # `is_only_local` is False for the uninitialized geometry in qlat too
        # (it tracks the initialized flag, not the zero volume).
        self.assertFalse(geo.is_only_local)

    def test_display_and_equality_are_safe(self):
        geo = qr.Geometry()
        self.assertEqual(repr(geo), "Geometry([0, 0, 0, 0])")
        self.assertEqual(str(geo), "Geometry([0, 0, 0, 0])")
        self.assertEqual(geo.show(), "Geometry([0, 0, 0, 0])")
        self.assertEqual(geo, qr.Geometry())
        self.assertNotEqual(geo, qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        self.assertEqual(geo.copy(), geo)
        self.assertEqual(geo.copy(), qr.Geometry())

    def test_coordinate_mapping_is_safe(self):
        # No site is owned, so the index -> coordinate maps return the zero
        # coordinate instead of dividing by a zero extent (qlat's guard).
        geo = qr.Geometry()
        x = qr.Coordinate([1, 1, 1, 1])
        self.assertEqual(geo.coordinate_from_index(0).to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.g_coordinate_from_index(0).to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.coordinate_l_from_g(x).to_list(), [1, 1, 1, 1])
        self.assertEqual(geo.coordinate_g_from_l(x).to_list(), [1, 1, 1, 1])
        self.assertEqual(geo.xg_arr().shape, (0, 4))

    def test_a_zero_extent_geometry_is_also_safe(self):
        # A geometry with one zero extent (but an initialized profile) is
        # empty as well; every volume is 0 and nothing divides by zero.
        geo = qr.Geometry(0, qr.Coordinate([1, 1, 1, 1]), qr.Coordinate([4, 4, 4, 0]))
        self.assertEqual(geo.total_site.to_list(), [4, 4, 4, 0])
        self.assertEqual(geo.num_node, 1)
        self.assertTrue(geo.is_only_local)
        self.assertEqual(geo.total_volume, 0)
        self.assertEqual(geo.spatial_volume, 0)
        self.assertEqual(geo.local_volume, 0)
        self.assertEqual(geo.local_volume_expanded, 0)
        self.assertEqual(repr(geo), "Geometry([4, 4, 4, 0])")
        self.assertEqual(geo.coordinate_from_index(0).to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.g_coordinate_from_index(0).to_list(), [0, 0, 0, 0])
        self.assertEqual(geo.xg_arr().shape, (0, 4))

    def test_a_valid_geometry_is_unaffected(self):
        geo = qr.Geometry(qr.Coordinate(TOTAL_SITE))
        self.assertEqual(geo.size_node.to_list(), [1, 1, 1, 1])
        self.assertEqual(geo.num_node, 1)
        self.assertTrue(geo.is_only_local)
        self.assertEqual(geo.total_volume, np.prod(TOTAL_SITE))
        self.assertEqual(geo.spatial_volume, np.prod(TOTAL_SITE) // TOTAL_SITE[3])
        for index in range(geo.local_volume):
            x = geo.coordinate_from_index(index)
            self.assertEqual(geo.index_from_coordinate(x), index)

class TestArrayBridge(unittest.TestCase):
    def test_field_round_trips_through_the_bridge(self):
        gf = qr.GaugeField(qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        gf.set_rand(qr.RngState(SEED), sigma=0.5, n_step=1)
        arr = fld.get_array_from_field(gf)
        self.assertEqual(arr.shape, (4,) + tuple(TOTAL_SITE) + (qr.NC, qr.NC))
        gf2 = fld.get_field_from_array(arr, gf.geo)
        self.assertTrue(np.array_equal(np.asarray(gf2), np.asarray(gf)))

    def test_np_asarray_is_a_live_view(self):
        # The documented bridge must not copy: writing the array writes the
        # field, which is what the HMC-style in-place updates rely on.
        gf = qr.GaugeField(qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        gf.set_unit(1.0)
        # set_unit sets only the (NC) diagonal entries, so the unit field has
        # NC nonzero entries per link: qnorm = n_sites * multiplicity * NC.
        n_links = int(np.prod(TOTAL_SITE)) * 4
        self.assertEqual(gf.qnorm(), n_links * qr.NC)
        # The bridge is a live view: zeroing one 3x3 link matrix removes NC.
        np.asarray(gf)[0, 0] = 0.0
        self.assertEqual(gf.qnorm(), n_links * qr.NC - qr.NC)

    def test_set_rand_and_set_rand_g_ranges(self):
        # Regression: FieldColorMatrix.set_rand/set_rand_g used to call
        # u_rand_arr/g_rand_arr with lower/upper/center/sigma keywords, which
        # those methods do not take (qlat's take only a shape), so both raised
        # TypeError.  Neither was covered by a test, so the bug survived.
        geo = qr.Geometry(qr.Coordinate(TOTAL_SITE))
        cm = qr.FieldColorMatrix(geo, 4)
        cm.set_rand(qr.RngState(f"{SEED}-u-rand"), 2.5, -1.5)
        re = np.asarray(cm).real
        self.assertTrue(np.all(re >= -1.5))
        self.assertTrue(np.all(re < 2.5))
        # The public order is (upper, lower), matching qlat's set_u_rand.
        cm2 = qr.FieldColorMatrix(geo, 4)
        cm2.set_rand(qr.RngState(f"{SEED}-u-rand"), upper=2.5, lower=-1.5)
        self.assertTrue(np.array_equal(np.asarray(cm), np.asarray(cm2)))
        # Defaults are qlat's: uniform in [0, 1).
        cm3 = qr.FieldColorMatrix(geo, 4)
        cm3.set_rand(qr.RngState(f"{SEED}-u-rand"))
        re3 = np.asarray(cm3).real
        self.assertTrue(np.all(re3 >= 0.0))
        self.assertTrue(np.all(re3 < 1.0))
        # set_rand_g(center, sigma): the real parts are N(center, sigma).
        cm4 = qr.FieldColorMatrix(geo, 4)
        cm4.set_rand_g(qr.RngState(f"{SEED}-g-rand"), 0.5, 2.0)
        re4 = np.asarray(cm4).real
        self.assertLess(abs(float(np.mean(re4)) - 0.5), 0.2)
        self.assertLess(abs(float(np.std(re4)) - 2.0), 0.3)

    def test_field_geometry_and_multiplicity(self):
        geo = qr.Geometry(qr.Coordinate(TOTAL_SITE))
        gf = qr.GaugeField(geo)
        self.assertEqual(gf.multiplicity, 4)
        self.assertEqual(gf.n_sites, geo.local_volume)
        self.assertEqual(gf.sizeof_m, qr.NC * qr.NC * 16)
        self.assertEqual(gf.total_site.to_list(), list(TOTAL_SITE))
        self.assertEqual(len(gf), geo.local_volume)
        gt = qr.GaugeTransform(geo)
        self.assertEqual(gt.multiplicity, 1)
        gm = qr.GaugeMomentum(geo)
        self.assertEqual(gm.multiplicity, 4)

class TestGaugeFieldOperations(unittest.TestCase):
    def get_field(self, seed=0):
        gf = qr.GaugeField(qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        gf.set_rand(qr.RngState(f"{SEED}-{seed}"), sigma=0.5, n_step=1)
        return gf

    def test_random_field_is_unitary(self):
        gf = self.get_field()
        self.assertLessEqual(qr.get_unitarity_error(gf), 1e-12)
        self.assertTrue(qr.check_unitarity(gf, eps=1e-12))

    def test_set_rand_is_deterministic(self):
        a = self.get_field(seed=1)
        b = self.get_field(seed=1)
        self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
        self.assertFalse(
            np.array_equal(np.asarray(a), np.asarray(self.get_field(seed=2)))
        )

    def test_set_rand_n_step_changes_the_field(self):
        a = qr.GaugeField(self.get_field().geo)
        b = qr.GaugeField(self.get_field().geo)
        rng_seed = f"{SEED}-nstep"
        a.set_rand(qr.RngState(rng_seed), sigma=0.5, n_step=1)
        b.set_rand(qr.RngState(rng_seed), sigma=0.5, n_step=2)
        self.assertFalse(np.array_equal(np.asarray(a), np.asarray(b)))

    def test_set_unit_and_clone(self):
        gf = self.get_field()
        gf.set_unit(2.0)
        self.assertEqual(gf.qnorm(), 2.0**2 * 3 * 4 * np.prod(TOTAL_SITE))
        other = gf.copy()
        self.assertTrue(np.array_equal(np.asarray(other), np.asarray(gf)))
        other.set_zero()
        self.assertEqual(other.qnorm(), 0.0)
        self.assertEqual(gf.qnorm(), 2.0**2 * 3 * 4 * np.prod(TOTAL_SITE))

    def test_unitarize_projects_onto_su3(self):
        gf = self.get_field()
        np.asarray(gf)[...] *= 1.3
        gf.unitarize()
        self.assertLessEqual(qr.get_unitarity_error(gf), 1e-12)

    def test_plaq_and_link_trace_of_the_unit_field(self):
        gf = qr.GaugeField(qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        gf.set_unit(1.0)
        self.assertAlmostEqual(gf.plaq(), 1.0, places=12)
        self.assertAlmostEqual(gf.link_trace(), 1.0, places=12)

    def test_plaq_is_real_and_below_one(self):
        gf = self.get_field()
        self.assertLess(gf.plaq(), 1.0)
        self.assertGreater(gf.plaq(), -1.0)

class TestGaugeTransformAlgebra(unittest.TestCase):
    def get_fields(self):
        geo = qr.Geometry(qr.Coordinate(TOTAL_SITE))
        gf = qr.GaugeField(geo)
        gf.set_rand(qr.RngState(f"{SEED}-gf"), sigma=0.5, n_step=1)
        gt = qr.GaugeTransform(geo)
        gt.set_rand(qr.RngState(f"{SEED}-gt"), sigma=0.5, n_step=1)
        gm = qr.GaugeMomentum(geo)
        gm.set_rand(qr.RngState(f"{SEED}-gm"), sigma=1.0)
        return gf, gt, gm

    def test_gt_times_gf_uses_the_shifted_transform(self):
        gf, gt, _ = self.get_fields()
        transformed = gt * gf
        self.assertIsInstance(transformed, qr.GaugeField)
        U = fld.get_array_from_field(gf)
        g = np.asarray(gt).reshape(tuple(TOTAL_SITE) + (qr.NC, qr.NC), order="F")
        expected = qr.apply_gauge_transform(U, g)
        self.assertLess(
            float(np.max(np.abs(fld.get_array_from_field(transformed) - expected))),
            1e-14,
        )

    def test_gt_times_gf_is_covariant(self):
        # A gauge-transformed field has the same plaquette: the plaquette is a
        # gauge invariant, so this checks the whole shift convention at once.
        gf, gt, _ = self.get_fields()
        self.assertAlmostEqual((gt * gf).plaq(), gf.plaq(), places=12)
        # The link trace and the gauge functional are NOT gauge invariant -- the
        # functional is exactly what gauge fixing maximizes -- so they must
        # change here; only the plaquette is invariant.
        self.assertNotAlmostEqual((gt * gf).link_trace(), gf.link_trace(), places=2)

    def test_gt_composition_order(self):
        _, gt, _ = self.get_fields()
        _, gt1, _ = self.get_fields()
        composed = gt * gt1
        self.assertLess(
            float(
                np.max(np.abs(np.asarray(composed) - np.asarray(gt1) @ np.asarray(gt)))
            ),
            1e-14,
        )

    def test_gt_inv_is_the_site_wise_adjoint(self):
        _, gt, _ = self.get_fields()
        inv = gt.inv()
        ident = np.einsum("...ij,...jk->...ik", np.asarray(inv), np.asarray(gt))
        expected = np.broadcast_to(np.eye(qr.NC, dtype=np.complex128), ident.shape)
        self.assertLess(float(np.max(np.abs(ident - expected))), 1e-13)

    def test_gt_times_gm_is_a_same_site_adjoint_action(self):
        # A momentum transforms at the SAME site: no x + mu shift.
        _, gt, gm = self.get_fields()
        transformed = gt * gm
        self.assertIsInstance(transformed, qr.GaugeMomentum)
        expected = qr.gm_apply_gauge_transformation(gm, gt)
        self.assertLess(
            float(np.max(np.abs(np.asarray(transformed) - np.asarray(expected)))),
            1e-14,
        )
        self.assertLess(
            float(np.max(np.abs(np.asarray(transformed) - np.asarray(gt * gm)))),
            1e-14,
        )

    def test_gm_apply_is_daggerable(self):
        _, gt, gm = self.get_fields()
        forward = qr.gm_apply_gauge_transformation(gm, gt)
        backward = qr.gm_apply_gauge_transformation(forward, gt, is_dagger=True)
        self.assertLess(
            float(np.max(np.abs(np.asarray(backward) - np.asarray(gm)))), 1e-13
        )

    def test_momentum_is_antihermitian(self):
        _, _, gm = self.get_fields()
        arr = np.asarray(gm)
        self.assertLess(
            float(np.max(np.abs(arr + np.conj(np.swapaxes(arr, -1, -2))))), 1e-13
        )

class TestUnitarityErrorAcceptedTypes(unittest.TestCase):
    """
    ``get_unitarity_error_typed`` is the typed entry point mirroring
    ``qlat.gauge_fixing.get_unitarity_error`` and must accept exactly the types
    qlat accepts: a ``GaugeField`` and a ``GaugeTransform``.\n
    The number is the maximal ``|U U^dagger - 1|`` matrix element, which is
    meaningful for any field of matrices but NOT for a momentum (whose elements
    are anti-Hermitian, so ``P P^dagger - 1`` is not a unitarity defect) nor for
    a raw array (no field type, so a wrong call site would go unnoticed).  This
    used to be a documented qlat/qlat_ref deviation; it is qlat-issue-11 and the
    two packages now share the same contract.
    """

    def get_fields(self):
        geo = qr.Geometry(qr.Coordinate(TOTAL_SITE))
        gf = qr.GaugeField(geo)
        gf.set_rand(qr.RngState(f"{SEED}-u-gf"), sigma=0.5, n_step=1)
        gt = qr.GaugeTransform(geo)
        gt.set_rand(qr.RngState(f"{SEED}-u-gt"), sigma=0.5, n_step=1)
        gm = qr.GaugeMomentum(geo)
        gm.set_rand(qr.RngState(f"{SEED}-u-gm"), sigma=1.0)
        return gf, gt, gm

    def test_accepts_gauge_field(self):
        gf, _, _ = self.get_fields()
        value = qr.observables.get_unitarity_error_typed(gf)
        self.assertIsInstance(value, float)
        # A near-identity random field is unitary to round-off.
        self.assertLessEqual(value, 1e-12)
        # ... and it is the same number as the array-level helper's.
        self.assertEqual(
            value, qr.mat.get_unitarity_error(fld.get_array_from_field(gf))
        )

    def test_accepts_gauge_transform(self):
        _, gt, _ = self.get_fields()
        value = qr.observables.get_unitarity_error_typed(gt)
        self.assertIsInstance(value, float)
        self.assertLessEqual(value, 1e-12)
        self.assertEqual(value, qr.mat.get_unitarity_error(np.asarray(gt)))

    def test_equal_content_gives_the_same_value(self):
        # Four copies of one transform's site matrix is a gauge field with
        # exactly that transform's unitarity error.
        _, gt, _ = self.get_fields()
        geo = qr.Geometry(qr.Coordinate(TOTAL_SITE))
        gf = qr.GaugeField(geo)
        gf._arr[...] = np.broadcast_to(np.asarray(gt), gf._arr.shape)
        self.assertEqual(
            qr.observables.get_unitarity_error_typed(gf),
            qr.observables.get_unitarity_error_typed(gt),
        )

    def test_rejects_a_gauge_momentum(self):
        _, _, gm = self.get_fields()
        with self.assertRaises(TypeError) as ctx:
            qr.observables.get_unitarity_error_typed(gm)
        self.assertIn("GaugeField or a GaugeTransform", str(ctx.exception))

    def test_rejects_plain_arrays_and_other_objects(self):
        gf, _, _ = self.get_fields()
        for obj in (fld.get_array_from_field(gf), np.eye(qr.NC), [1, 2, 3], None):
            with self.subTest(obj=type(obj).__name__):
                with self.assertRaises(TypeError):
                    qr.observables.get_unitarity_error_typed(obj)

class TestGaugeAction(unittest.TestCase):
    def test_accessors_are_methods(self):
        # qlat exposes beta() and c1() as METHODS, not properties.
        ga = qr.GaugeAction(5.5, -0.331)
        self.assertEqual(ga.beta(), 5.5)
        self.assertEqual(ga.c1(), -0.331)
        self.assertEqual(qr.GaugeAction(6.0).c1(), 0.0)

    def test_imatmul_assigns(self):
        ga = qr.GaugeAction(1.0, 2.0)
        ga @= qr.GaugeAction(5.5, -0.331)
        self.assertEqual(ga.beta(), 5.5)
        self.assertEqual(ga.c1(), -0.331)

    def test_action_of_the_unit_field_is_zero(self):
        gf = qr.GaugeField(qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        gf.set_unit(1.0)
        for c1 in (0.0, -0.331):
            self.assertAlmostEqual(
                qr.get_gauge_action(gf, qr.GaugeAction(5.5, c1)), 0.0, places=12
            )

    def test_action_is_positive_and_beta_scales_it(self):
        gf = qr.GaugeField(qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        gf.set_rand(qr.RngState(SEED), sigma=0.5, n_step=1)
        a1 = qr.get_gauge_action(gf, qr.GaugeAction(5.5))
        a2 = qr.get_gauge_action(gf, qr.GaugeAction(11.0))
        self.assertGreater(a1, 0.0)
        self.assertAlmostEqual(a2, 2.0 * a1, places=10)

class TestFileIO(unittest.TestCase):
    def get_field(self):
        gf = qr.GaugeField(qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        gf.set_rand(qr.RngState(SEED), sigma=0.5, n_step=1)
        return gf

    def test_nersc_lat_round_trip(self):
        gf = self.get_field()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "field.lat")
            n = gf.save(path)
            self.assertGreater(n, 0)
            other = qr.GaugeField()
            other.load(path)
            self.assertEqual(other.geo.total_site.to_list(), list(TOTAL_SITE))
            self.assertLess(
                float(np.max(np.abs(np.asarray(other) - np.asarray(gf)))), 1e-13
            )

    def test_nersc_header_records_the_observables(self):
        gf = self.get_field()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "field.lat")
            gf.save(path)
            with open(path, "rb") as f:
                head = f.read(4096).decode("ascii", "replace")
        self.assertIn("DATATYPE = 4D_SU3_GAUGE", head)
        self.assertIn("CHECKSUM = ", head)
        self.assertIn("CRC32HASH = ", head)
        self.assertIn("FLOATING_POINT = IEEE64BIG", head)

    def test_corrupt_file_is_rejected(self):
        gf = self.get_field()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "field.lat")
            gf.save(path)
            raw = bytearray(open(path, "rb").read())
            raw[-1] ^= 0xFF
            open(path, "wb").write(bytes(raw))
            with self.assertRaises(IOError):
                qr.GaugeField().load(path)

    def test_gauge_transform_generic_round_trip(self):
        gt = qr.GaugeTransform(qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        gt.set_rand(qr.RngState(SEED), sigma=0.5, n_step=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gt.field")
            gt.save(path)
            other = qr.GaugeTransform()
            other.load(path)
            self.assertLess(
                float(np.max(np.abs(np.asarray(other) - np.asarray(gt)))), 1e-14
            )

class TestCoulombConvention(unittest.TestCase):
    def test_coulomb_uses_the_correct_spatial_directions(self):
        # Component 3 is time in qlat's convention (`total_site` is
        # (Lx, Ly, Lz, Lt), `Geometry::spatial_volume` divides out component 3,
        # `get_psel_tslice` slices `t_dir=3`), so the spatial directions are
        # 0, 1, 2.  qlat's `gf_fix_coulomb_gauge` uses {1, 2, 3} instead -- that
        # is qlat-issue-8 in tmp/qlat-ref-reorg/qlat-issues.md, and qlat_ref
        # deliberately implements the correct set.  See also
        # tests/test_qlat_parity.py::TestGaugeFixingParity, which allowlists
        # this deviation explicitly.
        gf = qr.GaugeField(qr.Geometry(qr.Coordinate(TOTAL_SITE)))
        gf.set_rand(qr.RngState(SEED), sigma=0.5, n_step=1)
        _, info = qr.fix_coulomb_gauge(gf, max_iter=2000, eps=1e-9)
        self.assertEqual(tuple(info["dirs"]), (0, 1, 2))
        self.assertEqual(info["gauge"], "coulomb")
        landau_info = qr.fix_landau_gauge(gf, max_iter=200, eps=1e-8)[1]
        self.assertEqual(tuple(landau_info["dirs"]), (0, 1, 2, 3))
        self.assertEqual(landau_info["gauge"], "landau")

    def test_coulomb_condition_holds_for_0_1_2_only(self):
        gf = qr.GaugeField(qr.Geometry(qr.Coordinate((2, 2, 2, 2))))
        gf.set_rand(qr.RngState(SEED), sigma=0.5, n_step=1)
        gt, info = qr.fix_coulomb_gauge(gf, max_iter=2000, eps=1e-9)
        self.assertTrue(info["converged"])
        fixed = gt * gf
        self.assertLessEqual(qr.get_gauge_divergence(fixed, (0, 1, 2)), 1e-8)
        self.assertGreater(qr.get_gauge_divergence(fixed, (1, 2, 3)), 1e-2)

class TestTimeAxis(unittest.TestCase):
    def test_time_is_component_3(self):
        # The per-time-slice helpers must split along component 3 of
        # total_site, i.e. array axis 4 of the link field.  An asymmetric
        # lattice is required to catch a swap: with all extents equal any axis
        # passes.
        total_site = (2, 3, 4, 5)
        U = qr.get_random_gauge_field(total_site, seed=3, scale=0.8)
        self.assertEqual(len(qr.get_coulomb_functional_list(U)), total_site[3])
        self.assertEqual(len(qr.get_coulomb_divergence_list(U)), total_site[3])
        self.assertEqual(fld._TIME_DIR, 3)
        self.assertEqual(fld._TIME_AXIS, 4)

    def test_spatial_volume_divides_out_component_3(self):
        # Geometry.spatial_volume = total_volume / total_site[3], i.e. the
        # product of the first three components (2*3*4), confirming that
        # component 3 is time and that the spatial directions are 0, 1, 2.
        geo = qr.Geometry(qr.Coordinate((2, 3, 4, 5)))
        self.assertEqual(geo.spatial_volume, 2 * 3 * 4)
        self.assertEqual(geo.spatial_volume, geo.total_volume // 5)

    def test_coulomb_fixing_converges_per_time_slice(self):
        total_site = (2, 3, 4, 5)
        U = qr.get_random_gauge_field(total_site, seed=5, scale=0.8)
        before = qr.get_coulomb_functional_list(U)
        gt, info = qr.fix_coulomb_gauge(U, max_iter=3000, eps=1e-9)
        self.assertTrue(info["converged"])
        fixed = qr.apply_gauge_transform(U, gt)
        after = qr.get_coulomb_functional_list(fixed)
        # Each slice is an independent 3D Landau problem: every slice improves.
        self.assertTrue(all(a > b for b, a in zip(before, after)))
        divergence_list = qr.get_coulomb_divergence_list(fixed)
        self.assertLessEqual(max(divergence_list), 1e-8 * np.sqrt(len(divergence_list)))
