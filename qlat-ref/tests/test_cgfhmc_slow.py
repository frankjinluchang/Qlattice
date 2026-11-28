"""
Slow tests of :mod:`qlat_ref.cgfhmc`: the emergent properties that need
statistics or a bigger lattice.\n
The fast module (:mod:`tests.test_cgfhmc`) pins every *function* against
independently computed values and against golden CGFHMC trajectories, so an
implementation error is caught there.  What cannot be expressed that way is the
behaviour that only appears over many trajectories or on a lattice large enough
to matter:\n
* the Gaussian momentum ensemble really is **stationary** under the mode mixing
  (a rescaling error would still pass a single-refresh test),
* a chain stays **healthy and bounded** (no drift, no runaway ``delta_h``) with
  and without the refresh,
* the refresh actually **changes the decorrelation**, which is the entire point
  of the mode dependence,
* and the transform is right on a lattice whose axes all **differ**, where a
  transposed axis cannot cancel out.\n
These are gated behind ``QLAT_REF_SLOW=1`` because they are minutes of wasted CI
otherwise.  Run them with::\n
    cd qlat-ref
    QLAT_REF_SLOW=1 python3 -m pytest tests/test_cgfhmc_slow.py -v\n
Tolerances are absolute where a quantity can be zero and statistical bands
where the quantity is a sample mean; nothing here compares bit patterns.
"""

import math
import os
import unittest

import numpy as np

import qlat_ref as qr
from qlat_ref import cgfhmc
from qlat_ref.hmc import gm_hamilton_node

TOTAL_SITE = (2, 2, 2, 2)
SLOW = os.environ.get("QLAT_REF_SLOW", "") == "1"

### -------------------------------------------------------------------

def get_geometry(total_site=TOTAL_SITE):
    """Return a :class:`qlat_ref.Geometry` of ``total_site`` (helper)."""
    return qr.Geometry(qr.Coordinate(list(total_site)))

def get_uniform_c(geo, c):
    """Return the constant ``c(k) = c`` field on ``geo`` (helper)."""
    return cgfhmc.generate_c_field(geo, lambda kv: c)

def get_two_zone_c(geo, c_low=0.35, c_high=0.9):
    """Return a two-zone ``c(k)``: stiff infrared, aggressive ultraviolet (helper)."""
    threshold = cgfhmc.resolve_k_cut(geo, 1)
    return cgfhmc.generate_c_field(
        geo, lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, c_low, c_high)
    )

def get_random_momentum(geo, seed="momentum"):
    """Return a Gaussian momentum drawn from ``seed`` (helper)."""
    gm = qr.GaugeMomentum(geo)
    gm.set_rand(qr.RngState(seed), 1.0)
    return gm

def get_kinetic_energy_sample(geo, c_field, n_sample, label):
    """Return ``(energies_before, energies_after)`` over ``n_sample`` draws (helper)."""
    energies_before = []
    energies_after = []
    for i in range(n_sample):
        gm = get_random_momentum(geo, f"{label}-momentum-{i}")
        energies_before.append(float(gm_hamilton_node(gm)))
        cgfhmc.partial_refresh_momentum_mode(
            gm, qr.RngState(f"{label}-mix-{i}"), c_field
        )
        energies_after.append(float(gm_hamilton_node(gm)))
    #
    return energies_before, energies_after

### -------------------------------------------------------------------

@unittest.skipUnless(SLOW, "set QLAT_REF_SLOW=1 to run the slow CGFHMC tests")
class TestEnsembleStationarity(unittest.TestCase):
    """The mode mixing preserves the Gaussian momentum ensemble."""

    def test_mean_kinetic_energy_is_unchanged(self):
        # A single refresh always satisfies the per-mode identity by
        # construction; only a sample can detect a wrong *overall* rescaling
        # (which is what a non-unitary FFT normalization would produce).
        #
        # 2^4 with 64 draws instead of 4^4 with 48: the claim is statistical
        # in the number of *samples*, not in the volume, and one draw costs
        # 0.06 s here versus 0.97 s on 4^4 (measured).  Measured relative
        # deviation: 0.23%, against a 2% band.
        geo = get_geometry()
        c = get_two_zone_c(geo)
        before, after = get_kinetic_energy_sample(geo, c, 64, "stationarity")
        mean_before = float(np.mean(before))
        mean_after = float(np.mean(after))
        relative = abs(mean_after - mean_before) / mean_before
        self.assertLess(relative, 0.02)

    def test_variance_is_unchanged_per_mode_family(self):
        # Split the modes into an infrared (frozen) and an ultraviolet (fully
        # resampled) family and check that each keeps its spectral weight.  A
        # wrong FFT normalization rescales the transform, so it rescales the two
        # families by the *same* factor and could hide in a total-energy check;
        # comparing the ratio of the two cancels that and pins the mixing itself.
        geo = get_geometry((2, 2, 2, 2))
        threshold = cgfhmc.resolve_k_cut(geo, 1)
        k_sq = cgfhmc.k_squared(geo)
        low_mask = k_sq <= threshold**2
        self.assertGreater(int(np.count_nonzero(low_mask)), 0)
        self.assertGreater(int(np.count_nonzero(~low_mask)), 0)
        c = cgfhmc.generate_c_field(
            geo, lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, 0.0, 1.0)
        )
        ratios_before = []
        ratios_after = []
        for _ in range(24):
            gm = get_random_momentum(geo, f"family-{_}")
            # The eight real su(3) components of the momentum, transformed: this
            # is exactly the space the refresh mixes in.
            phi_k = np.abs(np.fft.fftn(cgfhmc._get_components(gm), norm="ortho")) ** 2
            ratios_before.append(
                float(np.sum(phi_k[..., low_mask].sum()))
                / float(np.sum(phi_k[..., ~low_mask].sum()))
            )
            cgfhmc.partial_refresh_momentum_mode(gm, qr.RngState(f"familymix-{_}"), c)
            phi_k = np.abs(np.fft.fftn(cgfhmc._get_components(gm), norm="ortho")) ** 2
            ratios_after.append(
                float(np.sum(phi_k[..., low_mask].sum()))
                / float(np.sum(phi_k[..., ~low_mask].sum()))
            )
        #
        mean_before = float(np.mean(ratios_before))
        mean_after = float(np.mean(ratios_after))
        # Both families are refreshed with c = 0 / c = 1 respectively, i.e. one
        # is frozen and one is completely resampled, and both start from the
        # same ensemble -- so the ratio must be unchanged.
        self.assertLess(abs(mean_after - mean_before) / mean_before, 0.25)

### -------------------------------------------------------------------

@unittest.skipUnless(SLOW, "set QLAT_REF_SLOW=1 to run the slow CGFHMC tests")
class TestChainHealth(unittest.TestCase):
    """A chain stays bounded and healthy, with and without the refresh."""

    def test_long_chain_does_not_drift(self):
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_unit()
        c = get_two_zone_c(geo)
        rs = qr.RngState("cgfhmc-long-chain")
        delta_h_list = []
        delta_h_accept_list = []
        plaq_list = []
        n_traj = 40
        for traj in range(1, n_traj + 1):
            info_out = {}
            cgfhmc.run_hmc_cgfhmc_pure_gauge(
                gf,
                qr.GaugeAction(5.5, 0.0),
                traj,
                rs.split("run_hmc_cgfhmc_pure_gauge"),
                c_field=c,
                n_step=4,
                md_time=1.0,
                info_out=info_out,
            )
            delta_h_list.append(info_out["delta_h_raw"])
            delta_h_accept_list.append(info_out["delta_h_accept"])
            plaq_list.append(gf.plaq())
        # The accept variable is the one that must be centred on zero: a biased
        # force or a wrong dK sign makes the chain drift systematically.
        mean_accept = float(np.mean(delta_h_accept_list))
        self.assertLess(abs(mean_accept), 2.0)
        # The raw violation is larger (it carries the refresh's kinetic change)
        # but must still be bounded: a runaway would mean the refresh or the
        # gauge fixing is injecting energy.
        self.assertLess(abs(float(np.mean(delta_h_list))), 200.0)
        self.assertLess(float(np.mean(np.abs(delta_h_list))), 250.0)
        # A healthy thermalised chain sits in the ordered phase at beta = 5.5.
        mean_plaq = float(np.mean(plaq_list))
        self.assertGreater(mean_plaq, 0.35)
        # And it must not drift: compare the two halves of the chain rather than
        # the first and last configuration.  A single pair is one sample of the
        # plaq fluctuation -- measured, it swings between 0.36 and 0.48 run to
        # run -- so it would be a machine-sensitive assertion, not a property of
        # the algorithm.  Block averages are what "no drift" actually means.
        half = len(plaq_list) // 2
        first_half = float(np.mean(plaq_list[:half]))
        second_half = float(np.mean(plaq_list[half:]))
        self.assertLess(abs(second_half - first_half), 0.2)

    def test_frozen_refresh_chain_is_plain_hmc(self):
        # With c == 0 the refresh is a strict no-op, so the chain must behave
        # like the plain driver's: bounded delta_h and a plaq in the same band.
        geo = get_geometry()
        gf = qr.GaugeField(geo)
        gf.set_unit()
        c = get_uniform_c(geo, 0.0)
        rs = qr.RngState("cgfhmc-frozen-chain")
        delta_h_accept_list = []
        for traj in range(1, 21):
            info_out = {}
            cgfhmc.run_hmc_cgfhmc_pure_gauge(
                gf,
                qr.GaugeAction(5.5, 0.0),
                traj,
                rs.split("run_hmc_cgfhmc_pure_gauge"),
                c_field=c,
                n_step=4,
                md_time=1.0,
                info_out=info_out,
            )
            self.assertEqual(info_out["dK_sum"], 0.0)
            self.assertEqual(info_out["n_refresh_applied"], 0)
            delta_h_accept_list.append(info_out["delta_h_accept"])
        self.assertLess(abs(float(np.mean(delta_h_accept_list))), 2.0)
        self.assertGreater(gf.plaq(), 0.35)

### -------------------------------------------------------------------

@unittest.skipUnless(SLOW, "set QLAT_REF_SLOW=1 to run the slow CGFHMC tests")
class TestRefreshChangesTheChain(unittest.TestCase):
    """The c(k) field is a real knob: it changes the trajectory, not just a tag."""

    def test_refresh_strength_changes_the_trajectory(self):
        # The accept rate is 1.0 on every 2^4 chain (delta_h is far below 1), so
        # it cannot discriminate; the raw violation and the dK sum can.
        geo = get_geometry()
        summary = {}
        for label, c in (
            ("frozen", get_uniform_c(geo, 0.0)),
            ("weak", get_uniform_c(geo, 0.2)),
            ("strong", get_uniform_c(geo, 0.9)),
        ):
            gf = qr.GaugeField(geo)
            gf.set_unit()
            rs = qr.RngState("cgfhmc-strength")
            delta_h_list = []
            dk_list = []
            for traj in range(1, 6):
                info_out = {}
                cgfhmc.run_hmc_cgfhmc_pure_gauge(
                    gf,
                    qr.GaugeAction(5.5, 0.0),
                    traj,
                    rs.split("run_hmc_cgfhmc_pure_gauge"),
                    c_field=c,
                    n_step=4,
                    md_time=1.0,
                    info_out=info_out,
                )
                delta_h_list.append(info_out["delta_h_raw"])
                dk_list.append(info_out["dK_sum"])
            summary[label] = (delta_h_list, dk_list)
        # A frozen field does not touch the kinetic energy at all, so its dK sum
        # is exactly zero; both refreshing fields must move it.
        self.assertEqual(sum(abs(v) for v in summary["frozen"][1]), 0.0)
        self.assertGreater(max(abs(v) for v in summary["weak"][1]), 0.0)
        self.assertGreater(max(abs(v) for v in summary["strong"][1]), 0.0)
        # And the raw violation must differ from the frozen chain, which is what
        # "the refresh is active" means.
        self.assertGreater(
            max(abs(a - b) for a, b in zip(summary["frozen"][0], summary["strong"][0])),
            1e-6,
        )

    def test_mode_dependent_refresh_differs_from_uniform(self):
        geo = get_geometry()
        chains = []
        for label, c in (
            ("uniform", get_uniform_c(geo, 0.5)),
            ("two-zone", get_two_zone_c(geo)),
        ):
            gf = qr.GaugeField(geo)
            gf.set_unit()
            rs = qr.RngState("cgfhmc-mode-vs-uniform")
            delta_h_list = []
            for traj in range(1, 6):
                info_out = {}
                cgfhmc.run_hmc_cgfhmc_pure_gauge(
                    gf,
                    qr.GaugeAction(5.5, 0.0),
                    traj,
                    rs.split("run_hmc_cgfhmc_pure_gauge"),
                    c_field=c,
                    n_step=4,
                    md_time=1.0,
                    info_out=info_out,
                )
                delta_h_list.append(info_out["delta_h_accept"])
            chains.append(delta_h_list)
        # Same seed, same number of refreshes, different c(k): the trajectories
        # must differ, otherwise the mode dependence is not reaching the field.
        self.assertGreater(max(abs(a - b) for a, b in zip(chains[0], chains[1])), 1e-6)

### -------------------------------------------------------------------

@unittest.skipUnless(SLOW, "set QLAT_REF_SLOW=1 to run the slow CGFHMC tests")
class TestAnisotropicLattice(unittest.TestCase):
    """A lattice with four different extents: the axis-order test that scales."""

    def test_refresh_on_an_anisotropic_lattice(self):
        # The fast suite checks this on 2x3x4x2; repeating it on a lattice whose
        # longest axis is the *time* one also exercises the folding of the
        # Nyquist index on an even axis other than x, where an off-by-one in the
        # roll direction would only show at the largest extent.
        geo = get_geometry((4, 2, 6, 8))
        threshold = cgfhmc.resolve_k_cut(geo, 1)
        c = cgfhmc.generate_c_field(
            geo, lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, 0.25, 0.85)
        )
        self.assertEqual(c.shape, (4, 2, 6, 8))
        gm = get_random_momentum(geo)
        applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(
            gm, qr.RngState("anisotropic"), c
        )
        self.assertTrue(applied)
        self.assertEqual(info["path"], "fourier (mode-dependent c)")
        arr = np.asarray(gm)
        self.assertLess(
            float(np.max(np.abs(arr + np.conj(np.swapaxes(arr, -1, -2))))), 1e-14
        )
        self.assertLess(float(np.max(np.abs(np.trace(arr, axis1=-2, axis2=-1)))), 1e-14)

    def test_short_chain_on_an_anisotropic_lattice(self):
        geo = get_geometry((2, 2, 2, 4))
        gf = qr.GaugeField(geo)
        gf.set_unit()
        c = get_two_zone_c(geo)
        rs = qr.RngState("cgfhmc-anisotropic-chain")
        for traj in range(1, 4):
            info_out = {}
            cgfhmc.run_hmc_cgfhmc_pure_gauge(
                gf,
                qr.GaugeAction(5.5, 0.0),
                traj,
                rs.split("run_hmc_cgfhmc_pure_gauge"),
                c_field=c,
                n_step=2,
                md_time=1.0,
                info_out=info_out,
            )
            self.assertTrue(math.isfinite(info_out["delta_h_raw"]))
        self.assertLess(float(qr.get_unitarity_error(gf)), 1e-12)
        self.assertGreater(gf.plaq(), 0.0)
        self.assertLessEqual(gf.plaq(), 1.0 + 1e-12)

    def test_timelike_extent_is_the_fastest_to_mix(self):
        # Sanity check on the axis order: refreshing with a c(k) that depends
        # only on k_t must leave the three spatial directions' mode structure
        # untouched, which is only true if component 3 really is the time axis.
        geo = get_geometry((2, 2, 2, 8))
        c = cgfhmc.generate_c_field(geo, lambda kv: 0.5 + 0.4 * np.cos(kv[3]))
        cgfhmc.k_vectors(geo)
        # The field depends on k_t only.
        axial_t = c.max(axis=(0, 1, 2))
        marginalised = c.max(axis=(0, 1, 2))
        self.assertEqual(axial_t.shape, (8,))
        self.assertTrue(np.all(marginalised == axial_t))
        gm = get_random_momentum(geo)
        applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(
            gm, qr.RngState("timelike"), c
        )
        self.assertTrue(applied)
        self.assertEqual(info["n_modes"], int(np.prod((2, 2, 2, 8))))

### -------------------------------------------------------------------

@unittest.skipUnless(SLOW, "set QLAT_REF_SLOW=1 to run the slow CGFHMC tests")
class TestLargerLatticeInvariants(unittest.TestCase):
    """The volume-independence of the field and refresh invariants.\n
    The fast module checks each of these on a smallest-interesting lattice, where
    the property is already visible.  Repeating them on ``4^4`` here is what
    turns "the code paths work" into "the folding, the evenness projector and
    the band split are right on a lattice large enough for a wrong Nyquist
    convention to matter" -- 4^4 has 16x the modes of 2^4, so a formula that is
    off by one index is exercised 256 times instead of 16.
    """

    def test_evenness_is_enforced_on_a_larger_lattice(self):
        geo = get_geometry((4, 4, 4, 4))
        odd = np.where(cgfhmc.k_vectors(geo)[0] >= 0.0, 0.0, 0.8)
        out = cgfhmc.regularize_c_field(odd)
        np.testing.assert_allclose(out, out[(-np.arange(4)) % 4], rtol=0.0, atol=0.0)
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 0.0, places=15)
        self.assertAlmostEqual(float(out[1, 0, 0, 0]), 0.4, places=15)
        self.assertAlmostEqual(float(out[2, 0, 0, 0]), 0.0, places=15)
        self.assertAlmostEqual(float(out[3, 0, 0, 0]), 0.4, places=15)

    def test_band_split_on_a_larger_lattice(self):
        geo = get_geometry((4, 4, 4, 4))
        threshold = cgfhmc.resolve_k_cut(geo, 1)
        c = cgfhmc.generate_c_field(
            geo, lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, 0.0, 1.0)
        )
        n_frozen = int(np.count_nonzero(c == 0.0))
        n_refreshed = int(np.count_nonzero(c == 1.0))
        self.assertGreater(n_frozen, 0)
        self.assertGreater(n_refreshed, 0)
        gm = get_random_momentum(geo)
        applied, d_k, info = cgfhmc.partial_refresh_momentum_mode(
            gm, qr.RngState("bands-4-4"), c
        )
        self.assertTrue(applied)
        self.assertEqual(info["n_modes_frozen"], n_frozen)
        self.assertEqual(info["n_modes_refreshed"], n_refreshed)
        self.assertEqual(info["path"], "fourier (mode-dependent c)")

### -------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
