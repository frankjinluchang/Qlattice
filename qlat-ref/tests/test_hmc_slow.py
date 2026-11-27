"""
Opt-in slow HMC scans (``QLAT_REF_SLOW=1``).\n
The trimmed suite (the default ``unittest discover`` run, and the nix install
check) keeps a few short chains; the physics scans that need many trajectories,
a larger lattice or the expensive rectangle-improved force live here.  They are
skipped unless the environment variable is set, so CI stays fast while the full
coverage remains one command away::
    QLAT_REF_SLOW=1 python3 -m unittest discover -s tests -t . -v
The guard is the same in every class, and the module is discovered either way:
an unset variable must skip, not fail.\n
Runtime (single core, numpy only): about 3-4 minutes, dominated by the
``4^4`` runs and the 200-trajectory always-accept chain.
"""

import math
import os
import unittest

import numpy as np

import qlat_ref as qr
from qlat_ref import hmc
from qlat_ref import hmc_utils
from qlat_ref.field import get_array_from_field

SLOW = os.environ.get("QLAT_REF_SLOW", "") not in ("", "0")

TOTAL_SITE = (2, 2, 2, 2)
N_STEP = 6

### -------------------------------------------------------------------

def get_geometry(total_site=TOTAL_SITE):
    """Return a geometry (helper)."""
    return qr.Geometry(qr.Coordinate(list(total_site)))

def get_unit_field(total_site=TOTAL_SITE):
    """Return the unit gauge field (helper)."""
    gf = qr.GaugeField(get_geometry(total_site))
    gf.set_unit()
    return gf

def get_max_difference(a, b):
    """Return ``max |a - b|`` over two equally shaped arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

def run_chain(ga, n_traj, seed, *, n_step=N_STEP, is_always_accept=False):
    """Run a chain from the unit field and return ``(gf, flags, delta_hs, plaqs)``."""
    gf = get_unit_field()
    rs = qr.RngState(seed)
    flags = []
    delta_hs = []
    plaqs = []
    for traj in range(1, n_traj + 1):
        flag, delta_h = hmc_utils.run_hmc_pure_gauge(
            gf,
            ga,
            traj,
            rs.split("run_hmc_pure_gauge"),
            n_step=n_step,
            is_always_accept=is_always_accept,
        )
        flags.append(flag)
        delta_hs.append(delta_h)
        plaqs.append(gf.plaq())
    return gf, flags, delta_hs, plaqs

### -------------------------------------------------------------------

@unittest.skipUnless(SLOW, "set QLAT_REF_SLOW=1 to run the slow HMC scans")
class TestBetaScan(unittest.TestCase):
    """The full acceptance / plaquette table over a range of couplings."""

    # 30 trajectories per beta on 2^4, n_step=6, md_time=1, from the unit field.
    # The plaquette is the average of the last 10 trajectories; the values are
    # the reference measurement of the design report and serve as a regression
    # band (30 trajectories on 2^4 have large autocorrelation, so this is a
    # health check, not a precision measurement).
    EXPECTED = {
        (1.0, 0.0): 0.0598,
        (2.0, 0.0): 0.1454,
        (3.0, 0.0): 0.2048,
        (5.5, 0.0): 0.5593,
        (6.0, 0.0): 0.5980,
        (2.13, -0.331): 0.5966,
    }

    def test_plaquette_table(self):
        for (beta, c1), expected in self.EXPECTED.items():
            ga = qr.GaugeAction(beta, c1)
            gf, flags, delta_hs, plaqs = run_chain(ga, 30, seed=f"scan-{beta}-{c1}")
            accepted = sum(1 for flag in flags if flag)
            mean_plaq = float(np.mean(plaqs[-10:]))
            # the c1 != 0 chain loses a trajectory now and then; the Wilson
            # chains are accepted essentially always at this step size
            self.assertGreaterEqual(accepted, 25, msg=f"beta={beta} c1={c1}")
            self.assertLessEqual(
                abs(mean_plaq - expected),
                0.06,
                msg=f"beta={beta} c1={c1}: plaq={mean_plaq} vs {expected}",
            )
            self.assertLessEqual(
                qr.mat.get_unitarity_error(get_array_from_field(gf)), 1e-12
            )

    def test_plaquette_increases_with_beta(self):
        plaquettes = {}
        for beta in (1.0, 2.0, 3.0, 5.5, 6.0):
            gf, _, _, plaqs = run_chain(
                qr.GaugeAction(beta, 0.0), 20, seed=f"ordering-{beta}"
            )
            plaquettes[beta] = gf.plaq()
        ordered = [plaquettes[beta] for beta in (1.0, 2.0, 3.0, 5.5, 6.0)]
        for previous, current in zip(ordered, ordered[1:]):
            self.assertLess(previous, current)

@unittest.skipUnless(SLOW, "set QLAT_REF_SLOW=1 to run the slow HMC scans")
class TestLongAlwaysAcceptChain(unittest.TestCase):
    """The drift gate: a long chain *without* Metropolis must not blow up."""

    def test_200_trajectory_chain(self):
        ga = qr.GaugeAction(5.5, 0.0)
        gf, flags, delta_hs, plaqs = run_chain(
            ga, 200, seed="always-accept-200", is_always_accept=True
        )
        mean_delta_h = float(np.mean(delta_hs))
        mean_abs_delta_h = float(np.mean(np.abs(delta_hs)))
        # the integrator error is unbiased, so the chain drifts only slowly
        self.assertLess(abs(mean_delta_h), 1e-2)
        self.assertLess(mean_abs_delta_h, 5e-2)
        self.assertGreater(gf.plaq(), 0.3)
        self.assertLessEqual(
            qr.mat.get_unitarity_error(get_array_from_field(gf)), 1e-12
        )
        self.assertFalse(any(math.isnan(delta_h) for delta_h in delta_hs))
        self.assertFalse(any(math.isnan(plaq) for plaq in plaqs))

@unittest.skipUnless(SLOW, "set QLAT_REF_SLOW=1 to run the slow HMC scans")
class TestStepCountScan(unittest.TestCase):
    """``|delta_h|`` over a wide range of step counts: the O(dt^4) law."""

    def test_delta_h_scan(self):
        ga = qr.GaugeAction(5.5, 0.0)
        gf = qr.GaugeField(get_geometry())
        gf.set_rand(qr.RngState("scan-field"), sigma=0.4, n_step=2)
        gm_seed = "scan-momenta"
        violations = {}
        for n_step in (2, 4, 6, 8, 12, 16, 24):
            values = []
            for k in range(5):
                gm = qr.GaugeMomentum(gf.geo)
                gm.set_rand(qr.RngState(f"{gm_seed}-{k}"), 1.0)
                gf_run = gf.copy()
                delta_h = hmc_utils.run_hmc_evolve_pure_gauge(gm, gf_run, ga, n_step)
                values.append(abs(delta_h))
            violations[n_step] = float(np.mean(values))
        # monotone down the scan (the O(dt^4) trajectory error)
        steps = sorted(violations)
        for smaller, larger in zip(steps, steps[1:]):
            self.assertLess(violations[larger], violations[smaller])
        # and the quartic law between 4 and 8 steps
        self.assertGreater(violations[4] / violations[8], 8.0)

@unittest.skipUnless(SLOW, "set QLAT_REF_SLOW=1 to run the slow HMC scans")
class TestLargerLattice(unittest.TestCase):
    """A ``4^4`` trajectory: the vectorised kernels, away from V = 16."""

    def test_4_to_the_4_chain(self):
        total_site = (4, 4, 4, 4)
        ga = qr.GaugeAction(5.5, 0.0)
        gf = get_unit_field(total_site)
        gf.set_rand(qr.RngState("large-lattice-start"), sigma=0.5, n_step=1)
        rs = qr.RngState("large-lattice")
        for traj in range(1, 4):
            flag, delta_h = hmc_utils.run_hmc_pure_gauge(
                gf, ga, traj, rs.split("run_hmc_pure_gauge")
            )
            self.assertFalse(math.isnan(delta_h))
            self.assertGreater(gf.plaq(), 0.0)
            self.assertLessEqual(gf.plaq(), 1.0)
        self.assertLessEqual(
            qr.mat.get_unitarity_error(get_array_from_field(gf)), 1e-12
        )
        # the momentum initialisation is the only sequential part of the port,
        # so a larger lattice is where its cost would show up; a silent
        # degeneration to O(V^2) would still pass the physics checks
        gm = qr.GaugeMomentum(gf.geo)
        gm.set_rand(qr.RngState("large-lattice-gm"), 1.0)
        self.assertGreater(hmc.gm_hamilton_node(gm), 0.0)

### -------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
