#!/usr/bin/env python3

# Deterministic molecular-dynamics commutation test for the gauge-transform
# action on the HMC momentum (``q.gm_apply_gauge_transformation`` and the
# ``gt * gm`` sugar), on a pure-gauge field.
#
# The two branches use the SAME integrator and the SAME gauge action and differ
# only in the order of the gauge transformation and the trajectory:
#
#   branch 1: evolve (gf, gm) for ``n_step`` steps, then apply ``Ad_gt`` to
#             both the links and the momentum;
#   branch 2: apply ``Ad_gt`` to (gf, gm) first, then evolve for ``n_step``
#             steps.
#
# A correct adjoint action gives branch agreement to round-off and the same
# ``delta_h`` in both branches.  ``gt = 1``, a wrong ``is_dagger`` or a
# shifted-site law all make the comparison fail, so the test additionally
# asserts that ``gt`` is far from the identity and that the fields move.
# ``run_hmc_evolve_pure_gauge`` applies no Metropolis step.
#
# The residuals are ~1e-15 and are NOT reproducible to their own relative
# round-off, so they are not recorded as ``.log.json`` floats
# (``q.check_log_json`` compares relatively): only PASS/FAIL flags are
# recorded, and the residual numbers appear in one human-readable diagnostic
# line.  The bounds carry ~3 orders of margin (measured at 2^4, one node,
# <= 64 steps); if a larger-lattice run ever exceeds them, switch to a
# hand-rolled leapfrog rather than loosening the bound blindly.
#
# Run through the examples-py test suite / nixpkgs/run-one-example-py.py
# (target hmc-gauge-commute.log).

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "false"

import numpy as np
from mpi4py import MPI

import qlat as q

size_node_list = [
    [1, 1, 1, 1],
    [1, 1, 1, 2],
    [1, 1, 1, 4],
    [2, 2, 2, 2],
]

TOTAL_SITE = [2, 2, 2, 2]
N_STEP = 8
MD_TIME = 1.0

BOUND_FIELD = 1.0e-12
BOUND_MOMENTUM = 1.0e-12
BOUND_DELTA_H = 1.0e-10
BOUND_INVERSE = 1.0e-12
BOUND_HAMILTON = 1.0e-12
BOUND_ANTI_HERMITIAN = 1.0e-12
BOUND_SUGAR = 1.0e-14
BOUND_GT_TRIVIAL = 0.1
BOUND_MOVED = 1.0e-3

def get_glb_max(x):
    # Global max over the nodes of max|...|; a max is exact, so the recorded
    # boolean thresholds do not depend on the node decomposition.
    local_max = float(np.max(np.abs(x)))
    comm = q.get_comm()
    return float(comm.allreduce(local_max, op=MPI.MAX))

def get_max_abs_diff(f0, f1):
    # Global max |f0 - f1| over the locally owned sites.
    return get_glb_max(np.asarray(f0) - np.asarray(f1))

def get_hamilton(gm, gf, ga):
    # Global Hamiltonian H = sum_x Tr(p^2)/2 + S_gauge, summed over the nodes.
    h_node = float(q.gm_hamilton_node(gm)) + float(q.gf_hamilton_node(gf, ga))
    return float(q.glb_sum(h_node))

def get_rel_diff(h0, h1):
    return abs(h0 - h1) / max(abs(h0), abs(h1), 1.0e-300)

def get_max_anti_hermitian_traceless_dev(gm):
    # Max deviation of the momentum entries from anti-Hermitian and traceless.
    arr = np.asarray(gm)
    dev_anti_hermitian = get_glb_max(arr + np.conj(np.swapaxes(arr, -1, -2)))
    dev_traceless = get_glb_max(np.trace(arr, axis1=-2, axis2=-1))
    return max(dev_anti_hermitian, dev_traceless)

def check_commute(fname):
    geo = q.Geometry(q.Coordinate(TOTAL_SITE))
    rs = q.RngState("hmc-gauge-commute")
    gf = q.GaugeField(geo)
    gf.set_rand(rs.split("gf"), 0.1, 2)
    gf.unitarize()
    gm = q.GaugeMomentum(geo)
    gm.set_rand(rs.split("gm"), 1.0)
    gt = q.GaugeTransform(geo)
    gt.set_rand(rs.split("gt"), 0.5, 2)
    gt.unitarize()
    ga = q.GaugeAction(2.13, -0.331)
    gf_before = np.asarray(gf).copy()
    gm_before = np.asarray(gm).copy()
    # Non-vacuity: the transform must be far from the identity (a trivial gt
    # makes any commutation test pass) and the links must actually move.
    dev_gt = get_glb_max(np.asarray(gt)[:, 0] - np.eye(3))
    # ---- branch 1: evolve, then transform both fields.
    gf1 = gf.copy()
    gm1 = gm.copy()
    d1 = q.run_hmc_evolve_pure_gauge(gm1, gf1, ga, N_STEP, MD_TIME)
    dev_moved = get_max_abs_diff(gf1, gf)
    gf1_t = gt * gf1
    gm1_t = gt * gm1
    # ---- branch 2: transform both fields, then evolve.
    gf2 = gt * gf
    gm2 = gt * gm
    # The initial Hamiltonian is gauge invariant: measure it before branch 2
    # evolves the transformed pair in place.
    h_orig = get_hamilton(gm, gf, ga)
    h_trans = get_hamilton(gm2, gf2, ga)
    d2 = q.run_hmc_evolve_pure_gauge(gm2, gf2, ga, N_STEP, MD_TIME)
    # ---- branch agreement.
    max_dgf = get_max_abs_diff(gf2, gf1_t)
    max_dgm = get_max_abs_diff(gm2, gm1_t)
    max_ddelta_h = abs(d1 - d2)
    # ---- the sugar and the explicit function are the same transform: both must
    # agree, and neither may modify its input momentum.
    gm_t_explicit = q.gm_apply_gauge_transformation(gm, gt)
    max_dsugar = get_max_abs_diff(gm_t_explicit, gt * gm)
    max_dsugar_t = get_max_abs_diff(q.gm_apply_gauge_transformation(gm1, gt), gt * gm1)
    max_dgm_in = get_max_abs_diff(gm, gm_before)
    # ---- inverse property: is_dagger=True undoes is_dagger=False.
    gm_inv = q.gm_apply_gauge_transformation(gm_t_explicit, gt, is_dagger=True)
    max_dinv = get_max_abs_diff(gm_inv, gm)
    # ---- Ad_gt preserves anti-Hermiticity and tracelessness of the momentum.
    dev_ah = max(
        get_max_anti_hermitian_traceless_dev(gm1_t),
        get_max_anti_hermitian_traceless_dev(gm2),
    )
    # ---- Hamiltonian invariance: H(gf, gm) is gauge invariant, so the
    # transformed pair starts at the same H and both branches end at the same H.
    h1 = get_hamilton(gm1_t, gf1_t, ga)
    h2 = get_hamilton(gm2, gf2, ga)
    max_dh = max(get_rel_diff(h_orig, h_trans), get_rel_diff(h1, h2))
    # ---- assertions on exactly the bounds recorded below.
    assert max_dgf < BOUND_FIELD
    assert max_dgm < BOUND_MOMENTUM
    assert max_ddelta_h < BOUND_DELTA_H
    assert max_dinv < BOUND_INVERSE
    assert max_dsugar < BOUND_SUGAR
    assert max_dsugar_t < BOUND_SUGAR
    assert max_dgm_in == 0.0
    assert np.array_equal(np.asarray(gf), gf_before)
    assert dev_ah < BOUND_ANTI_HERMITIAN
    assert max_dh < BOUND_HAMILTON
    assert dev_gt > BOUND_GT_TRIVIAL
    assert dev_moved > BOUND_MOVED
    q.displayln_info(
        f"{fname}: residuals: dgf = {max_dgf:.3e}; dgm = {max_dgm:.3e};"
        f" ddelta_h = {max_ddelta_h:.3e}; dinv = {max_dinv:.3e};"
        f" dsugar = {max(max_dsugar, max_dsugar_t):.3e};"
        f" dH/H = {max_dh:.3e}; anti-hermitian dev = {dev_ah:.3e};"
        f" gt non-triviality = {dev_gt:.3e}; field movement = {dev_moved:.3e}"
    )
    q.displayln_info(
        f"{fname}: delta_h: branch1 = {d1:.15e}; branch2 = {d2:.15e}; H = {h1:.15e}"
    )
    # Non-float outcomes are encoded in the recorded name, never as a
    # fabricated float (AGENTS.md): ``check_log_json`` compares relatively, so a
    # ~1e-15 residual is not reproducible against a committed reference float.
    # Only PASS/FAIL flags are recorded; the residual numbers stay in the
    # diagnostic line above.
    q.json_results_append(f"hmc-gauge-commute: links = {max_dgf < BOUND_FIELD}")
    q.json_results_append(f"hmc-gauge-commute: momentum = {max_dgm < BOUND_MOMENTUM}")
    q.json_results_append(
        f"hmc-gauge-commute: delta_h = {max_ddelta_h < BOUND_DELTA_H}"
    )
    q.json_results_append(f"hmc-gauge-commute: inverse = {max_dinv < BOUND_INVERSE}")
    q.json_results_append(
        f"hmc-gauge-commute: anti_hermitian = {dev_ah < BOUND_ANTI_HERMITIAN}"
    )
    q.json_results_append(f"hmc-gauge-commute: hamilton = {max_dh < BOUND_HAMILTON}")
    q.json_results_append(
        f"hmc-gauge-commute: sugar = {max(max_dsugar, max_dsugar_t) < BOUND_SUGAR}"
    )
    q.json_results_append(
        f"hmc-gauge-commute: momentum input unchanged = {max_dgm_in == 0.0}"
    )
    q.json_results_append(
        f"hmc-gauge-commute: gt is non-trivial = {dev_gt > BOUND_GT_TRIVIAL}"
    )
    q.json_results_append(
        f"hmc-gauge-commute: fields moved = {dev_moved > BOUND_MOVED}"
    )

def main():
    fname = q.get_fname()
    q.begin_with_mpi(size_node_list)
    check_commute(fname)
    q.timer_display()
    if q.is_test():
        q.check_log_json(__file__, check_eps=1.0e-10)
    q.end_with_mpi()
    q.displayln_info("CHECK: finished successfully.")

if __name__ == "__main__":
    main()
