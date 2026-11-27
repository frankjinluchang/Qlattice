#!/usr/bin/env python3

# Plain local (no Fourier acceleration) Landau and Coulomb gauge fixing test
# of the public qlat API (``qlat.gauge_fixing``), run with 2 MPI processes on
# a distributed gauge field.
#
# The fixing functions do not modify their input: they return ``(gt, info)``
# and the gauge-fixed field is ``gt * gf``.  This test therefore checks the
# fixed field through ``gt * gf`` and asserts explicitly that the input field
# is unchanged (bit-for-bit, plus its plaquette and gauge functional).
#
# Run through the examples-py test suite / nixpkgs/run-one-example-py.py
# (target gauge-fixing.log).

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "false"

import numpy as np

import qlat as q

size_node_list = [
    [1, 1, 1, 1],
    [1, 1, 1, 2],
    [1, 1, 1, 4],
    [2, 2, 2, 2],
]

TOTAL_SITE = [4, 4, 4, 4]
EPS = 1.0e-8
ALPHA = 0.1
MAX_ITER = 5000

def show_info(fname, gf, tag):
    plaq = gf.plaq()
    link_trace = gf.link_trace()
    q.displayln_info(
        f"{fname}: {tag}: plaq = {plaq:.15e}; link_trace = {link_trace:.15e}"
    )

def get_max_abs_diff(arr0, arr1):
    return float(np.max(np.abs(np.asarray(arr0) - np.asarray(arr1))))

def check_input_not_modified(
    fname, tag, gf, gf_arr_before, plaq_before, functional_before, dirs
):
    # The ``fix_*`` functions are documented to leave the input field untouched:
    # the fixed field is ``gt * gf`` and ``gf`` itself must not change.
    err_field = get_max_abs_diff(gf, gf_arr_before)
    err_plaq = abs(gf.plaq() - plaq_before)
    err_functional = abs(q.get_gauge_functional(gf, dirs) - functional_before)
    q.displayln_info(
        f"{fname}: {tag}: input unchanged: max|dU| = {err_field:.3e};"
        f" dplaq = {err_plaq:.3e}; dF = {err_functional:.3e}"
    )
    assert err_field == 0.0
    assert err_plaq == 0.0
    assert err_functional == 0.0

def check_landau_gauge_fixing(fname, geo, rs):
    gf = q.GaugeField(geo)
    gf.set_rand(rs.split("gf-landau"), 0.5, 1)
    show_info(fname, gf, "landau before")
    gf_arr_before = np.asarray(gf).copy()
    plaq_before = gf.plaq()
    functional_before = q.get_gauge_functional(gf, [0, 1, 2, 3])
    divergence_before = q.get_gauge_divergence(gf, [0, 1, 2, 3])
    gt, info = q.fix_landau_gauge(gf, max_iter=MAX_ITER, eps=EPS, alpha=ALPHA)
    gf_fixed = gt * gf
    show_info(fname, gf_fixed, "landau after")
    plaq_after = gf_fixed.plaq()
    unitarity_error = q.get_unitarity_error(gf_fixed)
    functional_after = q.get_gauge_functional(gf_fixed, [0, 1, 2, 3])
    q.displayln_info(
        f"{fname}: landau: functional {functional_before:.15e} ->"
        f" {info['functional']:.15e} in {info['n_iter']} sweeps;"
        f" divergence {divergence_before:.3e} -> {info['divergence']:.3e};"
        f" unitarity error = {unitarity_error:.3e}"
    )
    assert info["gauge"] == "landau"
    assert info["converged"]
    assert info["divergence"] <= EPS
    assert info["functional"] > functional_before
    # ``gt * gf`` must reproduce the field the sweeps actually fixed.
    assert abs(functional_after - info["functional"]) < 1.0e-12
    assert abs(plaq_after - plaq_before) < 1.0e-10
    assert unitarity_error < 1.0e-10
    check_input_not_modified(
        fname, "landau", gf, gf_arr_before, plaq_before, functional_before, [0, 1, 2, 3]
    )
    q.json_results_append(f"{fname}: landau plaq before", plaq_before, 1.0e-10)
    q.json_results_append(f"{fname}: landau plaq after", plaq_after, 1.0e-10)
    q.json_results_append(
        f"{fname}: landau functional before", functional_before, 1.0e-10
    )
    q.json_results_append(f"{fname}: landau functional", info["functional"], 1.0e-10)

def check_coulomb_gauge_fixing(fname, geo, rs):
    gf = q.GaugeField(geo)
    gf.set_rand(rs.split("gf-coulomb"), 0.5, 1)
    show_info(fname, gf, "coulomb before")
    gf_arr_before = np.asarray(gf).copy()
    plaq_before = gf.plaq()
    functional_before = q.get_gauge_functional(gf, [1, 2, 3])
    divergence_before = q.get_gauge_divergence(gf, [1, 2, 3])
    gt, info = q.fix_coulomb_gauge(gf, max_iter=MAX_ITER, eps=EPS, alpha=ALPHA)
    gf_fixed = gt * gf
    show_info(fname, gf_fixed, "coulomb after")
    plaq_after = gf_fixed.plaq()
    unitarity_error = q.get_unitarity_error(gf_fixed)
    functional_after = q.get_gauge_functional(gf_fixed, [1, 2, 3])
    q.displayln_info(
        f"{fname}: coulomb: functional {functional_before:.15e} ->"
        f" {info['functional']:.15e} in {info['n_iter']} sweeps;"
        f" divergence {divergence_before:.3e} -> {info['divergence']:.3e};"
        f" unitarity error = {unitarity_error:.3e}"
    )
    assert info["gauge"] == "coulomb"
    assert info["converged"]
    assert info["divergence"] <= EPS
    assert info["functional"] > functional_before
    # ``gt * gf`` must reproduce the field the sweeps actually fixed.
    assert abs(functional_after - info["functional"]) < 1.0e-12
    assert abs(plaq_after - plaq_before) < 1.0e-10
    assert unitarity_error < 1.0e-10
    check_input_not_modified(
        fname, "coulomb", gf, gf_arr_before, plaq_before, functional_before, [1, 2, 3]
    )
    q.json_results_append(f"{fname}: coulomb plaq before", plaq_before, 1.0e-10)
    q.json_results_append(f"{fname}: coulomb plaq after", plaq_after, 1.0e-10)
    q.json_results_append(
        f"{fname}: coulomb functional before", functional_before, 1.0e-10
    )
    q.json_results_append(f"{fname}: coulomb functional", info["functional"], 1.0e-10)

def check_distributed_field(fname, geo):
    gf = q.GaugeField(geo)
    rs = q.RngState("gauge-fixing-distributed")
    gf.set_rand(rs.split("gf"), 0.5, 1)
    gf_arr_before = np.asarray(gf).copy()
    plaq_before = gf.plaq()
    functional_before = q.get_gauge_functional(gf, [0, 1, 2, 3])
    gt, info = q.fix_landau_gauge(gf, max_iter=MAX_ITER, eps=EPS, alpha=ALPHA)
    assert info["converged"]
    gf_fixed = gt * gf
    assert q.get_gauge_divergence(gf_fixed, [0, 1, 2, 3]) <= EPS
    # Halo/boundary guard: the fixed field's functional must agree with the
    # functional accumulated by the C++ sweeps on the expanded working field.
    # A stale halo or a missing boundary refresh shows up here on 2 nodes.
    assert (
        abs(q.get_gauge_functional(gf_fixed, [0, 1, 2, 3]) - info["functional"])
        < 1.0e-12
    )
    check_input_not_modified(
        fname,
        "distributed",
        gf,
        gf_arr_before,
        plaq_before,
        functional_before,
        [0, 1, 2, 3],
    )
    q.displayln_info(
        f"{fname}: distributed: id_node = {q.get_id_node()};"
        f" num_node = {q.get_num_node()}; local_volume = {geo.local_volume};"
        f" total_volume = {geo.total_volume}"
    )
    q.json_results_append(f"{fname}: num_node = {q.get_num_node()}")
    q.json_results_append(f"{fname}: local_volume = {geo.local_volume}")
    q.json_results_append(f"{fname}: total_volume = {geo.total_volume}")

def check_unitarity_error_accepted_types(fname, geo, rs):
    # `get_unitarity_error` is defined by the matrix content alone, so it must
    # accept a GaugeTransform as well as a GaugeField (qlat-issue-11), and it
    # must reject the types where the number would be meaningless.
    scale = 1.01
    gf = q.GaugeField(geo)
    gf.set_unit()
    # A transform and a gauge field built from the same explicit matrices: the
    # error is analytic, so the check does not depend on the RNG at all.
    gt_scaled = q.GaugeTransform(geo)
    np.asarray(gt_scaled)[...] = scale * np.eye(3, dtype=np.complex128)
    gf_scaled = q.GaugeField(geo)
    arr_gf_scaled = np.asarray(gf_scaled)
    arr_gt_scaled = np.asarray(gt_scaled)
    for mu in range(4):
        arr_gf_scaled[:, mu] = arr_gt_scaled[:, 0]
    err_gf_unit = q.get_unitarity_error(gf)
    err_gt_scaled = q.get_unitarity_error(gt_scaled)
    err_gf_scaled = q.get_unitarity_error(gf_scaled)
    q.displayln_info(
        f"{fname}: unitarity: GaugeField(unit) = {err_gf_unit:.3e};"
        f" GaugeTransform({scale} * 1) = {err_gt_scaled:.3e};"
        f" GaugeField(same matrices) = {err_gf_scaled:.3e}"
    )
    # An exactly unitary field has exactly zero error, and the returned value
    # is a plain float (not an array).
    assert err_gf_unit == 0.0
    assert isinstance(err_gt_scaled, float)
    # |c^2 - 1| for a uniformly scaled unit matrix, to round-off.
    assert abs(err_gt_scaled - abs(scale**2 - 1.0)) < 1.0e-14
    assert err_gt_scaled > 1.0e-2
    # Same content -> bit-identical value: the two functions differ only in the
    # field type they accept, not in the arithmetic they perform.
    assert err_gf_scaled == err_gt_scaled
    # A random near-identity transform is unitary to round-off, like a gauge
    # field, and so is a gauge-fixed field and its accumulated transform.
    gt_rand = q.GaugeTransform(geo)
    gt_rand.set_rand(rs.split("gt-unitarity"), 0.5, 1)
    err_gt_rand = q.get_unitarity_error(gt_rand)
    gf_rand = q.GaugeField(geo)
    gf_rand.set_rand(rs.split("gf-unitarity"), 0.5, 1)
    gt_fix, info_fix = q.fix_landau_gauge(
        gf_rand, max_iter=MAX_ITER, eps=EPS, alpha=ALPHA
    )
    assert info_fix["converged"]
    err_fixed_field = q.get_unitarity_error(gt_fix * gf_rand)
    err_fixed_transform = q.get_unitarity_error(gt_fix)
    q.displayln_info(
        f"{fname}: unitarity: random transform = {err_gt_rand:.3e};"
        f" fixed field = {err_fixed_field:.3e};"
        f" fixed transform = {err_fixed_transform:.3e}"
    )
    assert err_gt_rand <= 1.0e-10
    assert err_fixed_field <= 1.0e-10
    assert err_fixed_transform <= 1.0e-10
    # The meaningless types are rejected with a clear TypeError.
    gm = q.GaugeMomentum(geo)
    rejected = 0
    for obj in (gm, np.asarray(gf_rand), np.eye(3)):
        try:
            q.get_unitarity_error(obj)
        except TypeError:
            rejected += 1
    assert rejected == 3
    q.json_results_append(f"{fname}: unitarity GaugeField(unit)", err_gf_unit, 0.0)
    q.json_results_append(f"{fname}: unitarity GaugeTransform", err_gt_scaled, 1.0e-10)
    q.json_results_append(
        f"{fname}: unitarity GaugeField(same matrices)", err_gf_scaled, 1.0e-10
    )
    # These three are round-off level (1e-15), so they cannot be compared with a
    # relative ``eps``: a last-bit difference is a large *relative* difference.
    # The bounds are already asserted above, so record the outcome, not the
    # value (AGENTS.md: only floating-point results go through the eps form).
    q.json_results_append(
        f"{fname}: unitarity random transform <= 1e-10 = {err_gt_rand <= 1.0e-10}"
    )
    q.json_results_append(
        f"{fname}: unitarity fixed field <= 1e-10 = {err_fixed_field <= 1.0e-10}"
    )
    q.json_results_append(
        f"{fname}: unitarity fixed transform <= 1e-10 = "
        f"{err_fixed_transform <= 1.0e-10}"
    )
    q.json_results_append(f"{fname}: unitarity rejected types = {rejected}")

def main():
    fname = q.get_fname()
    q.begin_with_mpi(size_node_list)
    total_site = q.Coordinate(TOTAL_SITE)
    geo = q.Geometry(total_site)
    q.json_results_append(f"{fname}: geo = {geo.show()}")
    rs = q.RngState("gauge-fixing")
    check_landau_gauge_fixing(fname, geo, rs)
    check_coulomb_gauge_fixing(fname, geo, rs)
    check_distributed_field(fname, geo)
    check_unitarity_error_accepted_types(fname, geo, rs)
    q.timer_display()
    if q.is_test():
        q.check_log_json(__file__, check_eps=1.0e-10)
    q.end_with_mpi()
    q.displayln_info("CHECK: finished successfully.")

if __name__ == "__main__":
    main()
