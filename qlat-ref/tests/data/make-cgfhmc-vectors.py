#!/usr/bin/env python3

"""
Regenerate ``cgfhmc-vectors.json`` from the live C++ ``qlat`` build.\n
Not part of the installed package and not run by the test suite: the committed
JSON is the reference.  This script exists so the vectors can be regenerated
deliberately (e.g. after an intentional change in ``qlat``) and so the
provenance of every value is reproducible.\n
Usage (from the repository root)::
    source result-py-local/bin/setenv-qlat.sh
    python3 qlat-ref/tests/data/make-cgfhmc-vectors.py
It writes ``cgfhmc-vectors.json`` next to itself.  Every trajectory is generated
twice -- once through the live ``qlat`` build and once through
:mod:`qlat_ref` -- and the script fails unless the two agree, so a wrong
expectation cannot be committed.\n
What the vectors pin: the whole CGFHMC trajectory, i.e. the RNG tree
(``split(str(traj))`` then ``"set_rand_gauge_momentum"``,
``"partial_refresh_momentum_base"`` and ``"metropolis_accept"``), the momentum
draw, the force-gradient integrator, **the insertion point of the per-step
Landau fixing and partial refresh**, the ``dK`` accounting and the Metropolis
step.  The insertion point is why ``n_step`` is 4 here and not 1: a driver that
hoists an integrator sub-update out of the step loop agrees with the correct one
for a single step and diverges from the second step on, so a one-step vector
could not catch it.\n
The ``field`` of each trajectory is the configuration **after** that
trajectory, i.e. ``gf`` as the driver leaves it: the proposal on acceptance and
unchanged on rejection.  That is the unambiguous quantity the public driver
exposes -- a rejected trajectory deliberately shows the previous field, so the
``flag`` sequence is visible in the field sequence too.  Storing the *proposal*
instead would require an ``is_always_accept=True`` replay, which is a different
chain (the accepted field feeds the next trajectory's momentum), so the field
and the recorded ``flag``/``plaq`` would then describe two different chains.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np

import qlat as q

import qlat_ref as qr

from qlat_ref import cgfhmc as cg_ref

# Tolerances used only by the cross-check below (the test suite has its own).
# The integrands are the same operations in the same order on both sides, so the
# measured difference is a handful of ULP of an O(V) double; the field after
# several matrix products accumulates a little more.
EPS_DELTA_H = 1e-9
EPS_DK = 1e-7
EPS_PLAQ = 1e-9
EPS_FIELD = 1e-12

# (name, total_site, beta, c1, n_step, md_time, c-field kind, uniform value,
#  seed, n_traj).  The two refresh paths are covered: ``uniform`` exercises the
# real-space fast path and ``twozone`` the Fourier path.
CASES = (
    (
        "beta5p5_uniform",
        (2, 2, 2, 2),
        5.5,
        0.0,
        4,
        1.0,
        "uniform",
        0.5,
        "cgfhmc-vectors-uniform",
        3,
    ),
    (
        "beta5p5_twozone",
        (2, 2, 2, 2),
        5.5,
        0.0,
        4,
        1.0,
        "twozone",
        None,
        "cgfhmc-vectors-twozone",
        3,
    ),
    (
        "beta2p13_rect_uniform",
        (2, 2, 2, 2),
        2.13,
        -0.331,
        4,
        1.0,
        "uniform",
        0.3,
        "cgfhmc-vectors-rect",
        2,
    ),
)

### --- c(k) fields -------------------------------------------------------

def get_c_func(total_site, kind, uniform):
    """Return the ``c(k)`` callable of a case (helper).\n
    The ``twozone`` profile is the one ``test_cgfhmc.get_two_zone_c`` rebuilds
    from the recorded ``c_field_kind``: a threshold at one unit of ``2*pi/Lx``,
    with ``c = 0.35`` below it and ``c = 0.9`` above.
    """
    if kind == "uniform":
        return lambda kv: uniform
    if kind == "twozone":
        threshold = 2.0 * np.pi / float(total_site[0])
        return lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, 0.35, 0.9)
    raise AssertionError(f"unknown c_field_kind {kind!r}")

### --- the live qlat driver ---------------------------------------------

def get_qlat_trajectories(
    total_site, beta, c1, n_step, md_time, kind, uniform, seed, n_traj
):
    """Run the chain through the live ``qlat`` driver (helper).\n
    Only the driver is used -- the momentum draw, the field evolution, the
    fixing, the refresh and the Metropolis step all happen inside it -- so the
    vectors pin the shipped code path rather than a re-derivation of it.
    """
    geo = q.Geometry(q.Coordinate(list(total_site)))
    ga = q.GaugeAction(beta, c1)
    c_field = q.generate_c_field(geo, get_c_func(total_site, kind, uniform))
    # The driver splits ``str(traj)`` off the state it is given, so the caller
    # sits one level above it -- the same tree the test reconstructs.
    rs = q.RngState(seed).split("run_hmc_cgfhmc_pure_gauge")
    gf = q.GaugeField(geo)
    gf.set_unit()
    out = []
    for traj in range(1, n_traj + 1):
        info = {}
        flag, delta_h = q.run_hmc_cgfhmc_pure_gauge(
            gf,
            ga,
            traj,
            rs,
            c_field=c_field,
            n_step=n_step,
            md_time=md_time,
            info_out=info,
        )
        out.append(
            {
                "traj": traj,
                "flag": int(flag),
                "delta_h": float(delta_h),
                "delta_h_accept": float(info["delta_h_accept"]),
                "dK_sum": float(info["dK_sum"]),
                "n_refresh_applied": int(info["n_refresh_applied"]),
                "fix_n_iter_total": int(info["fix_n_iter_total"]),
                "fix_n_iter_max": int(info["fix_n_iter_max"]),
                "plaq": float(gf.plaq()),
                # The field AFTER this trajectory: the accepted proposal, or the
                # previous field when the Metropolis test rejected.
                "field": [
                    [[float(x.real), float(x.imag)] for x in row]
                    for row in np.asarray(gf).reshape(-1, 9)
                ],
            }
        )
    return out

### --- the qlat_ref driver ----------------------------------------------

def get_ref_trajectories(
    total_site, beta, c1, n_step, md_time, kind, uniform, seed, n_traj
):
    """Run the same chain through ``qlat_ref`` (helper)."""
    geo = qr.Geometry(qr.Coordinate(list(total_site)))
    ga = qr.GaugeAction(beta, c1)
    c_field = cg_ref.generate_c_field(geo, get_c_func(total_site, kind, uniform))
    rs = qr.RngState(seed)
    gf = qr.GaugeField(geo)
    gf.set_unit()
    out = []
    for traj in range(1, n_traj + 1):
        info = {}
        flag, delta_h = cg_ref.run_hmc_cgfhmc_pure_gauge(
            gf,
            ga,
            traj,
            rs.split("run_hmc_cgfhmc_pure_gauge"),
            c_field=c_field,
            n_step=n_step,
            md_time=md_time,
            info_out=info,
        )
        out.append(
            {
                "traj": traj,
                "flag": int(flag),
                "delta_h": float(delta_h),
                "delta_h_accept": float(info["delta_h_accept"]),
                "dK_sum": float(info["dK_sum"]),
                "n_refresh_applied": int(info["n_refresh_applied"]),
                "fix_n_iter_total": int(info["fix_n_iter_total"]),
                "fix_n_iter_max": int(info["fix_n_iter_max"]),
                "plaq": float(gf.plaq()),
                "field": [
                    [[float(x.real), float(x.imag)] for x in row]
                    for row in np.asarray(gf).reshape(-1, 9)
                ],
            }
        )
    return out

### --- main --------------------------------------------------------------

def main():
    """Regenerate the vectors, cross-checking qlat against qlat_ref."""
    q.begin_with_mpi([[1, 1, 1, 1]])
    vec = {
        "metadata": {
            "source": "live qlat 1.21-current, run_hmc_cgfhmc_pure_gauge",
            "generated_by": "qlat-ref/tests/data/make-cgfhmc-vectors.py",
            "note": (
                "trajectories are deterministic: same seed gives the same "
                "flags/delta_h; `field` is the gauge field AFTER the "
                "trajectory (accepted proposal, or unchanged on rejection), "
                "`plaq` and `flag` are the post-Metropolis values"
            ),
            "n_step": 4,
            "md_time": 1.0,
            "gauge_fix": "max_iter=25 eps=1e-10 alpha=0.1",
            "insertion_point": (
                "fix + refresh exactly at the step boundary, i.e. after a "
                "self-contained lam,(1-2*lam),lam drift sandwich, so the "
                "accumulated drift is (i+1)*dt"
            ),
        },
        "chain_cases": {},
    }
    for (
        name,
        total_site,
        beta,
        c1,
        n_step,
        md_time,
        kind,
        uniform,
        seed,
        n_traj,
    ) in CASES:
        got = get_qlat_trajectories(
            total_site, beta, c1, n_step, md_time, kind, uniform, seed, n_traj
        )
        ref = get_ref_trajectories(
            total_site, beta, c1, n_step, md_time, kind, uniform, seed, n_traj
        )
        for x, y in zip(got, ref):
            d_dh = abs(x["delta_h"] - y["delta_h"])
            d_acc = abs(x["delta_h_accept"] - y["delta_h_accept"])
            d_dk = abs(x["dK_sum"] - y["dK_sum"])
            d_plaq = abs(x["plaq"] - y["plaq"])
            d_field = float(np.max(np.abs(np.array(x["field"]) - np.array(y["field"]))))
            assert x["flag"] == y["flag"], (name, x["traj"], x["flag"], y["flag"])
            assert x["n_refresh_applied"] == y["n_refresh_applied"], (name, x["traj"])
            assert x["fix_n_iter_total"] == y["fix_n_iter_total"], (name, x["traj"])
            assert x["fix_n_iter_max"] == y["fix_n_iter_max"], (name, x["traj"])
            assert d_dh < EPS_DELTA_H, (name, x["traj"], d_dh)
            assert d_acc < EPS_DELTA_H, (name, x["traj"], d_acc)
            assert d_dk < EPS_DK, (name, x["traj"], d_dk)
            assert d_plaq < EPS_PLAQ, (name, x["traj"], d_plaq)
            assert d_field < EPS_FIELD, (name, x["traj"], d_field)
            print(
                f"  {name} traj {x['traj']}: ok ddh={d_dh:.2e} dacc={d_acc:.2e}"
                f" ddk={d_dk:.2e} dplaq={d_plaq:.2e} dfield={d_field:.2e}"
            )
        vec["chain_cases"][name] = {
            "total_site": list(total_site),
            "beta": beta,
            "c1": c1,
            "n_step": n_step,
            "md_time": md_time,
            "seed": seed,
            "c_field_kind": kind,
            "c_uniform": uniform,
            "trajectories": got,
        }
    out = os.path.join(HERE, "cgfhmc-vectors.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(vec, fp, indent=1, sort_keys=True)
        fp.write("\n")
    print(f"wrote {out}")
    q.end_with_mpi()

if __name__ == "__main__":
    main()
