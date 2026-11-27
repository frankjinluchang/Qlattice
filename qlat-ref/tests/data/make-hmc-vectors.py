#!/usr/bin/env python3

"""
Regenerate ``hmc-vectors.json`` from the live C++ ``qlat`` build.\n
Not part of the installed package and not run by the test suite: the committed
JSON is the reference.  This script exists so the vectors can be regenerated
deliberately (e.g. after an intentional change in ``qlat``) and so the
provenance of every value is reproducible.\n
Usage (from the repository root)::
    source result-py-local/bin/setenv-qlat.sh
    python3 qlat-ref/tests/data/make-hmc-vectors.py
It writes ``hmc-vectors.json`` next to itself.  Every value is generated twice —
once through the live ``qlat`` build and once through :mod:`qlat_ref` — and the
script fails unless the two agree to floating-point precision (momentum and
trajectory fields) or within the stated tolerance (``delta_h``), so a wrong
expectation cannot be committed.\n
The two vector groups pin the two contracts of :mod:`qlat_ref.hmc`:
``gauge_momentum_cases`` the RNG scheme of the momentum initialisation (the
per-site ``split(gindex)``, then eight Gaussian coefficients per direction with
``sigma / sqrt(2)``), ``metropolis_decision_cases`` the accept/reject decision
of :func:`qlat_ref.hmc_utils.metropolis_accept` together with the raw 64-bit
draw that produces it (this is what pins the ``u_rand_gen(0.0, 1.0)`` argument
order), ``chain_flag_cases`` the per-trajectory flags of short chains run at a
step size where rejections are common (so a flipped decision cannot hide), and
``trajectory_cases`` the whole force-gradient trajectory (force,
matrix exponential, unitarisation, integrator and Metropolis step) through the
values -- including the accept/reject flag -- of one, two and three consecutive
trajectories.
"""

import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np

import qlat as q

import qlat_ref as qr

from qlat_ref import hmc_utils

TOTAL_SITE = [2, 2, 2, 2]
TRAJ_SEED = "qlat-ref-hmc-traj"

# Tolerances used only by the cross-check below (the test suite has its own):
# the momentum arithmetic is identical operation by operation, so the measured
# difference is a handful of ULP of an O(1) double; ``delta_h`` and the field
# after several matrix multiplications accumulate a little more.
EPS_MOMENTUM = 1e-14
EPS_FIELD = 1e-13
EPS_DELTA_H = 1e-11

### --- momentum cases ---------------------------------------------------

def momentum_case(name, seed_tree, make_q, make_ref, sigma):
    """Return one ``GaugeMomentum.set_rand`` vector case (helper)."""
    geo = q.Geometry(q.Coordinate(list(TOTAL_SITE)))
    gm = q.GaugeMomentum(geo)
    gm.set_rand(make_q(), sigma)
    arr = np.array(gm)
    flat = arr.reshape(-1)
    #
    geo_r = qr.Geometry(qr.Coordinate(list(TOTAL_SITE)))
    gm_r = qr.GaugeMomentum(geo_r)
    gm_r.set_rand(make_ref(), sigma)
    arr_r = np.asarray(gm_r).reshape(-1)
    diff = float(np.max(np.abs(arr_r - flat)))
    assert diff <= EPS_MOMENTUM, f"{name}: qlat_ref momentum differs by {diff}"
    #
    return {
        "name": name,
        "seed_tree": seed_tree,
        "total_site": list(TOTAL_SITE),
        "sigma": sigma,
        "shape": list(arr.shape),
        "flat_ordering": "index = ((site * 4 + mu) * 9 + (i * 3 + j)); "
        "site = index_from_coordinate(x, total_site) with x[0] fastest; "
        "matrix row-major p[i*3+j], matching ColorMatrixT::p",
        "values_real": [float(x.real) for x in flat],
        "values_imag": [float(x.imag) for x in flat],
        "gm_hamilton_node": float(q.gm_hamilton_node(gm)),
        "checksum_sum_real": float(np.sum(flat.real)),
        "checksum_sum_abs2": float(np.sum(np.abs(flat) ** 2)),
        "max_abs_diff_qlat_ref": diff,
    }

### --- trajectory cases -------------------------------------------------

def metropolis_decision_case(name, delta_h, seed):
    """Return one ``metropolis_accept`` decision vector (helper).\n
    The decision is recorded from **live qlat**, and the raw 64-bit draw is
    recorded alongside it so a reader can re-derive the answer without trusting
    either implementation: ``u = rand_gen() * 2**-64`` and the trajectory is
    accepted iff ``u <= min(1, exp(-delta_h))``.\n
    This is the vector that pins the ``u_rand_gen`` **argument order**.  qlat
    draws ``u_rand_gen(0.0, 1.0)``; passing ``(1.0, 0.0)`` draws ``1 - u``, so
    the recorded flag and the recorded raw draw disagree on every seed where
    ``u`` and ``1 - u`` fall on opposite sides of ``accept_prob`` -- which is
    every seed at ``accept_prob = 0.5``.
    """
    rs_q = q.RngState(seed)
    flag, accept_prob = q.metropolis_accept(delta_h, 1, rs_q)
    raw = q.RngState(seed).rand_gen()
    u = float(raw) * 2.0**-64
    accept_prob_expected = 1.0 if delta_h <= 0.0 else math.exp(-delta_h)
    flag_expected = u <= accept_prob_expected
    assert abs(accept_prob - accept_prob_expected) <= 1e-15
    assert bool(flag) == bool(flag_expected), (
        f"{name}: qlat flag {flag} disagrees with its own raw draw {u} "
        f"vs accept_prob {accept_prob_expected}"
    )
    return {
        "delta_h": delta_h,
        "seed": seed,
        "accept_prob": accept_prob,
        "raw_draw": raw,
        "u": u,
        "flag": int(flag),
        "provenance": "qlat.metropolis_accept",
    }

def metropolis_decision_cases():
    """Return the accept/reject decision vectors covering the probability range."""
    cases = {}
    # accept_prob = exp(-delta_h); the interesting region is around 1/2, where
    # u and 1-u are on opposite sides and the argument order decides the flag.
    for accept_prob in (0.95, 0.70, 0.50, 0.30, 0.05):
        delta_h = -math.log(accept_prob)
        for k in range(6):
            name = f"p{accept_prob:.2f}_s{k}"
            cases[name] = metropolis_decision_case(name, delta_h, f"metropolis-{k}")
    # the uncondititonal branch: delta_h <= 0 must accept without a draw
    for delta_h in (0.0, -1e-9, -25.0):
        name = "nonpositive_" + f"{delta_h}".replace("-", "m").replace(".", "p")
        cases[name] = metropolis_decision_case(name, delta_h, "metropolis-nonpositive")
    return cases

def trajectory_cases():
    """Return the trajectory vectors of a short pure-gauge HMC chain (helper)."""
    cases = {}
    for beta, c1 in ((5.5, 0.0), (1.0, 0.0), (2.13, -0.331)):
        geo = q.Geometry(q.Coordinate(list(TOTAL_SITE)))
        ga = q.GaugeAction(beta, c1)
        gf = q.GaugeField(geo)
        q.set_unit(gf)
        rs = q.RngState(TRAJ_SEED)
        #
        geo_r = qr.Geometry(qr.Coordinate(list(TOTAL_SITE)))
        ga_r = qr.GaugeAction(beta, c1)
        gf_r = qr.GaugeField(geo_r)
        gf_r.set_unit()
        rs_r = qr.RngState(TRAJ_SEED)
        #
        key = f"beta_{beta}_c1_{c1}".replace("-", "m").replace(".", "p")
        entry = {
            "beta": beta,
            "c1": c1,
            "seed": TRAJ_SEED,
            "n_step": 6,
            "md_time": 1.0,
            "start": "unit gauge field",
            "plaq_start": float(q.gf_avg_plaq(gf)),
            "gf_hamilton_start": float(q.gf_hamilton_node(gf, ga)),
            "trajectories": [],
            "flags": [],
        }
        assert entry["plaq_start"] == 1.0
        assert entry["gf_hamilton_start"] == 0.0
        max_dplaq = 0.0
        max_ddelta_h = 0.0
        max_dfield = 0.0
        for traj in range(1, 4):
            flag, delta_h = q.run_hmc_pure_gauge(
                gf,
                ga,
                traj,
                rs.split("run_hmc_pure_gauge"),
                n_step=6,
                md_time=1.0,
                is_always_accept=False,
            )
            flag_r, delta_h_r = hmc_utils.run_hmc_pure_gauge(
                gf_r,
                ga_r,
                traj,
                rs_r.split("run_hmc_pure_gauge"),
                n_step=6,
                md_time=1.0,
                is_always_accept=False,
            )
            plaq = float(q.gf_avg_plaq(gf))
            plaq_r = gf_r.plaq()
            field = np.array(gf).reshape(-1)
            field_r = np.asarray(gf_r).reshape(-1)
            dplaq = abs(plaq_r - plaq)
            ddelta_h = abs(delta_h_r - delta_h)
            dfield = float(np.max(np.abs(field_r - field)))
            assert int(flag_r) == int(flag), f"{key} traj {traj}: flag differs"
            assert dplaq <= EPS_FIELD, f"{key} traj {traj}: dplaq = {dplaq}"
            assert ddelta_h <= EPS_DELTA_H, f"{key} traj {traj}: ddelta_h = {ddelta_h}"
            assert dfield <= EPS_FIELD, f"{key} traj {traj}: dfield = {dfield}"
            #
            max_dplaq = max(max_dplaq, dplaq)
            max_ddelta_h = max(max_ddelta_h, ddelta_h)
            max_dfield = max(max_dfield, dfield)
            entry["trajectories"].append(
                {
                    "traj": traj,
                    "flag": int(flag),
                    "delta_h": float(delta_h),
                    "plaq": plaq,
                    "gf_hamilton_node": float(q.gf_hamilton_node(gf, ga)),
                    "field_real": [float(x.real) for x in field],
                    "field_imag": [float(x.imag) for x in field],
                }
            )
        #
        entry["max_abs_diff_qlat_ref"] = {
            "plaq": max_dplaq,
            "delta_h": max_ddelta_h,
            "field": max_dfield,
        }
        cases[key] = entry
    #
    return cases

### --- end-to-end accept/reject flags ----------------------------------

def chain_flag_case(beta, c1, seed, n_traj, n_step, md_time):
    """Return the per-trajectory accept/reject flags of one short chain (helper).\n
    ``md_time`` larger than the default puts the chain in the region where
    ``delta_h`` is of order one, so the Metropolis decision actually rejects
    sometimes; that is what makes a wrong ``u_rand_gen`` argument order visible
    as a divergent chain rather than as a rare one-trajectory difference.
    """
    geo = q.Geometry(q.Coordinate(list(TOTAL_SITE)))
    ga = q.GaugeAction(beta, c1)
    gf = q.GaugeField(geo)
    q.set_unit(gf)
    rs = q.RngState(seed)
    #
    geo_r = qr.Geometry(qr.Coordinate(list(TOTAL_SITE)))
    ga_r = qr.GaugeAction(beta, c1)
    gf_r = qr.GaugeField(geo_r)
    gf_r.set_unit()
    rs_r = qr.RngState(seed)
    #
    flags_q = []
    flags_r = []
    delta_hs = []
    for traj in range(1, n_traj + 1):
        flag, delta_h = q.run_hmc_pure_gauge(
            gf,
            ga,
            traj,
            rs.split("run_hmc_pure_gauge"),
            n_step=n_step,
            md_time=md_time,
            is_always_accept=False,
        )
        flag_r, delta_h_r = hmc_utils.run_hmc_pure_gauge(
            gf_r,
            ga_r,
            traj,
            rs_r.split("run_hmc_pure_gauge"),
            n_step=n_step,
            md_time=md_time,
            is_always_accept=False,
        )
        assert int(flag) == int(flag_r), f"{seed} traj {traj}: flag differs"
        assert abs(delta_h - delta_h_r) <= EPS_DELTA_H, f"{seed} traj {traj}"
        flags_q.append(int(flag))
        flags_r.append(int(flag_r))
        delta_hs.append(float(delta_h))
    #
    return {
        "beta": beta,
        "c1": c1,
        "seed": seed,
        "n_traj": n_traj,
        "n_step": n_step,
        "md_time": md_time,
        "start": "unit gauge field",
        "flags": flags_q,
        "flags_qlat_ref": flags_r,
        "delta_h": delta_hs,
        "plaq_end": float(q.gf_avg_plaq(gf)),
        "plaq_end_qlat_ref": gf_r.plaq(),
        "n_rejected": n_traj - sum(flags_q),
    }

def chain_flag_cases():
    """Return the end-to-end accept/reject flag vectors.\n
    The step sizes are chosen so that **both** branches are exercised and a
    flipped decision changes the result: each case has several rejections, and
    at least two trajectories whose flag differs when the accept test is
    evaluated with ``1 - u`` instead of ``u`` (verified while choosing them; a
    chain that rejects everything or accepts everything cannot tell the two
    apart and is useless as a regression vector).  ``delta_h`` grows with
    ``md_time`` because ``md_time / n_step`` is the step size.
    """
    return {
        "beta_5p5_md_time_2p0": chain_flag_case(5.5, 0.0, "x1-chain", 12, 6, 2.0),
        "beta_1p0_md_time_3p0": chain_flag_case(
            1.0, 0.0, "x1-chain-low-beta", 12, 6, 3.0
        ),
        "beta_2p13_md_time_1p5_rect": chain_flag_case(
            2.13, -0.331, "x1-chain-rect", 12, 6, 1.5
        ),
    }

### --- provenance -------------------------------------------------------

def git_commit():
    """Return the repository HEAD commit, or an empty string outside a git tree."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    try:
        return subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return ""

def main():
    """Build and write ``hmc-vectors.json``, cross-checking every value."""
    q.begin_with_mpi()
    try:
        cases = {}
        seeds = [
            (
                "gm_root_test_rng_sigma1",
                "RngState('test-rng')",
                lambda: q.RngState("test-rng"),
                lambda: qr.RngState("test-rng"),
                1.0,
            ),
            (
                "gm_root_test_rng_sigma0p5",
                "RngState('test-rng')",
                lambda: q.RngState("test-rng"),
                lambda: qr.RngState("test-rng"),
                0.5,
            ),
            (
                "gm_split_gm_sigma1",
                "RngState('test-gm').split('gm')",
                lambda: q.RngState("test-gm").split("gm"),
                lambda: qr.RngState("test-gm").split("gm"),
                1.0,
            ),
            (
                "gm_split_chain_sigma1",
                "RngState('test-gm').split('a').split('gm')",
                lambda: q.RngState("test-gm").split("a").split("gm"),
                lambda: qr.RngState("test-gm").split("a").split("gm"),
                1.0,
            ),
            (
                "gm_hmc_traj_chain",
                "RngState('hmc-seed').split('0').split('set_rand_gauge_momentum')",
                lambda: (
                    q.RngState("hmc-seed").split("0").split("set_rand_gauge_momentum")
                ),
                lambda: (
                    qr.RngState("hmc-seed").split("0").split("set_rand_gauge_momentum")
                ),
                1.0,
            ),
        ]
        for name, seed_tree, make_q, make_ref, sigma in seeds:
            cases[name] = momentum_case(name, seed_tree, make_q, make_ref, sigma)
        #
        payload = {
            "metadata": {
                "description": "C++ qlat HMC test vectors for qlat_ref.hmc "
                "(GaugeMomentum.set_rand, gm_hamilton_node and full "
                "force-gradient trajectories)",
                "provenance": {
                    "qlat_version": importlib.metadata.version("qlat"),
                    "qlat_utils_version": importlib.metadata.version("qlat-utils"),
                    "generator": "qlat-ref/tests/data/make-hmc-vectors.py "
                    "(run under result-py-local/bin/setenv-qlat.sh)",
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "libc": list(platform.libc_ver()),
                    "git_commit": git_commit(),
                },
                "exactness": {
                    "integer": "the RNG integer stream is bit-exact; not "
                    "recorded here because tests/data/rng-vectors.json is its "
                    "primary contract",
                    "float": "values are compared to 1e-14 (momentum) and "
                    "1e-12 (trajectory field/plaq); the C++ and qlat_ref "
                    "arithmetic is the same operation by operation, so the "
                    "measured differences are a few ULP.  Full precision "
                    "decimals are stored (repr of the double), so a "
                    "regeneration can be compared bit for bit by hand, but the "
                    "tests assert the tolerance, not bit equality.",
                },
                "maintenance_note": "These vectors are a committed reference, not "
                "a snapshot to regenerate casually: regenerating them after an "
                "unintended change in qlat_ref would defeat their purpose.",
            },
            "gauge_momentum_cases": cases,
            "metropolis_decision_cases": metropolis_decision_cases(),
            "chain_flag_cases": chain_flag_cases(),
            "trajectory_cases": trajectory_cases(),
        }
        path = os.path.join(HERE, "hmc-vectors.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=1)
            fp.write("\n")
        #
        print("wrote", path)
        print("momentum cases:", len(cases))
        print("metropolis decision cases:", len(payload["metropolis_decision_cases"]))
        print("chain flag cases:", len(payload["chain_flag_cases"]))
        for name, case in payload["chain_flag_cases"].items():
            print(
                f"  {name}: flags = {case['flags']}"
                f"  ({case['n_rejected']}/{case['n_traj']} rejected)"
            )
        print("trajectory cases:", len(payload["trajectory_cases"]))
        for name, case in payload["trajectory_cases"].items():
            print(
                f"  {name}: max|d plaq| = {case['max_abs_diff_qlat_ref']['plaq']:.3e}"
                f"  max|d delta_h| = {case['max_abs_diff_qlat_ref']['delta_h']:.3e}"
                f"  max|d field| = {case['max_abs_diff_qlat_ref']['field']:.3e}"
            )
    finally:
        q.end_with_mpi()

if __name__ == "__main__":
    main()
