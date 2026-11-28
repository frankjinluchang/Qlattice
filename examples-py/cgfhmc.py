#!/usr/bin/env python3

# Canonical Gauge-Fixed HMC (CGFHMC) on a pure-gauge field.
#
# CGFHMC = canonical HMC + Landau gauge fixing after every MD step (the same
# gauge transform is applied to BOTH the gauge field and the gauge momentum,
# so the pair stays canonical) + a partial momentum refresh with a Fourier
# mode-dependent noise weight c(k), with Metropolis applied to the energy
# violation corrected by the refresh-induced kinetic change.
#
#   P_k <- sqrt(1 - c(k)^2) P_k + c(k) xi_k        (c = fresh-noise weight)
#   delta_h_accept = delta_h - dK_refresh_sum
#
# This example exercises:
#   * a thermalized chain with a UNIFORM c (the real-space fast path),
#   * a thermalized chain with a MODE-DEPENDENT c(k) (the Fourier path),
#   * the strict no-op (c = 0 draws no random number, momentum bit-identical),
#   * the complete resampling limit (c = 1 replaces the momentum by the noise),
#   * the mode grid / c-field helpers,
#   * the not-time-reversible guard on ``is_reverse_test``.
#
# Parameter choice matters: on a small lattice the raw delta_h is dominated by
# the kinetic-energy fluctuation (E[K] = 16 V) and by dK_refresh_sum, so a cold
# start at large md_time rejects every trajectory.  beta = 6.0 with
# md_time = 1.0 / n_step = 3 (dt = 1/3) after N_THERM = 8 Metropolis
# thermalization trajectories gives a mixed outcome here: measured mean
# |delta_h_accept| is of order 1-2 and the accept rate lands around 40%-50%
# rather than 0% or 100%, so the Metropolis test is doing real work.  The raw
# |delta_h| is deliberately much larger (order 45-90) because the refresh injects
# a kinetic change that the Metropolis variable removes.
#
# Only floats are recorded with ``json_results_append(name, value, eps)``; every
# integer/boolean outcome is encoded in the recorded NAME instead, and no
# near-zero quantity is compared with a relative epsilon (AGENTS.md: the recent
# CI lesson).  Every recorded float is bounded away from zero (the c-field
# entries are >= 0.2, the energy/plaquette entries are O(1) or larger), so the
# relative comparison in ``q.check_log_json`` has ample margin.
#
# Run through the examples-py test suite / nixpkgs/run-one-example-py.py
# (target cgfhmc.log).  The runner invokes ``mpiexec -n 2``, hence the
# ``[2, 1, 1, 1]`` decomposition below.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "false"

import numpy as np

import qlat as q

size_node_list = [
    [1, 1, 1, 1],
    [2, 1, 1, 1],
]

TOTAL_SITE = [4, 4, 4, 4]
BETA = 6.0
# Noise weight c; the legacy correlation convention c_corr = 0.9 corresponds to
# c = sqrt(1 - c_corr^2) = 0.4358898943540674.
C_NOISE = 0.4358898943540674
# One trajectory of md_time = 1.0 in 3 steps (dt = 1/3).  The coarse step keeps
# the corrected integrator's Metropolis variable O(1) instead of O(dt^4): the
# 4th-order force-gradient integrator conserves energy so well that a small dt
# drives |delta_h_accept| down to ~1e-5, and a relative comparison of a
# difference of two O(50) energies at that size would be mostly round-off.  At
# dt = 1/3 the per-trajectory |delta_h_accept| is O(1) and the accept rate is
# mixed (~40-50%), so the Metropolis test is doing real work and every recorded
# float stays well away from zero.
MD_TIME = 1.0
N_STEP = 3
N_THERM = 8
N_MEAS = 10

EPS_PLAQ = 1.0e-10
EPS_RESULT = 1.0e-8

def get_c_field_uniform(geo, c_noise):
    # A constant c(k): the refresh takes the real-space fast path.
    return q.generate_c_field(geo, lambda kv: c_noise)

def get_c_field_mode(geo, c_low, c_high):
    # A two-zone profile: the low momenta are held stiff (small c, long
    # autocorrelation) and the high momenta are refreshed hard.  Because c(k)
    # varies with k, the refresh must go through the Fourier path.
    k_cut = q.resolve_k_cut(geo, 1)
    return q.generate_c_field(
        geo,
        lambda kv: np.where(np.sum(kv**2, axis=0) <= k_cut**2, c_low, c_high),
    )

def run_chain(fname, geo, ga, rs, c_field, label):
    # Metropolis chain: N_THERM trajectories to equilibrate, then N_MEAS
    # measured ones.  ``is_always_accept`` is deliberately NOT used for the
    # thermalization: forced acceptance drives the gauge action uphill and the
    # field never equilibrates.
    # Probe the refresh ONCE on a throwaway momentum to record the real route
    # (real-space fast path vs Fourier) and the applied coefficients.  The probe
    # uses its own RngState, so it never perturbs the chain's random stream.
    # NOTE: the labels are literal (not derived from ``rs``): str(RngState) is
    # the default object repr and would include a memory address.
    gm_probe = q.GaugeMomentum(geo)
    gm_probe.set_rand(q.RngState(f"{label}-probe-gm"), 1.0)
    applied, _, refresh_info = q.partial_refresh_momentum_mode(
        gm_probe, q.RngState(f"{label}-probe-refresh"), c_field
    )
    refresh_path = refresh_info["path"]
    c_min = float(refresh_info["c_min"])
    c_max = float(refresh_info["c_max"])
    c_mean = float(refresh_info["c_mean"])
    gf = q.GaugeField(geo)
    gf.set_rand(rs.split("init"), 0.3, 2)
    gf.unitarize()
    traj = 0
    for _ in range(N_THERM):
        traj += 1
        q.run_hmc_cgfhmc_pure_gauge(
            gf,
            ga,
            traj,
            rs.split("run_hmc_cgfhmc_pure_gauge"),
            c_field=c_field,
            n_step=N_STEP,
            md_time=MD_TIME,
        )
    n_accepted = 0
    abs_delta_h = []
    delta_h_accept = []
    dk_sum = []
    n_fix = 0
    n_fix_converged = 0
    n_refresh_applied = 0
    plaq_list = []
    for _ in range(N_MEAS):
        traj += 1
        info = {}
        flag, delta_h = q.run_hmc_cgfhmc_pure_gauge(
            gf,
            ga,
            traj,
            rs.split("run_hmc_cgfhmc_pure_gauge"),
            c_field=c_field,
            n_step=N_STEP,
            md_time=MD_TIME,
            info_out=info,
        )
        n_accepted += int(bool(flag))
        abs_delta_h.append(abs(delta_h))
        delta_h_accept.append(info["delta_h_accept"])
        dk_sum.append(info["dK_sum"])
        n_fix += info["n_fix"]
        n_fix_converged += info["fix_n_converged"]
        n_refresh_applied += info["n_refresh_applied"]
        plaq_list.append(float(gf.plaq()))
    # Absolute means only: a signed mean of delta_h could land near zero, where
    # a relative comparison against a committed reference is meaningless.
    q.json_results_append(
        f"cgfhmc {label}: mean |delta_h|", float(np.mean(abs_delta_h)), EPS_RESULT
    )
    q.json_results_append(
        f"cgfhmc {label}: mean |delta_h_accept|",
        float(np.mean(np.abs(delta_h_accept))),
        EPS_RESULT,
    )
    q.json_results_append(
        f"cgfhmc {label}: dK_sum mean abs", float(np.mean(np.abs(dk_sum))), EPS_RESULT
    )
    q.json_results_append(
        f"cgfhmc {label}: plaq mean", float(np.mean(plaq_list)), EPS_PLAQ
    )
    # c-field diagnostics: what the refresh ACTUALLY applied (from the probe
    # above, so this is the driver's own report, not a re-derivation).  The
    # regularizer has already clipped into [0, 1] and evened the profile.
    # All three are O(0.1) or larger, never a signed near-zero, so the relative
    # comparison in check_log_json has an ample margin.
    q.json_results_append(f"cgfhmc {label}: c_min", c_min, EPS_RESULT)
    q.json_results_append(f"cgfhmc {label}: c_max", c_max, EPS_RESULT)
    q.json_results_append(f"cgfhmc {label}: c_mean", c_mean, EPS_RESULT)
    q.json_results_append(
        f"cgfhmc {label}: c is uniform = {refresh_info['is_uniform']}"
        f" (refresh path = {refresh_path})"
    )
    # Structural invariant: with c > 0 somewhere the refresh is applied on every
    # single MD step (this is what makes dK_refresh_sum a complete correction).
    q.json_results_append(
        f"cgfhmc {label}: refresh applied on every step ="
        f" {bool(applied) and n_fix > 0 and n_refresh_applied == N_MEAS * N_STEP}"
        f" ({n_refresh_applied} of {N_MEAS * N_STEP} MD steps)"
    )
    # Exact, machine-independent counts/booleans: encoded in the name, never as
    # a fabricated float.
    q.json_results_append(f"cgfhmc {label}: accepted = {n_accepted}/{N_MEAS}")
    q.json_results_append(f"cgfhmc {label}: gauge fixations = {n_fix}")
    # Converged/not is NOT recorded as a boolean: the per-step fixing uses a
    # small max_iter and starts from a nearly-fixed field, so convergence can
    # flip on a last-bit change.  Count it instead.
    q.json_results_append(
        f"cgfhmc {label}: fix steps converged = {n_fix_converged}/{n_fix}"
    )
    q.displayln_info(
        f"{fname}: {label}: accepted {n_accepted}/{N_MEAS};"
        f" mean |delta_h| = {np.mean(abs_delta_h):.6f};"
        f" mean |delta_h_accept| = {np.mean(np.abs(delta_h_accept)):.6f};"
        f" plaq = {np.mean(plaq_list):.10f};"
        f" c = [{c_min:.4f}, {c_max:.4f}] mean {c_mean:.4f} ({refresh_path});"
        f" fix converged {n_fix_converged}/{n_fix}"
    )
    return float(np.mean(plaq_list))

def check_c_field_helpers(fname, geo):
    # Momentum grid: total_site axis order (Lx, Ly, Lz, Lt), folded to [-pi, pi).
    k_vec = q.k_vectors(geo)
    k_sqr = q.k_squared(geo)
    ok_shape = k_vec.shape == (4,) + tuple(geo.total_site.to_list())
    ok_consistent = bool(np.allclose(k_sqr, np.sum(k_vec**2, axis=0)))
    # The upper half of each axis must fold to a negative momentum: for L = 4
    # the index 3 maps to -2*pi/4 rather than +3*2*pi/4.
    ok_folded = bool(
        abs(k_vec[0, 3, 0, 0, 0] + 2.0 * np.pi / 4.0) < 1.0e-12
        and abs(k_vec[0, 2, 0, 0, 0] - np.pi) < 1.0e-12
    )
    # Regularization is always applied: clip to [0, 1] and enforce c(-k) = c(k).
    c_check = q.generate_c_field(
        geo, lambda kv: np.where(np.sum(kv**2, axis=0) <= 1.0, 0.25, 2.5)
    )
    ok_range = bool(float(c_check.min()) >= 0.0 and float(c_check.max()) <= 1.0)
    partner = c_check
    for axis in range(4):
        partner = np.roll(np.flip(partner, axis=axis), 1, axis=axis)
    ok_even = bool(np.allclose(c_check, 0.5 * (c_check + partner), atol=0.0))
    ok_dtype = bool(c_check.dtype == np.float64 and c_check.flags.c_contiguous)
    q.json_results_append(f"cgfhmc helpers: k grid shape = {ok_shape}")
    q.json_results_append(f"cgfhmc helpers: k^2 consistent = {ok_consistent}")
    q.json_results_append(f"cgfhmc helpers: k folded at Nyquist = {ok_folded}")
    q.json_results_append(f"cgfhmc helpers: c clipped to [0,1] = {ok_range}")
    q.json_results_append(f"cgfhmc helpers: c is even = {ok_even}")
    q.json_results_append(f"cgfhmc helpers: c is float64 C-contiguous = {ok_dtype}")
    q.displayln_info(
        f"{fname}: helpers: shape={ok_shape} k2={ok_consistent}"
        f" folded={ok_folded} range={ok_range} even={ok_even} dtype={ok_dtype}"
    )

def check_refresh_limits(fname, geo):
    rs = q.RngState("cgfhmc-limits")
    gm0 = q.GaugeMomentum(geo)
    gm0.set_rand(rs.split("gm"), 1.0)
    gm_before = np.asarray(gm0).copy()
    # c = 0 everywhere: strict no-op, bit-identical, and NO random number drawn
    # (compare the next draw against a fresh RngState).
    c_zero = q.generate_c_field(geo, lambda kv: 0.0)
    rng_probe = q.RngState("cgfhmc-probe")
    applied, dk, info = q.partial_refresh_momentum_mode(gm0, rng_probe, c_zero)
    is_noop = bool(
        applied is False and dk == 0.0 and np.array_equal(np.asarray(gm0), gm_before)
    )
    rng_fresh = q.RngState("cgfhmc-probe")
    rng_untouched = bool(
        rng_probe.u_rand_gen(0.0, 1.0) == rng_fresh.u_rand_gen(0.0, 1.0)
    )
    # c = 1 everywhere: complete resampling, P becomes exactly the fresh noise.
    c_one = q.generate_c_field(geo, lambda kv: 1.0)
    gm_full = q.GaugeMomentum(geo)
    gm_full.set_rand(q.RngState("cgfhmc-full"), 1.0)
    noise = q.GaugeMomentum(geo)
    noise.set_rand(q.RngState("cgfhmc-noise"), 1.0)
    q.partial_refresh_momentum_mode(gm_full, q.RngState("cgfhmc-noise"), c_one)
    is_full_resample = bool(
        float(np.max(np.abs(np.asarray(gm_full) - np.asarray(noise)))) < 1.0e-14
    )
    # The uniform path is exactly the real-space mixing law.
    c_uniform = q.generate_c_field(geo, lambda kv: C_NOISE)
    keep = float(np.sqrt(1.0 - C_NOISE**2))
    gm_mix = q.GaugeMomentum(geo)
    gm_mix.set_rand(q.RngState("cgfhmc-mix"), 1.0)
    expected = keep * np.asarray(gm_mix)
    xi = q.GaugeMomentum(geo)
    xi.set_rand(q.RngState("cgfhmc-xi"), 1.0)
    expected = expected + C_NOISE * np.asarray(xi)
    applied_mix, _, info_mix = q.partial_refresh_momentum_mode(
        gm_mix, q.RngState("cgfhmc-xi"), c_uniform
    )
    max_dev = float(np.max(np.abs(np.asarray(gm_mix) - expected)))
    is_exact = bool(max_dev < 1.0e-14)
    q.json_results_append(f"cgfhmc limits: c=0 is a strict no-op = {is_noop}")
    q.json_results_append(
        f"cgfhmc limits: c=0 draws no random number = {rng_untouched}"
    )
    q.json_results_append(
        f"cgfhmc limits: c=1 resamples completely = {is_full_resample}"
    )
    q.json_results_append(f"cgfhmc limits: uniform c matches explicit law = {is_exact}")
    q.json_results_append(
        f"cgfhmc limits: uniform c uses the real-space path ="
        f" {info_mix['path'].startswith('real-space')}"
    )
    q.displayln_info(
        f"{fname}: limits: no-op={is_noop} rng={rng_untouched}"
        f" full={is_full_resample} exact={is_exact} (max_dev={max_dev:.3e})"
        f" path={info_mix['path']!r}"
    )

def check_reverse_test_guard(fname, geo, ga, c_field):
    # CGFHMC is not time-reversible: the per-step gauge fixing is a projection.
    # The parameter is kept for signature compatibility with run_hmc_pure_gauge
    # but must refuse rather than return a meaningless verdict.
    gf = q.GaugeField(geo)
    gf.set_rand(q.RngState("cgfhmc-rev"), 0.3, 2)
    gf.unitarize()
    try:
        q.run_hmc_cgfhmc_pure_gauge(
            gf,
            ga,
            1,
            q.RngState("cgfhmc-rev-run"),
            c_field=c_field,
            is_reverse_test=True,
        )
        raised = False
        message_ok = False
    except NotImplementedError as exc:
        raised = True
        message_ok = "not time-reversible" in str(exc)
    q.json_results_append(
        f"cgfhmc guard: is_reverse_test raises NotImplementedError = {raised}"
    )
    q.json_results_append(
        f"cgfhmc guard: the error names the irreversibility = {message_ok}"
    )
    q.displayln_info(f"{fname}: guard: raised={raised} message_ok={message_ok}")

def main():
    fname = q.get_fname()
    q.begin_with_mpi(size_node_list)
    geo = q.Geometry(q.Coordinate(TOTAL_SITE))
    beta = BETA
    ga = q.GaugeAction(beta, 0.0)
    rs = q.RngState(f"cgfhmc-{TOTAL_SITE}")
    q.displayln_info(
        f"{fname}: total_site = {TOTAL_SITE}; beta = {beta};"
        f" md_time = {MD_TIME}; n_step = {N_STEP};"
        f" c_noise = {C_NOISE!r}; num_node = {q.get_num_node()}"
    )
    check_c_field_helpers(fname, geo)
    check_refresh_limits(fname, geo)
    check_reverse_test_guard(fname, geo, ga, get_c_field_mode(geo, 0.2, C_NOISE))
    # Uniform c: the real-space fast path.
    run_chain(
        fname,
        geo,
        ga,
        rs.split("uniform"),
        get_c_field_uniform(geo, C_NOISE),
        "uniform",
    )
    # Mode-dependent c(k): the Fourier path.
    run_chain(
        fname, geo, ga, rs.split("mode"), get_c_field_mode(geo, 0.2, C_NOISE), "mode"
    )
    q.timer_display()
    if q.is_test():
        q.check_log_json(__file__, check_eps=EPS_RESULT)
    q.end_with_mpi()
    q.displayln_info("CHECK: finished successfully.")

if __name__ == "__main__":
    main()
