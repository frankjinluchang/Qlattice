"""
Module ``qlat_ref.hmc_utils``
=============================\n
The Hybrid-Monte-Carlo driver layer of a pure-gauge lattice, mirroring
``qlat.hmc_utils``: the Metropolis accept/reject step, the force-gradient
(Omelyan / minimum-norm) molecular-dynamics integrator, and the two trajectory
drivers.\n
The functions, their argument order, defaults and mutation semantics are
qlat's; the only structural difference is that qlat runs the lattice over MPI
nodes (so it sums ``delta_h`` with ``glb_sum_double`` and broadcasts the
accept/reject flag), while this is a single-process reference where the
node-local quantities already are the global ones.\n
Algorithm (qlat's, ``qlat/qlat/hmc_utils.py``)
----------------------------------------------\n
One trajectory draws a momentum ``P`` from
:meth:`qlat_ref.field.GaugeMomentum.set_rand` (the Gaussian field of
:func:`qlat_ref.hmc.set_rand_gauge_momentum`), runs the force-gradient
integrator for ``n_step`` steps of size ``dt = md_time / n_step`` with the
Omelyan coefficients
``lam = (1 - 1/sqrt(3)) / 2`` and ``theta = (2 - sqrt(3)) / 48``, and then
accepts with probability ``min(1, exp(-delta_h))``.\n
RNG tree (qlat's, identical): the driver seed ``rs`` is split by the
trajectory number,
``rs_traj = rs.split(str(traj))``, and the two consumers get independent
streams, ``rs_traj.split("set_rand_gauge_momentum")`` for the momentum and
``rs_traj.split("metropolis_accept")`` for the accept draw.  They are
independent on purpose: a rejected trajectory does not disturb the momentum
stream.\n
Mutation semantics (qlat's):
:func:`gm_evolve_fg_pure_gauge` mutates ``gm`` only and never ``gf_init``;
:func:`run_hmc_evolve_pure_gauge` mutates **both** ``gm`` and ``gf``;
:func:`run_hmc_pure_gauge` mutates ``gf`` **in place on accept** and returns
``(flag, delta_h)`` -- a caller that needs the pre-trajectory configuration
must take ``np.array(gf)`` or ``gf.copy()`` *before* the call (both are genuine
copies; ``np.asarray(gf)`` is a live view and follows the update).\n
.. note:: qlat prints one progress line per trajectory through
   ``displayln_info``.  The reference is silent by default -- the test suite runs
   hundreds of trajectories -- and prints the same lines when the private
   ``_VERBOSE`` switch is set.  It is a logging switch: no value, and no RNG
   draw, depends on it.\n
Documentation: ``docs/qlat-ref/qlat_hmc.md``\n
.. note:: Update the documentation when updating this source file.
"""

import math

import numpy as np

from .field import GaugeField
from .field import GaugeMomentum

from .hmc import _get_lattice
from .hmc import _set_lattice
from .hmc import gf_evolve
from .hmc import gf_hamilton_node
from .hmc import gm_hamilton_node
from .hmc import set_gm_force

### -------------------------------------------------------------------

# qlat reports every trajectory through ``displayln_info``; the reference
# package is quiet by default (the test suite runs hundreds of trajectories) and
# prints the same lines when this switch is set.  It is a logging switch only:
# no value or stream depends on it.
_VERBOSE = False

def _displayln_info(*args):
    """Print an informational line when ``_VERBOSE`` is set (helper)."""
    if _VERBOSE:
        print(*args)

### -------------------------------------------------------------------

def metropolis_accept(delta_h, traj, rs):
    """
    Metropolis accept/reject step (qlat's ``metropolis_accept``).\n
    Accept with probability ``min(1, exp(-delta_h))`` and return
    ``(flag, accept_prob)``: ``delta_h <= 0`` is accepted unconditionally with
    ``accept_prob = 1.0``; otherwise ``accept_prob = exp(-delta_h)`` and the
    trajectory is accepted when ``rs.u_rand_gen(0.0, 1.0) <= accept_prob``.\n
    ``rs`` is the accept stream, ``rs_traj.split("metropolis_accept")``; the
    uniform draw is taken from it and the ``traj`` number is only used for the
    progress line.  The draw is ``rs.u_rand_gen(0.0, 1.0)`` -- qlat's argument
    order is ``(lower, upper)`` (it was ``(upper, lower)`` before the rename in
    both qlat and :mod:`qlat_ref.rng`, and the values were swapped with it), so
    this is a uniform number in ``[0, 1)``.  Passing ``(1.0, 0.0)`` would draw
    ``1 - u`` instead and flip every borderline decision: at ``p = 0.5`` it
    disagrees with qlat on every seed.  Single process: qlat computes this on
    node ``0`` and broadcasts ``flag``/``accept_prob``, which is the identity
    here.
    """
    flag_d = 0.0
    accept_prob = 0.0
    if delta_h <= 0.0:
        accept_prob = 1.0
        flag_d = 1.0
    else:
        accept_prob = math.exp(-delta_h)
        rand_num = rs.u_rand_gen(0.0, 1.0)
        if rand_num <= accept_prob:
            flag_d = 1.0
    #
    flag = flag_d > 0.5
    _displayln_info(
        f"metropolis_accept: flag={int(flag):d} with"
        f" accept_prob={accept_prob * 100.0:.1f}%"
        f" delta_h={delta_h:.16f} traj={traj}"
    )
    return flag, accept_prob

def gm_evolve_fg_pure_gauge(gm, gf_init, ga, fg_dt, dt):
    """
    Force-gradient momentum update, in place on ``gm`` (qlat's
    ``gm_evolve_fg_pure_gauge``).\n
    Evaluate the force ``F`` at ``gf_init``, evolve a **copy** of the links by
    ``fg_dt`` along ``F``, re-evaluate the force at that intermediate point,
    and add ``dt * F_intermediate`` to ``gm``.  ``gf_init`` itself is never
    mutated (qlat makes the same internal copy because the force is needed at
    the intermediate field and at the original one).
    """
    geo = gf_init.geo
    gf = GaugeField(geo)
    gf @= gf_init
    gm_force = GaugeMomentum(geo)
    set_gm_force(gm_force, gf, ga)
    gf_evolve(gf, gm_force, fg_dt)
    set_gm_force(gm_force, gf, ga)
    _set_lattice(gm, _get_lattice(gm) + dt * _get_lattice(gm_force))

def run_hmc_evolve_pure_gauge(gm, gf, ga, n_step, md_time=1.0):
    """
    Run the force-gradient molecular-dynamics evolution, in place (qlat's
    ``run_hmc_evolve_pure_gauge``).\n
    Evolve **both** ``gm`` and ``gf`` for ``n_step`` steps of size
    ``dt = md_time / n_step`` and return the energy violation
    ``delta_h = H(final) - H(initial)`` (single process: already the global
    value qlat obtains with ``glb_sum_double``).\n
    The scheme is the Omelyan / minimum-norm force-gradient integrator of
    ``qlat/qlat/hmc_utils.py`` with ``lam = (1 - 1/sqrt(3)) / 2`` and
    ``theta = (2 - sqrt(3)) / 48``; the final field is re-unitarised.
    ``md_time`` may be negative, which is the reversibility test: re-running
    from the output with ``-md_time`` returns to the input.
    """
    energy = gm_hamilton_node(gm) + gf_hamilton_node(gf, ga)
    dt = md_time / n_step
    lam = 0.5 * (1.0 - 1.0 / math.sqrt(3.0))
    theta = (2.0 - math.sqrt(3.0)) / 48.0
    ttheta = theta * dt * dt * dt
    gf_evolve(gf, gm, lam * dt)
    for i in range(n_step):
        gm_evolve_fg_pure_gauge(gm, gf, ga, 4.0 * ttheta / dt, 0.5 * dt)
        gf_evolve(gf, gm, (1.0 - 2.0 * lam) * dt)
        gm_evolve_fg_pure_gauge(gm, gf, ga, 4.0 * ttheta / dt, 0.5 * dt)
        if i < n_step - 1:
            gf_evolve(gf, gm, 2.0 * lam * dt)
        else:
            gf_evolve(gf, gm, lam * dt)
    #
    gf.unitarize()
    delta_h = gm_hamilton_node(gm) + gf_hamilton_node(gf, ga) - energy
    return delta_h

def run_hmc_pure_gauge(
    gf,
    ga,
    traj,
    rs,
    *,
    is_reverse_test=False,
    n_step=6,
    md_time=1.0,
    is_always_accept=False,
):
    """
    Run one pure-gauge HMC trajectory (qlat's ``run_hmc_pure_gauge``).\n
    Draw a Gaussian momentum, run the force-gradient MD evolution, apply the
    Metropolis step, and return ``(flag, delta_h)``.  On accept ``gf`` is
    updated **in place** (qlat's convention ``gf @= gf0``); the trial field is
    an internal ``gf.copy()``, so a rejected trajectory leaves ``gf``
    bit-unchanged.  Take ``gf.copy()`` **before** the call when the
    pre-trajectory configuration is needed.\n
    Parameters
    ----------
    gf : GaugeField
        Input/output gauge field, mutated in place on accept.
    ga : GaugeAction
        The gauge action (``beta``, ``c1``).
    traj : int
        Trajectory number; the driver splits the RNG with ``str(traj)``.
    rs : RngState
        Driver-level seed on top of which the trajectory and consumer splits
        are taken.
    is_reverse_test : bool
        Run the trajectory backwards from the output and assert that the field
        returns (qlat's cheap in-driver reversibility gate).
    n_step : int
        Number of MD steps per trajectory (qlat's default ``6``).
    md_time : float
        Total MD time (qlat's default ``1.0``); negative reverses the
        evolution.
    is_always_accept : bool
        Update ``gf`` even when the Metropolis test rejects.\n
    Returns
    -------
    (flag, delta_h)
        ``flag`` is whether the Metropolis test accepted; ``delta_h`` is the
        energy violation of the trajectory (always the trajectory's own, even
        when ``is_always_accept`` forces the update).
    """
    rs = rs.split(f"{traj}")
    geo = gf.geo
    gf0 = GaugeField(geo)
    gf0 @= gf
    gm = GaugeMomentum(geo)
    gm.set_rand(rs.split("set_rand_gauge_momentum"), 1.0)
    delta_h = run_hmc_evolve_pure_gauge(gm, gf0, ga, n_step, md_time)
    if is_reverse_test:
        gm_r = GaugeMomentum(geo)
        gm_r @= gm
        gf0_r = GaugeField(geo)
        gf0_r @= gf0
        delta_h_rev = run_hmc_evolve_pure_gauge(gm_r, gf0_r, ga, n_step, -md_time)
        gf_diff_norm = float(
            np.sum(np.abs(_get_lattice(gf0_r) - _get_lattice(gf)) ** 2)
        )
        gf_norm = gf0.qnorm()
        _displayln_info(
            f"run_hmc_pure_gauge: reversed delta_diff: {delta_h + delta_h_rev}"
            f" / {delta_h}"
        )
        _displayln_info(
            f"run_hmc_pure_gauge: reversed gf_diff: {gf_diff_norm} / {gf_norm}"
        )
        assert gf_diff_norm <= 1e-12 * gf_norm
    #
    flag, accept_prob = metropolis_accept(delta_h, traj, rs.split("metropolis_accept"))
    if flag or is_always_accept:
        _displayln_info(f"run_hmc_pure_gauge: update gf (traj={traj})")
        gf @= gf0
    #
    return flag, delta_h
