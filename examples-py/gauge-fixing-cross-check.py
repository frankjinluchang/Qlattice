#!/usr/bin/env python3

# Cross-check of the public qlat gauge-fixing API (``qlat.gauge_fixing``)
# against the pure-NumPy reference package `qlat-ref`.
#
# This script is NOT part of the CI test suite: it is a manual (non-CI) check
# with no reference ``.log``/``.log.json`` and it is not registered in
# ``examples-py/Makefile``.  Its `assert`s are self-checking, so it cannot fail
# silently from a missing reference, but it also gives no CI regression
# protection.  Run it manually from a checkout with the nix environment:
#
#   source result-py-local/bin/setenv-qlat.sh
#   mpiexec -n 2 --oversubscribe --bind-to none \
#     python3 -m mpi4py examples-py/gauge-fixing-cross-check.py
#
# `qlat-ref` is an installed package in the nix dev environment (it is part of
# ``qlat-jhub-env``), so ``import qlat_ref`` normally just works.  The
# ``sys.path`` fallback below only covers a checkout without a rebuilt
# environment: it adds the sibling ``qlat-ref/`` source directory.  The script
# is not run by the nix example build (that derivation does not depend on
# `qlat-ref`), which is the other reason it is not in CI.
#
# It builds one deterministic random gauge field with `qlat-ref`, fixes it
# with both implementations (same max_iter/eps/alpha) and compares the
# converged functional, the RMS divergence, the unitarity error and (on the
# locally owned sites of each MPI rank) the fixed link matrices.
#
# Both implementations return ``(gt, info)`` and neither mutates its input
# field.  The returned ``gt`` is **not unique**: it depends on the convergence
# path and is only defined up to the residual stabiliser (a global center
# element already gives a different ``gt`` for a bit-identical fixed field).
# The comparison therefore applies both transforms to the SAME original field
# and compares the resulting fixed fields, never the raw ``gt`` arrays.

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "false"

import numpy as np

import qlat as q

try:
    import qlat_ref as qr
except ImportError:
    # Fallback for a checkout whose nix environment has not been rebuilt since
    # `qlat-ref` became a packaged dependency: use the sibling source tree.
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(REPO_ROOT, "qlat-ref"))
    import qlat_ref as qr

size_node_list = [
    [1, 1, 1, 1],
    [1, 1, 1, 2],
    [1, 1, 1, 4],
    [2, 2, 2, 2],
]

TOTAL_SITE = (2, 2, 2, 2)
SEED = 731
SCALE = 0.7
MAX_ITER = 10000
EPS = 1.0e-10
ALPHA = 0.1

TOLERANCE_FUNCTIONAL = 1.0e-8
TOLERANCE_FIELD = 1.0e-6

def get_buffer_layout(U):
    """Global NumPy field -> (Nt,Nx,Ny,Nz,4,3,3), the qlat buffer layout."""
    return np.transpose(U, (1, 2, 3, 4, 0, 5, 6))

def get_local_buffer(arr, U, geo):
    """Pick the locally owned part of a global NumPy field."""
    xg_arr = geo.xg_arr()
    U_layout = get_buffer_layout(U)
    return U_layout[xg_arr[:, 0], xg_arr[:, 1], xg_arr[:, 2], xg_arr[:, 3]]

def gf_from_numpy(geo, U):
    """Create a qlat GaugeField from a global NumPy field (on every rank)."""
    gf = q.GaugeField(geo)
    arr = np.asarray(gf)
    assert arr.shape == (geo.local_volume, 4, 3, 3)
    arr[...] = get_local_buffer(arr, U, geo)
    return gf

def get_local_difference(gf, U, geo):
    """Max |gf - U| over the locally owned sites."""
    arr = np.asarray(gf)
    return float(np.max(np.abs(arr - get_local_buffer(arr, U, geo))))

def get_local_max_unitarity_error(arr):
    """Max |g g^dag - 1| of a local (local_volume, n, 3, 3) gauge-transform array."""
    return float(
        np.max(np.abs(arr @ np.conj(arr.transpose(0, 1, 3, 2)) - np.eye(3)[None, None]))
    )

def check_gauge(qr_fix, qg_fix, geo, fname):
    U = qr.get_random_gauge_field(TOTAL_SITE, seed=SEED, scale=SCALE)
    gf = gf_from_numpy(geo, U)
    gf_arr_before = np.asarray(gf).copy()
    gt_ref, ref_info = qr_fix(U, max_iter=MAX_ITER, eps=EPS, alpha=ALPHA)
    gt_qlat, gf_info = qg_fix(gf, max_iter=MAX_ITER, eps=EPS, alpha=ALPHA)
    gauge = ref_info["gauge"]
    functional_difference = abs(gf_info["functional"] - ref_info["functional"])
    # Apply both transforms to the SAME original field, then compare the
    # resulting fixed fields: the transforms themselves are path-dependent and
    # only defined up to the residual stabiliser.
    U_ref_fixed = qr.apply_gauge_transform(U, gt_ref)
    gf_qlat_fixed = gt_qlat * gf
    local_field_difference = get_local_difference(gf_qlat_fixed, U_ref_fixed, geo)
    unitarity_ref = qr.get_unitarity_error(U_ref_fixed)
    unitarity_qlat = q.get_unitarity_error(gf_qlat_fixed)
    # The transforms are not unique, so only compare their unitarity, through
    # numpy: `get_unitarity_error(gt)` is GaugeField-typed and rejects a
    # GaugeTransform (documented limitation of the getters).
    unitarity_gt_ref = qr.get_unitarity_error(gt_ref)
    unitarity_gt_qlat = get_local_max_unitarity_error(np.asarray(gt_qlat))
    assert ref_info["converged"]
    assert gf_info["converged"]
    assert ref_info["divergence"] <= EPS
    assert gf_info["divergence"] <= EPS
    assert functional_difference <= TOLERANCE_FUNCTIONAL
    assert local_field_difference <= TOLERANCE_FIELD
    assert unitarity_ref <= 1.0e-12
    assert unitarity_qlat <= 1.0e-12
    assert unitarity_gt_ref <= 1.0e-12
    assert unitarity_gt_qlat <= 1.0e-12
    # Neither implementation may modify its input field.
    assert np.array_equal(np.asarray(gf), gf_arr_before)
    q.displayln_info(
        f"{fname}: {gauge}: F_ref = {ref_info['functional']:.15e};"
        f" F_qlat = {gf_info['functional']:.15e};"
        f" |dF| = {functional_difference:.3e}"
    )
    q.displayln_info(
        f"{fname}: {gauge}: theta_ref = {ref_info['divergence']:.3e};"
        f" theta_qlat = {gf_info['divergence']:.3e}"
    )
    q.displayln_info(
        f"{fname}: {gauge}: max|U_ref_fixed - U_qlat_fixed| (local sites) ="
        f" {local_field_difference:.3e}; unitarity_ref = {unitarity_ref:.3e};"
        f" unitarity_qlat = {unitarity_qlat:.3e}"
    )
    q.displayln_info(
        f"{fname}: {gauge}: max|gt gt^dag - 1|: gt_ref = {unitarity_gt_ref:.3e};"
        f" gt_qlat = {unitarity_gt_qlat:.3e}"
    )

def main():
    fname = q.get_fname()
    q.begin_with_mpi(size_node_list)
    geo = q.Geometry(q.Coordinate(list(TOTAL_SITE)))
    q.displayln_info(
        f"{fname}: total_site = {TOTAL_SITE}; num_node = {q.get_num_node()};"
        f" local_volume = {geo.local_volume}"
    )
    check_gauge(qr.fix_landau_gauge, q.fix_landau_gauge, geo, fname)
    check_gauge(qr.fix_coulomb_gauge, q.fix_coulomb_gauge, geo, fname)
    q.end_with_mpi()
    q.displayln_info("CHECK: cross-check finished successfully.")

if __name__ == "__main__":
    main()
