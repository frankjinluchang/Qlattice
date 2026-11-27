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
# COULOMB GAUGE DOES NOT USE ``q.fix_coulomb_gauge``.  Component 3 is time in
# qlat's own convention (``Geometry::total_site`` is (Lx, Ly, Lz, Lt),
# ``Geometry::spatial_volume`` divides out component 3, ``get_psel_tslice``
# slices ``t_dir=3``, and ``qlat_gpt.gauge_fix_coulomb`` separates axis 3), so
# the spatial directions are {0, 1, 2}.  ``q.fix_coulomb_gauge`` uses
# {1, 2, 3} instead -- it fixes the three directions that *include* time and
# excludes spatial direction 0.  That is qlat-issue-8 in
# ``tmp/qlat-ref-reorg/qlat-issues.md``, and the user decided ``qlat-ref``
# implements the CORRECT set rather than reproducing the bug.
#
# The cross-check therefore compares ``qr.fix_coulomb_gauge`` against
# ``q.fix_gauge(gf, dirs=[0, 1, 2], ...)``, which runs the *same* local
# steepest-descent algorithm on the correct directions (verified: qlat's own
# ``fix_coulomb_gauge`` is exactly ``fix_gauge(dirs=[1, 2, 3])`` to round-off,
# so ``fix_gauge`` is a faithful reference and only the direction set differs).
# Comparing against ``q.fix_coulomb_gauge`` would make this script assert that
# ``qlat-ref`` reproduces a bug, which is not what it is for.
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

def import_qlat_ref():
    """
    Import `qlat_ref`, preferring a build that has the topic modules.\n
    The pure-NumPy package was reorganised from one ``qlat_ref.gauge`` module
    into ``base``/``mat``/``field``/``observables``/``gauge_fixing``/
    ``gauge_action``.  An installed copy from before that reorganisation still
    imports fine but has the pre-reorganisation API, and using it would make
    this cross-check measure stale code.  ``qlat_ref`` is pure Python, so
    falling back to the sibling source tree is safe and always current.
    """
    try:
        import qlat_ref as qr
        #
        if hasattr(qr, "gauge_fixing"):
            return qr
        # Stale installed build: drop it from the module cache so that the
        # source tree below is actually the one that gets imported.
        for name in [
            n for n in sys.modules if n == "qlat_ref" or n.startswith("qlat_ref.")
        ]:
            del sys.modules[name]
    except ImportError:
        pass
    repl = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.join(repl, "qlat-ref")
    if not os.path.isdir(src):
        raise ImportError(
            "neither an installed qlat_ref with the topic modules nor a"
            f" sibling qlat-ref/ source tree at {src}"
        )
    sys.path.insert(0, src)
    import qlat_ref as qr
    #
    if not hasattr(qr, "gauge_fixing"):
        raise ImportError(
            f"the qlat_ref at {qr.__file__} has no topic modules; rebuild the"
            " nix environment or update the checkout"
        )
    q.displayln_info(
        f"gauge-fixing-cross-check: using the source tree qlat_ref at"
        f" {qr.__file__} (the installed package is stale; rebuild the nix"
        " environment to test the installed copy)"
    )
    return qr

qr = import_qlat_ref()

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

# The spatial directions, i.e. all directions except time.  Component 3 is
# time, so this is (0, 1, 2) -- NOT qlat's (1, 2, 3).  See the top-of-file
# comment and qlat-issue-8.
SPATIAL_DIRS = (0, 1, 2)

def check_gauge(qr_fix, qg_fix, geo, fname, qg_dirs=None):
    U = qr.get_random_gauge_field(TOTAL_SITE, seed=SEED, scale=SCALE)
    gf = gf_from_numpy(geo, U)
    gf_arr_before = np.asarray(gf).copy()
    gt_ref, ref_info = qr_fix(U, max_iter=MAX_ITER, eps=EPS, alpha=ALPHA)
    if qg_dirs is None:
        gt_qlat, gf_info = qg_fix(gf, max_iter=MAX_ITER, eps=EPS, alpha=ALPHA)
    else:
        # Compare against qlat's own algorithm on EXPLICIT directions, so that
        # a qlat direction-set bug cannot silently become the reference.
        gt_qlat, gf_info = qg_fix(
            gf, dirs=list(qg_dirs), max_iter=MAX_ITER, eps=EPS, alpha=ALPHA
        )
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
    # The transforms are not unique, so only compare their unitarity.  Both
    # getters accept a GaugeField and a GaugeTransform (qlat-issue-11), so the
    # typed call works on both sides and no numpy workaround is needed.
    unitarity_gt_ref = qr.get_unitarity_error(gt_ref)
    unitarity_gt_qlat = q.get_unitarity_error(gt_qlat)
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

def check_coulomb_direction_bug(geo, fname):
    """
    Document qlat-issue-8: `q.fix_coulomb_gauge` fixes the WRONG directions.\n
    Component 3 is time, so the spatial directions are (0, 1, 2).  qlat's
    `fix_coulomb_gauge` uses (1, 2, 3) instead, which means:
      * its fixed field satisfies the gauge condition for (1, 2, 3) and NOT
        for (0, 1, 2), and
      * it is exactly `q.fix_gauge(gf, dirs=[1, 2, 3])`, so the algorithm is
        fine and only the direction set is wrong.\n
    This is a *reporting* check, not a correctness assert on qlat: it pins the
    observed qlat behaviour so that an upstream fix will show up here as a
    change, and it proves that qlat_ref's deviation is observable (the two
    gauge conditions disagree) rather than cosmetic.
    """
    U = qr.get_random_gauge_field(TOTAL_SITE, seed=SEED, scale=SCALE)
    gf = gf_from_numpy(geo, U)
    gt_qlat, info_qlat = q.fix_coulomb_gauge(
        gf, max_iter=MAX_ITER, eps=EPS, alpha=ALPHA
    )
    fixed_qlat = gt_qlat * gf
    theta_spatial = q.get_gauge_divergence(fixed_qlat, list(SPATIAL_DIRS))
    theta_qlat = q.get_gauge_divergence(fixed_qlat, [1, 2, 3])
    # qlat converged its own (wrong) condition but not the correct one.
    assert theta_qlat <= EPS, (
        f"{fname}: qlat fix_coulomb_gauge did not converge its own dirs:"
        f" theta[1,2,3] = {theta_qlat:.3e}"
    )
    assert theta_spatial > 1.0e-2, (
        f"{fname}: qlat fix_coulomb_gauge unexpectedly converged the spatial"
        f" dirs (0,1,2): theta = {theta_spatial:.3e}; has qlat-issue-8 been"
        " fixed upstream?  If so, update qlat-ref and its parity allowlist."
    )
    # The algorithm is the same, only the direction set differs.
    gt_explicit, info_explicit = q.fix_gauge(
        gf, dirs=[1, 2, 3], max_iter=MAX_ITER, eps=EPS, alpha=ALPHA
    )
    agree = np.max(np.abs(np.asarray(gt_explicit * gf) - np.asarray(fixed_qlat)))
    assert agree <= TOLERANCE_FIELD, (
        f"{fname}: fix_coulomb_gauge is not fix_gauge(dirs=[1,2,3]): {agree:.3e}"
    )
    q.displayln_info(
        f"{fname}: coulomb: qlat-issue-8: fix_coulomb_gauge uses dirs (1,2,3);"
        f" theta(1,2,3) = {theta_qlat:.3e} (converged);"
        f" theta(0,1,2) = {theta_spatial:.3e} (NOT converged);"
        f" it equals fix_gauge(dirs=[1,2,3]) to {agree:.3e}."
    )
    q.displayln_info(
        f"{fname}: coulomb: qlat_ref fixes the correct spatial dirs (0,1,2);"
        " no assert compares it against fix_coulomb_gauge on purpose."
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
    # Coulomb: qlat_ref fixes the correct spatial directions, so compare with
    # qlat's fix_gauge on those same directions (see the top-of-file comment).
    check_gauge(qr.fix_coulomb_gauge, q.fix_gauge, geo, fname, qg_dirs=SPATIAL_DIRS)
    check_coulomb_direction_bug(geo, fname)
    q.end_with_mpi()
    q.displayln_info("CHECK: cross-check finished successfully.")

if __name__ == "__main__":
    main()
