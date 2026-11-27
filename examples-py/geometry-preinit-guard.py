#!/usr/bin/env python3

# Pre-initialisation guard for Geometry(total_site), and the empty-but-safe
# contract of a default-constructed Geometry().
#
# ``q.Geometry(q.Coordinate(total_site))`` divides total_site by the global
# size_node, which is unknown until q.begin_with_mpi() (or q.begin()) has run.
# It used to die with SIGFPE (exit 136); it must now raise RuntimeError.
#
# The child interpreter is used because the failure mode was a native signal,
# which would kill this test process instead of being reported.
#
# ``q.Geometry()`` itself is an empty but safe object (qlat-issue-9): an
# uninitialized geometry node with zero volumes, whose size/volume accessors
# return 0.  ``spatial_volume`` used to kill the interpreter with SIGFPE
# (integer divide by zero) and now returns 0 as well.

import os
import subprocess
import sys

import qlat as q

child_code = (
    "import sys\n"
    "import qlat as q\n"
    "try:\n"
    "    q.Geometry(q.Coordinate([4, 4, 4, 4]))\n"
    "except RuntimeError:\n"
    "    print('CHILD: RuntimeError')\n"
    "except BaseException as e:\n"
    "    print('CHILD: OTHER', type(e).__name__, e)\n"
    "else:\n"
    "    print('CHILD: NO-EXCEPTION')\n"
    "print('CHILD-DONE', sys.version.split()[0])\n"
)

child_env = dict(os.environ)
# inherit the parent's import path (PYTHONPATH from the environment alone is not
# enough once we are running from a copied work directory)
child_env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
r = subprocess.run(
    [sys.executable, "-c", child_code],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    env=child_env,
    capture_output=True,
    text=True,
    timeout=300,
)
child_outcome = "unexpected"
child_ok = False
for line in r.stdout.splitlines():
    if line.startswith("CHILD: "):
        word = line[7:]
        if word == "RuntimeError":
            child_outcome = "RuntimeError"
            child_ok = True
        elif word == "NO-EXCEPTION":
            child_outcome = "NO-EXCEPTION"
        else:
            child_outcome = f"unexpected:{word}"
q.json_results_append(f"pre-init Geometry(total_site) outcome = {child_outcome}")
q.json_results_append(f"pre-init Geometry(total_site) ok = {child_ok}")
q.json_results_append(f"pre-init child exit code = {r.returncode}")
if not child_ok:
    print("CHILD-STDOUT:", r.stdout)
    print("CHILD-STDERR:", r.stderr)

# the guard must not fire once qlat is initialised
q.begin_with_mpi([q.Coordinate([1, 1, 1, 1])])
geo = q.Geometry(q.Coordinate([4, 4, 4, 4]))
q.json_results_append(f"initialised Geometry(total_site) = {geo.total_site.to_list()}")

# Geometry() stays usable without initialisation, and so does the explicit
# (id_node, size_node, node_site) form, which does not consult the global geon
geo0 = q.Geometry()
q.json_results_append(f"Geometry() total_volume = {geo0.total_volume}")
geo1 = q.Geometry(0, q.Coordinate([1, 1, 1, 1]), q.Coordinate([4, 4, 4, 4]))
q.json_results_append(
    f"explicit Geometry(id, size, node) = {geo1.total_site.to_list()}"
)

# The empty-but-safe contract for a default-constructed Geometry()
# (qlat-issue-9): it is an uninitialized geometry node with zero volumes, and
# every size/volume accessor returns 0 instead of dividing by a zero extent
# (spatial_volume used to kill the interpreter with SIGFPE here).
q.json_results_append(f"Geometry() total_site = {geo0.total_site.to_list()}")
q.json_results_append(f"Geometry() node_site = {geo0.node_site.to_list()}")
q.json_results_append(f"Geometry() size_node = {geo0.size_node.to_list()}")
q.json_results_append(f"Geometry() coor_node = {geo0.coor_node.to_list()}")
q.json_results_append(f"Geometry() expansion_left = {geo0.expansion_left.to_list()}")
q.json_results_append(f"Geometry() expansion_right = {geo0.expansion_right.to_list()}")
q.json_results_append(f"Geometry() num_node = {geo0.num_node}")
q.json_results_append(f"Geometry() id_node = {geo0.id_node}")
q.json_results_append(f"Geometry() eo = {geo0.eo}")
q.json_results_append(f"Geometry() is_only_local = {geo0.is_only_local}")
q.json_results_append(f"Geometry() spatial_volume = {geo0.spatial_volume}")
q.json_results_append(f"Geometry() local_volume = {geo0.local_volume}")
q.json_results_append(
    f"Geometry() local_volume_expanded = {geo0.local_volume_expanded}"
)
q.json_results_append(f"Geometry() show = {geo0.show()}")
q.json_results_append(f"Geometry() == fresh Geometry() = {geo0 == q.Geometry()}")
q.json_results_append(f"Geometry() copy() == Geometry() = {geo0.copy() == geo0}")
q.json_results_append(
    f"Geometry() coordinate_from_index(0) = {geo0.coordinate_from_index(0).to_list()}"
)
q.json_results_append(
    f"Geometry() g_coordinate_from_index(0) = {geo0.g_coordinate_from_index(0).to_list()}"
)
q.json_results_append(f"Geometry() xg_arr shape = {list(geo0.xg_arr().shape)}")

# A geometry with one zero extent is empty as well, and equally safe
geo_zero = q.Geometry(0, q.Coordinate([1, 1, 1, 1]), q.Coordinate([4, 4, 4, 0]))
q.json_results_append(
    f"zero-extent Geometry total_site = {geo_zero.total_site.to_list()}"
)
q.json_results_append(f"zero-extent Geometry num_node = {geo_zero.num_node}")
q.json_results_append(f"zero-extent Geometry total_volume = {geo_zero.total_volume}")
q.json_results_append(
    f"zero-extent Geometry spatial_volume = {geo_zero.spatial_volume}"
)
q.json_results_append(f"zero-extent Geometry local_volume = {geo_zero.local_volume}")

q.timer_display()
if q.is_test():
    q.check_log_json(__file__, check_eps=1e-14)
q.end_with_mpi()
q.displayln_info("CHECK: finished successfully.")
