#!/usr/bin/env python3

"""
Regenerate ``rng-vectors.json`` from the live C++ ``qlat`` build.\n
Not part of the installed package and not run by the test suite: the committed
JSON is the reference.  This script exists so the vectors can be regenerated
deliberately (e.g. after an algorithm change in ``qlat``) and so the provenance
of every value is reproducible.\n
Usage (from the repository root)::\n
    source result-py-local/bin/setenv-qlat.sh
    python3 qlat-ref/tests/data/make-rng-vectors.py\n
It writes ``rng-vectors.json`` next to itself.  Every value is generated twice —
once through the live ``qlat_utils.rng_state.RngState`` and once through
:mod:`qlat_ref.rng` — and the script fails unless the two agree exactly (integers)
or bitwise (``float.hex()``), so a wrong expectation cannot be committed.\n
One set of cases cannot be produced from qlat's Python API: ``newtype`` is
C++-only, so its vectors are read from the ``newtype-vectors.json`` file emitted
by ``newtype-probe.cpp`` in this directory (see the comment at the top of that
file for the compile command).  Everything else is generated from the live
build.
"""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np

import qlat as q

from qlat_ref import rng as refrng

N_INT = 20

### --- seed trees -------------------------------------------------------

def build_q(spec):
    """Build a live qlat ``RngState`` from a vector ``seed_spec``."""
    kind = spec["kind"]
    if kind == "fresh":
        return q.RngState()
    if kind == "seed":
        return q.RngState(spec["value"])
    if kind == "ctor_bytes":
        return q.RngState(spec["value"].encode("utf-8"))
    if kind == "chain":
        rs = build_q(spec["root"])
        for item in spec["splits"]:
            if isinstance(item, dict):
                # C++ split(Long) is split(show(Long)); qlat's Python split()
                # only takes str, so use the equivalent string form.
                rs = rs.split(str(item["split_int"]))
            else:
                rs = rs.split(item)
        return rs
    raise AssertionError(f"unknown seed_spec kind: {kind!r}")

def build_ref(spec):
    """The same tree in the pure-Python reference implementation."""
    kind = spec["kind"]
    if kind == "fresh":
        return refrng.RngState()
    if kind == "seed":
        return refrng.RngState(spec["value"])
    if kind == "ctor_bytes":
        return refrng.RngState(spec["value"].encode("utf-8"))
    if kind == "chain":
        rs = build_ref(spec["root"])
        for item in spec["splits"]:
            if isinstance(item, dict):
                # C++ split(Long) == split(show(Long)); qlat's Python split()
                # accepts only str/bytes, so both packages use the string form.
                rs = rs.split(str(item["split_int"]))
            else:
                rs = rs.split(item)
        return rs
    raise AssertionError(f"unknown seed_spec kind: {kind!r}")

### --- one call series --------------------------------------------------

def call_spec(func, args, kwargs, n, spec, float_out=False):
    """
    Return the vector entry of one call series, cross-checked against qlat_ref.\n
    ``spec`` is the ``seed_spec`` to build both states from; every case passes it
    explicitly so a stale module-level "current spec" cannot silently mislabel a
    case.
    """
    rs_q = build_q(spec)
    rs_r = build_ref(spec)
    call = {"func": func, "args": list(args)}
    if kwargs:
        call["kwargs"] = dict(kwargs)
    if func == "rand_gen":
        vals_q = [int(rs_q.rand_gen()) for _ in range(n)]
        vals_r = [int(rs_r.rand_gen()) for _ in range(n)]
    elif func == "u_rand_gen":
        vals_q = [float(rs_q.u_rand_gen(*args, **kwargs)) for _ in range(n)]
        vals_r = [float(rs_r.u_rand_gen(*args, **kwargs)) for _ in range(n)]
    elif func == "g_rand_gen":
        vals_q = [float(rs_q.g_rand_gen(*args, **kwargs)) for _ in range(n)]
        vals_r = [float(rs_r.g_rand_gen(*args, **kwargs)) for _ in range(n)]
    elif func == "c_rand_gen":
        size = q.Coordinate(list(args[0]))
        vals_q = [tuple(int(v) for v in rs_q.c_rand_gen(size)) for _ in range(n)]
        vals_r = [rs_r.c_rand_gen(list(args[0])) for _ in range(n)]
    elif func == "select":
        vals_q = [int(rs_q.select(list(args[0]))) for _ in range(n)]
        vals_r = [int(rs_r.select(list(args[0]))) for _ in range(n)]
    elif func == "random_permute":
        vals_q = [int(x) for x in q.random_permute(list(args[0]), build_q(spec))]
        vals_r = [int(x) for x in refrng.random_permute(list(args[0]), build_ref(spec))]
    elif func == "rand_arr":
        vals_q = [int(x) for x in rs_q.rand_arr(tuple(args[0])).ravel()]
        vals_r = [int(x) for x in rs_r.rand_arr(tuple(args[0])).ravel()]
    elif func == "u_rand_arr":
        vals_q = [float(x) for x in rs_q.u_rand_arr(tuple(args[0])).ravel()]
        vals_r = [float(x) for x in rs_r.u_rand_arr(tuple(args[0])).ravel()]
    elif func == "g_rand_arr":
        vals_q = [float(x) for x in rs_q.g_rand_arr(tuple(args[0])).ravel()]
        vals_r = [float(x) for x in rs_r.g_rand_arr(tuple(args[0])).ravel()]
    else:
        raise AssertionError(f"unknown func: {func!r}")
    #
    if float_out:
        assert [v.hex() for v in vals_q] == [v.hex() for v in vals_r], (
            f"{func}{args}{kwargs} disagrees between qlat and qlat_ref: "
            f"{[(a, b) for a, b in zip(vals_q, vals_r) if a.hex() != b.hex()][:3]}"
        )
        call["values_hex"] = [v.hex() for v in vals_q]
        call["values"] = [float(v) for v in vals_q]
    else:
        assert vals_q == vals_r, (
            f"{func}{args}{kwargs} disagrees between qlat and qlat_ref: "
            f"{[(a, b) for a, b in zip(vals_q, vals_r) if a != b][:3]}"
        )
        call["values"] = [list(v) if isinstance(v, tuple) else int(v) for v in vals_q]
    #
    return call

def add_case(target, name, spec, calls, note=None):
    """Insert one case (a seed tree plus its call series) into ``target``."""
    entry = {"seed_spec": spec, "calls": calls}
    if note is not None:
        entry["note"] = note
    target[name] = entry

### --- the cases --------------------------------------------------------

def build_integer_cases():
    """Raw ``rand_gen`` streams: the primary exact-parity contract."""
    cases = {}
    trees = [
        ("root_test_rng", {"kind": "seed", "value": "test-rng"}, None),
        (
            "root_test_rng_split_0",
            {
                "kind": "chain",
                "root": {"kind": "seed", "value": "test-rng"},
                "splits": ["0"],
            },
            None,
        ),
        (
            "root_test_rng_split_a",
            {
                "kind": "chain",
                "root": {"kind": "seed", "value": "test-rng"},
                "splits": ["a"],
            },
            None,
        ),
        (
            "chain_a_b",
            {
                "kind": "chain",
                "root": {"kind": "seed", "value": "test-rng"},
                "splits": ["a", "b"],
            },
            None,
        ),
        ("root_empty_seed", {"kind": "seed", "value": ""}, None),
        (
            "root_int_42",
            {"kind": "seed", "value": 42},
            "an int seed is str()ed, so this is RngState('42')",
        ),
        ("root_int_0", {"kind": "seed", "value": 0}, None),
        (
            "root_fresh_state",
            {"kind": "fresh"},
            "RngState() is a bare SHA-256 context: no seed split was applied",
        ),
        (
            "root_long_key_200",
            {"kind": "seed", "value": "z" * 200},
            "a 200-byte seed spans several 64-byte blocks",
        ),
        (
            "lattice_v1_split_site_0_0_0_0",
            {
                "kind": "chain",
                "root": {"kind": "seed", "value": "lattice-v1"},
                "splits": ["(0,0,0,0)"],
            },
            "the per-site label form used by qlat's lattice scripts",
        ),
    ]
    for name, spec, note in trees:
        add_case(cases, name, spec, [call_spec("rand_gen", [], {}, N_INT, spec)], note)
    #
    spec = {"kind": "seed", "value": "test-rng"}
    add_case(
        cases,
        "raw_words_test_rng",
        spec,
        [call_spec("rand_gen", [], {}, 12, spec)],
        "12 draws consume 3 SHA-256 invocations: pins the 3-deep LIFO word cache",
    )
    return cases

def build_split_cases():
    """Split-tree structure: int keys, depth, byte-length semantics, ctor quirks."""
    cases = {}
    trees = [
        (
            "split_int_0",
            {
                "kind": "chain",
                "root": {"kind": "seed", "value": "test-rng"},
                "splits": [{"split_int": 0}],
            },
            "C++ split(Long 0) == split(str(0))",
        ),
        (
            "split_int_neg1",
            {
                "kind": "chain",
                "root": {"kind": "seed", "value": "test-rng"},
                "splits": [{"split_int": -1}],
            },
            None,
        ),
        (
            "split_int_2p62",
            {
                "kind": "chain",
                "root": {"kind": "seed", "value": "test-rng"},
                "splits": [{"split_int": 2**62}],
            },
            None,
        ),
        (
            "chain_deep_6",
            {
                "kind": "chain",
                "root": {"kind": "seed", "value": "deep"},
                "splits": ["a", "b", "c", "d", "e", "f"],
            },
            "6-deep chain: numBytes accumulates 6*64 on top of the seed's 64",
        ),
        (
            "label_utf8",
            {
                "kind": "chain",
                "root": {"kind": "seed", "value": "utf8"},
                "splits": ["héllo-日本"],
            },
            "UTF-8 label: the block count uses the byte length, not the character count",
        ),
    ]
    # 64-byte block-boundary labels: the message is "[0] {<label>}", i.e. 6 bytes
    # more than the label.
    for i in (57, 58, 59, 62, 63, 64, 65, 66):
        trees.append(
            (
                f"label_{i}_bytes",
                {
                    "kind": "chain",
                    "root": {"kind": "seed", "value": "bytes"},
                    "splits": ["x" * i],
                },
                f"label of {i} bytes: the message is {i + 6} bytes, so "
                f"nBlocks = {(i + 6 - 1) // 64 + 1}",
            )
        )
    #
    for name, spec, note in trees:
        add_case(cases, name, spec, [call_spec("rand_gen", [], {}, 8, spec)], note)
    #
    spec = {
        "kind": "chain",
        "root": {"kind": "seed", "value": "test-rng"},
        "splits": ["5"],
    }
    add_case(
        cases,
        "ctor_parent_split_5",
        spec,
        [call_spec("rand_gen", [], {}, 8, spec)],
        "qlat's RngState(rs, 5) applies str(5), so it equals rs.split('5')",
    )
    #
    spec = {"kind": "ctor_bytes", "value": "abc"}
    add_case(
        cases,
        "ctor_bytes_quirk",
        spec,
        [call_spec("rand_gen", [], {}, 8, spec)],
        "qlat's RngState.__init__ does str(x) for a non-RngState x, so a bytes seed "
        "is seeded through its repr string b'abc' and not as the raw bytes",
    )
    #
    spec = {"kind": "seed", "value": "a\x00bcdefg"}
    add_case(
        cases,
        "label_nul_truncated",
        spec,
        [call_spec("rand_gen", [], {}, 8, spec)],
        "qlat formats split labels with printf's %s, so a label containing a NUL "
        "byte is silently truncated: this seed behaves exactly like 'a' "
        "(qlat-issue-1).  Reproduced on purpose for bit-exact parity; see the "
        "qlat-issues write-up.",
    )
    return cases

def build_float_cases():
    """The floating-point paths, including the Box-Muller cache."""
    cases = {}
    add_case(
        cases,
        "floats_u_rand_gen",
        {"kind": "seed", "value": "test-rng"},
        [
            call_spec(
                "u_rand_gen",
                [],
                {},
                10,
                {"kind": "seed", "value": "test-rng"},
                float_out=True,
            )
        ],
    )
    add_case(
        cases,
        "floats_u_rand_range",
        {"kind": "seed", "value": "test-rng"},
        [
            call_spec(
                "u_rand_gen",
                (-1.0, 2.5),
                {},
                10,
                {"kind": "seed", "value": "test-rng"},
                float_out=True,
            )
        ],
        "pins the mapping u * 2**-64 * (upper - lower) + lower",
    )
    add_case(
        cases,
        "floats_g_rand_gen",
        {"kind": "seed", "value": "test-rng"},
        [
            call_spec(
                "g_rand_gen",
                [],
                {},
                10,
                {"kind": "seed", "value": "test-rng"},
                float_out=True,
            )
        ],
        "pins the polar Box-Muller path and the extra index bump on entry",
    )
    add_case(
        cases,
        "floats_g_rand_nondefault",
        {"kind": "seed", "value": "test-rng"},
        [
            call_spec(
                "g_rand_gen",
                (),
                {"center": 1.5, "sigma": 0.5},
                10,
                {"kind": "seed", "value": "test-rng"},
                float_out=True,
            )
        ],
        "pins that center/sigma are applied to the cached draw of a pair too",
    )
    add_case(
        cases,
        "floats_g_rand_many",
        {"kind": "seed", "value": "g-many"},
        [
            call_spec(
                "g_rand_gen",
                [],
                {},
                32,
                {"kind": "seed", "value": "g-many"},
                float_out=True,
            )
        ],
        "32 draws: includes rejection retries and repeated cache pairs",
    )
    spec = {
        "kind": "chain",
        "root": {"kind": "seed", "value": "test-rng"},
        "splits": ["metropolis_accept"],
    }
    add_case(
        cases,
        "metropolis_uniform",
        spec,
        # (lower, upper) after the order swap; the committed values were drawn
        # with the old (upper=0.0, lower=1.0) spelling, i.e. this same range.
        [call_spec("u_rand_gen", (1.0, 0.0), {}, 5, spec, float_out=True)],
        "the uniform draw used by metropolis_accept",
    )
    return cases

def build_derived_cases():
    """Range-mapped and array helpers built on the integer stream."""
    cases = {}
    spec = {"kind": "seed", "value": "helpers"}
    add_case(
        cases,
        "c_rand_gen",
        spec,
        [call_spec("c_rand_gen", ([4, 4, 4, 4],), {}, 10, spec)],
        "c_rand_gen(size) = coordinate_from_index(rand_gen() % volume(size), size), "
        "component 0 fastest",
    )
    spec = {"kind": "seed", "value": "helpers"}
    add_case(
        cases,
        "select",
        spec,
        [call_spec("select", ([0, 10, 20, 30, 40, 50, 60],), {}, 8, spec)],
        "select(l) = l[rand_gen() % len(l)]",
    )
    spec = {"kind": "seed", "value": "perm2"}
    add_case(
        cases,
        "random_permute",
        spec,
        [call_spec("random_permute", ([0, 1, 2, 3, 4, 5, 6, 7],), {}, 1, spec)],
        "Fisher-Yates with k ascending and kk = rand_gen() % (size - k); leaves the "
        "input list and the passed RngState alone",
    )
    for name, value, func, args, float_out, note in (
        (
            "rand_arr_fill_order",
            "arr2",
            "rand_arr",
            ([2, 2],),
            False,
            "filled in C (row-major) order, as the ravel() view does",
        ),
        ("rand_arr_1d", "arr2", "rand_arr", ([4],), False, None),
        ("u_rand_arr", "uarr", "u_rand_arr", ([3],), True, None),
        ("g_rand_arr", "garr", "g_rand_arr", ([3],), True, None),
    ):
        spec = {"kind": "seed", "value": value}
        add_case(
            cases,
            name,
            spec,
            [call_spec(func, args, {}, 1, spec, float_out=float_out)],
            note,
        )
    return cases

def build_newtype_cases():
    """``newtype`` cases come from the C++ probe (qlat Python does not expose it)."""
    path = os.path.join(HERE, "newtype-vectors.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing; compile and run newtype-probe.cpp first "
            f"(see the comment at the top of that file)"
        )
    #
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)["newtype_cases"]

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
    """Build and write ``rng-vectors.json``, cross-checking every value."""
    q.begin_with_mpi()
    try:
        payload = {
            "metadata": {
                "description": "C++ qlat RngState test vectors for qlat_ref.rng "
                "(SHA-256 counter mode)",
                "provenance": {
                    "qlat_version": importlib.metadata.version("qlat"),
                    "qlat_utils_version": importlib.metadata.version("qlat-utils"),
                    "generator": "qlat-ref/tests/data/make-rng-vectors.py "
                    "(run under result-py-local/bin/setenv-qlat.sh)",
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "libc": list(platform.libc_ver()),
                    "git_commit": git_commit(),
                    "note": "Every value was cross-checked against qlat_ref.rng before being "
                    "written, so a mismatch between the two implementations aborts "
                    "this script instead of committing a wrong expectation.",
                },
                "exactness": {
                    "integer": "bit-exact uint64; the tests compare Python ints with ==",
                    "float": "values_hex is the bit pattern produced by the x86-64 glibc "
                    "build; the tests compare hex exactly on x86-64 Linux and fall "
                    "back to 1e-13 relative tolerance elsewhere",
                },
                "split_semantics": "split(key) formats '[index] {key}' (or '[type,index] "
                "{key}') padded with ASCII spaces to a multiple of 64 "
                "bytes, feeds it to the raw SHA-256 compression function "
                "without final padding, and advances numBytes by the "
                "padded byte count",
                "python_api_notes": {
                    "split_accepts_int_in_qlat_python": False,
                    "split_int_note": "split() accepts only str/bytes in BOTH packages "
                    "(it routes through as_bytes), so an int raises TypeError; C++ has "
                    "split(Long) == split(show(Long)), and the int spelling lives on the "
                    "constructor: RngState(rs, 7) == rs.split('7').  qlat_ref.rng no "
                    "longer accepts an int in split() (it used to, as a documented "
                    "extension, and that was removed for parity).",
                    "ctor_bytes_note": "FIXED: RngState.__init__ uses as_seed_bytes, so a "
                    "bytes seed is used VERBATIM (b'abc' == 'abc') instead of going "
                    "through repr(); labels also carry embedded NULs verbatim",
                    "no_newtype_in_python": True,
                    "state_export_in_python": "RngState.get_state()/set_state() + "
                    "__reduce__ expose the 104-byte state (magic b'QLATRNG' + version 1 "
                    "+ 24 LE uint32 in exportRngState order)",
                },
                "maintenance_note": "These vectors are a committed reference, not a snapshot "
                "to regenerate casually: the tests compare them exactly, "
                "so a mismatch means the implementation changed.  "
                "Regenerate only if the C++ algorithm changes.",
                "algorithm": "SHA-256 counter mode over a fixed chaining value: rand_gen "
                "hashes '[index]' continuing the SHA-256 prefix with a byte "
                "counter, returns the last two 32-bit words as one big-endian "
                "uint64, and caches the first three words LIFO",
            },
            "integer_cases": build_integer_cases(),
            "split_cases": build_split_cases(),
            "float_cases": build_float_cases(),
            "integer_derived_cases": build_derived_cases(),
            "newtype_cases": build_newtype_cases(),
        }
    finally:
        q.end_with_mpi()
    #
    out = os.path.join(HERE, "rng-vectors.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=1)
        fp.write("\n")
    #
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    print(
        f"{len(payload['integer_cases'])} integer cases, "
        f"{len(payload['split_cases'])} split cases, "
        f"{len(payload['float_cases'])} float cases, "
        f"{len(payload['integer_derived_cases'])} derived cases, "
        f"{len(payload['newtype_cases'])} newtype cases"
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
