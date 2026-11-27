#!/usr/bin/env python3

import numpy as np

import qlat as q

size_node_list = [
    [1, 1, 1, 1],
    [1, 1, 1, 2],
    [1, 1, 1, 4],
    [1, 1, 1, 8],
    [2, 2, 2, 2],
    [2, 2, 2, 4],
]

def get_rng_draws(rs, n=4):
    return [rs.rand_gen() for i in range(n)]

def append_rng_draw_floats(name, draws):
    """
    Record floating-point draws as one float entry per value, with a relative
    ``eps``.\n
    The draws are obtained from the integer stream by a floating-point mapping,
    so their last bit may legitimately differ between machines (FMA,
    vectorisation, libm).  Recording a single string of exact ``float.hex()``
    values instead would make ``check_log_json`` compare that text exactly and
    fail on such a 1 ulp difference, which is not a change of the stream.  The
    integer stream itself is pinned exactly by the ``rand_gen`` entries, so a
    relative ``eps`` here keeps the values pinned without that fragility.
    """
    for idx, x in enumerate(draws):
        q.json_results_append(f"{name} draw {idx}", x, 1.0e-12)

def check_rng_labels():
    """
    RngState seed/split labels are byte strings (qlat-issue-1 and qlat-issue-6).\n
    The reference values below pin the streams of labels without an embedded NUL
    byte, so a change in the label formatting is caught.  Labels are length-aware
    byte strings: an embedded NUL byte takes part in the hash instead of
    truncating the label, and a ``bytes`` seed is used as raw bytes instead of
    being converted through ``repr()``.
    """
    # Fixed reference streams for labels without a NUL byte.
    rs = q.RngState("test-rng")
    q.json_results_append(f"RngState('abc') draws = {get_rng_draws(q.RngState('abc'))}")
    q.json_results_append(f"RngState('test-rng') draws = {get_rng_draws(rs)}")
    q.json_results_append(
        f"RngState('test-rng').split('a').split('b') = "
        f"{get_rng_draws(rs.split('a').split('b'))}"
    )
    q.json_results_append(
        f"RngState('test-rng').split(42) = {get_rng_draws(q.RngState(rs, 42))}"
    )
    q.json_results_append(f"RngState(42) = {get_rng_draws(q.RngState(42))}")
    # An int seed is its decimal string (C++ RngState(Long) uses show(Long)).
    assert get_rng_draws(q.RngState(rs, 42)) == get_rng_draws(rs.split("42"))
    assert get_rng_draws(q.RngState(42)) == get_rng_draws(q.RngState("42"))
    # qlat-issue-1: an embedded NUL byte is significant (it used to truncate).
    assert get_rng_draws(q.RngState("a\0bcdefg")) != get_rng_draws(q.RngState("a"))
    assert get_rng_draws(q.RngState("x\0")) != get_rng_draws(q.RngState("x"))
    assert get_rng_draws(q.RngState("\0abc")) != get_rng_draws(q.RngState(""))
    assert get_rng_draws(q.RngState("\0abc")) != get_rng_draws(q.RngState("abc"))
    assert get_rng_draws(rs.split("k\0zzz")) != get_rng_draws(rs.split("k"))
    assert get_rng_draws(rs.split("\0k")) != get_rng_draws(rs.split("k"))
    assert get_rng_draws(rs.split("a\0b").split("c\0d")) != (
        get_rng_draws(rs.split("a").split("c"))
    )
    q.json_results_append(
        f"RngState('a\\0bcdefg') draws = {get_rng_draws(q.RngState('a\0bcdefg'))}"
    )
    q.json_results_append(
        f"RngState('x').split('k\\0zzz') draws = "
        f"{get_rng_draws(q.RngState('x').split('k\0zzz'))}"
    )
    q.json_results_append(
        f"RngState('\\0abc') draws = {get_rng_draws(q.RngState('\0abc'))}"
    )
    # The NUL byte is what makes the difference: same bytes, same stream.
    assert get_rng_draws(q.RngState("a\0bcdefg")) == (
        get_rng_draws(q.RngState("a".encode() + b"\0bcdefg"))
    )
    # qlat-issue-6: a bytes seed is the raw bytes, not its repr().
    assert get_rng_draws(q.RngState(b"abc")) == get_rng_draws(q.RngState("abc"))
    assert get_rng_draws(q.RngState(b"abc")) != get_rng_draws(q.RngState("b'abc'"))
    assert get_rng_draws(q.RngState(b"a\0b")) == get_rng_draws(q.RngState("a\0b"))
    # Non-UTF-8 bytes are accepted as a label (no decoding through repr()).
    assert get_rng_draws(q.RngState(b"\xff")) != get_rng_draws(
        q.RngState(b"\xff".decode("latin-1"))
    )
    assert get_rng_draws(q.RngState(b"\xff")) == get_rng_draws(q.RngState(b"\xff"))
    # split already used raw bytes; keep it aligned with the constructor.
    assert get_rng_draws(rs.split(b"abc")) == get_rng_draws(rs.split("abc"))
    q.json_results_append(
        f"RngState(b'abc') draws = {get_rng_draws(q.RngState(b'abc'))}"
    )
    q.json_results_append(
        f"RngState(b'\\xff') draws = {get_rng_draws(q.RngState(b'\xff'))}"
    )

def check_u_rand_gen_bounds():
    """
    ``u_rand_gen(lower, upper)`` takes the lower bound first (qlat-issue-7).\n
    The order used to be ``(upper, lower)``.  Keyword calls keep their meaning
    and the draws must be bit-identical, while positional calls now mean
    ``[lower, upper)``.
    """
    # The default is [0, 1) and the keyword form is unchanged.
    rs = q.RngState("test-rng")
    q.json_results_append(f"u_rand_gen() draws = {get_rng_draws(rs, 4)}")
    rs = q.RngState("test-rng")
    append_rng_draw_floats(
        "u_rand_gen(lower=-1.0, upper=1.0)",
        [rs.u_rand_gen(lower=-1.0, upper=1.0) for _ in range(4)],
    )
    # Positional args are (lower, upper): every draw lies in [lower, upper).
    rs = q.RngState("test-rng")
    draws = [rs.u_rand_gen(2.0, 5.0) for _ in range(64)]
    assert all(2.0 <= x < 5.0 for x in draws)
    assert max(draws) - min(draws) > 1.0
    # The positional form equals the keyword form with the same bounds.
    rs_a = q.RngState("test-rng")
    rs_b = q.RngState("test-rng")
    assert [rs_a.u_rand_gen(2.0, 5.0) for _ in range(8)] == [
        rs_b.u_rand_gen(lower=2.0, upper=5.0) for _ in range(8)
    ]
    # These are the numbers the old ``u_rand_gen(upper=2.0, lower=5.0)`` drew.
    rs = q.RngState("test-rng")
    append_rng_draw_floats(
        "u_rand_gen(2.0, 5.0)", [rs.u_rand_gen(2.0, 5.0) for _ in range(4)]
    )
    # u_rand_arr follows the same (lower, upper) order: the filler helpers are
    # internal (qlat-issue-3), so build the range from the array API.
    rs = q.RngState("test-rng")
    u = rs.u_rand_arr((2, 3))
    assert np.all((0.0 <= u) & (u < 1.0))
    rs = q.RngState("test-rng")
    buf = rs.u_rand_arr((16,)) * 2.0 - 1.0
    assert np.all((-1.0 <= buf) & (buf < 1.0))
    append_rng_draw_floats("u_rand_arr(-1, 1)", [float(x) for x in buf[:4]])
    # Higher-level generators built on u_rand_gen are unchanged; these values
    # are pinned in the reference log.
    rs = q.RngState("test-rng")
    append_rng_draw_floats("g_rand_gen", [rs.g_rand_gen() for _ in range(4)])
    rs = q.RngState("test-rng")
    append_rng_draw_floats(
        "g_rand_arr", [float(x) for x in rs.g_rand_arr((2, 2)).ravel()]
    )
    rs = q.RngState("test-rng")
    q.json_results_append(
        f"rand_arr draws = {[int(x) for x in rs.rand_arr((2, 2)).ravel()]}"
    )

def check_rng_select():
    """
    ``RngState.select`` accepts any sized/indexable sequence (qlat-issue-7).\n
    It used to require a ``list``.  One draw is consumed per call, so the chosen
    indices for a given seed are the same for every container of one length.
    """
    lst = ["a", "b", "c", "d", "e"]
    rs = q.RngState("test-rng")
    q.json_results_append(f"select list = {[rs.select(lst) for _ in range(8)]}")
    # list / tuple / range / ndarray of the same length select the same indices.
    n = 7
    seeds = [f"select-{i}" for i in range(8)]
    picks = {}
    for kind, seq in [
        ("list", list(range(n))),
        ("tuple", tuple(range(n))),
        ("range", range(n)),
        ("ndarray", np.arange(n)),
    ]:
        picks[kind] = [q.RngState(sd).select(seq) for sd in seeds]
    assert picks["list"] == picks["tuple"] == picks["range"] == picks["ndarray"]
    assert picks["ndarray"] == [int(x) for x in picks["ndarray"]]
    q.json_results_append(
        f"select same indices for list/tuple/range/ndarray = {picks['list']}"
    )
    #
    # A custom sequence (only __len__ + __getitem__) works too.
    class CyclicSeq:
        def __len__(self):
            return 4
        #
        def __getitem__(self, i):
            return f"custom-{i}"
    #
    assert q.RngState("test-rng").select(CyclicSeq()).startswith("custom-")
    # A str is a sized/indexable sequence; its elements are 1-char strings.
    assert q.RngState("test-rng").select("abc") in "abc"
    #
    # Draw-stream identity: select() consumes exactly one rand_gen() draw.
    rs_a = q.RngState("test-rng")
    rs_b = q.RngState("test-rng")
    assert [rs_a.select(lst) for _ in range(8)] == [
        lst[rs_b.rand_gen() % len(lst)] for _ in range(8)
    ]
    #
    # Error contract: TypeError without len/getitem, ValueError when empty.
    for bad in (None, 5, 3.5):
        try:
            q.RngState("t").select(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"select({bad!r}) must raise TypeError")
    for empty in ([], (), range(0), np.array([])):
        try:
            q.RngState("t").select(empty)
        except ValueError:
            pass
        else:
            raise AssertionError(f"select({empty!r}) must raise ValueError")
    # A failed select() must not consume a draw when it raises before drawing.
    rs_c = q.RngState("test-rng")
    try:
        rs_c.select([])
    except ValueError:
        pass
    q.json_results_append(f"select after empty-sequence error = {rs_c.rand_gen()}")

def check_rng_state_api():
    """
    ``get_state`` / ``set_state`` / pickle round-trip the exact generator state
    (qlat-issue-7 "not picklable"), with a byte format shared with ``qlat_ref``.
    """
    import pickle
    #
    rs = q.RngState("state-api")
    [rs.rand_gen() for _ in range(5)]
    rs.g_rand_gen()  # leaves a value in the Box-Muller cache
    state = rs.get_state()
    assert isinstance(state, bytes) and len(state) == 104
    assert state[:7] == b"QLATRNG" and state[7] == 1
    # Continue from the original, then restore the saved state: must match.
    expected = [rs.rand_gen() for _ in range(8)]
    rs2 = q.RngState()
    rs2.set_state(state)
    assert [rs2.rand_gen() for _ in range(8)] == expected
    # The Box-Muller cache is part of the state too: two generators restored
    # from the same bytes produce the same cached Gaussian.
    rs_a = q.RngState()
    rs_a.set_state(state)
    rs_b = q.RngState()
    rs_b.set_state(state)
    assert rs_a.g_rand_gen() == rs_b.g_rand_gen()
    # pickle round-trip gives an independent generator on the same stream.
    rs_p = pickle.loads(pickle.dumps(q.RngState("state-api")))
    rs_q = q.RngState("state-api")
    assert [rs_p.rand_gen() for _ in range(4)] == [rs_q.rand_gen() for _ in range(4)]
    # bytearray / memoryview are accepted.
    rs4 = q.RngState()
    rs4.set_state(bytearray(state))
    rs5 = q.RngState()
    rs5.set_state(memoryview(state))
    assert rs4.get_state() == state and rs5.get_state() == state
    # Malformed states raise ValueError and leave the generator untouched.
    rs6 = q.RngState("keep-me")
    before = [rs6.rand_gen() for _ in range(3)]
    rs6 = q.RngState("keep-me")
    for bad in (
        b"",
        b"QLATRNG",
        state[:-1],
        state + b"\x00",
        b"XXXXXXXX" + state[8:],
        state[:7] + b"\x02" + state[8:],
    ):
        try:
            rs6.set_state(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"set_state({bad[:9]!r}...) must raise ValueError")
        assert [rs6.rand_gen() for _ in range(3)] == before
        rs6 = q.RngState("keep-me")
    # split() + state: a child restored from bytes continues the same stream.
    child = q.RngState("tree").split("branch").split("leaf")
    child_state = child.get_state()
    expected_child = [child.rand_gen() for _ in range(6)]
    child2 = q.RngState("tree").split("branch").split("leaf")
    child2.set_state(child_state)
    assert [child2.rand_gen() for _ in range(6)] == expected_child
    q.json_results_append(f"RngState state bytes = {state.hex()}")
    q.json_results_append(f"RngState state roundtrip draws = {expected}")

def check_rng_api_validation():
    """
    Argument validation raises explicitly, so it also applies under ``python -O``
    (qlat-issue-5).  Checked in a subprocess with and without ``-O``.
    """
    import subprocess
    import sys
    import tempfile
    #
    prog = (
        "import sys\n"
        "import qlat as q\n"
        "out = []\n"
        "for args in [(1, 2), (None, 2), ('s', 'x')]:\n"
        "    try:\n"
        "        q.RngState(*args)\n"
        "    except TypeError:\n"
        "        out.append('TypeError')\n"
        "    else:\n"
        "        out.append('no-error')\n"
        "print(','.join(out), file=open(sys.argv[1], 'w'))\n"
    )
    results = {}
    tmpdir = tempfile.mkdtemp()
    for mode, flags in (("normal", []), ("optimized", ["-O"])):
        # A file carries the result: `import qlat` also writes a banner to stdout.
        path = f"{tmpdir}/rng-validation-{mode}.txt"
        run = subprocess.run(
            [sys.executable] + flags + ["-c", prog, path],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr
        with open(path) as fp:
            results[mode] = fp.read().strip()
    normal_out = results["normal"]
    assert normal_out == "TypeError,TypeError,TypeError", normal_out
    assert normal_out == results["optimized"], results
    q.json_results_append(f"RngState validation identical under -O = {normal_out}")
    # select: TypeError without a sequence, ValueError when empty (qlat-issue-7).
    for bad in (None, 5):
        try:
            q.RngState("t").select(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"select({bad!r}) must raise TypeError")
        try:
            q.RngState("t").select(())
        except ValueError:
            pass
        else:
            raise AssertionError("select(()) must raise ValueError")
    # The whole obsolete fill* family is gone from the Python API
    # (qlat-issue-3): no fill name is public on RngState any more.
    rs = q.RngState("gone")
    for name in (
        "u_rand_fill",
        "g_rand_fill",
        "u_rand_fill_real_d",
        "g_rand_fill_real_d",
        "rand_fill_u_long",
    ):
        assert not hasattr(rs, name), f"{name} must not be public"
    assert [n for n in dir(rs) if "fill" in n.lower() and not n.startswith("_")] == []
    # Module level names are gone too.
    assert not hasattr(q, "u_rand_fill") and not hasattr(q, "g_rand_fill")
    # The supported array API still works.
    assert rs.rand_arr((2, 2)).dtype == np.uint64
    assert rs.u_rand_arr((2, 2)).dtype == np.float64
    assert rs.g_rand_arr((2, 2)).dtype == np.float64
    q.json_results_append("RngState fill* family removed from Python API = True")

q.begin_with_mpi(size_node_list)

check_rng_labels()
check_u_rand_gen_bounds()
check_rng_select()
check_rng_state_api()
check_rng_api_validation()

q.json_results_append("test LRUCache")

cache = q.LRUCache(4)
cache.clear()
v = 12 in cache
q.json_results_append(f"{v}")
cache[12] = "hello"
v = 12 in cache
q.json_results_append(f"{v}")
v = cache[12]
q.json_results_append(f"{v}")
for i in range(8):
    cache[i] = f"{i}"
v = len(cache.cache)
q.json_results_append(f"{v}")
v = cache.get(12)
q.json_results_append(f"{v}")
v = cache.get(12, "not found")
q.json_results_append(f"{v}")
v = list(cache.cache.items())
q.json_results_append(f"{v}")

q.json_results_append("test cache_call")

block_size = 10

block_size_dict = {
    "48I": 10,
}

@q.cache_call(
    maxsize=4,
)
def func(x):
    q.json_results_append(
        f"run: func({x}) ; block_size={block_size} ; block_size_dict={block_size_dict}"
    )
    return x, block_size, block_size_dict

for i in range(10):
    q.json_results_append(f"i={i} ; func(i % 4)")

@q.cache_call(
    maxsize=4,
    get_state=lambda: (
        block_size,
        block_size_dict,
    ),
)
def func(x):
    q.json_results_append(
        f"run: func({x}) ; block_size={block_size} ; block_size_dict={block_size_dict}"
    )
    return x, block_size, block_size_dict

for i in range(10):
    q.json_results_append(f"i={i} ; func(i % 4)")

block_size = 2

for i in range(10):
    q.json_results_append(f"i={i} ; func(i % 4)")

block_size_dict["64I"] = 20

for i in range(10):
    q.json_results_append(f"i={i} ; func(i % 4)")

for i in range(20):
    q.json_results_append(f"i={i} ; func(i % 10)")

@q.cache_call(
    maxsize=2,
    get_state=lambda: (
        block_size,
        block_size_dict,
    ),
    path="cache/func",
)
def func(x):
    q.json_results_append(
        f"run: func({x}) ; block_size={block_size} ; block_size_dict={block_size_dict}"
    )
    return x, block_size, block_size_dict

for i in range(10):
    q.json_results_append(f"i={i} ; func(i % 4)")

block_size = 2

for i in range(10):
    q.json_results_append(f"i={i} ; func(i % 4)")

block_size_dict["64I"] = 20

for i in range(20):
    q.json_results_append(f"i={i} ; func(i % 10)")

@q.cache_call(
    maxsize=4,
    get_state=lambda: (block_size,),
    is_hash_args=False,
)
def func(x):
    q.json_results_append(
        f"run: func({x}) ; block_size={block_size} ; block_size_dict={block_size_dict}"
    )
    return x, block_size, block_size_dict

for i in range(10):
    q.json_results_append(f"i={i} ; {func(i % 4)}")

with q.TimerFork():
    q.json_results_append("run: in TimerFork")

# Regression asserts for matrix scalar arithmetic / reflected ops / g5_herm.
# Asserts only: nothing is printed or recorded, so the reference logs are unchanged.
wm = q.WilsonMatrix()
wm[:] = 0.0
wm[0, 0] = 1.0 + 2.0j
wm[1, 0] = 3.0 - 4.0j
wm_ref = wm.copy()
assert 2 * wm == wm * 2
assert np.float64(2.0) * wm == wm * 2
assert 2 * wm == q.mat_mul_a_wm(2.0, wm)
assert wm == wm_ref
sm = q.SpinMatrix()
sm[:] = 0.0
sm[0, 0] = 1.0 + 2.0j
sm_ref = sm.copy()
assert 2 * sm == sm * 2
assert sm == sm_ref
cm = q.ColorMatrix()
cm[:] = 0.0
cm[0, 0] = 1.0 + 2.0j
cm_ref = cm.copy()
assert 2 * cm == cm * 2
assert cm == cm_ref
wm_g5_ref = q.wilson_matrix_g5_herm(wm_ref)
assert wm.g5_herm() is wm
assert wm == wm_g5_ref
for m in (wm, sm, cm):
    try:
        hash(m)
    except TypeError:
        pass
    else:
        raise AssertionError("matrix types must be unhashable")

q.timer_display()
if q.is_test():
    q.check_log_json(__file__, check_eps=1e-14)
q.end_with_mpi()
q.displayln_info("CHECK: finished successfully.")
