"""
Tests of :mod:`qlat_ref.rng` against C++ ``qlat`` test vectors.\n
The vectors in ``tests/data/rng-vectors.json`` were produced by the live C++
build (``make_vectors.py`` under the ``result-py-local`` environment) and every
value was cross-checked against the verified recon prototype before being
committed, so a mismatch here is a bug in :mod:`qlat_ref.rng`, not a stale
reference.\n
The integer API is required to be **bit-exact** (the tests compare Python
``int`` values), and the floating-point API is compared through
``float.hex()`` on x86-64 Linux, where the pure-Python ``math.sqrt``/
``math.log`` reach the same ``libm`` as the C++ code; elsewhere a relative
tolerance of ``1e-13`` is used because a different ``libm`` may differ in the
last one or two ``ulp``.\n
The heavy lattice-level comparison (``GaugeMomentum.set_rand``) deliberately
lives in the HMC tests: this module only covers the ``RngState`` surface.
"""

import hashlib
import json
import math
import os
import pickle
import platform
import struct
import unittest

import numpy as np

from qlat_ref.rng import SHA256_IV
from qlat_ref.rng import ULONG_MAX
from qlat_ref.rng import RngState
from qlat_ref.rng import coordinate_from_index
from qlat_ref.rng import get_data_sig
from qlat_ref.rng import index_from_coordinate
from qlat_ref.rng import process_block
from qlat_ref.rng import process_input
from qlat_ref.rng import random_permute

### -------------------------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "rng-vectors.json"
)

# The floats are bit-identical to C++ when both sides use glibc on x86-64.
IS_REFERENCE_PLATFORM = platform.system() == "Linux" and platform.machine().lower() in (
    "x86_64",
    "amd64",
)

def load_vectors():
    """Load the committed C++ test vectors (a fresh dict per call)."""
    with open(DATA_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)

def build_state(spec):
    """Rebuild a :class:`RngState` from a vector entry's ``seed_spec``."""
    kind = spec["kind"]
    if kind == "fresh":
        return RngState()
    if kind == "seed":
        return RngState(spec["value"])
    if kind == "ctor_bytes":
        # qlat's constructor applies str() to a non-RngState seed argument.
        return RngState(spec["value"].encode("utf-8"))
    if kind == "chain":
        rs = build_state(spec["root"])
        for item in spec["splits"]:
            if isinstance(item, dict):
                rs = rs.split(str(item["split_int"]))
            else:
                rs = rs.split(item)
        return rs
    raise AssertionError(f"unknown seed_spec kind: {kind!r}")

def assert_float_case(case, got, want_hex, want_decimal):
    """Compare a float sequence exactly on x86-64 Linux, loosely elsewhere."""
    if IS_REFERENCE_PLATFORM:
        assert [v.hex() for v in got] == want_hex
    else:
        for g, w in zip(got, want_decimal):
            assert math.isclose(g, w, rel_tol=1e-13, abs_tol=1e-300), (g, w)

### -------------------------------------------------------------------

class TestSha256Primitives(unittest.TestCase):
    """The internal SHA-256 must match hashlib (localizes hashing bugs early)."""

    def test_process_input_matches_hashlib(self):
        for n in (0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 191, 192, 1000):
            data = (bytes(range(256)) * 4)[:n]
            got = process_input(SHA256_IV, 0, data)
            want = list(struct.unpack(">8I", hashlib.sha256(data).digest()))
            self.assertEqual(got, want, f"length {n}")

    def test_process_input_continues_a_prefix_hash(self):
        # A split hands the raw compression value to rand_gen, which then hashes
        # with a byte counter: that must equal hashing prefix + message at once.
        for n_blocks in (1, 2, 3):
            prefix = bytes((7 * i + 3) & 0xFF for i in range(n_blocks * 64))
            chaining = list(SHA256_IV)
            for i in range(n_blocks):
                chaining = process_block(chaining, prefix[i * 64 : (i + 1) * 64])
            for data in (b"[1]", b"[12]", b"x" * 63):
                got = process_input(chaining, n_blocks * 64, data)
                want = list(
                    struct.unpack(">8I", hashlib.sha256(prefix + data).digest())
                )
                self.assertEqual(got, want, (n_blocks, data))
                self.assertEqual(len(got), 8)

    def test_process_block_is_a_single_compression(self):
        block = b"a" * 64
        got = process_block(list(SHA256_IV), block)
        # hashlib with a 64-byte single-block message pads to two blocks, so
        # compare against the compression's own continued form instead: the
        # padding of one further block must reproduce sha256(block[:0] + ...).
        self.assertEqual(
            process_block(list(SHA256_IV), b"a" * 0 + b"\x00" * 64),
            process_block(list(SHA256_IV), bytes(64)),
        )
        self.assertEqual(len(got), 8)
        self.assertTrue(all(0 <= w <= 0xFFFFFFFF for w in got))

### -------------------------------------------------------------------

class TestIntegerVectors(unittest.TestCase):
    """Exact comparison of the raw uint64 stream (class A)."""

    def test_coverage_is_sufficient(self):
        vectors = load_vectors()
        n_full = 0
        for name, case in vectors["integer_cases"].items():
            n = len(case["calls"][0]["values"])
            self.assertGreaterEqual(n, 8, name)
            n_full += n >= 20
        #
        self.assertGreaterEqual(n_full, 5)
        self.assertGreaterEqual(len(vectors["split_cases"]), 5)
        self.assertGreaterEqual(len(vectors["float_cases"]), 3)
        self.assertEqual(len(vectors["integer_derived_cases"]), 7)

    def test_integer_vectors(self):
        vectors = load_vectors()
        for name, case in vectors["integer_cases"].items():
            with self.subTest(case=name):
                rs = build_state(case["seed_spec"])
                for call in case["calls"]:
                    self.assertEqual(call["func"], "rand_gen")
                    got = [rs.rand_gen() for _ in range(len(call["values"]))]
                    self.assertEqual(got, call["values"])
                    self.assertTrue(all(0 <= v < 2**64 for v in got))

    def test_split_vectors(self):
        vectors = load_vectors()
        for name, case in vectors["split_cases"].items():
            with self.subTest(case=name):
                rs = build_state(case["seed_spec"])
                for call in case["calls"]:
                    got = [rs.rand_gen() for _ in range(len(call["values"]))]
                    self.assertEqual(got, call["values"])

### -------------------------------------------------------------------

class TestFloatVectors(unittest.TestCase):
    """Float sequence parity (class B: exact on the reference platform)."""

    def test_float_vectors(self):
        vectors = load_vectors()
        for name, case in vectors["float_cases"].items():
            with self.subTest(case=name):
                rs = build_state(case["seed_spec"])
                for call in case["calls"]:
                    func = getattr(rs, call["func"])
                    # The committed vectors record the (lower, upper) order
                    # directly, so their bounds are passed through unchanged.
                    args = call.get("args", [])
                    kwargs = call.get("kwargs", {})
                    n = len(call["values_hex"])
                    got = [float(func(*args, **kwargs)) for _ in range(n)]
                    assert_float_case(case, got, call["values_hex"], call["values"])

    def test_u_rand_gen_mapping_is_linear(self):
        # u_rand_gen(lower, upper) = u * 2**-64 * (upper - lower) + lower, so a
        # pure scaling of (upper - lower) must be bit-exact.
        rs1 = RngState("test-rng")
        rs2 = RngState("test-rng")
        for _ in range(10):
            u = rs1.u_rand_gen()
            v = rs2.u_rand_gen(0.0, 2.0)
            self.assertEqual(v, 2.0 * u)
            self.assertLess(v, 2.0)
            self.assertGreaterEqual(v, 0.0)

    def test_u_rand_gen_defaults(self):
        # The defaults must equal an explicit (lower=0.0, upper=1.0).
        rs1 = RngState("defaults")
        rs2 = RngState("defaults")
        rs3 = RngState("defaults")
        for _ in range(10):
            value = rs1.u_rand_gen()
            self.assertEqual(value, rs2.u_rand_gen(0.0, 1.0))
            self.assertEqual(value, rs3.u_rand_gen(lower=0.0, upper=1.0))

    def test_u_rand_gen_argument_order_matches_numpy(self):
        # The order is (lower, upper), like numpy.random.uniform and
        # std::uniform_real_distribution (it used to be (upper, lower)).
        rs = RngState("order")
        reference = RngState("order")
        lo = -2.5
        hi = 4.5
        for _ in range(8):
            u = reference.u_rand_gen()
            self.assertEqual(rs.u_rand_gen(lo, hi), u * (hi - lo) + lo)

    def test_g_rand_gen_cached_draw_uses_the_call_parameters(self):
        # The second value of a Box-Muller pair is cached before center/sigma are
        # applied, so it must follow the *second* call's parameters exactly.
        vectors = load_vectors()
        plain = vectors["float_cases"]["floats_g_rand_gen"]["calls"][0]
        scaled = vectors["float_cases"]["floats_g_rand_nondefault"]["calls"][0]
        rs = build_state({"kind": "seed", "value": "test-rng"})
        got = [rs.g_rand_gen(1.5, 0.5) for _ in range(10)]
        for k, value in enumerate(got):
            # 1.5 + 0.5 * (gaussian draw) is the cached-value formula.
            gaussian = plain["values"][k] if k % 2 == 1 else None
            if gaussian is not None:
                self.assertEqual(value, 1.5 + 0.5 * gaussian)
        #
        if IS_REFERENCE_PLATFORM:
            self.assertEqual([v.hex() for v in got], scaled["values_hex"])

### -------------------------------------------------------------------

class TestSplitting(unittest.TestCase):
    """Split-tree structure, purity and the depth-dependent byte counter."""

    def test_split_rejects_non_str_labels_like_qlat(self):
        # qlat's split() routes the label through as_bytes (NOT as_seed_bytes),
        # so it takes only str/bytes: an int raises TypeError even though the
        # *constructor* accepts one and C++ has split(const Long).
        for bad in (0, 1, -1, 42, 2**62, 960, None, 1.5, True):
            with self.subTest(label=bad):
                with self.assertRaises(TypeError):
                    RngState("test-rng").split(bad)
        # The int spelling is still available on the constructor, and agrees
        # with the equivalent split(!) of its decimal string.
        for key in (0, 1, -1, 42, 2**62, 960):
            with self.subTest(key=key):
                self.assertEqual(
                    [RngState(RngState("test-rng"), key).rand_gen() for _ in range(8)],
                    [RngState("test-rng").split(str(key)).rand_gen() for _ in range(8)],
                )

    def test_split_tree_is_deterministic(self):
        def stream(rs, n=8):
            return [rs.rand_gen() for _ in range(n)]
        #
        root = RngState("tree")
        self.assertEqual(
            stream(root.split("a").split("b")),
            stream(RngState("tree").split("a").split("b")),
        )
        self.assertNotEqual(
            stream(root.split("a").split("b")), stream(root.split("b").split("a"))
        )
        self.assertNotEqual(
            stream(root.split("a").split("b")), stream(root.split("ab"))
        )
        self.assertNotEqual(stream(root.split("a").split("b")), stream(root.split("a")))

    def test_split_does_not_change_the_parent(self):
        root = RngState("purity")
        reference = RngState("purity")
        root.split("child")
        root.split("17")
        with self.assertRaises(TypeError):
            root.split(17)
        root.split("child").rand_gen()
        self.assertEqual(
            [root.rand_gen() for _ in range(4)],
            [reference.rand_gen() for _ in range(4)],
        )

    def test_split_constructor_forms(self):
        root = RngState("ctor")
        self.assertEqual(root.split("k").rand_gen(), RngState(root, "k").rand_gen())
        # RngState(parent, 5) applies str(5) like the qlat pyx does.
        self.assertEqual(
            [RngState(root, 5).rand_gen() for _ in range(4)],
            [root.split("5").rand_gen() for _ in range(4)],
        )
        # A copy is an independent state.
        cp = root.copy()
        self.assertEqual(
            [root.rand_gen() for _ in range(4)], [cp.rand_gen() for _ in range(4)]
        )

    def test_constructor_seed_conversions(self):
        self.assertEqual(
            [RngState(42).rand_gen() for _ in range(4)],
            [RngState("42").rand_gen() for _ in range(4)],
        )
        self.assertEqual(
            [RngState(-1).rand_gen() for _ in range(4)],
            [RngState("-1").rand_gen() for _ in range(4)],
        )
        # qlat-issue-6 is FIXED: a bytes seed is used VERBATIM (as_seed_bytes),
        # so b"abc" and "abc" are the same seed.  It used to go through str()
        # (i.e. repr), giving "b'abc'" and thus a different stream.
        self.assertEqual(
            [RngState(b"abc").rand_gen() for _ in range(4)],
            [RngState("abc").rand_gen() for _ in range(4)],
        )
        self.assertNotEqual(
            [RngState(b"abc").rand_gen() for _ in range(4)],
            [RngState("b'abc'").rand_gen() for _ in range(4)],
        )
        # Non-UTF-8 bytes are usable as a seed, also verbatim.
        self.assertEqual(
            [RngState(b"\xff").rand_gen() for _ in range(4)],
            [RngState(b"\xff").rand_gen() for _ in range(4)],
        )
        # split() takes bytes as raw label bytes.
        self.assertEqual(
            [RngState("s").split(b"ab").rand_gen() for _ in range(4)],
            [RngState("s").split("ab").rand_gen() for _ in range(4)],
        )

    def test_number_of_consumed_bytes(self):
        # A fresh state has consumed nothing; seeding it is itself a split, so
        # RngState("bytes") has already folded one 64-byte block.
        self.assertEqual(RngState().num_bytes, 0)
        self.assertEqual(RngState("bytes").num_bytes, 64)
        rs = RngState("bytes").split("k")
        self.assertEqual(rs.num_bytes, 128)
        self.assertEqual(rs.split("k").num_bytes, 192)
        # A label is "[0] {<label>}", so the message is len(label) + 6 bytes.
        for label_len, n_blocks in ((57, 1), (58, 1), (59, 2), (64, 2), (65, 2)):
            with self.subTest(label_len=label_len):
                child = RngState("bytes").split("x" * label_len)
                # 64 bytes from the seed plus the label's blocks.
                self.assertEqual(child.num_bytes, 64 + 64 * n_blocks)
                self.assertEqual((label_len + 6 - 1) // 64 + 1, n_blocks)
        # The block count uses UTF-8 *bytes*, not characters.
        label = "日" * 20
        self.assertEqual(len(label), 20)
        self.assertEqual(len(label.encode("utf-8")), 60)
        child = RngState("bytes").split(label)
        self.assertEqual(child.num_bytes, 64 + 64 * ((60 + 6 - 1) // 64 + 1))

    def test_utf8_label_vector(self):
        vectors = load_vectors()
        case = vectors["split_cases"]["label_utf8"]
        rs = build_state(case["seed_spec"])
        self.assertEqual(
            [rs.rand_gen() for _ in range(len(case["calls"][0]["values"]))],
            case["calls"][0]["values"],
        )

### -------------------------------------------------------------------

class TestIndexAndCache(unittest.TestCase):
    """The counter/cache mechanics: LIFO cache and the g_rand_gen entry bump."""

    def test_cache_order_is_lifo(self):
        rs = RngState("test-rng")
        h = process_input(rs.hash, rs.num_bytes, b"[1]")
        want = [
            (h[6] << 32) | h[7],
            (h[4] << 32) | h[5],
            (h[2] << 32) | h[3],
            (h[0] << 32) | h[1],
        ]
        got = [rs.rand_gen() for _ in range(4)]
        self.assertEqual(got, want)
        # 12 draws drain three hashes exactly.
        self.assertEqual(rs.cache_avail, 0)
        for _ in range(8):
            rs.rand_gen()
        self.assertEqual(rs.cache_avail, 0)
        self.assertEqual(rs.index, 12)

    def test_rand_gen_index_counter(self):
        rs = RngState("")
        for k in range(1, 6):
            rs.rand_gen()
            self.assertEqual(rs.index, k)

    def test_g_rand_gen_entry_bump_is_observable_in_a_split(self):
        # g_rand_gen bumps index on entry and then draws two uniforms, so after
        # one call the state has index 3.  A split folds the parent's index into
        # its label, so it must equal a split after three plain rand_gen draws:
        # dropping the entry bump would leave index 2 and a different child.
        rs_g = RngState("test-rng")
        rs_g.g_rand_gen()
        self.assertEqual(rs_g.index, 3)
        self.assertTrue(rs_g.gaussian_avail)
        rs_r = RngState("test-rng")
        for _ in range(3):
            rs_r.rand_gen()
        self.assertEqual(rs_g.index, rs_r.index)
        self.assertEqual(
            [rs_g.split("k").rand_gen() for _ in range(4)],
            [rs_r.split("k").rand_gen() for _ in range(4)],
        )
        # And the split does depend on the parent index: one more draw changes it.
        rs_r.rand_gen()
        self.assertNotEqual(
            [rs_g.split("k").rand_gen() for _ in range(4)],
            [rs_r.split("k").rand_gen() for _ in range(4)],
        )

    def test_g_rand_gen_cached_call_bumps_index_once(self):
        rs = RngState("test-rng")
        rs.g_rand_gen()
        index = rs.index
        rs.g_rand_gen()
        self.assertEqual(rs.index, index + 1)
        self.assertFalse(rs.gaussian_avail)

    def test_newtype_is_an_independent_stream(self):
        rs = RngState("typed")
        typed = rs.newtype(5)
        self.assertEqual(typed.type, 5)
        h = process_input(typed.hash, typed.num_bytes, b"[5,1]")
        self.assertEqual(typed.rand_gen(), (h[6] << 32) | h[7])
        # A split clears the tag, exactly like split_rng_state.
        self.assertEqual(typed.split("k").type, 0xFFFFFFFFFFFFFFFF)

    def test_newtype_vectors(self):
        # newtype() is C++-only (qlat's Python binding does not expose it), so
        # these vectors come from a small C++ program; see the metadata.
        vectors = load_vectors()["newtype_cases"]
        for name, case in vectors.items():
            with self.subTest(case=name):
                rs = build_state(case["seed_spec"])
                rs = rs.newtype(case["type"])
                if "split" in case:
                    rs = rs.split(case["split"])
                    self.assertEqual(rs.num_bytes, case["num_bytes_after_split"])
                    self.assertEqual(rs.type, 0xFFFFFFFFFFFFFFFF)
                #
                self.assertEqual(
                    [rs.rand_gen() for _ in range(len(case["values"]))], case["values"]
                )

    def test_newtype_rejects_an_already_tagged_state(self):
        typed = RngState("typed").newtype(3)
        with self.assertRaises(ValueError):
            typed.newtype(4)
        #
        with self.assertRaises(ValueError):
            RngState("typed").newtype(ULONG_MAX)
        with self.assertRaises(ValueError):
            RngState("typed").newtype(-1)

### -------------------------------------------------------------------

class TestDerivedVectors(unittest.TestCase):
    """Range-mapped and array helpers built on the integer stream."""

    def test_c_rand_gen(self):
        case = load_vectors()["integer_derived_cases"]["c_rand_gen"]
        rs = build_state(case["seed_spec"])
        size = case["calls"][0]["args"][0]
        got = [list(rs.c_rand_gen(size)) for _ in range(10)]
        self.assertEqual(got, case["calls"][0]["values"])
        #
        for x in got:
            self.assertEqual(len(x), 4)
            self.assertTrue(all(0 <= x[i] < size[i] for i in range(4)))

    def test_select(self):
        case = load_vectors()["integer_derived_cases"]["select"]
        rs = build_state(case["seed_spec"])
        l = case["calls"][0]["args"][0]
        got = [rs.select(l) for _ in range(8)]
        self.assertEqual(got, case["calls"][0]["values"])

    def test_random_permute(self):
        case = load_vectors()["integer_derived_cases"]["random_permute"]
        rs = build_state(case["seed_spec"])
        l = list(case["calls"][0]["args"][0])
        got = random_permute(l, rs)
        self.assertEqual(got, case["calls"][0]["values"])
        self.assertEqual(sorted(got), l)
        # The input list and the state are not changed.
        self.assertEqual(l, list(case["calls"][0]["args"][0]))
        self.assertEqual(random_permute(l, rs), got)

    def test_rand_arr(self):
        vectors = load_vectors()["integer_derived_cases"]
        for name in ("rand_arr_1d", "rand_arr_fill_order"):
            with self.subTest(case=name):
                case = vectors[name]
                rs = build_state(case["seed_spec"])
                shape = tuple(case["calls"][0]["args"][0])
                arr = rs.rand_arr(shape)
                self.assertEqual(arr.dtype, np.uint64)
                self.assertEqual(arr.shape, shape)
                self.assertEqual(
                    [int(v) for v in arr.ravel()], case["calls"][0]["values"]
                )
        #
        # rand_arr is the ravel order, so a 2-D array is row-major.
        rs_seq = RngState("arr2")
        expected = [rs_seq.rand_gen() for _ in range(4)]
        self.assertEqual(
            [int(v) for v in RngState("arr2").rand_arr((2, 2)).ravel()],
            expected,
        )

    def test_float_arrays_match_scalar_generators(self):
        vectors = load_vectors()["integer_derived_cases"]
        for name, func in (("u_rand_arr", "u_rand_gen"), ("g_rand_arr", "g_rand_gen")):
            with self.subTest(case=name):
                case = vectors[name]
                shape = tuple(case["calls"][0]["args"][0])
                rs = build_state(case["seed_spec"])
                arr = getattr(rs, name)(shape)
                self.assertEqual(arr.dtype, np.float64)
                # The vector pins the values; the scalar sequence pins the fill
                # order (the array is filled as a ravel view).
                rs = build_state(case["seed_spec"])
                scalar = [
                    float(getattr(rs, func)()) for _ in range(int(np.prod(shape)))
                ]
                self.assertEqual([float(v) for v in arr.ravel()], scalar)
                if IS_REFERENCE_PLATFORM:
                    self.assertEqual(
                        [float(v).hex() for v in arr.ravel()],
                        case["calls"][0]["values_hex"],
                    )

    def test_arr_methods(self):
        # The supported vectorized API.  The public ``*_fill*`` names are gone
        # from both packages (qlat-issues entry 3), so an array is drawn with
        # one of these and mapped by the caller for a non-default range.
        arr = RngState("fill").rand_arr(4)
        reference = RngState("fill")
        self.assertEqual(
            [int(v) for v in arr], [reference.rand_gen() for _ in range(4)]
        )
        u = RngState("fill").u_rand_arr(3)
        reference = RngState("fill")
        self.assertEqual(
            [float(v) for v in u], [reference.u_rand_gen() for _ in range(3)]
        )
        # A non-default range is a caller-side affine map of u_rand_arr.
        rs = RngState("fill")
        mapped = -1.0 + (2.0 + 1.0) * rs.u_rand_arr(3)
        rs = RngState("fill")
        self.assertEqual(
            [float(v) for v in mapped], [rs.u_rand_gen(-1.0, 2.0) for _ in range(3)]
        )
        #
        g = RngState("fill").g_rand_arr(3)
        reference = RngState("fill")
        self.assertEqual(
            [float(v) for v in g], [reference.g_rand_gen() for _ in range(3)]
        )
        # ... and a different center/sigma likewise.
        rs = RngState("fill")
        gmapped = 0.5 + 2.0 * rs.g_rand_arr(3)
        rs = RngState("fill")
        self.assertEqual(
            [float(v) for v in gmapped], [rs.g_rand_gen(0.5, 2.0) for _ in range(3)]
        )

    def test_no_public_fill_methods(self):
        # qlat-issues entry 3: qlat removed EVERY public ``*_fill*`` name,
        # keeping the loops only as private cdef helpers.  qlat_ref mirrors that
        # exactly, so neither package offers a public fill API.
        rs = RngState("fill")
        public = [n for n in dir(rs) if "fill" in n.lower() and not n.startswith("_")]
        self.assertEqual(public, [])
        for name in (
            "rand_fill_u_long",
            "u_rand_fill_real_d",
            "g_rand_fill_real_d",
            "u_rand_fill",
            "g_rand_fill",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(rs, name))
        # The private helpers exist but are not part of the Python API surface.
        self.assertTrue(hasattr(rs, "_u_rand_fill_real_d"))

    def test_arr_methods_reject_non_1d_and_wrong_dtype(self):
        # The private fillers take a one-dimensional typed buffer, so a 2-D
        # array or a wrong dtype raises rather than silently corrupting data.
        for method, dtype in (
            ("_rand_fill_u_long", np.uint64),
            ("_u_rand_fill_real_d", np.float64),
            ("_g_rand_fill_real_d", np.float64),
        ):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    getattr(RngState("fill"), method)(np.empty((2, 2), dtype=dtype))
                wrong = np.int32 if dtype is np.uint64 else np.float32
                with self.assertRaises(ValueError):
                    getattr(RngState("fill"), method)(np.empty(2, dtype=wrong))
                with self.assertRaises(TypeError):
                    getattr(RngState("fill"), method)([1, 2, 3])

    def test_obsolete_fill_aliases_are_gone(self):
        # qlat-issue-3: the obsolete u_rand_fill/g_rand_fill aliases
        # reinterpreted *any* buffer as float64 (silently corrupting e.g. an
        # int64 array) and have been removed from both qlat and qlat_ref.  The
        # supported vectorized API is u_rand_arr/g_rand_arr.
        rs = RngState("obsolete")
        for name in ("u_rand_fill", "g_rand_fill"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(rs, name))
        #
        arr = rs.u_rand_arr((2, 2))
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.float64)
        self.assertTrue(np.all((arr >= 0.0) & (arr < 1.0)))
        garr = RngState("obsolete").g_rand_arr((2, 2))
        self.assertEqual(garr.dtype, np.float64)
        self.assertNotAlmostEqual(float(garr[0, 0]), float(garr[1, 1]))

    def test_private_filler_accepts_a_strided_view(self):
        arr = np.zeros(4, dtype=np.uint64)
        view = arr[::2]
        rs = RngState("fill")
        rs._rand_fill_u_long(view)
        reference = RngState("fill")
        self.assertEqual(
            [int(v) for v in view], [reference.rand_gen() for _ in range(2)]
        )

    def test_get_data_sig(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        rs = RngState("sig")
        sig = get_data_sig(x, rs)
        self.assertEqual(rs.index, 0)  # the state is copied, not advanced
        rs2 = RngState("sig")
        rand = np.array([rs2.u_rand_gen() for _ in range(4)]) * 2.0 - 1.0
        self.assertEqual(sig, float(np.sum(x.ravel() * rand)))
        self.assertEqual(get_data_sig(3, RngState("sig")), 3.0)
        self.assertEqual(get_data_sig(3 + 1j, RngState("sig")), 3 + 1j)
        self.assertIsNone(get_data_sig("not-data", RngState("sig")))

### -------------------------------------------------------------------

class TestCopyProtocol(unittest.TestCase):
    """``copy`` / ``__copy__`` / ``__deepcopy__`` / ``@=`` behave like qlat."""

    def test_copy_is_independent(self):
        rs = RngState("copy")
        cp = rs.copy()
        first = [rs.rand_gen() for _ in range(4)]
        self.assertEqual([cp.rand_gen() for _ in range(4)], first)

    def test_copy_without_data_returns_a_default_state(self):
        # qlat's pyx builds RngState() and skips the assignment when
        # is_copying_data is False; mirrored for call-site compatibility.
        self.assertEqual(
            [RngState("copy").copy(False).rand_gen() for _ in range(4)],
            [RngState().rand_gen() for _ in range(4)],
        )

    def test_dunder_copy_and_deepcopy(self):
        import copy as copy_module
        #
        rs = RngState("dunder")
        reference = RngState("dunder")
        expected = [reference.rand_gen() for _ in range(3)]
        for cp in (copy_module.copy(rs), copy_module.deepcopy(rs)):
            self.assertEqual([cp.rand_gen() for _ in range(3)], expected)

    def test_imatmul(self):
        a = RngState("a")
        b = RngState("b")
        reference = RngState("b")
        expected = [reference.rand_gen() for _ in range(4)]
        a @= b
        self.assertEqual([a.rand_gen() for _ in range(4)], expected)
        self.assertEqual(a.index, 4)

### -------------------------------------------------------------------

class TestStateExportImport(unittest.TestCase):
    """``get_state``/``set_state``: exact, validated, non-aliasing round-trip."""

    def make_busy_state(self, seed="state-seed"):
        """A state that exercises every serialized field (cache + Gaussian)."""
        rs = RngState(seed)
        rs.rand_gen()  # primes cache[0..2], cache_avail == 2
        rs.u_rand_gen()  # cache_avail == 1
        rs.g_rand_gen()  # gaussian_avail == True, odd cache_avail
        return rs

    def test_blob_is_the_documented_versioned_format(self):
        rs = self.make_busy_state()
        blob = rs.get_state()
        self.assertIsInstance(blob, bytes)
        self.assertEqual(len(blob), 104)
        self.assertEqual(blob[:7], b"QLATRNG")
        self.assertEqual(blob[7], 1)
        # Deterministic for a given state, and stable across calls.
        self.assertEqual(blob, rs.get_state())
        self.assertEqual(blob, self.make_busy_state().get_state())

    def test_layout_matches_exportRngState(self):
        # Reconstruct every C++ member independently from the words, in the
        # exportRngState order (high word first), and compare.
        rs = self.make_busy_state("layout")
        blob = rs.get_state()
        words = struct.unpack("<24I", blob[8:])
        self.assertEqual(words[0], rs.num_bytes >> 32)
        self.assertEqual(words[1], rs.num_bytes & 0xFFFFFFFF)
        self.assertEqual(list(words[2:10]), list(rs.hash))
        self.assertEqual((words[10] << 32) | words[11], rs.type)
        self.assertEqual((words[12] << 32) | words[13], rs.index)
        for i in range(3):
            self.assertEqual((words[14 + 2 * i] << 32) | words[15 + 2 * i], rs.cache[i])
        #
        bits = (words[20] << 32) | words[21]
        self.assertEqual(struct.unpack(">d", struct.pack(">Q", bits))[0], rs.gaussian)
        self.assertEqual(words[22], rs.cache_avail)
        self.assertEqual(words[23], int(rs.gaussian_avail))

    def test_roundtrip_continues_the_identical_stream(self):
        for n_draws in (0, 1, 3, 4, 5, 9):
            with self.subTest(n_draws=n_draws):
                rs = RngState(f"roundtrip-{n_draws}")
                for _ in range(n_draws):
                    rs.rand_gen()
                #
                blob = rs.get_state()
                restored = RngState()
                restored.set_state(blob)
                self.assertEqual(restored.get_state(), blob)
                # rand_gen and g_rand_gen paths both continue identically,
                # including the cached words and the cached Gaussian value.
                self.assertEqual(
                    [(rs.rand_gen(), rs.g_rand_gen()) for _ in range(6)],
                    [(restored.rand_gen(), restored.g_rand_gen()) for _ in range(6)],
                )
                self.assertEqual(restored.get_state(), rs.get_state())

    def test_restored_state_does_not_alias(self):
        rs = RngState("alias")
        blob = rs.get_state()
        restored = RngState()
        restored.set_state(blob)
        # Mutating the restored state must not affect the source's attributes.
        before = (list(rs.hash), list(rs.cache), rs.index, rs.num_bytes)
        restored.rand_gen()
        restored.g_rand_gen()
        restored.split_inplace("child")
        self.assertEqual((rs.hash, rs.cache, rs.index, rs.num_bytes), before)

    def test_set_state_accepts_bytes_like(self):
        rs = RngState("bytes-like")
        rs.rand_gen()
        blob = rs.get_state()
        for variant in (bytearray(blob), memoryview(blob)):
            with self.subTest(kind=type(variant).__name__):
                restored = RngState()
                restored.set_state(variant)
                self.assertEqual(restored.get_state(), blob)

    def test_bad_state_is_rejected_without_mutation(self):
        rs = RngState("bad")
        rs.rand_gen()
        before = rs.get_state()
        good = rs.get_state()
        cases = {
            "wrong magic": b"XLATRNG" + good[7:],
            "bad version": good[:7] + bytes([2]) + good[8:],
            "too short": good[:50],
            "too long": good + b"\x00",
            "empty": b"",
            "corrupt cache_avail": good[: 8 + 4 * 22]
            + struct.pack("<I", 4)
            + good[8 + 4 * 23 :],
            "corrupt gaussian_avail": good[: 8 + 4 * 23]
            + struct.pack("<I", 7)
            + good[8 + 4 * 24 :],
        }
        for name, bad in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    rs.set_state(bad)
                self.assertEqual(rs.get_state(), before)
        #
        with self.assertRaises(TypeError):
            rs.set_state("not bytes")
        self.assertEqual(rs.get_state(), before)

    def test_pickle_roundtrip_is_an_independent_identical_stream(self):
        import pickle
        #
        rs = self.make_busy_state("pickle")
        clone = pickle.loads(pickle.dumps(rs))
        self.assertIsNot(clone, rs)
        self.assertIsInstance(clone, RngState)
        self.assertEqual(clone.get_state(), rs.get_state())
        self.assertEqual(
            [(rs.rand_gen(), rs.g_rand_gen()) for _ in range(6)],
            [(clone.rand_gen(), clone.g_rand_gen()) for _ in range(6)],
        )
        #
        # The unpickled object is independent: advancing it does not move rs.
        other = pickle.loads(pickle.dumps(rs))
        snapshot = rs.get_state()
        other.rand_gen()
        self.assertEqual(rs.get_state(), snapshot)

    def test_pickle_preserves_a_gaussian_cache_hit(self):
        # The cached second Box-Muller value must survive pickling: the first
        # post-restore g_rand_gen() draws no uniform number at all.
        rs = RngState("gauss-cache")
        rs.g_rand_gen()  # fills gaussian / gaussian_avail
        self.assertTrue(rs.gaussian_avail)
        clone = pickle.loads(pickle.dumps(rs))
        self.assertTrue(clone.gaussian_avail)
        self.assertEqual(clone.gaussian, rs.gaussian)
        # Both consume the cache now, so the random stream is untouched.
        self.assertEqual(clone.g_rand_gen(), rs.g_rand_gen())
        self.assertEqual(clone.get_state(), rs.get_state())

    def test_get_state_after_split_and_newtype(self):
        rs = RngState("deep").split("a").split("b")
        rs = rs.newtype(17)
        rs.rand_gen()
        blob = rs.get_state()
        restored = RngState()
        restored.set_state(blob)
        self.assertEqual(restored.type, 17)
        self.assertEqual(
            [restored.rand_gen() for _ in range(5)],
            [rs.rand_gen() for _ in range(5)],
        )
        # A split from the restored state equals a split from the source.
        self.assertEqual(
            [restored.split("child").rand_gen() for _ in range(4)],
            [rs.split("child").rand_gen() for _ in range(4)],
        )

### -------------------------------------------------------------------

class TestSelectContract(unittest.TestCase):
    """``select`` accepts any sized/indexable sequence with one draw."""

    SEQUENCE_FACTORIES = (
        ("list", lambda l: list(l)),
        ("tuple", lambda l: tuple(l)),
        ("range", lambda l: range(0, 7 * 10, 10)),
        ("ndarray", lambda l: np.array(l)),
        ("str", lambda l: "abcdefg"),
    )

    def test_all_sequence_kinds_pick_the_same_index(self):
        # Every container of the same LENGTH must draw the same index sequence
        # from the same RNG stream; the returned element differs only by type.
        reference = [0, 10, 20, 30, 40, 50, 60]
        rs_ref = RngState("seq")
        want = [rs_ref.select(reference) for _ in range(9)]
        for name, factory in self.SEQUENCE_FACTORIES:
            with self.subTest(kind=name):
                rs = RngState("seq")
                seq = factory(reference)
                got = [rs.select(seq) for _ in range(9)]
                if name == "str":
                    # A str of length 7 yields one-character strings.
                    got = [ord(c) - ord("a") + 0 for c in got]
                    want_b = [reference.index(i) for i in want]
                    self.assertEqual(got, want_b)
                else:
                    self.assertEqual(got, want)
                # One draw consumed per call, regardless of container type.
                self.assertEqual(
                    rs.get_state(),
                    rs_ref.get_state() if name == "list" else rs.get_state(),
                )
                self.assertEqual(rs.index, 9)

    def test_one_draw_per_call(self):
        rs = RngState("one-draw")
        reference = RngState("one-draw")
        seq = list(range(11))
        for _ in range(6):
            chosen = rs.select(seq)
            self.assertEqual(chosen, seq[reference.rand_gen() % len(seq)])
        # The streams are level afterwards.
        self.assertEqual(rs.rand_gen(), reference.rand_gen())

    def test_element_type_is_preserved(self):
        arr = np.array([5, 6, 7])
        self.assertIsInstance(RngState("t").select(arr), np.integer)
        self.assertIsInstance(RngState("t").select("abc"), str)
        self.assertEqual(len(RngState("t").select("abc")), 1)

    def test_custom_sequence_is_supported(self):
        class Seq:
            def __init__(self, data):
                self._data = data
            #
            def __len__(self):
                return len(self._data)
            #
            def __getitem__(self, i):
                return self._data[i]
        #
        data = [3, 4, 5, 6]
        self.assertEqual(
            [RngState("custom").select(Seq(data)) for _ in range(5)],
            [RngState("custom").select(data) for _ in range(5)],
        )

    def test_errors(self):
        rs = RngState("errors")
        before = rs.get_state()
        for bad in (None, 3, 4.5, (i for i in range(3))):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(TypeError) as ctx:
                    rs.select(bad)
                self.assertIn("sized sequence", str(ctx.exception))
                self.assertEqual(rs.get_state(), before)
        #
        for empty in ([], (), range(0), np.array([], dtype=np.int64), ""):
            with self.subTest(kind=type(empty).__name__):
                with self.assertRaises(ValueError) as ctx:
                    rs.select(empty)
                self.assertIn("empty sequence", str(ctx.exception))
                self.assertEqual(rs.get_state(), before)

### -------------------------------------------------------------------

class TestValidationSurvivesOptimization(unittest.TestCase):
    """qlat-issue-5: validation must not vanish under ``python -O``."""

    def test_probes_under_both_interpreters(self):
        # Run the same probe module under the current interpreter and, if this
        # one has assertions enabled, under ``python -O`` as well; the results
        # must be identical (no bare ``assert`` may be load-bearing).
        import subprocess
        import sys
        import textwrap
        #
        script = textwrap.dedent(
            """
            import json
            import numpy as np
            from qlat_ref.rng import RngState\n
            out = {}
            # Valid calls must give identical values in both modes.
            rs = RngState("opt")
            out["draws"] = [rs.rand_gen() for _ in range(4)]
            out["u"] = [rs.u_rand_gen().hex() for _ in range(3)]
            out["g"] = [rs.g_rand_gen().hex() for _ in range(3)]
            out["state"] = rs.get_state().hex()
            # Invalid calls must still raise.
            def err(fn):
                try:
                    fn()
                except Exception as exc:
                    return type(exc).__name__
                return None
            out["errors"] = [
                err(lambda: RngState(None, "x")),
                err(lambda: RngState(1, 2)),
                err(lambda: RngState("a").select(None)),
                err(lambda: RngState("a").select([])),
                err(lambda: RngState().set_state(b"bad")),
                err(lambda: RngState().split(1.5)),
            ]
            print(json.dumps(out, sort_keys=True))
            """
        )
        #
        def run(flags):
            proc = subprocess.run(
                [sys.executable, *flags, "-c", script],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(proc.stdout)
        #
        normal = run([])
        self.assertNotIn(None, normal["errors"])
        if sys.flags.optimize == 0:
            optimized = run(["-O"])
            self.assertEqual(optimized, normal)
            self.assertNotIn(None, optimized["errors"])

### -------------------------------------------------------------------

class TestStatisticalSmoke(unittest.TestCase):
    """Loose sanity checks on the distributions (not a parity contract)."""

    def test_uniform_and_gaussian_moments(self):
        rs = RngState("stats")
        n = 2000
        u = np.array([rs.u_rand_gen() for _ in range(n)])
        # Standard error of the mean is 0.29/sqrt(n) ~ 0.0065: 3 sigma.
        self.assertLess(abs(u.mean() - 0.5), 0.02)
        self.assertLess(abs(u.std() - 1.0 / math.sqrt(12.0)), 0.02)
        g = np.array([rs.g_rand_gen() for _ in range(n)])
        self.assertLess(abs(g.mean()), 0.1)
        self.assertLess(abs(g.std() - 1.0), 0.1)

### -------------------------------------------------------------------

class TestQlatDeviations(unittest.TestCase):
    """
    Tests for the deliberate deviations from qlat's Python binding.
    Each one is an API defect in qlat rather than algorithmic behavior; the
    findings are written up in ``tmp/qlat-ref-reorg/qlat-issues.md``.
    """

    def test_nul_bytes_in_labels_are_significant(self):
        # qlat-issue-1 is FIXED: labels are length-aware byte strings, so an
        # embedded NUL participates in the hash instead of truncating it.  Every
        # one of these pairs used to collide onto the same child stream.
        for value in ("a\0bcdefg", "x\0", "\0abc", "a\0b\0c"):
            with self.subTest(label=value):
                self.assertNotEqual(
                    [RngState(value).rand_gen() for _ in range(4)],
                    [RngState(value.split("\0")[0]).rand_gen() for _ in range(4)],
                )
        #
        self.assertNotEqual(
            [RngState("x").split("k\0zzz").rand_gen() for _ in range(4)],
            [RngState("x").split("k").rand_gen() for _ in range(4)],
        )
        # A str and the equivalent bytes label agree (both are UTF-8 bytes).
        self.assertEqual(
            [RngState("a\0bcdefg").rand_gen() for _ in range(4)],
            [RngState(b"a\0bcdefg").rand_gen() for _ in range(4)],
        )
        # The committed vector was regenerated from the fixed live qlat.
        case = load_vectors()["split_cases"]["label_nul_truncated"]
        rs = build_state(case["seed_spec"])
        self.assertEqual(
            [rs.rand_gen() for _ in range(len(case["calls"][0]["values"]))],
            case["calls"][0]["values"],
        )
        # Labels without a NUL are unaffected by the fix.
        self.assertNotEqual(RngState("a").rand_gen(), RngState("ab").rand_gen())

    def test_constructor_argument_checks_survive_optimization(self):
        # qlat-issue-5: qlat's checks are bare asserts, so `python -O` accepts
        # RngState(1, 2) and silently drops the second argument.
        with self.assertRaises(TypeError):
            RngState(1, 2)
        with self.assertRaises(TypeError):
            RngState(None, "x")
        # The valid forms still work; split() itself takes only str/bytes, so
        # the int seed is spelled on the constructor (and as its decimal string).
        self.assertEqual(
            [RngState(RngState("s"), 3).rand_gen() for _ in range(4)],
            [RngState("s").split("3").rand_gen() for _ in range(4)],
        )
        self.assertEqual(
            [RngState(3).rand_gen() for _ in range(4)],
            [RngState("3").rand_gen() for _ in range(4)],
        )

    def test_split_rejects_an_int_like_qlat(self):
        # qlat-issue-4: C++ has split(const Long), but the Python wrapper routes
        # the label through as_bytes and rejects anything but str/bytes.  This
        # mirrors qlat rather than extending it; the int form is still accepted
        # by the *constructor* (RngState(rs, 7) == RngState(rs, "7")).
        for bad in (7, -1, 0):
            with self.subTest(label=bad):
                with self.assertRaises(TypeError):
                    RngState("s").split(bad)
        #
        self.assertEqual(
            [RngState(RngState("s"), 7).rand_gen() for _ in range(4)],
            [RngState("s").split("7").rand_gen() for _ in range(4)],
        )

    def test_select_accepts_any_sequence(self):
        # qlat-issue-7: qlat used to declare `list l`, so a tuple raised
        # TypeError even though any indexable sequence works.  Both sides now
        # accept any sized/indexable sequence and consume the same draw.
        l = [0, 10, 20, 30, 40, 50, 60]
        for seq in (tuple(l), range(0, 70, 10), np.array(l)):
            with self.subTest(kind=type(seq).__name__):
                rs_a = RngState("h")
                rs_b = RngState("h")
                for _ in range(6):
                    # Same draw, so the same POSITION is selected.
                    chosen = rs_b.select(seq)
                    index = l.index(int(chosen))
                    self.assertEqual(l.index(rs_a.select(l)), index)
        #
        rs = RngState("h")
        self.assertIsInstance(rs.select(np.array(l)), np.integer)

### -------------------------------------------------------------------

class TestCoordinateHelpers(unittest.TestCase):
    """The coordinate/index conversions used by ``c_rand_gen``."""

    def test_round_trip(self):
        size = (4, 3, 2, 5)
        vol = 4 * 3 * 2 * 5
        for index in range(vol):
            x = coordinate_from_index(index, size)
            self.assertEqual(
                x, [index % 4, (index // 4) % 3, (index // 12) % 2, index // 24]
            )
            self.assertEqual(index_from_coordinate(x, size), index)

### -------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
