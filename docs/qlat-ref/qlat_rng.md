# `qlat_ref.rng` — Deterministic SHA-256 Random Number Generator (NumPy Reference)

Source: `qlat-ref/qlat_ref/rng.py`

> **Note:** Update this document when updating the source file.

## Outline

1. [Overview](#overview)
2. [Algorithm](#algorithm)
3. [Array Layout and Exactness](#array-layout-and-exactness)
4. [Public Interface](#public-interface)
5. [State Export, Import and Pickling](#state-export-import-and-pickling)
6. [Serialized State Format](#serialized-state-format)
7. [Sequence Selection](#sequence-selection)
8. [Examples](#examples)
9. [Deviations from `qlat`](#deviations-from-qlat)
10. [Self-Checks](#self-checks)

---

## Overview

`qlat_ref.rng` is a pure-Python, NumPy-only re-implementation of
`qlat_utils.RngState`, the deterministic random number generator of `qlat`.

It is not a NumPy `Generator`: the generator is a **SHA-256 counter mode**
stream.  The state is a 256-bit SHA-256 chaining value plus a byte counter, and
every draw hashes the ASCII counter string `"[index]"` (or `"[type,index]"`
when the stream is tagged) *continuing* that chaining value.

The module exists to cross-check the C++/Cython implementation in `qlat`; it is
meant for reference-scale lattices (`2^4` to `4^4`) and for test vectors, not
for production volumes.  One draw costs roughly 10 microseconds and one split
roughly 300 microseconds.

## Algorithm

Each `RngState` stores

| Attribute | C++ member | Meaning |
|---|---|---|
| `num_bytes` | `numBytes` | message length already consumed by the hash |
| `hash` | `hash` | 8 × `uint32`, the 256-bit SHA-256 chaining value |
| `type` | `type` | optional stream tag (`ULONG_MAX` when unset) |
| `index` | `index` | number of draws made on this stream |
| `cache` | `cache` | 3 cached `uint64` draws |
| `cache_avail` | `cacheAvail` | how many of `cache` are still unused (LIFO) |
| `gaussian` | `gaussian` | cached second Box-Muller value |
| `gaussian_avail` | `gaussianAvail` | whether `gaussian` is valid |

A call to `rand_gen()` increments `index`, returns a cached value if
`cache_avail > 0`, and otherwise hashes `"[index]"` to obtain a 256-bit digest.
The digest is split into four big-endian `uint64` words: the last one is
returned immediately and the first three are cached for the following three
calls in LIFO order.

Splitting (`split(label)`) is the recommended way to obtain independent
streams: it feeds the padded label `"[index] {label}"` through the raw SHA-256
compression function keyed by the parent hash, so children are
cryptographically independent of the parent and of each other.

`g_rand_gen()` is a hand-written polar (Marsaglia) Box-Muller with rejection,
not `std::normal_distribution`; the second value of each pair is cached and
reused by the next call.

## Array Layout and Exactness

The generator exposes no lattice arrays itself; the vectorized helpers fill
1-D NumPy views:

    rs.rand_arr(shape)     -> numpy.uint64,   rand_gen()
    rs.u_rand_arr(shape)   -> numpy.float64,  uniform in [0, 1)
    rs.g_rand_arr(shape)   -> numpy.float64,  standard normal

Arrays are filled in C (row-major) order, i.e. as a ravel view, so
`rs.rand_arr((2, 2)).ravel()` equals four successive `rand_gen()` calls.

Exactness classes:

- The integer API (`rand_gen`, `c_rand_gen`, `select`, `random_permute`,
  `rand_arr`) is **bit-exact** against the C++ implementation on every
  platform: it only uses 32-bit integer arithmetic.
- The floating-point API (`u_rand_gen`, `g_rand_gen`, `u_rand_arr`,
  `g_rand_arr`) follows the same draws in the same order and is bit-identical
  on x86-64 with glibc, because it makes exactly the same IEEE-754 operations.
  On another `libm` it may differ in the last one or two `ulp`.

## Public Interface

    class RngState:
        RngState()
        RngState(seed)                 # seed: str | bytes | int
        RngState(parent: RngState)
        RngState(parent: RngState, seed)

        reset() -> None
        copy(is_copying_data=True) -> RngState
        split(seed) -> RngState
        split_inplace(seed) -> None
        newtype(type_) -> RngState        # documented extension

        rand_gen() -> int                 # [0, 2**64 - 1]
        u_rand_gen(lower=0.0, upper=1.0) -> float
        g_rand_gen(center=0.0, sigma=1.0) -> float
        select(seq)
        c_rand_gen(size) -> tuple

        rand_arr(shape) -> numpy.ndarray   # uint64
        u_rand_arr(shape) -> numpy.ndarray # float64, [0, 1)
        g_rand_arr(shape) -> numpy.ndarray # float64, standard normal

        get_state() -> bytes
        set_state(state) -> None

Module-level helpers: `random_permute(l, rs)`, `get_data_sig(x, rs)`, and the
SHA-256 primitives `process_block(old_hash, data)`, `process_input(old_hash,
num_bytes, data)`, `get_hash(data)`.

The `(lower, upper)` argument order of `u_rand_gen` matches
`numpy.random.uniform` and `std::uniform_real_distribution`.  It used to be
`(upper, lower)`; the values passed were swapped with the signature, so the
streams of every caller are unchanged.  Note the public order of
`FieldColorMatrix.set_rand` is still `(upper, lower)` (qlat's), which is a
different method — see below.

There is **no public `*_fill*` method** in either package (qlat removed them;
see qlat-issues entry 3).  `rand_arr`/`u_rand_arr`/`g_rand_arr` take a `shape`
and return a correctly typed array; for a non-default range map the result:

    u = rs.u_rand_arr(shape)                 # [0, 1)
    x = lower + (upper - lower) * u          # [lower, upper)
    g = rs.g_rand_arr(shape)                 # N(0, 1)
    y = center + sigma * g                   # N(center, sigma)

### State export and import

```python
blob = rs.get_state()      # 104 bytes, format version 1
...
rs2 = qr.RngState()
rs2.set_state(blob)        # rs2 continues rs's stream exactly
assert [rs2.rand_gen() for _ in range(4)] == [rs.rand_gen() for _ in range(4)]
```

`get_state()` captures **everything** that determines the remaining stream: the
chaining value, the byte counter, `type`/`index`, the three cached `rand_gen`
words and the cached Box-Muller Gaussian.  A restored state therefore
continues the identical sequence, including cache hits.

`set_state()` validates the whole blob *before* assigning anything, so a
corrupt or foreign blob raises `ValueError` (and `TypeError` for a non-bytes
argument) and leaves the generator exactly as it was — there is no
half-restored state.

### Pickling

`__getstate__`/`__setstate__` return and accept the same blob, so pickling is
just a convenient alias:

```python
import pickle
clone = pickle.loads(pickle.dumps(rs))
assert clone is not rs
assert clone.get_state() == rs.get_state()
assert [clone.rand_gen() for _ in range(4)] == [rs.rand_gen() for _ in range(4)]
```

The unpickled object is an independent copy.  `copy.copy`, `copy.deepcopy` and
`rs.copy()` are independent too.

## Serialized State Format

The blob is **byte-identical to C++ `qlat`**, so a state exported by a
`result-py-local` build can be imported by this package and vice versa.

| Offset | Size | Content |
|---|---|---|
| 0 | 7 | magic `b"QLATRNG"` |
| 7 | 1 | format version, currently `0x01` |
| 8 | 96 | 24 × `uint32`, little-endian |

The 24 words are exactly the words written by C++
`exportRngState(uint32_t*, const RngState&)`.  That function splits every
64-bit member with `patchTwoUint32(high, low)`, and `splitTwoUint32` writes the
**high** word first:

| Word(s) | Content |
|---|---|
| 0, 1 | `num_bytes` high, low |
| 2 … 9 | `hash[0]` … `hash[7]` |
| 10, 11 | `type` high, low |
| 12, 13 | `index` high, low |
| 14, 15 | `cache[0]` high, low |
| 16, 17 | `cache[1]` high, low |
| 18, 19 | `cache[2]` high, low |
| 20, 21 | bit pattern of `gaussian` high, low |
| 22 | `cache_avail` (0 … 3) |
| 23 | `gaussian_avail` (0 or 1) |

The format is versioned: a future change must bump the version byte, and
`set_state` rejects every version it does not implement.  `cache_avail` and
`gaussian_avail` are range-checked as part of the validation.

## Sequence Selection

`select(seq)` returns a uniformly chosen element of **any sized and indexable
sequence** — a `list`, `tuple`, `range`, `numpy.ndarray`, a `str`, or any
object implementing `__len__` and `__getitem__`:

```python
rs.select([10, 20, 30])          # a list element
rs.select((10, 20, 30))          # same index, so same element
rs.select(range(10, 40, 10))     # same index again
rs.select(np.array([10, 20, 30]))
```

Exactly one `rand_gen()` draw is consumed, `rand_gen() % len(seq)`, so the
stream is identical for any container of the same length and unchanged from the
former list-only implementation.  The element is returned exactly as
`seq[index]` gives it, so an `ndarray` yields a NumPy scalar and a `str` yields
a one-character `str`.

Errors: `TypeError` if `seq` is not sized and indexable (e.g. `None`, an
`int`, a generator) and `ValueError` if it is empty.  This matches
`qlat_utils.RngState.select` exactly.

## Examples

Seeding and splitting:

```python
import qlat_ref as qr

rs = qr.RngState("my-seed")
child = rs.split("site 0")        # rs itself is not changed
draws = [child.rand_gen() for _ in range(4)]
```

Reproducible sub-streams per trajectory:

```python
root = qr.RngState("hmc")
for traj in range(4):
    rs_traj = root.split(str(traj))
    momentum_rs = rs_traj.split("set_rand_gauge_momentum")
    accept_rs = rs_traj.split("metropolis_accept")
```

Interruptible long runs — save and resume the exact position:

```python
blob = accept_rs.get_state()
with open("state.bin", "wb") as fp:
    fp.write(blob)
...
accept_rs = qr.RngState()
with open("state.bin", "rb") as fp:
    accept_rs.set_state(fp.read())
```

## Deviations from `qlat`

The shared surface mirrors `qlat_utils.RngState` exactly so a test can swap
`qlat` for `qlat_ref` without changing a call site — method names, argument
order, defaults, draw consumption and error types are all the same.  Two places
extend the surface, and each is documented in the docstring of the affected
method:

| Extension | Why |
|---|---|
| `newtype(type_)` is exposed | it is three lines of C++ (it tags the stream so the counter string becomes `"[type,index]"`); the qlat binding does not expose it |
| `c_rand_gen` accepts any length-4 sequence and returns a plain tuple | `qlat_ref.base.Coordinate` exists, so narrowing it is possible, but that is an API change needing a parity-allowlist entry; the drawn values are already identical |

Both are additions only: nothing that exists in `qlat` behaves differently here.

Conversely, the whole public `*_fill*` family that `qlat` used to expose —
`u_rand_fill`, `g_rand_fill`, `u_rand_fill_real_d`, `g_rand_fill_real_d` and
`rand_fill_u_long` — has been **removed from both packages**.  The first two
reinterpreted any buffer as `float64` and could silently corrupt e.g. an
`int64` array; the other three were only ever the internals of the `*_arr`
helpers and are now private on both sides (`qlat`: `cdef void _...`, no Python
binding; here: a leading-underscore method that is not part of the API).  The
supported API is the `*_arr` trio:

    rs.u_rand_fill(x, -1.0, 1.0)                  # removed
    xv = np.asarray(x).ravel().view(np.float64)   # supported replacement
    xv[...] = -1.0 + 2.0 * rs.u_rand_arr(xv.shape)

The dtype and dimensionality checks the old fillers made are still enforced by
the private helpers, so a wrong dtype raises instead of corrupting the buffer.

### Seed and label byte semantics

A seed or split label is a **byte string**, not text, and every byte
participates in the hash:

| Seed type | Bytes hashed |
|---|---|
| `str` | the UTF-8 encoding |
| `bytes` | used **verbatim** (may be non-UTF-8, may contain NUL) |
| anything else | `str(x)` first, so `RngState(42) == RngState("42")` |

Consequences, all of them covered by the parity tests:

- `RngState(b"abc") == RngState("abc")` — a `bytes` seed is *not* rendered
  through `str()` (which would have given `"b'abc'"`).
- `RngState("a\0bcdefg") != RngState("a")` — an embedded NUL is significant.
  qlat used to format labels with `printf("%s")`, which silently truncated
  them, so distinct labels could collapse onto the same stream
  (qlat-issues entry 1, FIXED).
- `RngState(b"\xff")` is a legal, distinct seed.

`split(seed)` takes only `str`/`bytes` and raises
`TypeError("expected str or bytes, got <type>")` otherwise — including for an
`int` and for a NumPy integer.  This is qlat's contract: C++ has
`split(const Long)`, but the Python binding routes the label through
`as_bytes`, which accepts only `str`/`bytes`.  The *constructor* additionally
accepts an `int` seed (`RngState(42) == RngState("42")`) and
`RngState(parent, 7) == parent.split("7")`.  `split("k\0zzz")` likewise
differs from `split("k")`.

Because `split` rejects an integer, internal call sites that need the
`split(Long)` semantics must spell the label: the field layer uses
`rng.split(str(gindex))`, which hashes exactly the bytes that C++
`RngState rs(rs0, gindex)` hashes.

One quirk is reproduced on purpose: `copy(False)` returns an unseeded state
(qlat's binding builds a fresh `RngState()` and skips the assignment).  The full
write-up, with minimal reproductions, is in
`tmp/qlat-ref-reorg/qlat-issues.md`.

### Consumers in the field and HMC layers

`RngState` is used by the field layer, so its argument conventions matter
beyond this module.  **Two different per-site derivations exist, and they are
not interchangeable** — do not "simplify" one into the other:

- `core.h` `set_u_rand` / `set_g_rand` (behind
  `FieldColorMatrix.set_rand` / `set_rand_g`) use `rs.newtype(gindex)`, which
  *tags* the stream so the counter string becomes `"[type,index]"`;
- `qcd-utils.cpp` `set_g_rand_anti_hermitian_matrix_field` /
  `set_g_rand_color_matrix_field` (behind `GaugeMomentum.set_rand`,
  `GaugeField.set_rand`, `GaugeTransform.set_rand`) use the `RngState rsi(rs,
  gindex)` **split** constructor, i.e. a split of the *decimal string* of the
  global index — `rng.split(str(gindex))` here.

Swapping them shifts every momentum/`SU(3)` value by `O(1)` while looking
locally plausible, so both paths are pinned by the field/momentum parity
tests (max abs difference vs live `qlat` is `<= 1.6e-15`).

  In particular the public order of
`FieldColorMatrix.set_rand` is `(rng, upper=1.0, lower=0.0)` — upper **first**,
which is `qlat`'s public order and is kept because ~148 repo call sites pass
the bounds positionally.  C++ `set_u_rand(f, rs, lower, upper)` takes them
lower-first, and the Cython binding and `qlat_ref` both forward
`(lower, upper)`, so the two packages agree numerically.  The per-site stream
is `rng.newtype(global_site_index)` (not `split`), and within a site the real
and imaginary parts are drawn element-major, which
`Field._get_rand_field` reproduces with one `(mult * NC * NC * 2,)` draw
viewed as `complex128`.

## Self-Checks

```bash
cd qlat-ref
python3 -m unittest tests.test_rng -v
python3 -O -m unittest tests.test_rng -v     # validation must survive -O
```

`tests/test_rng.py` covers the committed C++ vectors (`tests/data/rng-vectors.json`),
the SHA-256 primitives against `hashlib`, the state blob layout and round-trip,
pickling, `select` over every container kind, and the `python -O` probes.
