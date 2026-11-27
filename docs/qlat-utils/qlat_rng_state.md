# `qlat_utils.rng_state` — Deterministic SHA-256-Based Random Number Generator Module

Source: `qlat-utils/qlat_utils/rng_state.pyx`

> **Note:** Update this document when updating the source file.

## Outline

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [RngState Class](#rngstate-class)
   - [Constructor](#constructor)
   - [Copying and Assignment](#copying-and-assignment)
   - [Splitting](#splitting)
   - [Scalar Random Generators](#scalar-random-generators)
   - [Array Random Generators](#array-random-generators)
   - [Selection and Permutation Helpers](#selection-and-permutation-helpers)
   - [State Export, Restore and Pickling](#state-export-restore-and-pickling)
4. [Module-Level Utility Functions](#module-level-utility-functions)
5. [C++ Internals Reference](#c-internals-reference)
6. [Examples](#examples)

---

## Overview

`rng_state` is the random number generator module of `qlat_utils`. It provides:

- **`RngState`** — a deterministic, reproducible, split-based PRNG built on
  SHA-256 hashing, designed for parallel lattice-QCD simulations where every
  MPI rank, GPU, or thread must produce independent yet fully reproducible
  random streams without a central coordinator.
- **`get_data_sig()`** — compute a deterministic floating-point signature of
  arbitrary data by dotting with a random ±1 vector.
- **`random_permute()`** — Fisher-Yates shuffle returning a new permuted list.

Key properties of `RngState`:

| Property | Description |
|---|---|
| Determinism | Same seed + same split history = identical sequence on every platform |
| Splitting | Fork child generators by name/index; children are independent of each other |
| No period | SHA-256 based; no short-cycle issues common in LCG/Xorshift generators |
| Thread safe | Each `RngState` instance owns its own state; no global lock needed |
| 64-bit output | `rand_gen()` returns `uint64` values uniformly over `[0, 2^64-1]` |

Python access:

```python
import qlat_utils as q
rs = q.RngState("my-seed")
```

---

## Design Philosophy

### SHA-256 counter mode

Internally, each `RngState` stores a 256-bit SHA-256 hash (`hash[8]` of `uint32`)
plus a byte counter (`numBytes`). When a random number is requested, the
generator increments its `index`, formats a string like `"[index]"` (or
`"[type,index]"` if `type` is set), and feeds it into SHA-256 keyed by the
current hash. The resulting 256-bit digest yields **four** `uint64` values: one
is returned immediately, the other three are cached for subsequent calls. This
gives a 4x throughput boost per SHA-256 invocation.

### Split-based forking

Instead of using a single global seed and skipping ahead (which requires
coordination), `RngState` uses **splitting**: calling `split("label")` produces
a *new* `RngState` whose hash is derived from the parent hash plus the label.
The child's sequence is cryptographically independent of the parent and of all
other children. This is the recommended way to assign independent random streams
to lattice sites, directions, MPI ranks, etc.

### Gaussian via Box-Muller

`g_rand_gen()` uses the Box-Muller transform. One of the two generated Gaussian
values is cached in `gaussian` / `gaussianAvail` and returned on the next call,
so every pair of uniform draws produces two Gaussian draws efficiently.

---

## RngState Class

### Constructor

```python
RngState()
RngState(seed: str | int)
RngState(parent: RngState)
RngState(parent: RngState, seed: str | int)
```

#### Parameters

| Form | Behavior |
|---|---|
| `RngState()` | Create an RNG with the default (zero) initial hash. Useful as a blank slate before assignment. |
| `RngState(seed)` | Create an RNG seeded with `seed`. The seed is converted to a string internally. Equivalent to `RngState()` then `reset(seed)`. |
| `RngState(parent)` | Copy constructor — create an independent but *identical* clone of `parent`. |
| `RngState(parent, seed)` | Split `parent` by `seed` to produce a new independent child. Equivalent to `parent.split(seed)`. |

#### Examples

```python
import qlat_utils as q

rs0 = q.RngState()           # default (zero-hash) state
rs1 = q.RngState("hello")    # seeded with string "hello"
rs2 = q.RngState(42)         # seeded with integer 42 (converted to "42")
rs3 = q.RngState(rs1)        # copy of rs1
rs4 = q.RngState(rs1, "sub") # split child of rs1 with label "sub"
```

#### Seed and label byte semantics

A seed or split label is a **byte string**, not a C string:

| Argument type | Meaning |
|---|---|
| `str` | Encoded as UTF-8, so its bytes (including any embedded NUL) are hashed. |
| `bytes` | Used **verbatim** as the label bytes; no decoding, so a non-UTF-8 label such as `b"\xff"` is valid. |
| other (e.g. `int`) | Rendered with `str()` first, e.g. `RngState(42)` seeds with `"42"`. |

Argument validation raises explicit exceptions (`TypeError` / `ValueError`), so
it is active in every mode, including `python -O` (it used to rely on bare
`assert`s, which `-O` silently removed).

Key consequences, covered by the `examples-py/utils.py` regression test:

- **An embedded NUL byte is significant.** `RngState("a\0bcdefg")` is
  *different* from `RngState("a")` — the whole label contributes to the
  SHA-256 hash. Before qlat-issue-1 was fixed, the label was formatted with
  `printf("%s", label.c_str())`, which silently truncated it at the first NUL
  byte and made such seeds collide.
- **A `bytes` seed is raw label bytes**, the same as `split()` already did.
  `RngState(b"abc")` equals `RngState("abc")`. Before qlat-issue-6 was fixed,
  the constructor went through `str(x)` and hashed the *`repr`* (`"b'abc'"`)
  instead.
- Labels **without** a NUL byte hash exactly as they always did, so existing
  seeds and split trees are unchanged.

```python
q.RngState("a\0bcdefg") == q.RngState("a")      # False (used to be True)
q.RngState(b"abc") == q.RngState("abc")           # True
q.RngState(b"abc") == q.RngState("b'abc'")        # False (used to be True)
q.RngState(b"\xff")                               # valid: raw non-UTF-8 label
```

---

### Copying and Assignment

#### `copy(is_copying_data=True) -> RngState`

Return a copy of this `RngState`. If `is_copying_data` is `False`, return a
default-initialized (blank) `RngState`.

```python
rs_copy = rs.copy()            # identical copy
rs_blank = rs.copy(False)      # default state, not a copy of rs
```

#### `__copy__` / `__deepcopy__`

Both delegate to `copy()`. Standard `copy.copy(rs)` and `copy.deepcopy(rs)`
work as expected.

#### `__imatmul__` (in-place assignment `@=`)

```python
rs1 @= rs2   # rs1 now has identical state to rs2
```

---

### Splitting

#### `split(seed: str | bytes) -> RngState`

Produce a new `RngState` deterministically derived from `self` and `seed`.
The child is fully independent of the parent and of all other children with
different seeds. `str` is encoded as UTF-8 and `bytes` is used verbatim (see
[Seed and label byte semantics](#seed-and-label-byte-semantics)), so labels
containing NUL or non-UTF-8 bytes are distinct and participate fully in the
hash.

```python
rs_root = q.RngState("experiment-1")

rs_site_0   = rs_root.split("site-0")
rs_site_1   = rs_root.split("site-1")
rs_direction = rs_root.split("mu=2")
rs_rank     = rs_root.split("3")
```

**Splitting is the primary mechanism for assigning independent streams in
parallel computations.** Typical pattern for a lattice simulation:

```python
rs_global = q.RngState("simulation-v1")

for site in all_sites:
    rs_site = rs_global.split(site.to_tuple())  # or any unique label
    # use rs_site for all randomness at this site
```

---

### Scalar Random Generators

#### `rand_gen() -> int`

Generate a uniformly distributed random integer in `[0, 2^64 - 1]`.

```python
r = rs.rand_gen()   # e.g. 14873649204851673523
```

#### `u_rand_gen(lower=0.0, upper=1.0) -> float`

Generate a uniformly distributed random `float64` in `[lower, upper)`.

The argument order is `(lower, upper)`, matching `numpy.random.uniform` and
`std::uniform_real_distribution`. It used to be `(upper, lower)`, so positional
calls written against the old signature must swap their two arguments; calls
that pass `lower=` / `upper=` by keyword keep their meaning.

```python
x = rs.u_rand_gen()          # uniform in [0.0, 1.0)
x = rs.u_rand_gen(2.0, 5.0)  # uniform in [2.0, 5.0)
```

#### `g_rand_gen(center=0.0, sigma=1.0) -> float`

Generate a Gaussian (normal) distributed random `float64` with the given
`center` (mean) and `sigma` (standard deviation). Uses Box-Muller internally;
caches one of the two generated values for efficiency.

```python
z = rs.g_rand_gen()              # standard normal N(0,1)
x = rs.g_rand_gen(3.0, 0.5)     # N(3.0, 0.5)
```

#### `c_rand_gen(size: Coordinate) -> Coordinate`

Generate a uniformly random lattice coordinate within the hyper-rectangle
defined by `size` (a 4-component `Coordinate`). Useful for picking a random
lattice site given the lattice `total_site`.

```python
total_site = q.Coordinate((16, 8, 8, 8))
coord = rs.c_rand_gen(total_site)  # e.g. Coordinate((7, 3, 5, 2))
```

---

### Array Random Generators

These return NumPy arrays filled with random values of the specified shape.

> **Removed in this release:** the obsolete `u_rand_fill` / `g_rand_fill`
> family (`u_rand_fill`, `g_rand_fill`, `u_rand_fill_real_d`,
> `g_rand_fill_real_d`, `rand_fill_u_long`) is **no longer part of the Python
> API**. Those methods reinterpreted an arbitrary buffer as `float64` and could
> silently corrupt a non-`float64` array, and they required the argument to be a
> view. Use `rand_arr` / `u_rand_arr` / `g_rand_arr` and do the arithmetic in
> NumPy, e.g. `rs.u_rand_arr(shape) * (upper - lower) + lower`.
> `Field.set_rand` / `Field.set_rand_g` keep the old bound order and still do
> this internally.

#### `rand_arr(shape) -> np.ndarray`

Return a `uint64` array of the given `shape` filled with uniform random
integers in `[0, 2^64-1]`.

```python
arr = rs.rand_arr((4, 4))  # shape (4,4), dtype=uint64
```

#### `u_rand_arr(shape) -> np.ndarray`

Return a `float64` array of the given `shape` filled with uniform random
values in `[0.0, 1.0)`.

```python
arr = rs.u_rand_arr((100,))  # 100 uniform floats
```

#### `g_rand_arr(shape) -> np.ndarray`

Return a `float64` array of the given `shape` filled with Gaussian random
values with `center=0.0, sigma=1.0`.

```python
arr = rs.g_rand_arr((8, 8, 8, 8))  # 4D lattice of N(0,1) values
```

---

### Selection and Permutation Helpers

#### `select(seq) -> Any`

Pick and return a uniformly random element of `seq`. Any **sized and indexable**
sequence works: a `list`, `tuple`, `range`, `numpy.ndarray`, `str`, or a custom
object implementing `__len__` and `__getitem__` (this used to require a `list`).
The element is returned exactly as `seq[index]` would give it, so an `ndarray`
input yields a NumPy scalar.

Exactly one `rand_gen()` draw is consumed per call, independently of the
container type, so the same seed chooses the same index for any container of
the same length.

Errors: `TypeError` when `seq` has no length or is not indexable (e.g. `None`,
an `int`); `ValueError` when `seq` is empty.

```python
sites = [(0,0,0,0), (1,1,1,1), (2,2,2,2)]
chosen = rs.select(sites)
chosen = rs.select(range(16))          # any sized/indexable sequence
chosen = rs.select(np.arange(16))      # NumPy scalar returned
```

#### `get_state() -> bytes` / `set_state(state)`

`get_state()` serializes the **complete** generator state — the SHA-256 chaining
value, the byte counter, `type`, `index`, the three-word output cache and the
Box-Muller Gaussian cache — into 104 bytes. `set_state()` restores it exactly,
so the restored generator continues the *identical* stream, including a pending
cached Gaussian.

The layout is versioned and shared with `qlat_ref` (a state written by one
package can be restored by the other):

| Offset | Size | Content |
|---|---|---|
| 0 | 7 | magic `b"QLATRNG"` |
| 7 | 1 | version `0x01` |
| 8 | 96 | 24 little-endian `uint32` words, in C++ `exportRngState` order |

The 24 words are `numBytes` (high, low), `hash[0..7]`, `type` (high, low),
`index` (high, low), `cache[0..2]` (high, low each), the `gaussian` bit pattern
(high, low), `cacheAvail` and `gaussianAvail`.

`set_state()` accepts `bytes`, `bytearray`, `memoryview` or any buffer-like
object and validates length, magic, version and the two small integer fields
**before** mutating the generator: a malformed state raises `ValueError` and
leaves the generator untouched.

```python
state = rs.get_state()          # bytes, length 104
rs2 = q.RngState()
rs2.set_state(state)            # continues rs's stream exactly
```

#### `pickle` support

`RngState` pickles via `__reduce__`/`__setstate__`, so a round-trip yields an
**independent** copy on the same stream:

```python
import pickle
rs3 = pickle.loads(pickle.dumps(rs))
assert rs3.rand_gen() == rs.rand_gen()   # same next value
```

---

## Module-Level Utility Functions

These are free functions in the `qlat_utils` namespace (not methods of
`RngState`).

### `get_data_sig(x, rs: RngState) -> float | complex`

Compute a deterministic signature of data `x` (an ndarray, `LatData`,
`SpinMatrix`, etc.) by dotting the flattened data with a random `{-1, +1}`
vector drawn from `rs`. The result depends only on the *value* of `x`, not its
shape. Useful for checksumming field data.

```python
sig = q.get_data_sig(field_array, rs)
```

### `random_permute(l: list, rs: RngState) -> list`

Return a new list that is a random permutation of `l` (Fisher-Yates shuffle).
Does **not** modify the original list.

```python
permuted = q.random_permute([0, 1, 2, 3, 4], rs)
```

---

## C++ Internals Reference

For developers extending or debugging the C++ layer.

### Struct Layout (`rng-state.h`)

```cpp
struct RngState {
    uint64_t numBytes;       // total bytes processed by SHA-256
    uint32_t hash[8];        // 256-bit SHA-256 state
    uint64_t type;           // optional type tag (ULONG_MAX = unset)
    uint64_t index;          // number of rand_gen calls made
    uint64_t cache[3];       // cached random values (3 unused from last hash)
    RealD    gaussian;       // cached Gaussian value (Box-Muller)
    Int      cacheAvail;     // number of valid entries in cache[]
    bool     gaussianAvail;  // whether `gaussian` holds a valid cached value
};
```

### Free Functions (`rng-state.h`)

| C++ Signature | Description |
|---|---|
| `void reset(RngState& rs)` | Reset to initial (zero-hash) state |
| `void reset(RngState& rs, const std::string& seed)` | Reset and seed from string |
| `void reset(RngState& rs, const Long seed)` | Reset and seed from integer |
| `void split_rng_state(RngState& rs, const RngState& rs0, const std::string& sindex)` | Derive `rs` from `rs0` + label |
| `void split_rng_state(RngState& rs, const RngState& rs0, const Long sindex)` | Same, with integer label |
| `void set_type(RngState& rs, const uint64_t type)` | Set the type tag (must be unset) |
| `uint64_t rand_gen(RngState& rs)` | Return uniform `uint64` |
| `RealD u_rand_gen(RngState& rs, RealD lower, RealD upper)` | Uniform `RealD` in `[lower, upper)` |
| `RealD g_rand_gen(RngState& rs, RealD center, RealD sigma)` | Gaussian `RealD` |
| `void compute_hash_with_input(uint32_t hash[8], const RngState& rs, const std::string& input)` | Core SHA-256 computation |
| `std::string get_rng_state_bytes(const RngState& rs)` | Serialize the full state to the 104-byte format |
| `void set_rng_state_bytes(RngState& rs, const std::string& data)` | Restore it; throws `std::invalid_argument` (maps to Python `ValueError`) on a malformed blob |
| `void exportRngState(uint32_t* v, const RngState& rs)` / `void importRngState(RngState& rs, const uint32_t* v)` | The 24-word state transfer used by the above and by `operator<<`/`>>` |

### Member Functions

| Method | Description |
|---|---|
| `rs.split(sindex)` | Return new `RngState` split by string label |
| `rs.newtype(type)` | Return a copy with the type tag set |

### How `rand_gen` Works Internally

1. Increment `index`.
2. If `cacheAvail > 0`, return `cache[--cacheAvail]`.
3. Otherwise, compute `SHA-256(hash ‖ "[index]")` (or `"[type,index]"` if `type` is set).
4. The 256-bit digest yields 4 × `uint64`. Return the last one immediately; cache the first three.
5. Update `hash` and increment `numBytes`.

### How `split_rng_state` Works Internally

1. Format a label string: `"[index] {sindex}"` or `"[type,index] {sindex}"`.
2. Pad the label to a multiple of 64 bytes.
3. Feed the padded label through `SHA-256::processBlock` keyed by `rs0.hash`.
4. The new `rs` gets the resulting hash, `numBytes += nBlocks * 64`, and all
   cache/gaussian state is cleared.

---

## Examples

### Basic Usage

```python
import qlat_utils as q

rs = q.RngState("my-seed")

# Single random values
print(rs.rand_gen())      # random uint64
print(rs.u_rand_gen())    # random float in [0, 1)
print(rs.g_rand_gen())    # random Gaussian N(0,1)

# Arrays
arr = rs.u_rand_arr((3, 3))  # 3x3 uniform float array
print(arr)
```

### Splitting for Parallel Sites

```python
import qlat_utils as q

rs = q.RngState("lattice-run-1")

# Assign independent streams per site
total_site = q.Coordinate((8, 8, 8, 8))
for t in range(total_site[0]):
    for z in range(total_site[1]):
        for y in range(total_site[2]):
            for x in range(total_site[3]):
                rs_site = rs.split(f"({t},{z},{y},{x})")
                # Use rs_site for all randomness at this site
                link_value = rs_site.g_rand_gen()
```

### Reproducible Sub-streams

```python
import qlat_utils as q

rs = q.RngState("experiment")

# These two calls always produce the same results regardless of ordering
rs_a = rs.split("stream-a")
rs_b = rs.split("stream-b")

# Each stream is deterministic
val_a = rs_a.u_rand_gen()  # always the same
val_b = rs_b.u_rand_gen()  # always the same, independent of val_a
```

### Random Permutation

```python
import qlat_utils as q

rs = q.RngState("shuffle-seed")
sites = list(range(64))
permuted_sites = q.random_permute(sites, rs)
print(permuted_sites)  # deterministic permutation
```

### Checksumming with `get_data_sig`

```python
import qlat_utils as q
import numpy as np

rs = q.RngState("checksum")
data = np.random.randn(100)
sig = q.get_data_sig(data, rs)
# sig is a deterministic float depending on data values
```
