"""
Module ``qlat_ref.rng``
=======================\n
Pure-Python re-implementation of qlat's ``RngState``, the deterministic
SHA-256-based random number generator of ``qlat_utils``.\n
Documentation: ``docs/qlat-ref/qlat_rng.md``\n
.. note:: Update the documentation when updating this source file.\n
The generator is a SHA-256 *counter mode* stream, not a Mersenne Twister and
not NumPy's PCG64: the state is a 256-bit SHA-256 chaining value plus a byte
counter, and each draw hashes the ASCII string ``"[index]"`` *continuing* that
chaining value (the message-length counter starts at the bytes already
consumed).  The 256-bit digest is split into four big-endian ``uint64`` words:
the last one is returned at once and the first three are cached and returned by
the following three calls in LIFO order.\n
Exactness classes
-----------------\n
The integer API (``rand_gen``, ``c_rand_gen``, ``select``, ``random_permute``,
``rand_arr``) is **bit-exact** against the C++ implementation on every
platform: it only uses 32-bit integer arithmetic.  The floating-point API
(``u_rand_gen``, ``g_rand_gen``, ``u_rand_arr``, ``g_rand_arr``) follows the
same draws in the same order and is bit-identical on x86-64 with glibc, because
it makes exactly the same IEEE-754 operations (``u_rand_gen`` is one
multiply-add; ``g_rand_gen`` calls ``math.sqrt``/``math.log``, i.e. the same
``libm``).  On another ``libm`` it may differ in the last one or two ``ulp``,
which is why the tests compare ``float.hex()`` exactly on x86-64 Linux and
fall back to a small relative tolerance elsewhere.\n
Interface
---------\n
The shared surface mirrors qlat's Python ``RngState`` (``qlat_utils.rng_state``)
exactly, so a test can swap ``qlat`` for ``qlat_ref`` without changing a call
site.  Two documented extensions exist:\n
- ``split`` also accepts an ``int``, which is what C++ ``split(const Long)``
  does (``split(0)`` and ``split("0")`` agree in C++); qlat's Python binding
  rejects it with ``TypeError`` because it converts the argument to
  ``str``/``bytes`` directly.
- ``newtype(type)`` is offered because it is three lines of C++ (it tags the
  stream so the counter string becomes ``"[type,index]"``); qlat's Python
  binding does not expose it, so it is an extension, not part of the parity
  contract.\n
State export, import and pickling
---------------------------------\n
:meth:`RngState.get_state` returns the complete generator state as a
104-byte ``bytes`` blob and :meth:`RngState.set_state` restores it exactly, so
a saved state continues the identical stream -- including the three cached
``rand_gen`` words and the cached Box-Muller Gaussian value.  The blob is
versioned (magic ``b"QLATRNG"`` + version byte) and byte-compatible with C++
``qlat``, so a state can be moved between a ``result-py-local`` build and this
package in either direction.  ``set_state`` validates the whole blob first and
raises ``ValueError`` (or ``TypeError`` for a non-``bytes`` argument) without
touching the generator, so a corrupt or foreign blob can never leave a
half-restored state.\n
``pickle`` uses the same blob through ``__getstate__``/``__setstate__``, and
``copy.copy``/``copy.deepcopy``/``copy()`` remain independent copies::\n
    blob = rs.get_state()
    ...
    rs2 = RngState()
    rs2.set_state(blob)          # rs2 continues rs's stream
    rs3 = pickle.loads(pickle.dumps(rs))   # same, independent object
\n
qlat quirks and deviations
--------------------------\n
Bit-exact parity with qlat is the contract of this module, so the quirks that
are still observable are reproduced deliberately and documented where they
happen (``copy(False)`` returns an unseeded state).  Two quirks that the
original binding had are **gone from both qlat and this module** and are no
longer reproduced: a ``bytes`` seed is used as the raw label bytes rather than
being rendered with ``repr``, and a split label is no longer cut at an embedded
NUL byte (both were API defects of the Cython binding, fixed in
``qlat-utils/qlat_utils/rng_state.pyx``; the old behavior is kept in
``_truncate_at_nul``'s docstring only as history).  Three places still deviate
because qlat's behavior is an API defect rather than algorithmic output, and
each is marked in its docstring: :meth:`split` accepts an ``int`` (C++ has
``split(const Long)``), :meth:`RngState.__init__` raises ``TypeError`` for
invalid argument combinations instead of relying on ``assert`` (which vanishes
under ``python -O``), and every public method validates its arguments
explicitly for the same reason.  The full write-up, with minimal reproductions
and the recommended upstream fixes, is in
``tmp/qlat-ref-reorg/qlat-issues.md``.\n
Performance
-----------\n
SHA-256 is written out in plain Python because ``hashlib`` cannot continue a
hash from a given chaining value *and* a given message length, which is exactly
what counter mode needs.  One draw costs roughly 10 microseconds, one split
roughly 300 microseconds, so this module is meant for reference-scale lattices
(``2^4`` to ``4^4``) and for test vectors, not for production volumes.
"""

import math
import struct

import numpy as np

### -------------------------------------------------------------------

ULONG_MAX = 0xFFFFFFFFFFFFFFFF
MASK32 = 0xFFFFFFFF
# 2**-64, computed the way the C++ code computes it: 1/2**32 / 2**32.
TWO_POW_M64 = 1.0 / (256.0 * 256.0 * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 * 256.0)

# ``sha256::processBlock`` round constants (sha256.cpp).
_K = (
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
)

# ``sha256::setInitialHash()``: the standard SHA-256 IV as 8 big-endian uint32.
SHA256_IV = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)

### --- SHA-256 primitives (qlat-utils/lib/sha256.cpp) -------------------

def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK32

def process_block(old_hash, data):
    """
    ``sha256::processBlock``: raw SHA-256 compression of exactly 64 bytes.\n
    ``old_hash`` is the 8-word chaining value; the returned list is the new
    chaining value (this is one *compression*, not one whole hash).
    """
    assert len(data) == 64
    w = list(struct.unpack(">16I", data))
    for i in range(16, 64):
        s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & MASK32)
    a, b, c, d, e, f, g, h = old_hash
    for i in range(64):
        s1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = (h + s1 + ch + _K[i] + w[i]) & MASK32
        s0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = ((a | b) & c) | (a & b)
        t2 = (s0 + maj) & MASK32
        h, g, f, e, d, c, b, a = (
            g,
            f,
            e,
            (d + t1) & MASK32,
            c,
            b,
            a,
            (t1 + t2) & MASK32,
        )
    #
    return [
        (old_hash[0] + a) & MASK32,
        (old_hash[1] + b) & MASK32,
        (old_hash[2] + c) & MASK32,
        (old_hash[3] + d) & MASK32,
        (old_hash[4] + e) & MASK32,
        (old_hash[5] + f) & MASK32,
        (old_hash[6] + g) & MASK32,
        (old_hash[7] + h) & MASK32,
    ]

def process_input(old_hash, num_bytes, data):
    """
    ``sha256::processInput``: hash a message, continuing a prefix hash.\n
    ``old_hash`` is the chaining value after ``num_bytes`` message bytes have
    already been consumed, so the appended 64-bit big-endian length field is
    ``8 * (num_bytes + len(data))``.  Unlike a split this *does* apply the
    standard SHA-256 padding (``0x80``, zero fill, length).
    """
    h = list(old_hash)
    n_blocks = len(data) // 64
    for i in range(n_blocks):
        h = process_block(h, data[i * 64 : (i + 1) * 64])
    #
    buffer_size = len(data) - n_blocks * 64
    padded_bits = buffer_size * 8 + 1
    lower11 = padded_bits & 511
    padded_bits += 448 - lower11 if lower11 <= 448 else 512 + 448 - lower11
    padded_len = padded_bits // 8
    buf = bytearray(64)
    buf[:buffer_size] = data[n_blocks * 64 :]
    extra = bytearray(64)
    if buffer_size < 64:
        buf[buffer_size] = 128
    else:
        extra[0] = 128
    msg_bits = (8 * (num_bytes + len(data))) & 0xFFFFFFFFFFFFFFFF
    length_bytes = struct.pack(">Q", msg_bits)
    if padded_len < 64:
        buf[padded_len : padded_len + 8] = length_bytes
    else:
        extra[padded_len - 64 : padded_len - 64 + 8] = length_bytes
    h = process_block(h, bytes(buf))
    if padded_len > 64:
        h = process_block(h, bytes(extra))
    return h

def _patch(a, b):
    """``patchTwoUint32``: big-endian packing of two 32-bit words into a uint64."""
    return (a << 32) | b

def get_hash(data):
    """Return the SHA-256 digest of ``data`` (bytes) as a list of 8 uint32 words."""
    return process_input(SHA256_IV, 0, data)

### --- coordinate helpers (qlat-utils/lib/coordinate.h) -----------------

def coordinate_from_index(index, size):
    """
    ``coordinate_from_index``: component 0 varies fastest.\n
    ``size`` is a length-4 sequence (``(Nt, Nx, Ny, Nz)``).
    """
    x = [0, 0, 0, 0]
    for i in range(4):
        x[i] = index % size[i]
        index //= size[i]
    #
    return x

def index_from_coordinate(x, size):
    """``index_from_coordinate``: the inverse of :func:`coordinate_from_index`."""
    return (((x[3] * size[2]) + x[2]) * size[1] + x[1]) * size[0] + x[0]

### -------------------------------------------------------------------

def _seed_bytes(x):
    """
    Return the seed/label bytes for ``x``, mirroring qlat's ``as_seed_bytes``.\n
    A ``str`` is encoded as UTF-8, a ``bytes`` object is used **verbatim** (so
    it can carry a non-UTF-8 or NUL-containing label), and any other object is
    rendered with ``str()`` first (so an ``int`` seed gives its decimal
    string).  Critically, ``bytes`` must not go through ``str()``: that would
    render ``b"abc"`` as ``"b'abc'"``.
    """
    if isinstance(x, (str, bytes, bytearray)):
        return x.encode("utf-8") if isinstance(x, str) else bytes(x)
    return str(x).encode("utf-8")

def _truncate_at_nul(label):
    """
    Return ``label`` unchanged (every byte participates in the hash).\n
    Historical note: qlat used to cut a label at its first NUL byte, because
    ``split_rng_state`` formatted it with ``ssprintf("[%lu] {%s}", ...,
    sindex.c_str())``; two different labels then collapsed onto the same child
    stream (qlat-issues entry 1).  qlat appends the label verbatim now and this
    module mirrors that, so a label containing a NUL byte is *not* truncated.
    Non-NUL labels hash exactly as before, so no committed vector moves.
    """
    return label

def _get_buffer(arr, dtype, c_name):
    """
    Return ``arr`` as a 1-D NumPy array of ``dtype``, like a typed memoryview.\n
    qlat's (now private) array fillers take a 1-D typed memoryview, so a
    higher-dimensional
    array, a wrong dtype or a non-buffer raise the same ``ValueError`` /
    ``TypeError`` here.  Reproducing this is not pedantry: silently accepting an
    ``int32`` array would corrupt it (the 64-bit draws do not fit) rather than
    fail.
    """
    if not hasattr(arr, "dtype"):
        raise TypeError(f"a bytes-like object is required, not '{type(arr).__name__}'")
    #
    if arr.dtype != dtype:
        raise ValueError(
            f"Buffer dtype mismatch, expected '{c_name}' but got "
            f"'{np.dtype(arr.dtype).name}'"
        )
    #
    if arr.ndim != 1:
        raise ValueError(
            f"Buffer has wrong number of dimensions (expected 1, got {arr.ndim})"
        )
    #
    return arr

### --- state blob packing (shared format with C++ qlat) -----------------
#
# The header constants live on :class:`RngState` (``_STATE_MAGIC`` etc.) so they
# are part of the documented API; these helpers only do the struct packing.

def _pack_state_words(words):
    """Pack the 24 ``exportRngState`` words into little-endian ``uint32`` bytes."""
    assert len(words) == RngState._STATE_WORDS
    #
    return struct.pack("<%dI" % len(words), *[int(w) & MASK32 for w in words])

def _unpack_state(state):
    """
    Validate a state blob and return its 24 ``uint32`` words (never mutates).\n
    The checks and their order mirror ``qlat_utils.RngState.set_state`` exactly,
    so a bad blob produces the same error type (and a very similar message) in
    both packages: ``TypeError`` for a non-buffer argument and ``ValueError``
    for a wrong length, magic, version or out-of-range cache flag.
    """
    if isinstance(state, (bytes, bytearray)):
        blob = bytes(state)
    else:
        # Accept any object exposing the buffer protocol (memoryview, numpy
        # arrays of uint8, array.array, ...), like the C++ binding.
        try:
            blob = bytes(memoryview(state))
        except TypeError:
            raise TypeError(
                f"RngState.set_state: a bytes-like state is required, got "
                f"'{type(state).__name__}'"
            ) from None
    #
    expected = RngState._STATE_SIZE
    if len(blob) != expected:
        raise ValueError(
            f"RngState.set_state: expected {expected} bytes, got {len(blob)}"
        )
    #
    magic = RngState._STATE_MAGIC
    if blob[: len(magic)] != magic:
        raise ValueError(
            f"RngState.set_state: not a qlat RNG state (expected magic "
            f"{magic!r}, got {blob[: len(magic)]!r})"
        )
    #
    version = blob[len(magic)]
    if version != RngState._STATE_VERSION:
        raise ValueError(
            f"RngState.set_state: unsupported version {version} "
            f"(expected {RngState._STATE_VERSION})"
        )
    #
    words = list(struct.unpack("<%dI" % RngState._STATE_WORDS, blob[len(magic) + 1 :]))
    if not 0 <= words[22] <= 3:
        raise ValueError(
            f"RngState.set_state: corrupt state (cache_avail is {words[22]}, "
            "expected 0..3)"
        )
    #
    if words[23] not in (0, 1):
        raise ValueError(
            f"RngState.set_state: corrupt state (gaussian_avail is {words[23]}, "
            "expected 0 or 1)"
        )
    #
    return words

### -------------------------------------------------------------------

class RngState:
    """
    Deterministic SHA-256 counter-mode random number generator.\n
    ``RngState()`` hashes from the standard SHA-256 initial value;
    ``RngState(seed)`` obtains its state from ``RngState().split(seed)``, which
    is exactly what C++ ``RngState(const std::string&)`` does.  ``RngState(rs)``
    copies ``rs`` and ``RngState(rs, seed)`` copies then splits it.  Because the
    qlat Python binding applies ``str()`` to a non-``RngState`` argument,
    ``RngState(42)`` is ``RngState("42")`` and ``RngState(b"x")`` is
    ``RngState("b'x'")``; the same holds in C++, where ``RngState(Long)``
    formats the integer with ``show``.  See :meth:`split` for the one
    documented extension (an ``int`` split key).\n
    The attributes mirror the C++ ``RngState`` members: ``num_bytes``
    (``numBytes``), ``hash`` (``hash``, a list of 8 uint32 words), ``type``
    (``type``, ``ULONG_MAX`` when unset), ``index`` (``index``), ``cache`` /
    ``cache_avail`` (``cache`` / ``cacheAvail``), and ``gaussian`` /
    ``gaussian_avail`` (``gaussian`` / ``gaussianAvail``).  They are exposed
    because the tests and the HMC momentum code need to inspect them; a user
    only calls the generation methods.\n
    Use :meth:`get_state`/:meth:`set_state` (or ``pickle``) to save and restore
    a generator exactly, and :meth:`select` to draw an element of any
    sized/indexable sequence.\n
    Still inherited from qlat on purpose (bit-exact parity is the contract; see
    ``tmp/qlat-ref-reorg/qlat-issues.md`` for the full write-up):
    ``copy(False)`` returns an unseeded default state, like the qlat binding.
    Two former quirks are **no longer reproduced** because they were fixed in
    qlat itself: a ``bytes`` seed is now the raw label bytes (not its
    ``repr``), and a split label is no longer cut at an embedded NUL byte.\n
    Documented deviations from qlat's Python binding (both are API defects
    rather than algorithmic behavior): :meth:`split` accepts an ``int``
    (matching C++ ``split(const Long)``, which the Python wrapper forgot), and
    the constructor raises ``TypeError`` for invalid argument combinations
    instead of relying on ``assert`` (which vanishes under ``python -O``).
    """

    def __init__(self, x=None, y=None):
        if x is None:
            if y is not None:
                raise TypeError("RngState(None, y) is not a valid state")
            #
            self.reset()
        elif isinstance(x, RngState):
            self._assign(x)
            if y is not None:
                self.split_inplace(_seed_bytes(y))
        else:
            if y is not None:
                raise TypeError(
                    "RngState(x, y) requires x to be an RngState; got "
                    f"{type(x).__name__}"
                )
            #
            self.reset()
            self.split_inplace(_seed_bytes(x))

    ### --- state handling ---------------------------------------------

    def reset(self):
        """``reset``: back to the all-zero (standard SHA-256 IV) state."""
        self.num_bytes = 0
        self.hash = list(SHA256_IV)
        self.type = ULONG_MAX
        self.index = 0
        self.cache = [0, 0, 0]
        self.gaussian = 0.0
        self.cache_avail = 0
        self.gaussian_avail = False

    def _assign(self, o):
        self.num_bytes = o.num_bytes
        self.hash = list(o.hash)
        self.type = o.type
        self.index = o.index
        self.cache = list(o.cache)
        self.gaussian = o.gaussian
        self.cache_avail = o.cache_avail
        self.gaussian_avail = o.gaussian_avail

    ### --- state export / import ---------------------------------------
    #
    # The serialized form is shared byte-for-byte with C++ ``qlat`` so a state
    # exported by one implementation can be imported by the other:
    #
    #   bytes[0:7]   b"QLATRNG"                      magic
    #   bytes[7]     0x01                            format version
    #   bytes[8:104] 24 x uint32, little-endian      the ``exportRngState``
    #                                                words (see below)
    #
    # ``exportRngState`` (qlat-utils/lib/rng-state.cpp) splits each 64-bit
    # member into ``patchTwoUint32(high, low)``; ``splitTwoUint32`` writes the
    # HIGH word first, so the order is:
    #
    #   [0]  num_bytes  >> 32          [1]  num_bytes  & 0xFFFFFFFF
    #   [2..9]  hash[0..7]
    #   [10] type       >> 32          [11] type       & 0xFFFFFFFF
    #   [12] index      >> 32          [13] index      & 0xFFFFFFFF
    #   [14] cache[0]   >> 32          [15] cache[0]   & 0xFFFFFFFF
    #   [16] cache[1]   >> 32          [17] cache[1]   & 0xFFFFFFFF
    #   [18] cache[2]   >> 32          [19] cache[2]   & 0xFFFFFFFF
    #   [20] bit pattern of ``gaussian`` >> 32
    #   [21] bit pattern of ``gaussian`` & 0xFFFFFFFF
    #   [22] cache_avail (0..3)        [23] gaussian_avail (0 or 1)
    #
    # The format is versioned: a future change must bump ``_STATE_VERSION``
    # and keep ``set_state`` rejecting the versions it does not implement.

    _STATE_MAGIC = b"QLATRNG"
    _STATE_VERSION = 1
    _STATE_WORDS = 24
    _STATE_SIZE = len(_STATE_MAGIC) + 1 + 4 * _STATE_WORDS

    def get_state(self):
        """
        Return the complete state as ``bytes`` (104 bytes, format version 1).\n
        The blob captures *everything* that determines the remaining stream:
        the 256-bit SHA-256 chaining value, the byte counter and type/index
        bookkeeping, the three cached ``rand_gen`` words and the Box-Muller
        Gaussian cache.  It is byte-compatible with ``qlat_utils.RngState``
        (see the layout documented above), so a state can be moved between a
        C++ ``qlat`` build and this package.\n
        Use :meth:`set_state` to restore it, or just ``pickle`` the object.
        """
        words = [self.num_bytes >> 32, self.num_bytes & MASK32]
        words += list(self.hash)
        words += [self.type >> 32, self.type & MASK32]
        words += [self.index >> 32, self.index & MASK32]
        for c in self.cache:
            words += [c >> 32, c & MASK32]
        #
        gaussian_bits = struct.unpack(">Q", struct.pack(">d", self.gaussian))[0]
        words += [gaussian_bits >> 32, gaussian_bits & MASK32]
        words += [self.cache_avail, 1 if self.gaussian_avail else 0]
        #
        return (
            self._STATE_MAGIC + bytes([self._STATE_VERSION]) + _pack_state_words(words)
        )

    def set_state(self, state):
        """
        Restore this generator from a :meth:`get_state` blob (or a pickle).\n
        ``state`` is the 104-byte ``bytes``-like blob of format version 1; it is
        validated completely *before* anything is assigned, so a corrupt or
        foreign blob raises ``ValueError`` and leaves the generator exactly as
        it was (no partial mutation).\n
        Raises ``TypeError`` if ``state`` is not ``bytes``-like.
        """
        words = _unpack_state(state)
        num_bytes = (words[0] << 32) | words[1]
        hash_words = list(words[2:10])
        type_ = (words[10] << 32) | words[11]
        index = (words[12] << 32) | words[13]
        cache = [(words[14 + 2 * i] << 32) | words[15 + 2 * i] for i in range(3)]
        gaussian_bits = (words[20] << 32) | words[21]
        gaussian = struct.unpack(">d", struct.pack(">Q", gaussian_bits))[0]
        cache_avail = words[22]
        gaussian_avail = bool(words[23])
        # Everything validated; only now touch the receiver's attributes.
        self.num_bytes = num_bytes
        self.hash = hash_words
        self.type = type_
        self.index = index
        self.cache = cache
        self.gaussian = gaussian
        self.cache_avail = cache_avail
        self.gaussian_avail = gaussian_avail

    def __getstate__(self):
        """
        Pickle protocol: return the :meth:`get_state` blob.\n
        ``pickle.loads(pickle.dumps(rs))`` therefore yields an independent copy
        that continues exactly the same stream, including cached values.
        """
        return self.get_state()

    def __setstate__(self, state):
        """Pickle protocol: restore from a :meth:`get_state` blob."""
        self.set_state(state)

    def __reduce__(self):
        """
        Pickle protocol, mirroring qlat's ``__reduce__``.\n
        Returns ``(RngState, (), state_blob)``: the unpickled object is built
        with the default constructor and then updated through
        :meth:`__setstate__`, so it is an independent copy of this generator.
        """
        return (RngState, (), self.get_state())

    def copy(self, is_copying_data=True):
        """
        Return a copy, like qlat's ``copy(is_copying_data=True)``.\n
        qlat returns a *default* state for ``copy(False)`` (its pyx builds
        ``RngState()`` and skips the assignment); this implementation keeps that
        behaviour for call-site compatibility.
        """
        r = RngState()
        if is_copying_data:
            r._assign(self)
        return r

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __imatmul__(self, v1):
        self._assign(v1)
        return self

    def __repr__(self):
        return (
            f"RngState(index={self.index}, type={self.type}, "
            f"num_bytes={self.num_bytes}, cache_avail={self.cache_avail}, "
            f"gaussian_avail={self.gaussian_avail})"
        )

    ### --- splitting --------------------------------------------------

    def split_inplace(self, sindex):
        """
        ``split_rng_state(rs, rs, sindex)``: fork this state, in place.\n
        The label is formatted as ``"[<index>] {<sindex>}"`` (or
        ``"[<type>,<index>] {<sindex>}"`` with a type tag), padded with ASCII
        spaces to a multiple of 64 *bytes*, and fed to the raw SHA-256
        compression function *without* final padding; ``num_bytes`` grows by the
        padded block count so later draws continue the hash at that offset.
        Because of that, a split depends on the accumulated ``num_bytes``, i.e.
        on the depth of the chain, not only on the labels.  Every byte of the
        label participates in the hash, including an embedded NUL: qlat used to
        cut the label at the first NUL because it formatted it with
        ``printf("%s")``, but that defect is fixed on both sides, so
        :func:`_truncate_at_nul` is now the identity (its docstring keeps the
        history).
        """
        if not isinstance(sindex, (str, bytes, bytearray)):
            # qlat's split() goes through as_bytes (NOT as_seed_bytes), so only
            # str/bytes are accepted here; note the constructor *does* accept an
            # int seed.  Return the same TypeError message as as_bytes.
            raise TypeError(f"expected str or bytes, got {type(sindex).__name__}")
        #
        sindex = _seed_bytes(sindex)
        sindex = _truncate_at_nul(sindex)
        if self.type == ULONG_MAX:
            data = b"[%d] {%s}" % (self.index, sindex)
        else:
            data = b"[%d,%d] {%s}" % (self.type, self.index, sindex)
        #
        # The block count uses the BYTE length and the padding is 0x20.
        n_blocks = (len(data) - 1) // 64 + 1
        data = data + b" " * (n_blocks * 64 - len(data))
        h = process_block(list(self.hash), data[0:64])
        for i in range(1, n_blocks):
            h = process_block(h, data[i * 64 : (i + 1) * 64])
        #
        self.hash = h
        self.num_bytes = self.num_bytes + n_blocks * 64
        self.type = ULONG_MAX
        self.index = 0
        self.cache = [0, 0, 0]
        self.gaussian = 0.0
        self.cache_avail = 0
        self.gaussian_avail = False

    def split(self, seed):
        """
        Return the child state for ``seed`` (this state is not changed).\n
        ``seed`` is a ``str`` (or ``bytes``, used verbatim as the label bytes);
        anything else raises ``TypeError``, exactly like qlat's binding, which
        goes through ``as_bytes``.  Note the *constructor* additionally accepts
        an ``int`` seed (rendered with ``str()``), matching C++
        ``RngState(const Long)``.
        """
        r = self.copy()
        r.split_inplace(seed)
        return r

    def newtype(self, type_):
        """
        Return a copy of this state with the stream type tag set.\n
        ``newtype`` is C++ ``RngState::newtype``: the counter string becomes
        ``"[type,index]"``, so the stream is independent of the untagged one.
        This is an extension: qlat's Python binding does not expose it.
        ``type_`` must be a non-negative integer other than ``ULONG_MAX`` (which
        marks an untagged state), and the state itself must be untagged; both
        violations raise ``ValueError`` (C++ uses ``qassert``).
        """
        if self.type != ULONG_MAX:
            raise ValueError(
                "newtype() needs an untagged state; this one already has type "
                f"{self.type}"
            )
        #
        if int(type_) in (ULONG_MAX, -1):
            raise ValueError(f"newtype({type_}) is not a valid type tag")
        #
        if int(type_) < 0:
            raise ValueError(f"newtype({type_}) is not a valid type tag")
        r = self.copy()
        r.type = int(type_) & ULONG_MAX
        r.cache_avail = 0
        r.gaussian_avail = False
        return r

    ### --- core generation --------------------------------------------

    def _hash_with_input(self, data):
        return process_input(self.hash, self.num_bytes, data)

    def rand_gen(self):
        """Return a uniform ``int`` in ``[0, 2**64 - 1]`` (bit-exact vs C++)."""
        assert 0 <= self.cache_avail <= 3
        self.index += 1
        if self.cache_avail > 0:
            self.cache_avail -= 1
            r = self.cache[self.cache_avail]
            self.cache[self.cache_avail] = 0
            return r
        #
        if self.type == ULONG_MAX:
            h = self._hash_with_input(b"[%d]" % self.index)
        else:
            h = self._hash_with_input(b"[%d,%d]" % (self.type, self.index))
        #
        self.cache[0] = _patch(h[0], h[1])
        self.cache[1] = _patch(h[2], h[3])
        self.cache[2] = _patch(h[4], h[5])
        self.cache_avail = 3
        return _patch(h[6], h[7])

    def u_rand_gen(self, lower=0.0, upper=1.0):
        """
        Return a uniform float in ``[lower, upper)``.\n
        Same mapping as C++ ``u_rand_gen``: ``u * 2**-64 * (upper - lower) +
        lower``, i.e. a single multiply-add on the raw 64-bit draw.\n
        The ``(lower, upper)`` argument order matches
        ``numpy.random.uniform`` and ``std::uniform_real_distribution``; it used
        to be ``(upper, lower)`` in both qlat and this module, and the streams of
        every caller are unchanged by the swap (the values passed were swapped
        with it).
        """
        u = self.rand_gen()
        return float(u) * TWO_POW_M64 * (upper - lower) + lower

    def g_rand_gen(self, center=0.0, sigma=1.0):
        """
        Return a Gaussian float with mean ``center`` and width ``sigma``.\n
        C++ ``g_rand_gen`` uses a hand-written polar (Marsaglia) Box-Muller
        with rejection, *not* ``std::normal_distribution``.  Two details are
        part of the observable stream and must be kept: the entry
        ``index += 1`` happens on *every* call (including the one that returns
        the cached second value of a pair), and the cached value is reused by
        the next call with that call's ``center``/``sigma``.
        """
        # C++ bumps the index on entry, then each u_rand_gen bumps it again.
        self.index += 1
        if self.gaussian_avail:
            self.gaussian_avail = False
            return self.gaussian * sigma + center
        #
        num_try = 1
        while True:
            v1 = self.u_rand_gen(-1.0, 1.0)
            v2 = self.u_rand_gen(-1.0, 1.0)
            rsq = v1 * v1 + v2 * v2
            num_try += 1
            if not (num_try < 10000 and (rsq >= 1.0 or rsq == 0.0)):
                break
        #
        # The C++ loop caps at 10000 tries and then returns 1e+10 with a
        # warning; unreachable in practice, ported for parity.
        if num_try > 9999:
            return 1e10
        #
        fac = math.sqrt(-2.0 * math.log(rsq) / rsq)
        self.gaussian = v1 * fac
        self.gaussian_avail = True
        return v2 * fac * sigma + center

    ### --- vectorized helpers (qlat_utils.rng_state.RngState) ----------

    def rand_arr(self, shape):
        """
        Return a ``numpy.uint64`` array of ``shape`` (bit-exact vs C++).\n
        The array is filled in C (row-major) order.
        """
        arr = np.empty(shape, dtype=np.uint64)
        self._rand_fill_u_long(arr.ravel())
        return arr

    def u_rand_arr(self, shape):
        """
        Return a ``numpy.float64`` array of ``shape`` uniform in ``[0, 1)``.\n
        This is the supported way to draw a float array; there is no public
        ``*_fill*`` method (qlat removed them, see qlat-issues entry 3).  For a
        different range map the result: ``lower + (upper - lower) * u``.
        """
        arr = np.empty(shape, dtype=np.float64)
        self._u_rand_fill_real_d(arr.ravel(), 0.0, 1.0)
        return arr

    def g_rand_arr(self, shape):
        """
        Return a ``numpy.float64`` array of ``shape``, standard normal.\n
        As with :meth:`u_rand_arr` there is no public ``*_fill*`` method; map
        the result for a different ``center``/``sigma``:
        ``center + sigma * g``.
        """
        arr = np.empty(shape, dtype=np.float64)
        self._g_rand_fill_real_d(arr.ravel(), 0.0, 1.0)
        return arr

    ### --- private array fillers (NOT public API) ---------------------
    #
    # qlat-issues entry 3: qlat removed every public ``*_fill*`` name, keeping
    # the loops only as private ``cdef`` helpers behind rand_arr/u_rand_arr/
    # g_rand_arr.  The same is done here (leading underscore, no Python API),
    # so no ``fill`` name is public on either side.

    def _rand_fill_u_long(self, arr):
        """Fill a 1-D ``uint64`` array with ``rand_gen`` values (private)."""
        arr = _get_buffer(arr, np.uint64, "ULong")
        for i in range(arr.shape[0]):
            arr[i] = self.rand_gen()

    def _u_rand_fill_real_d(self, arr, lower=0.0, upper=1.0):
        """Fill a 1-D ``float64`` array with ``u_rand_gen`` values (private)."""
        arr = _get_buffer(arr, np.float64, "double")
        for i in range(arr.shape[0]):
            arr[i] = self.u_rand_gen(lower, upper)

    def _g_rand_fill_real_d(self, arr, center=0.0, sigma=1.0):
        """Fill a 1-D ``float64`` array with ``g_rand_gen`` values (private)."""
        arr = _get_buffer(arr, np.float64, "double")
        for i in range(arr.shape[0]):
            arr[i] = self.g_rand_gen(center, sigma)

    ### --- range-mapped helpers ---------------------------------------

    def c_rand_gen(self, size):
        """
        Return a random coordinate in ``size`` as a 4-tuple.\n
        C++ ``c_rand_gen`` is ``coordinate_from_index(rand_gen() % volume(size),
        size)``, i.e. plain modulo (a negligible bias when ``volume`` does not
        divide ``2**64``; "fixing" it would break parity).  qlat's binding takes a
        ``Coordinate``; this accepts any length-4 sequence of ints and returns a
        plain tuple, although ``qlat_ref.base.Coordinate`` now exists (an
        extension: narrowing the parameter would be an API change needing a
        parity-allowlist entry, and the values are already identical — see
        qlat-issue-7).
        """
        total_vol = 1
        for v in size:
            total_vol *= v
        #
        return tuple(coordinate_from_index(self.rand_gen() % total_vol, size))

    def select(self, seq):
        """
        Return a uniformly chosen element of the sized/indexable sequence ``seq``.\n
        ``seq`` may be a ``list``, ``tuple``, ``range``, ``numpy.ndarray`` or any
        other object implementing ``__len__`` and ``__getitem__``; the element
        is returned exactly as ``seq[index]`` gives it (so an ``ndarray`` yields
        a NumPy scalar and a ``str`` a one-character ``str``).  Exactly one
        ``rand_gen()`` draw is consumed, so the stream is identical for any
        container of the same length and unchanged from the former list-only
        implementation.\n
        Raises ``TypeError`` if ``seq`` is not sized and indexable (e.g. ``None``,
        an ``int``, or a generator) and ``ValueError`` if it is empty.  This
        matches ``qlat_utils.RngState.select`` exactly (see
        ``tmp/qlat-ref-reorg/qlat-issues.md``, qlat-issue-7).
        """
        try:
            n = len(seq)
        except TypeError:
            raise TypeError(
                f"select() requires a sized sequence (list, tuple, range, "
                f"ndarray or __len__/__getitem__ object), got "
                f"{type(seq).__name__}"
            ) from None
        #
        if n == 0:
            raise ValueError("select() cannot choose from an empty sequence")
        #
        return seq[self.rand_gen() % n]

### -------------------------------------------------------------------

def random_permute(l, rs):
    """
    Return a new list with the elements of ``l`` permuted (``l`` is unchanged).\n
    Same biased-but-deterministic Fisher-Yates as C++ ``random_permute``:
    ``kk = rand_gen() % (size - k)`` with ``k`` ascending, swapping ``l[k]`` and
    ``l[k + kk]``.  The passed ``RngState`` is not advanced (C++ takes it by
    value), so this copies it.
    """
    rs = rs.copy()
    l_new = list(l)
    size = len(l_new)
    for k in range(size):
        kk = rs.rand_gen() % (size - k)
        l_new[k], l_new[k + kk] = l_new[k + kk], l_new[k]
    #
    return l_new

def get_data_sig(x, rs):
    """
    Return a real or complex signature of ``x`` viewed as a 1-D array.\n
    Same as qlat's ``get_data_sig``: an ``np.ndarray`` is multiplied
    element-wise by a uniform random array in ``[-1, 1]`` and summed (the passed
    ``RngState`` is copied first, so it is not advanced).  ``int``/``float`` and
    ``complex`` return themselves, and an object without a signature returns
    ``None``.
    """
    if isinstance(x, np.ndarray):
        rs = rs.copy()
        arr = x.ravel()
        arr_rand = rs.u_rand_arr(arr.shape) * 2.0 - 1.0
        return np.sum(arr * arr_rand)
    #
    if isinstance(x, (int, float)):
        return float(x)
    #
    if isinstance(x, complex):
        return complex(x)
    #
    if hasattr(x, "get_data_sig"):
        return x.get_data_sig(rs)
    #
    return None
