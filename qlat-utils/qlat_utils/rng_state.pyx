# cython: binding=True, embedsignature=True

"""
Module ``qlat_utils.rng_state``
===============================\n
Deterministic SHA-256-based random number generator and data signature utilities.\n
Documentation: ``docs/qlat-utils/qlat_rng_state.md``\n
.. note:: Update the documentation when updating this source file.
"""

from . cimport everything as cc

from .mat cimport *
from .coordinate cimport *
from .lat_data cimport *

from cpython cimport Py_buffer
from cpython.buffer cimport PyBUF_FORMAT
cimport numpy

class q:
    from .timer import (
        timer,
    )

import numpy as np
import functools

cdef inline bytes as_seed_bytes(object x):
    """
    Return the seed/label bytes for ``x``, used by both the constructor and ``split``.\n
    A ``str`` is encoded as UTF-8, a ``bytes`` object is used verbatim (so it can
    carry a non-UTF-8 or NUL-containing label), and any other object is rendered
    with ``str()`` first (e.g. an ``int`` seed gives its decimal string).
    """
    if isinstance(x, (str, bytes,)):
        return cc.as_bytes(x)
    return cc.as_bytes(str(x))

cdef inline bytes to_state_bytes(object x):
    """
    Coerce a buffer-like state object to ``bytes`` for ``RngState.set_state``.\n
    Accepts ``bytes``, ``bytearray`` and ``memoryview`` (and anything exposing the
    buffer protocol); everything else raises ``TypeError``.
    """
    if isinstance(x, bytes):
        return x
    try:
        return bytes(memoryview(x))
    except TypeError:
        raise TypeError(
            "RngState.set_state: expected a bytes-like object, got "
            + type(x).__name__
        ) from None

cdef class RngState:

    def __init__(self, x=None, y=None):
        """
        ``RngState()`` | ``RngState(seed)`` | ``RngState(parent)`` |
        ``RngState(parent, seed)``.\n
        ``seed`` may be ``str`` (encoded as UTF-8), ``bytes`` (used verbatim) or
        any other object (rendered with ``str()``, e.g. an ``int``).  A ``bytes``
        seed is the raw label bytes, not its ``repr``.\n
        Argument validation raises ``TypeError`` and is active in every mode,
        including ``python -O``.
        """
        cdef cc.std_string seed
        if x is None:
            if y is not None:
                raise TypeError(
                    "RngState(): a second argument requires an RngState as the "
                    f"first argument, got {type(x).__name__} and {type(y).__name__}"
                )
            # make a new rng
            self.xx = cc.RngState()
        elif isinstance(x, RngState):
            if y is None:
                # make a copy of x
                self.xx = (<RngState>x).xx
            else:
                # split into a new rng
                seed = as_seed_bytes(y)
                self.xx = cc.RngState((<RngState>x).xx, seed)
        else:
            if y is not None:
                raise TypeError(
                    "RngState(): a second argument requires an RngState as the "
                    f"first argument, got {type(x).__name__} and {type(y).__name__}"
                )
            # seed a new rng
            seed = as_seed_bytes(x)
            self.xx = cc.RngState(seed)

    def __imatmul__(self, RngState v1):
        self.xx = v1.xx
        return self

    def copy(self, cc.Bool is_copying_data=True):
        cdef RngState x = RngState()
        if is_copying_data:
            x.xx = self.xx
        return x

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def get_state(self):
        """
        Return the complete generator state as ``bytes`` (104 bytes).\n
        The format is versioned and shared with ``qlat_ref``: the 8-byte header
        ``b"QLATRNG" + bytes([0x01])`` followed by the 24 little-endian
        ``uint32`` words of the C++ ``exportRngState`` (byte counter, hash,
        type, index, word cache, Box-Muller cache).  A state restored with
        :meth:`set_state` continues the identical random stream.
        """
        return cc.as_bytes_std(cc.get_rng_state_bytes(self.xx))

    def set_state(self, state):
        """
        Restore a state produced by :meth:`get_state`.\n
        ``state`` may be ``bytes``, ``bytearray``, ``memoryview`` or any object
        supporting the buffer protocol.  Raises ``ValueError`` on a malformed
        state (wrong length, magic or version) and leaves this generator
        unchanged in that case.
        """
        cdef cc.std_string state_std = to_state_bytes(state)
        cc.set_rng_state_bytes(self.xx, state_std)

    def __reduce__(self):
        return (RngState, (), self.get_state())

    def __setstate__(self, state):
        self.set_state(state)

    def split(self, seed):
        """
        Return a new ``RngState`` derived from this one and the label ``seed``.\n
        ``seed`` may be ``str`` (hashed as its UTF-8 bytes) or ``bytes`` (hashed
        verbatim, so it may contain any byte including NUL).
        """
        cdef bytes seed_b = cc.as_bytes(seed)
        cdef RngState x = RngState()
        x.xx = self.xx.split(seed_b)
        return x

    def rand_gen(self):
        """Generate a uniformly distributed random integer ranges from 0 up to 2**64 - 1"""
        return cc.rand_gen(self.xx)

    def u_rand_gen(self, double lower=0.0, double upper=1.0):
        """
        Return a uniform random number in ``[lower, upper)``.\n
        The ``(lower, upper)`` argument order matches ``numpy.random.uniform``
        and ``std::uniform_real_distribution``; it used to be ``(upper, lower)``.
        """
        return cc.u_rand_gen(self.xx, lower, upper)

    def g_rand_gen(self, double center=0.0, double sigma=1.0):
        return cc.g_rand_gen(self.xx, center, sigma)

    def c_rand_gen(self, Coordinate size):
        """``size`` can be ``total_site`` of the lattice"""
        cdef Coordinate x = Coordinate()
        x.xx = cc.c_rand_gen(self.xx, size.xx)
        return x

    def select(self, seq):
        """
        Return a uniformly chosen element of the sized/indexable sequence ``seq``.\n
        ``seq`` may be a ``list``, ``tuple``, ``range``, ``numpy.ndarray`` or any
        other object implementing ``__len__`` and ``__getitem__``; the element
        is returned exactly as ``seq[index]`` gives it.  One ``rand_gen()`` draw
        is consumed, so the stream is identical for any container of the same
        length.\n
        Raises ``TypeError`` if ``seq`` is not sized and indexable, and
        ``ValueError`` if it is empty.
        """
        cdef Py_ssize_t n
        try:
            n = len(seq)
        except TypeError:
            raise TypeError(
                f"select() requires a sized sequence (list, tuple, range, "
                f"ndarray or __len__/__getitem__ object), got "
                f"{type(seq).__name__}"
            ) from None
        if n == 0:
            raise ValueError("select() cannot choose from an empty sequence")
        return seq[self.rand_gen() % n]

    def rand_arr(self, shape):
        """
        return numpy array with `shape` and dtype=np.uint64
        Uniformly distributed random integer ranges from 0 up to 2**64 - 1
        """
        arr = np.empty(shape, dtype=np.uint64)
        self._rand_fill_u_long(arr.ravel())
        return arr

    def u_rand_arr(self, shape):
        """
        return numpy array with `shape` and dtype=np.float64
        Uniform distribution from 0.0 to 1.0
        """
        arr = np.empty(shape, dtype=np.float64)
        self._u_rand_fill_real_d(arr.ravel(), 0.0, 1.0)
        return arr

    def g_rand_arr(self, shape):
        """
        return numpy array with `shape` and dtype=np.float64
        Gaussian distribution center=0.0 with sigma=1.0
        """
        arr = np.empty(shape, dtype=np.float64)
        self._g_rand_fill_real_d(arr.ravel(), 0.0, 1.0)
        return arr

    # The three ``*_fill*`` helpers below are internal implementation details of
    # rand_arr()/u_rand_arr()/g_rand_arr() and are deliberately NOT part of the
    # Python API (they are cdef, not def): the old public u_rand_fill() /
    # g_rand_fill() family was removed in qlat-issue-3 because it reinterpreted
    # arbitrary buffers as float64 and could silently corrupt them.  Use
    # rand_arr()/u_rand_arr()/g_rand_arr() instead.

    cdef void _rand_fill_u_long(self, cc.ULong[:] arr):
        cdef cc.Long size = arr.size
        cdef cc.Long i
        for i in range(size):
            arr[i] = cc.rand_gen(self.xx)

    cdef void _u_rand_fill_real_d(self, double[:] arr, const double lower,
                                  const double upper):
        cdef cc.Long size = arr.size
        cdef cc.Long i
        for i in range(size):
            arr[i] = cc.u_rand_gen(self.xx, lower, upper)

    cdef void _g_rand_fill_real_d(self, double[:] arr, const double center,
                                 const double sigma):
        cdef cc.Long size = arr.size
        cdef cc.Long i
        for i in range(size):
            arr[i] = cc.g_rand_gen(self.xx, center, sigma)

### -------------------------------------------------------------------

@q.timer
def get_data_sig(x, RngState rs):
    """
    Return a signature (a floating point number, real or complex) of data viewed as a 1-D array of numbers.\n
    Result only depends on the value of the data, not the structure.
    ``x`` can be an instance of ``LatData``, ``np.ndarray``, etc.
    """
    if isinstance(x, np.ndarray):
        rs = rs.copy()
        arr = x.ravel()
        arr_rand = rs.u_rand_arr(arr.shape) * 2.0 - 1.0
        return np.sum(arr * arr_rand)
    elif isinstance(x, (LatData, LatDataInt, LatDataRealF, LatDataLong)):
        return get_data_sig(np.asarray(x), rs)
    elif isinstance(x, (SpinMatrix, ColorMatrix, WilsonMatrix,)):
        return get_data_sig(np.asarray(x), rs)
    elif isinstance(x, (int, float,)):
        return float(x)
    elif isinstance(x, (complex,)):
        return complex(x)
    elif hasattr(x, "get_data_sig"):
        return x.get_data_sig(rs)
    else:
        return None

@q.timer
def random_permute(list l, RngState rs):
    """
    Do not change `l`.
    Return a new permuted list.
    """
    cdef cc.Long size = len(l)
    cdef cc.std_vector[cc.PyObject*] vec
    vec.resize(size)
    cdef cc.Long i = 0
    for i in range(size):
        vec[i] = <cc.PyObject*>l[i]
    cc.random_permute(vec, rs.xx)
    cdef list l_new = []
    for i in range(size):
        l_new.append(<object>vec[i])
    return l_new
