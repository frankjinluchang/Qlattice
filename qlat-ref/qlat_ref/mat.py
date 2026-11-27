"""
Module ``qlat_ref.mat``
=======================\n
Colour-matrix algebra: the :class:`ColorMatrix` type, the adjoint, the
anti-hermitian traceless projection, the truncated matrix exponential, the
random ``SU(3)`` and anti-hermitian generators, the unitarity measures and the
Gram-Schmidt unitarization.\n
Documentation: ``docs/qlat-ref/qlat_types.md``\n
.. note:: Update the documentation when updating this source file.\n
Conventions
-----------\n
:class:`ColorMatrix` stores a ``3 x 3`` complex matrix internally as a
``numpy.complex128`` array and exposes it through the buffer protocol, so
``np.asarray(cm)`` is a zero-copy ``(3, 3)`` view -- that is the documented
array bridge between the reference types and plain NumPy.\n
The residual helpers (:func:`get_matrix_adjoint`,
:func:`get_traceless_antihermitian_part`, :func:`get_su3_exp`,
:func:`get_unitarity_error` and friends) work on plain NumPy arrays whose last
two axes are the colour axes, so they apply to a single matrix, a site field
``(Nt, Nx, Ny, Nz, NC, NC)`` and a link field ``(4, Nt, Nx, Ny, Nz, NC, NC)``
alike.
"""

import numpy as np

from .base import NC

### -------------------------------------------------------------------

_GELL_MANN = np.asarray(
    [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]] / np.sqrt(3.0),
    ],
    dtype=np.complex128,
)

# qlat's `make_anti_hermitian_matrix` (qlat-utils/.../matrix-hmc.h) expands its
# 8 real coefficients into the anti-Hermitian basis ``T_a = 1j * lambda_a``,
# where ``lambda_a`` are the standard Hermitian Gell-Mann matrices of
# ``_GELL_MANN``.  That basis is anti-Hermitian (``T_a^dag = -T_a``) and
# normalized as ``Tr[T_a T_b] = -2 delta_ab``, which is the convention the
# random-matrix constructors in :mod:`qlat_ref.field` rely on.  The C++
# interleaved-double index layout of that expansion is reproduced inline in
# :func:`qlat_ref.field._get_basis_antihermitian_matrix`, not here.
_QLAT_GENERATORS = 1j * _GELL_MANN

### -------------------------------------------------------------------

class ColorMatrix:
    """
    A ``3 x 3`` complex (colour) matrix.\n
    Mirrors ``qlat_utils.mat.ColorMatrix``: element access and the buffer
    protocol via ``numpy``, in-place and out-of-place ``+``/``-``/scalar ``*``,
    ``@=`` assignment, ``copy``, ``set_zero``, ``qnorm``, ``adjoint``,
    ``conjugate``, ``transpose`` and comparison by ``qnorm`` difference.\n
    .. note:: ``np.asarray(cm)`` is a **zero-copy** view of the internal
       buffer (the documented array bridge), so writing to it modifies the
       matrix in place.  A plain ``numpy`` array is *not* accepted where a
       :class:`ColorMatrix` is expected (qlat is equally strict); use
       ``np.asarray`` to get one and :meth:`_from_array` to wrap one.
    """

    __slots__ = ("_x",)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Reflected operators must win over numpy's, which would otherwise
        # convert the matrix through its buffer to an ndarray.
        return NotImplemented

    def __init__(self, *args):
        # qlat's ColorMatrix accepts and ignores any positional arguments and
        # always starts zeroed; the same here.
        self._x = np.zeros((NC, NC), dtype=np.complex128)

    def __array__(self, dtype=None, copy=None):
        """
        Return the internal ``(3, 3)`` buffer.\n
        ``numpy.asarray(m)`` is a zero-copy view of the matrix;
        ``numpy.array(m)`` is an independent copy (NumPy passes ``copy=True``
        to ``__array__`` and the request is honoured here, matching the field
        types and qlat's buffer behaviour).
        """
        arr = self._x if copy is not True else self._x.copy()
        if dtype is None:
            return arr
        return arr.astype(dtype, copy=False)

    @classmethod
    def _from_array(cls, arr):
        """Return a new matrix holding a copy of the ``(3, 3)`` array ``arr``."""
        x = cls()
        x._x[...] = arr
        return x

    def __imatmul__(self, v1):
        self._x[...] = v1._x
        return self

    def copy(self, is_copying_data=True):
        """Return a copy of the matrix (``is_copying_data=False``: zeros)."""
        if not is_copying_data:
            return ColorMatrix()
        x = ColorMatrix()
        x._x[...] = self._x
        return x

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __getstate__(self):
        return np.asarray(self._x.tolist(), dtype=np.complex128)

    def __setstate__(self, state):
        self._x = np.asarray(state, dtype=np.complex128).reshape(NC, NC)

    def __setitem__(self, idx, val):
        """Index the internal buffer directly (implemented with ``numpy``)."""
        self._x[idx] = val

    def __getitem__(self, idx):
        """Index the internal buffer directly (implemented with ``numpy``)."""
        return self._x[idx]

    def __repr__(self):
        return f"ColorMatrix({self._x.tolist()!r})"

    def set_zero(self):
        """Set every element to zero (in place)."""
        self._x[...] = 0.0

    def qnorm(self):
        """Return ``sum_{i,j} |m[i, j]|^2``."""
        return float(np.sum(np.abs(self._x) ** 2))

    def __eq__(self, v1):
        if not isinstance(v1, ColorMatrix):
            return NotImplemented
        # qlat compares `qnorm(self - v1) == 0.0`, i.e. all elements equal.
        return float(np.sum(np.abs(self._x - v1._x) ** 2)) == 0.0

    def __ne__(self, v1):
        r = self.__eq__(v1)
        if r is NotImplemented:
            return r
        return not r

    def __iadd__(self, v1):
        self._x += v1._x
        return self

    def __isub__(self, v1):
        self._x -= v1._x
        return self

    def __imul__(self, v1):
        self._x *= v1
        return self

    def __add__(self, v1):
        x = ColorMatrix()
        x._x[...] = self._x + v1._x
        return x

    def __sub__(self, v1):
        x = ColorMatrix()
        x._x[...] = self._x - v1._x
        return x

    def __mul__(self, v1):
        x = ColorMatrix()
        x._x[...] = v1 * self._x
        return x

    def __rmul__(self, v1):
        # Scalar multiplication is commutative: ``2 * cm == cm * 2``.
        return self.__mul__(v1)

    def conjugate(self):
        """Return the element-wise complex conjugate as a new matrix."""
        x = ColorMatrix()
        x._x[...] = np.conjugate(self._x)
        return x

    def transpose(self):
        """Return the transpose (no conjugation) as a new matrix."""
        x = ColorMatrix()
        x._x[...] = self._x.T
        return x

    def adjoint(self):
        """Return the conjugate transpose as a new matrix."""
        x = ColorMatrix()
        x._x[...] = get_matrix_adjoint(self._x)
        return x

### -------------------------------------------------------------------

def get_matrix_adjoint(a):
    """Return the conjugate transpose over the last two (colour) axes."""
    return np.swapaxes(np.conjugate(a), -1, -2)

def get_traceless_antihermitian_part(a):
    """
    Return ``(a - a^dagger) / 2`` with its trace removed.\n
    The result is anti-hermitian and traceless; it is the Lie-algebra element
    used for the local gauge update.
    """
    h = (a - get_matrix_adjoint(a)) / 2
    tr = np.trace(h, axis1=-2, axis2=-1)
    return h - tr[..., None, None] * np.eye(NC) / NC

def get_su3_exp(h, *, max_order=20):
    """
    Return ``exp(h)`` for an anti-hermitian traceless matrix field ``h``.\n
    The exponential is evaluated with the same truncated Taylor (Horner)
    recurrence as qlat's ``make_matrix_exp`` (truncation order ``max_order``),
    so that the reference and the qlat implementation use identical
    arithmetic::\n
        t2 = h
        for j in range(max_order, 1, -1):
            t2 = h @ (1 + t2 / j)
        return 1 + t2\n
    .. note:: ``make_matrix_exp`` is a C++ inline function that qlat does not
       export to Python; this is the reference implementation of it.
    """
    h = np.asarray(h, dtype=np.complex128)
    unit = np.broadcast_to(np.eye(NC), h.shape)
    t2 = h.copy()
    for order in range(max_order, 1, -1):
        t2 = h @ (unit + t2 / order)
    return unit + t2

def get_unitarized(m):
    """
    Return the Gram-Schmidt unitarization of the last two axes of ``m``.\n
    This is qlat's ``unitarize`` (``qlat-utils/.../matrix-hmc.h``): the rows
    are taken as three vectors, row ``1`` is normalized, row ``2`` is
    orthogonalized against row ``1`` and normalized, and row ``3`` is replaced
    by ``(row_1 x row_2)^*``.  The result is an ``SU(3)`` matrix (or a field
    of them) up to the rounding of the Gram-Schmidt step.
    """
    m = np.asarray(m, dtype=np.complex128)
    p1 = m[..., 0, :].copy()
    p2 = m[..., 1, :].copy()
    p3 = m[..., 2, :].copy()
    # normalize_array_complex(p1)
    n1 = np.sqrt(np.sum(np.abs(p1) ** 2, axis=-1))
    p1 = _divide_rows(p1, n1)
    # orthogonalize_array_complex(p2, p1)
    c = np.sum(np.conjugate(p1) * p2, axis=-1)
    p2 = p2 - _multiply_rows(p1, c)
    # normalize_array_complex(p2)
    n2 = np.sqrt(np.sum(np.abs(p2) ** 2, axis=-1))
    p2 = _divide_rows(p2, n2)
    # cross_product_conj(p3, p1, p2)
    p3 = np.conjugate(
        np.stack(
            [
                p1[..., 1] * p2[..., 2] - p1[..., 2] * p2[..., 1],
                p1[..., 2] * p2[..., 0] - p1[..., 0] * p2[..., 2],
                p1[..., 0] * p2[..., 1] - p1[..., 1] * p2[..., 0],
            ],
            axis=-1,
        )
    )
    return np.stack([p1, p2, p3], axis=-2)

def _divide_rows(v, s):
    """Divide each row of ``v`` by the scalar ``s`` (helper)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return v / s[..., None]

def _multiply_rows(v, s):
    """Multiply each row of ``v`` by the scalar ``s`` (helper)."""
    return v * s[..., None]

def get_unitarity_error(gf):
    """
    Return the maximal ``|U U^dagger - 1|`` matrix element of the field.\n
    The argument is named ``gf`` and the signature is qlat's
    ``get_unitarity_error(gf)``; the accepted *types* are a superset of qlat's
    (a justified extra, see ``docs/qlat-ref/qlat_ref_overview.md``): a
    :class:`ColorMatrix`, a link field ``(4, Nt, Nx, Ny, Nz, NC, NC)``, a site
    field ``(Nt, Nx, Ny, Nz, NC, NC)`` or a single ``(NC, NC)`` matrix all
    work, where qlat requires a ``GaugeField`` or a ``GaugeTransform``.  The
    strict, qlat-identical type contract is the typed entry point
    :func:`qlat_ref.observables.get_unitarity_error_typed`, which is what a
    cross-check of the *type layer* should use.
    """
    arr = np.asarray(gf, dtype=np.complex128)
    deviation = arr @ get_matrix_adjoint(arr) - np.eye(NC)
    return float(np.max(np.abs(deviation)))

def check_unitarity(gf, *, eps=1e-12):
    """Return whether every link of ``gf`` is unitary within ``eps``."""
    return get_unitarity_error(gf) <= eps

### -------------------------------------------------------------------

def _get_random_su3_field(shape, *, seed, scale):
    """Return a deterministic random ``SU(3)`` field of the given shape (internal)."""
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal(tuple(shape) + (8,))
    h = np.zeros(tuple(shape) + (NC, NC), dtype=np.complex128)
    for a in range(8):
        h += 1j * scale * coef[..., a][..., None, None] * (_GELL_MANN[a] / 2)
    return get_su3_exp(h, max_order=30)

def get_random_su3_field(shape, *, seed=0, scale=1.0):
    """
    Return a deterministic random ``SU(3)`` field of the given shape.\n
    Eight Gaussian random coefficients per site are exponentiated into an
    ``SU(3)`` matrix; ``scale`` controls their spread (``scale=0.0`` gives the
    unit field).  This is the reference random field used by the gauge-fixing
    cross-check, not a bit-compatible reproduction of qlat's RNG.
    """
    return _get_random_su3_field(shape, seed=seed, scale=scale)
