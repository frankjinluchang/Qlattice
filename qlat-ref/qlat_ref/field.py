"""
Module ``qlat_ref.field``
=========================\n
Lattice gauge field types and the gauge transformations that act on them:\n
- :class:`FieldColorMatrix` -- the base site field of colour matrices
  (``(local_volume, multiplicity, NC, NC)`` buffer);
- :class:`GaugeField` -- a link field (multiplicity ``4``);
- :class:`GaugeTransform` -- a gauge transformation (multiplicity ``1``);
- :class:`GaugeMomentum` -- a conjugate momentum field (multiplicity ``4``).\n
Documentation: ``docs/qlat-ref/qlat_types.md``\n
.. note:: Update the documentation when updating this source file.\n
Array bridge
------------\n
Every field type stores its data as a C-contiguous ``numpy`` array of shape::\n
    (local_volume, multiplicity, NC, NC)\n
with ``dtype = numpy.complex128`` and the **flat local site index** on axis
``0`` (the first coordinate varying fastest, matching
:meth:`qlat_ref.base.Geometry.coordinate_from_index`).  ``np.asarray(f)``
returns a zero-copy view of that buffer, which is the documented bridge
between these types and plain NumPy: read or write the array in place,
perform the numerical work with plain NumPy, and the field object follows.
That is also how the gauge-fixing reference
(:mod:`qlat_ref.gauge_fixing`) accepts both a :class:`GaugeField` and a plain
``(4, Nt, Nx, Ny, Nz, NC, NC)`` array.\n
.. note:: x/y/z lattice axes are **not** part of the buffer; use
   :func:`qlat_ref.field.get_array_from_field` and
   :func:`qlat_ref.field.get_field_from_array` to convert between the buffer
   and the ``(Nt, Nx, Ny, Nz, ...)`` array convention used by the rest of the
   package.
"""

import numpy as np

from .base import NC
from .base import Coordinate
from .base import Geometry

from .mat import get_matrix_adjoint
from .mat import get_su3_exp
from .mat import get_unitarized

### -------------------------------------------------------------------

def _get_geo_checked(geo, multiplicity):
    """Return the geometry of a field, validating the multiplicity (helper)."""
    if geo is None:
        raise ValueError(
            "qlat_ref: a field needs a Geometry (qlat's generic `Field` types"
            " raise ZeroDivisionError for geo=None; pass a Geometry instead)"
        )
    if not isinstance(geo, Geometry):
        raise TypeError(f"qlat_ref: expected a Geometry, got {type(geo).__name__}")
    if multiplicity <= 0:
        raise ValueError(f"qlat_ref: multiplicity must be positive: {multiplicity}")
    return geo

def _get_site_array(geo, multiplicity):
    """Return a zeroed ``(local_volume, multiplicity, NC, NC)`` buffer (helper)."""
    local_volume = geo.local_volume
    return np.zeros((local_volume, multiplicity, NC, NC), dtype=np.complex128)

def _get_site_index(geo, x):
    """Flat local site index of a coordinate (helper)."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    return geo.index_from_coordinate(x)

### -------------------------------------------------------------------

class FieldColorMatrix:
    """
    A lattice field of ``NC x NC`` complex matrices, one per site and
    multiplicity slot.\n
    Mirrors ``qlat.field_types.FieldColorMatrix``: the NumPy buffer
    ``(local_volume, multiplicity, NC, NC)``, ``copy``, ``set_zero``,
    ``set_unit``, ``set_rand``, ``set_rand_g``, ``qnorm``, ``qnorm_field``,
    ``swap``, ``glb_sum``, ``@=`` assignment and the geometry properties
    (``geo``, ``total_site``, ``n_sites``, ``multiplicity``, ``sizeof_m``).\n
    .. note:: ``np.asarray(f)`` is a zero-copy view of the internal buffer, so
       ``np.asarray(f)[...] = x`` writes through to the field.  That is the
       documented array bridge; see the module docstring.
    """

    _MULTIPLICITY = 0
    _ELEM_SIZE = NC * NC * 16

    def __init__(self, *args):
        geo = args[0] if len(args) > 0 else None
        multiplicity = args[1] if len(args) > 1 else self._MULTIPLICITY
        self._geo = geo
        self._multiplicity = int(multiplicity)
        if geo is None:
            self._arr = np.zeros((0, self._multiplicity, NC, NC), dtype=np.complex128)
        else:
            if not isinstance(geo, Geometry):
                raise TypeError(
                    f"qlat_ref: expected a Geometry, got {type(geo).__name__}"
                )
            self._arr = _get_site_array(geo, self._multiplicity)

    def _check_init(self):
        """Raise if the field was created without a geometry (helper)."""
        if self._geo is None:
            raise ValueError(
                "qlat_ref: the field was created without a Geometry; reinitialize"
                " it with `field.init_from_geo(geo, multiplicity)` first"
            )

    def init_from_geo(self, geo=None, multiplicity=0):
        """(Re)initialize the field on ``geo`` with the given multiplicity."""
        if geo is None:
            self._geo = None
            self._arr = np.zeros((0, int(multiplicity), NC, NC), dtype=np.complex128)
            self._multiplicity = int(multiplicity)
            return
        if not isinstance(geo, Geometry):
            raise TypeError(f"qlat_ref: expected a Geometry, got {type(geo).__name__}")
        mult = int(multiplicity) if multiplicity else self._multiplicity
        if mult <= 0:
            raise ValueError(f"qlat_ref: multiplicity must be positive: {mult}")
        self._geo = geo
        self._multiplicity = mult
        self._arr = _get_site_array(geo, mult)

    def __array__(self, dtype=None, copy=None):
        """
        Return the internal buffer -- the array bridge (qlat's buffer protocol).\n
        ``numpy.asarray(f)`` returns a zero-copy **view** of the field, so
        writing into it writes through to the field; ``numpy.array(f)`` returns
        an independent **copy**, as it does for qlat's own field types (NumPy
        passes ``copy=True`` to ``__array__`` in that case, and the request is
        honoured here).  Taking a snapshot of a field before an in-place
        operation therefore means ``numpy.array(f)`` or ``f.copy()``.
        """
        arr = self._arr if copy is not True else self._arr.copy()
        if dtype is None:
            return arr
        return arr.astype(dtype, copy=False)

    @classmethod
    def _from_array(cls, arr, geo, multiplicity):
        """Return a new field holding a copy of the ``(V, mult, NC, NC)`` array."""
        f = cls(geo, multiplicity)
        f._arr[...] = arr
        return f

    def __imatmul__(self, other):
        """``self @= other``: copy the data of another field (geometry kept)."""
        if isinstance(other, FieldColorMatrix):
            if self._geo is None:
                self.init_from_geo(other.geo, other.multiplicity)
            self._arr[...] = other._arr
            return self
        raise TypeError(
            f"qlat_ref: FieldColorMatrix @= type mismatch: {type(other).__name__}"
        )

    def copy(self, is_copying_data=True):
        """Return a copy of the field (``is_copying_data=False``: empty object)."""
        f = type(self)()
        if is_copying_data:
            f @= self
        return f

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __getstate__(self):
        return [np.asarray(self).copy(), self.geo, self.multiplicity]

    def __setstate__(self, state):
        data_arr, geo, multiplicity = state
        self.__init__()
        self.init_from_geo(geo, multiplicity)
        self._arr[...] = data_arr

    def __len__(self):
        """Return the number of local sites (qlat's ``n_sites``)."""
        return self.n_sites

    def __getitem__(self, idx):
        """Index the internal buffer ``(local_volume, multiplicity, NC, NC)``."""
        return self._arr[tuple(idx) if isinstance(idx, tuple) else idx]

    def __setitem__(self, idx, val):
        """Index the internal buffer ``(local_volume, multiplicity, NC, NC)``."""
        self._arr[tuple(idx) if isinstance(idx, tuple) else idx] = val

    def __repr__(self):
        cls = type(self).__name__
        return (
            f"{cls}(geo={self.show_geo()}, multiplicity={self.multiplicity},"
            f" qnorm={self.qnorm():.6e})"
        )

    def show_geo(self):
        """Return the geometry repr of the field (helper of ``__repr__``)."""
        return repr(self._geo)

    @property
    def geo(self):
        """The :class:`Geometry` the field lives on."""
        return self._geo

    @property
    def total_site(self):
        """The global lattice size as a :class:`Coordinate`."""
        self._check_init()
        return self._geo.total_site

    @property
    def n_sites(self):
        """The number of locally owned sites."""
        if self._geo is None:
            return 0
        return self._geo.local_volume

    @property
    def multiplicity(self):
        """The number of matrices stored per site."""
        return self._multiplicity

    @property
    def sizeof_m(self):
        """The size in bytes of one per-site matrix (``NC * NC * 16``)."""
        return self._ELEM_SIZE

    def qnorm(self):
        """Return ``sum_{x, m, i, j} |f[x, m, i, j]|^2``."""
        return float(np.sum(np.abs(self._arr) ** 2))

    def qnorm_field(self):
        """Return the per-site ``qnorm`` as a plain ``(local_volume,)`` array."""
        return np.sum(np.abs(self._arr) ** 2, axis=(1, 2, 3))

    def set_zero(self):
        """Set every element to zero (in place)."""
        self._check_init()
        self._arr[...] = 0.0

    def set_unit(self, coef=1.0):
        """Set every per-site matrix to ``coef * 1`` (in place)."""
        self._check_init()
        self._arr[...] = np.eye(NC) * coef

    def set_rand(self, rng, upper=1.0, lower=0.0):
        """
        Fill with uniform random real/imaginary parts (qlat's ``set_u_rand``).\n
        The public argument order (``upper`` then ``lower``) is qlat's, and is
        kept: 148 repo call sites pass them positionally.  C++
        ``set_u_rand(f, rs, lower, upper)`` takes them lower-first, so the call
        below forwards ``(lower, upper)`` — exactly like
        ``field_types.pyx.in``.  Each real component of each element is drawn
        independently and uniformly from ``[lower, upper)``; the per-site
        stream is ``rng.newtype(global_site_index)``.
        """
        self._check_init()
        self._arr[...] = self._get_rand_field(
            rng,
            lambda rs, shape: rs.u_rand_arr(shape) * (upper - lower) + lower,
        )

    def set_rand_g(self, rng, center=0.0, sigma=1.0):
        """
        Fill with Gaussian random real/imaginary parts (qlat's ``set_g_rand``).\n
        Each real component is drawn independently from a normal distribution
        of mean ``center`` and standard deviation ``sigma``; the per-site
        stream is ``rng.newtype(global_site_index)``.
        """
        self._check_init()
        self._arr[...] = self._get_rand_field(
            rng, lambda rs, shape: rs.g_rand_arr(shape) * sigma + center
        )

    def _get_rand_field(self, rng, draw):
        """Draw a per-site random complex field with the qlat RNG layout (helper)."""
        geo = self._geo
        local_volume = geo.local_volume
        mult = self._multiplicity
        ret = np.zeros((local_volume, mult, NC, NC), dtype=np.complex128)
        for index in range(local_volume):
            xg = geo.g_coordinate_from_index(index)
            gindex = geo.index_from_g_coordinate(xg)
            rs = rng.newtype(gindex)
            # qlat fills the raw real/imaginary components in memory order,
            # i.e. element-major over the whole per-site block; drawing a real
            # array of twice the size and viewing it as complex reproduces it.
            real = draw(rs, (mult * NC * NC * 2,))
            ret[index] = (
                real.astype(np.float64).view(np.complex128).reshape(mult, NC, NC)
            )
        return ret

    def swap(self, f1):
        """Swap the data and geometry with another field of the same type."""
        self._arr, f1._arr = f1._arr, self._arr
        self._geo, f1._geo = f1._geo, self._geo
        self._multiplicity, f1._multiplicity = f1._multiplicity, self._multiplicity

    def glb_sum(self):
        """
        Return the global sum of every element.\n
        Single-process reference implementation: the sum is over the local
        (== global) sites, so this is just a plain ``numpy`` sum.
        """
        self._check_init()
        return np.sum(self._arr, axis=0)

### -------------------------------------------------------------------

def _get_link_array(gf):
    """Return the ``(4, Nt, Nx, Ny, Nz, NC, NC)`` array of a link field (helper)."""
    if isinstance(gf, FieldColorMatrix):
        if gf.geo is None:
            raise ValueError("qlat_ref: the field has no geometry")
        return get_array_from_field(gf)
    arr = np.asarray(gf, dtype=np.complex128)
    if arr.ndim != 7 or arr.shape[0] != 4:
        raise ValueError(
            f"qlat_ref: expected a link field with shape (4, Nt, Nx, Ny, Nz, NC,"
            f" NC), got shape {arr.shape}"
        )
    return arr

def get_staple_sum(gf, dirs=None):
    """
    Return the local ``K`` matrix on each site,\n
        ``K(x) = sum_{mu in dirs} (U_mu(x) + U_mu(x - mu)^dagger)``.\n
    It collects the links that couple to the local functional at ``x``.
    ``gf`` may be a :class:`GaugeField` or a plain ``(4, ...)`` array; the
    returned site field is a plain ``(Nt, Nx, Ny, Nz, NC, NC)`` array.
    """
    from .base import get_dirs
    #
    U = _get_link_array(gf)
    K = np.zeros_like(U[0])
    for mu in get_dirs(dirs):
        K = K + U[mu] + get_matrix_adjoint(np.roll(U[mu], 1, axis=mu))
    return K

def get_local_gauge_gradient(gf, dirs=None):
    """
    Return the local gradient matrix ``D(x)`` of the gauge functional.\n
    ``D(x)`` is the anti-hermitian traceless part of ``K(x)^dagger`` (with
    ``K`` from :func:`get_staple_sum`); it is the ascent direction of the
    functional and ``exp(alpha * D(x))`` with ``alpha > 0`` is the local gauge
    transformation that increases the functional.  The gauge condition is
    ``D(x) = 0`` for all ``x``.  This matches qlat's
    ``gf_set_random_local_gradient`` / ``make_tr_less_anti_herm_matrix``
    combination.\n
    ``gf`` may be a :class:`GaugeField` or a plain ``(4, ...)`` array.
    """
    from .mat import get_traceless_antihermitian_part
    #
    K = get_staple_sum(gf, dirs=dirs)
    return get_traceless_antihermitian_part(get_matrix_adjoint(K))

def get_unit_gauge_field(total_site):
    """Return the unit gauge field ``U_mu(x) = 1`` on ``total_site``."""
    shape = (4,) + tuple(total_site) + (NC, NC)
    return np.broadcast_to(np.eye(NC, dtype=np.complex128), shape).copy()

def get_random_gauge_field(total_site, *, seed=0, scale=1.0):
    """
    Return a random unitary gauge field on ``total_site``.\n
    The field is deterministic given ``seed``: eight Gaussian random
    coefficients per site and direction are exponentiated into ``SU(3)``
    matrices.  ``scale`` controls the spread of the Gaussian coefficients
    (``scale=1.0`` gives a strongly disordered field, ``scale=0.0`` the unit
    field).  This is the plain-array reference field used by the gauge-fixing
    cross-check.
    """
    from .mat import get_random_su3_field
    #
    return get_random_su3_field((4,) + tuple(total_site), seed=seed, scale=scale)

def get_random_gauge_transform(total_site, *, seed=0, scale=0.3):
    """
    Return a random ``SU(3)`` gauge transformation on ``total_site``.\n
    The transformation is deterministic given ``seed``.
    """
    from .mat import get_random_su3_field
    #
    return get_random_su3_field(tuple(total_site), seed=seed, scale=scale)

def apply_gauge_transform(U, g):
    """
    Return the gauge-transformed field ``U^g`` with\n
        ``U^g_mu(x) = g(x) U_mu(x) g(x + mu)^dagger``.\n
    ``U`` has shape ``(4, Nt, Nx, Ny, Nz, NC, NC)`` and ``g`` has shape
    ``(Nt, Nx, Ny, Nz, NC, NC)``.  This is the plain-array counterpart of
    ``GaugeTransform * GaugeField``.
    """
    U = np.asarray(U, dtype=np.complex128)
    g = np.asarray(g, dtype=np.complex128)
    if U.shape[0] != 4:
        raise ValueError(f"qlat_ref: not a link field: shape={U.shape}")
    if U.shape[1:] != g.shape:
        raise ValueError(
            f"qlat_ref: shape mismatch: U{tuple(U.shape)} g{tuple(g.shape)}"
        )
    return np.stack(
        [g @ U[mu] @ get_matrix_adjoint(np.roll(g, -1, axis=mu)) for mu in range(4)]
    )

def apply_gauge_transform_to_momentum(P, g, is_dagger=False):
    """
    Return the gauge-transformed momentum ``P^g`` with\n
        ``P^g_mu(x) = g(x) P_mu(x) g(x)^dagger``.\n
    ``P`` has shape ``(4, Nt, Nx, Ny, Nz, NC, NC)`` and ``g`` has shape
    ``(Nt, Nx, Ny, Nz, NC, NC)``, the same layouts as ``U`` and ``g`` in
    :func:`apply_gauge_transform`.  A momentum (a Lie-algebra valued site
    field in the representation of the link field) is attached to one site, so
    this transformation has **no** shift: it is
    :func:`apply_gauge_transform` without the ``numpy.roll`` of ``g``.
    With ``is_dagger=True`` return ``g(x)^dagger P_mu(x) g(x)``, which is the
    inverse transformation.  The inputs are not modified; a new array is
    returned.
    """
    P = np.asarray(P, dtype=np.complex128)
    g = np.asarray(g, dtype=np.complex128)
    if P.shape[0] != 4:
        raise ValueError(f"qlat_ref: not a momentum field: shape={P.shape}")
    if P.shape[1:] != g.shape:
        raise ValueError(
            f"qlat_ref: shape mismatch: P{tuple(P.shape)} g{tuple(g.shape)}"
        )
    g_dag = get_matrix_adjoint(g)
    if is_dagger:
        return np.stack([g_dag @ P[mu] @ g for mu in range(4)])
    return np.stack([g @ P[mu] @ g_dag for mu in range(4)])

def get_array_from_field(gf):
    """
    Return the ``(4, Nt, Nx, Ny, Nz, NC, NC)`` array of a link field.\n
    Inverse of :func:`get_field_from_array`.  ``gf`` may be a
    :class:`GaugeField`, a :class:`GaugeMomentum` or a plain buffer-shaped
    array (which is returned unchanged after a shape check).
    """
    geo = getattr(gf, "geo", None)
    arr = np.asarray(gf, dtype=np.complex128)
    if geo is None:
        return arr
    total_site = geo.total_site.to_tuple()
    multiplicity = getattr(gf, "multiplicity", 4)
    flat = arr.reshape(tuple(total_site) + (multiplicity, NC, NC), order="F")
    return np.moveaxis(flat, -3, 0)

def get_field_from_array(arr, geo=None):
    """
    Return a :class:`GaugeField` built from a ``(4, Nt, Nx, Ny, Nz, NC, NC)``
    array.\n
    Inverse of :func:`get_array_from_field`.  ``geo`` may be omitted when the
    array already has the link-field shape, in which case the geometry is
    built from it.
    """
    arr = np.asarray(arr, dtype=np.complex128)
    if arr.ndim != 7 or arr.shape[0] != 4:
        raise ValueError(f"qlat_ref: not a link-field array: shape={arr.shape}")
    if geo is None:
        geo = Geometry(Coordinate(arr.shape[1:5]))
    gf = GaugeField(geo)
    n = geo.local_volume
    flat = np.moveaxis(arr, 0, -3)
    gf._arr[...] = flat.reshape((n, 4, NC, NC), order="F")
    return gf

### -------------------------------------------------------------------

class GaugeField(FieldColorMatrix):
    """
    A lattice gauge field: one ``SU(3)`` matrix per site and link direction.\n
    Mirrors ``qlat.qcd.GaugeField``: multiplicity ``4`` (one matrix per link
    direction), :meth:`set_rand`, :meth:`unitarize`, :meth:`plaq`,
    :meth:`link_trace`, :meth:`su_projection`, :meth:`save` and
    :meth:`load` (the standard NERSC ``4D_SU3_GAUGE`` format), plus the
    inherited :class:`FieldColorMatrix` surface.
    """

    _MULTIPLICITY = 4

    def __init__(self, geo=None, multiplicity=4):
        if multiplicity != 4:
            raise ValueError(
                f"qlat_ref: a GaugeField has multiplicity 4, got {multiplicity}"
            )
        super().__init__(geo, 4)

    def set_rand(self, rng, sigma=0.5, n_step=1):
        """
        Fill with random ``SU(3)`` links (qlat's ``set_g_rand_color_matrix_field``).\n
        Each link is ``make_color_matrix_exp(make_g_rand_anti_hermitian_matrix(rs,
        sigma))`` and is multiplied by ``n_step - 1`` further such random
        matrices; the per-site stream is ``rng.split(str(global_site_index))``,
        which is what C++ ``set_g_rand_color_matrix_field`` uses (a split of
        the decimal string, *not* ``newtype``).
        ``sigma=0.5`` is qlat's default.
        """
        self._check_init()
        self[:] = _get_rand_color_matrix_field(
            self.geo, rng, sigma=sigma, n_step=n_step, multiplicity=4
        )

    def unitarize(self):
        """Gram-Schmidt unitarize every link in place (qlat's ``unitarize``)."""
        self._check_init()
        self._arr[...] = get_unitarized(self._arr)

    def su_projection(self, tolerance=1.0e-8):
        """
        Project every link onto ``SU(3)`` in place (qlat's ``gf_su_projection``).\n
        The reference implementation only needs the rounding-level projection,
        so this is :meth:`unitarize` (a Gram-Schmidt step); the ``tolerance``
        of qlat's trace-maximizing algorithm is accepted and ignored.
        """
        self.unitarize()

    def plaq(self):
        """Return the average plaquette (qlat's ``gf_avg_plaq``)."""
        self._check_init()
        return _get_avg_plaq(get_array_from_field(self))

    def link_trace(self):
        """Return the average link trace (qlat's ``gf_avg_link_trace``)."""
        self._check_init()
        return _get_avg_link_trace(get_array_from_field(self))

    def twist_boundary_at_boundary(self, lmom=-0.5, mu=3):
        """
        Multiply the links in direction ``mu`` at the far boundary by
        ``exp(2i pi lmom)`` (in place), matching
        ``qf_twist_boundary_at_boundary``.
        """
        self._check_init()
        arr = get_array_from_field(self)
        arr[mu, -1, ...] *= np.exp(2j * np.pi * lmom)
        self._arr[...] = np.moveaxis(arr, 0, -3).reshape(
            (self.geo.local_volume, 4, NC, NC), order="F"
        )

    def save(self, path):
        """Save with the standard NERSC ``4D_SU3_GAUGE`` format."""
        self._check_init()
        return _save_gauge_field_nersc(self, path)

    def load(self, path):
        """Load with the standard NERSC ``4D_SU3_GAUGE`` format."""
        return _load_gauge_field_nersc(self, path)

### -------------------------------------------------------------------

class GaugeTransform(FieldColorMatrix):
    """
    A gauge transformation: one ``SU(3)`` matrix per site.\n
    Mirrors ``qlat.qcd.GaugeTransform`` (multiplicity ``1``) including the
    ``*`` algebra: ``gt * gt`` composes (``t = t1 * t0``), ``gt * gf``
    transforms the gauge field with
    ``U_mu(x) <- gt(x) U_mu(x) gt(x + mu)^dagger``, ``gt * gm`` transforms a
    momentum at the same site (through the momentum's ``__rmul__``), and
    :meth:`inv` returns the site-wise adjoint.\n
    .. note:: The composition order follows qlat's
       ``gt_apply_gauge_transformation``: ``t0 <- t1 * t0``, i.e. the operand
       acts **first**.
    """

    _MULTIPLICITY = 1

    def __init__(self, geo=None, multiplicity=1):
        if multiplicity != 1:
            raise ValueError(
                f"qlat_ref: a GaugeTransform has multiplicity 1, got {multiplicity}"
            )
        super().__init__(geo, 1)

    def set_rand(self, rng, sigma=0.5, n_step=1):
        """
        Fill with random ``SU(3)`` transformations (like :meth:`GaugeField.set_rand`).\n
        ``sigma=0.5`` is qlat's default.
        """
        self._check_init()
        self[:] = _get_rand_color_matrix_field(
            self.geo, rng, sigma=sigma, n_step=n_step, multiplicity=1
        )

    def unitarize(self):
        """Gram-Schmidt unitarize every site in place (qlat's ``unitarize``)."""
        self._check_init()
        self._arr[...] = get_unitarized(self._arr)

    def inv(self):
        """Return the site-wise adjoint, i.e. the inverse transformation."""
        self._check_init()
        return _get_gauge_transform_inv(self)

    def __mul__(self, other):
        """Apply the transformation to a gauge field, momentum or transform."""
        self._check_init()
        if isinstance(other, GaugeTransform):
            return _get_gauge_transform_composed(self, other)
        if isinstance(other, GaugeMomentum):
            # same-site adjoint action, no shift (qlat's own `__rmul__` path)
            return gm_apply_gauge_transformation(other, self)
        if isinstance(other, GaugeField):
            return _get_gauge_transformed_field(self, other)
        if isinstance(other, (list, tuple)):
            return [self * p for p in other]
        return NotImplemented

    def save(self, path):
        """Save as double precision with the generic field format."""
        self._check_init()
        return _save_double_field_nersc(self, path)

    def load(self, path):
        """Load as double precision with the generic field format."""
        return _load_double_field_nersc(self, path)

### -------------------------------------------------------------------

class GaugeMomentum(FieldColorMatrix):
    """
    The conjugate momentum of a gauge field: one anti-Hermitian ``3 x 3``
    matrix per site and link direction (multiplicity ``4``).\n
    Mirrors ``qlat.hmc.GaugeMomentum``: :meth:`set_rand`,
    :meth:`set_rand_fa` and the ``gt * gm`` sugar, which is
    ``gm_apply_gauge_transformation(self, gt)`` (adjoint action at the **same**
    site, no shift).
    """

    _MULTIPLICITY = 4

    def __init__(self, geo=None):
        super().__init__(geo, 4)

    def set_rand(self, rng, sigma=1.0):
        """
        Fill with Gaussian anti-Hermitian matrices (qlat's
        ``set_rand_gauge_momentum``).\n
        Each matrix is ``make_g_rand_anti_hermitian_matrix(rng.split(str(gindex)),
        sigma)``, so it is distributed as ``exp[-Tr(m^2) / (2 sigma^2)]``.
        ``sigma=1.0`` is qlat's default.
        """
        self._check_init()
        self._arr[...] = _get_rand_antihermitian_field(self.geo, rng, sigma)

    def set_rand_fa(self, mf, rng):
        """
        Fill with Gaussian anti-Hermitian matrices scaled per site by
        ``sqrt(mf[x])`` (qlat's ``set_rand_gauge_momentum(gm, mf, rng)``).\n
        A ``+inf`` entry zeroes that site's matrix, matching qlat.
        """
        self._check_init()
        self.set_rand(rng, sigma=1.0)
        mf_arr = np.asarray(mf, dtype=np.float64).reshape(-1)
        if mf_arr.shape[0] != self.geo.local_volume:
            raise ValueError(
                f"qlat_ref: mf has {mf_arr.shape[0]} entries but the lattice has"
                f" {self.geo.local_volume} local sites"
            )
        fac = np.where(np.isinf(mf_arr), 0.0, np.sqrt(mf_arr))
        self._arr[...] = self._arr * fac[:, None, None, None]

    def __rmul__(self, other):
        """``gt * gm`` sugar for :func:`gm_apply_gauge_transformation`."""
        if isinstance(other, GaugeTransform):
            return gm_apply_gauge_transformation(self, other)
        return NotImplemented

### -------------------------------------------------------------------

def _get_basis_antihermitian_matrix(basis):
    """Assemble ``m = sum_a basis[a] T_a`` as qlat's C++ does (helper)."""
    p = np.zeros((NC, NC), dtype=np.complex128).reshape(-1).view(np.float64)
    s3 = basis[7] / np.sqrt(3.0)
    p[0] = 0.0
    p[8] = 0.0
    p[16] = 0.0
    p[1] = basis[2] + s3
    p[9] = -basis[2] + s3
    p[17] = -2.0 * s3
    p[2] = basis[1]
    p[3] = basis[0]
    p[4] = basis[4]
    p[5] = basis[3]
    p[6] = -basis[1]
    p[7] = basis[0]
    p[10] = basis[6]
    p[11] = basis[5]
    p[12] = -basis[4]
    p[13] = basis[3]
    p[14] = -basis[6]
    p[15] = basis[5]
    return p.view(np.complex128).reshape(NC, NC).copy()

def _get_g_rand_antihermitian_matrix(rs, sigma):
    """Return one random anti-Hermitian matrix (qlat's helper) (internal)."""
    s = sigma / np.sqrt(2.0)
    basis = np.asarray([rs.g_rand_gen(center=0.0, sigma=s) for _ in range(8)])
    return _get_basis_antihermitian_matrix(basis)

def _get_color_matrix_exp(a, max_order=20):
    """Return ``unitarize(make_matrix_exp(a))`` (qlat's helper) (internal)."""
    return get_unitarized(get_su3_exp(a, max_order=max_order))

def _get_rand_antihermitian_field(geo, rng, sigma, multiplicity=4):
    """Return a random anti-Hermitian field buffer (internal)."""
    local_volume = geo.local_volume
    ret = np.zeros((local_volume, multiplicity, NC, NC), dtype=np.complex128)
    for index in range(local_volume):
        xg = geo.g_coordinate_from_index(index)
        # qcd-utils.cpp uses ``RngState rsi(rs, gindex)``, i.e. a split of the
        # DECIMAL STRING of the global index -- not ``newtype`` (which core.h
        # set_u_rand/set_g_rand use).  Both matter for parity.
        rs = rng.split(str(geo.index_from_g_coordinate(xg)))
        for m in range(multiplicity):
            ret[index, m] = _get_g_rand_antihermitian_matrix(rs, sigma)
    return ret

def _get_rand_color_matrix_field(geo, rng, *, sigma, n_step, multiplicity=4):
    """Return a random ``SU(3)`` field buffer (internal)."""
    if n_step < 1:
        raise ValueError(f"qlat_ref: n_step must be positive: {n_step}")
    local_volume = geo.local_volume
    ret = np.zeros((local_volume, multiplicity, NC, NC), dtype=np.complex128)
    for index in range(local_volume):
        xg = geo.g_coordinate_from_index(index)
        # qcd-utils.cpp uses ``RngState rsi(rs, gindex)``, i.e. a split of the
        # DECIMAL STRING of the global index -- not ``newtype`` (which core.h
        # set_u_rand/set_g_rand use).  Both matter for parity.
        rs = rng.split(str(geo.index_from_g_coordinate(xg)))
        for m in range(multiplicity):
            v = _get_color_matrix_exp(_get_g_rand_antihermitian_matrix(rs, sigma))
            for _ in range(1, n_step):
                v = v @ _get_color_matrix_exp(
                    _get_g_rand_antihermitian_matrix(rs, sigma)
                )
            ret[index, m] = v
    return ret

### -------------------------------------------------------------------

# Time is component 3 of `total_site` (the qlat convention: `total_site` is
# (Lx, Ly, Lz, Lt), `Geometry::spatial_volume` divides out component 3).  There
# are two array conventions to keep apart: in the (mu, L0, L1, L2, L3, NC, NC)
# link layout the time axis is array axis 4, while after indexing `arr[mu]`
# the lattice axis of direction `mu` is simply array axis `mu`.
_TIME_DIR = 3
_TIME_AXIS = 4

def _get_shifted(arr, mu, shift=1):
    """Shift a ``(4, Nt, Nx, Ny, Nz, NC, NC)`` array by ``shift`` along ``mu``."""
    return np.roll(arr[mu], shift, axis=mu)

def _get_avg_plaq(arr):
    """Return the average plaquette of a link-field array (internal)."""
    total = 0.0
    for m1 in range(1, 4):
        for m2 in range(m1):
            # qlat: v[m1] * vms[m1][m2] * adjoint(v[m2] * vms[m2][m1])
            # with vms[m][n] = U_n(x + e_m), i.e.
            # P = U_m1(x) U_m2(x + m1) adjoint(U_m2(x) U_m1(x + m2))
            u_m1 = arr[m1]
            u_m2_at_m1 = np.roll(arr[m2], -1, axis=m1)
            u_m2 = arr[m2]
            u_m1_at_m2 = np.roll(arr[m1], -1, axis=m2)
            plaq = u_m1 @ u_m2_at_m1 @ get_matrix_adjoint(u_m2 @ u_m1_at_m2)
            total += float(np.mean(np.trace(plaq, axis1=-2, axis2=-1).real)) / NC
    total *= 2.0 / (4.0 * 3.0)
    return total

def _get_avg_link_trace(arr):
    """Return the average link trace of a link-field array (internal)."""
    total = 0.0
    for mu in range(4):
        total += float(np.mean(np.trace(arr[mu], axis1=-2, axis2=-1).real)) / NC
    return total / 4.0

### -------------------------------------------------------------------

def _get_gauge_transform_inv(gt):
    """Return the site-wise adjoint of a gauge transformation (internal)."""
    ret = GaugeTransform(gt.geo)
    ret._arr[...] = get_matrix_adjoint(gt._arr)
    return ret

def _get_gauge_transform_composed(gt1, gt0):
    """Return ``gt1 * gt0`` as qlat's ``gt0 <- gt1 * gt0`` (internal)."""
    ret = GaugeTransform(gt0.geo)
    ret._arr[...] = gt1._arr @ gt0._arr
    return ret

def _get_gauge_transformed_field(gt, gf0):
    """Return ``gt * gf0`` for a link field or momentum field (internal)."""
    geo = gf0.geo
    arr = get_array_from_field(gf0)
    t0 = np.asarray(gt._arr).reshape((geo.local_volume, NC, NC), order="F")
    # site-field buffer -> (Nt, Nx, Ny, Nz, NC, NC) for the shift
    total_site = geo.total_site.to_tuple()
    t_site = t0.reshape(tuple(total_site) + (NC, NC), order="F")
    v = np.empty_like(arr)
    for mu in range(4):
        t1 = np.roll(t_site, -1, axis=mu)
        v[mu] = t_site @ arr[mu] @ get_matrix_adjoint(t1)
    ret = GaugeField(geo)
    ret._arr[...] = np.moveaxis(v, 0, -3).reshape(
        (geo.local_volume, 4, NC, NC), order="F"
    )
    return ret

def gm_apply_gauge_transformation(gm, gt, is_dagger=False):
    """
    Return the adjoint action of ``gt`` on the momentum ``gm``,\n
        ``gm_mu(x) <- gt(x) gm_mu(x) gt(x)^dagger``,\n
    at the **same** site for every ``mu`` (no shift).  With
    ``is_dagger=True`` return the inverse action
    ``gt(x)^dagger gm_mu(x) gt(x)``.\n
    Mirrors ``qlat.hmc.gm_apply_gauge_transformation``.  The inputs are not
    modified; a new :class:`GaugeMomentum` is returned.
    """
    geo = gm.geo
    t0 = np.asarray(gt._arr).reshape((geo.local_volume, NC, NC)).copy()
    t0 = t0.reshape(tuple(geo.total_site.to_tuple()) + (NC, NC), order="F")
    t0_dag = get_matrix_adjoint(t0)
    arr = get_array_from_field(gm)  # (4, Nt, Nx, Ny, Nz, NC, NC)
    if is_dagger:
        v = np.einsum("...ij,m...jk,...kl->m...il", t0_dag, arr, t0)
    else:
        v = np.einsum("...ij,m...jk,...kl->m...il", t0, arr, t0_dag)
    ret = GaugeMomentum(geo)
    ret._arr[...] = np.moveaxis(v, 0, -3).reshape(
        (geo.local_volume, 4, NC, NC), order="F"
    )
    return ret

### -------------------------------------------------------------------

def _get_crc32(data):
    """Return the zlib CRC32 of a bytes-like object (helper)."""
    import zlib
    #
    return zlib.crc32(data) & 0xFFFFFFFF

def _get_simple_checksum(arr, *, is_big_endian=False):
    """
    Return qlat's ``field_simple_checksum`` of a complex128 array (helper).\n
    The checksum is the 32-bit sum of the machine-endian 32-bit words, i.e. it
    is computed on the **pre-swap** data.  For the file contents (post-swap,
    big-endian) the words must be read big-endian, hence ``is_big_endian``.
    """
    flat = np.ascontiguousarray(arr).reshape(-1)
    if is_big_endian:
        flat = flat.astype(np.dtype(">c16")).view(np.dtype(">u4"))
    else:
        flat = flat.view(np.uint32)
    return int(np.sum(flat.astype(np.uint64)) & 0xFFFFFFFF)

def _make_gauge_field_header(total_site, link_trace, plaq, checksum, crc32, beta):
    """Return the NERSC ``4D_SU3_GAUGE`` header text (helper)."""
    import time
    #
    lines = [
        "BEGIN_HEADER",
        "HDR_VERSION = 1.0",
        "DATATYPE = 4D_SU3_GAUGE",
        f"DIMENSION_1 = {total_site[0]}",
        f"DIMENSION_2 = {total_site[1]}",
        f"DIMENSION_3 = {total_site[2]}",
        f"DIMENSION_4 = {total_site[3]}",
        f"LINK_TRACE = {link_trace:.12f}",
        f"PLAQUETTE = {plaq:.12f}",
        f"CHECKSUM = {checksum:08x}",
        f"CRC32HASH = {crc32:08x}",
        "CREATOR = Qlat",
        f"ARCHIVE_DATE = {time.ctime()}",
        "ENSEMBLE_ID = 42",
        "ENSEMBLE_LABEL = default-ensemble",
        f"BETA = {beta:.12f}",
        "SEQUENCE_NUMBER = 0",
        "FLOATING_POINT = IEEE64BIG",
        "END_HEADER",
    ]
    return "\n".join(lines) + "\n"

def _read_gauge_field_header(path):
    """Return the parsed NERSC header of a ``4D_SU3_GAUGE`` file (helper)."""
    with open(path, "rb") as f:
        raw = f.read(4096)
    if not raw.startswith(b"BEGIN_HEADER\n"):
        raise IOError(f"qlat_ref: not a NERSC gauge field file: '{path}'")
    end = raw.find(b"END_HEADER\n")
    if end < 0:
        raise IOError(f"qlat_ref: truncated NERSC header in '{path}'")
    header_text = raw[: end + len(b"END_HEADER\n")].decode("ascii")
    header = {}
    for line in header_text.splitlines():
        if " = " in line:
            key, val = line.split(" = ", 1)
            header[key.strip()] = val.strip()
    n_bytes = end + len(b"END_HEADER\n")
    return header, n_bytes

def _save_gauge_field_nersc(gf, path):
    """
    Write ``gf`` in the NERSC ``4D_SU3_GAUGE`` format (internal helper).\n
    The layout follows qlat's ``save_gauge_field`` exactly: only the first two
    rows of every link are stored (6 complex numbers per link, big-endian),
    because the third row is rebuilt by ``unitarize`` on load.  The header
    records ``LINK_TRACE``, ``PLAQUETTE``, ``CHECKSUM`` (sum of the native
    endian 32-bit words, computed **before** the endianness swap) and
    ``CRC32HASH`` (CRC32 of the big-endian data).
    """
    geo = gf.geo
    arr = get_array_from_field(gf)
    # (4, Nt, Nx, Ny, Nz, 3, 3) -> (local_volume, 4, 6) in flat-site order
    flat = np.moveaxis(arr, 0, -3).reshape((geo.local_volume, 4, NC, NC), order="F")
    data = np.empty((geo.local_volume, 4, 6), dtype=np.complex128)
    data[..., 0:3] = flat[..., 0, :]
    data[..., 3:6] = flat[..., 1, :]
    checksum = _get_simple_checksum(data)
    plaq = _get_avg_plaq(arr)
    link_trace = _get_avg_link_trace(arr)
    data_big_endian = data.astype(data.dtype.newbyteorder(">"), copy=True)
    crc32 = _get_crc32(np.ascontiguousarray(data_big_endian).tobytes())
    header = _make_gauge_field_header(
        geo.total_site.to_tuple(), link_trace, plaq, checksum, crc32, 0.0
    )
    tmp_path = path + ".partial"
    with open(tmp_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data_big_endian).tobytes())
    import os
    #
    os.replace(tmp_path, path)
    return os.path.getsize(path)

def _load_gauge_field_nersc(gf, path):
    """
    Read a NERSC ``4D_SU3_GAUGE`` file into ``gf`` (internal helper).\n
    The third row of every link is **rebuilt** with the Gram-Schmidt
    ``unitarize``, exactly as qlat's ``load_gauge_field`` does, so the format
    is lossy for non-unitary input; a ``CHECKSUM``/``CRC32HASH`` mismatch
    raises with qlat's warning text.\n
    .. note:: qlat only *warns* on a checksum mismatch and keeps going;
       ``qlat_ref`` raises ``IOError``, which is the behavior a reference
       implementation should have when the data is corrupt.
    """
    import os
    #
    if not os.path.isfile(path):
        raise IOError(f"qlat_ref: gauge field file does not exist: '{path}'")
    header, offset = _read_gauge_field_header(path)
    total_site = tuple(int(header[f"DIMENSION_{i + 1}"]) for i in range(4))
    geo = Geometry(Coordinate(total_site))
    if gf.geo is None:
        gf.init_from_geo(geo, 4)
    else:
        gf.init_from_geo(geo, 4)
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read()
    n_bytes = geo.local_volume * 4 * 6 * 16
    if len(raw) != n_bytes:
        raise IOError(
            f"qlat_ref: gauge field file '{path}' has {len(raw)} data bytes,"
            f" expected {n_bytes}"
        )
    data_big_endian = np.frombuffer(raw, dtype=np.dtype(">c16")).reshape(
        geo.local_volume, 4, 6
    )
    checksum = _get_simple_checksum(data_big_endian, is_big_endian=True)
    crc32 = _get_crc32(raw)
    if "CHECKSUM" in header:
        if int(header["CHECKSUM"], 16) != checksum:
            raise IOError(
                f"qlat_ref: '{path}': CHECKSUM mismatch"
                f" (header {header['CHECKSUM']}, data {checksum:08x});"
                " the file is corrupt"
            )
    if "CRC32HASH" in header:
        if int(header["CRC32HASH"], 16) != crc32:
            raise IOError(
                f"qlat_ref: '{path}': CRC32HASH mismatch"
                f" (header {header['CRC32HASH']}, data {crc32:08x});"
                " the file is corrupt"
            )
    data = data_big_endian.astype(np.complex128)
    flat = np.zeros((geo.local_volume, 4, NC, NC), dtype=np.complex128)
    flat[..., 0, :] = data[..., 0:3]
    flat[..., 1, :] = data[..., 3:6]
    flat[...] = get_unitarized(flat)
    gf._arr[...] = flat
    return n_bytes

def _make_field_header(total_site, multiplicity, sizeof_m, crc32):
    """Return the generic ``BEGIN_FIELD_HEADER`` text of qlat (helper)."""
    lines = [
        "BEGIN_FIELD_HEADER",
        "field_version = 1.0",
        f"total_site[0] = {total_site[0]}",
        f"total_site[1] = {total_site[1]}",
        f"total_site[2] = {total_site[2]}",
        f"total_site[3] = {total_site[3]}",
        f"multiplicity = {multiplicity}",
        f"sizeof(M) = {sizeof_m}",
        f"field_crc32 = {crc32:08X}",
        "END_HEADER",
    ]
    return "\n".join(lines) + "\n"

def _read_field_header(path):
    """Return the parsed generic (``BEGIN_FIELD_HEADER``) header (helper)."""
    with open(path, "rb") as f:
        raw = f.read(4096)
    if not raw.startswith(b"BEGIN_FIELD_HEADER\n"):
        raise IOError(f"qlat_ref: not a generic field file: '{path}'")
    end = raw.find(b"END_HEADER\n")
    if end < 0:
        raise IOError(f"qlat_ref: truncated field header in '{path}'")
    header_text = raw[: end + len(b"END_HEADER\n")].decode("ascii")
    header = {}
    for line in header_text.splitlines():
        if " = " in line:
            key, val = line.split(" = ", 1)
            header[key.strip()] = val.strip()
    return header, end + len(b"END_HEADER\n")

def _save_double_field_nersc(f, path):
    """
    Write a generic double-precision field (internal helper).\n
    This is qlat's generic ``Field.save_double`` format used by
    ``GaugeTransform.save``: a ``BEGIN_FIELD_HEADER`` text header (geometry,
    multiplicity, element size and the CRC32 of the big-endian payload)
    followed by the big-endian complex128 payload with no per-element
    padding.
    """
    arr = np.asarray(f)
    total_site = f.geo.total_site.to_tuple() if f.geo is not None else (0, 0, 0, 0)
    data_big_endian = arr.astype(arr.dtype.newbyteorder(">"), copy=True)
    payload = np.ascontiguousarray(data_big_endian).tobytes()
    crc32 = _get_crc32(payload)
    header = _make_field_header(total_site, f.multiplicity, f.sizeof_m, crc32)
    tmp_path = path + ".partial"
    with open(tmp_path, "wb") as out:
        out.write(header.encode("ascii"))
        out.write(payload)
    import os
    #
    os.replace(tmp_path, path)
    return os.path.getsize(path)

def _load_double_field_nersc(f, path):
    """
    Read the generic double-precision field format (internal helper).\n
    The header's ``field_crc32`` is verified against the payload; qlat only
    warns on a mismatch, ``qlat_ref`` raises ``IOError``.
    """
    import os
    #
    if not os.path.isfile(path):
        raise IOError(f"qlat_ref: field file does not exist: '{path}'")
    header, offset = _read_field_header(path)
    with open(path, "rb") as fp:
        raw = fp.read()
    payload = raw[offset:]
    if "field_crc32" in header:
        if int(header["field_crc32"], 16) != _get_crc32(payload):
            raise IOError(
                f"qlat_ref: '{path}': field_crc32 mismatch; the file is corrupt"
            )
    total_site = tuple(int(header[f"total_site[{i}]"]) for i in range(4))
    multiplicity = int(header["multiplicity"])
    geo = Geometry(Coordinate(total_site))
    if f.geo is None:
        f.init_from_geo(geo, multiplicity)
    else:
        f.init_from_geo(geo, multiplicity)
    arr = np.frombuffer(payload, dtype=np.dtype(">c16")).astype(np.complex128)
    f._arr[...] = arr.reshape(np.asarray(f).shape)
    return len(payload)
