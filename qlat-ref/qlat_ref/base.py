"""
Module ``qlat_ref.base``
========================\n
Shared conventions of the reference package: the colour dimension ``NC``, the
link-direction sets with the spatial-direction convention, the array shape
helpers, and the two qlat value types :class:`Coordinate` and
:class:`Geometry`.\n
Documentation: ``docs/qlat-ref/qlat_types.md``\n
.. note:: Update the documentation when updating this source file.\n
Conventions
-----------\n
``NC = 3`` is the number of colours.  A link field has shape::\n
    U.shape == (4, Nt, Nx, Ny, Nz, NC, NC)\n
so axis ``0`` is the link direction ``mu`` (``0 = t``, ``1 = x``, ``2 = y``,
``3 = z``) and the colour matrix of direction ``mu`` is ``U[mu]`` of shape
``(Nt, Nx, Ny, Nz, NC, NC)``.  A site field drops the leading direction axis.
The lattice is periodic, so a hop by one site in direction ``mu`` is
``numpy.roll(..., 1, axis=mu)`` for a site field.\n
.. note:: :class:`Geometry` and :class:`Coordinate` mirror the qlat API
   (``qlat.geometry.Geometry`` and ``qlat_utils.coordinate.Coordinate``),
   including the names of the properties and methods.  ``qlat_ref`` is a
   single-process reference implementation, so the MPI-node quantities are the
   global ones: ``node_site == total_site``, ``coor_node == 0``,
   ``id_node == 0`` and ``num_node == 1`` for a non-degenerate geometry.
"""

import numpy as np

### -------------------------------------------------------------------

NC = 3

_ALL_DIRS = (0, 1, 2, 3)
# The spatial directions.  Component 3 is time in qlat's convention
# (`Geometry::total_site` = (Lx, Ly, Lz, Lt); `Geometry::spatial_volume` divides
# out component 3 and `get_psel_tslice` slices along `t_dir=3`), so the spatial
# directions are 0, 1, 2.  qlat's `gf_fix_coulomb_gauge` instead uses
# {1, 2, 3} (`qlat/qlat/lib/gauge-fixing.cpp`), which is qlat-issue-8; the
# reference deliberately uses the correct set.  See the note in `get_dirs`.
_SPATIAL_DIRS = (0, 1, 2)

### -------------------------------------------------------------------

def get_total_site(shape):
    """Return ``total_site`` from the shape of a link, site or momentum field."""
    shape = tuple(shape)
    if len(shape) == 6:
        return shape
    if len(shape) == 7:
        return shape[1:]
    raise ValueError(f"qlat_ref: not a gauge field shape: {shape}")

def get_dirs(dirs=None):
    """
    Return the tuple of link directions to include in the gauge functional.\n
    ``dirs=None`` means all four directions ``(0, 1, 2, 3)`` (Landau gauge);
    ``dirs=(0, 1, 2)`` means the spatial directions only (Coulomb gauge).\n
    .. note:: The spatial set is ``(0, 1, 2)`` because component 3 is **time**
       in qlat's convention: ``Geometry::total_site`` is ``(Lx, Ly, Lz, Lt)``,
       ``Geometry::spatial_volume`` divides out component 3, and
       ``get_psel_tslice`` slices along ``t_dir=3``.  qlat's
       ``gf_fix_coulomb_gauge`` uses ``{1, 2, 3}`` instead, i.e. it fixes the
       three directions that *include* time and excludes a spatial one; that
       is qlat-issue-8 in ``tmp/qlat-ref-reorg/qlat-issues.md``.  The
       reference deliberately uses the correct set and the deviation from
       ``q.fix_coulomb_gauge`` is intentional, documented and tested.
    """
    if dirs is None:
        return _ALL_DIRS
    dirs = tuple(int(mu) for mu in dirs)
    if len(dirs) == 0:
        raise ValueError("qlat_ref: dirs must not be empty")
    if len(set(dirs)) != len(dirs):
        raise ValueError(f"qlat_ref: dirs must not repeat: {dirs}")
    if not (0 <= min(dirs) and max(dirs) < 4):
        raise ValueError(f"qlat_ref: dirs must be link directions 0..3: {dirs}")
    return dirs

### -------------------------------------------------------------------

def _coordinate_components(x):
    """Return the 4 integer components of a coordinate-like value (helper)."""
    if isinstance(x, Coordinate):
        return x.to_tuple()
    if isinstance(x, np.ndarray):
        if x.shape != (4,):
            raise ValueError(f"qlat_ref: not a coordinate array: shape={x.shape}")
        return tuple(int(v) for v in x)
    if isinstance(x, (list, tuple)):
        if len(x) != 4:
            raise ValueError(f"qlat_ref: not a coordinate: len={len(x)}")
        return tuple(int(v) for v in x)
    raise TypeError(f"qlat_ref: not a coordinate: {type(x).__name__}")

def _get_key_index(key, dim=4):
    """Return the normalized component index for a coordinate (helper)."""
    if not hasattr(key, "__index__"):
        raise TypeError(
            f"coordinate indices must be integers, not {type(key).__name__}"
        )
    i = key.__index__()
    if i < 0:
        i += dim
    if i < 0 or dim <= i:
        raise IndexError(
            f"coordinate index {key} is out of range (coordinate has {dim} components)"
        )
    return i

### -------------------------------------------------------------------

class Coordinate:
    """
    A 4-component integer lattice coordinate.\n
    Mirrors ``qlat_utils.coordinate.Coordinate``: element-wise arithmetic
    (``+``, ``-``, ``*`` by an integer or element-wise, ``%``, ``//``, unary
    ``-``/``+``), sized-sequence behaviour (``len(c) == 4``, indexing with
    Python-style negative indices, iteration, ``numpy`` conversion), hashing
    consistent with ``==``, and index <-> coordinate conversion.\n
    .. note:: A ``Coordinate`` is always truthy, even ``Coordinate()``, so that
       ``if c:`` does not silently mean "non-zero coordinate".  Test for the
       origin explicitly, e.g. ``c == Coordinate()``.
    """

    __slots__ = ("_x",)

    def __init__(self, x=None):
        if x is None:
            self._x = (0, 0, 0, 0)
        else:
            self._x = _coordinate_components(x)

    def __imatmul__(self, v1):
        self._x = _coordinate_components(v1)
        return self

    def copy(self, is_copying_data=True):
        """Return a copy of the coordinate (``is_copying_data=False``: zeros)."""
        if is_copying_data:
            return Coordinate(self._x)
        return Coordinate()

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __repr__(self):
        return f"Coordinate({self.to_list()})"

    def __iter__(self):
        return iter(self._x)

    def to_list(self):
        """Return a list composed of the 4 components of the coordinate."""
        return list(self._x)

    def to_tuple(self):
        """Return a tuple composed of the 4 components of the coordinate."""
        return tuple(self._x)

    def to_numpy(self):
        """Return a ``numpy`` int32 array of the 4 components of the coordinate."""
        return np.asarray(self._x, dtype=np.int32)

    def from_list(self, x):
        """Set the value from a list/tuple/array of the 4 components."""
        self._x = _coordinate_components(x)

    def sqr(self):
        """Return the square sum of all the components as ``int``."""
        return sum(v * v for v in self._x)

    def r_sqr(self):
        """Return the spatial distance square ``x_1^2 + x_2^2 + x_3^2`` as ``int``."""
        return self._x[0] ** 2 + self._x[1] ** 2 + self._x[2] ** 2

    def volume(self):
        """Return the product of all the components."""
        v = 1
        for c in self._x:
            v *= c
        return v

    def spatial_volume(self):
        """
        Return the product of the first three components ``x_0 * x_1 * x_2``.\n
        .. note:: This mirrors qlat exactly (``qlat-utils/.../coordinate.h``),
           where the name is a misnomer: the product drops the *last*
           component, which is not the time component under qlat's own
           convention (axis ``3`` is time, see :func:`qlat_ref.base.get_dirs`
           and qlat-issue-8 in ``tmp/qlat-ref-reorg/qlat-issues.md``).
        """
        return self._x[0] * self._x[1] * self._x[2]

    def __getitem__(self, key):
        """Get component ``key`` (supports Python-style negative indexing)."""
        return self._x[_get_key_index(key)]

    def __setitem__(self, key, val):
        """Set component ``key`` (supports Python-style negative indexing)."""
        i = _get_key_index(key)
        x = list(self._x)
        x[i] = int(val)
        self._x = tuple(x)

    def __len__(self):
        """Return the number of components (always 4)."""
        return 4

    def __bool__(self):
        """A ``Coordinate`` is always truthy (see the class docstring)."""
        return True

    def __eq__(self, other):
        return isinstance(other, Coordinate) and self._x == other._x

    def __hash__(self):
        return hash(self._x)

    def __add__(self, c2):
        b = _coordinate_components(c2)
        return Coordinate([a + b_ for a, b_ in zip(self._x, b)])

    def __sub__(self, c2):
        b = _coordinate_components(c2)
        return Coordinate([a - b_ for a, b_ in zip(self._x, b)])

    def __mul__(self, c2):
        if isinstance(c2, int):
            return Coordinate([c * c2 for c in self._x])
        b = _coordinate_components(c2)
        return Coordinate([a * b_ for a, b_ in zip(self._x, b)])

    def __rmul__(self, c2):
        return self * c2

    def __mod__(self, c2):
        b = _coordinate_components(c2)
        return Coordinate([a % b_ for a, b_ in zip(self._x, b)])

    def __floordiv__(self, c2):
        b = _coordinate_components(c2)
        return Coordinate([a // b_ for a, b_ in zip(self._x, b)])

    def __neg__(self):
        return Coordinate([-c for c in self._x])

    def __pos__(self):
        return self

    def from_index(self, index, size):
        """Set the value from the flat index ``index`` in a lattice of ``size``."""
        size = _coordinate_components(size)
        self._x = _coordinate_from_index(int(index), size)

    def to_index(self, size):
        """Return the flat index of this coordinate in a lattice of ``size``."""
        return _index_from_coordinate(self._x, _coordinate_components(size))

### -------------------------------------------------------------------

def _coordinate_from_index(index, size):
    """Flat site index -> coordinate, first coordinate fastest (helper)."""
    index = int(index)
    x = []
    for mu in range(4):
        x.append(index % size[mu])
        index //= size[mu]
    return tuple(x)

def _index_from_coordinate(x, size):
    """Coordinate -> flat site index, first coordinate fastest (helper)."""
    index = 0
    for mu in range(3, -1, -1):
        index = index * size[mu] + int(x[mu])
    return index

def _get_volume(x):
    """Return the product of the 4 components of a coordinate-like value (helper)."""
    v = 1
    for c in _coordinate_components(x):
        v *= c
    return v

### -------------------------------------------------------------------

class Geometry:
    """
    The lattice geometry: total and local lattice size, node decomposition,
    halo expansion and even/odd partitioning.\n
    Mirrors ``qlat.geometry.Geometry``.  The constructor forms are::\n
        Geometry()                                    # all zero, uninitialized
        Geometry(total_site)                          # one MPI node: local == total
        Geometry(id_node, size_node, node_site)
        Geometry(coor_node, size_node, node_site)\n
    where every lattice-size argument is a :class:`Coordinate` (like qlat, a
    plain list is **not** accepted for the ``1``- and ``3``-argument forms).\n
    .. note:: Single-process reference implementation: there is no MPI, so the
       quantities that qlat splits across nodes are the global ones.
       ``total_site``, ``node_site``, ``local_site`` and the volumes are all
       consistent with each other for any ``Geometry(...)`` with a non-zero
       ``total_site``; ``id_node`` is ``0`` and ``num_node`` is
       ``1`` (``0`` for the all-zero ``Geometry()``, like qlat).
    """

    __slots__ = (
        "_total_site",
        "_node_site",
        "_expansion_left",
        "_expansion_right",
        "_eo",
        "_initialized",
    )

    def __init__(self, *args):
        n_args = len(args)
        if n_args == 0:
            self._init(Coordinate(), Coordinate(), 0)
            # qlat's bare Geometry() stays uninitialized (and reports
            # is_only_local == False), unlike a geometry built from a lattice.
            self._initialized = False
        elif n_args == 1:
            if not isinstance(args[0], Coordinate):
                raise Exception(f"qlat_ref: Geometry: args={args}")
            self._init(args[0], args[0], 0)
        elif n_args == 3:
            size_node = _coordinate_components(args[1])
            node_site = _coordinate_components(args[2])
            if isinstance(args[0], int):
                # id_node form: the node coordinate is derived from the index.
                if not isinstance(args[1], Coordinate) or not isinstance(
                    args[2], Coordinate
                ):
                    raise Exception(f"qlat_ref: Geometry: args={args}")
            elif isinstance(args[0], Coordinate):
                if not isinstance(args[1], Coordinate) or not isinstance(
                    args[2], Coordinate
                ):
                    raise Exception(f"qlat_ref: Geometry: args={args}")
            else:
                raise Exception(f"qlat_ref: Geometry: args={args}")
            # total_site = size_node * node_site
            total_site = Coordinate([size_node[mu] * node_site[mu] for mu in range(4)])
            self._init(total_site, node_site, 0)
        else:
            raise Exception(f"qlat_ref: Geometry: args={args}")

    def _init(self, total_site, node_site, eo):
        """Initialize from a total size, a local size and an even/odd flag (internal)."""
        self._total_site = Coordinate(total_site)
        self._node_site = Coordinate(node_site)
        self._expansion_left = Coordinate()
        self._expansion_right = Coordinate()
        self._eo = int(eo)
        self._initialized = True

    def __imatmul__(self, v1):
        self._total_site = v1.total_site
        self._node_site = v1.local_site
        self._expansion_left = v1.expansion_left
        self._expansion_right = v1.expansion_right
        self._eo = v1.eo
        self._initialized = v1._initialized
        return self

    def copy(self, is_copying_data=True):
        """Return a copy of the geometry (``is_copying_data=False``: all zero)."""
        if not is_copying_data:
            return Geometry()
        x = Geometry()
        x @= self
        return x

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __eq__(self, other):
        return (
            isinstance(other, Geometry)
            and self._total_site == other._total_site
            and self._node_site == other._node_site
            and self._expansion_left == other._expansion_left
            and self._expansion_right == other._expansion_right
            and self._eo == other._eo
            and self._initialized == other._initialized
        )

    def __str__(self):
        return self.show()

    def __repr__(self):
        return self.show()

    def show(self):
        """Return the one-line representation, e.g. ``Geometry([2, 2, 3, 4])``."""
        zero = Coordinate()
        if (
            self._expansion_left == zero
            and self._expansion_right == zero
            and self._eo == 0
        ):
            return f"Geometry({self.total_site.to_list()})"
        return (
            f"Geometry({self.total_site.to_list()},"
            f" expan_left={self.expansion_left.to_list()},"
            f" expan_right={self.expansion_right.to_list()}, eo={self.eo})"
        )

    @property
    def total_site(self):
        """The global lattice size as a :class:`Coordinate`."""
        return self._total_site.copy()

    @property
    def total_volume(self):
        """The global number of sites."""
        return self._volume_of(self._total_site)

    @property
    def spatial_volume(self):
        """
        The global number of sites per time slice,
        ``total_volume / total_site[3]``.\n
        .. note:: Mirrors qlat's ``Geometry::spatial_volume``, which divides out
           component ``3`` (time under qlat's convention).  A degenerate
           ``Geometry()`` has ``total_site == [0, 0, 0, 0]`` and therefore an
           empty-but-safe ``total_volume == 0``, so this returns ``0``; qlat
           used to die there with SIGFPE (integer divide-by-zero) but now has
           the same empty-but-safe guard (qlat-issue-9 in
           ``tmp/qlat-ref-reorg/qlat-issues.md``).
        """
        if self._total_site[3] == 0:
            return 0
        return self.total_volume // self._total_site[3]

    @property
    def local_site(self):
        """The locally owned lattice size as a :class:`Coordinate`."""
        return self._node_site.copy()

    @property
    def local_volume(self):
        """The number of locally owned sites."""
        return self._volume_of(self._node_site)

    @property
    def local_volume_expanded(self):
        """The number of locally owned sites including the halo expansion."""
        return self._volume_of(self._node_site_expanded())

    @property
    def eo(self):
        """The even/odd partition flag (``0`` for an unpartitioned geometry)."""
        return self._eo

    @property
    def expansion_left(self):
        """The halo expansion on the low side as a :class:`Coordinate`."""
        return self._expansion_left.copy()

    @property
    def expansion_right(self):
        """The halo expansion on the high side as a :class:`Coordinate`."""
        return self._expansion_right.copy()

    @property
    def id_node(self):
        """The MPI node index (always ``0``: this package is single-process)."""
        return 0

    @property
    def num_node(self):
        """
        The number of MPI nodes: ``product(size_node)``, i.e. ``1`` for a
        single-process geometry and ``0`` for the degenerate ``Geometry()``.
        """
        return _get_volume(self.size_node)

    @property
    def coor_node(self):
        """The MPI node coordinate (always the origin for a single process)."""
        return Coordinate()

    @property
    def size_node(self):
        """
        The MPI node-grid size (``1x1x1x1`` for a single process).\n
        .. note:: ``[0, 0, 0, 0]`` for the uninitialized ``Geometry()``, whose
           global geometry node is not known yet — matching qlat's bare
           ``Geometry()`` (qlat-issue-9).  This is also what makes
           ``num_node``, ``_mirror`` and ``is_local`` degenerate the same way
           as in qlat for the empty geometry.
        """
        if not self._initialized:
            return Coordinate()
        return Coordinate([1, 1, 1, 1])

    @property
    def node_site(self):
        """The site size of one MPI node (equal to ``total_site`` in one process)."""
        return self._node_site.copy()

    @property
    def is_only_local(self):
        """
        Whether the local volume covers the whole lattice (no halo).\n
        .. note:: ``False`` for the uninitialized ``Geometry()``, matching
           qlat's bare ``Geometry()``.
        """
        zero = Coordinate()
        return (
            self._initialized
            and self._expansion_left == zero
            and self._expansion_right == zero
        )

    def _node_site_expanded(self):
        """Return the local size including the halo expansion (helper)."""
        return Coordinate(
            [
                self._expansion_left[mu]
                + self._node_site[mu]
                + self._expansion_right[mu]
                for mu in range(4)
            ]
        )

    @staticmethod
    def _volume_of(x):
        """Return the product of the 4 components of a coordinate (helper)."""
        return _get_volume(x)

    def coordinate_g_from_l(self, xl):
        """Local coordinate -> global coordinate."""
        coalesced = _coordinate_components(xl)
        coor_node = self.coor_node.to_tuple()
        node_site = self._node_site.to_tuple()
        return Coordinate(
            [coalesced[mu] + coor_node[mu] * node_site[mu] for mu in range(4)]
        )

    def coordinate_l_from_g(self, xg):
        """Global coordinate -> local coordinate."""
        coalesced = _coordinate_components(xg)
        coor_node = self.coor_node.to_tuple()
        node_site = self._node_site.to_tuple()
        return Coordinate(
            [coalesced[mu] - coor_node[mu] * node_site[mu] for mu in range(4)]
        )

    def index_from_coordinate(self, xl):
        """
        Local coordinate -> flat local site index.\n
        .. note:: Mirrors qlat's ``Geometry::index_from_coordinate``, which
           first applies ``mirror``: along an axis with ``size_node[mu] == 1``
           the coordinate is reduced modulo ``node_site[mu]``, so an
           out-of-range coordinate wraps around instead of raising.  In this
           single-process package every axis has ``size_node[mu] == 1``, so
           every coordinate is wrapped.
        """
        x = self._mirror(_coordinate_components(xl))
        return _index_from_coordinate(x, self._node_site.to_tuple())

    def _mirror(self, x):
        """Reduce ``x`` modulo ``node_site`` on axes owned by a single node (helper)."""
        node_site = self._node_site.to_tuple()
        size_node = self.size_node.to_tuple()
        return tuple(
            x[mu] % node_site[mu]
            if size_node[mu] == 1 and node_site[mu] != 0
            else x[mu]
            for mu in range(4)
        )

    def coordinate_from_index(self, index):
        """
        Flat local site index -> local coordinate.\n
        .. note:: An empty geometry (zero extent) owns no site, so this returns
           the zero coordinate instead of dividing by a zero extent — the same
           empty-but-safe contract as qlat (qlat-issue-9).
        """
        node_site = self._node_site.to_tuple()
        if _get_volume(node_site) == 0:
            return Coordinate()
        return Coordinate(_coordinate_from_index(int(index), node_site))

    def index_from_g_coordinate(self, xg):
        """Global coordinate -> flat local site index."""
        return self.index_from_coordinate(self.coordinate_l_from_g(xg))

    def g_coordinate_from_index(self, index):
        """Flat local site index -> global coordinate."""
        return self.coordinate_g_from_l(self.coordinate_from_index(index))

    def is_local(self, xl):
        """
        Return whether ``xl`` is a locally owned coordinate.\n
        .. note:: Mirrors qlat's ``Geometry::is_local``, which also returns
           ``True`` along any axis with ``size_node[mu] == 1`` (there is no
           other node to own it).  Since this package is single-process, every
           axis has ``size_node[mu] == 1`` and therefore ``is_local`` is always
           ``True``, exactly as in qlat run on one node.
        """
        x = _coordinate_components(xl)
        node_site = self._node_site.to_tuple()
        size_node = self.size_node.to_tuple()
        return all(
            (0 <= x[mu] < node_site[mu]) or size_node[mu] == 1 for mu in range(4)
        )

    def is_local_xg(self, xg):
        """Return whether the global coordinate ``xg`` is inside the local volume."""
        return self.is_local(self.coordinate_l_from_g(xg))

    def xg_arr(self):
        """
        Return the global coordinates of all local sites.\n
        Shape ``(local_volume, 4)`` with the flat local site index on axis
        ``0``, matching ``qlat.geometry.Geometry.xg_arr``.
        """
        local_volume = self.local_volume
        arr = np.zeros((local_volume, 4), dtype=np.int32)
        node_site = self._node_site.to_tuple()
        coor_node = self.coor_node.to_tuple()
        for index in range(local_volume):
            xl = _coordinate_from_index(index, node_site)
            for mu in range(4):
                arr[index, mu] = xl[mu] + coor_node[mu] * node_site[mu]
        return arr

    def __getstate__(self):
        return [
            self.size_node,
            self.coor_node,
            self.node_site,
            self.expansion_left,
            self.expansion_right,
        ]

    def __setstate__(self, state):
        self.__init__()
        self._initialized = True
        size_node, coor_node, node_site, expan_left, expan_right = state
        total_site = Coordinate(
            [
                _coordinate_components(size_node)[mu]
                * _coordinate_components(node_site)[mu]
                for mu in range(4)
            ]
        )
        self._init(total_site, node_site, 0)
        self._expansion_left = Coordinate(expan_left)
        self._expansion_right = Coordinate(expan_right)
