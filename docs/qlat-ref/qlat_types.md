# `qlat_ref` types — `Coordinate`, `Geometry`, `ColorMatrix`, fields (NumPy Reference)

Source: `qlat-ref/qlat_ref/base.py`, `qlat-ref/qlat_ref/mat.py`,
`qlat-ref/qlat_ref/field.py`

> **Note:** Update this document when updating the source files.

## Outline

1. [Overview](#overview)
2. [`NC` and the Link-Direction Convention](#nc-and-the-link-direction-convention)
3. [`Coordinate`](#coordinate)
4. [`Geometry`](#geometry)
5. [`ColorMatrix` and the Matrix Helpers](#colormatrix-and-the-matrix-helpers)
6. [`FieldColorMatrix`](#fieldcolormatrix)
7. [`GaugeField`, `GaugeTransform`, `GaugeMomentum`](#gaugefield-gaugetransform-gaugemomentum)
8. [The Array Bridge](#the-array-bridge)
9. [Deviation: No MPI](#deviation-no-mpi)
10. [Self-Checks](#self-checks)

---

## Overview

`qlat_ref` re-implements qlat's value types (`Coordinate`, `Geometry`, the
`ColorMatrix` and the four field types) with **qlat's names, properties and
methods**.  These are the types the rest of the package is written against, so
the gauge fixing, the observables and the HMC are all expressed on them.

Everything here is a thin wrapper around a plain NumPy array; the wrapper
exists so that the numerical code can be ported line by line against qlat
without every call site having to know about buffer layout.

## `NC` and the Link-Direction Convention

`NC = 3` is the number of colours.  A **link field** has shape

```
U.shape == (4, Nt, Nx, Ny, Nz, NC, NC)
```

so axis `0` is the link direction `mu` (`0 = t`, `1 = x`, `2 = y`, `3 = z`)
and the colour matrix of direction `mu` is `U[mu]` of shape
`(Nt, Nx, Ny, Nz, NC, NC)`.  A **site field** drops the leading direction
axis.  The lattice is periodic, so a hop by one site in direction `mu` is
`numpy.roll(..., 1, axis=mu)` on a site field and `numpy.roll(..., 1,
axis=mu)` on `U[mu]` (because the direction axis is dropped).

## `Coordinate`

`Coordinate` is an integer 4-vector used both as a lattice size and as a
lattice coordinate.  It mirrors `qlat_utils.Coordinate`:

| Member | Meaning |
|---|---|
| `Coordinate(x=None)` | all-zero, or from a length-4 list/tuple/array |
| `to_list()`, `to_tuple()`, `to_numpy()` | the four components |
| `from_list(x)`, `from_index(index, size)`, `to_index(size)` | construction and the flat-index mapping |
| `sqr()`, `r_sqr()`, `volume()`, `spatial_volume()` | the volume products (`spatial_volume` divides out component 3, the time axis) |
| `__getitem__`, `__setitem__`, `__len__`, `__iter__` | component access |
| `__add__`, `__sub__`, `__mul__`, `__rmul__`, `__mod__`, `__floordiv__`, `__neg__` | component-wise arithmetic |
| `copy()`, `__eq__`, `__hash__` | value semantics |

## `Geometry`

`Geometry` mirrors `qlat.geometry.Geometry`: the total and local lattice size,
the node decomposition, the halo expansion and the even/odd partitioning.  The
constructor forms are

    Geometry()                                   # all zero, uninitialized
    Geometry(total_site)                         # one MPI node: local == total
    Geometry(id_node, size_node, node_site)
    Geometry(coor_node, size_node, node_site)

where every lattice size is a `Coordinate` (as in qlat, a plain list is not
accepted by the `1`- and `3`-argument forms).

The geometry members are the ones qlat's numerical code reads:

| Group | Members |
|---|---|
| lattice sizes | `total_site`, `node_site`, `local_site`, `total_volume`, `spatial_volume`, `local_volume`, `local_volume_expanded` |
| node decomposition | `id_node`, `num_node`, `coor_node`, `size_node`, `eo`, `is_only_local` |
| halo | `expansion_left`, `expansion_right` |
| index arithmetic | `coordinate_from_index`, `index_from_coordinate`, `coordinate_g_from_l`, `coordinate_l_from_g`, `g_coordinate_from_index`, `index_from_g_coordinate` |
| locality | `is_local`, `is_local_xg`, `xg_arr` |
| value semantics | `copy`, `show`, `__str__`, `__eq__`, `__imatmul__` |

`xg_arr()` returns the `(local_volume, 4)` array of the locally owned global
coordinates, which is what the array bridge below uses to scatter a global
array into a field.

## `ColorMatrix` and the Matrix Helpers

`ColorMatrix` is a `3 x 3` complex matrix with qlat's operations:

- `copy`, `set_zero`, `qnorm`;
- `conjugate`, `transpose`, `adjoint`;
- `__getitem__`, `__setitem__`, `__iadd__`, `__isub__`, `__imul__`,
  `__mul__`, `__rmul__`, `__eq__`.

The module-level helpers in `qlat_ref.mat` are the routines the reference
formulas are built from.  They all work on a plain NumPy array whose **last two
axes are the colour axes**, so they apply to a single matrix, a site field and
a link field alike:

| Helper | Meaning |
|---|---|
| `get_matrix_adjoint(a)` | `a^dagger`, colour axes only |
| `get_traceless_antihermitian_part(a)` | `(a - a^dagger)/2 - Tr((a - a^dagger)/2)/3` |
| `get_su3_exp(h, *, max_order=20)` | truncated Taylor (Horner) matrix exponential |
| `get_unitarized(m)` | Gram-Schmidt projection onto `SU(3)` |
| `get_unitarity_error(gf)` | `max |U U^dagger - 1|` |
| `check_unitarity(gf, *, eps=1e-12)` | `get_unitarity_error(gf) <= eps` |
| `get_random_su3_field(shape, *, seed=0, scale=1.0)` | deterministic random `SU(3)` field |

`get_su3_exp` is the matrix exponential of the gauge fixing
(`qlat_ref.gauge_fixing`) and of `gf_evolve` (`qlat_ref.hmc`), and
`get_unitarized` is the final projection of both; they reproduce qlat's
`make_matrix_exp` and Gram-Schmidt step.  `get_unitarity_error` accepts a
`ColorMatrix`, a link field, a site field or a single matrix — a documented
extra over qlat's typed entry point (see
[`qlat_ref` overview](qlat_ref_overview.md)).

The last section of this page lists the exact deviations.

## `FieldColorMatrix`

`FieldColorMatrix` is a **site field**: one `NC x NC` matrix per lattice site.
It is the base class of `GaugeTransform` (multiplicity `1`).

    FieldColorMatrix(geo=None, multiplicity=1)

Its buffer is the C-contiguous

```
(local_volume, multiplicity, NC, NC)        dtype = complex128
```

with the **flat local site index** on axis 0, and the members are:

| Member | Meaning |
|---|---|
| `geo`, `total_site`, `n_sites`, `multiplicity`, `sizeof_m` | geometry and layout |
| `qnorm()`, `qnorm_field()` | global and per-site norm |
| `set_zero()`, `set_unit(coef=1.0)` | constant fields |
| `set_rand(rng, upper=1.0, lower=0.0)` | uniform random real/imaginary parts |
| `set_rand_g(rng, center=0.0, sigma=1.0)` | Gaussian random real/imaginary parts |
| `glb_sum()`, `swap(f1)` | reductions and buffer exchange |
| `copy(is_copying_data=True)`, `init_from_geo(geo=None, multiplicity=0)` | construction |
| `__len__`, `__getitem__`, `__setitem__` | buffer access |
| `__getstate__`, `__setstate__` | pickling (the buffer plus the geometry) |

`init_from_geo` and `set_unit` are the two entry points qlat's own code uses
most, so the argument order and defaults are qlat's: `set_rand` takes `upper`
**before** `lower` although it forwards `(lower, upper)` to the RNG, which is
qlat's public order and is kept because the repo call sites pass the bounds
positionally (see [`qlat_rng`](qlat_rng.md)).

## `GaugeField`, `GaugeTransform`, `GaugeMomentum`

The three subclasses differ only in their multiplicity and in the operators the
reference needs:

| Type | Multiplicity | Extra members |
|---|---|---|
| `GaugeField(geo=None, multiplicity=4)` | `4` (one link per direction) | `plaq()`, `link_trace()`, `unitarize()`, `su_projection()`, `save(path)`, `load(path)`, `twist_boundary_at_boundary()` |
| `GaugeTransform(geo=None, multiplicity=1)` | `1` (one matrix per site) | `inv()`, `unitarize()`, `save(path)`, `load(path)`, `__mul__` |
| `GaugeMomentum(geo=None, multiplicity=4)` | `4` | `set_rand(rng, sigma=1.0)`, `__rmul__`, `__matmul__` |

The `*` algebra is qlat's exactly:

- `gt * gf` applies the transformation
  `U_mu(x) -> gt(x) U_mu(x) gt(x+mu)^dagger`;
- `gt * gt` composes as `t = t1 * t0` (the right operand acts first);
- `gt * gm` acts on a momentum at the **same** site, with no `x + mu` shift;
- `gt.inv()` is the site-wise adjoint.

`GaugeMomentum.set_rand(rng, sigma=1.0)` fills anti-Hermitian traceless
matrices drawn from `exp[-Tr(M^2) / (2 sigma^2)]`; the free function
`qlat_ref.hmc.set_rand_gauge_momentum(gm, sigma, rng)` is qlat's name and
argument order for the same operation.

`save`/`load` use qlat's NERSC `.lat` chunked file format, so a field written
by either package is readable by the other (pinned by
`tests/test_qlat_parity.py::TestFileFormatParity`).  A corrupt file raises
`IOError` here, where qlat warns and continues — a documented deviation.

`plaq()` returns the average plaquette of the field and `link_trace()` the
average `Tr U_mu(x) / 3`; both are the quantities the HMC tests monitor.

## The Array Bridge

The bridge is the documented contract between the types and plain NumPy:
`np.asarray(field)` is a **zero-copy view** of the buffer, so

```python
np.asarray(gf)[site, mu] = something      # writes through to the field
```

and a plain array can be handed to any function that accepts a field.  To move
between the buffer and the `(Nt, Nx, Ny, Nz, ...)` convention the numerical
code uses,

```python
from qlat_ref.field import get_array_from_field, get_field_from_array

U = get_array_from_field(gf)      # (4, Nt, Nx, Ny, Nz, NC, NC)
gf = get_field_from_array(U, geo)
```

The conversion is `np.moveaxis(arr, 0, -3).reshape(local_volume, 4, NC, NC,
order='F')` and back, i.e. the first coordinate varies fastest and the lattice
directions come *after* the site in the buffer.  `get_site_array(f)` and
`get_link_array(f)` in `qlat_ref.observables` are the same conversion for a
site field and a link field, and they accept a field **or** a plain array.

`np.array(f)` is a genuine copy (`copy=True`), while `np.asarray(f)` is a live
view.  The HMC drivers mutate their field in place, so a caller that needs the
pre-trajectory configuration must take `np.array(gf)` or `gf.copy()` **before**
the call.

## Deviation: No MPI

`qlat_ref` is single-process.  The quantities qlat splits across MPI nodes are
the global ones here:

- `node_site == total_site`, `coor_node == 0`, `id_node == 0`, `num_node == 1`
  for a non-degenerate geometry (and `num_node == 0` for the all-zero
  `Geometry()`);
- `glb_sum()` is the plain NumPy sum over the local (== global) sites;
- there is no broadcast, so `metropolis_accept` returns the flag directly.

The names keep qlat's `_node` suffix for the HMC quantities because they are
qlat's names, even though here the node-local value is the global one.

Everything else follows qlat: `Geometry()` is empty but safe in **both**
packages (qlat-issue-9, fixed upstream), the `.lat` format is shared, and the
pickling state of a field is the buffer plus the geometry.

## Self-Checks

```bash
cd qlat-ref
python3 -m unittest discover -s tests -t . -v
```

The type checks are in `tests/test_types.py` (construction, the buffer layout,
the array bridge, `set_rand` determinism, the `*` algebra, `.lat` round trips,
the typed unitarity contract and pickling) and `tests/test_invariants.py`
(unitarity and determinism of the random fixtures, trivially fixed
configurations).  `tests/test_interface_parity.py` additionally asserts that
every method and attribute listed here exists on the live qlat type with the
same signature, and `tests/test_qlat_parity.py` compares the numerical values.
