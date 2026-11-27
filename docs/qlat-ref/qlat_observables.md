# `qlat_ref.observables` — Gauge Functionals, Divergences and Monitoring (NumPy Reference)

Source: `qlat-ref/qlat_ref/observables.py`

> **Note:** Update this document when updating the source file.

## Outline

1. [Overview](#overview)
2. [Input Types](#input-types)
3. [Gauge Functionals](#gauge-functionals)
4. [Gauge Conditions and Divergences](#gauge-conditions-and-divergences)
5. [Coulomb Slices](#coulomb-slices)
6. [Unitarity](#unitarity)
7. [Using the Observables](#using-the-observables)
8. [Deviations from `qlat`](#deviations-from-qlat)
9. [Self-Checks](#self-checks)

---

## Overview

`qlat_ref.observables` holds the scalar quantities that monitor a gauge field:
the gauge **functional** that gauge fixing maximises, the RMS **divergence**
that measures how well the gauge condition holds, the same two quantities split
by time slice for the Coulomb gauge, and the **unitarity** checks.

The functions are the ones `qlat.gauge_fixing` provides (`get_gauge_functional`
and `get_gauge_divergence`), plus the pieces qlat keeps as private C++ helpers
(the per-slice lists, the gradient divergence and the array bridge), which the
reference exposes because the fixing code and the tests need them.

All of them are **read-only** and work on the whole lattice at once with plain
NumPy array operations.

## Input Types

Every function accepts **either** a field object **or** a plain NumPy array:

- a `qlat_ref.field.GaugeField` (a link field, multiplicity `4`) or a
  `(4, Nt, Nx, Ny, Nz, NC, NC)` link array;
- a `qlat_ref.field.GaugeTransform` (a site field, multiplicity `1`) or a
  `(Nt, Nx, Ny, Nz, NC, NC)` site array, for the site-field entry points.

The two conventions are equivalent through the documented array bridge
(see the [`qlat_ref` types](qlat_types.md) page), and reconciling them in one
place is exactly what the two converters do:

| Converter | Meaning |
|---|---|
| `get_link_array(gf)` | field or array → `(4, Nt, Nx, Ny, Nz, NC, NC)` |
| `get_site_array(gf)` | field or array → `(Nt, Nx, Ny, Nz, NC, NC)` |

A wrong shape or an out-of-range direction raises `ValueError`; there is no
silent reinterpretation of the buffer.

## Gauge Functionals

The normalized gauge functional is

$$
F = \frac{1}{V \cdot |\text{dirs}| \cdot N_c} \sum_{x,\, \mu \in \text{dirs}} \operatorname{Re} \operatorname{Tr} U_\mu(x)
\tag{1}
$$

with $V$ the number of lattice sites.  `get_gauge_functional(gf, dirs)` takes
the direction set as a required positional argument, exactly like
`qlat.gauge_fixing.get_gauge_functional(gf, dirs)`; the Landau functional uses
all four directions, the Coulomb functional the spatial ones (see below).

The Coulomb functional splits into one independent three-dimensional Landau
functional per time slice:

$$
F_t = \frac{1}{V_s \cdot 3 \cdot N_c} \sum_{x_s} \sum_{i = 0}^{2} \operatorname{Re} \operatorname{Tr} U_i(t, x_s)
\tag{2}
$$

which is `get_coulomb_functional_list(gf)`, a Python list of length `Lt`.
$V_s$ is the spatial volume and the indices $i$ are the spatial directions.

## Gauge Conditions and Divergences

Gauge fixing maximises the functional; the gauge condition is that the local
gradient vanishes.  The local gradient of a link field is the anti-Hermitian
traceless part of the adjoint staple sum,

$$
D(x) = \frac{K(x)^\dagger - K(x)}{2} - \frac{1}{3} \operatorname{Tr} \frac{K(x)^\dagger - K(x)}{2}
\tag{3}
$$

with `K(x)` from `qlat_ref.field.get_staple_sum(gf, dirs)`; it is the ascent
direction of the functional, so `exp(alpha * D(x))` with a positive `alpha` is
the local gauge transformation that increases it.  The gradient itself is
`qlat_ref.field.get_local_gauge_gradient(gf, dirs=None)`, and
`get_divergence_from_gradient(D)` turns a gradient field into the divergence.

The convergence measure is the RMS divergence

$$
\theta = \sqrt{\frac{1}{V \cdot N_c} \sum_{x} \operatorname{Tr}\bigl( D(x) D(x)^\dagger \bigr)}
\tag{4}
$$

which vanishes exactly when the gauge condition holds.  `get_gauge_divergence(gf,
dirs)` computes it from a link field, and it is the quantity the fixing drivers
compare against `eps`.  The same divergence is also available per time slice
for the Coulomb gauge, `get_coulomb_divergence_list(gf)`.

Because the two functionals are normalised by the number of directions, a
divergence computed over a *subset* of directions is meaningful: it measures
whether the gauge condition holds along those links only.  That is what makes
the direction-set deviation of the Coulomb fixing observable (below).

## Coulomb Slices

The Coulomb gauge transformation acts inside a single time slice, so a Coulomb
fixing is `Lt` independent three-dimensional problems.  Both per-slice lists
are indexed by time, i.e. by component `3` of `total_site` (qlat's convention:
`Geometry.spatial_volume` divides out component 3 and `get_psel_tslice` slices
`t_dir=3`).  For a link array `U` of shape `(4, Nt, Nx, Ny, Nz, NC, NC)` the
time axis is array axis `4`, because indexing `U[mu]` drops the direction
axis.

The spatial directions are therefore `(0, 1, 2)`, **not** qlat's `(1, 2, 3)`;
see the deviation note below and on the
[gauge fixing](qlat_gauge.md) page.

## Unitarity

Two entry points exist, with different contracts:

| Function | Accepted types | Use |
|---|---|---|
| `get_unitarity_error(gf)` (also exported by `qlat_ref.mat`) | any field of matrices: a `ColorMatrix`, a site/link field, a plain array | the generic matrix routine, used inside the reference formulas |
| `observables.get_unitarity_error_typed(gf)` | a `GaugeField` **or** a `GaugeTransform` only | the strict, qlat-identical contract of `qlat.gauge_fixing.get_unitarity_error(gf)` |

Both return the maximal `|U U^dagger - 1|` matrix element as a `float`, and
both reject a `GaugeMomentum` (whose anti-Hermitian elements are not unitarity
defects) from the *typed* entry point with `TypeError`.  The typed entry point
takes a `GaugeField` over every site and every direction, or a `GaugeTransform`
over every site.

## Using the Observables

```python
import qlat_ref as qr

U = qr.get_random_gauge_field((2, 2, 2, 2), seed=0)
dirs = (0, 1, 2, 3)

print(qr.get_gauge_functional(U, dirs))     # qlat's signature: dirs required
print(qr.get_gauge_divergence(U, dirs))

gt, info = qr.fix_landau_gauge(U, max_iter=2000)
U_fixed = qr.apply_gauge_transform(U, gt)
assert qr.get_gauge_divergence(U_fixed, dirs) <= info["eps"]
assert qr.get_unitarity_error(U_fixed) <= 1e-10

gt_c, info_c = qr.fix_coulomb_gauge(U, max_iter=2000)
U_coulomb = qr.apply_gauge_transform(U, gt_c)
assert qr.get_gauge_divergence(U_coulomb, (0, 1, 2)) <= info_c["eps"]
print(qr.get_coulomb_divergence_list(U_coulomb))
print(qr.get_coulomb_functional_list(U_coulomb))
```

The `dirs` default (`(0, 1, 2, 3)`) lives in the facade `qlat_ref.gauge`, so
the historical one-argument calls keep working there:

```python
import qlat_ref as qr

U = qr.get_random_gauge_field((2, 2, 2, 2), seed=0)
print(qr.gauge.get_gauge_functional(U))     # facade: dirs defaults
```

The top level `qlat_ref` deliberately keeps qlat's exact signature
(`get_gauge_functional(gf, dirs)`), so `qr.get_gauge_functional(U)` is a
`TypeError` — the same thing it is in qlat.

## Deviations from `qlat`

1. **`get_gauge_functional` / `get_gauge_divergence` keep qlat's required
   `dirs`** at the top level and in this module; the default is provided by the
   `qlat_ref.gauge` facade only (a compatibility layer, not a signature change).
2. **Coulomb slices use the spatial directions `(0, 1, 2)`.**  Component 3 is
   time in qlat's own convention, so qlat's `fix_coulomb_gauge` direction set
   `{1, 2, 3}` includes time — qlat-issue-8, decided to be fixed here rather
   than reproduced.  The divergence of a Coulomb-fixed field is `O(1)` over
   `(1, 2, 3)` and at `eps` over `(0, 1, 2)`, so the deviation is observable.
3. **The array-level `get_unitarity_error` accepts more types than qlat.**
   qlat's version takes only a `GaugeField` or a `GaugeTransform`; here the
   generic matrix routine is exported under the same name and takes any field of
   matrices.  The strict contract is
   `get_unitarity_error_typed`, which is qlat-identical.
4. **The per-slice lists are public here.**  `get_coulomb_functional_list` and
   `get_coulomb_divergence_list` mirror the C++ helpers qlat keeps private;
   they are additions, not renames of a qlat Python entry point.

## Self-Checks

```bash
cd qlat-ref
python3 -m unittest discover -s tests -t . -v
```

`tests/test_invariants.py` checks the functional and the divergence of a
trivially fixed configuration (unit field), the per-slice lists against the
three-dimensional Landau problem, and the gradient divergence against the link
divergence.  `tests/test_interface_parity.py` asserts that
`get_gauge_functional`/`get_gauge_divergence`/`get_unitarity_error` exist in
qlat with the same signatures, and `tests/test_qlat_parity.py` compares their
values against a live build and pins the Coulomb direction deviation.
