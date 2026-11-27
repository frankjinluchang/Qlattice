# `qlat_ref.gauge` — Landau and Coulomb Gauge Fixing (NumPy Reference)

Source: `qlat-ref/qlat_ref/gauge_fixing.py` (facade:
`qlat-ref/qlat_ref/gauge.py`)

> **Note:** Update this document when updating the source file.

## Outline

1. [Overview](#overview)
2. [Array Layout and Conventions](#array-layout-and-conventions)
3. [Algorithm](#algorithm)
4. [Convergence and Monitoring](#convergence-and-monitoring)
5. [Public Interface](#public-interface)
6. [Momentum Fields](#momentum-fields)
7. [Examples](#examples)
8. [Self-Checks](#self-checks)

---

## Overview

`qlat_ref.gauge` is a deliberately small, pure-NumPy reference implementation
of **Landau gauge fixing** and **Coulomb gauge fixing** on a
four-dimensional lattice.  It is the reference used to cross-check the
efficient C++/Cython implementation in `qlat`; it is not a performance target.

The iteration scheme is **plain local steepest descent**: every site is
updated at once by the local gauge transformation that increases the gauge
functional, without Fourier acceleration and without FFT.  The whole lattice is
treated with NumPy array operations, so there is no loop over sites in the
code.

## Array Layout and Conventions

A gauge field is a NumPy array `U` with shape `(4, Nt, Nx, Ny, Nz, 3, 3)` and
`dtype` `numpy.complex128`:

    U[mu, t, x, y, z, i, j] = U_mu(t, x, y, z)[i, j]

`U[mu]` is the field of links in direction `mu` (with `0 = t`, `1 = x`,
`2 = y`, `3 = z`), so the site axes of `U[mu]` are the time and the three
spatial directions, and the last two axes are the (row, column) color indices
of the `SU(3)` matrix.

A gauge transformation is a site field of shape `(Nt, Nx, Ny, Nz, 3, 3)`.
The lattice is periodic in every direction: a shift by one hop in direction
`mu` is `numpy.roll(..., 1, axis=mu)`.  For a link field `U[mu]` the
coordinate axis of direction `mu` is `axis=mu`, because `U[mu]` drops the
leading direction axis of `U`.

The gauge transformation convention is

$$
U_\mu(x) \to g(x) U_\mu(x) g(x+\mu)^\dagger
\tag{1}
$$

which is implemented by `apply_gauge_transform`.

## Algorithm

The Landau gauge functional is

$$
F = \frac{1}{V \cdot 4 \cdot 3} \sum_{x,\mu} \operatorname{Re} \operatorname{Tr} U_\mu(x)
\tag{2}
$$

and the Coulomb gauge functional is the same sum restricted to the spatial
directions `i = 0, 1, 2`.  Component 3 is **time** in qlat's convention
(`total_site` is `(Lx, Ly, Lz, Lt)`; `Geometry.spatial_volume` divides out
component 3, and `get_psel_tslice` slices `t_dir=3`), so the spatial directions
are `0, 1, 2`, not `1, 2, 3`.  The Coulomb functional splits into one
independent three-dimensional Landau functional per time slice, because a
Coulomb gauge transformation `g(t, x)` acts inside a single time slice on the
spatial links.

> **Deviation from qlat.** `q.fix_coulomb_gauge` uses the directions
> `{1, 2, 3}`, i.e. it fixes the three directions that *include* time and
> excludes the spatial direction `0`.  That is qlat-issue-8 in
> `tmp/qlat-ref-reorg/qlat-issues.md`, and the user decided `qlat_ref`
> implements the correct set.  The deviation is intentional and observable:
> the field fixed by `qlat_ref` satisfies the gauge condition for `(0, 1, 2)`
> and not for `(1, 2, 3)`, while the field fixed by `q.fix_coulomb_gauge`
> shows the opposite.  `q.fix_coulomb_gauge` is exactly
> `q.fix_gauge(gf, dirs=[1, 2, 3])`, so
> `examples-py/gauge-fixing-cross-check.py` compares against
> `q.fix_gauge(gf, dirs=[0, 1, 2])` — the same algorithm on the correct
> directions — instead of against the buggy fixer.

At every site the local gradient matrix is

$$
D(x) = \frac{K(x)^\dagger - K(x)}{2} - \frac{1}{3} \operatorname{Tr}\frac{K(x)^\dagger - K(x)}{2}
\tag{3}
$$

with the staple sum

$$
K(x) = \sum_{\mu} \left( U_\mu(x) + U_\mu(x-\mu)^\dagger \right)
\tag{4}
$$

the sum running over all four directions for Landau gauge and over the
spatial directions `(0, 1, 2)` for Coulomb gauge (see the note above).
`D(x)` is anti-hermitian and traceless,
so the local gauge transformation

$$
g(x) = \exp(\alpha D(x))
\tag{5}
$$

is in `SU(3)` (the matrix exponential is a truncated Taylor series, see
`get_su3_exp`).  The step size `alpha` defaults to `0.1`; it is the
over-relaxation parameter of the plain scheme.  Larger steps converge faster
but become unstable, so `alpha = 0.1` is the safe default.

One sweep computes `D(x)` on the whole lattice and then applies the
transformation to every site at once (Jacobi sweep, no site loop and no
ordering dependence).  The gauge field `U` passed to the fixing functions is
not mutated: the fixing functions accumulate the gauge transformation of all
sweeps and return it.  The accumulated transformation is multiplied on the
left by every new sweep,

$$
g_{n+1} = g_{\text{step}} \, g_n
\tag{6}
$$

so that applying it once reproduces the field after every sweep, and
`apply_gauge_transform(U, gt)` is the gauge-fixed field.

## Convergence and Monitoring

The convergence measure is the RMS lattice divergence

$$
\theta = \sqrt{\frac{1}{V \cdot 3} \sum_x \operatorname{Tr} \left( D(x) D(x)^\dagger \right)}
\tag{7}
$$

which vanishes exactly when the gauge condition `D(x) = 0` holds.  The
iteration stops when `theta <= eps` (default `eps = 1e-10`) or after
`max_iter` sweeps (default `10000`).  Both the functional and the divergence
of every sweep are recorded in the returned `info` dictionary, which makes it
easy to monitor the convergence.

## Public Interface

- `get_unit_gauge_field(total_site)` — unit gauge field.
- `get_random_gauge_field(total_site, *, seed=0, scale=1.0)` — deterministic
  random unitary gauge field.
- `get_random_gauge_transform(total_site, *, seed=0, scale=0.3)` — deterministic
  random `SU(3)` gauge transformation.
- `apply_gauge_transform(U, g)` — apply a gauge transformation to a link
  field.
- `apply_gauge_transform_to_momentum(P, g, is_dagger=False)` — apply the same
  gauge transformation to a momentum field (no shift).
- `get_gauge_functional(gf, dirs=_ALL_DIRS)` — normalized gauge functional.
  The required `dirs` of `qlat` is given a default here, so the historical
  `get_gauge_functional(U)` call keeps working.
- `get_coulomb_functional_list(U)` — Coulomb functional of each time slice.
- `get_gauge_divergence(gf, dirs=_ALL_DIRS)` — RMS divergence `theta` (same
  `dirs` default).
- `get_coulomb_divergence_list(U)` — RMS divergence of each time slice.
- `get_local_gauge_gradient(gf, dirs=None)` — local gradient field `D(x)`.
- `get_staple_sum(gf, dirs=None)` — staple sum `K(x)`.
- `get_unitarity_error(gf)` / `check_unitarity(gf, *, eps=1e-12)` — unitarity
  monitoring.  `observables.get_unitarity_error_typed(gf)` is the typed entry
  point that mirrors `qlat.gauge_fixing.get_unitarity_error`: it accepts a
  `GaugeField` **or** a `GaugeTransform` and raises `TypeError` for anything
  else (a `GaugeMomentum`, a raw array, ...), so both packages share one
  accepted-type contract.  The array-level `mat.get_unitarity_error(U)` stays
  the generic matrix routine and takes any array.
- `fix_landau_gauge(gf, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)`
  — Landau gauge fixing.
- `fix_coulomb_gauge(gf, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)`
  — Coulomb gauge fixing.
- `fix_gauge(gf, dirs, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)` —
  gauge fixing with an explicit direction list.

Every positional signature above matches `qlat`'s exactly, with one
deliberate exception: the facade gives `dirs` a default for
`get_gauge_functional` and `get_gauge_divergence` so the historical
one-argument calls keep working.  The top level `qlat_ref` and the topic module
`qlat_ref.observables` keep qlat's required `dirs`, so
`qr.get_gauge_functional(U)` is a `TypeError` there, exactly as in qlat.
`qlat-ref/tests/test_interface_parity.py` asserts the top-level signatures
against a live build and pins the facade default;
`qlat-ref/tests/test_qlat_parity.py` additionally compares the numerical
results.
`gf` may be a plain `(4, Nt, Nx, Ny, Nz, NC, NC)` array or a
`qlat_ref.field.GaugeField`; the two are equivalent through the array bridge
described in [`qlat_ref` overview](qlat_ref_overview.md).

`fix_landau_gauge` and `fix_coulomb_gauge` return a tuple `(gt, info)`, where
`gt` is the accumulated gauge transformation: the gauge-fixed field is
`apply_gauge_transform(U, gt)`.  `info` contains the gauge name, the direction
set, the final functional and divergence, the convergence flag, the number of
sweeps and the per-sweep history.

`gt` is **not unique**.  It depends on the fixing path (`alpha`, `max_iter`,
rounding), and every transformation that leaves the gauge condition invariant
multiplies it without changing the fixed field — for Landau gauge a global
center element, for Coulomb gauge one per time slice.  Compare fixed *fields*
or gauge-invariant observables, never the transformations themselves.

## Momentum Fields

The conjugate momentum of the gauge field (the `GaugeMomentum` of `qlat`) is an
anti-hermitian traceless color matrix on every site and direction, so it has
the same array layout as a link field:

    P.shape == (4, Nt, Nx, Ny, Nz, 3, 3)

Unlike a link, a momentum is attached to a single site: it carries no
neighbor index and the gauge transformation therefore acts without the shift
of Eq. (1),

$$
P_\mu(x) \to g(x) P_\mu(x) g(x)^\dagger
\tag{8}
$$

which is the adjoint action of `SU(3)` on its Lie algebra.  It preserves both
the anti-hermiticity and the tracelessness of `P`, and it is the same
expression as Eq. (1) with the `numpy.roll` of the link transformation
removed — that shift is exactly the part that belongs to a link and not to a
site field.  With `is_dagger=True` the transformation acts with $g^\dagger$ on
the left and with $g$ on the right, which is its inverse, so the two calls
with the same `g` restore the input.  Like the link transformation, it does
not modify its inputs and returns a new array.

## Examples

```python
import numpy as np
import qlat_ref as qr

total_site = (2, 2, 2, 2)
U = qr.get_random_gauge_field(total_site, seed=0)
gt, info = qr.fix_landau_gauge(U)
U_fixed = qr.apply_gauge_transform(U, gt)
assert info["converged"]
assert qr.get_gauge_divergence(U_fixed) <= info["eps"]
assert qr.check_unitarity(U_fixed, eps=1e-10)
assert qr.check_unitarity(gt, eps=1e-10)  # gt is a site field; no np.stack needed

gt_coulomb, info_coulomb = qr.fix_coulomb_gauge(U)
U_coulomb = qr.apply_gauge_transform(U, gt_coulomb)
assert info_coulomb["converged"]
# U is (4, L0, L1, L2, L3, 3, 3); time is component 3, i.e. array axis 4.
n_t = U.shape[4]
assert max(qr.get_coulomb_divergence_list(U_coulomb)) <= np.sqrt(n_t) * info_coulomb["eps"]
```

The momentum of the gauge-fixed configuration is obtained with the same
transformation:

```python
rng = np.random.default_rng(0)
H = rng.standard_normal(U.shape) + 1j * rng.standard_normal(U.shape)
P = qr.get_traceless_antihermitian_part(H)  # a momentum field
P_fixed = qr.apply_gauge_transform_to_momentum(P, gt)
```

so that `(U_fixed, P_fixed)` is the gauge-fixed configuration.  Applying `gt`
a second time is not the inverse: use `is_dagger=True`, which restores `P`
exactly.

## Self-Checks

The package owns a standard-library `unittest` suite in `qlat-ref/tests/`.
Run it from the checkout (no install needed):

```bash
cd qlat-ref
python3 -m unittest discover -s tests -t . -v
```

From the repository root the equivalent is:

```bash
python3 -m unittest discover -s qlat-ref/tests -t qlat-ref -v
```

The historical `python3 -m qlat_ref.tests` entry point still works as a thin
shim around the same suite.  The nix derivation runs the identical suite as an
install check (`unittestCheckHook`), and `./scripts/qlat-ref-test.sh` runs it in
the `./scripts` flow; a failing test fails those builds.

The suite is split into `tests/test_gauge_fixing.py` (fixing),
`tests/test_momentum.py` (the momentum transform) and
`tests/test_invariants.py` (unitarity, determinism and array conventions), with
shared deterministic helpers in `tests/helpers.py`.  It checks that the random
fields are unitary, that the functional increases
monotonically during the plain iteration, that both gauges converge and
satisfy their gauge condition, that a `1^4` lattice is trivially gauge fixed
(the functional is invariant and the divergence vanishes), that the iteration
is exactly covariant under a global gauge transformation (both the fixed field
and the accumulated transformation), that the accumulated transformation
reproduces the iterated field, that the transform is only unique up to the
center stabiliser (the Landau and Coulomb fixed fields are unaffected while the
transforms differ), and that the converged functional does not depend on the
step size.  The momentum transformation is checked against the
site-by-site adjoint action, for the `is_dagger` inverse, for the composition
order `g_2 g_1`, for the preservation of the anti-hermitian traceless algebra
and of the invariant `sum_x Tr P^2`, and for undoing the gauge fixing.
