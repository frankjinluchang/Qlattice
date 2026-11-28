# `qlat.gauge_fixing` — Landau and Coulomb Gauge Fixing

Source: `qlat/qlat/gauge_fixing.pyx` (C++ core in
`qlat/qlat/include/qlat/gauge-fixing.h` and `qlat/qlat/lib/gauge-fixing.cpp`)

> **Note:** Update this document when updating the source file.

## Outline

1. [Overview](#overview)
2. [Algorithm](#algorithm)
3. [Convergence](#convergence)
4. [Python API](#python-api)
5. [MPI and Performance](#mpi-and-performance)
6. [Example and Cross-Check](#example-and-cross-check)

---

## Overview

`qlat.gauge_fixing` implements **Landau gauge fixing** and **Coulomb gauge
fixing** on a distributed `GaugeField` (`Field<ColorMatrix>`) with a plain
local iterative scheme: every site is updated by the local gauge
transformation that increases the gauge functional.  There is **no Fourier
acceleration and no FFT**.  The site loops live in C++; the Python layer only
calls the Cython wrapper.

## Algorithm

The normalized gauge functional is

$$
F = \frac{1}{V \cdot n_{\mathrm{dir}} \cdot 3} \sum_{x,\mu \in \mathrm{dirs}} \operatorname{Re} \operatorname{Tr} U_\mu(x)
\tag{1}
$$

with all four directions for Landau gauge and the three spatial directions
for Coulomb gauge.  The Coulomb functional splits into one independent
three-dimensional Landau functional per time slice, because a Coulomb gauge
transformation acts inside a single time slice on the spatial links; the
resulting transformation is still applied to all four link directions.

At every site the local staple sum and the local gradient matrix are

$$
K(x) = \sum_{\mu \in \mathrm{dirs}} \left( U_\mu(x) + U_\mu(x-\mu)^\dagger \right)
\tag{2}
$$

$$
D(x) = \frac{K(x)^\dagger - K(x)}{2} - \frac{1}{3} \operatorname{Tr} \frac{K(x)^\dagger - K(x)}{2}
\tag{3}
$$

`D(x)` is anti-hermitian and traceless and is the ascent direction of the
functional.  One Jacobi sweep computes `D(x)` on the whole lattice and then
applies the local gauge transformation to every site at once:

$$
g(x) = \exp(\alpha D(x))
\tag{4}
$$

$$
U_\mu(x) \to g(x) U_\mu(x) g(x+\mu)^\dagger
\tag{5}
$$

The matrix exponential is the truncated Taylor (Horner) recurrence of
`qlat_utils.make_matrix_exp` with order 20.  The step size `alpha` is the
over-relaxation parameter of the plain scheme; it defaults to `0.1`, which is
stable and monotone for the tested lattices.  Larger steps converge faster
but become unstable.

## Convergence

The convergence measure is the RMS lattice divergence

$$
\theta = \sqrt{\frac{1}{V \cdot 3} \sum_x \operatorname{Tr} \left( D(x) D(x)^\dagger \right)}
\tag{6}
$$

which vanishes exactly when the gauge condition `D(x) = 0` holds.  The
iteration stops when `theta <= eps` (default `1e-10`) or after `max_iter`
sweeps (default `10000`).  The returned dictionary contains the full
per-sweep functional and divergence history.

## Python API

All three fixing functions leave their input `qlat.GaugeField` **untouched**
and return a tuple `(gt, info)`: `gt` is a `qlat.GaugeTransform` and the
gauge-fixed field is the new field `gt * gf`.  The getters are read-only:

- `q.fix_gauge(gf, dirs, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)`
  — fix with an explicit list of link directions; returns `(gt, info)`.
- `q.fix_landau_gauge(gf, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)`
  — Landau gauge fixing (all four directions); returns `(gt, info)`.
- `q.fix_coulomb_gauge(gf, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)`
  — Coulomb gauge fixing (spatial directions, per time slice); returns
  `(gt, info)`.
- `q.get_gauge_functional(gf, dirs)` — normalized gauge functional.
- `q.get_gauge_divergence(gf, dirs)` — RMS divergence `theta`.
- `q.get_unitarity_error(gf)` — maximal `|U U^dagger - 1|` matrix element.  It
  accepts a `qlat.GaugeField` (over every site and all four link directions)
  and a `qlat.GaugeTransform` (over every site), since the number only depends
  on the field's matrices; any other type (`qlat.GaugeMomentum`, a raw array,
  a plain list) raises `TypeError`.

The returned `gt` acts on the left exactly as in [Eq. (5)](#algorithm), so
`gt * gf` allocates a new `GaugeField` and the caller keeps the original.  In
place into a destination field: `gf_dest @= gt * gf`.

The `info` dictionary has the keys `gauge`, `converged`, `n_iter`, `max_iter`,
`eps`, `alpha`, `functional`, `divergence`, `functional_list` and
`divergence_list`; `functional` and `divergence` describe the field
`gt * gf`.

### The returned transform is not unique

`gt` depends on the convergence path and is only defined up to the residual
stabiliser of the gauge condition: multiplying `gt` by a global third root of
unity times the identity gives a different `gt` for a bit-identical fixed
field.  Code that needs to compare two implementations must therefore compare
the **applied** fixed fields `gt * gf`, never the raw `gt` arrays.

## MPI and Performance

The gauge field is distributed over the `qlat` geometry.  Each sweep only
needs the neighbouring links at the node boundaries: the implementation keeps
one expanded one-hop working copy of the gauge field (seeded once from the
input, `refresh_expanded` with the cached `CommPlan` once per sweep), a compact
`gt` that is distributed like `gf`, an expanded gradient `D`, and a compact
per-sweep step `gt_step`.  The functional, the divergence and the unitarity
error are reduced with `field_glb_sum`/`field_glb_max`, so all nodes stay in
sync.  Site loops are in C++ (`qacc_for`); there is no Python loop over sites.

Two sweeps' worth of passes are fused, which matters on a GPU where every
`qacc_for` is a kernel launch followed by a device synchronisation:

- the per-site functional and divergence densities are evaluated in **one**
  pass, held as the two components of a single `Field<RealD>`, and reduced by
  **one** `field_glb_sum`, so the whole reduction is a single host sum plus a
  single `MPI_Allreduce` instead of two of each;
- the four link updates of [Eq. (5)](#algorithm) run as **one** pass with the
  direction loop inside it.  The four directions are independent, because a
  link's update reads only `g(x)`, `U_mu(x)` and the read-only expanded
  `g(x+mu)`, so fusing them changes neither the update order nor the result.

A sweep therefore issues 6 `qacc_for` passes and 1 reduction instead of 10
passes and 2 reductions; the values are unchanged bit for bit, including the
per-sweep `functional_list` and `divergence_list` histories.

Peak memory is the working copy plus `gt` plus `D`'s halo: at `64^3 * 128`
that is **+5.012 GiB** over the previous implementation (about `+0.28x` a
`GaugeField`), and **less** in absolute terms than the old peak because the
old `gf_gauge_gradient` transient is gone.

## Example and Cross-Check

The example `examples-py/gauge-fixing.py` runs on a `4^4` lattice with two
MPI processes, fixes both gauges and checks the convergence, the gauge
condition, unitarity, the monotone increase of the functional and the
invariance of the plaquette.  It obtains the fixed field as `gt * gf` and
asserts that the input field is left exactly unchanged (link matrices,
plaquette and functional).  It is registered in `examples-py/Makefile` and
runs through the examples-py test suite / `nixpkgs/run-one-example-py.py`.

`examples-py/gauge-fixing-cross-check.py` is a manual (non-CI) script — it has
no reference log and is not a `Makefile` target — that fixes the same
deterministic random field with this implementation and with the pure-NumPy
reference package `qlat-ref`, and compares the converged functional, the
divergence, the unitarity error and the fixed link matrices.  Both
implementations return a transform, so the comparison applies both transforms
to the same original field and compares the resulting fixed fields (see the
non-uniqueness note above).  Run it manually with `mpiexec`.

**Coulomb gauge is compared with `q.fix_gauge`, not `q.fix_coulomb_gauge`.**
Component 3 is time in this implementation's own convention (`total_site` is
`(Lx, Ly, Lz, Lt)`, `Geometry::spatial_volume` divides out component 3,
`get_psel_tslice` slices `t_dir=3`), so the spatial directions are `(0, 1, 2)`.
`fix_coulomb_gauge` here uses `{1, 2, 3}` instead — it fixes the three
directions that include time and excludes spatial direction 0.  That is
**qlat-issue-8** (`tmp/qlat-ref-reorg/qlat-issues.md`); the user decided
`qlat-ref` implements the correct set rather than reproducing the bug.  The
cross-check therefore drives the qlat side with
`q.fix_gauge(gf, dirs=[0, 1, 2])`, which runs the same local
steepest-descent algorithm on the correct directions — `fix_coulomb_gauge` is
exactly `fix_gauge(dirs=[1, 2, 3])` to round-off, so nothing but the direction
set differs.  The script additionally asserts the observed qlat behaviour
(`fix_coulomb_gauge` converges `(1, 2, 3)` and not `(0, 1, 2)`), so an
upstream fix of qlat will surface here as a failure.

`qlat-ref` is an installed package in the nix development environment (it is
part of `qlat-jhub-env` and has its own `qlat_ref` nix attribute), so the
script just imports it — but it prefers a build that has the topic modules
(`qlat_ref.gauge_fixing`) and otherwise falls back to the sibling source tree,
so a stale installed copy cannot make the cross-check measure old code:

```bash
source result-py-local/bin/setenv-qlat.sh
mpiexec -n 2 --oversubscribe --bind-to none \
  python3 -m mpi4py examples-py/gauge-fixing-cross-check.py
```

It is still not in CI because the nix *example* derivation does not depend on
`qlat-ref`.  A fallback adds the sibling `qlat-ref/` source tree when the
environment has not been rebuilt since `qlat-ref` was packaged (or still has
the pre-reorganisation single-module layout), so a plain checkout also works.  The reference package's own tests are run with
`cd qlat-ref && python3 -m unittest discover -s tests -t . -v` (see
`docs/qlat-ref.rst`).

### End-to-end flow with HMC

The same `gt` transforms an HMC momentum at the same site, by the adjoint
action

$$
P'_\mu(x) = gt(x) P_\mu(x) gt(x)^\dagger
\tag{7}
$$

(`q.gm_apply_gauge_transformation`, also available as `gt * gm`), so a
gauge-fixed configuration can be prepared together with its momentum:

```python
gt, info = q.fix_landau_gauge(gf, max_iter=5000, eps=1.0e-8, alpha=0.1)
assert info["converged"]
gf_fixed = gt * gf                                  # gf is untouched
gm_fixed = q.gm_apply_gauge_transformation(gm, gt)  # or gt * gm; gm is untouched
delta_h = q.run_hmc_evolve_pure_gauge(gm_fixed, gf_fixed, ga, n_step=8)
```

Neither transform mutates its input; the momentum law is the same-site
adjoint action, while the link law in [Eq. (5)](#algorithm) has the shifted
`g(x+\mu)^dagger` factor.  See `docs/qlat/qlat_hmc.md` for the momentum
conventions (and for the `_dual` momentum, which this transform does not
cover).
