# `qlat_ref` — Pure-NumPy Reference Package

Source: `qlat-ref/qlat_ref/`

> **Note:** Update this document when updating the source tree.

## Outline

1. [Purpose](#purpose)
2. [Module Layout](#module-layout)
3. [The Array Bridge](#the-array-bridge)
4. [qlat Compatibility Contract](#qlat-compatibility-contract)
5. [Known Deviations from qlat](#known-deviations-from-qlat)
6. [Self-Checks](#self-checks)

---

## Purpose

`qlat_ref` is a pure-NumPy reference implementation of the parts of `qlat`
that need an independent cross-check: the lattice geometry and coordinates,
the colour-matrix algebra, the gauge field types, gauge fixing, the gauge
action and the RNG.  It is **not** a performance target and it is **not**
distributed over MPI: every quantity is computed on the whole lattice in one
process, with NumPy array operations.

The package depends only on `numpy`.  It never imports `qlat`, so it can be
installed and imported without the C++/Cython build; the parity tests against a
live `qlat` are therefore optional and skip themselves when `qlat` is absent.

## Module Layout

The implementation is split by topic.  The dependency order is acyclic:

```
base.py          Coordinate, Geometry, NC          (no internal deps)
mat.py           ColorMatrix, matrix helpers       (no internal deps)
rng.py           RngState                          (no internal deps)
field.py         FieldColorMatrix, GaugeField,     (base, mat, rng)
                 GaugeTransform, GaugeMomentum
observables.py   functionals, divergences          (base, field, mat)
gauge_fixing.py  fix_gauge, fix_landau/coulomb     (base, field, mat, observables)
gauge_action.py  GaugeAction, action value         (mat)
hmc.py           HMC building blocks               (base, field, gauge_action, mat)
hmc_utils.py     the trajectory drivers            (field, hmc)
cgfhmc.py        Canonical Gauge-Fixed HMC         (hmc, hmc_utils, gauge_fixing)
gauge.py         the stable facade (re-exports)    (all of the above)
__init__.py      the same names as gauge.py, plus every topic module
tests.py         historical shim: `python3 -m qlat_ref.tests`
```

| Module | Contents |
|---|---|
| `qlat_ref.base` | `Coordinate`, `Geometry`, `NC`, `get_dirs`, `get_total_site` |
| `qlat_ref.mat` | `ColorMatrix`, `get_matrix_adjoint`, `get_traceless_antihermitian_part`, `get_su3_exp`, `get_unitarized`, `get_unitarity_error`, `check_unitarity`, `get_random_su3_field` |
| `qlat_ref.field` | `FieldColorMatrix`, `GaugeField`, `GaugeTransform`, `GaugeMomentum`, `get_array_from_field`, `get_field_from_array`, `apply_gauge_transform`, `apply_gauge_transform_to_momentum`, `get_staple_sum`, `get_local_gauge_gradient`, `get_unit_gauge_field`, `get_random_gauge_field`, `get_random_gauge_transform`, `gm_apply_gauge_transformation` |
| `qlat_ref.observables` | `get_link_array`, `get_site_array`, `get_gauge_functional`, `get_gauge_divergence`, `get_coulomb_functional_list`, `get_coulomb_divergence_list`, `get_divergence_from_gradient`, `get_unitarity_error_typed` |
| `qlat_ref.gauge_fixing` | `fix_gauge`, `fix_landau_gauge`, `fix_coulomb_gauge`, `_fix_gauge` |
| `qlat_ref.gauge_action` | `GaugeAction`, `get_gauge_action` |
| `qlat_ref.rng` | `RngState` |
| `qlat_ref.hmc` | `set_rand_gauge_momentum`, `gm_hamilton_node`, `gf_hamilton_node`, `set_gm_force`, `set_gm_force_dual`, `gf_evolve`, `gf_evolve_dual`, `gm_apply_gauge_transformation` |
| `qlat_ref.hmc_utils` | `metropolis_accept`, `gm_evolve_fg_pure_gauge`, `run_hmc_evolve_pure_gauge`, `run_hmc_pure_gauge` |
| `qlat_ref.cgfhmc` | `k_vectors`, `k_squared`, `resolve_k_cut`, `regularize_c_field`, `generate_c_field`, `c_field_from_file`, `partial_refresh_momentum_mode`, `run_hmc_cgfhmc_pure_gauge` |
| `qlat_ref.gauge` | the facade: every name above except the HMC ones, plus the compatibility wrappers whose `dirs` arguments have defaults |

`import qlat_ref` re-exports the names of every topic module, and also binds the
modules themselves, so `qr.GaugeField(...)` and `qr.hmc.set_gm_force(...)` both
work.  `qlat_ref.gauge` remains the gauge-only facade and keeps the historical
one-argument accessor calls, so both
`import qlat_ref as qr; qr.GaugeField(...)` and
`from qlat_ref import gauge as g; g.GaugeField(...)` work.

## The Array Bridge

Every field type stores its data in one C-contiguous NumPy array

```
(local_volume, multiplicity, NC, NC)        dtype = complex128
```

with the **flat local site index** on axis 0.  `np.asarray(field)` returns a
**zero-copy view** of that buffer, so

```python
np.asarray(gf)[site, mu] = something      # writes through to the field
```

and a plain array can be handed to any function that accepts a field.  This is
the documented bridge between the types and plain NumPy, and it is what makes
the gauge-fixing reference usable on both.

The lattice axes are **not** part of the buffer.  To move between the buffer
and the `(Nt, Nx, Ny, Nz, ...)` convention used by the examples,

```python
from qlat_ref.field import get_array_from_field, get_field_from_array

U = get_array_from_field(gf)      # (4, Nt, Nx, Ny, Nz, NC, NC)
gf = get_field_from_array(U, geo)
```

The conversion is `np.moveaxis(arr, 0, -3).reshape(local_volume, 4, NC, NC,
order='F')` and back, i.e. the first coordinate varies fastest and the lattice
directions come *after* the site in the buffer.

## qlat Compatibility Contract

For every shared name the rule is: **same name, same positional arguments,
same defaults, same return convention** as `qlat`.  Two test modules assert it
mechanically against a live `qlat`:

- `qlat-ref/tests/test_interface_parity.py` is the cheap half: it compares
  names and `inspect.signature` for the functions, checks the type members
  through an explicit table, and enforces the subset rule (every public name is
  either shared with `qlat` or a registered extra with a justification).
- `qlat-ref/tests/test_qlat_parity.py` is the numerical half: it compares the
  results of every deterministic operation.

Both skip themselves when `qlat` is not importable, because the package must
install and import with `numpy` alone.

| Name | Signature (identical to `qlat`) |
|---|---|
| `fix_gauge` | `(gf, dirs, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)` |
| `fix_landau_gauge` | `(gf, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)` |
| `fix_coulomb_gauge` | `(gf, max_iter=10000, eps=1e-10, alpha=0.1, verbose=False)` |
| `get_gauge_functional` | `(gf, dirs)` |
| `get_gauge_divergence` | `(gf, dirs)` |
| `get_unitarity_error` | `(gf)` (array bridge; strict form: `observables.get_unitarity_error_typed(gf)`) |
| `GaugeAction` | `(beta, c1=0.0)`, with **methods** `beta()` and `c1()` |
| `GaugeField` | `(geo=None, multiplicity=4)`, `set_rand(rng, sigma=0.5, n_step=1)` |
| `GaugeTransform` | `(geo=None, multiplicity=1)`, `set_rand(rng, sigma=0.5, n_step=1)` |
| `GaugeMomentum` | `(geo=None)`, `set_rand(rng, sigma=1.0)` |
| `set_rand_gauge_momentum` | `(gm, sigma, rng)` |
| `gm_hamilton_node` | `(gm)` |
| `gf_hamilton_node` | `(gf, ga)` |
| `set_gm_force` | `(gm_force, gf, ga)` |
| `set_gm_force_dual` | `(gm_force_dual, gf, gm_force)` |
| `gf_evolve` | `(gf, gm, step_size)` |
| `gf_evolve_dual` | `(gf, gm_dual, step_size)` |
| `gm_apply_gauge_transformation` | `(gm, gt, is_dagger=False)` |
| `metropolis_accept` | `(delta_h, traj, rs)` |
| `gm_evolve_fg_pure_gauge` | `(gm, gf_init, ga, fg_dt, dt)` |
| `run_hmc_evolve_pure_gauge` | `(gm, gf, ga, n_step, md_time=1.0)` |
| `run_hmc_pure_gauge` | `(gf, ga, traj, rs, *, is_reverse_test=False, n_step=6, md_time=1.0, is_always_accept=False)` |

The `*` algebra follows qlat exactly:

- `gt * gf` transforms a gauge field with `U_mu(x) -> gt(x) U_mu(x) gt(x+mu)^dag`;
- `gt * gt` composes as `t = t1 * t0` (the right operand acts first);
- `gt * gm` acts on a momentum at the **same** site, with no `x + mu` shift;
- `gt.inv()` is the site-wise adjoint.

Deterministic operations reproduce the C++ results to machine precision:
`set_rand` for all three field types, `unitarize`, `plaq`, `link_trace`, the
`gt *` products, the gauge-action value, the gauge-fixing fields, and the
NERSC `.lat` files in both directions.

## Known Deviations from qlat

Everything below is deliberate and is listed in `DEVIATIONS` in
`qlat-ref/tests/test_qlat_parity.py`, so it is checked rather than skipped.

1. **`info` carries one extra key.**  The fixing dictionary contains every key
   `qlat` provides plus `dirs`, the tuple of link directions that was actually
   used.  This is a superset, never a substitution.
2. **Coulomb gauge fixes the correct spatial directions `(0, 1, 2)`.**
   Component 3 is time throughout qlat's own convention (`total_site` is
   `(Lx, Ly, Lz, Lt)`, `Geometry.spatial_volume` divides out component 3,
   `get_psel_tslice` slices `t_dir=3`, `qlat_gpt.gauge_fix_coulomb` separates
   axis 3), so the spatial directions are `0, 1, 2`.  `q.fix_coulomb_gauge`
   uses `{1, 2, 3}` instead: a qlat bug (qlat-issue-8), decided by the user to
   be fixed here rather than reproduced.  The name, signature and `(gt, info)`
   return are unchanged and the algorithm is identical — `q.fix_coulomb_gauge`
   is exactly `q.fix_gauge(gf, dirs=[1, 2, 3])`, so the cross-check compares
   against `q.fix_gauge(gf, dirs=[0, 1, 2])`.  The deviation is observable: the
   gauge condition holds for `(0, 1, 2)` and not for `(1, 2, 3)`.
3. **Corrupt files raise instead of warning.**  On a `CHECKSUM`/`CRC32HASH`
   mismatch `qlat` prints a warning and continues; `qlat_ref` raises `IOError`,
   because silently accepting corrupt data defeats the purpose of a reference.
4. **A default `Geometry()` is empty but safe, in both packages.**  It used to
   be a deviation: `q.Geometry().spatial_volume` raised `SIGFPE` (an integer
   divide-by-zero that killed the interpreter) while `qlat_ref` returned `0`.
   qlat now guards its zero-extent divisions (qlat-issue-9, FIXED), so both
   packages give the same empty-but-safe values: `num_node == 0`,
   `is_only_local == False`, `size_node == [0, 0, 0, 0]`, and
   `total_volume`/`spatial_volume`/`local_volume`/`local_volume_expanded` all
   `0`.  `coordinate_from_index`/`g_coordinate_from_index` return the zero
   coordinate and `xg_arr()` has shape `(0, 4)` instead of raising.  This is no
   longer a deviation;
   `qlat-ref/tests/test_qlat_parity.py::TestEmptyGeometryParity` pins the
   agreement against a live qlat.
5. **No MPI.**  `qlat_ref` is single-process: `_node` quantities are global,
   `glb_sum` is an identity in the sense that the local sum is the global sum,
   and there is no broadcast.  The types are thin NumPy wrappers, so there is
   no Cython object identity either.
6. **Unimplemented `qlat` methods are absent, not stubbed.**  The reference
   implements a subset and does not pretend otherwise.
7. **The HMC is silent by default.**  `qlat` prints one progress line per
   trajectory through `displayln_info`; the reference sets the private
   `qlat_ref.hmc_utils._VERBOSE` switch instead.  It is a logging switch only:
   no value and no RNG draw depends on it.

The reasoning, the reproductions and the upstream status of every underlying
`qlat` defect are recorded in `tmp/qlat-ref-reorg/qlat-issues.md`.

## Self-Checks

The suite is 359 standard-library `unittest` tests (295 run + 64 skipped
without a live `qlat`).  It runs fastest under
**pytest + pytest-xdist**, whose settings live in `qlat-ref/pyproject.toml`
(`[tool.pytest.ini_options]`), so the command needs no flags:

```bash
cd qlat-ref
python3 -m pytest                 # 15.2 s, -n auto --maxprocesses=8 --dist loadfile
python3 -m pytest -n 0            #  37.0 s, one interpreter (debugging)
```

pytest and pytest-xdist are a **test extra** (`pip install -e '.[test]'`), never
a runtime dependency: a plain `pip install .` still requires `numpy` only.
`--dist loadfile` puts each test *file* on one worker, which is required
because `tests/test_qlat_parity.py` owns process-global state
(`q.begin_with_mpi()` once per module).

Without pytest the identical suite runs on the standard library alone — the
fallback when the test extra is not installed:

```bash
cd qlat-ref
python3 -m unittest discover -s tests -t . -v     # 37.0 s
```

The suite needs only `numpy`.  The long multi-trajectory statistics are opt-in
and are the one part that still costs a minute:

```bash
cd qlat-ref
QLAT_REF_SLOW=1 python3 -m pytest                 # 62.6 s (116.9 s with -n 0)
QLAT_REF_SLOW=1 python3 -m pytest tests/test_hmc_slow.py -v
QLAT_REF_SLOW=1 python3 -m pytest tests/test_cgfhmc_slow.py -v
```

`python -O` is supported as well: no test depends on a bare `assert`, and
`tests/test_rng.py::TestValidationSurvivesOptimization` re-runs the RNG
validation under `-O` to prove the checks survive.

When a live `qlat` is importable, the two parity modules additionally verify the
names, the signatures and the numerical results against it; otherwise those
tests skip.

## Where to Read What

| Page | Contents |
|---|---|
| [`qlat_types.md`](qlat_types.md) | `Coordinate`, `Geometry`, `ColorMatrix`, the field types and the array bridge |
| [`qlat_gauge.md`](qlat_gauge.md) | Landau and Coulomb gauge fixing |
| [`qlat_observables.md`](qlat_observables.md) | functionals, divergences, Coulomb slices and unitarity |
| [`qlat_rng.md`](qlat_rng.md) | `RngState` and the reference vectors |
| [`qlat_hmc.md`](qlat_hmc.md) | the pure-gauge HMC: Hamiltonian, force, integrator and drivers |
| [`qlat_cgfhmc.md`](qlat_cgfhmc.md) | Canonical Gauge-Fixed HMC: gauge fixing per MD step and the mode-dependent partial momentum refresh |
