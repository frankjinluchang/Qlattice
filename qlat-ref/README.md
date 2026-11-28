# `qlat-ref`

Pure-NumPy reference implementations used to cross-check the C++/Cython code
in `qlat`.  Every type, function name, signature and default the reference
shares with `qlat` is `qlat`'s, so a reader can diff the two side by side.

```python
import qlat_ref as qr

U = qr.get_random_gauge_field((2, 2, 2, 2), seed=0)
gt, info = qr.fix_landau_gauge(U)
U_fixed = qr.apply_gauge_transform(U, gt)
print(info["functional"], info["divergence"])
```

The package covers the lattice types (`Coordinate`, `Geometry`, `ColorMatrix`,
`FieldColorMatrix`, `GaugeField`, `GaugeTransform`, `GaugeMomentum`,
`GaugeAction`), Landau and Coulomb gauge fixing, the gauge functionals and
divergences that monitor them, the gauge action, the RNG and a pure-gauge HMC
(the Hamiltonian, the force, the Omelyan force-gradient integrator and the
trajectory driver).

## Documentation

The pages live in [`docs/qlat-ref/`](../docs/qlat-ref/):

| Page | Contents |
|---|---|
| [`qlat_ref_overview.md`](../docs/qlat-ref/qlat_ref_overview.md) | the package as a whole: module layout, the array bridge, the compatibility contract and the deviations from `qlat` |
| [`qlat_types.md`](../docs/qlat-ref/qlat_types.md) | `Coordinate`, `Geometry`, `ColorMatrix`, the field types and the array bridge |
| [`qlat_gauge.md`](../docs/qlat-ref/qlat_gauge.md) | Landau and Coulomb gauge fixing |
| [`qlat_observables.md`](../docs/qlat-ref/qlat_observables.md) | functionals, divergences, Coulomb slices and unitarity |
| [`qlat_rng.md`](../docs/qlat-ref/qlat_rng.md) | `RngState` and the reference vectors |
| [`qlat_hmc.md`](../docs/qlat-ref/qlat_hmc.md) | the pure-gauge HMC |

## Module Layout

The implementation is split by topic, in dependency order:
`base` → `mat` → `rng` → `field` → `observables` → `gauge_fixing` →
`gauge_action` → `hmc` → `hmc_utils`.  `qlat_ref.gauge` is the stable
gauge-only facade and `qlat_ref/__init__.py` re-exports the public names of
every module, so both of these work:

```python
import qlat_ref as qr

U = qr.get_random_gauge_field((2, 2, 2, 2), seed=0)
print(qr.hmc.gm_hamilton_node(qr.GaugeMomentum(qr.Geometry(qr.Coordinate((2, 2, 2, 2))))))
print(qr.gauge.get_gauge_functional(U))     # facade: dirs defaults to all four
```

The top level keeps `qlat`'s exact signatures, so
`qr.get_gauge_functional(U)` is a `TypeError` (qlat requires `dirs`); the
facade is the one place that defaults it.

## Gauge Fixing

`fix_landau_gauge` and `fix_coulomb_gauge` return the accumulated gauge
transformation `gt` (a site field of shape `(Nt, Nx, Ny, Nz, 3, 3)`), not the
gauge-fixed field: apply it with `apply_gauge_transform` as above.  `gt` is not
unique (it depends on the fixing path, and a global center element multiplies
it without changing the fixed field), so compare fixed *fields*, not
transformations.

Coulomb gauge fixes the **spatial** directions `(0, 1, 2)`; `qlat`'s
`fix_coulomb_gauge` uses `(1, 2, 3)`, which includes time.  This is a
deliberate deviation (qlat-issue-8): component 3 is time throughout qlat's own
convention.  `q.fix_gauge(gf, dirs=[0, 1, 2])` is the same algorithm, and
[`examples-py/gauge-fixing-cross-check.py`](../examples-py/gauge-fixing-cross-check.py)
compares against that.

## Fields, Arrays and Momenta

A field wraps one C-contiguous NumPy buffer `(local_volume, multiplicity, 3,
3)`; `np.asarray(field)` is a **zero-copy view** of it, so a plain array and a
field are interchangeable at every function boundary.  `np.array(field)` is a
copy; the HMC drivers mutate their field in place, so take `gf.copy()` before
a trajectory if the old configuration is needed.

A momentum field is a Lie-algebra valued site field `P` of shape
`(4, Nt, Nx, Ny, Nz, 3, 3)` that lives on one site, so a gauge transformation
acts on it without the link shift:

```python
P = np.zeros((4,) + U.shape[1:], dtype=np.complex128)  # or any momentum field
P_fixed = qr.apply_gauge_transform_to_momentum(P, gt)
```

The action is the adjoint one, `P_mu(x) -> g(x) P_mu(x) g(x)^dagger` (with
`is_dagger=True` the inverse `g(x)^dagger P_mu(x) g(x)`); the inputs are not
modified and a new array is returned.

## A Trajectory

```python
import qlat_ref as qr

geo = qr.Geometry(qr.Coordinate((2, 2, 2, 2)))
gf = qr.GaugeField(geo)
gf.set_unit()
ga = qr.GaugeAction(5.5, 0.0)
flag, delta_h = qr.run_hmc_pure_gauge(gf, ga, 0, qr.RngState("traj-seed"))
```

`run_hmc_pure_gauge` mutates `gf` in place on accept and returns
`(flag, delta_h)`.  The RNG tree is qlat's: the driver seed is split by the
trajectory number and then into independent momentum and accept streams, so a
rejected trajectory does not disturb the next one's momentum.

The package is pure Python (only `numpy` is required), single process, single
thread, and vectorized over all lattice sites.

## Installation

`qlat-ref` is a regular packaged Python package (built with `meson-python`),
installed like the other packages:

```bash
# nix (it is part of the qlat-jhub-env / qlat-tests package sets)
./nixpkgs/install-py-local-kernel-with-nix.py
source result-py-local/bin/setenv-qlat.sh

# ./scripts flow
./scripts/qlat-ref.sh

# or directly with pip
pip install ./qlat-ref
```

After that `import qlat_ref` works without any `PYTHONPATH` setup.  If you do
not want to install it, adding the `qlat-ref` directory to `PYTHONPATH` (or
running from inside it) still works as a no-install fallback.

## Tests

The tests are a standard-library `unittest` suite in `tests/` (they are part of
the source tree and of the sdist, but are not installed with the package).  Two
runners are supported and run **the same 283 tests**; only the speed differs.

**pytest (parallel, the fast runner).**  The runner settings live in
`[tool.pytest.ini_options]` of `pyproject.toml`, so the plain command needs no
flags:

```bash
cd qlat-ref
python3 -m pytest                     # -n auto --maxprocesses=8 --dist loadfile
python3 -m pytest -n 0                # sequential, in one interpreter
python3 -m pytest -n 8                # explicit worker count
```

pytest and pytest-xdist are a **test extra**, never a runtime dependency:

```bash
pip install -e '.[test]'              # or: pip install pytest pytest-xdist
```

A runtime install (`pip install .`) still pulls in `numpy` only.

The defaults are `-n auto --maxprocesses=8 --dist loadfile`:

- `--dist loadfile` sends every test **file** to exactly one worker.  This is a
  correctness requirement, not a tuning choice: `tests/test_qlat_parity.py` owns
  process-global state (`q.begin_with_mpi()` once per module), so all of a
  file's tests must share one interpreter.  Do not switch to `--dist load` or
  `loadscope` without re-checking the `setUpModule` hooks.
- `--maxprocesses=8` caps the worker count.  The suite is 13 files, so beyond
  ~13 workers only startup cost grows; measured `-n 4`, `-n 8`, `-n 13` and
  `-n auto` all land at ~14.7 s on this machine, so the cap is free and keeps a
  64-core host from spawning 64 interpreters.

**unittest (no extra dependencies).**  Still fully supported — this is the
fallback when pytest is not installed, and the historical entry point:

```bash
python3 -m unittest discover -s tests -t . -v
```

From the repository root:

```bash
python3 -m unittest discover -s qlat-ref/tests -t qlat-ref -v
```

The historical `python3 -m qlat_ref.tests` entry point still works as a thin
shim around the same suite.

**Timings** (16 test files, 359 tests = 295 run + 64 skipped without a live
`qlat`; 8 workers on this 64-core host):

| Invocation | Wall time |
|---|---|
| `pytest` (parallel, default) | **15.2 s** |
| `pytest -n 0` (sequential) | 37.0 s |
| `unittest discover` (sequential) | 37.0 s |
| `pytest` with `QLAT_REF_SLOW=1` | 62.6 s |
| `pytest -n 0` with `QLAT_REF_SLOW=1` | 116.9 s |
| `python -O -m pytest` | 15.9 s |

The speedup is bounded by `tests/test_hmc_driver.py`, which alone takes 13.2 s
and is a single file, so it cannot be split across workers: the parallel run can
never beat that critical path.  `--dist loadfile` is what keeps the suite safe
rather than maximally fast.

11 tests in the two opt-in slow modules are skipped unless `QLAT_REF_SLOW=1` is
set (5 in `tests/test_hmc_slow.py`, 6 in `tests/test_cgfhmc_slow.py`):

```bash
QLAT_REF_SLOW=1 python3 -m pytest tests/test_hmc_slow.py tests/test_cgfhmc_slow.py -v
QLAT_REF_SLOW=1 python3 -m unittest tests.test_hmc_slow tests.test_cgfhmc_slow -v
```

The parity tests against a live `qlat` skip themselves when `qlat` is not
importable, so the suite needs only `numpy`:

| File | Contents |
|---|---|
| `tests/__init__.py` | package marker (needed by `discover -t .`), no tests |
| `tests/helpers.py` | shared deterministic field/momentum helpers |
| `tests/test_types.py` | construction, buffer layout, the array bridge, `set_rand` determinism, the `*` algebra, `.lat` round trips, the typed unitarity contract and pickling |
| `tests/test_gauge_fixing.py` | Landau/Coulomb convergence, gauge condition, covariance under a global transform, step-size independence, non-uniqueness, and that the returned transform reproduces the fixed field |
| `tests/test_momentum.py` | adjoint momentum action, `is_dagger` inverse, composition order, Lie-algebra/invariant preservation, commutation with the fixing |
| `tests/test_invariants.py` | unitarity and determinism of the random fields, trivially fixed configurations, array layout/shift conventions |
| `tests/test_rng.py` | `RngState` bit-for-bit against the C++ vectors, the split tree, and the `newtype`/`split_inplace` extensions |
| `tests/test_hmc_momentum.py` | `set_rand_gauge_momentum` against the C++ vectors, the Gaussian distribution and the split layout |
| `tests/test_hmc_action.py` | `gf_hamilton_node` (plaquette and rectangle), the unit-field zero, `gm_hamilton_node` |
| `tests/test_hmc_force.py` | `set_gm_force`/`set_gm_force_dual` by finite differences, including the wrong-form negative controls |
| `tests/test_hmc_integrator.py` | `gf_evolve`/`gf_evolve_dual`, the `c1` rectangle term, unitarity after evolution |
| `tests/test_hmc_driver.py` | `metropolis_accept`, `gm_evolve_fg_pure_gauge`, `run_hmc_evolve_pure_gauge`, `run_hmc_pure_gauge`, reversibility and RNG-tree reproducibility |
| `tests/test_hmc_slow.py` | long multi-trajectory statistics (**opt-in**: `QLAT_REF_SLOW=1`) |
| `tests/test_cgfhmc.py` | the CGFHMC `c(k)` grid and its folding, the regularizer, the partial momentum refresh laws and the driver's `dK` accounting, against independently computed values and the golden chains in `tests/data/cgfhmc-vectors.json` |
| `tests/test_cgfhmc_slow.py` | CGFHMC ensemble stationarity, chain health, and the field/refresh invariants repeated on a larger lattice (**opt-in**: `QLAT_REF_SLOW=1`) |
| `tests/test_qlat_parity.py` | numerical parity against a live `qlat` (skips without it) |
| `tests/test_interface_parity.py` | names, signatures and type members against a live `qlat`, plus the subset rule and the declared extras/deviations (skips without it) |

`tests/data/` holds the reference vectors (`rng-vectors.json`,
`newtype-vectors.json`, `hmc-vectors.json`, `cgfhmc-vectors.json`) and the
scripts that regenerate them from the C++ implementation (`make-*.py`,
`newtype-probe.cpp`).  `cgfhmc-vectors.json` records full CGFHMC chains
(flags, `delta_h`, `dK_sum`, fixing iteration counts and plaquettes) written by
the experimental implementation on a live `qlat`; the post-trajectory fields are
checked in `tests/test_cgfhmc.py` as well.

The nix derivation runs the same suite as an install check with
`pytestCheckHook` + pytest-xdist (the same runner, so a pytest-only collection
failure cannot hide behind a green unittest run), so a failing test fails the
nix build; `./scripts/qlat-ref-test.sh` runs it in the `./scripts` flow.

`python -O` is supported too: no test relies on a bare `assert`, and
`tests/test_rng.py::TestValidationSurvivesOptimization` re-runs the RNG
validation under `-O` to prove it.
