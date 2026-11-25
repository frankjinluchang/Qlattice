# `qlat-ref`

Pure-NumPy reference implementations used to cross-check the C++/Cython code
in `qlat`.

The gauge-fixing reference is `qlat_ref.gauge`; its documentation is
[`docs/qlat-ref/qlat_gauge.md`](../docs/qlat-ref/qlat_gauge.md).

```python
import numpy as np
import qlat_ref as qr

U = qr.get_random_gauge_field((2, 2, 2, 2), seed=0)
gt, info = qr.fix_landau_gauge(U)
U_fixed = qr.apply_gauge_transform(U, gt)
print(info["functional"], info["divergence"])
```

`fix_landau_gauge` and `fix_coulomb_gauge` return the accumulated gauge
transformation `gt` (a site field of shape `(Nt, Nx, Ny, Nz, 3, 3)`), not the
gauge-fixed field: apply it with `apply_gauge_transform` as above.  `gt` is not
unique (it depends on the fixing path, and a global center element multiplies
it without changing the fixed field), so compare fixed *fields*, not
transformations.

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
the source tree and of the sdist, but are not installed with the package):

```bash
cd qlat-ref
python3 -m unittest discover -s tests -t . -v
```

From the repository root:

```bash
python3 -m unittest discover -s qlat-ref/tests -t qlat-ref -v
```

The historical `python3 -m qlat_ref.tests` entry point still works as a thin
shim around the same suite.  The tests use tiny lattices and finish in a few
seconds.

`tests/` layout:

| File | Contents |
|---|---|
| `tests/__init__.py` | package marker (needed by `discover -t .`), no tests |
| `tests/helpers.py` | shared deterministic field/momentum helpers |
| `tests/test_gauge_fixing.py` | Landau/Coulomb convergence, gauge condition, covariance under a global transform, step-size independence, non-uniqueness, and that the returned transform reproduces the fixed field |
| `tests/test_momentum.py` | adjoint momentum action, `is_dagger` inverse, composition order, Lie-algebra/invariant preservation, commutation with the fixing |
| `tests/test_invariants.py` | unitarity and determinism of the random fields, trivially fixed configurations, array layout/shift conventions |

The nix derivation runs the same suite as an install check
(`unittestCheckHook`), so a failing test fails the nix build; `./scripts/qlat-ref-test.sh`
runs it in the `./scripts` flow.
