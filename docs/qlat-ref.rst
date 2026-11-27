qlat-ref
========

Pure-NumPy reference implementations used to cross-check the C++/Cython code
in `qlat`.  It is a regular packaged Python package (pure Python, only `numpy`
is required, built with `meson-python` like the other packages) and is
imported as `qlat_ref`.

It is installed like the other packages: with nix it is the `qlat_ref`
attribute of `nixpkgs/q-pkgs.nix` and part of the `qlat-jhub-env` /
`qlat-tests` package sets, so it is present in the environment created by
`./nixpkgs/install-py-local-kernel-with-nix.py`; in the `./scripts` flow use
`./scripts/qlat-ref.sh`.

Its tests are a standard-library `unittest` suite in `qlat-ref/tests/` (not
log-comparison examples).  They run fastest under pytest + pytest-xdist, which
is a *test* extra (`pip install -e '.[test]'`) and never a runtime dependency,
so the runtime requirement stays `numpy` alone::

    cd qlat-ref
    python3 -m pytest            # -n auto --maxprocesses=8 --dist loadfile
    python3 -m pytest -n 0       # sequential, one interpreter

`--dist loadfile` keeps every test file on a single worker, which the modules
holding process-global state (`q.begin_with_mpi()` in
`tests/test_qlat_parity.py`) require.

Without the test extra the identical suite runs on the standard library::

    cd qlat-ref
    python3 -m unittest discover -s tests -t . -v

or, from the repository root::

    python3 -m unittest discover -s qlat-ref/tests -t qlat-ref -v

The nix derivation runs the same suite as an install check with
`pytestCheckHook` + pytest-xdist, so a failing test fails the nix build.  The
long multi-trajectory tests are opt-in with `QLAT_REF_SLOW=1`.

.. toctree::
   :maxdepth: 1

   qlat-ref/qlat_ref_overview.md
   qlat-ref/qlat_types.md
   qlat-ref/qlat_gauge.md
   qlat-ref/qlat_observables.md
   qlat-ref/qlat_rng.md
   qlat-ref/qlat_hmc.md
