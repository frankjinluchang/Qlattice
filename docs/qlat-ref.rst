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
log-comparison examples)::

    cd qlat-ref
    python3 -m unittest discover -s tests -t . -v

or, from the repository root::

    python3 -m unittest discover -s qlat-ref/tests -t qlat-ref -v

The nix derivation runs the same suite as an install check, so a failing test
fails the nix build.

.. toctree::
   :maxdepth: 1

   qlat-ref/qlat_ref_overview.md
   qlat-ref/qlat_types.md
   qlat-ref/qlat_gauge.md
   qlat-ref/qlat_observables.md
   qlat-ref/qlat_rng.md
   qlat-ref/qlat_hmc.md
