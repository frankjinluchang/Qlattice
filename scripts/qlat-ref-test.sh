#!/usr/bin/env bash

# Manual helper (the same role scripts/qlat-test.sh plays): run the `qlat-ref`
# test suite.  No build.*.sh or CI step calls this by default.
#
# `-e -o pipefail` so a failing test makes this script exit non-zero instead of
# being masked by the trailing `| tee`.
set -e -o pipefail

name=qlat-ref-test

source qcore/set-prefix.sh $name

{ time {
    echo "!!!! build $name !!!!"
    source qcore/conf.sh ..

    # `python3 -m pytest` is the canonical runner: `[tool.pytest.ini_options]`
    # in qlat-ref/pyproject.toml supplies `testpaths` plus the pytest-xdist
    # defaults (`-n auto --maxprocesses=8 --dist loadfile`), so no flags are
    # needed here.  `--dist loadfile` keeps one test file per worker, which
    # `tests/test_qlat_parity.py` requires because it owns process-global
    # state (`q.begin_with_mpi()` once per module).
    #
    # `python3 -m unittest discover -s tests -t . -v` (run from the qlat-ref
    # directory) remains the equivalent sequential fallback and needs no
    # pytest install at all.
    cd "$wd"/qlat-ref
    time-run python3 -m pytest

    echo "!!!! $name build !!!!"
    rm -rf "$temp_dir" || true
} } 2>&1 | tee $prefix/log.$name.txt
