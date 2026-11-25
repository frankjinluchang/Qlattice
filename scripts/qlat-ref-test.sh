#!/usr/bin/env bash

# Manual helper (the same role scripts/qlat-test.sh plays): run the `qlat-ref`
# stdlib unittest suite.  No build.*.sh or CI step calls this by default.
#
# `-e -o pipefail` so a failing test makes this script exit non-zero instead of
# being masked by the trailing `| tee`.
set -e -o pipefail

name=qlat-ref-test

source qcore/set-prefix.sh $name

{ time {
    echo "!!!! build $name !!!!"
    source qcore/conf.sh ..

    # `-t "$wd"/qlat-ref` makes `tests` importable as a package and puts
    # `$wd/qlat-ref` on sys.path, so the suite runs against the checkout
    # whether or not qlat-ref is installed in this prefix.
    time-run python3 -m unittest discover -s "$wd"/qlat-ref/tests -t "$wd"/qlat-ref -v

    echo "!!!! $name build !!!!"
    rm -rf "$temp_dir" || true
} } 2>&1 | tee $prefix/log.$name.txt
