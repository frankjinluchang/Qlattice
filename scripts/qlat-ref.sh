#!/usr/bin/env bash

# `pipefail` so a failed build step is not hidden by the `| tee`
# below when this script is invoked from build.default*.sh.
set -o pipefail

name=qlat-ref

source qcore/set-prefix.sh $name

{ time {
    echo "!!!! build $name !!!!"
    source qcore/conf.sh ..

    build="$prefix/build"
    mkdir -p "$build"

    cd "$build"

    rm -rfv "$prefix"/build-successfully.txt

    is_fail=false

    # qlat-ref is pure Python (numpy only), so unlike scripts/qlat-utils.sh and
    # scripts/qlat.sh there is no CXX/CFLAGS/LDFLAGS/LIBS glue to set up here.
    time-run meson setup "$wd/qlat-ref" \
        --prefix="$prefix" \
        -Dpython.platlibdir="$prefix/lib/python3/qlat-packages" \
        -Dpython.purelibdir="$prefix/lib/python3/qlat-packages" \
        || is_fail=true

    if $is_fail ; then
        echo "cat meson-logs/meson-log.txt"
        cat meson-logs/meson-log.txt
        exit 1
    fi

    # Unlike scripts/qlat-utils.sh and scripts/qlat.sh, a failed `meson compile`
    # or `meson install` is checked too: without `set -e` the surrounding block
    # would keep going and end on the `touch build-successfully.txt` below, so
    # the failure would be masked (the same masking that `set -o pipefail` above
    # was added to expose).
    if ! time-run meson compile -j$num_proc ; then
        echo "qlat-ref: 'meson compile' failed. Stop."
        exit 1
    fi

    # NOTE: deliberately NO `rm -rf "$prefix"/bin "$prefix"/lib` here.
    # scripts/qlat-utils.sh, scripts/qlat.sh and scripts/qlat-grid.sh do that
    # because each of them reinstalls the whole prefix, but qlat-ref is a
    # pure-Python add-on installed into a prefix that already contains
    # qlat-utils/qlat: removing lib/ would delete every sibling package.
    if ! time-run meson install ; then
        echo "qlat-ref: 'meson install' failed. Stop."
        exit 1
    fi

    mk-setenv.sh
    echo "!!!! $name build !!!!"
    rm -rf "$temp_dir" || true
    touch "$prefix"/build-successfully.txt
} } 2>&1 | tee $prefix/log.$name.txt
