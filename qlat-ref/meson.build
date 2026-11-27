project('qlat-ref',
  version: '1.21',
  license: 'GPL-3.0-or-later',
  default_options: [
    'libdir=lib',
    ])

py_mod = import('python')
# Pure Python: there are no compiled extensions, so the files go to purelib and
# the wheel is platform independent (py3-none-any).
py3 = py_mod.find_installation('python3', pure: true)
message(py3.full_path())
message('Will directly use python3 command.')
message(py3.get_install_dir())

# Only the package itself is installed.  The ``tests`` directory is part of the
# source tree (and of the sdist) but is deliberately NOT installed, so an
# installed qlat-ref carries no test code.  For the same reason
# ``qlat_ref/tests.py`` -- the historical ``python3 -m qlat_ref.tests`` shim
# that discovers the source-tree suite -- has no entry here: it cannot run
# without ``tests/`` and says so when installed.  The paths must be relative to
# the top of the source tree: this is a single top-level meson.build.
#
# Every module of the package MUST be listed here: meson has no directory glob
# (`fs.glob` does not exist), so a module that is missing from this list is
# silently absent from the wheel and only fails later, at import time, with a
# ``ModuleNotFoundError`` that looks like a packaging bug in the caller.
# ``qlat-ref/tests/test_interface_parity.py`` checks the list against the
# package directory, so a new module cannot be forgotten silently.
py3.install_sources(
  files(
    'qlat_ref/__init__.py',
    'qlat_ref/base.py',
    'qlat_ref/field.py',
    'qlat_ref/gauge.py',
    'qlat_ref/gauge_action.py',
    'qlat_ref/gauge_fixing.py',
    'qlat_ref/hmc.py',
    'qlat_ref/hmc_utils.py',
    'qlat_ref/mat.py',
    'qlat_ref/observables.py',
    'qlat_ref/rng.py',
    ),
  subdir: 'qlat_ref',
  )
