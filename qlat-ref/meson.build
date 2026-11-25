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
# installed qlat-ref carries no test code.  The paths must be relative to the
# top of the source tree: this is a single top-level meson.build.
py3.install_sources(
  files(
    'qlat_ref/__init__.py',
    'qlat_ref/gauge.py',
    ),
  subdir: 'qlat_ref',
  )
