"""
Standard-library ``unittest`` suite of :mod:`qlat_ref`.\n
The package marker is required: ``python3 -m unittest discover -s tests -t .``
fails with ``ImportError: Start directory is not importable`` without it.  The
test modules import the shared helpers as ``from tests.helpers import ...``,
which works both from this directory (``-t .``) and from the repository root
(``-s qlat-ref/tests -t qlat-ref``).
"""
