"""
Backward-compatible entry point of the :mod:`qlat_ref` self-checks.\n
The checks now live in the standard-library ``unittest`` package ``tests/``
next to this package; this module only discovers and runs them, so the
historical command::\n
    cd qlat-ref && python3 -m qlat_ref.tests\n
keeps working.  Prefer running unittest directly::\n
    cd qlat-ref && python3 -m unittest discover -s tests -t . -v\n
The tests are part of the source tree only and are not installed with the
package.
"""

import os
import sys
import unittest

def get_tests_dir():
    """Return the path of the ``tests`` directory of the source tree."""
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return package_dir, os.path.join(package_dir, "tests")

def main(argv=None):
    """Run the unittest suite and return 0 when it passed, 1 otherwise."""
    package_dir, tests_dir = get_tests_dir()
    if not os.path.isdir(tests_dir):
        print(
            f"qlat_ref.tests: the test package is not installed here"
            f" (looked for {tests_dir}); run unittest in the source tree"
            f" instead",
            file=sys.stderr,
        )
        return 1
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=tests_dir, top_level_dir=package_dir, pattern="test_*.py"
    )
    # A suite with no test would pass vacuously; refuse to report success.
    if suite.countTestCases() == 0:
        print(
            f"qlat_ref.tests: no tests found under {tests_dir}",
            file=sys.stderr,
        )
        return 1
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stderr)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main())
