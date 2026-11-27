{ stdenv
, buildPythonPackage
, meson-python
, numpy
, pytestCheckHook
, pytest
, pytest-xdist
, qlat-src
, qlat-name ? ""
, version ? "current"
}:

let

  pname = "qlat_ref";

in buildPythonPackage.override { stdenv = stdenv; } {

  pname = pname;
  version = version;
  # The variant suffix must not live in pname: pythonMetadataCheckPhase
  # resolves the installed dist-info by pname, and only the project name
  # matches it.
  name = "qlat_ref${qlat-name}-${version}";

  pyproject = true;

  src = qlat-src.qlat-ref;

  # Pure NumPy package: no C/Cython/MPI, no cuda or cuda-mpi-qlat.sh glue, so
  # the very same derivation is valid for every variant (only the source
  # differs; see qlat-src.qlat-ref in q-pkgs.nix).
  build-system = [
    meson-python
  ];

  # RUNTIME dependency: numpy only.  pytest / pytest-xdist below are
  # check-time inputs and never enter the closure of an environment that
  # installs qlat_ref.
  dependencies = [
    numpy
  ];

  # The tests are a standard-library unittest suite in the source tree (they
  # are not installed into the wheel), run with pytest + pytest-xdist so the
  # nix check exercises the documented runner instead of a second,
  # differently-configured code path: a pytest-only collection failure cannot
  # hide behind a green unittest run.
  #
  # pytestCheckHook appends pytestCheckPhase to preDistPhases; a non-zero
  # pytest exit fails the build.  doInstallCheck is already true by default for
  # python packages, but state it explicitly: the test suite is the point of
  # this derivation.
  nativeCheckInputs = [
    pytestCheckHook
    pytest
    pytest-xdist
  ];

  # The paths and parallel defaults live in [tool.pytest.ini_options] of
  # pyproject.toml (testpaths = ["tests"], addopts = "-n auto --maxprocesses=8
  # --dist loadfile"), so a developer's `cd qlat-ref && python3 -m pytest` needs
  # no flags and the local and nix invocations cannot drift apart.
  #
  # pytest-xdist ships its own nix setup hook that appends
  # `--numprocesses=$NIX_BUILD_CORES` during preInstallCheck.  In the sandbox
  # that is the *host* core count (64 on the machine that built this), which for
  # a 13-file suite means ~51 idle interpreters, and it lands after the
  # pyproject defaults so it would silently override the worker cap.
  # dontUsePytestXdist disables that hook, letting the pyproject settings govern
  # so the nix check is the same command a developer runs.
  dontUsePytestXdist = true;

  # -p no:cacheprovider: never write .pytest_cache into the build directory.
  # -v:                    a failure names the test in the nix log.
  pytestFlags = [
    "-p" "no:cacheprovider"
    "-v"
  ];

  doInstallCheck = true;

  pythonImportsCheck = [ "qlat_ref" ];

  postFixup = ''
    mkdir -pv "$out"/share/version
    echo ${version} >"$out"/share/version/${pname}${qlat-name}
  '';

}
