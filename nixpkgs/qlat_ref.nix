{ stdenv
, buildPythonPackage
, meson-python
, numpy
, unittestCheckHook
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

  dependencies = [
    numpy
  ];

  # The tests are a standard-library unittest package in the source tree (not
  # installed into the wheel).  buildPythonPackage has no checkPhase, so
  # unittestCheckHook adds unittestCheckPhase to preDistPhases and runs
  # `python3 -m unittest discover` there; a non-zero unittest exit fails the
  # build.  doInstallCheck is already true by default for python packages, but
  # state it explicitly: the test suite is the point of this derivation.
  nativeCheckInputs = [
    unittestCheckHook
  ];

  unittestFlags = [ "-s" "tests" "-t" "." "-v" ];

  doInstallCheck = true;

  pythonImportsCheck = [ "qlat_ref" ];

  postFixup = ''
    mkdir -pv "$out"/share/version
    echo ${version} >"$out"/share/version/${pname}${qlat-name}
  '';

}
