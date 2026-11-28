"""
Interface-parity tests: the public :mod:`qlat_ref` API against a live ``qlat``.\n
These tests are the **cheap** half of the qlat comparison (the numerical half
is :mod:`tests.test_qlat_parity`): they only look at names and signatures, so
they run in milliseconds and never touch a lattice or MPI.  They are skipped as
a whole when ``qlat`` is not importable, because the pure-NumPy package (and
its nix derivation, which depends on ``numpy`` only) must not need it.\n
The contract they enforce is the one documented in
``docs/qlat-ref/qlat_ref_overview.md``:\n
1. **Shared names are qlat's.**  Every ``qlat_ref`` function that has a ``qlat``
   counterpart exposes the same *name*, the same parameter names, the same
   parameter kinds and the same defaults (compared through
   ``inspect.signature``).  ``inspect`` works on qlat's Cython *functions*;
   it does not work on its classes, so the types are compared by an explicit
   ``hasattr`` table of methods and attributes instead.
2. **The subset rule.**  A name in ``qlat_ref.__all__`` is either a shared name
   (checked by 1.) or a registered extra in :data:`EXTRAS` with a one-line
   justification.  An unregistered new name fails these tests, and an extra
   that turns out to exist in ``qlat`` after all is rejected too: if the
   functionality is shared, the name must be shared.
3. **Module-level subsets.**  The topic modules that mirror a ``qlat`` module
   (``hmc``, ``hmc_utils``, ``gauge_fixing``, ``gauge_action``) may only define
   functions whose names and signatures ``qlat``'s module also defines.  This is
   what stops a plausible-looking ``gm_hamilton``/``gm_evolve`` invention from
   creeping back in.
4. **Deviations are declared.**  The intentional behavioural differences are
   listed in :data:`DEVIATIONS` and pinned here (the Coulomb direction set and
   the extra ``info`` key are checked from the ``qlat_ref`` side, where they are
   observable without a live lattice).\n
Run it with the qlat environment sourced::\n
    source result-py-local/bin/setenv-qlat.sh
    cd qlat-ref && python3 -m unittest tests.test_interface_parity -v
"""

import importlib
import inspect
import os
import re
import unittest

import qlat_ref as qr

_REQUIRED_QLAT_NAMES = (
    "GaugeAction",
    "GaugeField",
    "GaugeMomentum",
    "GaugeTransform",
    "Geometry",
    "RngState",
    "fix_coulomb_gauge",
    "fix_gauge",
    "fix_landau_gauge",
    "get_gauge_divergence",
    "get_gauge_functional",
    "get_unitarity_error",
    "gf_evolve",
    "gm_hamilton_node",
    "run_hmc_pure_gauge",
)

try:
    import qlat as q

    _missing = [n for n in _REQUIRED_QLAT_NAMES if not hasattr(q, n)]
    HAS_QLAT = not _missing
    _NO_QLAT_REASON = (
        f"the installed qlat lacks {_missing}"
        if _missing
        else "a live qlat build is required for the interface-parity tests"
    )
except Exception as exc:  # noqa: BLE001 - any import failure means "no qlat here"
    q = None
    HAS_QLAT = False
    _NO_QLAT_REASON = f"qlat is not importable ({exc})"

### -------------------------------------------------------------------
### The declared contract
### -------------------------------------------------------------------

# Shared function names: qlat_ref's exported name -> qlat's name.  They are the
# same string everywhere, and the mapping exists so a future rename is one line.
SHARED_FUNCTIONS = {
    "c_field_from_file": "c_field_from_file",
    "fix_coulomb_gauge": "fix_coulomb_gauge",
    "fix_gauge": "fix_gauge",
    "fix_landau_gauge": "fix_landau_gauge",
    "get_gauge_divergence": "get_gauge_divergence",
    "get_gauge_functional": "get_gauge_functional",
    "get_unitarity_error": "get_unitarity_error",
    "gf_evolve": "gf_evolve",
    "gf_evolve_dual": "gf_evolve_dual",
    "generate_c_field": "generate_c_field",
    "gf_hamilton_node": "gf_hamilton_node",
    "gm_apply_gauge_transformation": "gm_apply_gauge_transformation",
    "gm_evolve_fg_pure_gauge": "gm_evolve_fg_pure_gauge",
    "gm_hamilton_node": "gm_hamilton_node",
    "k_squared": "k_squared",
    "k_vectors": "k_vectors",
    "metropolis_accept": "metropolis_accept",
    "partial_refresh_momentum_mode": "partial_refresh_momentum_mode",
    "regularize_c_field": "regularize_c_field",
    "resolve_k_cut": "resolve_k_cut",
    "run_hmc_cgfhmc_pure_gauge": "run_hmc_cgfhmc_pure_gauge",
    "run_hmc_evolve_pure_gauge": "run_hmc_evolve_pure_gauge",
    "run_hmc_pure_gauge": "run_hmc_pure_gauge",
    "set_gm_force": "set_gm_force",
    "set_gm_force_dual": "set_gm_force_dual",
    "set_rand_gauge_momentum": "set_rand_gauge_momentum",
}

# Public names with no ``qlat`` counterpart, with the reason they are exported.
# ``qlat``'s Python API deliberately has no equivalent of these (they are the
# array bridge, the deterministic fixtures and the matrix helpers of the
# reference implementation), so exporting them is an addition, not an invented
# alias for something qlat already has.
EXTRAS = {
    "NC": "the colour dimension, a module constant of the reference",
    "_fix_gauge": "internal fixing core that also returns the iterated field",
    "apply_gauge_transform": "array bridge: apply gt to a plain link array",
    "apply_gauge_transform_to_momentum": "array bridge: adjoint action on a momentum array",
    "check_unitarity": "array bridge: unitarity predicate on any matrix field",
    "get_array_from_field": "array bridge: field -> (4, ...) NumPy array",
    "get_coulomb_divergence_list": "per-time-slice convergence monitor",
    "get_coulomb_functional_list": "per-time-slice Coulomb functional",
    "get_dirs": "direction-set helper used by the fixing defaults",
    "get_divergence_from_gradient": "divergence of a gradient field, used by the monitors",
    "get_field_from_array": "array bridge: (4, ...) NumPy array -> field",
    "get_gauge_action": "the action value behind qlat's gf_hamilton_node",
    "get_link_array": "array bridge: link field of any type -> (4, ...) array",
    "get_local_gauge_gradient": "the local gradient D(x) of the fixing",
    "get_matrix_adjoint": "matrix helper used by the reference formulas",
    "get_random_gauge_field": "deterministic fixture (qlat has no such constructor)",
    "get_random_gauge_transform": "deterministic fixture",
    "get_random_su3_field": "deterministic fixture",
    "get_site_array": "array bridge: site field of any type -> (Nt, ...) array",
    "get_staple_sum": "the staple sum K(x) of the fixing",
    "get_su3_exp": "the matrix exponential of the fixing and of gf_evolve",
    "get_total_site": "shape helper used by the fixtures",
    "get_traceless_antihermitian_part": "Lie-algebra projection helper",
    "get_unit_gauge_field": "deterministic fixture",
    "get_unitarity_error_typed": "the strict, qlat-identical type contract",
    "get_unitarized": "Gram-Schmidt unitarization of gf_evolve",
}

# The topic modules are exported so that ``qr.hmc.set_gm_force`` (say) works the
# way ``q.hmc.set_gm_force`` does.  A module name may or may not exist at qlat's
# top level (``qlat.hmc`` and ``qlat.mat`` do, ``qlat.rng`` does not), so the
# modules are classified separately from the function extras.
TOPIC_MODULES = {
    "base": "topic split of the reference package",
    "cgfhmc": "mirrors qlat.cgfhmc",
    "field": "topic split of the reference package",
    "gauge": "backward-compatible facade (kept permanently, see AGENTS.md)",
    "gauge_action": "mirrors qlat.gauge_action",
    "gauge_fixing": "mirrors qlat.gauge_fixing",
    "hmc": "mirrors qlat.hmc",
    "hmc_utils": "mirrors qlat.hmc_utils",
    "mat": "mirrors qlat.mat",
    "observables": "topic split of the reference package",
    "rng": "mirrors qlat_utils.rng_state",
}

# Functions a topic module may define that its qlat counterpart does not.
# Every entry needs a reason; an unlisted definition fails the subset test.
MODULE_EXTRAS = {
    ("gauge_action", "get_gauge_action"): (
        "the action value itself; qlat only exposes it through gf_hamilton_node,"
        " which qlat_ref provides as the qlat-named wrapper"
    ),
}

# Type -> the methods and attributes both packages must expose.  ``inspect``
# cannot read the signature of a Cython class (or of its ``__init__``), so the
# types are compared by presence; the method *signatures* are compared wherever
# both sides are inspectable (see ``TestMethodSignatures``).
TYPE_INTERFACES = {
    "Geometry": {
        "methods": (
            "coordinate_from_index",
            "copy",
            "coordinate_g_from_l",
            "coordinate_l_from_g",
            "g_coordinate_from_index",
            "index_from_coordinate",
            "index_from_g_coordinate",
            "is_local",
            "is_local_xg",
            "show",
            "xg_arr",
        ),
        "attributes": (
            "coor_node",
            "eo",
            "expansion_left",
            "expansion_right",
            "id_node",
            "is_only_local",
            "local_site",
            "local_volume",
            "local_volume_expanded",
            "node_site",
            "num_node",
            "size_node",
            "spatial_volume",
            "total_site",
            "total_volume",
        ),
    },
    "Coordinate": {
        "methods": (
            "copy",
            "from_index",
            "from_list",
            "r_sqr",
            "spatial_volume",
            "sqr",
            "to_index",
            "to_list",
            "to_numpy",
            "to_tuple",
            "volume",
        ),
        "attributes": (),
    },
    "ColorMatrix": {
        "methods": ("adjoint", "conjugate", "copy", "qnorm", "set_zero", "transpose"),
        "attributes": (),
        "dunders": ("__getitem__", "__iadd__", "__imul__", "__isub__", "__mul__"),
    },
    "FieldColorMatrix": {
        "methods": (
            "copy",
            "glb_sum",
            "init_from_geo",
            "qnorm",
            "qnorm_field",
            "set_rand",
            "set_rand_g",
            "set_unit",
            "set_zero",
            "swap",
        ),
        "attributes": ("geo", "multiplicity", "n_sites", "sizeof_m", "total_site"),
    },
    "GaugeField": {
        "methods": (
            "copy",
            "glb_sum",
            "init_from_geo",
            "link_trace",
            "load",
            "plaq",
            "qnorm",
            "qnorm_field",
            "save",
            "set_rand",
            "set_rand_g",
            "set_unit",
            "set_zero",
            "su_projection",
            "swap",
            "twist_boundary_at_boundary",
            "unitarize",
        ),
        "attributes": ("geo", "multiplicity", "n_sites", "sizeof_m", "total_site"),
    },
    "GaugeTransform": {
        "methods": (
            "copy",
            "glb_sum",
            "init_from_geo",
            "inv",
            "load",
            "qnorm",
            "qnorm_field",
            "save",
            "set_rand",
            "set_rand_g",
            "set_unit",
            "set_zero",
            "swap",
            "unitarize",
        ),
        "attributes": ("geo", "multiplicity"),
        "dunders": ("__mul__",),
    },
    "GaugeMomentum": {
        "methods": (
            "copy",
            "glb_sum",
            "init_from_geo",
            "qnorm",
            "qnorm_field",
            "set_rand",
            "set_rand_fa",
            "set_rand_g",
            "set_unit",
            "set_zero",
            "swap",
        ),
        "attributes": ("geo", "multiplicity"),
        "dunders": ("__rmul__",),
    },
    "GaugeAction": {
        "methods": ("beta", "c1"),
        "attributes": (),
        "dunders": ("__imatmul__",),
    },
    "RngState": {
        "methods": (
            "c_rand_gen",
            "copy",
            "g_rand_arr",
            "g_rand_gen",
            "get_state",
            "rand_arr",
            "rand_gen",
            "select",
            "set_state",
            "split",
            "u_rand_arr",
            "u_rand_gen",
        ),
        "attributes": (),
    },
}

# Public members a qlat_ref type has that qlat's does not, with the reason.
TYPE_EXTRAS = {
    ("RngState", "newtype"): "C++ RngState::newtype is not exposed by qlat's binding",
    ("RngState", "reset"): "the C++ RngState::reset is not exposed by qlat's binding",
    (
        "RngState",
        "split_inplace",
    ): "the in-place split the constructor uses; no qlat binding",
    (
        "FieldColorMatrix",
        "show_geo",
    ): "a __repr__ helper of the reference, not part of qlat's API",
}

# Intentional behavioural deviations, name -> reason.  A key must be a name that
# exists in both packages: a deviation is about *behaviour*, never about a
# missing or invented interface.
DEVIATIONS = {
    "fix_gauge": (
        "qlat_ref's info dict is a superset: it also records 'dirs', the link"
        " directions actually fixed.  Every qlat key is present."
    ),
    "fix_coulomb_gauge": (
        "qlat_ref fixes the CORRECT spatial directions (0, 1, 2); qlat uses"
        " {1, 2, 3}, i.e. three directions that include time.  Component 3 is"
        " time throughout qlat's own convention, so (0, 1, 2) is the spatial set"
        " (qlat-issue-8, fixed here by decision).  Name, signature and the"
        " (gt, info) return are unchanged, and q.fix_gauge(gf, dirs=[1, 2, 3])"
        " is the same algorithm for numerical cross-checks."
    ),
    "get_unitarity_error": (
        "qlat_ref's array-level version also accepts a ColorMatrix, a plain"
        " NumPy link/site array and any field of matrices; qlat accepts only a"
        " GaugeField or a GaugeTransform.  The strict qlat-identical contract is"
        " observables.get_unitarity_error_typed."
    ),
    "RngState": (
        "the shared surface is identical; qlat_ref adds newtype/reset/"
        "split_inplace (see TYPE_EXTRAS) and no MPI broadcast"
    ),
    "Geometry": "single process: node quantities are the global ones, no MPI",
    "metropolis_accept": "single process: no broadcast of flag/accept_prob",
    "run_hmc_pure_gauge": "single process: no glb_sum_double of delta_h",
    "regularize_c_field": (
        "same name, signature, clipping and evenness projection; qlat_ref's"
        " last_report adds is_even next to qlat's diagnostics keys, and its"
        " error messages carry a 'qlat_ref:' prefix"
    ),
    "generate_c_field": (
        "same name, signature, always-regularized output and last_report keys;"
        " the report also records total_site (qlat's does not)"
    ),
    "partial_refresh_momentum_mode": (
        "same name, signature, (applied, dK, info) return, c=0 no-op with no"
        " RNG draw, uniform fast path and Fourier path.  qlat_ref validates the"
        " c(k) field itself (range/evenness/finiteness) so its info adds"
        " is_even, max_asymmetry and n_site; every qlat key is present with the"
        " same value.  Single process instead of a distributed FFT."
    ),
    "run_hmc_cgfhmc_pure_gauge": (
        "same name, signature, defaults, in-place-on-accept semantics, dK"
        " accounting and the identical NotImplementedError text for"
        " is_reverse_test; single process instead of MPI, and no verbose line"
        " unless qlat_ref.hmc_utils._VERBOSE is set"
    ),
}

### -------------------------------------------------------------------
### Helpers
### -------------------------------------------------------------------

def get_signature(fn):
    """Return the comparable view of a signature, or ``None`` if unavailable."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    return [(p.name, p.kind.name, p.default) for p in sig.parameters.values()]

def get_public_defined(module):
    """Return the public functions and classes *defined in* ``module``."""
    out = {}
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            out[name] = obj
    return out

def find_meson_build():
    """Return the path of ``qlat-ref/meson.build``, or ``None``."""
    candidates = []
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(qr.__file__)))
    candidates.append(os.path.join(package_parent, "meson.build"))
    candidates.append(os.path.join(os.getcwd(), "meson.build"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

def get_installed_sources():
    """Return the ``qlat_ref/*.py`` files meson.build installs (sorted).\n
    ``meson.build`` is meson's own language, not Python, so the ``files(...)``
    list is read with a regex over the quoted paths.  A commented-out entry is
    still an entry: the test only needs to detect a *missing* module, and a
    stale comment cannot hide one.
    """
    path = find_meson_build()
    if path is None:
        return None
    with open(path, encoding="utf-8") as fp:
        text = fp.read()
    pattern = re.compile(r"['\"](qlat_ref/[^'\"]+\.py)['\"]")
    return {match.group(1)[len("qlat_ref/") :] for match in pattern.finditer(text)}

def get_package_sources():
    """Return the ``*.py`` files of the package directory (sorted)."""
    package_dir = os.path.dirname(os.path.abspath(qr.__file__))
    return {
        name
        for name in os.listdir(package_dir)
        if name.endswith(".py") and not name.startswith(".")
    }

### -------------------------------------------------------------------
### Tests that need only numpy (they run in the nix install check too)
### -------------------------------------------------------------------

class TestPublicSurface(unittest.TestCase):
    """``qlat_ref.__all__`` must resolve and be exactly the declared surface."""

    def test_every_public_name_imports(self):
        missing = [name for name in qr.__all__ if not hasattr(qr, name)]
        self.assertEqual(
            missing, [], f"qlat_ref.__all__ names that do not import: {missing}"
        )

    def test_all_is_sorted_and_unique(self):
        names = list(qr.__all__)
        self.assertEqual(len(names), len(set(names)), "duplicate names in __all__")
        rest = [name for name in names if name != "NC"]
        self.assertEqual(rest, sorted(rest), "__all__ is not sorted")

    def test_no_undeclared_public_names(self):
        declared = (
            set(SHARED_FUNCTIONS)
            | set(EXTRAS)
            | set(TOPIC_MODULES)
            | set(TYPE_INTERFACES)
        )
        undeclared = sorted(set(qr.__all__) - declared)
        self.assertEqual(
            undeclared,
            [],
            "public names must be registered in SHARED_FUNCTIONS or EXTRAS:"
            f" {undeclared}",
        )

    def test_every_topic_module_imports(self):
        for name in sorted(TOPIC_MODULES):
            with self.subTest(name=name):
                self.assertTrue(inspect.ismodule(getattr(qr, name)))
                self.assertTrue(TOPIC_MODULES[name].strip())

    def test_every_extra_is_justified(self):
        for name in sorted(EXTRAS):
            with self.subTest(name=name):
                self.assertTrue(
                    EXTRAS[name].strip(),
                    f"the extra {name!r} needs a one-line justification",
                )

    def test_module_extras_are_justified(self):
        for key, reason in MODULE_EXTRAS.items():
            with self.subTest(key=str(key)):
                self.assertTrue(
                    reason.strip(), f"the module extra {key!r} needs a reason"
                )

    def test_measured_deviations_are_declared(self):
        for name in sorted(DEVIATIONS):
            with self.subTest(name=name):
                self.assertTrue(
                    DEVIATIONS[name].strip(), f"the deviation {name!r} needs a reason"
                )
                self.assertTrue(
                    hasattr(qr, name), f"the deviation {name!r} is not a qlat_ref name"
                )

class TestGaugeFacadeBackwardCompatibility(unittest.TestCase):
    """``qlat_ref.gauge`` keeps the historical gauge-only names working."""

    HISTORICAL_NAMES = (
        "NC",
        "apply_gauge_transform",
        "apply_gauge_transform_to_momentum",
        "check_unitarity",
        "fix_coulomb_gauge",
        "fix_gauge",
        "fix_landau_gauge",
        "get_coulomb_divergence_list",
        "get_coulomb_functional_list",
        "get_dirs",
        "get_divergence_from_gradient",
        "get_gauge_divergence",
        "get_gauge_functional",
        "get_local_gauge_gradient",
        "get_matrix_adjoint",
        "get_random_gauge_field",
        "get_random_gauge_transform",
        "get_staple_sum",
        "get_su3_exp",
        "get_traceless_antihermitian_part",
        "get_unit_gauge_field",
        "get_unitarity_error",
    )

    def test_historical_names_still_exist(self):
        missing = [n for n in self.HISTORICAL_NAMES if not hasattr(qr.gauge, n)]
        self.assertEqual(missing, [], f"qlat_ref.gauge lost {missing}")

    def test_the_one_argument_accessors_still_work(self):
        # The facade is the one place where `dirs` defaults, so the historical
        # qr.get_gauge_functional(U) / qr.get_gauge_divergence(U) calls keep
        # working.  The top level keeps qlat's exact signatures instead.
        U = qr.get_random_gauge_field((2, 2, 2, 2), seed=0)
        # The one-argument call must equal the explicit all-directions call,
        # and the two accessors must agree with their topic-module versions.
        self.assertEqual(
            qr.gauge.get_gauge_functional(U),
            qr.observables.get_gauge_functional(U, (0, 1, 2, 3)),
        )
        self.assertEqual(
            qr.gauge.get_gauge_divergence(U),
            qr.observables.get_gauge_divergence(U, (0, 1, 2, 3)),
        )
        self.assertGreater(qr.gauge.get_gauge_divergence(U), 1e-2)
        # ... and the top level requires `dirs`, like qlat.
        self.assertEqual(len(inspect.signature(qr.get_gauge_functional).parameters), 2)
        self.assertEqual(
            len(inspect.signature(qr.gauge.get_gauge_functional).parameters), 2
        )
        self.assertTrue(
            inspect.signature(qr.gauge.get_gauge_functional).parameters["dirs"].default
            == (0, 1, 2, 3)
        )

class TestDeclaredBehaviourWithoutQlat(unittest.TestCase):
    """The deviations that are observable from the reference side alone."""

    def test_coulomb_gauge_fixes_the_spatial_directions(self):
        U = qr.get_random_gauge_field((2, 2, 2, 2), seed=1)
        gt, info = qr.fix_coulomb_gauge(U, max_iter=2000, eps=1e-10, alpha=0.1)
        self.assertEqual(tuple(info["dirs"]), (0, 1, 2))
        self.assertTrue(info["converged"])
        fixed = qr.apply_gauge_transform(U, gt)
        # The gauge condition holds for the spatial set (0, 1, 2) only; the
        # divergence over qlat's (1, 2, 3) is O(1), which is what makes the
        # deviation observable rather than cosmetic.
        self.assertLessEqual(qr.get_gauge_divergence(fixed, (0, 1, 2)), info["eps"])
        self.assertGreater(qr.get_gauge_divergence(fixed, (1, 2, 3)), 1e-2)

    def test_landau_info_is_a_superset_of_qlat_keys(self):
        # The qlat keys are the ones the qlat side of the cross-check reads; the
        # one addition is 'dirs'.
        qlat_info_keys = {
            "gauge",
            "dirs",
            "functional",
            "functional_list",
            "divergence",
            "divergence_list",
            "converged",
            "n_iter",
            "eps",
            "max_iter",
            "alpha",
        }
        U = qr.get_random_gauge_field((2, 2, 2, 2), seed=2)
        _, info = qr.fix_landau_gauge(U, max_iter=200, eps=1e-10, alpha=0.1)
        self.assertTrue(qlat_info_keys <= set(info), sorted(qlat_info_keys - set(info)))
        self.assertIn("dirs", DEVIATIONS["fix_gauge"])

    def test_unitarity_error_accepts_arrays_and_the_typed_entry_point_rejects_them(
        self,
    ):
        U = qr.get_random_gauge_field((2, 2, 2, 2), seed=3)
        gf = qr.field.get_field_from_array(U, qr.Geometry(qr.Coordinate((2, 2, 2, 2))))
        # The exported get_unitarity_error is the array bridge: any field works.
        self.assertLessEqual(qr.get_unitarity_error(U), 1e-12)
        self.assertLessEqual(qr.get_unitarity_error(gf), 1e-12)
        # The typed entry point has qlat's exact (strict) type contract.
        self.assertLessEqual(qr.observables.get_unitarity_error_typed(gf), 1e-12)
        with self.assertRaises(TypeError):
            qr.observables.get_unitarity_error_typed(U)
        self.assertIn("get_unitarity_error", DEVIATIONS)

class TestPackaging(unittest.TestCase):
    """Every module of the package must be in ``meson.build``."""

    def test_every_module_is_installed(self):
        installed = get_installed_sources()
        if installed is None:
            self.skipTest("meson.build is not available (installed package)")
        package_sources = get_package_sources()
        # ``tests.py`` is the historical shim: it discovers the source-tree
        # ``tests/`` directory, which is not installed, so it is deliberately
        # not part of the wheel.
        package_sources.discard("tests.py")
        missing = sorted(package_sources - installed)
        stale = sorted(installed - package_sources)
        self.assertEqual(
            missing,
            [],
            "these modules are missing from qlat-ref/meson.build, so the"
            f" installed package would fail to import them: {missing}",
        )
        self.assertEqual(
            stale, [], f"meson.build lists files that do not exist: {stale}"
        )

    def test_the_tests_shim_is_deliberately_not_installed(self):
        installed = get_installed_sources()
        if installed is None:
            self.skipTest("meson.build is not available (installed package)")
        self.assertNotIn("tests.py", installed)
        self.assertTrue(
            os.path.isfile(os.path.join(os.path.dirname(qr.__file__), "tests.py"))
        )

### -------------------------------------------------------------------
### Tests that need a live qlat
### -------------------------------------------------------------------

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestSharedFunctionSignatures(unittest.TestCase):
    """Every shared function must have qlat's exact positional signature."""

    def test_shared_functions_exist_in_qlat(self):
        missing = [n for n in SHARED_FUNCTIONS if not hasattr(q, n)]
        self.assertEqual(missing, [], f"the installed qlat lacks {missing}")

    def test_shared_function_signatures_match(self):
        mismatches = []
        for ref_name, qlat_name in SHARED_FUNCTIONS.items():
            with self.subTest(name=ref_name):
                ref_sig = get_signature(getattr(qr, ref_name))
                qlat_sig = get_signature(getattr(q, qlat_name))
                self.assertIsNotNone(qlat_sig, f"qlat.{qlat_name} has no signature")
                self.assertIsNotNone(ref_sig, f"qlat_ref.{ref_name} has no signature")
                if ref_sig != qlat_sig:
                    mismatches.append((ref_name, qlat_sig, ref_sig))
        self.assertEqual(
            mismatches,
            [],
            "signature mismatches (name, qlat, qlat_ref):" + repr(mismatches),
        )

    def test_shared_function_parameter_names_match(self):
        # The parameter *names* are part of qlat's interface (keyword calls and
        # error messages depend on them), so they are asserted separately from
        # kinds/defaults: this is the check that would have caught the old
        # `U` / `gf` divergence.
        for ref_name, qlat_name in SHARED_FUNCTIONS.items():
            with self.subTest(name=ref_name):
                ref_sig = inspect.signature(getattr(qr, ref_name))
                qlat_sig = inspect.signature(getattr(q, qlat_name))
                self.assertEqual(list(ref_sig.parameters), list(qlat_sig.parameters))

    def test_no_extra_shared_functions_are_hidden(self):
        # A function that exists in qlat at the top level *and* in qlat_ref must
        # be declared in SHARED_FUNCTIONS; being in EXTRAS is a contradiction
        # (we would claim a name has no counterpart when it has one).
        contradictions = []
        for name in EXTRAS:
            if hasattr(q, name):
                contradictions.append(name)
        self.assertEqual(
            contradictions,
            [],
            "these names exist in qlat, so they belong in SHARED_FUNCTIONS,"
            f" not EXTRAS: {contradictions}",
        )

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestModuleSubsets(unittest.TestCase):
    """A mirrored topic module may only define what qlat's module defines."""

    MIRRORED = ("hmc", "hmc_utils", "cgfhmc", "gauge_fixing", "gauge_action")

    def test_mirrored_modules_exist(self):
        missing = [name for name in self.MIRRORED if not hasattr(q, name)]
        self.assertEqual(missing, [], f"the installed qlat lacks the modules {missing}")

    def test_module_definitions_are_a_subset(self):
        for module_name in self.MIRRORED:
            with self.subTest(module=module_name):
                ref_module = importlib.import_module(f"qlat_ref.{module_name}")
                qlat_module = getattr(q, module_name)
                ref_names = set(get_public_defined(ref_module))
                qlat_names = {n for n in dir(qlat_module) if not n.startswith("_")}
                extras = {
                    name
                    for name in ref_names - qlat_names
                    if (module_name, name) not in MODULE_EXTRAS
                }
                self.assertEqual(
                    extras,
                    set(),
                    f"qlat_ref.{module_name} defines names qlat.{module_name} does"
                    f" not have (register them in MODULE_EXTRAS if intended):"
                    f" {sorted(extras)}",
                )

    def test_module_function_signatures_match(self):
        mismatches = []
        for module_name in self.MIRRORED:
            ref_module = importlib.import_module(f"qlat_ref.{module_name}")
            qlat_module = getattr(q, module_name)
            for name, ref_obj in sorted(get_public_defined(ref_module).items()):
                if not inspect.isfunction(ref_obj) or not hasattr(qlat_module, name):
                    continue
                qlat_obj = getattr(qlat_module, name)
                ref_sig = get_signature(ref_obj)
                qlat_sig = get_signature(qlat_obj)
                if ref_sig is not None and qlat_sig is not None and ref_sig != qlat_sig:
                    mismatches.append((f"{module_name}.{name}", qlat_sig, ref_sig))
        self.assertEqual(
            mismatches,
            [],
            "module function signature mismatches (name, qlat, qlat_ref):"
            + repr(mismatches),
        )

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestTypeInterfaces(unittest.TestCase):
    """The shared types must expose the same methods and attributes."""

    def test_shared_types_exist(self):
        missing = [name for name in TYPE_INTERFACES if not hasattr(q, name)]
        self.assertEqual(missing, [], f"the installed qlat lacks the types {missing}")

    def test_type_members_exist_on_both_sides(self):
        problems = []
        for type_name, table in TYPE_INTERFACES.items():
            ref_type = getattr(qr, type_name)
            qlat_type = getattr(q, type_name)
            for kind in ("methods", "attributes", "dunders"):
                for member in table.get(kind, ()):
                    if not hasattr(qlat_type, member):
                        problems.append(f"qlat.{type_name}.{member} is missing")
                    if not hasattr(ref_type, member):
                        problems.append(f"qlat_ref.{type_name}.{member} is missing")
        self.assertEqual(problems, [], "member mismatches: " + repr(problems))

    def test_ref_only_type_members_are_declared(self):
        undeclared = []
        for type_name in TYPE_INTERFACES:
            ref_type = getattr(qr, type_name)
            qlat_type = getattr(q, type_name)
            for name in dir(ref_type):
                if name.startswith("_"):
                    continue
                if hasattr(qlat_type, name):
                    continue
                if (type_name, name) in TYPE_EXTRAS:
                    continue
                if self.is_inherited_declared_extra(name, qlat_type):
                    continue
                undeclared.append(f"{type_name}.{name}")
        self.assertEqual(
            undeclared,
            [],
            "qlat_ref members with no qlat counterpart must be registered in"
            f" TYPE_EXTRAS: {undeclared}",
        )

    @staticmethod
    def is_inherited_declared_extra(name, qlat_type):
        """Whether ``name`` is declared for a qlat_ref base class instead."""
        return (
            name in ("show_geo",)
            and isinstance(qlat_type, type)
            and not hasattr(qlat_type, "show_geo")
            and ("FieldColorMatrix", name) in TYPE_EXTRAS
        )

    def test_declared_type_extras_really_are_extras(self):
        for type_name, member in TYPE_EXTRAS:
            with self.subTest(member=f"{type_name}.{member}"):
                self.assertTrue(hasattr(getattr(qr, type_name), member))
                self.assertFalse(hasattr(getattr(q, type_name), member))

    def test_method_signatures_match_where_inspectable(self):
        mismatches = []
        for type_name, table in TYPE_INTERFACES.items():
            ref_type = getattr(qr, type_name)
            qlat_type = getattr(q, type_name)
            for member in table.get("methods", ()):
                ref_sig = get_signature(getattr(ref_type, member))
                qlat_sig = get_signature(getattr(qlat_type, member))
                if ref_sig is None or qlat_sig is None:
                    continue
                if ref_sig != qlat_sig:
                    mismatches.append((f"{type_name}.{member}", qlat_sig, ref_sig))
        self.assertEqual(
            mismatches,
            [],
            "method signature mismatches (name, qlat, qlat_ref):" + repr(mismatches),
        )

@unittest.skipUnless(HAS_QLAT, _NO_QLAT_REASON)
class TestDeviationsAgainstLiveQlat(unittest.TestCase):
    """The declared deviations are exactly the ones a live qlat exhibits."""

    def test_every_deviation_is_a_shared_name(self):
        not_shared = sorted(
            set(DEVIATIONS) - set(SHARED_FUNCTIONS) - set(TYPE_INTERFACES)
        )
        self.assertEqual(
            not_shared,
            [],
            f"a deviation must be about a name both packages have: {not_shared}",
        )

    def test_qlat_coulomb_uses_a_different_direction_set(self):
        # qlat's `fix_coulomb_gauge` is `fix_gauge(dirs=[1, 2, 3])`, i.e. it is
        # the same algorithm on a different direction set -- the documented
        # deviation is a direction choice, not a different implementation.  The
        # signatures already match (TestSharedFunctionSignatures), so the
        # direction set is the only difference the cross-check has to bridge.
        self.assertIn("fix_coulomb_gauge", DEVIATIONS)
        self.assertIn(qr.base._SPATIAL_DIRS, ((0, 1, 2),))
        self.assertNotEqual(tuple(qr.base._SPATIAL_DIRS), (1, 2, 3))

    def test_gauge_action_accessors_are_methods_on_both_sides(self):
        self.assertTrue(callable(q.GaugeAction(5.5, 0.1).beta))
        self.assertTrue(callable(qr.GaugeAction(5.5, 0.1).beta))
        self.assertEqual(
            q.GaugeAction(5.5, 0.1).beta(), qr.GaugeAction(5.5, 0.1).beta()
        )
        self.assertEqual(q.GaugeAction(5.5, 0.1).c1(), qr.GaugeAction(5.5, 0.1).c1())

    def test_rng_shared_surface_behaves_identically(self):
        self.assertEqual(
            [q.RngState("parity-seed").rand_gen() for _ in range(4)],
            [qr.RngState("parity-seed").rand_gen() for _ in range(4)],
        )
        self.assertEqual(
            q.RngState("parity-split").split("a").rand_gen(),
            qr.RngState("parity-split").split("a").rand_gen(),
        )
