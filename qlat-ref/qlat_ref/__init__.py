"""
Package ``qlat_ref``
====================\n
Pure-NumPy reference implementations used to cross-check the qlat C++/Cython
code.  The modules are split by topic and mirror qlat's names, signatures and
behaviour:\n
======================  ================================================
module                  contents
======================  ================================================
:mod:`qlat_ref.base`    ``Coordinate``, ``Geometry``, ``NC``
:mod:`qlat_ref.mat`     ``ColorMatrix`` and the matrix algebra
:mod:`qlat_ref.field`   ``FieldColorMatrix``, ``GaugeField``,
                        ``GaugeTransform``, ``GaugeMomentum``
:mod:`qlat_ref.observables`  gauge functionals and divergences
:mod:`qlat_ref.gauge_fixing` Landau and Coulomb gauge fixing
:mod:`qlat_ref.gauge_action` ``GaugeAction`` and the gauge action
:mod:`qlat_ref.rng`     ``RngState``
:mod:`qlat_ref.hmc`     pure-gauge HMC building blocks
:mod:`qlat_ref.hmc_utils` the HMC trajectory drivers
:mod:`qlat_ref.cgfhmc`  Canonical Gauge-Fixed HMC (mode-dependent
                        partial momentum refresh)
======================  ================================================\n
``import qlat_ref`` re-exports the qlat-parity surface of every module, so the
types (``Geometry``, ``Coordinate``, ``ColorMatrix``, ``FieldColorMatrix``,
``GaugeField``, ``GaugeMomentum``, ``GaugeTransform``, ``GaugeAction``), the
gauge-fixing and observable functions, ``RngState`` and the HMC functions are
all reachable as ``qr.<name>``.  :mod:`qlat_ref.gauge` is the stable facade
that keeps the historical gauge-only call style (plain NumPy arrays, ``dirs``
defaults) working.  Every module docstring names its own documentation file.\n
Documentation: ``docs/qlat-ref/qlat_ref_overview.md``\n
.. note:: Update the documentation when updating this source file.
"""

from .base import NC
from .base import Coordinate
from .base import Geometry
from .base import get_dirs
from .base import get_total_site

from .mat import ColorMatrix
from .mat import check_unitarity
from .mat import get_matrix_adjoint
from .mat import get_random_su3_field
from .mat import get_su3_exp
from .mat import get_traceless_antihermitian_part
from .mat import get_unitarity_error
from .mat import get_unitarized

from .rng import RngState

from .field import FieldColorMatrix
from .field import GaugeField
from .field import GaugeMomentum
from .field import GaugeTransform
from .field import apply_gauge_transform
from .field import apply_gauge_transform_to_momentum
from .field import get_array_from_field
from .field import get_field_from_array
from .field import get_local_gauge_gradient
from .field import get_random_gauge_field
from .field import get_random_gauge_transform
from .field import get_staple_sum
from .field import get_unit_gauge_field
from .field import gm_apply_gauge_transformation

# ``gm_apply_gauge_transformation`` is NOT imported again below: ``.field`` is
# its single implementation and ``qlat_ref.hmc`` re-exports the same function
# object, so a second binding here would only suggest two implementations.
from .hmc import gf_evolve
from .hmc import gf_evolve_dual
from .hmc import gf_hamilton_node
from .hmc import gm_hamilton_node
from .hmc import set_gm_force
from .hmc import set_gm_force_dual
from .hmc import set_rand_gauge_momentum

from .hmc_utils import gm_evolve_fg_pure_gauge
from .hmc_utils import metropolis_accept
from .hmc_utils import run_hmc_evolve_pure_gauge
from .hmc_utils import run_hmc_pure_gauge

from .cgfhmc import c_field_from_file
from .cgfhmc import generate_c_field
from .cgfhmc import k_squared
from .cgfhmc import k_vectors
from .cgfhmc import partial_refresh_momentum_mode
from .cgfhmc import regularize_c_field
from .cgfhmc import resolve_k_cut
from .cgfhmc import run_hmc_cgfhmc_pure_gauge

from .observables import get_coulomb_divergence_list
from .observables import get_coulomb_functional_list
from .observables import get_divergence_from_gradient
from .observables import get_gauge_divergence
from .observables import get_gauge_functional
from .observables import get_link_array
from .observables import get_site_array
from .observables import get_unitarity_error_typed

# The compatibility fixers re-export the facade's wrappers (they forward to the
# topic module, so the object is qlat's).  ``get_gauge_functional`` and
# ``get_gauge_divergence`` are qlat's exact signatures here, with ``dirs`` a
# required positional argument, because the top level of ``qlat_ref`` mirrors
# ``qlat``; ``qlat_ref.gauge`` keeps the historical one-argument call style by
# defaulting ``dirs`` to all four directions.
from .gauge import _fix_gauge
from .gauge import fix_coulomb_gauge
from .gauge import fix_gauge
from .gauge import fix_landau_gauge

from .gauge_action import GaugeAction
from .gauge_action import get_gauge_action

from . import base
from . import cgfhmc
from . import field
from . import gauge
from . import gauge_action
from . import gauge_fixing
from . import hmc
from . import hmc_utils
from . import mat
from . import observables
from . import rng

__all__ = [
    "NC",
    "ColorMatrix",
    "Coordinate",
    "FieldColorMatrix",
    "GaugeAction",
    "GaugeField",
    "GaugeMomentum",
    "GaugeTransform",
    "Geometry",
    "RngState",
    "_fix_gauge",
    "apply_gauge_transform",
    "apply_gauge_transform_to_momentum",
    "base",
    "c_field_from_file",
    "cgfhmc",
    "check_unitarity",
    "field",
    "fix_coulomb_gauge",
    "fix_gauge",
    "fix_landau_gauge",
    "gauge",
    "gauge_action",
    "gauge_fixing",
    "generate_c_field",
    "get_array_from_field",
    "get_coulomb_divergence_list",
    "get_coulomb_functional_list",
    "get_dirs",
    "get_divergence_from_gradient",
    "get_field_from_array",
    "get_gauge_action",
    "get_gauge_divergence",
    "get_gauge_functional",
    "get_link_array",
    "get_local_gauge_gradient",
    "get_matrix_adjoint",
    "get_random_gauge_field",
    "get_random_gauge_transform",
    "get_random_su3_field",
    "get_site_array",
    "get_staple_sum",
    "get_su3_exp",
    "get_total_site",
    "get_traceless_antihermitian_part",
    "get_unit_gauge_field",
    "get_unitarity_error",
    "get_unitarity_error_typed",
    "get_unitarized",
    "gf_evolve",
    "gf_evolve_dual",
    "gf_hamilton_node",
    "gm_apply_gauge_transformation",
    "gm_evolve_fg_pure_gauge",
    "gm_hamilton_node",
    "hmc",
    "hmc_utils",
    "k_squared",
    "k_vectors",
    "mat",
    "metropolis_accept",
    "observables",
    "partial_refresh_momentum_mode",
    "regularize_c_field",
    "resolve_k_cut",
    "rng",
    "run_hmc_cgfhmc_pure_gauge",
    "run_hmc_evolve_pure_gauge",
    "run_hmc_pure_gauge",
    "set_gm_force",
    "set_gm_force_dual",
    "set_rand_gauge_momentum",
]
