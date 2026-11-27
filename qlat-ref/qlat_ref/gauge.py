"""
Module ``qlat_ref.gauge``
=========================\n
The stable public facade of ``qlat_ref``: it re-exports the field types, the
observables, the gauge-fixing drivers and the array helpers, and keeps the
historical call style working (a plain NumPy ``(4, Nt, Nx, Ny, Nz, NC, NC)``
array wherever a gauge field is accepted).\n
Documentation: ``docs/qlat-ref/qlat_gauge.md``\n
.. note:: Update the documentation when updating this source file.\n
Layout
------\n
The implementation is split by topic, with no import cycles:\n
======================  ================================================
module                  contents
======================  ================================================
:mod:`qlat_ref.base`    ``Coordinate``, ``Geometry``, ``NC``
:mod:`qlat_ref.mat`     ``ColorMatrix``, matrix algebra, ``unitarize``
:mod:`qlat_ref.field`   ``FieldColorMatrix``, ``GaugeField``,
                        ``GaugeTransform``, ``GaugeMomentum``, array bridge
:mod:`qlat_ref.observables`  functionals, divergences, unitarity checks
:mod:`qlat_ref.gauge_fixing` gauge-fixing drivers
:mod:`qlat_ref.gauge_action` ``GaugeAction``, gauge action value
:mod:`qlat_ref.rng`     ``RngState``
======================  ================================================\n
``qlat_ref.gauge`` itself only aggregates them, so importing it never pulls in
something it does not need, and the modules can be imported individually.\n
Compatibility notes
-------------------\n
- ``dirs`` arguments default to all four directions, so
  ``get_gauge_functional(U)`` keeps working; the qlat-parity functions in the
  topic modules take ``dirs`` positionally like ``qlat`` does.
- ``fix_landau_gauge`` / ``fix_coulomb_gauge`` return ``(gt, info)``, like
  ``qlat``; the internal :func:`_fix_gauge` additionally returns the iterated
  field.
- Both a plain array and a :class:`qlat_ref.field.GaugeField` are accepted
  everywhere a gauge field is expected.
"""

from . import gauge_fixing as _gauge_fixing
from . import observables as _observables

# Re-exported names.  They are listed explicitly in ``__all__`` below, which is
# what keeps them here: the facade's whole purpose is to aggregate the topic
# modules, so these imports are the interface, not unused bindings.
from .base import NC

from .base import Coordinate

from .base import Geometry

from .base import _coordinate_from_index

from .base import _index_from_coordinate

from .base import get_dirs

from .base import get_total_site

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

from .gauge_action import GaugeAction

from .gauge_action import get_gauge_action

from .gauge_fixing import _fix_gauge

from .mat import ColorMatrix

from .mat import check_unitarity

from .mat import get_matrix_adjoint

from .mat import get_random_su3_field

from .mat import get_su3_exp

from .mat import get_traceless_antihermitian_part

from .mat import get_unitarity_error

from .mat import get_unitarized

from .observables import get_coulomb_divergence_list

from .observables import get_coulomb_functional_list

from .observables import get_divergence_from_gradient

from .observables import get_link_array

from .observables import get_site_array

from .observables import get_unitarity_error_typed

from .rng import RngState

_dirs_default = (0, 1, 2, 3)

### -------------------------------------------------------------------

def get_gauge_functional(gf, dirs=_dirs_default):
    """
    Return the normalized gauge functional (see
    :func:`qlat_ref.observables.get_gauge_functional`).\n
    ``dirs`` defaults to all four link directions, so the historical
    ``get_gauge_functional(U)`` call keeps working.
    """
    return _observables.get_gauge_functional(gf, dirs)

def get_gauge_divergence(gf, dirs=_dirs_default):
    """
    Return the RMS lattice divergence (see
    :func:`qlat_ref.observables.get_gauge_divergence`).\n
    ``dirs`` defaults to all four link directions.
    """
    return _observables.get_gauge_divergence(gf, dirs)

def fix_gauge(gf, dirs, max_iter=10000, eps=1.0e-10, alpha=0.1, verbose=False):
    """
    Fix the gauge with an explicit ``dirs`` list (see
    :func:`qlat_ref.gauge_fixing.fix_gauge`).
    """
    return _gauge_fixing.fix_gauge(
        gf, dirs, max_iter=max_iter, eps=eps, alpha=alpha, verbose=verbose
    )

def fix_landau_gauge(gf, max_iter=10000, eps=1.0e-10, alpha=0.1, verbose=False):
    """
    Return ``(gt, info)`` with ``gt * gf`` in Landau gauge (see
    :func:`qlat_ref.gauge_fixing.fix_landau_gauge`).
    """
    return _gauge_fixing.fix_landau_gauge(
        gf, max_iter=max_iter, eps=eps, alpha=alpha, verbose=verbose
    )

def fix_coulomb_gauge(gf, max_iter=10000, eps=1.0e-10, alpha=0.1, verbose=False):
    """
    Return ``(gt, info)`` with ``gt * gf`` in Coulomb gauge (see
    :func:`qlat_ref.gauge_fixing.fix_coulomb_gauge`).
    """
    return _gauge_fixing.fix_coulomb_gauge(
        gf, max_iter=max_iter, eps=eps, alpha=alpha, verbose=verbose
    )

### -------------------------------------------------------------------

# The facade re-exports; every name below is part of the public interface.
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
    "_coordinate_from_index",
    "_fix_gauge",
    "_index_from_coordinate",
    "apply_gauge_transform",
    "apply_gauge_transform_to_momentum",
    "check_unitarity",
    "fix_coulomb_gauge",
    "fix_gauge",
    "fix_landau_gauge",
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
    "gm_apply_gauge_transformation",
]
