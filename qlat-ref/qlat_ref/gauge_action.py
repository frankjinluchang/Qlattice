"""
Module ``qlat_ref.gauge_action``
================================\n
The :class:`GaugeAction` type (the ``beta`` and ``c1`` parameters of a lattice
gauge action) and the reference implementation of the gauge-action value\n
    ``S_gauge = (beta / 3) [ (1 - 8 c1) sum_P (3 - Re Tr U_P)
                            + c1 sum_R (3 - Re Tr U_R) ]``\n
where the sum runs over all oriented plaquettes and ``1 x 2`` rectangles.
This is ``qlat.gf_hamilton_node`` (a positive energy).\n
Documentation: ``docs/qlat-ref/qlat_hmc.md``\n
.. note:: Update the documentation when updating this source file.\n
.. note:: :meth:`GaugeAction.beta` and :meth:`GaugeAction.c1` are **methods**,
   not properties, because that is what ``qlat.gauge_action`` exposes
   (``qlat/qlat/gauge_action.pyx``); calling them follows qlat's API.
"""

import numpy as np

from .mat import get_matrix_adjoint

### -------------------------------------------------------------------

class GaugeAction:
    """
    The parameters of a lattice gauge action.\n
    Mirrors ``qlat.gauge_action.GaugeAction``: ``GaugeAction(beta, c1=0.0)``,
    ``beta()`` and ``c1()`` as methods, and ``@=`` assignment.\n
    .. note:: qlat's C++ default is ``beta=5.5, c1=-0.331``, but its Python
       constructor has no defaults and requires ``beta``; the reference
       mirrors the Python signature.
    """

    __slots__ = ("_beta", "_c1")

    def __init__(self, beta, c1=0.0):
        self._beta = float(beta)
        self._c1 = float(c1)

    def __imatmul__(self, v1):
        self._beta = v1._beta
        self._c1 = v1._c1
        return self

    def __repr__(self):
        return f"GaugeAction(beta={self._beta!r}, c1={self._c1!r})"

    def __eq__(self, other):
        return (
            isinstance(other, GaugeAction)
            and self._beta == other._beta
            and self._c1 == other._c1
        )

    def __ne__(self, other):
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def beta(self):
        """Return the inverse coupling ``beta``."""
        return self._beta

    def c1(self):
        """Return the rectangle improvement coefficient ``c1``."""
        return self._c1

### -------------------------------------------------------------------

def _get_shift(a, mu, n=1):
    """Shift a link-field array ``n`` sites forward in direction ``mu`` (helper)."""
    return np.roll(a, -n, axis=mu)

def _get_plaq(arr, mu, nu):
    """Return the plaquette field ``U_mu(x) U_nu(x+mu) U_mu(x+nu)^dag U_nu(x)^dag``
    (helper)."""
    return (
        arr[mu]
        @ _get_shift(arr[nu], mu)
        @ get_matrix_adjoint(_get_shift(arr[mu], nu))
        @ get_matrix_adjoint(arr[nu])
    )

def _get_rect(arr, mu, nu):
    """
    Return the ``1 x 2`` rectangle field (helper).\n
    ``U_R = U_mu(x) U_mu(x+mu) U_nu(x+2mu) U_mu(x+mu+nu)^dag U_mu(x+nu)^dag
    U_nu(x)^dag``, the oriented rectangle whose long side is ``mu``.
    """
    return (
        arr[mu]
        @ _get_shift(arr[mu], mu)
        @ _get_shift(_get_shift(arr[nu], mu), mu)
        @ get_matrix_adjoint(_get_shift(_get_shift(arr[mu], mu), nu))
        @ get_matrix_adjoint(_get_shift(arr[mu], nu))
        @ get_matrix_adjoint(arr[nu])
    )

def get_gauge_action(gf, ga):
    """
    Return the gauge action value of ``gf`` for the action ``ga``.\n
    This is qlat's ``gf_hamilton_node`` (the gauge part of the Hamiltonian,
    a positive energy),\n
        ``S = (beta / 3) [ (1 - 8 c1) sum_P (3 - Re Tr U_P)
                          + c1 sum_R (3 - Re Tr U_R) ]``\n
    summed over all oriented plaquettes (``6`` per site) and ``1 x 2``
    rectangles (``12`` per site).  ``gf`` may be a
    :class:`qlat_ref.field.GaugeField` or a plain ``(4, ...)`` array.
    """
    from .observables import get_link_array
    #
    arr = get_link_array(gf)
    total_plaq = 0.0
    total_rect = 0.0
    for mu in range(3):
        for nu in range(mu + 1, 4):
            re_tr_plaq = np.trace(_get_plaq(arr, mu, nu), axis1=-2, axis2=-1).real
            total_plaq += float(np.sum(3.0 - re_tr_plaq))
            # both orientations of the 1 x 2 rectangle
            re_tr_rect = np.trace(_get_rect(arr, mu, nu), axis1=-2, axis2=-1).real
            total_rect += float(np.sum(3.0 - re_tr_rect))
            re_tr_rect = np.trace(_get_rect(arr, nu, mu), axis1=-2, axis2=-1).real
            total_rect += float(np.sum(3.0 - re_tr_rect))
    beta = ga.beta()
    c1 = ga.c1()
    return beta / 3.0 * ((1.0 - 8.0 * c1) * total_plaq + c1 * total_rect)
