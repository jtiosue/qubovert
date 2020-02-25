#   Copyright 2020 Joseph T. Iosue
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""_approximate_extrema.py.

This file functions that estimate the min and max of boolean and spin systems.

"""

__all__ = (
    'approximate_pubo_extrema', 'approximate_puso_extrema',
    'approximate_qubo_extrema', 'approximate_quso_extrema'
)


def approximate_pubo_extrema(P):
    """approximate_pubo_extrema.

    Find the approximate minimum and maximum possible values that a PUBO can
    take. This is very approximate! The estimated minimum value that this
    function returns will definitely be less than or equal to the true
    minimum (most likely, it will be much less). Similarly, the estimated
    maximum value that this function returns will definitely be greater than or
    equal to the true maximum (most likely, it will be much more).

    Parameters
    ----------
    P : dict, PUBO, or PUBOMatrix object.
        ``P`` represents a PUBO. Please see ``qubovert.PUBO`` or
        ``qubovert.utils.PUBOMatrix``.

    Return
    ------
    bounds : tuple (min, max).
        The approximate minimum and maximum of the PUBO.

    """
    min_, max_ = 0, 0
    for k, v in P.items():
        if not k:  # offset
            min_ += v
            max_ += v
        elif v < 0:
            min_ += v
        else:
            max_ += v
    return min_, max_


def approximate_puso_extrema(H):
    """approximate_puso_extrema.

    Find the approximate minimum and maximum possible values that a PUSO can
    take. This is very approximate! The estimated minimum value that this
    function returns will definitely be less than or equal to the true
    minimum (most likely, it will be much less). Similarly, the estimated
    maximum value that this function returns will definitely be greater than or
    equal to the true maximum (most likely, it will be much more).

    Parameters
    ----------
    H : dict, PUSO, or PUSOMatrix object.
        ``H`` represents a PUSO. Please see ``qubovert.PUSO`` or
        ``qubovert.utils.PUSOMatrix``.

    Return
    ------
    bounds : tuple (min, max).
        The approximate minimum and maximum of the PUSO.

    """
    min_, max_ = 0, 0
    for k, v in H.items():
        if not k:  # offset
            min_ += v
            max_ += v
        else:
            min_ -= abs(v)
            max_ += abs(v)
    return min_, max_


def approximate_qubo_extrema(Q):
    """approximate_qubo_extrema.

    Find the approximate minimum and maximum possible values that a QUBO can
    take. This is very approximate! The estimated minimum value that this
    function returns will definitely be less than or equal to the true
    minimum (most likely, it will be much less). Similarly, the estimated
    maximum value that this function returns will definitely be greater than or
    equal to the true maximum (most likely, it will be much more).

    Parameters
    ----------
    Q : dict, QUBO, or QUBOMatrix object.
        ``Q`` represents a QUBO. Please see ``qubovert.QUBO`` or
        ``qubovert.utils.QUBOMatrix``.

    Return
    ------
    bounds : tuple (min, max).
        The approximate minimum and maximum of the QUBO.

    """
    return approximate_pubo_extrema(Q)


def approximate_quso_extrema(L):
    """approximate_quso_extrema.

    Find the approximate minimum and maximum possible values that a QUSO can
    take. This is very approximate! The estimated minimum value that this
    function returns will definitely be less than or equal to the true
    minimum (most likely, it will be much less). Similarly, the estimated
    maximum value that this function returns will definitely be greater than or
    equal to the true maximum (most likely, it will be much more).

    Parameters
    ----------
    L : dict, QUSO, or QUSOMatrix object.
        ``L`` represents a QUSO. Please see ``qubovert.QUSO`` or
        ``qubovert.utils.QUSOMatrix``.

    Return
    ------
    bounds : tuple (min, max).
        The approximate minimum and maximum of the QUSO.

    """
    return approximate_puso_extrema(L)
