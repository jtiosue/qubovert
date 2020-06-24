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

"""_values.py.

This file contains QUBO/PUBO and QUSO/PUSO objective function evaluators.

"""

__all__ = (
    'pubo_value', 'qubo_value', 'puso_value', 'quso_value'
)


def pubo_value(x, P):
    r"""pubo_value.

    Find the value of
    :math:`\sum_{i,...,j} P_{i...j} x_{i} ... x_{j}`

    Parameters
    ----------
    x : dict or iterable.
        Maps boolean variable indices to their boolean values, 0 or 1. Ie
        ``x[i]`` must be the boolean value of variable i.
    P : dict, qubovert.utils.PUBOMatrix, or qubovert.PUBO object.
        Maps tuples of boolean variables indices to the P value.

    Returns
    -------
    value : float.
        The value of the PUBO with the given assignment `x`. Ie

    Example
    -------
    >>> P = {(0, 0): 1, (0, 1): -1}
    >>> x = {0: 1, 1: 0}
    >>> pubo_value(x, P)
    1

    """
    return sum(v for k, v in P.items() if all(x[i] for i in k))


def qubo_value(x, Q):
    r"""qubo_value.

    Find the value of
    :math:`\sum_{i,j} Q_{ij} x_{i} x_{j}`

    Parameters
    ----------
    x : dict or iterable.
        Maps boolean variable indices to their boolean values, 0 or 1. Ie
        ``x[i]`` must be the boolean value of variable i.
    Q : dict or qubovert.utils.QUBOMatrix object.
        Maps tuples of boolean variables indices to the Q value.

    Returns
    -------
    value : float.
        The value of the QUBO with the given assignment `x`. Ie

        >>> sum(
                Q[(i, j)] * x[i] * x[j]
                for i in range(n) for j in range(n)
            )

    Example
    -------
    >>> Q = {(0, 0): 1, (0, 1): -1}
    >>> x = {0: 1, 1: 0}
    >>> qubo_value(x, Q)
    1

    """
    return pubo_value(x, Q)


def puso_value(z, H):
    r"""puso_value.

    Find the value of
        :math:`\sum_{i,...,j} H_{i...j} z_{i} ... z_{j}`.

    Parameters
    ----------
    z: dict or iterable.
        Maps variable labels to their values, 1 or -1. Ie z[i] must be the
        value of variable i.
    H : dict, qubovert.utils.PUSOMatrix, or qubovert.PUSO object.
        Maps spin labels to values.

    Returns
    -------
    value : float.
        The value of the PUSO with the given assignment `z`.

    Example
    -------
    >>> H = {(0, 1): -1, (0,): 1}
    >>> z = {0: -1, 1: 1}
    >>> puso_value(z, H)
    0

    """
    return sum(
        v * pow(-1, [z[i] for i in k].count(-1) % 2)
        for k, v in H.items()
    )


def quso_value(z, L):
    r"""quso_value.

    Find the value of
        :math:`\sum_{i,j} J_{ij} z_{i} z_{j} + \sum_{i} h_{i} z_{i}`.

    The J's are encoded by keys with pairs of labels in L, and the h's are
    encoded by keys with a single label in L.

    Parameters
    ----------
    z: dict or iterable.
        Maps variable labels to their values, 1 or -1. Ie z[i] must be the
        value of variable i.
    L : dict, qubovert.utils.QUSOMatrix, or qubovert.QUSO object.
        Maps spin labels to values.

    Returns
    -------
    value : float.
        The value of the QUSO with the given assignment `z`.

    Example
    -------
    >>> L = {(0, 1): -1, (0,): 1}
    >>> z = {0: -1, 1: 1}
    >>> quso_value(z, L)
    0

    """
    return puso_value(z, L)
