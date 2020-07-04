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

    Find the value of the PUBO for a given assignment of the boolean variables
    ``x``.

    Parameters
    ----------
    x : dict or iterable.
        Maps boolean variable indices to their boolean values, 0 or 1. Ie
        ``x[i]`` must be the boolean value of variable i.
    P : dict, or any object in ``qv.BOOLEAN_MODELS``.
        Maps tuples of boolean variables indices to the P value.

    Returns
    -------
    value : float.
        The value of the PUBO with the given assignment `x`. Ie

    Example
    -------
    >>> P = {(0,): 1, (0, 1): -1}
    >>> x = {0: 1, 1: 0}
    >>> pubo_value(x, P)
    1

    """
    # map is faster than all(x[i] for i in k)
    return sum(v for k, v in P.items() if all(map(lambda i: x[i], k)))


def qubo_value(x, Q):
    r"""qubo_value.

    Find the value of the QUBO for a given assignment of the boolean variables
    ``x``.

    Please note that THIS FUNCTION WILL NOT RAISE AN EXCEPTION IF ``Q`` IS
    NOT A QUBO!! If you input a ``Q`` that is, for example, degree 3 instead
    of degree 2, then this function will return an incorrect value!

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

    Example
    -------
    >>> Q = {(0,): 1, (0, 1): -1}
    >>> x = {0: 1, 1: 0}
    >>> qubo_value(x, Q)
    1

    """
    # we could just return pubo_value(x, Q), but instead let's take
    # advantage of a maximum degree of 2 to not have to loop through keys
    return sum(
        v for k, v in Q.items() if (
            not k or (len(k) == 1 and x[k[0]]) or
            (len(k) == 2 and x[k[0]] and x[k[1]])
        )
    )


def puso_value(z, H):
    r"""puso_value.

    Find the value of the PUSO for a given assignment of the spin variables
    ``z``.

    Parameters
    ----------
    z: dict or iterable.
        Maps variable labels to their values, 1 or -1. Ie z[i] must be the
        value of variable i.
    H : dict, or any object in ``qv.SPIN_MODELS``.
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

    Find the value of the QUSO for a given assignment of the spin variables
    ``z``.

    Please note that THIS FUNCTION WILL NOT RAISE AN EXCEPTION IF ``L`` IS
    NOT A QUSO!! If you input a ``L`` that is, for example, degree 3 instead
    of degree 2, then this function will return an incorrect value!

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
    # we could just return puso_value(z, L), but instead let's take
    # advantage of a maximum degree of 2 to not have to loop through keys
    return sum(
        v * (z[k[0]] if k else 1) * (z[k[1]] if len(k) > 1 else 1)
        for k, v in L.items()
    )
