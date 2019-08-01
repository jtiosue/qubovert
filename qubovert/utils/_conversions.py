#   Copyright 2019 Joseph T. Iosue
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

"""_conversions.py.

This file contains methods to convert to and from QUBO/Ising problems and
variables.

"""

from . import QUBOMatrix, IsingCoupling, IsingField


def qubo_to_ising(Q, offset=0):
    """qubo_to_ising.

    Convert the specified QUBO problem into an Ising problem. Note that
    QUBO {0, 1} values go to Ising {-1, 1} values in that order!

    Parameters
    ----------
    Q : dictionary or qubovert.utils.QUBOMatrix object.
        Maps tuples of binary variables indices to the Q value.
    offset : float (optional, defaults to 0).
             The part of the objective function that does not depend on the
             variables.

    Returns
    ------
    result : tuple (h, J, offset).
        h : qubovert.utils.IsingField object.
            The field of each spin in the Ising formulation.
            h is a IsingField object. For most practical purposes, you can
            use IsingField in he same way as an ordinary dictionary. For
            more information, see ``help(qubovert.utils.IsingField)``.
        J : qubovert.utils.IsingCoupling object.
            The upper triangular coupling matrix, an IsingCoupling object.
            For most practical purposes, you can use IsingCoupling in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.IsingCoupling)``.
        offset : float.
            It is the sum of the terms in the formulation in
            the cited paper that don't involve any variables.

    Example
    -------
    >>> Q = {(0, 0): 1, (0, 1): -1, (1, 1): 3}
    >>> h, J, offset = qubo_to_ising(Q)

    """
    # IsingCoupling deals with keeping J upper triangular, so we don't have to
    # worry about it!
    h, J = IsingField(), IsingCoupling()

    for (i, j), v in Q.items():
        if i != j:
            J[(i, j)] += v / 4
            h[i] += v / 4
            h[j] += v / 4
            offset += v / 4
        else:
            h[i] += v / 2
            offset += v / 2

    return h, J, offset


def ising_to_qubo(h, J, offset=0):
    """ising_to_qubo.

    Convert the specified Ising problem into an upper triangular QUBO problem.
    Note that Ising {-1, 1} values go to QUBO {0, 1} values in that order!

    Parameters
    ----------
    h : dictionary or qubovert.utils.IsingField object.
        Maps spin indices to the field value.
    J : dictionary or qubovert.utils.IsingCoupling object.
        Maps tuples of spin indices to the coupling value. Note
        that J cannot have a key that has a repeated index, ie (1, 1) is an
        invalid key.
    offset : float (optional, defaults to 0).
        The part of the objective function that does not depend on the
        variables.

    Returns
    -------
    result : tuple (Q, offset).
        Q : qubovert.utils.QUBOMatrix object.
            The upper triangular QUBO matrix, a QUBOMatrix object.
            For most practical purposes, you can use QUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUBOMatrix)``.
        offset : float.
            The sum of the terms in the formulation that don't involve any
            variables.

    Example
    -------
    >>> h = {0: 1, 1: -1}
    >>> J = {(0, 1): -1}
    >>> Q, offset = ising_to_qubo(h, J)

    """
    # QUBOMarix deals with keeping ! upper triangular, so we don't have to
    # worry about it!
    Q = QUBOMatrix()

    for (i, j), v in J.items():
        if i == j:
            raise KeyError("J formatted incorrectly, key cannot "
                           "have repeated indices")
        Q[(i, j)] += 4 * v
        Q[(i, i)] -= 2 * v
        Q[(j, j)] -= 2 * v
        offset += v

    for i, v in h.items():
        Q[(i, i)] += 2 * v
        offset -= v

    return Q, offset


def binary_to_spin(x):
    """binary_to_spin.

    Convert a binary number in {0, 1} to a spin in {-1, 1}, in that order.

    Parameters
    ----------
    x : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 0 or 1.

    Returns
    -------
    z : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either -1 or 1.

    Example
    -------
    >>> binary_to_spin(0)  # will print -1
    >>> binary_to_spin(1)  # will print 1
    >>> binary_to_spin([0, 1, 1])  # will print [-1, 1, 1]
    >>> binary_to_spin({"a": 0, "b": 1})  # will print {"a": -1, "b": 1}

    """
    convert = {0: -1, 1: 1}
    if isinstance(x, (int, float)) and x in convert:
        return convert[x]
    elif isinstance(x, dict):
        return {k: convert[v] for k, v in x.items()}
    return type(x)(convert[i] for i in x)


def spin_to_binary(z):
    """spin_to_binary.

    Convert a spin in {-1, 1} to a binary variable in {0, 1}, in that order.

    Parameters
    ----------
    z : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either -1 or 1.

    Returns
    -------
    x : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 0 or 1.

    Example
    -------
    >>> spin_to_binary(-1)  # will print 0
    >>> spin_to_binary(1)  # will print 1
    >>> spin_to_binary([-1, 1, 1])  # will print [0, 1, 1]
    >>> spin_to_binary({"a": -1, "b": 1})  # will print {"a": 0, "b": 1}

    """
    convert = {-1: 0, 1: 1}
    if isinstance(z, (int, float)) and z in convert:
        return convert[z]
    elif isinstance(z, dict):
        return {k: convert[v] for k, v in z.items()}
    return type(z)(convert[i] for i in z)
