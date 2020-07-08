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

"""_conversions.py.

This file contains methods to convert to and from QUBO/QUSO/PUBO/PUSO
problems, as well as various misc converters

"""

from . import QUBOMatrix, QUSOMatrix, PUBOMatrix, PUSOMatrix
# for QUBO, QUSO, PUBO, PUSO, can't import directly because it will cause
# circular imports, so instead just import qubovert.
import qubovert as qv


__all__ = (
    'boolean_to_spin', 'spin_to_boolean',
    'decimal_to_spin', 'spin_to_decimal',
    'decimal_to_boolean', 'boolean_to_decimal',
    'qubo_to_quso', 'quso_to_qubo', 'pubo_to_puso', 'puso_to_pubo',
    'Conversions'
)


def boolean_to_spin(x):
    """boolean_to_spin.

    Convert a boolean number in {0, 1} to a spin in {1, -1}, in that order.

    Parameters
    ----------
    x : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 0 or 1.

    Returns
    -------
    z : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 1 or -1.

    Example
    -------
    >>> boolean_to_spin(0)  # will print 1
    >>> boolean_to_spin(1)  # will print -1
    >>> boolean_to_spin([0, 1, 1])  # will print [1, -1, -1]
    >>> boolean_to_spin({"a": 0, "b": 1})  # will print {"a": 1, "b": -1}

    """
    convert = {0: 1, 1: -1}
    if isinstance(x, (int, float)) and x in convert:
        return convert[x]
    elif isinstance(x, dict):
        return {k: convert[v] for k, v in x.items()}
    return type(x)(convert[i] for i in x)


def spin_to_boolean(z):
    """spin_to_boolean.

    Convert a spin in {1, -1} to a boolean variable in {0, 1}, in that order.

    Parameters
    ----------
    z : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 1 or -1.

    Returns
    -------
    x : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 0 or 1.

    Example
    -------
    >>> spin_to_boolean(-1)  # will print 1
    >>> spin_to_boolean(1)  # will print 0
    >>> spin_to_boolean([-1, 1, 1])  # will print [1, 0, 0]
    >>> spin_to_boolean({"a": -1, "b": 1})  # will print {"a": 1, "b": 0}

    """
    convert = {-1: 1, 1: 0}
    if isinstance(z, (int, float)) and z in convert:
        return convert[z]
    elif isinstance(z, dict):
        return {k: convert[v] for k, v in z.items()}
    return type(z)(convert[i] for i in z)


def decimal_to_boolean(d, num_bits=None):
    """decimal_to_boolean.

    Convert the integer ``d`` to its boolean representation.

    Parameters
    ----------
    d : int >= 0.
        Number to convert to binary.
    num_bits : int >= 0 (optional, defaults to None).
        Number of bits in the representation. If ``num_bits is None``, then
        the minimum number of bits required will be used.

    Returns
    -------
    b : tuple of length ``num_bits``.
        Each element of ``b`` is a 0 or 1.

    Example
    -------
    >>> decimal_to_boolean(10, 7)
    (0, 0, 0, 1, 0, 1, 0)

    >>> decimal_to_boolean(10)
    (1, 0, 1, 0)

    """
    if int(d) != d or d < 0:
        raise ValueError("``d`` must be an integer >- 0.")
    b = bin(d)[2:]
    lb = len(b)
    if num_bits is None:
        num_bits = lb
    elif num_bits < lb:
        raise ValueError("Not enough bits to represent the number.")
    return (0,) * (num_bits - lb) + tuple(int(x) for x in b)


def boolean_to_decimal(b):
    """boolean_to_decimal.

    Convert a bit string to its decimal form.

    Parameters
    ----------
    b : tuple or list of 0s and 1s.
        The binary bit string.

    Returns
    -------
    d : int.

    Examples
    --------
    >>> boolean_to_decimal((1, 1, 0))
    6

    """
    return int("".join(str(x) for x in b), base=2) if b else 0


def decimal_to_spin(d, num_spins=None):
    """decimal_to_spin.

    Convert the integer ``d`` to its spin representation (ie its binary
    representation, but with 1 and -1 instead of 0 and 1).

    Parameters
    ----------
    d : int >= 0.
        Number to convert to binary.
    num_spins : int >= 0 (optional, defaults to None).
        Number of bits in the representation. If ``num_spins is None``, then
        the minimum number of bits required will be used.

    Returns
    -------
    b : tuple of length ``num_spins``.
        Each element of ``b`` is a 0 or 1.

    Example
    -------
    >>> decimal_to_spin(10, 7)
    (1, 1, 1, -1, 1, -1, 1)

    >>> decimal_to_spin(10)
    (-1, 1, -1, 1)

    """
    return boolean_to_spin(decimal_to_boolean(d, num_spins))


def spin_to_decimal(b):
    """spin_to_decimal.

    Convert a spin string to its decimal form.

    Parameters
    ----------
    b : tuple or list of 1s and -1s.
        The spin bit string.

    Returns
    -------
    d : int.

    Examples
    --------
    >>> spin_to_decimal((-1, -1, 1))
    6

    """
    return boolean_to_decimal(spin_to_boolean(b))


def qubo_to_quso(Q):
    """qubo_to_quso.

    Convert the specified QUBO problem into an QUSO problem. Note that
    QUBO {0, 1} values go to QUSO {1, -1} values in that order!

    Parameters
    ----------
    Q : dictionary, qubovert.QUBO, or qubovert.utils.QUBOMatrix object.
        Maps tuples of boolean variables indices to the Q value. See
        ``help(qubovert.QUBO)`` and ``help(qubovert.utils.QUBOMatrix)`` for
        info on formatting.

    Returns
    -------
    L : qubovert.QUSO or qubovert.utils.QUSOMatrix object.
        tuple of spin labels map to QUSO values. If ``Q`` is a
        ``qubovert.utils.QUBOMatrix`` object, then ``L`` will be a
        ``qubovert.utils.QUSOMatrix``, otherwise ``L`` will be a
        ``qubovert.QUSO`` object.

    Example
    -------
    >>> Q = {(0,): 1, (0, 1): -1, (1,): 3}
    >>> L = qubo_to_quso(Q)
    >>> isinstance(L, qubovert.utils.QUSOMatrix)
    True

    >>> Q = {('a',): 1, ('a', 'b'): -1, ('b',): 3}
    >>> L = qubo_to_quso(Q)
    >>> isinstance(L, qubovert.QUSO)
    True

    """
    # could just use QUSOMatrix(pubo_to_puso(Q)), but then we spend a lot of
    # time converting from a PUSOMatrix to QUSOMatrix, so instead we
    # explictly write out the conversion.

    # not isinstance! because isinstance(QUBO, QUBOMatrix) is True
    if type(Q) in (QUBOMatrix, qv.QUBO):
        # key will already be squashed
        def squash_key(k): return k
    else:
        squash_key = qv.QUBO.squash_key

    L = QUSOMatrix() if type(Q) == QUBOMatrix else qv.QUSO()

    for kp, v in Q.items():
        k = squash_key(kp)
        if not k:
            L[k] += v
        elif len(k) == 1:
            L[k] -= v / 2
            L[()] += v / 2
        else:
            # len(k) must be 2 because of squash_key
            i, j = k
            L[k] += v / 4
            L[(i,)] -= v / 4
            L[(j,)] -= v / 4
            L[()] += v / 4

    return L


def quso_to_qubo(L):
    """quso_to_qubo.

    Convert the specified QUSO problem into an upper triangular QUBO problem.
    Note that QUSO {1, -1} values go to QUBO {0, 1} values in that order!

    Parameters
    ----------
    L : dictionary, qubovert.QUSO, or qubovert.utils.QUSOMatrix object.
        Tuple of spin labels map to QUSO values. See
        ``help(qubovert.QUSO)`` and ``help(qubovert.utils.QUSOMatrix)`` for
        info on formatting.


    Returns
    -------
    Q : qubovert.QUBO or qubovert.utils.QUBOMatrix object.
        If ``L`` is a ``qubovert.utils.QUSOMatrix`` object, then ``Q`` will be
        a ``qubovert.utils.QUBOMatrix``, otherwise ``Q`` will be a
        ``qubovert.QUBO`` object. See ``help(qubovert.QUBO)`` and
        ``help(qubovert.utils.QUBOMatrix)`` for info on formatting.

    Example
    -------
    >>> L = {(0,): 1, (1,): -1, (0, 1): -1}
    >>> Q = quso_to_qubo(L)
    >>> isinstance(Q, qubovert.utils.QUBOMatrix)
    True

    >>> L = {('a',): 1, ('b',): -1, (0, 1): -1}
    >>> Q = quso_to_qubo(L)
    >>> isinstance(Q, qubovert.QUBO)
    True

    """
    # could just use QUBOMatrix(puso_to_pubo(L)), but then we spend a lot of
    # time converting from a PUBOMatrix to QUBOMatrix, so instead we explictly
    # write out the conversion.

    # not isinstance! because isinstance(QUSO, QUSOMatrix) is True
    if type(L) in (QUSOMatrix, qv.QUSO):
        # key will already be squashed
        def squash_key(k): return k
    else:
        squash_key = qv.QUSO.squash_key

    Q = QUBOMatrix() if type(L) == QUSOMatrix else qv.QUBO()

    for kp, v in L.items():
        k = squash_key(kp)
        if not k:
            Q[k] += v
        elif len(k) == 1:
            Q[k] -= 2 * v
            Q[()] += v
        else:
            # len(k) must be 2 because of squash_key
            i, j = k
            Q[k] += 4 * v
            Q[(i,)] -= 2 * v
            Q[(j,)] -= 2 * v
            Q[()] += v

    return Q


def pubo_to_puso(P):
    """pubo_to_puso.

    Convert the specified PUBO problem into an PUSO problem. Note that
    PUBO {0, 1} values go to PUSO {1, -1} values in that order!

    Parameters
    ----------
    P : dictionary, qubovert.PUBO, or qubovert.utils.PUBOMatrix object.
        Maps tuples of boolean variables indices to the P value. See
        ``help(qubovert.PUBO)`` and ``help(qubovert.utils.PUBOMatrix)`` for
        info on formatting.

    Returns
    -------
    H : qubovert.utils.PUSOMatrix object or qubovert.PUSO object.
        tuple of spin labels map to PUSO values. If ``P`` is a
        ``qubovert.utils.PUBOMatrix`` object, then ``H`` will be a
        ``qubovert.utils.PUSOMatrix``, otherwise ``H`` will be a
        ``qubovert.PUSO`` object..

    Example
    -------
    >>> P = {(0,): 1, (0, 1): -1, (1,): 3}
    >>> H = pubo_to_puso(P)
    >>> isinstance(H, qubovert.utils.PUSOMatrix)
    True

    >>> P = {(0,): 1, (0, 1): -1, (1,): 3}
    >>> H = pubo_to_puso(P)
    >>> isinstance(H, qubovert.PUSO)
    True

    """
    def generate_new_key_value(k):
        """generate_new_key_value.

        Recursively generate the PUBO key, value pairs for converting the
        product ``x[k[0]] * ... * x[k[-1]]``, where each ``x`` is a boolean
        variable in {0, 1}, to the product
        ``(1-z[k[0]])/2 * ... * (1-z[k[1]])/2``., where each ``z`` is a spin
        in {1, -1}.

        Parameters
        ----------
        k : tuple.
            Each element of the tuple corresponds to a boolean label.

        Yields
        ------
        res : tuple (key, value)
            key : tuple.
                Each element of the tuple corresponds to a spin label.
            value : float.
                The value to multiply the value corresponding with ``k`` by.

        """
        if not k:
            yield k, 1
        else:
            for key, value in generate_new_key_value(k[1:]):
                yield (k[0],) + key, -value / 2
                yield key, value / 2

    # not isinstance! because isinstance(PUBO, PUBOMatrix) is True
    H = PUSOMatrix() if type(P) == PUBOMatrix else qv.PUSO()

    for k, v in P.items():
        for key, value in generate_new_key_value(k):
            H[key] += value * v

    return H


def puso_to_pubo(H):
    """puso_to_pubo.

    Convert the specified PUSO problem into an upper triangular PUBO problem.
    Note that PUSO {1, -1} values go to PUBO {0, 1} values in that order!

    Parameters
    ----------
    H : dictionary or qubovert.utils.PUSOMatrix object.
        Tuple of spin labels map to PUSO values. See
        ``help(qubovert.PUSO)`` and ``help(qubovert.utils.PUSOMatrix)`` for
        info on formatting.

    Returns
    -------
    P : qubovert.utils.PUBOMatrix object or qubovert.PUBO object.
        If ``H`` is a ``qubovert.utils.PUSOMatrix`` object, then ``P`` will
        be a ``qubovert.utils.PUBOMatrix``, otherwise ``P`` will be a
        ``qubovert.PUBO`` object. See ``help(qubovert.PUBO)`` and
        ``help(qubovert.utils.PUBOMatrix)`` for info on formatting.

    Example
    -------
    >>> H = {(0,): 1, (1,): -1, (0, 1): -1}
    >>> P = puso_to_pubo(H)
    >>> isinstance(P, qubovert.utils.PUBOMatrix)
    True

    >>> H = {('0',): 1, ('1',): -1, (0, 1): -1}
    >>> P = puso_to_pubo(H)
    >>> isinstance(P, qubovert.PUBO)
    True

    """
    def generate_new_key_value(k):
        """generate_new_key_value.

        Recursively generate the PUBO key, value pairs for converting the
        product ``z[k[0]] * ... * z[k[-1]]``, where each ``z`` is a spin in
        {1, -1}, to the product ``(1-2*x[k[0]]) * ... * (1-2*x[k[1]])``, where
        each ``x`` is a boolean variables in {0, 1}.

        Parameters
        ----------
        k : tuple.
            Each element of the tuple corresponds to a spin label.

        Yields
        ------
        res : tuple (key, value)
            key : tuple.
                Each element of the tuple corresponds to a binary label.
            value : float.
                The value to multiply the value corresponding with ``k`` by.

        """
        if not k:
            yield k, 1
        else:
            for key, value in generate_new_key_value(k[1:]):
                yield (k[0],) + key, -2 * value
                yield key, value

    # not isinstance! because isinstance(PUSO, PUSOMatrix) is True
    P = PUBOMatrix() if type(H) == PUSOMatrix else qv.PUBO()

    for k, v in H.items():
        for key, value in generate_new_key_value(k):
            P[key] += value * v

    return P


class Conversions:
    """Conversions.

    This is a parent class that defines the functions ``to_qubo``,
    ``to_quso``, ``to_pubo``, and ``to_puso``. Any subclass that inherits
    from ``Conversions`` `must` supply at least one of ``to_qubo`` or
    ``to_quso``. And at least one of ``to_pubo`` or ``to_puso``.


    """

    def to_qubo(self, *args, **kwargs):
        """to_qubo.

        Create and return upper triangular QUBO representing the problem.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls ``to_quso`` and
        converts the quso formulation to a QUBO formulation.

        Parameters
        ----------
        arguments : Defined in the child class.
            They should be parameters that define lagrange multipliers or
            factors in the QUBO.

        Returns
        -------
        Q : qubovert.utils.QUBOMatrix object.
            The upper triangular QUBO matrix, a QUBOMatrix object.
            For most practical purposes, you can use QUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUBOMatrix)``.

        Raises
        ------
        ``RecursionError`` if neither ``to_qubo`` nor ``to_quso`` are defined
        in the subclass.

        """
        return quso_to_qubo(self.to_quso(*args, **kwargs))

    def to_quso(self, *args, **kwargs):
        """to_quso.

        Create and return QUSO model representing the problem.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls ``to_qubo`` and
        converts the QUBO formulation to an QUSO formulation.

        Parameters
        ----------
        arguments : Defined in the child class.
            They should be parameters that define lagrange multipliers or
            factors in the QUSO model.

        Return
        ------
        L : qubovert.utils.QUSOMatrix object.
            The upper triangular coupling matrix, where two element tuples
            represent couplings and one element tuples represent fields.
            For most practical purposes, you can use IsingCoupling in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUSOMatrix)``.

        Raises
        ------
        ``RecursionError`` if neither ``to_qubo`` nor ``to_quso`` are defined
        in the subclass.

        """
        return qubo_to_quso(self.to_qubo(*args, **kwargs))

    def to_pubo(self, *args, **kwargs):
        """to_pubo.

        Create and return upper triangular PUBO representing the problem.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls ``to_puso`` or
        ``to_qubo`` and converts the puso or QUBO formulations to a
        PUBO formulation.

        Parameters
        ----------
        arguments : Defined in the child class.
            They should be parameters that define lagrange multipliers or
            factors in the QUBO.

        Returns
        -------
        P : qubovert.utils.PUBOMatrix object.
            The upper triangular PUBO matrix, a PUBOMatrix object.
            For most practical purposes, you can use PUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.PUBOMatrix)``.

        Raises
        ------
        ``RecursionError`` if neither ``to_pubo`` nor ``to_puso`` are defined
        in the subclass.

        """
        return puso_to_pubo(self.to_puso(*args, **kwargs))

    def to_puso(self, *args, **kwargs):
        """to_puso.

        Create and return PUSO model representing the problem.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls ``to_pubo`` or
        ``to_quso`` and converts to a PUSO formulation.

        Parameters
        ----------
        arguments : Defined in the child class.
            They should be parameters that define lagrange multipliers or
            factors in the QUSO model.

        Return
        ------
        H : qubovert.utils.PUSOMatrix object.
            For most practical purposes, you can use PUSOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.PUSOMatrix)``.

        Raises
        ------
        ``RecursionError`` if neither ``to_pubo`` nor ``to_puso`` are defined
        in the subclass.

        """
        return pubo_to_puso(self.to_pubo(*args, **kwargs))
