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

This file contains methods to convert to and from QUBO/Ising/PUBO/HIsing
problems, as well as various misc converters

"""

from . import QUBOMatrix, IsingMatrix, PUBOMatrix, HIsingMatrix
# for QUBO, Ising, PUBO, HIsing, can't import directly because it will cause
# circular imports, so instead just import qubovert.
import qubovert


__all__ = (
    'qubo_to_ising', 'ising_to_qubo', 'pubo_to_hising', 'hising_to_pubo',
    'Conversions'
)


def qubo_to_ising(Q):
    """qubo_to_ising.

    Convert the specified QUBO problem into an Ising problem. Note that
    QUBO {0, 1} values go to Ising {-1, 1} values in that order!

    Parameters
    ----------
    Q : dictionary, qubovert.QUBO, or qubovert.utils.QUBOMatrix object.
        Maps tuples of binary variables indices to the Q value. See
        ``help(qubovert.QUBO)`` and ``help(qubovert.utils.QUBOMatrix)`` for
        info on formatting.

    Returns
    ------
    L : qubovert.Ising or qubovert.utils.IsingMatrix object.
        tuple of spin labels map to Ising values. If ``Q`` is a
        ``qubovert.utils.QUBOMatrix`` object, then ``L`` will be a
        ``qubovert.utils.IsingMatrix``, otherwise ``L`` will be a
        ``qubovert.Ising`` object.

    Example
    -------
    >>> Q = {(0,): 1, (0, 1): -1, (1,): 3}
    >>> L = qubo_to_ising(Q)
    >>> isinstance(L, qubovert.utils.IsingMatrix)
    True

    >>> Q = {('a',): 1, ('a', 'b'): -1, ('b',): 3}
    >>> L = qubo_to_ising(Q)
    >>> isinstance(L, qubovert.Ising)
    True

    """
    # could just use IsingMatrix(pubo_to_hising(Q)), but then we spend a lot of
    # time converting from a HIsingMatrix to IsingMatrix, so instead we
    # explictly write out the conversion.

    # not isinstance! because isinstance(QUBO, QUBOMatrix) is True
    if type(Q) == QUBOMatrix:
        L = IsingMatrix()
        squash_key = QUBOMatrix.squash_key
    else:
        L = qubovert.Ising()
        squash_key = qubovert.QUBO.squash_key

    for kp, v in Q.items():
        k = squash_key(kp)
        if not k:
            L[k] += v
        elif len(k) == 1:
            L[k] += v / 2
            L[()] += v / 2
        elif len(k) == 2:
            i, j = k
            L[k] += v / 4
            L[(i,)] += v / 4
            L[(j,)] += v / 4
            L[()] += v / 4
        # len(k) cannot be greater than 2 because the squash_key checks

    return L


def ising_to_qubo(L):
    """ising_to_qubo.

    Convert the specified Ising problem into an upper triangular QUBO problem.
    Note that Ising {-1, 1} values go to QUBO {0, 1} values in that order!

    Parameters
    ----------
    L : dictionary, qubovert.Ising, or qubovert.utils.IsingMatrix object.
        Tuple of spin labels map to Ising values. See
        ``help(qubovert.Ising)`` and ``help(qubovert.utils.IsingMatrix)`` for
        info on formatting.


    Returns
    -------
    Q : qubovert.QUBO or qubovert.utils.QUBOMatrix object.
        If ``L`` is a ``qubovert.utils.IsingMatrix`` object, then ``Q`` will be
        a ``qubovert.utils.QUBOMatrix``, otherwise ``Q`` will be a
        ``qubovert.QUBO`` object. See ``help(qubovert.QUBO)`` and
        ``help(qubovert.utils.QUBOMatrix)`` for info on formatting.

    Example
    -------
    >>> L = {(0,): 1, (1,): -1, (0, 1): -1}
    >>> Q = ising_to_qubo(L)
    >>> isinstance(Q, qubovert.utils.QUBOMatrix)
    True

    >>> L = {('a',): 1, ('b',): -1, (0, 1): -1}
    >>> Q = ising_to_qubo(L)
    >>> isinstance(Q, qubovert.QUBO)
    True

    """
    # could just use QUBOMatrix(hising_to_pubo(L)), but then we spend a lot of
    # time converting from a PUBOMatrix to QUBOMatrix, so instead we explictly
    # write out the conversion.

    # not isinstance! because isinstance(Ising, IsingMatrix) is True
    if type(L) == IsingMatrix:
        Q = QUBOMatrix()
        squash_key = IsingMatrix.squash_key
    else:
        Q = qubovert.QUBO()
        squash_key = qubovert.Ising.squash_key

    for kp, v in L.items():
        k = squash_key(kp)
        if not k:
            Q[k] += v
        elif len(k) == 1:
            Q[k] += 2 * v
            Q[()] -= v
        elif len(k) == 2:
            i, j = k
            Q[k] += 4 * v
            Q[(i,)] -= 2 * v
            Q[(j,)] -= 2 * v
            Q[()] += v
        # squash key ensures that len(k) <= 2

    return Q


def pubo_to_hising(P):
    """pubo_to_hising.

    Convert the specified PUBO problem into an HIsing problem. Note that
    PUBO {0, 1} values go to HIsing {-1, 1} values in that order!

    Parameters
    ----------
    P : dictionary, qubovert.PUBO, or qubovert.utils.PUBOMatrix object.
        Maps tuples of binary variables indices to the P value. See
        ``help(qubovert.PUBO)`` and ``help(qubovert.utils.PUBOMatrix)`` for
        info on formatting.

    Returns
    ------
    H : qubovert.utils.HIsingMatrix object.
        tuple of spin labels map to HIsing values. If ``P`` is a
        ``qubovert.utils.PUBOMatrix`` object, then ``H`` will be a
        ``qubovert.utils.HIsingMatrix``, otherwise ``H`` will be a
        ``qubovert.HIsing`` object..

    Example
    -------
    >>> P = {(0,): 1, (0, 1): -1, (1,): 3}
    >>> H = pubo_to_hising(P)
    >>> isinstance(H, qubovert.utils.HIsingMatrix)
    True

    >>> P = {(0,): 1, (0, 1): -1, (1,): 3}
    >>> H = pubo_to_hising(P)
    >>> isinstance(H, qubovert.Ising)
    True

    """
    def generate_new_key_value(k):
        """generate_new_key_value.

        Recursively generate the PUBO key, value pairs for converting the
        product ``x[k[0]] * ... * x[k[-1]]``, where each ``x`` is a binary
        variable in {0, 1}, to the product
        ``(z[k[0]]+1)/2 * ... * (z[k[1]]+1)/2``., where each ``z`` is a spin
        in {-1, 1}.

        Parameters
        ----------
        k : tuple.
            Each element of the tuple corresponds to a binary label.

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
                yield (k[0],) + key, value / 2
                yield key, value / 2

    # not isinstance! because isinstance(PUBO, PUBOMatrix) is True
    if type(P) == PUBOMatrix:
        H = HIsingMatrix()
        squash_key = PUBOMatrix.squash_key
    else:
        H = qubovert.HIsing()
        squash_key = qubovert.PUBO.squash_key

    for k, v in P.items():
        for key, value in generate_new_key_value(squash_key(k)):
            H[key] += value * v

    return H


def hising_to_pubo(H):
    """hising_to_pubo.

    Convert the specified HIsing problem into an upper triangular PUBO problem.
    Note that HIsing {-1, 1} values go to PUBO {0, 1} values in that order!

    Parameters
    ----------
    H : dictionary or qubovert.utils.HIsingMatrix object.
        Tuple of spin labels map to HIsing values. See
        ``help(qubovert.HIsing)`` and ``help(qubovert.utils.HIsingMatrix)`` for
        info on formatting.

    Returns
    -------
    P : qubovert.utils.PUBOMatrix object.
        If ``H`` is a ``qubovert.utils.HIsingMatrix`` object, then ``P`` will
        be a ``qubovert.utils.PUBOMatrix``, otherwise ``P`` will be a
        ``qubovert.PUBO`` object. See ``help(qubovert.PUBO)`` and
        ``help(qubovert.utils.PUBOMatrix)`` for info on formatting.

    Example
    -------
    >>> H = {(0,): 1, (1,): -1, (0, 1): -1}
    >>> P = hising_to_pubo(H)
    >>> isinstance(P, qubovert.utils.PUBOMatrix)
    True

    >>> H = {('0',): 1, ('1',): -1, (0, 1): -1}
    >>> P = hising_to_pubo(H)
    >>> isinstance(P, qubovert.PUBO)
    True

    """
    def generate_new_key_value(k):
        """generate_new_key_value.

        Recursively generate the PUBO key, value pairs for converting the
        product ``z[k[0]] * ... * z[k[-1]]``, where each ``z`` is a spin in
        {-1, 1}, to the product ``(2*x[k[0]]-1) * ... * (2*x[k[1]]-1)``, where
        each ``x`` is a binary variables in {0, 1}.

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
                yield (k[0],) + key, 2 * value
                yield key, -value

    # not isinstance! because isinstance(Ising, IsingMatrix) is True
    if type(H) == HIsingMatrix:
        P = PUBOMatrix()
        squash_key = HIsingMatrix.squash_key
    else:
        P = qubovert.PUBO()
        squash_key = qubovert.HIsing.squash_key

    for k, v in H.items():
        for key, value in generate_new_key_value(squash_key(k)):
            P[key] += value * v

    return P


class Conversions:
    """Conversions.

    This is a parent class that defines the functions ``to_qubo``,
    ``to_ising``, ``to_pubo``, and ``to_hising``. Any subclass that inherits
    from ``Conversions`` `must` supply at least one of ``to_qubo`` or
    ``to_ising``. And at least one of ``to_pubo`` or ``to_hising``.


    """

    def to_qubo(self, *args, **kwargs):
        """to_qubo.

        Create and return upper triangular QUBO representing the problem.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls ``to_ising`` and
        converts the ising formulation to a QUBO formulation.

        Parameters
        ----------
        arguments : Defined in the child class.
            They should be parameters that define lagrange multipliers or
            factors in the QUBO.

        Return
        -------
        Q : qubovert.utils.QUBOMatrix object.
            The upper triangular QUBO matrix, a QUBOMatrix object.
            For most practical purposes, you can use QUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUBOMatrix)``.

        Raises
        -------
        ``RecursionError`` if neither ``to_qubo`` nor ``to_ising`` are defined
        in the subclass.

        """
        return ising_to_qubo(self.to_ising(*args, **kwargs))

    def to_ising(self, *args, **kwargs):
        """to_ising.

        Create and return Ising model representing the problem.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls ``to_qubo`` and
        converts the QUBO formulation to an Ising formulation.

        Parameters
        ----------
        arguments : Defined in the child class.
            They should be parameters that define lagrange multipliers or
            factors in the Ising model.

        Return
        ------
        L : qubovert.utils.IsingMatrix object.
            The upper triangular coupling matrix, where two element tuples
            represent couplings and one element tuples represent fields.
            For most practical purposes, you can use IsingCoupling in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.IsingMatrix)``.

        Raises
        -------
        ``RecursionError`` if neither ``to_qubo`` nor ``to_ising`` are defined
        in the subclass.

        """
        return qubo_to_ising(self.to_qubo(*args, **kwargs))

    def to_pubo(self, *args, **kwargs):
        """to_pubo.

        Create and return upper triangular PUBO representing the problem.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls ``to_hising`` or
        ``to_qubo`` and converts the hising or QUBO formulations to a
        PUBO formulation.

        Parameters
        ----------
        arguments : Defined in the child class.
            They should be parameters that define lagrange multipliers or
            factors in the QUBO.

        Return
        -------
        P : qubovert.utils.PUBOMatrix object.
            The upper triangular PUBO matrix, a PUBOMatrix object.
            For most practical purposes, you can use PUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.PUBOMatrix)``.

        Raises
        -------
        ``RecursionError`` if neither ``to_pubo`` nor ``to_hising`` are defined
        in the subclass.

        """
        return hising_to_pubo(self.to_hising(*args, **kwargs))

    def to_hising(self, *args, **kwargs):
        """to_hising.

        Create and return HIsing model representing the problem.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls ``to_pubo`` or
        ``to_ising`` and converts to a HIsing formulation.

        Parameters
        ----------
        arguments : Defined in the child class.
            They should be parameters that define lagrange multipliers or
            factors in the Ising model.

        Return
        ------
        H : qubovert.utils.HIsingMatrix object.
            For most practical purposes, you can use HIsingMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.HIsingMatrix)``.

        Raises
        -------
        ``RecursionError`` if neither ``to_pubo`` nor ``to_hising`` are defined
        in the subclass.

        """
        return pubo_to_hising(self.to_pubo(*args, **kwargs))
