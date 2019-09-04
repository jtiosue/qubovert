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

"""_pubo.py.

Contains the PUBO class. See ``help(qubovert.PUBO)``.

"""

from .utils import BO, PUBOMatrix, QUBOMatrix
from . import QUBO
# in PUBO._reduce_degree, we use HOBO.add_constraint_AND. But HOBO inherits
# from PUBO, so can't say `from . import HOBO` here. Instead, just import
# qubovert
import qubovert


__all__ = 'PUBO',


class PUBO(BO, PUBOMatrix):
    """PUBO.

    Class to manage converting general PUBO problems to and from their
    PUBO, HIsing, QUBO, and Ising formluations. In general, this class
    deals with unconstrained optimization problems that have arbitrary degree.
    To convert this to a QUBO (see ``to_qubo``) or Ising (``to_ising``) we have
    to introduce ancilla variables. The ``convert_solution`` method deals with
    converting a solution to the problem with ancilla variables back to the
    solution to the original problem.

    This class deals with PUBOs that have binary labels that do not range from
    0 to n-1. Note that it is generally
    more efficient to initialize an empty PUBO object and then build the
    PUBO, rather than initialize a PUBO object with an already built dict.

    PUBO inherits some methods and attributes from the ``PUBOMatrix`` class.
    See ``help(qubovert.utils.PUBOMatrix)``.

    PUBO inherits some methods and attributes the ``BO`` class. See
    ``help(qubovert.utils.BO)``.

    Example usage
    -------------
    >>> pubo = PUBO()
    >>> pubo[('a',)] += 5
    >>> pubo[(0, 'a', 1)] -= 2
    >>> pubo -= 1.5
    >>> pubo
    {('a',): 5, ('a', 0, 1): -2, (): -1.5}

    >>> pubo = PUBO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
    >>> pubo
    {('a',): 5, ('a', 0, 1): -2, (): -1.5}
    >>> P = pubo.to_pubo()
    >>> P
    {(0,): 5, (0, 1): -2, (): -1.5}
    >>> pubo.convert_solution({0: 1, 1: 0, 2: 1})
    {'a': 1, 0: 0, 1: 1}

    In the next example, notice that we introduce ancilla variables to
    represent that ```(0, 1)`` term. See the ``to_qubo`` method for more info.

    >>> pubo = PUBO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
    >>> pubo.mapping
    {'a': 0, 0: 1, 1: 2}
    >>> Q = pubo.to_qubo()
    >>> Q
    {(0,): 5, (3,): 9, (0, 1): 3, (0, 3): -6, (1, 3): -6, (2, 3): -2, (): -1.5}
    >>> pubo.convert_solution({0: 1, 1: 0, 2: 1, 2: 0})
    {'a': 1, 0: 0, 1: 1}

    Note 1
    ------
    Note that keys will end up sorted by their hash. Hashes will not be
    consistent across Python sessions (unless they are integers)! For example,
    both of the following can happen:

    >>> print(PUBO({('a', 0): 1, (0, 1): -1}))
    {('a', 0): 1, (0, 1): -1}

    or

    >>> print(PUBO({('a', 0): 1, (0, 1): -1}))
    {(0, 'a'): 1, (0, 1): -1}

    But the following will never happen:

    >>> print(PUBO({('a', 0): 1, (0, 1): -1}))
    {('a', 0): 1, (1, 0): -1}

    Ie integers will always be correctly sorted.

    Note 2
    ------
    For efficiency, many internal variables including mappings are computed as
    the problemis being built. This can cause these
    values to be wrong for some specific situations. Calling ``refresh``
    will rebuild the dictionary, resetting all of the values. See
    ``help(PUBO.refresh)``

    Examples
    --------
    >>> from qubovert import PUBO
    >>> P = PUBO()
    >>> P[('a',)] += 1
    >>> P, P.mapping, P.reverse_mapping
    {('a',): 1}, {'a': 0}, {0: 'a'}
    >>> P[('a',)] -= 1
    >>> P, P.mapping, P.reverse_mapping
    {}, {'a': 0}, {0: 'a'}
    >>> P.refresh()
    >>> P, P.mapping, P.reverse_mapping
    {}, {}, {}

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        This class deals with PUBOs that have binary labels that do not range
        from 0 to n-1. Note that it is generally more efficient
        to initialize an empty PUBO object and then build the PUBO, rather than
        initialize a PUBO object with an already built dict.

        Parameters
        ----------
        args and kwargs : define a dictionary with ``dict(*args, **kwargs)``.
            The dictionary will be initialized to follow all the convensions of
            the class.

        Examples
        -------
        >>> pubo = PUBO()
        >>> pubo[('a',)] += 5
        >>> pubo[(0, 'a')] -= 2
        >>> pubo -= 1.5
        >>> pubo
        {('a',): 5, ('a', 0): -2, (): -1.5}

        >>> pubo = PUBO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> pubo
        {('a',): 5, ('a', 0, 1): -2, (): -1.5}

        """
        BO.__init__(self, *args, **kwargs)
        PUBOMatrix.__init__(self, *args, **kwargs)

    @staticmethod
    def default_lam(v):
        """default_lam.

        This is the default function used in ``to_qubo``. It returns
        ``1 + abs(v)``. It weights the penalties used to enforce the constraint
        ``xy = z``. See the ``to_qubo`` method.

        Parameters
        ----------
        v : float.

        Return
        ------
        res : float.
            Penalty weight.

        """
        return 1 + abs(v)

    def _reduce_degree(self, D, deg, lam):
        """_reduce_degree.

        Reduce the degree of the higher order model to a degree ``deg`` model.
        This is a private used only internally and in ``qubovert.HIsing``. See
        ``qubovert.PUBO.to_pubo``, ``qubovert.HIsing.to_hising``,
        ``qubovert.PUBO.to_qubo``, and ``qubovert.HIsing.to_ising`` to see
        an example of this function being used.

        The labels will be integers from 0 to n-1. We introduce ancilla
        variables in order to reduce the degree. The solution to the higher
        order model can be read from the solution to the lower degree model by
        using the ``self.convert_solution`` method.

        Parameters
        ----------
        D : ``qubovert.utils.QUBOMatrix`` or ``qubovert.utils.IsingMatrix``.
            The dictionary to fill. For reducing PUBOs to QUBOs, ``D`` should
            be a ``qubovert.utils.QUBOMatrix`` object. For reducing HIsings
            to Isings, ``D`` should be a ``qubovert.utils.IsingMatrix`` object.
            ``D`` should be empty to start.
        deg : int >= 2.
            The degree of the model to reduce to. If``deg`` is None, then
            the model's degree will not be reduced.
        lam : function.
            If ``lam`` is None, the function ``PUBO.default_lam`` will be used.
            ``lam`` is the penalty factor to introduce in order to enforce the
            ancilla constraints. When we reduce the degree of the model, we add
            penalties to the lower order model in order to enforce ancilla
            variable constraints. These constraints will be multiplied by
            ``lam(v)``, where ``v`` is the value associated with the term that
            it is reducing. For example, a term ``(0, 1, 2): 3`` in the higher
            order model may be reduced to a term ``(0, 3): 3`` for the lower
            order model, and then the fact that ``3`` should be the product
            of ``1`` and ``2`` will be enforced with a penalty weight
            ``lam(3)``.

        Return
        -------
        None. ``D`` will be updated in place!

        """
        if deg is not None and deg < 2:
            raise ValueError("deg must be >= 2")
        if lam is None:
            lam = PUBO.default_lam
        if deg is None:
            deg = self.degree

        # next available label
        ancilla = self.num_binary_variables
        reductions = {}

        for kp, v in self.items():
            key = tuple(sorted(self._mapping[i] for i in kp))
            if key in reductions:
                D[(reductions[key],)] += v
            else:
                # find a reduction if len(key) > deg
                k = key
                while len(k) > deg:

                    # find a variable pair in k that has already been reduced.
                    found = False
                    for i, x in enumerate(k[:-1]):
                        for y in k[i+1:]:
                            if (x, y) in reductions:
                                found = True
                                break
                        if found:
                            break

                    if found:
                        # z is the ancilla variable for this reduction
                        z = reductions[(x, y)]
                    else:
                        # found is False so we haven't already reduced the
                        # variable pair (x, y), so just take the first two and
                        # reduce them.
                        # TODO: come up with a better way to choose x, y here.
                        x, y, z = k[0], k[1], ancilla
                        reductions[(x, y)] = z
                        ancilla += 1

                    # note we add the constraint even if we've already added
                    # it before (if found is True). This is because if we use
                    # the reduction multiple times, we need to enforce it
                    # multiple times.
                    D += qubovert.HOBO().add_constraint_AND(x, y, z, lam(v))
                    k = tuple(sorted(
                            tuple(i for i in k if i not in (x, y)) + (z,)
                        ))
                D[k] += v

    def to_pubo(self, deg=None, lam=None):
        """to_pubo.

        Create and return upper triangular degree ``deg`` PUBO representing the
        problem. The labels will be integers from 0 to n-1.

        Parameters
        ----------
        deg : int >= 0 (optional, defaults to None).
            The degree of the final PUBO. If ``deg`` is None, then the degree
            of the output PUBO will be the same as the degree of ``self``,
            ie see ``self.degree``.
        lam : function (optional, defaults to None).
            Note that if ``deg`` is None or ``deg >= self.degree``, then
            ``lam`` is unneccessary and will not be used.
            If ``lam`` is None, the function ``PUBO.default_lam`` will be used.
            ``lam`` is the penalty factor to introduce in order to enforce the
            ancilla constraints. When we reduce the degree of the model, we add
            penalties to the lower order model in order to enforce ancilla
            variable constraints. These constraints will be multiplied by
            ``lam(v)``, where ``v`` is the value associated with the term that
            it is reducing. For example, a term ``(0, 1, 2): 3`` in the higher
            order model may be reduced to a term ``(0, 3): 3`` for the lower
            order model, and then the fact that ``3`` should be the product
            of ``1`` and ``2`` will be enforced with a penalty weight
            ``lam(3)``.

        Return
        -------
        P : qubovert.utils.PUBOMatrix object.
            The upper triangular PUBO matrix, a PUBOMatrix object.
            For most practical purposes, you can use PUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.PUBOMatrix)``.

        Notes
        -----
        The penalty that we use to enforce the constraints that the ancilla
        variable ``z`` is equal to the product of the two variables that it is
        replacing, ``xy``, is:

            ``0`` if ``z == xy``,
            ``3*lam(v)`` if ``x == y == 0 and z == 1``, and
            ``lam(v)`` else.

        See https://arxiv.org/pdf/1307.8041.pdf equation 6.

        """
        P = PUBOMatrix()
        self._reduce_degree(P, deg, lam)
        return P

    def to_qubo(self, lam=None):
        """to_qubo.

        Create and return upper triangular QUBO representing the problem.
        The labels will be integers from 0 to n-1. We introduce ancilla
        variables in order to reduce the degree of the PUBO to a QUBO. The
        solution to the PUBO can be read from the solution to the QUBO by
        using the ``convert_solution`` method.

        Parameters
        ----------
        lam : function (optional, defaults to None).
            If ``lam`` is None, the function ``PUBO.default_lam`` will be used.
            ``lam`` is the penalty factor to introduce in order to enforce the
            ancilla constraints. When we reduce the degree of the PUBO to a
            QUBO, we add penalties to the QUBO in order to enforce ancilla
            variable constraints. These constraints will be multiplied by
            ``lam(v)``, where ``v`` is the value associated with the term that
            it is reducing. For example, a term ``(0, 1, 2): 3`` in the PUBO
            may be reduced to a term ``(0, 3): 3`` for the QUBO, and then the
            fact that ``3`` should be the product of ``1`` and ``2`` will be
            enforced with a penalty weight ``lam(3)``.

        Return
        -------
        Q : qubovert.utils.QUBOMatrix object.
            The upper triangular QUBO matrix, a QUBOMatrix object.
            For most practical purposes, you can use QUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUBOMatrix)``.

        """
        Q = QUBOMatrix()
        self._reduce_degree(Q, 2, lam)
        return Q

    def convert_solution(self, solution):
        """convert_solution.

        Convert the solution to the integer labeled PUBO to the solution to
        the originally labeled PUBO.

        Parameters
        ----------
        solution : iterable or dict.
            The PUBO, HIsing, QUBO, or Ising solution output. The PUBO solution
            output is either a list or tuple where indices specify the label of
            the variable and the element specifies whether it's 0 or 1 for PUBO
            (or -1 or 1 for Ising), or it can be a dictionary that maps the
            label of the variable to is value. The QUBO/Ising solution output
            includes the assignment for the ancilla variables used to reduce
            the degree of the PUBO.

        Return
        -------
        res : dict.
            Maps binary variable labels to their PUBO solutions values {0, 1}.

        Example
        -------
        >>> pubo = PUBO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> pubo
        {('a',): 5, ('a', 0, 1): -2, (): -1.5}
        >>> P = pubo.to_pubo()
        >>> P
        {(0,): 5, (0, 1): -2, (): -1.5}
        >>> pubo.convert_solution({0: 1, 1: 0, 2: 1})
        {'a': 1, 0: 0, 1: 1}

        In the next example, notice that we introduce ancilla variables to
        represent that ```(0, 1)`` term. See the ``to_qubo`` method for more
        info.

        >>> pubo = PUBO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> pubo.mapping
        {'a': 0, 0: 1, 1: 2}
        >>> Q = pubo.to_qubo(3)
        >>> Q
        {(0,): 5, (0, 3):-2, ():-1.5, (1, 2): 3, (3,): 3, (1, 3): 3, (2, 3): 3}
        >>> pubo.convert_solution({0: 1, 1: 0, 2: 1, 2: 0})
        {'a': 1, 0: 0, 1: 1}

        Notes
        -----
        We take ignore the ancilla variable assignments when we convert the
        solution. For example if the conversion from PUBO to QUBO introduced
        an ancilla varable ``z = xy`` where ``x`` and ``y`` are variables of
        the PUBO, then ``solution`` must have values for ``x``, ``y``, and
        ``z``. If the QUBO solver found that ``x = 1``, ``y = 0``, and
        ``z = 1``, then the constraint that ``z = xy`` is not satisfied (one
        possible cause for this is if the ``lam`` argument in ``to_qubo`` is
        too small). ``convert_solution`` will return that ``x = 1`` and
        ``y = 0`` and ignore the value of ``z``.

        """
        # this works for converting a solution to the pubo, qubo, hising, or
        # ising formulations, since in the to_qubo function all ancilla
        # variables are labeled with integers >= self.num_binary_variables.
        return QUBO.convert_solution(self, solution)

    @staticmethod
    def _check_key_valid(key):
        """_check_key_valid.

        Internal method to check if an input key to the dictionary is valid.
        Checks to see if ``key`` is a tuple.

        Parameters
        ---------
        key : anything, but must be a tuple to be valid.

        Returns
        -------
        None.

        Raises
        ------
        KeyError if the key is invalid.

        """
        # override PUBOMatrix._check_key_valid to allow for noninteger keys.
        if not isinstance(key, tuple):
            raise KeyError(
                "Key formatted incorrectly, must be tuple")
