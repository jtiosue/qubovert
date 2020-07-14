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

"""_puso.py.

Contains the PUSO class. See ``help(qubovert.PUSO)``.

"""

from .utils import BO, PUSOMatrix, puso_to_pubo, QUSOMatrix
from . import QUSO


__all__ = 'PUSO',


class PUSO(BO, PUSOMatrix):
    """PUSO.

    Class to manage converting general PUSO problems to and from their
    PUSO, PUBO, QUSO, and QUBO formluations. In general, this class
    deals with unconstrained optimization problems that have arbitrary degree.
    To convert this to a QUSO (see ``to_quso``) or QUBO (``to_qubo``) we have
    to introduce ancilla variables. The ``convert_solution`` method deals with
    converting a solution to the problem with ancilla variables back to the
    solution to the original problem.

    This class deals with PUSOs that have spin labels that do not range
    from 0 to n-1. Note that it is generally
    more efficient to initialize an empty PUSO object and then build the
    PUSO, rather than initialize a PUSO object with an already built dict.

    PUSO inherits some methods and attributes the ``PUSOMatrix`` class. See
    ``help(qubovert.utils.PUSOMatrix)``.

    PUSO inherits some methods and attributes the ``BO`` class. See
    ``help(qubovert.utils.BO)``.

    Example
    -------
    >>> puso = PUSO()
    >>> puso[('a',)] += 5
    >>> puso[(0, 'a', 1)] -= 2
    >>> puso -= 1.5
    >>> puso
    {('a',): 5, ('a', 0, 1): -2, (): -1.5}

    >>> puso = PUSO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
    >>> puso
    {('a',): 5, ('a', 0, 1): -2, (): -1.5}
    >>> H = puso.to_puso()
    >>> H
    {(0,): 5, (0, 1): -2, (): -1.5}
    >>> puso.convert_solution({0: 1, 1: -1, 2: 1})
    {'a': 1, 0: -1, 1: 1}

    In the next example, notice that we introduce ancilla variables to
    represent that ```(0, 1)`` term. See the ``to_quso`` method for more info.

    >>> puso = PUSO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
    >>> puso.mapping
    {'a': 0, 0: 1, 1: 2}
    >>> L = puso.to_quso()
    >>> L
    {(0,): 0.75, (): 11.25, (3,): 8.5, (0, 1): 4.25, (1,): -4.25, (0, 3): -8.5,
     (1, 3): -8.5, (2, 3): -4.0, (1, 2): 2.0, (0, 2): 2.0, (2,): -2.0}
    >>> puso.convert_solution({0: 1, 1: -1, 2: 1, 3: -1})
    {'a': 1, 0: -1, 1: 1}

    Note
    ----
    For efficiency, many internal variables including mappings are computed as
    the problemis being built. This can cause these
    values to be wrong for some specific situations. Calling ``refresh``
    will rebuild the dictionary, resetting all of the values. See
    ``help(PUSO.refresh)``

    Examples
    --------
    >>> from qubovert import PUSO
    >>> H = PUSO()
    >>> H[('a',)] += 1
    >>> H, H.mapping, H.reverse_mapping
    {('a',): 1}, {'a': 0}, {0: 'a'}
    >>> H[('a',)] -= 1
    >>> H, H.mapping, H.reverse_mapping
    {}, {'a': 0}, {0: 'a'}
    >>> H.refresh()
    >>> H, H.mapping, H.reverse_mapping
    {}, {}, {}

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        This class deals with PUSOs that have spin labels that do not range
        from 0 to n-1. Note that it is generally more efficient to initialize
        an empty PUSO object and then build the PUSO, rather than
        initialize a PUSO object with an already built dict.

        Parameters
        ----------
        arguments : define a dictionary with ``dict(*args, **kwargs)``.
            The dictionary will be initialized to follow all the convensions of
            the class.

        Examples
        --------
        >>> puso = PUSO()
        >>> puso[('a',)] += 5
        >>> puso[(0, 'a')] -= 2
        >>> puso -= 1.5
        >>> puso
        {('a',): 5, ('a', 0): -2, (): -1.5}

        >>> puso = PUSO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> puso
        {('a',): 5, ('a', 0, 1): -2, (): -1.5}

        """
        BO.__init__(self, *args, **kwargs)
        PUSOMatrix.__init__(self, *args, **kwargs)

    def _to_puso(self):
        """to_puso.

        Internal helper method.

        Create and return upper triangular degree PUSO representing
        the problem. The labels will be integers from 0 to n-1.

        Return
        ------
        H : qubovert.utils.PUSOMatrix object.
            The upper triangular PUSO matrix, a PUSOMatrix object.
            For most practical purposes, you can use PUSOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.PUSOMatrix)``.

        """
        H = PUSOMatrix()
        for k, v in self.items():
            key = tuple(sorted(self._mapping[i] for i in k))
            H[key] += v
        return H

    def _create_pubo(self):
        """_create_pubo.

        Internal method to create the PUBO object from self.

        Returns
        -------
        P : qubovert.PUBO object.

        """
        P = puso_to_pubo(self)
        P._mapping = self.mapping
        P._reverse_mapping = self.reverse_mapping
        return P

    def to_pubo(self, deg=None, lam=None, pairs=None):
        """to_pubo.

        Create and return upper triangular degree ``deg`` PUBO representing the
        PUSO problem. The labels will be integers from 0 to n-1.

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
        pairs : set (optional, defaults to None).
            A set of tuples of variable pairs to prioritize pairing together in
            to degree reduction. If a pair in ``pairs`` is found together in
            the PUSO, it will be chosen as a pair to reduce to a single
            ancilla. You should supply this parameter if you have a good idea
            of an efficient way to reduce the degree of the PUSO. If ``pairs``
            is None, then it will be the empty set ``set()``. In other words,
            no variable pairs will be prioritized, and instead variable pairs
            will be chosen to reduce to an ancilla bases solely on frequency
            of occurrance.

        Return
        ------
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
        return self._create_pubo().to_pubo(deg, lam, pairs)

    def to_puso(self, deg=None, lam=None, pairs=None):
        """to_puso.

        Create and return upper triangular degree ``deg`` PUSO representing
        the problem. The labels will be integers from 0 to n-1.

        Parameters
        ----------
        deg : int >= 0 (optional, defaults to None).
            The degree of the final PUSO. If ``deg`` is None, then the degree
            of the output PUSO will be the same as the degree of ``self``,
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
        pairs : set (optional, defaults to None).
            A set of tuples of variable pairs to prioritize pairing together in
            to degree reduction. If a pair in ``pairs`` is found together in
            the PUSO, it will be chosen as a pair to reduce to a single
            ancilla. You should supply this parameter if you have a good idea
            of an efficient way to reduce the degree of the PUSO. If ``pairs``
            is None, then it will be the empty set ``set()``. In other words,
            no variable pairs will be prioritized, and instead variable pairs
            will be chosen to reduce to an ancilla bases solely on frequency
            of occurrance.

        Return
        ------
        H : qubovert.utils.PUSOMatrix object.
            The upper triangular PUSO matrix, a PUSOMatrix object.
            For most practical purposes, you can use PUSOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.PUSOMatrix)``.

        Notes
        -----
        See the ``to_pubo`` docstring for more information on the penalties and
        ``lam``.

        """
        if deg is None or deg >= self.degree:
            return self._to_puso()
        return self._create_pubo().to_puso(deg, lam, pairs)

    def to_qubo(self, lam=None, pairs=None):
        """to_qubo.

        Create and return upper triangular QUBO representing the problem.
        The labels will be integers from 0 to n-1. We introduce ancilla
        variables in order to reduce the degree of the PUSO to a QUBO. The
        solution to the PUSO can be read from the solution to the QUBO by
        using the ``convert_solution`` method.

        Parameters
        ----------
        lam : function (optional, defaults to None).
            If ``lam`` is None, the function ``PUBO.default_lam`` will be used.
            ``lam`` is the penalty factor to introduce in order to enforce the
            ancilla constraints. When we reduce the degree of the PUSO to a
            QUBO, we add penalties to the QUBO in order to enforce ancilla
            variable constraints. These constraints will be multiplied by
            ``lam(v)``, where ``v`` is the value associated with the term that
            it is reducing. For example, a term ``(0, 1, 2): 3`` in the Hquso
            may be reduced to a term ``(0, 3): 3`` for the QUSO, and then the
            fact that ``3`` should be the product of ``1`` and ``2`` will be
            enforced with a penalty weight ``lam(3)``.
        pairs : set (optional, defaults to None).
            A set of tuples of variable pairs to prioritize pairing together in
            to degree reduction. If a pair in ``pairs`` is found together in
            the PUSO, it will be chosen as a pair to reduce to a single
            ancilla. You should supply this parameter if you have a good idea
            of an efficient way to reduce the degree of the PUSO. If ``pairs``
            is None, then it will be the empty set ``set()``. In other words,
            no variable pairs will be prioritized, and instead variable pairs
            will be chosen to reduce to an ancilla bases solely on frequency
            of occurrance.

        Return
        ------
        Q : qubovert.utils.QUBOMatrix object.
            The upper triangular QUBO matrix, an QUBOMatrix object.
            For most practical purposes, you can use QUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUBOMatrix)``.

        """
        return self._create_pubo().to_qubo(lam, pairs)

    def to_quso(self, lam=None, pairs=None):
        """to_quso.

        Create and return upper triangular QUSO representing the problem.
        The labels will be integers from 0 to n-1. We introduce ancilla
        variables in order to reduce the degree of the PUSO to a QUSO. The
        solution to the PUSO can be read from the solution to the QUSO by
        using the ``convert_solution`` method.

        Parameters
        ----------
        lam : function (optional, defaults to None).
            If ``lam`` is None, the function ``PUBO.default_lam`` will be used.
            ``lam`` is the penalty factor to introduce in order to enforce the
            ancilla constraints. When we reduce the degree of the PUSO to a
            QUSO, we add penalties to the QUSO in order to enforce ancilla
            variable constraints. These constraints will be multiplied by
            ``lam(v)``, where ``v`` is the value associated with the term that
            it is reducing. For example, a term ``(0, 1, 2): 3`` in the PUSO
            may be reduced to a term ``(0, 3): 3`` for the QUSO, and then the
            fact that ``3`` should be the product of ``1`` and ``2`` will be
            enforced with a penalty weight ``lam(3)``.
        pairs : set (optional, defaults to None).
            A set of tuples of variable pairs to prioritize pairing together in
            to degree reduction. If a pair in ``pairs`` is found together in
            the PUSO, it will be chosen as a pair to reduce to a single
            ancilla. You should supply this parameter if you have a good idea
            of an efficient way to reduce the degree of the PUSO. If ``pairs``
            is None, then it will be the empty set ``set()``. In other words,
            no variable pairs will be prioritized, and instead variable pairs
            will be chosen to reduce to an ancilla bases solely on frequency
            of occurrance.

        Return
        ------
        L : qubovert.utils.QUSOMatrix object.
            The upper triangular QUSO matrix, an QUSOMatrix object.
            For most practical purposes, you can use QUSOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUSOMatrix)``.

        """
        if self.degree <= 2:
            return QUSOMatrix(self._to_puso())
        return super().to_quso(lam, pairs)

    def convert_solution(self, solution, spin=True):
        """convert_solution.

        Convert the solution to the integer labeled PUSO to the solution to
        the originally labeled PUSO.

        Parameters
        ----------
        solution : iterable or dict.
            The PUSO, PUSO, QUSO, or QUSO solution output. The PUSO
            solution output is either a list or tuple where indices specify the
            label of the variable and the element specifies whether it's 0 or 1
            for PUSO (or 1 or -1 for QUSO), or it can be a dictionary that
            maps the label of the variable to is value. The QUSO/QUSO
            solution output includes the assignment for the ancilla variables
            used to reduce the degree of the PUSO.
        spin : bool (optional, defaults to True).
            `spin` indicates whether ``solution`` is the solution to the
            boolean {0, 1} formulation of the problem or the spin {1, -1}
            formulation of the problem. This parameter usually does not matter,
            and it will be ignored if possible. The only time it is used is if
            ``solution`` contains all 1's. In this case, it is unclear whether
            ``solution`` came from a spin or boolean formulation of the
            problem, and we will figure it out based on the ``spin`` parameter.

        Return
        ------
        res : dict.
            Maps spin variable labels to their PUSO solutions values
            {1, -1}.

        Example
        -------
        >>> puso = PUSO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> puso
        {('a',): 5, ('a', 0, 1): -2, (): -1.5}
        >>> H = puso.to_puso()
        >>> H
        {(0,): 5, (0, 1): -2, (): -1.5}
        >>> puso.convert_solution({0: 1, 1: -1, 2: 1})
        {'a': 1, 0: -1, 1: 1}

        In the next example, notice that we introduce ancilla variables to
        represent that ```(0, 1)`` term. See the ``to_quso`` method for more
        info.

        >>> puso = PUSO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> puso.mapping
        {'a': 0, 0: 1, 1: 2}
        >>> L = puso.to_quso(3)
        >>> L
        {(0,): 5, (0, 3): -2, (): -2.25,
         (1, 2): 3/4, (2, 3): 3/4, (1, 3): 3/4, (1,): 3/2, (2,): 3/2, (3,): 3}
        >>> puso.convert_solution({0: 1, 1: -1, 2: 1, 2: -1})
        {'a': 1, 0: -1, 1: 1}

        Notes
        -----
        We take ignore the ancilla variable assignments when we convert the
        solution. For example if the conversion from PUSO to QUSO introduced
        an ancilla varable ``z = xy`` where ``x`` and ``y`` are variables of
        the PUSO, then ``solution`` must have values for ``x``, ``y``, and
        ``z``. If the QUSO solver found that ``x = 1``, ``y = -1``, and
        ``z = 1``, then the constraint that ``z = xy`` is not satisfied (one
        possible cause for this is if the ``lam`` argument in ``to_quso`` is
        too small). ``convert_solution`` will return that ``x = 1`` and
        ``y = -1`` and ignore the value of ``z``.

        """
        # this works for converting a solution to the pubo, qubo, puso, or
        # quso formulations, since in the to_quso function all ancilla
        # variables are labeled with integers >= self.num_binary_variables.
        return QUSO.convert_solution(self, solution, spin)

    @staticmethod
    def _check_key_valid(key):
        """_check_key_valid.

        Internal method to check if an input key to the dictionary is valid.
        Checks to see if ``key`` is a tuple.

        Parameters
        ----------
        key : anything, but must be a tuple to be valid.

        Returns
        -------
        None.

        Raises
        ------
        KeyError if the key is invalid.

        """
        # override PUSOMatrix._check_key_valid to allow for noninteger keys.
        if not isinstance(key, tuple):
            raise KeyError(
                "Key formatted incorrectly, must be tuple")
