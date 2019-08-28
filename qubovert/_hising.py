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

"""_hising.py.

Contains the HIsing class. See ``help(qubovert.HIsing)``.

"""

from .utils import BO, HIsingMatrix, hising_to_pubo
from . import PUBO


__all__ = 'HIsing',


class HIsing(BO, HIsingMatrix):
    """HIsing.

    Class to manage converting general HIsing problems to and from their
    HIing, PUBO, Ising, and QUBO formluations. In general, this class
    deals with unconstrained optimization problems that have arbitrary degree.
    To convert this to a Ising (see ``to_ising``) or QUBO (``to_qubo``) we have
    to introduce ancilla variables. The ``convert_solution`` method deals with
    converting a solution to the problem with ancilla variables back to the
    solution to the original problem.

    This class deals with HIsings that have binary labels that do not range
    from 0 to n-1. Note that it is generally
    more efficient to initialize an empty HIsing object and then build the
    HIsing, rather than initialize a HIsing object with an already built dict.

    HIsing inherits some methods and attributes the ``HIsingMatrix`` class. See
    ``help(qubovert.utils.HIsingMatrix)``.

    HIsing inherits some methods and attributes the ``BO`` class. See
    ``help(qubovert.utils.BO)``.

    Example usage
    -------------
    >>> hising = HIsing()
    >>> hising[('a',)] += 5
    >>> hising[(0, 'a', 1)] -= 2
    >>> hising -= 1.5
    >>> hising
    {('a',): 5, ('a', 0, 1): -2, (): -1.5}

    >>> hising = HIsing({('a',): 5, (0, 'a', 1): -2, (): -1.5})
    >>> hising
    {('a',): 5, ('a', 0, 1): -2, (): -1.5}
    >>> H = hising.to_hising()
    >>> H
    {(0,): 5, (0, 1): -2, (): -1.5}
    >>> hising.convert_solution({0: 1, 1: -1, 2: 1})
    {'a': 1, 0: -1, 1: 1}

    In the next example, notice that we introduce ancilla variables to
    represent that ```(0, 1)`` term. See the ``to_ising`` method for more info.

    >>> hising = HIsing({('a',): 5, (0, 'a', 1): -2, (): -1.5})
    >>> hising.mapping
    {'a': 0, 0: 1, 1: 2}
    >>> L = hising.to_ising()
    >>> L
    {(0,): 0.75, (): 11.25, (3,): 8.5, (0, 1): 4.25, (1,): -4.25, (0, 3): -8.5,
     (1, 3): -8.5, (2, 3): -4.0, (1, 2): 2.0, (0, 2): 2.0, (2,): -2.0}
    >>> hising.convert_solution({0: 1, 1: -1, 2: 1, 3: -1})
    {'a': 1, 0: -1, 1: 1}

    Note 1
    ------
    Note that keys will end up sorted by their hash. Hashes will not be
    consistent across Python sessions (unless they are integers)! For example,
    both of the following can happen:

    >>> print(HIsing({('a', 0): 1, (0, 1): -1}))
    {('a', 0): 1, (0, 1): -1}

    or

    >>> print(HIsing({('a', 0): 1, (0, 1): -1}))
    {(0, 'a'): 1, (0, 1): -1}

    But the following will never happen:

    >>> print(HIsing({('a', 0): 1, (0, 1): -1}))
    {('a', 0): 1, (1, 0): -1}

    Ie integers will always be correctly sorted.

    Note 2
    ------
    For efficiency, many internal variables including mappings are computed as
    the problemis being built. This can cause these
    values to be wrong for some specific situations. Calling ``refresh``
    will rebuild the dictionary, resetting all of the values. See
    ``help(HIsing.refresh)``

    Examples
    --------
    >>> from qubovert import HIsing
    >>> H = HIsing()
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

        This class deals with HIsings that have binary labels that do not range
        from 0 to n-1. Note that it is generally more efficient to initialize
        an empty HIsing object and then build the HIsing, rather than
        initialize a HIsing object with an already built dict.

        Parameters
        ----------
        args and kwargs define a dictionary. The dictionary will be initialized
        to follow all the convensions of the class.

        Examples
        -------
        >>> hising = HIsing()
        >>> hising[('a',)] += 5
        >>> hising[(0, 'a')] -= 2
        >>> hising -= 1.5
        >>> hising
        {('a',): 5, ('a', 0): -2, (): -1.5}

        >>> hising = HIsing({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> hising
        {('a',): 5, ('a', 0, 1): -2, (): -1.5}

        """
        BO.__init__(self, *args, **kwargs)
        HIsingMatrix.__init__(self, *args, **kwargs)

    def to_pubo(self, deg=None, lam=None):
        """to_pubo.

        Create and return upper triangular degree ``deg`` PUBO representing the
        HIsing problem. The labels will be integers from 0 to n-1.

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

        """
        return PUBO(hising_to_pubo(self._to_hising())).to_pubo(deg, lam)

    def _to_hising(self):
        """to_hising.

        Internal helper method.

        Create and return upper triangular degree HIsing representing
        the problem. The labels will be integers from 0 to n-1.

        Return
        -------
        H : qubovert.utils.HIsingMatrix object.
            The upper triangular HIsing matrix, a HIsingMatrix object.
            For most practical purposes, you can use HIsingMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.HIsingMatrix)``.

        """
        H = HIsingMatrix()
        for k, v in self.items():
            key = tuple(sorted(self._mapping[i] for i in k))
            H[key] += v
        return H

    def to_hising(self, deg=None, lam=None):
        """to_hising.

        Create and return upper triangular degree ``deg`` HIsing representing
        the problem. The labels will be integers from 0 to n-1.

        Parameters
        ----------
        deg : int >= 0 (optional, defaults to None).
            The degree of the final HIsing. If ``deg`` is None, then the degree
            of the output HIsing will be the same as the degree of ``self``,
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
        H : qubovert.utils.HIsingMatrix object.
            The upper triangular HIsing matrix, a HIsingMatrix object.
            For most practical purposes, you can use HIsingMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.HIsingMatrix)``.

        """
        if deg is None or deg >= self.degree:
            return self._to_hising()
        return PUBO(hising_to_pubo(self._to_hising())).to_hising(deg, lam)

    def to_ising(self, lam=None):
        """to_ising.

        Create and return upper triangular Ising representing the problem.
        The labels will be integers from 0 to n-1. We introduce ancilla
        variables in order to reduce the degree of the HIsing to a Ising. The
        solution to the HIsing can be read from the solution to the Ising by
        using the ``convert_solution`` method.

        Parameters
        ----------
        lam : function (optional, defaults to None).
            If ``lam`` is None, the function ``PUBO.default_lam`` will be used.
            ``lam`` is the penalty factor to introduce in order to enforce the
            ancilla constraints. When we reduce the degree of the HIsing to a
            Ising, we add penalties to the Ising in order to enforce ancilla
            variable constraints. These constraints will be multiplied by
            ``lam(v)``, where ``v`` is the value associated with the term that
            it is reducing. For example, a term ``(0, 1, 2): 3`` in the Hising
            may be reduced to a term ``(0, 3): 3`` for the Ising, and then the
            fact that ``3`` should be the product of ``1`` and ``2`` will be
            enforced with a penalty weight ``lam(3)``.

        Return
        -------
        L : qubovert.utils.IsingMatrix object.
            The upper triangular Ising matrix, an IsingMatrix object.
            For most practical purposes, you can use IsingMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.IsingMatrix)``.

        """
        return PUBO(hising_to_pubo(self._to_hising())).to_ising(lam)

    def convert_solution(self, solution):
        """convert_solution.

        Convert the solution to the integer labeled HIsing to the solution to
        the originally labeled HIsing.

        Parameters
        ----------
        solution : iterable or dict.
            The HIsing, HIsing, Ising, or Ising solution output. The HIsing
            solution output is either a list or tuple where indices specify the
            label of the variable and the element specifies whether it's 0 or 1
            for HIsing (or -1 or 1 for Ising), or it can be a dictionary that
            maps the label of the variable to is value. The Ising/Ising
            solution output includes the assignment for the ancilla variables
            used to reduce the degree of the HIsing.

        Return
        -------
        res : dict.
            Maps binary variable labels to their HIsing solutions values
            {-1, 1}.

        Example
        -------
        >>> hising = HIsing({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> hising
        {('a',): 5, ('a', 0, 1): -2, (): -1.5}
        >>> H = hising.to_hising()
        >>> H
        {(0,): 5, (0, 1): -2, (): -1.5}
        >>> hising.convert_solution({0: 1, 1: -1, 2: 1})
        {'a': 1, 0: -1, 1: 1}

        In the next example, notice that we introduce ancilla variables to
        represent that ```(0, 1)`` term. See the ``to_ising`` method for more
        info.

        >>> hising = HIsing({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> hising.mapping
        {'a': 0, 0: 1, 1: 2}
        >>> L = hising.to_ising(3)
        >>> L
        {(0,): 5, (0, 3): -2, (): -2.25,
         (1, 2): 3/4, (2, 3): 3/4, (1, 3): 3/4, (1,): 3/2, (2,): 3/2, (3,): 3}
        >>> hising.convert_solution({0: 1, 1: -1, 2: 1, 2: -1})
        {'a': 1, 0: -1, 1: 1}

        Notes
        -----
        We take ignore the ancilla variable assignments when we convert the
        solution. For example if the conversion from HIsing to Ising introduced
        an ancilla varable ``z = xy`` where ``x`` and ``y`` are variables of
        the HIsing, then ``solution`` must have values for ``x``, ``y``, and
        ``z``. If the Ising solver found that ``x = 1``, ``y = -1``, and
        ``z = 1``, then the constraint that ``z = xy`` is not satisfied (one
        possible cause for this is if the ``lam`` argument in ``to_ising`` is
        too small). ``convert_solution`` will return that ``x = 1`` and
        ``y = -1`` and ignore the value of ``z``.

        """
        # this works for converting a solution to the pubo, qubo, hising, or
        # ising formulations, since in the to_ising function all ancilla
        # variables are labeled with integers >= self.num_binary_variables.
        return {
            self._reverse_mapping[i]: 1 if solution[i] == 1 else -1
            for i in range(self.num_binary_variables)
        }

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
        # override HIsingMatrix._check_key_valid to allow for noninteger keys.
        if not isinstance(key, tuple):
            raise KeyError(
                "Key formatted incorrectly, must be tuple")
