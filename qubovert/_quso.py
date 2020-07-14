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

"""_quso.py.

Contains the QUSO class. See ``help(qubovert.QUSO)``.

"""

from .utils import (
    BO, QUSOMatrix, PUSOMatrix, is_solution_spin, boolean_to_spin
)


__all__ = 'QUSO',


class QUSO(BO, QUSOMatrix):
    """QUSO.

    Class to manage converting general QUSO problems to and from their
    QUBO and QUSO formluations.

    This class deals with QUSOs that have spin labels that do not range from
    0 to n-1. If your labels are nonnegative integers, consider using
    ``qubovert.utils.QUSOMatrix``. Note that it is generally
    more efficient to initialize an empty QUSO object and then build the
    QUSO, rather than initialize a QUSO object with an already built dict.

    QUSO inherits some methods and attributes the ``QUSOMatrix`` class. See
    ``help(qubovert.utils.QUSOMatrix)``.

    QUSO inherits some methods and attributes the ``BO`` class. See
    ``help(qubovert.utils.BO)``.

    Example
    -------
    >>> quso = QUSO()
    >>> quso[('a',)] += 5
    >>> quso[(0, 'a')] -= 2
    >>> quso -= 1.5
    >>> quso
    {('a',): 5, ('a', 0): -2, (): -1.5}

    >>> quso = QUSO({('a',): 5, (0, 'a'): -2, (): -1.5})
    >>> quso
    {('a',): 5, ('a', 0): -2, (): -1.5}
    >>> L = quso.to_quso()
    >>> L
    {(0,): 5, (0, 1): -2, (): -1.5}
    >>> quso.convert_solution({0: 1, 1: 0})
    {'a': 1, 0: 0}

    Note
    ----
    For efficiency, many internal variables including mappings are computed as
    the problemis being built. This can cause these
    values to be wrong for some specific situations. Calling ``refresh``
    will rebuild the dictionary, resetting all of the values. See
    ``help(QUSO.refresh)``

    Examples
    --------
    >>> from qubovert import QUSO
    >>> L = QUSO()
    >>> L[('a',)] += 1
    >>> L, L.mapping, L.reverse_mapping
    {('a',): 1}, {'a': 0}, {0: 'a'}
    >>> L[('a',)] -= 1
    >>> L, L.mapping, L.reverse_mapping
    {}, {'a': 0}, {0: 'a'}
    >>> L.refresh()
    >>> L, L.mapping, L.reverse_mapping
    {}, {}, {}

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        This class deals with QUSOs that have spin labels that do not range
        from 0 to n-1. If your labels are nonnegative integers, consider using
        ``qubovert.utils.QUSOMatrix``. Note that it is generally more
        efficient to initialize an empty QUSO object and then build the QUSO,
        rather than initialize a QUSO object with an already built dict.

        Parameters
        ----------
        argguments : define a dictionary with ``dict(*args, **kwargs)``.
            The dictionary will be initialized to follow all the convensions of
            the class.

        Examples
        --------
        >>> quso = QUSO()
        >>> quso[('a',)] += 5
        >>> quso[(0, 'a')] -= 2
        >>> quso -= 1.5
        >>> quso
        {('a',): 5, ('a', 0): -2, (): -1.5}

        >>> quso = QUSO({('a',): 5, (0, 'a'): -2, (): -1.5})
        >>> quso
        {('a',): 5, ('a', 0): -2, (): -1.5}

        """
        BO.__init__(self, *args, **kwargs)
        QUSOMatrix.__init__(self, *args, **kwargs)

    def to_quso(self):
        """to_quso.

        Create and return upper triangular QUSO representing the problem.
        The labels will be integers from 0 to n-1.

        Return
        ------
        L : qubovert.utils.QUSOMatrix object.
            The upper triangular QUSO matrix, a QUSOMatrix object.
            For most practical purposes, you can use QUSOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUSOMatrix)``.

        """
        L = QUSOMatrix()

        for k, v in self.items():
            key = tuple(self._mapping[i] for i in k)
            L[key] += v

        return L

    def to_puso(self):
        """to_puso.

        Since the model is already a QUSO, ``self.to_puso`` will simply
        return ``qubovert.utils.PUSOMatrix(self.to_quso())``.

        Return
        ------
        H : qubovert.utils.PUSOMatrix object.
            The upper triangular PUSO matrix, a PUSOMatrix object.
            For most practical purposes, you can use PUSOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.PUSOMatrix)``.

        """
        return PUSOMatrix(self.to_quso())

    def convert_solution(self, solution, spin=True):
        """convert_solution.

        Convert the solution to the integer labeled QUSO to the solution to
        the originally labeled QUSO.

        Parameters
        ----------
        solution : iterable or dict.
            The QUBO or QUSO solution output. The QUSO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or 1 or -1 for QUSO), or it can be a dictionary that maps the
            label of the variable to is value.
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
            Maps spin variable labels to their QUSO solutions values {1, -1}.

        Example
        -------
        >>> quso = QUSO({('a',): 1, ('a', 'b'): -2, ('c',): -1})
        >>> L = quso.to_quso()
        >>> L
        {(0,): 1, (0, 1): -2, (2,): -1}
        >>> solution = solve_quso(L)  # any solver you want
        >>> solution
        [1, 1, -1]  # or {0: 1, 1: 1, 2: -1}
        >>> sol = quso.convert_solution(solution)
        >>> sol
        {'a': 1, 'b': 1, 'c': -1}

        >>> quso = QUSO({('a',): 1, ('a', 'b'): -2, ('c',): -1})
        >>> L = quso.to_qubo()
        >>> solution = solve_quso(L)  # any solver you want
        >>> solution
        [0, 0, 1]  # or {0: 0, 1: 0, 2: 1}
        >>> sol = quso.convert_solution(solution)
        >>> sol
        {'a': 1, 'b': 1, 'c': -1}

        """
        if not is_solution_spin(solution, spin):
            solution = boolean_to_spin(solution)
        return {
            self._reverse_mapping[i]: solution[i]
            for i in range(self.num_binary_variables)
        }

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
        # override QUSOMatrix._check_key_valid to allow for noninteger keys.
        invalid = (
            not isinstance(key, tuple) or
            len(set(x for x in set(key) if key.count(x) % 2)) > 2
        )
        if invalid:
            raise KeyError(
                "Key formatted incorrectly, must be tuple of <= 2 unique "
                "element. See PUSO for arbitrary number of unique elements.")
