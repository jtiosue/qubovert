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

"""_ising.py.

Contains the Ising class. See ``help(qubovert.Ising)``.

"""

from .utils import BO, IsingMatrix


__all__ = 'Ising',


class Ising(BO, IsingMatrix):
    """Ising.

    Class to manage converting general Ising problems to and from their
    QUBO and Ising formluations.

    This class deals with Isings that have binary labels that do not range from
    0 to n-1. If your labels are nonnegative integers, consider using
    ``qubovert.utils.IsingMatrix``. Note that it is generally
    more efficient to initialize an empty Ising object and then build the
    Ising, rather than initialize a Ising object with an already built dict.

    Ising inherits some methods and attributes the ``IsingMatrix`` class. See
    ``help(qubovert.utils.IsingMatrix)``.

    Ising inherits some methods and attributes the ``BO`` class. See
    ``help(qubovert.utils.BO)``.

    Example usage
    -------------
    >>> ising = Ising()
    >>> ising[('a',)] += 5
    >>> ising[(0, 'a')] -= 2
    >>> ising -= 1.5
    >>> ising
    {('a',): 5, ('a', 0): -2, (): -1.5}

    >>> ising = Ising({('a',): 5, (0, 'a'): -2, (): -1.5})
    >>> ising
    {('a',): 5, ('a', 0): -2, (): -1.5}
    >>> L = ising.to_ising()
    >>> L
    {(0,): 5, (0, 1): -2, (): -1.5}
    >>> ising.convert_solution({0: 1, 1: 0})
    {'a': 1, 0: 0}

    Note 1
    ------
    Note that keys will end up sorted by their hash. Hashes will not be
    consistent across Python sessions (unless they are integers)! For example,
    both of the following can happen:

    >>> print(Ising({('a', 0): 1, (0, 1): -1}))
    {('a', 0): 1, (0, 1): -1}

    or

    >>> print(Ising({('a', 0): 1, (0, 1): -1}))
    {(0, 'a'): 1, (0, 1): -1}

    But the following will never happen:

    >>> print(Ising({('a', 0): 1, (0, 1): -1}))
    {('a', 0): 1, (1, 0): -1}

    Ie integers will always be correctly sorted.

    Note 2
    ------
    For efficiency, many internal variables including mappings are computed as
    the problemis being built. This can cause these
    values to be wrong for some specific situations. Calling ``refresh``
    will rebuild the dictionary, resetting all of the values. See
    ``help(Ising.refresh)``

    Examples
    --------
    >>> from qubovert import Ising
    >>> L = Ising()
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

        This class deals with Isings that have binary labels that do not range
        from 0 to n-1. If your labels are nonnegative integers, consider using
        ``qubovert.utils.IsingMatrix``. Note that it is generally more
        efficient to initialize an empty Ising object and then build the Ising,
        rather than initialize a Ising object with an already built dict.

        Parameters
        ----------
        args and kwargs : define a dictionary with ``dict(*args, **kwargs)``.
            The dictionary will be initialized to follow all the convensions of
            the class.

        Examples
        -------
        >>> ising = Ising()
        >>> ising[('a',)] += 5
        >>> ising[(0, 'a')] -= 2
        >>> ising -= 1.5
        >>> ising
        {('a',): 5, ('a', 0): -2, (): -1.5}

        >>> ising = Ising({('a',): 5, (0, 'a'): -2, (): -1.5})
        >>> ising
        {('a',): 5, ('a', 0): -2, (): -1.5}

        """
        BO.__init__(self, *args, **kwargs)
        IsingMatrix.__init__(self, *args, **kwargs)

    def to_ising(self):
        """to_ising.

        Create and return upper triangular Ising representing the problem.
        The labels will be integers from 0 to n-1.

        Return
        -------
        L : qubovert.utils.IsingMatrix object.
            The upper triangular Ising matrix, a IsingMatrix object.
            For most practical purposes, you can use IsingMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.IsingMatrix)``.

        """
        L = IsingMatrix()

        for k, v in self.items():
            key = tuple(self._mapping[i] for i in k)
            L[key] += v

        return L

    def convert_solution(self, solution):
        """convert_solution.

        Convert the solution to the integer labeled Ising to the solution to
        the originally labeled Ising.

        Parameters
        ----------
        solution : iterable or dict.
            The QUBO or Ising solution output. The Ising solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or -1 or 1 for Ising), or it can be a dictionary that maps the
            label of the variable to is value.

        Return
        -------
        res : dict.
            Maps binary variable labels to their Ising solutions values {0, 1}.

        Example
        -------
        >>> ising = Ising({('a',): 1, ('a', 'b'): -2, ('c',): -1})
        >>> L = ising.to_ising()
        >>> L
        {(0,): 1, (0, 1): -2, (2,): -1}
        >>> solution = solve_ising(L)  # any solver you want
        >>> solution
        [1, 1, -1]  # or {0: 1, 1: 1, 2: -1}
        >>> sol = ising.convert_solution(solution)
        >>> sol
        {'a': 1, 'b': 1, 'c': -1}

        >>> ising = Ising({('a',): 1, ('a', 'b'): -2, ('c',): -1})
        >>> L = ising.to_qubo()
        >>> solution = solve_ising(L)  # any solver you want
        >>> solution
        [1, 1, 0]  # or {0: 1, 1: 1, 2: 0}
        >>> sol = ising.convert_solution(solution)
        >>> sol
        {'a': 1, 'b': 1, 'c': -1}

        """
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
        # override IsingMatrix._check_key_valid to allow for noninteger keys.
        invalid = (
            not isinstance(key, tuple) or
            len(set(x for x in set(key) if key.count(x) % 2)) > 2
        )
        if invalid:
            raise KeyError(
                "Key formatted incorrectly, must be tuple of <= 2 unique "
                "elements")
