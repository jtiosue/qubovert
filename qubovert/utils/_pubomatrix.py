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

"""_pubomatrix.py.

This file contains the PUBOMatrix object.

"""

from . import (
    DictArithmetic, ordering_key,
    pubo_value, solve_pubo_bruteforce
)


__all__ = 'PUBOMatrix',


class PUBOMatrix(DictArithmetic):
    """PUBOMatrix.

    ``PUBOMatrix`` inherits some methods from ``DictArithmetic``, see
    ``help(qubovert.utils.DictArithmetic)``.

    A class to handle PUBO matrices. It is the same thing as a dictionary
    with some methods modified. Note that each key must be a tuple of integers
    >= 0.

    Note that below we only consider keys that are of length 1 or 2, but they
    can in general be arbitrarily long! See ``qubovert.utils.QUBOMatrix``
    for an object that restricts the length to <= 2.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = PUBOMatrix()
    >>> print(d[(0,)]) # will print 0
    >>> d[(0,)] += 1
    >>> print(d) # will print {(0,): 1}

    Compared to an ordinary dictionary.

    >>> g = dict()
    >>> print(g[(0,)]) # will raise KeyError
    >>> g[(0,)] += 1 # will raise KeyError, since (0,) was never set

    One method of PUBOMatrix is that it will always keep the PUBO
    upper triangular! Consider the following example:

    >>> d = PUBOMatrix()
    >>> d[(1, 0)] += 2
    >>> print(d)
    >>> # will print {(0, 1): 2}

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = PUBOMatrix()
    >>> d[(0,)] += 1
    >>> d[(0,)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize PUBOMatrix with a previous dictionary
    it will be reinitialized to ensure that the PUBOMatrix is upper
    triangular and contains no zero values. Consider the following example:

    >>> d = PUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0, (2, 0, 1): 1})
    >>> print(d) # will print {(0,): 1, (0, 1): 2, (0, 1, 2): 1}

    We also change the update method so that it follows all the conventions.

    >>> d = PUBOMatrix({(0, 0): 1, (0, 1): 2})
    >>> d.update({(0,): 0, (1, 0): 1, (1, 2): -1})
    >>> print(d)  # will print {(0, 1): 1, (1, 2): -1}

    We also include arithmetic, addition, subtraction, scalar division,
    multiplication, and all those in place. For example,

    >>> d = PUBOMatrix({(0, 0): 1, (0, 1): -2})
    >>> g = d + {(0,): -1}
    >>> print(g) # will print {(0, 1): -2}
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8}
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {}

    >>> d = PUBOMatrix({(0, 0): 1, (0, 1): -1})
    >>> g = {(0,): -1, (2,): 1}
    >>> d *= g
    >>> print(d)
    {(0,): -1, (0, 2): 1, (0, 1): 1, (0, 1, 2): -1}

    >>> d = PUBOMatrix({(0, 0): 1, (0, 1): -1})
    >>> print(d ** 2 == d * d)
    True

    Adding or subtracting constants will update the () element of the
    dict.

    >>> d = PUBOMatrix()
    >>> d += 5
    >>> print(d)
    {(): 5}

    Finally, if you try to access a key out of order, it will sort the key. Be
    careful with this, it can cause unexpected behavior if you don't know it.
    For example,

    >>> d = PUBOMatrix()
    >>> d[(0, 1)] += 2
    >>> print(d[(1, 0)])  # will print 2

    >>> d = PUBOMatrix()
    >>> d[(0, 0)] += 2
    >>> print(d[(0,)])  # will print 2

    >>> d = PUBOMatrix()
    >>> d[(0, 0, 3, 2, 2)] += 2
    >>> print(d)  # will print {(0, 2, 3): 2}
    >>> print(d[(0, 3, 2)])  # will print 2

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        This class deals with Binary Optimization matrices. See child classes
        or ``qubovert.utils.DictArithmetic`` for details on the inputs.

        Parameters
        ----------
        arguments : parameters.
            Defined in child classes or in ``qubovert.utils.DictArithmetic``.

        """
        self._degree = -float("inf")
        self._variables, self._num_binary_variables = set(), 0
        super().__init__(*args, **kwargs)

    def refresh(self):
        """refresh.

        For efficiency, the internal variables for ``degree``,
        ``num_binary_variables``, ``max_index`` are computed as the dictionary
        is being built (and in subclasses such as ``qubovert.PUBO``, properties
        such as ``mapping`` and ``reverse_mapping``). This can cause these
        values to be wrong for some specific situations. Calling ``refresh``
        will rebuild the dictionary, resetting all of the values.

        Examples
        --------
        >>> from qubovert.utils import PUBOMatrix
        >>> P = PUBOMatrix()
        >>> P[(0,)] += 1
        >>> P, P.degree, P.num_binary_variables
        {(0,): 1}, 1, 1
        >>> P[(0,)] -= 1
        >>> P, P.degree, P.num_binary_variables
        {}, 1, 1
        >>> P.refresh()
        >>> P, P.degree, P.num_binary_variables
        {}, 0, 0

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
        d = self.copy()
        super().clear()
        self.__init__(d)

    def clear(self):
        """clear.

        For efficiency, the internal variables for ``degree``,
        ``num_binary_variables``, ``max_index`` are computed as the dictionary
        is being built (and in subclasses such as ``qubovert.PUBO``, properties
        such as ``mapping`` and ``reverse_mapping``). This can cause these
        values to be wrong for some specific situations. Thus, when we clear,
        we also need to reset all of these cached values. This function
        remove all the elments from ``self`` and resets the cached values.

        """
        super().clear()
        self.__init__()

    @property
    def degree(self):
        """degree.

        Return the degree of the problem.

        Return
        ------
        deg : int.

        """
        return self._degree

    @property
    def variables(self):
        """variables.

        Return a set of all the variables in the dict.

        Returns
        -------
        res : set.

        """
        return self._variables.copy()

    @property
    def offset(self):
        """offset.

        Get the part that does not depend on any variables. Ie the value
        corresponding to the () key.

        Returns
        -------
        offset : float.

        """
        return self[()]

    @property
    def num_binary_variables(self):
        """num_binary_variables.

        Return the number of binary variables in the problem.

        Return
        ------
        n : int.
            Number of binary variables in the problem.

        """
        return self._num_binary_variables

    @property
    def max_index(self):
        """max_index.

        Returns the maximum label index of the problem. If the problem is
        labeled with integers from 0 to ``n-1``, then ``max_index`` will give
        the same result as ``num_binary_variables - 1``.

        Return
        ------
        n : int or None.
            Max label index of the problem dictionary. If the dict is empty,
            then this returns None.

        """
        return max(self._variables) if self._variables else None

    @classmethod
    def squash_key(cls, key):
        """squash_key.

        Will convert the input key into the standard form for PUBOMatrix /
        QUBOMatrix. It will get rid of duplicates and sort. This method
        will check to see if the input key is valid.

        Parameters
        ----------
        key : tuple of integers.

        Return
        ------
        k : tuple of integers.
            A sorted and squashed version of ``key``.

        Raises
        ------
        KeyError if the key is invalid.

        Example
        -------
        >>> squash_key((0, 4, 0, 3, 3, 2))
        >>> (0, 2, 3, 4)

        """
        # if f is not None, then it is the squashed key (see QUBOMatrix)
        f = cls._check_key_valid(key)
        # use ordering_key here because in subclasses x may not always
        # be an int.
        return f or tuple(sorted(set(key), key=ordering_key))

    @staticmethod
    def _check_key_valid(key):
        """_check_key_valid.

        Internal method to check if an input key to the dictionary is valid.
        Checks to see if ``key`` is a tuple of non negative integers.

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
        incorrect_format = (
            not isinstance(key, tuple) or
            any(not isinstance(k, int) or k < 0 for k in key)
        )

        if incorrect_format:
            raise KeyError(
                "Key formatted incorrectly, must be tuple of non negative "
                "integers")

    def __getitem__(self, key):
        """__getitem__.

        Overrides the dict.__getitem__ command so that a KeyError is not
        thrown if `key` is not in the dictionary. The key will also be
        reformatted according to ``squash_key`` before indexing.

        Parameters
        ----------
        key : tuple of integers.
            Element of the dictionary.

        Return
        ------
        value : numeric
            the value corresponding to the key if the key is in the dictionary,
            otherwise returns 0.

        """
        return super().__getitem__(self.__class__.squash_key(key))

    def __setitem__(self, key, value):
        """__setitem__.

        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements will ever
        have zero value. Additionally, this method will keep self upper
        triangular, so if key[0] > key[1], then we will call
        ``__setitem__((key[1], key[0]), value)``. Finally, keys
        will be squashed, see ``squash_keys``.

        Parameters
        ----------
        key : tuple of integers.
            Element of the dictionary.
        value : numeric.
            Value corresponding to the key.

        """
        k = self.__class__.squash_key(key)
        if value:
            self._degree = max(self._degree, len(k))
            for i in filter(lambda x: x not in self._variables, k):
                self._variables.add(i)
                self._num_binary_variables += 1
        super().__setitem__(k, value)

    def is_solution_valid(self, solution):
        """is_solution_valid.

        Included for consistency with other problem classes. Always returns
        True since this is an unconstrainted problem.

        Parameters
        ----------
        solution : iterable or dict.

        Return
        ------
        valid : bool.
            Always returns True.

        """
        return True

    def solve_bruteforce(self, all_solutions=False):
        """solve_bruteforce.

        Solve the problem bruteforce. THIS SHOULD NOT BE USED FOR LARGE
        PROBLEMS! This is the exact same as calling
        ``qubovert.utils.solve_pubo_bruteforce(
            self, all_solutions, self.is_solution_valid)[1]``.

        Parameters
        ----------
        all_solutions : bool.
            See the description of the ``all_solutions`` parameter in
            ``qubovert.utils.solve_pubo_bruteforce``.

        Return
        ------
        res : the second element of the two element tuple that is returned from
            ``qubovert.utils.solve_pubo_bruteforce``.

        """
        return solve_pubo_bruteforce(self,
                                     all_solutions, self.is_solution_valid)[1]

    def value(self, x):
        r"""value.

        Find the value of
        :math:`\sum_{i,...,j} P_{i...j} x_{i} ... x_{j}`. Calling
        ``self.value(x)`` is the same as calling
        ``qubovert.utils.pubo_value(x, self)``.

        Parameters
        ----------
        x : dict or iterable.
            Maps boolean variable indices to their boolean values, 0 or 1. Ie
            ``x[i]`` must be the boolean value of variable i.

        Return
        ------
        value : float.
            The value of the PUBO with the given assignment `x`. Ie

        Example
        -------
        >>> from qubovert.utils import QUBOMatrix, PUBOMatrix
        >>> from qubovert import QUBO, PUBO

        >>> P = PUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> x = {0: 1, 1: 0}
        >>> P.value(x)
        1

        >>> Q = QUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> x = {0: 1, 1: 0}
        >>> Q.value(x)
        1

        >>> P = PUBO({(0, 0): 1, (0, 1): -1})
        >>> x = {0: 1, 1: 0}
        >>> P.value(x)
        1

        >>> Q = QUBO({(0, 0): 1, (0, 1): -1})
        >>> x = {0: 1, 1: 0}
        >>> Q.value(x)
        1

        """
        return pubo_value(x, self)
