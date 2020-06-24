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

"""_qusomatrix.py.

This file contains the QUSOMatrix object.

"""

from . import PUSOMatrix, quso_value, solve_quso_bruteforce


__all__ = 'QUSOMatrix',


class QUSOMatrix(PUSOMatrix):
    """QUSOMatrix.

    ``QUSOMatrix`` inherits some methods from ``PUSOMatrix``, see
    ``help(qubovert.utils.PUSOMatrix)``.

    A class to handle QUSO matrices. It is the same thing as a dictionary
    with some methods modified. Note that each key must be a tuple integers
    >= 0.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = QUSOMatrix()
    >>> print(d[(0,)]) # will print 0
    >>> d[(0,)] += 1
    >>> print(d) # will print {(0,): 1}

    One method of QUSOMatrix is that it will always keep the QUSO
    upper triangular! Consider the following example:

    >>> d = QUSOMatrix()
    >>> d[(1, 0)] += 2
    >>> print(d)
    >>> # will print {(0, 1): 2}

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = QUSOMatrix()
    >>> d[(0, 1)] += 1
    >>> d[(0, 1)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize QUSOMatrix with a previous dictionary
    it will be reinitialized to ensure that the QUSOMatrix is upper
    triangular and contains no zero values. Consider the following example:

    >>> d = QUSOMatrix({(0,): 1, (1, 0): 2, (2, 0): 0})
    >>> print(d) # will print {(0,): 1, (0, 1): 2}

    We also change the update method so that it follows all the conventions.

    >>> d = Isingatrix({(0,): 1, (0, 1): 2})
    >>> d.update({(0,): 0, (1, 0): 1, (1,): -1})
    >>> print(d)  # will print {(0, 1): 1, (1,): -1}

    We also include arithmetic, addition, subtraction, scalar division,
    multiplication, and all those in place. For example,

    >>> d = QUSOMatrix((0,)=1, (0, 1)=-2)
    >>> g = d + {(0,): -1}
    >>> print(g) # will print {(0, 1): -2}
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8}
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {}

    >>> d = QUSOMatrix({(0, 0): 1, (0, 1): -1})
    >>> print(d)
    {(): 1, (0, 1): -1}
    >>> g = {(0,): -1, (1,): 1}
    >>> d *= g
    >>> print(d)
    {(0,): -2, (1,): 2}

    >>> d = QUSOMatrix({(0, 0): 1, (0, 1): -1})
    >>> print(d ** 2 == d * d)
    True

    Adding or subtracting constants will update the () element of the
    dict.

    >>> d = QUSOMatrix()
    >>> d += 5
    >>> print(d)
    {(): 5}

    Finally, if you try to access a key out of order, it will sort the key. Be
    careful with this, it can cause unexpected behavior if you don't know it.
    For example,

    >>> d = QUSOMatrix()
    >>> d[(0, 1)] += 2
    >>> print(d[(1, 0)])  # will print 2

    """

    @staticmethod
    def _check_key_valid(key):
        """_check_key_valid.

        Internal method to check if an input key to the dictionary is valid.
        Checks to see if ``key`` is a tuple of non negative integers with <=
        2 unique integers after squashing.

        Parameters
        ----------
        key : anything, but must be a tuple to be valid.

        Returns
        -------
        k : tuple.
            The squashed key.

        Raises
        ------
        KeyError if the key is invalid.

        """
        k = PUSOMatrix.squash_key(key)
        if len(k) > 2:
            raise KeyError(
                "Key formatted incorrectly, must be tuple of <= 2 integers "
                "See PUSOMatrix instead.")
        return k

    def value(self, z):
        r"""value.

        Find the value of the QUSO. Calling
            ``self.value(z)`` is the same as calling
            ``qubovert.utils.quso_value(z, self)``.

        Parameters
        ----------
        z: dict or iterable.
            Maps variable labels to their values, -1 or 1. Ie z[i] must be the
            value of variable i.

        Return
        ------
        value : float.
            The value of the QUSO with the given assignment `z`.

        Example
        -------
        >>> from qubovert.utils import QUSOMatrix, PUSOMatrix
        >>> from qubovert import QUSO, PUSO

        >>> H = PUSOMatrix({(0, 1): -1, (0,): 1})
        >>> z = {0: -1, 1: 1}
        >>> H.value(z)
        0

        >>> H = PUSO({(0, 1): -1, (0,): 1})
        >>> z = {0: -1, 1: 1}
        >>> H.value(z)
        0

        >>> L = QUSOMatrix({(0, 1): -1, (0,): 1})
        >>> z = {0: -1, 1: 1}
        >>> L.value(z)
        0

        >>> L = QUSO({(0, 1): -1, (0,): 1})
        >>> z = {0: -1, 1: 1}
        >>> L.value(z)
        0

        """
        return quso_value(z, self)

    def solve_bruteforce(self, all_solutions=False):
        """solve_bruteforce.

        Solve the problem bruteforce. THIS SHOULD NOT BE USED FOR LARGE
        PROBLEMS! This is the exact same as calling
        ``qubovert.utils.solve_quso_bruteforce(
            self, all_solutions, self.is_solution_valid)[1]``.

        Parameters
        ----------
        all_solutions : bool.
            See the description of the ``all_solutions`` parameter in
            ``qubovert.utils.solve_quso_bruteforce``.

        Return
        ------
        res : the second element of the two element tuple that is returned from
            ``qubovert.utils.solve_quso_bruteforce``.

        """
        return solve_quso_bruteforce(self, all_solutions,
                                     self.is_solution_valid)[1]

    @property
    def h(self):
        """h.

        Return a plain dictionary representing the QUSO field values. Each key
        is an integer.

        Returns
        -------
        h : dict.
            Plain dictionary representing the QUSO field values.

        """
        return {k[0]: v for k, v in self.items() if len(k) == 1}

    @property
    def J(self):
        """J.

        Return a plain dictionary representing the QUSO coupling values. Each
        key is an integer.

        Returns
        -------
        J : dict.
            Plain dictionary representing the QUSO coupling values.

        """
        return {k: v for k, v in self.items() if len(k) == 2}
