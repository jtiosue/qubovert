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

"""_hisingmatrix.py.

This file contains the HIsingMatrix object.

"""

from . import (
    PUBOMatrix, hash_function,
    hising_value, solve_hising_bruteforce
)


__all__ = 'HIsingMatrix',


class HIsingMatrix(PUBOMatrix):
    """HIsingMatrix.

    ``HIsingMatrix`` inherits some methods from ``DictArithmetic``, see
    ``help(qubovert.utils.DictArithmetic)``.

    ``HIsingMatrix`` inherits some methods from ``PUBOMatrix``, see
    ``help(qubovert.utils.PUBOMatrix)``.

    A class to handle HIsing matrices. It is the same thing as a dictionary
    with some methods modified. Note that each key must be a tuple of integers
    >= 0.

    Note that below we only consider keys that are of length 1 or 2, but they
    can in general be arbitrarily long! See ``qubovert.utils.IsingMatrix``
    for an object that restricts the length to <= 2.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = HIsingMatrix()
    >>> print(d[(0,)]) # will print 0
    >>> d[(0,)] += 1
    >>> print(d) # will print {(0,): 1}

    One method of HIsingMatrix is that it will always keep the HIsing
    upper triangular! Consider the following example:

    >>> d = HIsingMatrix()
    >>> d[(1, 0)] += 2
    >>> print(d)
    >>> # will print {(0, 1): 2}

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = HIsingMatrix()
    >>> d[(0,)] += 1
    >>> d[(0,)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize HIsingMatrix with a previous dictionary
    it will be reinitialized to ensure that the HIsingMatrix is upper
    triangular and contains no zero values. Consider the following example:

    >>> d = HIsingMatrix({(0,): 1, (1, 0): 2, (2, 0): 0, (2, 0, 1): 1})
    >>> print(d) # will print {(0,): 1, (0, 1): 2, (0, 1, 2): 1}

    We also change the update method so that it follows all the conventions.

    >>> d = HIsingMatrix({(0,): 1, (0, 1): 2})
    >>> d.update({(0,): 0, (1, 0): 1, (1, 2): -1})
    >>> print(d)  # will print {(0, 1): 1, (1, 2): -1}

    We also include arithmetic, addition, subtraction, scalar division,
    scalar multiplication, and all those in place. For example,

    >>> d = HIsingMatrix((0,)=1, (0, 1)=-2)
    >>> g = d + {(0,): -1}
    >>> print(g) # will print {(0, 1): -2}
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8}
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {}

    >>> d = HIsingMatrix({(0, 0): 1, (0, 1): -1})
    >>> print(d)
    {(): 1, (0, 1): -1}
    >>> g = {(0,): -1, (1,): 1}
    >>> d *= g
    >>> print(d)
    {(0,): -2, (1,): 2}

    >>> d = HIsingMatrix({(0, 0): 1, (0, 1): -1})
    >>> print(d ** 2 == d * d)
    True

    If we try to set a key with duplicated indices, it will squash the key
    into its equivalent form. For example ``(0, 1, 1, 2, 2, 2)`` is equivalent
    to ``(0, 2)``. Thus,

    >>> d = HIsingMatrix()
    >>> d[(0, 1, 1, 2, 2, 2)] += 1
    >>> d[(0, 2)] += 2
    >>> print(d)
    {(0, 2): 3}

    Adding or subtracting constants will update the () element of the
    dict.

    >>> d = HIsingMatrix()
    >>> d += 5
    >>> print(d)
    {(): 5}

    Finally, if you try to access a key out of order, it will sort the key. Be
    careful with this, it can cause unexpected behavior if you don't know it.
    For example,

    >>> d = HIsingMatrix()
    >>> d[(0, 1)] += 2
    >>> print(d[(1, 0)])  # will print 2

    """

    @classmethod
    def squash_key(cls, key):
        """squash_key.

        Will convert the input key into the standard form for HIsingMatrix /
        IsingMatrix. It will get rid of pairs of duplicates and sort. This
        method will check to see if the input key is valid.

        Parameters
        ----------
        key : tuple of integers.

        Return
        ------
        k : tuple of integers.
            A sorted and squashed version of ``key``.

        Example
        -------
        >>> squash_key((0, 4, 0, 3, 3, 2, 3))
        >>> (2, 3, 4)

        """
        cls._check_key_valid(key)
        # here we use hash because some other classes that are subclasses of
        # this class will allow elements of the key to be strings! So we want
        # to still have something consistent to sort by. But for this class,
        # it doesn't make a difference, because hash_function(i) == i when i is
        # an int.
        return tuple(sorted(
            (x for x in set(key) if key.count(x) % 2),
            key=lambda x: hash_function(x)
        ))

    def solve_bruteforce(self, all_solutions=False):
        """solve_bruteforce.

        Solve the problem bruteforce. THIS SHOULD NOT BE USED FOR LARGE
        PROBLEMS! This is the exact same as calling
        ``qubovert.utils.solve_hising_bruteforce(self, all_solutions)[1]``.

        Parameters
        ----------
        all_solutions : bool.
            See the description of the ``all_solutions`` parameter in
            ``qubovert.utils.solve_hising_bruteforce``.

        Return
        ------
        res : the second element of the two element tuple that is returned from
            ``qubovert.utils.solve_hising_bruteforce``.

        """
        return solve_hising_bruteforce(self, all_solutions)[1]

    def value(self, x):
        r"""value.

        Find the value of
            :math:`\sum_{i,...,j} H_{i...j} z_{i} ... z_{j}`. Calling
            ``self.value(z)` is the same as calling
            ``qubovert.utils.hising_value(z, self)``.

        Parameters
        ----------
        z: dict or iterable.
            Maps variable labels to their values, -1 or 1. Ie z[i] must be the
            value of variable i.

        Return
        -------
        value : float.
            The value of the HIsing/Ising with the given assignment `z`.

        Example
        -------
        >>> from qubovert.utils import IsingMatrix, HIsingMatrix
        >>> from qubovet import Ising, HIsing

        >>> H = HIsingMatrix({(0, 1): -1, (0,): 1})
        >>> z = {0: -1, 1: 1}
        >>> H.value(z)
        0

        >>> H = HIsing({(0, 1): -1, (0,): 1})
        >>> z = {0: -1, 1: 1}
        >>> H.value(z)
        0

        >>> L = IsingMatrix({(0, 1): -1, (0,): 1})
        >>> z = {0: -1, 1: 1}
        >>> L.value(z)
        0

        >>> L = Ising({(0, 1): -1, (0,): 1})
        >>> z = {0: -1, 1: 1}
        >>> L.value(z)
        0

        """
        return hising_value(x, self)
