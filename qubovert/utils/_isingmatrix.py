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

"""_isingmatrix.py.

This file contains the IsingMatrix object.

"""

from . import HIsingMatrix


__all__ = 'IsingMatrix',


class IsingMatrix(HIsingMatrix):
    """IsingMatrix.

    ``IsingMatrix`` inherits some methods from ``HIsingMatrix``, see
    ``help(qubovert.utils.HIsingMatrix)``.

    A class to handle Ising matrices. It is the same thing as a dictionary
    with some methods modified. Note that each key must be a tuple integers
    >= 0.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = IsingMatrix()
    >>> print(d[(0,)]) # will print 0
    >>> d[(0,)] += 1
    >>> print(d) # will print {(0,): 1}

    One method of IsingMatrix is that it will always keep the Ising
    upper triangular! Consider the following example:

    >>> d = IsingMatrix()
    >>> d[(1, 0)] += 2
    >>> print(d)
    >>> # will print {(0, 1): 2}

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = IsingMatrix()
    >>> d[(0, 1)] += 1
    >>> d[(0, 1)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize IsingMatrix with a previous dictionary
    it will be reinitialized to ensure that the IsingMatrix is upper
    triangular and contains no zero values. Consider the following example:

    >>> d = IsingMatrix({(0,): 1, (1, 0): 2, (2, 0): 0})
    >>> print(d) # will print {(0,): 1, (0, 1): 2}

    We also change the update method so that it follows all the conventions.

    >>> d = Isingatrix({(0,): 1, (0, 1): 2})
    >>> d.update({(0,): 0, (1, 0): 1, (1,): -1})
    >>> print(d)  # will print {(0, 1): 1, (1,): -1}

    We also include arithmetic, addition, subtraction, scalar division,
    multiplication, and all those in place. For example,

    >>> d = IsingMatrix((0,)=1, (0, 1)=-2)
    >>> g = d + {(0,): -1}
    >>> print(g) # will print {(0, 1): -2}
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8}
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {}

    >>> d = IsingMatrix({(0, 0): 1, (0, 1): -1})
    >>> print(d)
    {(): 1, (0, 1): -1}
    >>> g = {(0,): -1, (1,): 1}
    >>> d *= g
    >>> print(d)
    {(0,): -2, (1,): 2}

    >>> d = IsingMatrix({(0, 0): 1, (0, 1): -1})
    >>> print(d ** 2 == d * d)
    True

    Adding or subtracting constants will update the () element of the
    dict.

    >>> d = IsingMatrix()
    >>> d += 5
    >>> print(d)
    {(): 5}

    Finally, if you try to access a key out of order, it will sort the key. Be
    careful with this, it can cause unexpected behavior if you don't know it.
    For example,

    >>> d = IsingMatrix()
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
        ---------
        key : anything, but must be a tuple to be valid.

        Returns
        -------
        None.

        Raises
        ------
        KeyError if the key is invalid.

        """
        if len(HIsingMatrix.squash_key(key)) > 2:
            raise KeyError(
                "Key formatted incorrectly, must be tuple of <= 2 integers "
                "See HIsingMatrix instead.")

    @property
    def h(self):
        """h.

        Return a plain dictionary representing the Ising field values. Each key
        is an integer.

        Returns
        -------
        h : dict.
            Plain dictionary representing the Ising field values.

        """
        return {k[0]: v for k, v in self.items() if len(k) == 1}

    @property
    def J(self):
        """J.

        Return a plain dictionary representing the Ising coupling values. Each
        key is an integer.

        Returns
        -------
        J : dict.
            Plain dictionary representing the Ising coupling values.

        """
        return {k: v for k, v in self.items() if len(k) == 2}
