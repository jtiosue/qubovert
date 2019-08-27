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

"""_qubo_matrix.py.

This file contains QUBOMatrix, IsingMatrix, PUBOMatrix, and HIsingMatrix
objects.
"""

from . import DictArithmetic


__all__ = (
    'PUBOMatrix', 'HIsingMatrix', 'QUBOMatrix', 'IsingMatrix'
)


def _hash(x):
    """_hash.

    Internal function to return (usually) unique hashes for ``x`` such
    that multiple ``x``s can be ordered. Note that the hash is not consistent
    across Python sessions (except for when ``x`` is an integer)!

    Parameters
    ----------
    x : hashable object.

    Returns
    -------
    res : int.
        If ``x`` is an integer, then ``res == x``. Otherwise,
        ``res == hash(x)``.

    """
    return x if isinstance(x, int) else hash(x)


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
        if not self:
            return 0

        s = set()
        for x in self:
            s.update(set(x))
        return len(s)

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
        if not self:
            return

        return max(max(x) for x in self)

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
        cls._check_key_valid(key)
        # here we use hash because some other classes that are subclasses of
        # this class will allow elements of the key to be strings! So we want
        # to still have something consistent to sort by. But for this class,
        # it doesn't make a difference, because _hash(i) == i when i is an int.
        return tuple(sorted(set(key), key=lambda x: _hash(x)))

    @staticmethod
    def _check_key_valid(key):
        """_check_key_valid.

        Internal method to check if an input key to the dictionary is valid.
        Checks to see if ``key`` is a tuple of non negative integers.

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
        ---------
        key : tuple of integers.
            Element of the dictionary.

        Return
        -------
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
        ---------
        key : tuple of integers.
            Element of the dictionary.
        value : numeric.
            Value corresponding to the key.

        """
        super().__setitem__(self.__class__.squash_key(key), value)


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
        # it doesn't make a difference, because _hash(i) == i when i is an int.
        return tuple(sorted(
            (x for x in set(key) if key.count(x) % 2),
            key=lambda x: _hash(x)
        ))


class QUBOMatrix(PUBOMatrix):
    """QUBOMatrix.

    ``QUBOMatrix`` inherits some methods from ``PUBOMatrix``, see
    ``help(qubovert.utils.PUBOMattrix)``.

    A class to handle QUBO matrices. It is the same thing as a dictionary
    with some methods modified. Note that each key must be a tuple of two
    integers >= 0.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = QUBOMatrix()
    >>> print(d[(0,)]) # will print 0
    >>> d[(0,)] += 1
    >>> print(d) # will print {(0,): 1}

    Compared to an ordinary dictionary.

    >>> g = dict()
    >>> print(g[(0,)]) # will raise KeyError
    >>> g[(0,)] += 1 # will raise KeyError, since (0,) was never set

    One method of QUBOMatrix is that it will always keep the QUBO
    upper triangular! Consider the following example:

    >>> d = QUBOMatrix()
    >>> d[(1, 0)] += 2
    >>> print(d)
    >>> # will print {(0, 1): 2}

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = QUBOMatrix()
    >>> d[(0,)] += 1
    >>> d[(0,)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize QUBOMatrix with a previous dictionary
    it will be reinitialized to ensure that the QUBOMatrix is upper
    triangular and contains no zero values. Consider the following example:

    >>> d = QUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0})
    >>> print(d) # will print {(0, 0): 1, (0, 1): 2}

    We also change the update method so that it follows all the conventions.

    >>> d = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    >>> d.update({(0, 0): 0, (1, 0): 1, (1, 1): -1})
    >>> print(d)  # will print {(0, 1): 1, (1,): -1}

    We also include arithmetic, addition, subtraction, scalar division,
    multiplication, and all those in place. For example,

    >>> d = QUBOMatrix((0, 0)=1, (0, 1)=-2)
    >>> g = d + {(0, 0): -1}
    >>> print(g) # will print {(0, 1): -2}
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8}
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {}

    >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
    >>> g = {(0,): -1, (1,): 1}
    >>> d *= g
    >>> print(d)
    {(0,): -1, (0, 1): 1}

    >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
    >>> print(d ** 2 == d * d)
    True

    Adding or subtracting constants will update the () element of the
    dict.

    >>> d = QUBOMatrix()
    >>> d += 5
    >>> print(d)
    {(): 5}

    Finally, if you try to access a key out of order, it will sort and squash
    the key. Be careful with this, it can cause unexpected behavior if you
    don't know it. For example,

    >>> d = QUBOMatrix()
    >>> d[(0, 1)] += 2
    >>> print(d[(1, 0)])  # will print 2

    >>> d = QUBOMatrix()
    >>> d[(0, 0)] += 2
    >>> print(d[(0,)])  # will print 2
    >>> print(d[(0, 0)])  # will print 2
    >>> print(d)  # will print {(0,): 2}

    """

    @staticmethod
    def _check_key_valid(key):
        """_check_key_valid.

        Internal method to check if an input key to the dictionary is valid.
        Checks to see if ``key`` is a tuple of non negative integers with <=
        2 unique integers.

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
        if len(PUBOMatrix.squash_key(key)) > 2:
            raise KeyError(
                "Key formatted incorrectly, must be tuple of <= 2 integers "
                "See PUBOMatrix instead.")

    @property
    def Q(self):
        """Q.

        Return a plain dictionary representing the QUBO. Each key is a tuple
        of two integers, ie (1, 1) corresponds to (1,). Note that the offset
        in the QUBOMatrix is ignored (ie the value corresponding to the key
        ()). See the ``offset`` property to access it.

        Returns
        -------
        Q : dict.
            Plain dictionary representing the QUBO in standard form.

        """
        return {k * (3 - len(k)): v for k, v in self.items() if k}


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
