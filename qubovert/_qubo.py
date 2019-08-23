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

"""_qubo.py.

Contains the QUBO class. See ``help(qubovert.QUBO)``.

"""

from .utils import BO, QUBOMatrix


def _convert_key(key):
    """_convert_key.

    A user can input a key into the QUBO class as either a 1 or 2 element
    tuple or as a single object. It must then be converted into a 2 element
    tuple.

    Parameters
    ----------
    key : 1 or 2 element tuple, or an object.

    Return
    ------
    res : two element tuple.
        Converted key.

    Example
    -------
    >>> _convert_key(0)
    (0, 0)

    >>> _convert_key((0,))
    (0, 0)

    >>> _convert_key((0, 0))
    (0, 0)

    """
    if not isinstance(key, (tuple, list)):
        k = key, key
    elif len(key) == 1:
        k = key * 2
    elif len(key) == 2:
        k = key
    else:
        raise ValueError("QUBO key cannot have length > 2")

    return k


class QUBO(BO):
    """QUBO.

    Class to manage converting general QUBO problems to and from their
    QUBO and Ising formluations.

    This class deals with QUBOs that contain offsets and/or have binary
    labels that do not range from 0 to n-1. Note that it is generally
    more efficient to initialize an empty QUBO object and then build the
    QUBO, rather than initialize a QUBO object with an already built dict.

    QUBO inherits some methods and attributes the ``BO`` class. See
    ``help(qubovert.utils.BO)``.

    Example usage
    -------------
    >>> qubo = QUBO()
    >>> qubo['a'] += 5
    >>> qubo[(0, 'a')] -= 2
    >>> qubo -= 1.5
    >>> qubo
    {('a' 'a'): 5, (0, 'a'): -2} - 1.5

    >>> qubo = QUBO({'a': 5, (0, 'a'): -2}, -1.5)
    >>> qubo
    {('a' 'a'): 5, (0, 'a'): -2} - 1.5
    >>> Q, offset = qubo.to_qubo()
    >>> offset
    -1.5
    >>> Q
    {(0, 0): 5, (0, 1): -2}
    >>> qubo.convert_solution({0: 1, 1: 0})
    {'a': 1, 0: 0}

    Details
    -------
    We define additional methods for the QUBO object. Many of these are very
    similar to method in qubovert.utils.QUBOMatrix. The difference is that
    the QUBOMatrix only allows integer binary labels, whereas the QUBO object
    allows for arbitrary labels. Also, the QUBOMatrix only allows to be indexed
    by tuples, whereas the QUBO object can be indexed by a single label for
    linear terms. QUBOMatrix['a'] will raise an error, but QUBO['a'] will
    interperet it as QUBO[('a', 'a')] (ie standard form). Also note one major
    difference between QUBO and QUBOMatrix is that QUBOMatrix always keeps the
    QUBO upper triangular, but QUBO does not! See the ``QUBO.to_qubo`` method
    to convert a QUBO into a QUBOMatrix.

    The following examples will only use integer labels just for ease, but in
    general the labels can be arbitrary.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = QUBO()
    >>> print(d[(0, 0)]) # will print 0
    >>> d[(0, 0)] += 1
    >>> print(d) # will print {(0, 0): 1}

    Compared to an ordinary dictionary.

    >>> g = dict()
    >>> print(g[(0, 0)]) # will raise KeyError
    >>> g[(0, 0)] += 1 # will raise KeyError, since (0, 0) was never set

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = QUBO()
    >>> d[(0, 0)] += 1
    >>> d[(0, 0)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize QUBO with a previous dictionary
    it will be reinitialized to ensure that it contains no zero values.
    Consider the following example:

    >>> d = QUBO({('0', '0'): 1, (1, '0'): 2, (2, '0'): 0})
    >>> print(d) # will print {('0', '0'): 1, (1, '0'): 2}

    We also change the update method so that it follows all the conventions.

    >>> d = QUBO({(0, 0): 1, (0, 1): 2})
    >>> d.update({(0, 0): 0, (1, 0): 1, (1, 1): -1})
    >>> print(d)  # will print {(0, 1): 1, (1, 1): -1}

    We also include arithmetic, addition, subtraction, scalar division,
    scalar multiplication, and all those in place. For example,

    >>> d = QUBOMatrix(dict((0, 0)=1, (0, 1)=-2), 3)
    >>> g = d + {(0, 0): -1}
    >>> print(g)  # will print {(0, 1): -2} + 3
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8} + 12
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {} + 12
    >>> g -= 12
    >>> print(g)  # will print {}

    """

    def __init__(self, Q=None, offset=0):
        """__init__.

        This class deals with QUBOs that contain offsets and/or have binary
        labels that do not range from 0 to n-1. Note that it is generally
        more efficient to initialize an empty QUBO object and then build the
        QUBO, rather than initialize a QUBO object with an already built dict.

        Parameters
        ----------
        Q : dict (optional, defaults to None).
            QUBO dictionary. This dictionary should map labels or pairs of
            labels to numbers. The labels can be arbitrary. Q defaults to None.
            If it is None, then this QUBO object will be initialized with an
            empty dictionary.
        offset: numeric (optional, defaults to 0).
            The part of the QUBO objective function that does not depend on
            any variables.

        Examples
        -------
        >>> qubo = QUBO()
        >>> qubo['a'] += 5
        >>> qubo[(0, 'a')] -= 2
        >>> qubo -= 1.5
        >>> qubo
        {('a' 'a'): 5, (0, 'a'): -2} - 1.5

        >>> qubo = QUBO({'a': 5, (0, 'a'): -2}, -1.5)
        >>> qubo
        {('a' 'a'): 5, (0, 'a'): -2} - 1.5

        """
        # we include this function just for the new docstring, since the
        # function itself is exactly the same as BO.__init__.
        super().__init__(Q, offset)

    @property
    def Q(self):
        """Q.

        Return a copy of the unconverted QUBO dictionary.

        Return
        ------
        Q : dict.
            Copy of he unconverted QUBO dictionary.

        """
        return dict(self)

    def to_qubo(self):
        """to_qubo.

        Create and return upper triangular QUBO representing the problem.
        The labels will be integers from 0 to n-1.

        Return
        -------
        result : tuple (Q, offset).
            Q : qubovert.utils.QUBOMatrix object.
                The upper triangular QUBO matrix, a QUBOMatrix object.
                For most practical purposes, you can use QUBOMatrix in the
                same way as an ordinary dictionary. For more information,
                see ``help(qubovert.utils.QUBOMatrix)``.
            offset : float.
                The sum of the terms in the formulation that don't involve any
                variables.

        """
        Q = QUBOMatrix()

        for (i, j), v in self.items():
            Q[(self._mapping[i], self._mapping[j])] += v

        return Q, self._offset

    def convert_solution(self, solution):
        """convert_solution.

        Convert the solution to the integer labeled QUBO to the solution to
        the originally labeled QUBO.

        Parameters
        ----------
        solution : iterable or dict.
            The QUBO or Ising solution output. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or -1 or 1 for Ising), or it can be a dictionary that maps the
            label of the variable to is value.

        Return
        -------
        res : dict.
            Maps binary variable labels to their QUBO solutions values {0, 1}.

        Example
        -------
        >>> qubo = QUBO({'a': 1, ('a', 'b'): -2, 'c': 1})
        >>> Q, offset = qubo.to_qubo()
        >>> Q
        {(0, 0): 1, (0, 1): -2, (2, 2): 1}
        >>> solution = solve_qubo(Q)  # any solver you want
        >>> solution
        [1, 1, 0]  # or {0: 1, 1: 1, 2: 0}
        >>> sol = qubo.convert_solution(solution)
        >>> sol
        {'a': 1, 'b': 1, 'c': 0}

        >>> qubo = QUBO({'a': 1, ('a', 'b'): -2, 'c': 1})
        >>> h, J, offset = qubo.to_ising()
        >>> solution = solve_ising(h, J)  # any solver you want
        >>> solution
        [1, 1, -1]  # or {0: 1, 1: 1, 2: -1}
        >>> sol = qubo.convert_solution(solution)
        >>> sol
        {'a': 1, 'b': 1, 'c': 0}

        """
        sol = {}
        for i in range(self.num_binary_variables):
            sol[self._reverse_mapping[i]] = 1 if solution[1] == 1 else 0
        return sol

    def __getitem__(self, key):
        """__getitem__.

        Overrides the dict.__getitem__ command so that a KeyError is not
        thrown if `key` is not in the dictionary.

        Parameters
        ---------
        key : tuple of one or two elements.
            Element of the dictionary. If the tuple is only length one, then
            it will be doubled. Ie ``(0,)`` goes to ``(0, 0)``. Or if the key
            is not a tuple, then it will be made into one. Ie ``0`` becomes
            ``(0, 0)``.

        Return
        -------
        value : numeric
            the value corresponding to the key if the key is in the dictionary,
            otherwise returns 0.

        """
        k = _convert_key(key)
        return super().__getitem__(k)

    def __setitem__(self, key, value):
        """__setitem__.

        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        QUBO dictionary will ever have zero value.

        Parameters
        ---------
        key : tuple of one or two elements.
            Element of the dictionary. If the tuple is only length one, then
            it will be doubled. Ie ``(0,)`` goes to ``(0, 0)``. Or if the key
            is not a tuple, then it will be made into one. Ie ``0`` becomes
            ``(0, 0)``.
        value : numeric.
            Value corresponding to the key.

        """
        k = _convert_key(key)
        keys = set(k)
        for i in keys:
            if i not in self._mapping:
                self._mapping[i] = self._next_label
                self._reverse_mapping[self._next_label] = i
                self._next_label += 1

        super().__setitem__(k, value)
