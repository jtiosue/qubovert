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

from .utils import BO, IsingField, IsingCoupling


class Ising(BO):
    """Ising.

    Class to manage converting general Ising problems to and from their
    QUBO and Ising formluations.

    This class deals with Isings that contain offsets and/or have binary
    labels that do not range from 0 to n-1. Note that it is generally
    more efficient to initialize an empty Ising object and then build the
    Ising, rather than initialize a Ising object with an already built dict.

    Keys of the Ising dict can either be a single label or a tuple of two
    different labels. If it is a single label, then it is interpreted as a
    field value. If it is a tuple of two different labels, then it is
    interpreted as a coupling value. See ``__setitem`` for conventions.

    Ising inherits some methods and attributes the ``BO`` class. See
    ``help(qubovert.utils.BO)``.

    Example usage
    -------------
    >>> ising = Ising()
    >>> ising['a'] += 5
    >>> ising[(0, 'a')] -= 2
    >>> ising -= 1.5
    >>> print(ising)
    {'a': 5, (0, 'a'): -2} - 1.5
    >>> print(ising.h)
    {'a': 5}
    >>> print(ising.J)
    {(0, 'a'): -2}
    >>> print(ising.offset)
    -1.5

    >>> ising = Ising({'a': 5, (0, 'a'): -2}, -1.5)
    >>> print(ising)
    {'a': 5, (0, 'a'): -2} - 1.5
    >>> h, J, offset = ising.to_ising()
    >>> print(offset)
    -1.5
    >>> print(h)
    {0: 5}
    >>> print(J)
    {(0, 1): -2}
    >>> ising.convert_solution({0: 1, 1: -1})
    {'a': 1, 0: -1}

    Details
    -------
    The following examples will only use integer labels just for ease, but in
    general the labels can be arbitrary.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = Ising()
    >>> print(d[0]) # will print 0
    >>> d[0] += 1
    >>> print(d) # will print {0: 1}

    Compared to an ordinary dictionary.

    >>> g = dict()
    >>> print(g[0]) # will raise KeyError
    >>> g[0] += 1 # will raise KeyError, since 0 was never set

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = Ising()
    >>> d[0] += 1
    >>> d[0] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize Ising with a previous dictionary
    it will be reinitialized to ensure that it contains no zero values.
    Consider the following example:

    >>> d = Ising({'0': 1, (1, '0'): 2, (2, '0'): 0})
    >>> print(d) # will print {'0': 1, (1, '0'): 2}

    We also change the update method so that it follows all the conventions.

    >>> d = Ising({0: 1, (0, 1): 2})
    >>> d.update({0: 0, (1, 0): 1, 1: -1})
    >>> print(d)  # will print {(0, 1): 1, 1: -1}

    We also include arithmetic, addition, subtraction, scalar division,
    scalar multiplication, and all those in place. For example,

    >>> d = IsingMatrix(dict(0=1, (0, 1)=-2), 3)
    >>> g = d + {0: -1}
    >>> print(g)  # will print {(0, 1): -2} + 3
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8} + 12
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {} + 12
    >>> g -= 12
    >>> print(g)  # will print {}

    """

    def __init__(self, ising=None, offset=0):
        """__init__.

        This class deals with Isings that contain offsets and/or have binary
        labels that do not range from 0 to n-1. Note that it is generally
        more efficient to initialize an empty Ising object and then build the
        Ising, rather than initialize a Ising object with an already built
        dict.

        Parameters
        ----------
        ising : dict (optional, defaults to None).
            Ising dictionary. This dictionary should map labels or pairs of
            labels to numbers. The labels can be arbitrary. A label that is a
            tuple of elements is interpreted as a coupling value, a label that
            is just a single object is interpreted as a field value. Note that
            a key that is a tuple of repeated labels will raise a KeyError.
            ``ising`` defaults to ``None``. If it is ``None``, then this
            ``Ising`` object will be initialized with an empty dictionary.
        offset: numeric (optional, defaults to 0).
            The part of the Ising objective function that does not depend on
            any variables.

        Examples
        -------
        >>> ising = Ising()
        >>> ising['a'] += 5
        >>> ising[(0, 'a')] -= 2
        >>> ising -= 1.5
        >>> ising
        {'a': 5, (0, 'a'): -2} - 1.5

        >>> ising = Ising({'a': 5, (0, 'a'): -2}, -1.5)
        >>> ising
        {'a': 5, (0, 'a'): -2} - 1.5

        """
        # this method is included just to update the docstring. There is no
        # new code, it just calls BO.__init__.
        super().__init__(ising, offset)

    @property
    def h(self):
        """h.

        Return a copy of the unconverted Ising Field dictionary.

        Return
        ------
        h : dict.
            Copy of he unconverted Ising Field dictionary.

        """
        return {k: v for k, v in self if not isinstance(k, tuple)}

    @property
    def J(self):
        """J.

        Return a copy of the unconverted Ising Coupling dictionary.

        Return
        ------
        J : dict.
            Copy of he unconverted Ising Coupling dictionary.

        """
        return {k: v for k, v in self if isinstance(k, tuple)}

    def to_ising(self):
        """to_ising.

        Create and return upper triangular J representing the coupling of the
        Ising formulation of the problem and the h representing the field.
        The labels of each spin will be integers from 0 to n-1.

        Return
        ------
        result : tuple (h, J, offset).
            h : qubovert.utils.IsingField object.
                The field of each spin in the Ising formulation.
                h is a IsingField object. For most practical purposes, you can
                use IsingField in he same way as an ordinary dictionary. For
                more information, see ``help(qubovert.utils.IsingField)``.
            J : qubovert.utils.IsingCoupling object.
                The upper triangular coupling matrix, an IsingCoupling object.
                For most practical purposes, you can use IsingCoupling in the
                same way as an ordinary dictionary. For more information,
                see ``help(qubovert.utils.IsingCoupling)``.
            offset : float.
                It is the sum of the terms in the formulation that don't
                involve any variables.

        """
        h, J = IsingField(), IsingCoupling()

        for k, v in self.items():
            if isinstance(k, tuple):
                i, j = k
                J[(self._mapping[i], self._mapping[j])] += v
            else:
                h[self._mapping[k]] += v

        return h, J, self._offset

    def convert_solution(self, solution):
        """convert_solution.

        Convert the solution to the integer labeled Ising to the solution to
        the originally labeled Ising.

        Parameters
        ----------
        solution : iterable or dict.
            The Ising or Ising solution output. The Ising solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for Ising
            (or -1 or 1 for Ising), or it can be a dictionary that maps the
            label of the variable to is value.

        Return
        -------
        res : dict.
            Maps binary variable labels to their Ising solution values {-1, 1}.

        Example
        -------
        >>> ising = Ising({'a': 1, ('a', 'b'): -2})
        >>> h, J, offset = ising.to_ising()
        >>> solution = solve_ising(h, J)  # any solver you want
        >>> solution
        [-1, 1]  # or {0: -1, 1: 1}
        >>> sol = ising.convert_solution(solution)
        >>> sol
        {'a': -1, 'b': 1}

        >>> ising = Ising({'a': 1, ('a', 'b'): -2})
        >>> Q, offset = ising.to_qubo()
        >>> solution = solve_qubo(Q)  # any solver you want
        >>> solution
        [0, 1]  # or {0: 0, 1: 1}
        >>> sol = ising.convert_solution(solution)
        >>> sol
        {'a': -1, 'b': 1}

        """
        sol = {}
        for i in range(self.num_binary_variables):
            sol[self._reverse_mapping[i]] = 1 if solution[i] == 1 else -1
        return sol

    def __setitem__(self, key, value):
        """__setitem__.

        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        Ising dictionary will ever have zero value. Also, if the key is a
        tuple with repeated indices then we raise a KeyError.

        Parameters
        ---------
        key : tuple of two elements, or an object.
            Element of the dictionary. If the key is not a tuple, then it will
            be interpreted as a field value, otherwise it will be interpreted
            as a coupling value.
        value : numeric.
            Value corresponding to the key.

        """
        keys = set(key) if isinstance(key, tuple) else {key}
        if len(keys) > 2:
            raise KeyError("Invalid key type")
        for i in keys:
            if i not in self._mapping:
                self._mapping[i] = self._next_label
                self._reverse_mapping[self._next_label] = i
                self._next_label += 1

        if isinstance(key, tuple) and key[0] == key[1]:
            raise KeyError("Cannot have repeated indices in Ising keys")

        super().__setitem__(key, value)
