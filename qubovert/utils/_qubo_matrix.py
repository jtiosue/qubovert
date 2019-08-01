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

This file contains QUBOMatrix, IsingCoupling, and IsingField objects, which
are used for QUBO matrices Q, Ising coupling matrices J, and Ising fields h.
"""


class QUBOMatrix(dict):
    """QUBOMatrix.

    A class to handle QUBO matrices. It is the same thing as a dictionary
    with some methods modified. Note that each key must be a tuple of two
    integers >= 0.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = QUBOMatrix()
    >>> print(d[(0, 0)]) # will print 0
    >>> d[(0, 0)] += 1
    >>> print(d) # will print {(0, 0): 1}

    Compared to an ordinary dictionary.

    >>> g = dict()
    >>> print(g[(0, 0)]) # will raise KeyError
    >>> g[(0, 0)] += 1 # will raise KeyError, since (0, 0) was never set

    One method of QUBOMatrix is that it will always keep the QUBO
    upper triangular! Consider the following example:

    >>> d = QUBOMatrix()
    >>> d[(1, 0)] += 2
    >>> print(d)
    >>> # will print {(0, 1): 2}

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = QUBOMatrix()
    >>> d[(0, 0)] += 1
    >>> d[(0, 0)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize QUBOMatrix with a previous dictionary
    it will be reinitialized to ensure that the QUBOMatrix is upper
    triangular and contains no zero values. Consider the following example:

    >>> d = QUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0})
    >>> print(d) # will print {(0, 0): 1, (0, 1): 2}

    We also change the update method so that it follows all the conventions.

    >>> d = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    >>> d.update({(0, 0): 0, (1, 0): 1, (1, 1): -1})
    >>> print(d)  # will print {(0, 1): 1, (1, 1): -1}

    We also include arithmetic, addition, subtraction, scalar division,
    scalar multiplication, and all those in place. For example,

    >>> d = QUBOMatrix((0, 0)=1, (0, 1)=-2)
    >>> g = d + {(0, 0): -1}
    >>> print(g) # will print {(0, 1): -2}
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8}
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {}

    Finally, if you try to access a key out of order, it will sort the key. Be
    careful with this, it can cause unexpected behavior if you don't know it.
    For example,

    >>> d = QUBOMatrix()
    >>> d[(0, 1)] += 2
    >>> print(d[(1, 0)])  # will print 2

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        Initialize a QUBOMatrix object. If you supply args and kwargs that
        represent a dictionary, they will be reinitialized to ensure that
        the QUBOMatrix is upper triangular and contains no zero values.

        Parameters
        ---------
        *args and **kwargs : see the docstring for dict.

        """
        super().__init__(*args, **kwargs)

        # reset to make sure everything is in the proper form
        items = tuple(self.items())
        self.clear()
        for key, value in items:
            self[key] += value

    def __getitem__(self, key):
        """__getitem__.

        Overrides the dict.__getitem__ command so that a KeyError is not
        thrown if `key` is not in the dictionary. Also sorts the key. So
        if we you try to access the key (1, 0), it will return the value for
        the key (0, 1).

        Parameters
        ---------
        key : tuple of two integers.
            Element of the dictionary.

        Return
        -------
        value : numeric
            the value corresponding to the key if the key is in the dictionary,
            otherwise returns 0.

        """
        try:
            k = tuple(sorted(key))
        except TypeError:
            k = key

        return self.get(k, 0)

    def __setitem__(self, key, value):
        """__setitem__.

        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        QUBOMatrix dictionary will ever have zero value. Additionally, this
        method will keep the QUBO upper triangular, so if key[0] > key[1],
        then we will call __setitem__((key[1], key[0]), value).

        Parameters
        ---------
        key : tuple of two integers.
            Element of the dictionary.
        value : numeric.
            Value corresponding to the key.

        """
        incorrect_format = (
            not isinstance(key, tuple) or not len(key) == 2 or
            not isinstance(key[0], int) or not isinstance(key[1], int) or
            key[0] < 0 or key[1] < 0
        )

        if incorrect_format:
            raise KeyError(
                "Key formatted incorrectly, must be tuple of two integers")

        k = tuple(sorted(key))
        if value:
            super().__setitem__(k, value)
        else:
            self.pop(k, 0)

    def copy(self, *args, **kwargs):
        """copy.

        Same as dict.copy, but we adjust the method so that it returns a
        QUBOMatrix object, or an IsingField or IsingCoupling object in those
        subclasses.

        Parameters
        ----------
        *args and **kwargs : see dict.copy.

        """
        return type(self)(super().copy(*args, **kwargs))

    def update(self, *args, **kwargs):
        """update.

        Update the dictionary but following all the conventions of this class.

        Parameters
        ----------
        *args and **kwargs : defines a dictionary.
            Ie ``d = dict(*args, **kwargs)``.
            Each element in d will be added in place to this instance following
            all the required convensions.

        """
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __add__(self, other):
        """__add__.

        Add two QUBOMatrices or dicts, return the sum.

        Parameters
        ----------
        other : a QUBOMatrix or dict object.

        Return
        ------
        Q : a QUBOMatrix object, self + other.

        Examples
        --------
        >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (1, 0): 1}
        >>> print(d + g)
        {(0, 0): 3}

        """
        d = self.copy()
        d += other
        return d

    def __radd__(self, other):
        """__radd__.

        Add two QUBO Matrices. This will be called if the left one is just a
        dict object. This is the same as the __add__ method, but is included
        in case we are adding a dict and a QUBOMatrix object, instead of two
        QUBOMatrix objects.

        Parameters
        ----------
        other : a QUBOMatrix or dict object.

        Return
        -------
        Q : a QUBOMatrix object, self + other.

        Examples
        --------
        >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (1, 0): 1}
        >>> print(g + d)
        {(0, 0): 3}

        """
        return self + other

    def __iadd__(self, other):
        """__iadd__.

        Same as the __add__ method, but done in place.

        Parameters
        ----------
        other : a QUBOMatrix or dict object.

        Return
        -------
        Q : a QUBOMatrix object, self.

        Examples
        --------
        >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (1, 0): 1}
        >>> d += g
        >>> print(d)
        {(0, 0): 3}

        """
        for k, v in other.items():
            self[k] += v
        return self

    def __sub__(self, other):
        """__sub__.

        Subtract two QUBOMatrices or dicts, return the difference.

        Parameters
        ----------
        other : a QUBOMatrix or dict object.

        Return
        -------
        Q : a QUBOMatrix object, self - other.

        Examples
        --------
        >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (1, 0): 1}
        >>> print(d - g)  # will print {(0, 0): -1}

        """
        d = self.copy()
        d -= other
        return d

    def __rsub__(self, other):
        """__rsub__.

        Subtract two QUBO Matrices. This will be called if the left one is just
        a dict object. This is the same as the __sub__ method, but is included
        in case we are adding a dict and a QUBOMatrix object, instead of two
        QUBOMatrix objects.

        Parameters
        ----------
        other : a QUBOMatrix or dict object.

        Return
        -------
        Q : a QUBOMatrix object, other - self.

        Examples
        --------
        >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (1, 0): 1}
        >>> print(g - d)  # will print {(0, 0): 1}

        """
        return -1*self + other

    def __isub__(self, other):
        """__isub__.

        Same as the __sub__ method, but done in place.

        Parameters
        ----------
        other : a QUBOMatrix or dict object.

        Returns
        -------
        Q : a QUBOMatrix object, self.

        Examples
        --------
        >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (1, 0): 1}
        >>> d -= g
        >>> print(d)  # will print {(0, 0): -1}

        """
        for k, v in other.items():
            self[k] -= v
        return self

    def __mul__(self, other):
        """__mul__.

        Multiplying a QUBOMatrix by a scalar.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        Q : a QUBOMatrix object.

        Example
        -------
        >>> d = QUBOMatrix((0, 0)=1, (0, 1)=-1)
        >>> print(d * 2)
        {(0, 0): 2, (0, 1): -1}

        """
        d = self.copy()
        d *= other
        return d

    def __rmul__(self, other):
        """__rmul__.

        Same as __mul__, but for different ordering.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        Q : a QUBOMatrix object.

        Example
        -------
        >>> d = QUBOMatrix((0, 0)=1, (0, 1)=-1)
        >>> print(2 * d)
        {(0, 0): 2, (0, 1): -1}

        """
        return self * other

    def __imul__(self, other):
        """__imul__.

        Same as __mul__, but done in place.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        Q : a QUBOMatrix object, self.

        Example
        -------
        >>> d = QUBOMatrix((0, 0)=1, (0, 1)=-1)
        >>> d *= 2
        >>> print(d)
        {(0, 0): 2, (0, 1): -1}

        """
        for k in tuple(self.keys()):
            self[k] *= other
        return self

    def __truediv__(self, other):
        """__truediv__.

        Dividing a QUBOMatrix by a scalar.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        Q : a QUBOMatrix object.

        Example
        -------
        >>> d = QUBOMatrix((0, 0)=1, (0, 1)=-1)
        >>> print(d / 2)
        {(0, 0): .5, (0, 1): -.5}

        """
        d = self.copy()
        d /= other
        return d

    def __itruediv__(self, other):
        """__itruediv__.

        Same as __truediv__, but done in place.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        Q : a QUBOMatrix object, self.

        Example
        -------
        >>> d = QUBOMatrix((0, 0)=1, (0, 1)=-1)
        >>> d /= 2
        >>> print(d)
        {(0, 0): .5, (0, 1): -.5}

        """
        for k in tuple(self.keys()):
            self[k] /= other
        return self

    def __floordiv__(self, other):
        """__floordiv__.

        Floor dividing a QUBOMatrix by a scalar.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        Q : a QUBOMatrix object.

        Example
        -------
        >>> d = QUBOMatrix((0, 0)=2, (0, 1)=-1)
        >>> print(d // 2)
        {(0, 0): 1, (0, 1): 0}

        """
        d = self.copy()
        d //= other
        return d

    def __ifloordiv__(self, other):
        """__ifloordiv__.

        Same as __floordiv__, but done in place.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        Q : a QUBOMatrix object, self.

        Example
        -------
        >>> d = QUBOMatrix((0, 0)=2, (0, 1)=-1)
        >>> d //= 2
        >>> print(d)
        {(0, 0): 1, (0, 1): 0}

        """
        for k in tuple(self.keys()):
            self[k] //= other
        return self


class IsingCoupling(QUBOMatrix):
    """IsingCoupling.

    A class to handle the J coupling Ising matrices, inherits from QUBOMatrix.
    It is the same thing as a dictionary with some methods modified. Note that
    each key must be a tuple of two integers >= 0. Note that this is almost
    exactly the same as QUBOMatrix, except that the keys cannot be tuples of
    the same index. For example, ``QUBOMatrix({(0, 0): 1})`` is valid but
    ``IsingCoupling({(0, 0): 1})`` is invalid.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = IsingCoupling()
    >>> print(d[(0, 1)]) # will print 0
    >>> d[(0, 1)] += 1
    >>> print(d) # will print {(0, 1): 1}

    Compared to an ordinary dictionary.

    >>> g = dict()
    >>> print(g[(0, 1)]) # will raise KeyError
    >>> g[(0, 1)] += 1 # will raise KeyError, since (0, 0) was never set

    One method of IsingCoupling is that it will always keep the coupling
    upper triangular! Consider the following example:

    >>> d = IsingCoupling()
    >>> d[(1, 0)] += 2
    >>> print(d)
    {(0, 1): 2}

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = IsingCoupling()
    >>> d[(0, 1)] += 1
    >>> d[(0, 1)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize IsingCoupling with a previous
    dictionary, it will be reinitialized to ensure that the IsingCoupling is
    upper triangular and contains no zero values. Consider the following
    example:

    >>> d = IsingCoupling({(0, 1): 1, (1, 0): 2, (2, 0): 0})
    >>> print(d) # will print {(0, 1): 3}

    We also change the update method so that it follows all the conventions
    .
    >>> d = IsingCoupling({(0, 1): 1, (0, 2): 2})
    >>> d.update({(1, 0): 0, (2, 1): 1})
    >>> print(d)  # will print {(1, 2): 1, (0, 2): 2}

    We also include arithmetic, addition, subtraction, scalar division,
    scalar multiplication, and all those in place. For example,

    >>> d = IsingCoupling((0, 2)=1, (0, 1)=-2)
    >>> g = d + {(0, 2): -1}
    >>> print(g) # will print {(0, 1): -2}
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8}
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {}

    Finally, if you try to access a key out of order, it will sort the key. Be
    careful with this, it can unexpected behavior if you don't know it.
    For example,

    >>> d = IsingCoupling()
    >>> d[(0, 1)] += 2
    >>> print(d[(1, 0)])  # will print 2

    """

    def __setitem__(self, key, value):
        """__setitem__.

        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        IsingCoupling dictionary will ever have zero value. Additionally, this
        method will keep the coupling upper triangular, so if key[0] > key[1],
        then we will call __setitem__((key[1], key[0]), value). Finally,
        key[0] cannot equal key[1], if so a KeyError will be raised.

        Parameters
        ----------
        key : tuple of two different integers.
            Element of the dictionary.
        value : numeric.
            Value corresponding to the key.

        """
        if not isinstance(key, tuple) or key[0] == key[1]:
            raise KeyError(
                "Key formatted incorrectly, "
                "must be tuple of two different integers")

        super().__setitem__(key, value)


class IsingField(QUBOMatrix):
    """IsingField.

    A class to handle the h field Ising matrices, inherits from QUBOMatrix.
    It is the same thing as a dictionary with some methods modified. Note that
    each key must be an an integer >= 0. Note that this is almost exactly the
    same as QUBOMatrix, except that the keys are integers instead of tuples.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = IsingField()
    >>> print(d[0]) # will print 0
    >>> d[0] += 1
    >>> print(d) # will print {0: 1}

    Compared to an ordinary dict.

    >>> g = dict()
    >>> print(g[0]) # will raise KeyError
    >>> g[0] += 1 # will raise KeyError, since (0, 0) was never set

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = IsingField()
    >>> d[1] += 1
    >>> d[1] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize IsingField with a previous
    dictionary, it will be reinitialized to ensure that the IsingField is
    contains no zero values and is all valid. Consider the following
    example:

    >>> d = IsingField({0: 1, 1: 2, 2: 0})
    >>> print(d) # will print {0: 1, 1: 2}

    We also change the update method so that it follows all the conventions.

    >>> d = IsingField({0: 1, 2: -2})
    >>> d.update({0: 0, 1: 1})
    >>> print(d)  # will print {1: 1, 2: -2}

    We also include arithmetic, addition, subtraction, scalar division,
    scalar multiplication, and all those in place. For example,

    >>> d = IsingField(0=1, 1=-2)
    >>> g = d + {0: -1}
    >>> print(g) # will print {1: -2}
    >>> g *= 4
    >>> print(g) # will print {1: -8}
    >>> g -= {1: -8}
    >>> print(g) # will print {}

    """

    def __setitem__(self, key, value):
        """__setitem__.

        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        IsingCoupling dictionary will ever have zero value. Additionally, this
        method will keep the coupling upper triangular, so if key[0] > key[1],
        then we will call ``__setitem__((key[1], key[0]), value)``. Finally,
        key[0] cannot equal key[1], if so a KeyError will be raised.

        Parameters
        ----------
        key: int.
            Element of the dictionary.
        value: numeric.
            Value corresponding to the key.

        """
        if not isinstance(key, int) or key < 0:
            raise KeyError(
                "Key formatted incorrectly, must be a positive integer")

        if value:
            dict.__setitem__(self, key, value)
        else:
            self.pop(key, 0)
