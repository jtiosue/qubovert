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

"""_dict_arithmetic.py.

Contains the DictArithmetic class. See ``help(qubovert.utils.DictArithmetic)``.
It is used to define operations for dictionaries, and is a parent class to some
other classes, such as ``qubovert.utils.QUBOMatrix`` and ``qubovert.utils.BO``.

"""

__all__ = 'DictArithmetic',


def _generate_key_value_pairs(*args, **kwargs):
    """_generate_key_value_pairs.

    Generate the key, value pairs from the parameter inputs to a dictionary
    without explicitly recreating the dictionary.

    Parameters
    ----------
    args and kwargs define a dictionary. See ``help(dict)``.

    Yields
    ------
    pairs : tuple (key, value).

    """
    if not kwargs and len(args) == 1:
        a = args[0]
        k_v_pairs = (x for x in (a.items() if isinstance(a, dict) else a))
    elif not args and kwargs:
        k_v_pairs = (x for x in kwargs.items())
    else:
        k_v_pairs = (x for x in dict(*args, **kwargs).items())

    yield from k_v_pairs


class DictArithmetic(dict):
    """DictArithmetic.

    A class to handle dictionaries. It is the same thing as a dictionary
    with some methods modified.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = DictArithmetic()
    >>> print(d[(0, 0)]) # will print 0
    >>> d[(0, 0)] += 1
    >>> print(d) # will print {(0, 0): 1}

    Compared to an ordinary dictionary.

    >>> g = dict()
    >>> print(g[(0, 0)]) # will raise KeyError
    >>> g[(0, 0)] += 1 # will raise KeyError, since (0, 0) was never set

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = DictArithmetic()
    >>> d[(0, 0)] += 1
    >>> d[(0, 0)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize DictArithmetic with a previous
    dictionary it will be reinitialized to ensure that the DictArithmetic
    contains no zero values. Consider the following example:

    >>> d = DictArithmetic({0: 1, (1, 0): 2, (2, 0): 0})
    >>> print(d) # will print {0: 1, (1, 0): 2}

    We also change the update method so that it follows all the conventions.

    >>> d = DictArithmetic({'a': 1, (0, 1): 2})
    >>> d.update({'a': 0, (1, 1): -1})
    >>> print(d)  # will print {(0, 1): 2, (1, 1): -1}

    We also include arithmetic, addition, subtraction, scalar division,
    scalar multiplication, and all those in place. For example,

    >>> d = DictArithmetic((0, 0)=1, (0, 1)=-2)
    >>> g = d + {(0, 0): -1}
    >>> print(g) # will print {(0, 1): -2}
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8}
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {}

    Finally, adding or subtracting constants will update the () element of the
    dict.

    >>> d = DictArithmetic()
    >>> d += 5
    >>> print(d)
    {(): 5}

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        Initialize the object. If you supply args and kwargs that
        represent a dictionary, they will be reinitialized to follow the
        conventions set in ``__setitem__``.

        Parameters
        ---------
        arguments : define a dictionary with ``dict(*args, **kwargs)``.
            The dictionary will be initialized to follow all the convensions of
            the class.

        """
        super().__init__()
        # reset to make sure everything is in the proper form

        for key, value in _generate_key_value_pairs(*args, **kwargs):
            self[key] += value

    def __getitem__(self, key):
        """__getitem__.

        Overrides the dict.__getitem__ command so that a KeyError is not
        thrown if `key` is not in the dictionary. If the key is not present,
        then we return 0. See the ``dict.__getitem__`` docstring.

        """
        return self.get(key, 0)

    def __setitem__(self, key, value):
        """__setitem__.

        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. See the
        ``dict.__setitem__`` for more.

        """
        if value:
            super().__setitem__(key, value)
        else:
            self.pop(key, 0)

    def copy(self):
        """copy.

        Same as dict.copy, but we adjust the method so that it returns a
        DictArithmetic object, or whatever object is the subclass.

        Returns
        -------
        d : DictArithmetic object, or subclass of.
            Same as ``self.__class__``.

        """
        return self.__class__(self)

    def update(self, *args, **kwargs):
        """update.

        Update the dictionary but following all the conventions of this class.

        Parameters
        ----------
        arguments : defines a dictionary, ie ``d = dict(*args, **kwargs)``.
            Each element in d will be added in place to this instance following
            all the required convensions.

        """
        for k, v in _generate_key_value_pairs(*args, **kwargs):
            self[k] = v

    def __add__(self, other):
        """__add__.

        Add two DictArithmetics or dicts, return the sum.

        Parameters
        ----------
        other : a DictArithmetic or dict object.

        Return
        ------
        d : a DictArithmetic object, self + other.

        Examples
        --------
        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (0, 1): 1}
        >>> print(d + g)
        {(0, 0): 3}

        """
        d = self.copy()
        d += other
        return d

    def __radd__(self, other):
        """__radd__.

        Add two DictArithmtic objects or dicts.

        Parameters
        ----------
        other : a DictArithmetic or dict object.

        Return
        -------
        d : a DictArithmetic object, self + other.

        Examples
        --------
        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (0, 1): 1}
        >>> print(g + d)
        {(0, 0): 3}

        """
        return self + other

    def __iadd__(self, other):
        """__iadd__.

        Same as the __add__ method, but done in place.

        Parameters
        ----------
        other : a DictArithmetic or dict object, or number.

        Return
        -------
        d : a DictArithmetic object, self.

        Examples
        --------
        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (0, 1): 1}
        >>> d += g
        >>> print(d)
        {(0, 0): 3}

        """
        if isinstance(other, dict):
            for k, v in other.items():
                self[k] += v
        else:
            self[()] += other
        return self

    def __sub__(self, other):
        """__sub__.

        Subtract two DictArithmetic or dicts, return the difference.

        Parameters
        ----------
        other : a DictArithetic or dict object.

        Return
        -------
        d : a DictArithmetic object, self - other.

        Examples
        --------
        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (0, 1): 1}
        >>> print(d - g)  # will print {(0, 0): -1}

        """
        d = self.copy()
        d -= other
        return d

    def __rsub__(self, other):
        """__rsub__.

        Subtract two DictArithmetic or dicts.

        Parameters
        ----------
        other : a DictArithmetic or dict object.

        Return
        -------
        d : a DictArithmetic object, other - self.

        Examples
        --------
        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (0, 1): 1}
        >>> print(g - d)  # will print {(0, 0): 1}

        """
        return -1*self + other

    def __isub__(self, other):
        """__isub__.

        Same as the __sub__ method, but done in place.

        Parameters
        ----------
        other : a DictArithmetic or dict object.

        Returns
        -------
        d : a DictArithmetic object, self.

        Examples
        --------
        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {(0, 0): 2, (0, 1): 1}
        >>> d -= g
        >>> print(d)  # will print {(0, 0): -1}

        """
        if isinstance(other, dict):
            for k, v in other.items():
                self[k] -= v
        else:
            self[()] -= other
        return self

    def __mul__(self, other):
        """__mul__.

        Multiplying a DictArithmetic by a scalar or another dict.

        Parameters
        ----------
        other : numeric or dict/DictArithmetic object.

        Return
        -------
        d : a DictArithmetic object.

        Example
        -------
        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> print(d * 2)
        {(0, 0): 2, (0, 1): -1}

        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {(0,): -1, (0, 2): 1}
        >>> print(d * g)
        {(0, 0, 0): -1, (0, 0, 0, 2): 1, (0, 1, 0): 1, (0, 1, 0, 2): -1}

        Note that if the keys are not tuples, then they will be made tuples! Ie

        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {0: -1, 2: 1}
        >>> print(d * g)
        {(0, 0, 0): -1, (0, 0, 2): 1, (0, 1, 0): 1, (0, 1, 2): -1}

        """
        d = self.copy()
        d *= other
        return d

    def __rmul__(self, other):
        """__rmul__.

        Same as __mul__, but for different ordering.

        Parameters
        ----------
        other : numeric or dict/DictArithmetic object.

        Return
        -------
        d : a DictArithmetic object.

        Example
        -------
        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> print(2 * d)
        {(0, 0): 2, (0, 1): -1}

        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {(0,): -1, (0, 2): 1}
        >>> print(g * d)
        {(0, 0, 0): -1, (0, 0, 0, 2): 1, (0, 1, 0): 1, (0, 1, 0, 2): -1}

        Note that if the keys are not tuples, then they will be made tuples! Ie

        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {0: -1, 2: 1}
        >>> print(g * d)
        {(0, 0, 0): -1, (0, 0, 2): 1, (0, 1, 0): 1, (0, 1, 2): -1}

        """
        return self * other

    def __imul__(self, other):
        """__imul__.

        Same as __mul__, but done in place.

        Parameters
        ----------
        other : numeric or dict/DictArithmetic object.

        Return
        -------
        d : a DictArithmetic object, self.

        Example
        -------
        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> d *= 2
        >>> print(d)
        {(0, 0): 2, (0, 1): -1}

        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {(0,): -1, (0, 2): 1}
        >>> d *= g
        >>> print(d)
        {(0, 0, 0): -1, (0, 0, 0, 2): 1, (0, 1, 0): 1, (0, 1, 0, 2): -1}

        Note that if the keys are not tuples, then they will be made tuples! Ie

        >>> d = DictArithmetic({(0, 0): 1, (0, 1): -1})
        >>> g = {0: -1, 2: 1}
        >>> d *= g
        >>> print(d)
        {(0, 0, 0): -1, (0, 0, 2): 1, (0, 1, 0): 1, (0, 1, 2): -1}

        """
        if isinstance(other, dict):
            items, oitems = tuple(self.items()), tuple(other.items())
            self.clear()
            for k, v in items:
                kp = k if isinstance(k, tuple) else (k,)
                for ko, vo in oitems:
                    kop = ko if isinstance(ko, tuple) else (ko,)
                    self[kp + kop] += v * vo

        else:
            for k in tuple(self.keys()):
                self[k] *= other

        return self

    def __pow__(self, exponent):
        """__pow__.

        Raise the object to an integer power. Note that for example
        ``self ** 3 == self * self * self``.

        Parameters
        ----------
        exponent : int.
            Integer power to raise the DictArithmetic to.

        Returns
        ------
        res : DictArithmetic.

        """
        d = self.copy()
        d **= exponent
        return d

    def __ipow__(self, exponent):
        """__ipow__.

        Same as ``__pow__`` but in place.

        Raise the object to an integer power. Note that for example
        ``self ** 3 == self * self * self``.

        Parameters
        ----------
        exponent : int.
            Integer power to raise the DictArithmetic to.

        Returns
        ------
        self : DictArithmetic.

        """
        if not isinstance(exponent, int) or exponent <= 0:
            raise ValueError("Exponent must be a positive integer")

        if exponent > 1:
            old = self.copy()
            for _ in range(exponent-1):
                self *= old

        return self

    def __truediv__(self, other):
        """__truediv__.

        Dividing a DictArithmetic by a scalar.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        d : a DictArithmetic object.

        Example
        -------
        >>> d = DictArithmetic((0, 0)=1, (0, 1)=-1)
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
        d : a DictArithmetic object, self.

        Example
        -------
        >>> d = DictArithmetic((0, 0)=1, (0, 1)=-1)
        >>> d /= 2
        >>> print(d)
        {(0, 0): .5, (0, 1): -.5}

        """
        for k in tuple(self.keys()):
            self[k] /= other
        return self

    def __floordiv__(self, other):
        """__floordiv__.

        Floor dividing a DictArithmetic by a scalar.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        d : a DictArithmetic object.

        Example
        -------
        >>> d = DictArithmetic((0, 0)=2, (0, 1)=-1)
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
        d : a DictArithmetic object, self.

        Example
        -------
        >>> d = DictArithmetic((0, 0)=2, (0, 1)=-1)
        >>> d //= 2
        >>> print(d)
        {(0, 0): 1, (0, 1): 0}

        """
        for k in tuple(self.keys()):
            self[k] //= other
        return self

    def __pos__(self):
        """__pos__.

        Called when unary + is used. Returns copy of self.

        Returns
        -------
        res : DictArithmetic object.
            Copy of self.

        """
        return self.copy()

    def __neg__(self):
        """__neg__.

        Called when unary - is used. Returns -1 * self.

        Returns
        -------
        res : DictArithmetic object.
            -1 * self.

        """
        return -1 * self

    def __round__(self, ndigits=None):
        """round.

        Round values of the DictArithmetic object.

        Parameters
        ----------
        ndigits : int.
            Number of decimal digits to round to.

        Returns
        -------
        res : DictArithmetic object.
            Copy of self but with each value rounded to ``ndigits`` decimal
            digits. Each value has a type according to the docstring
            specifications of ``round``, see ``help(round)``.

        """
        d = self.__class__()
        for k, v in self.items():
            try:
                d[k] = round(v, ndigits)
            except TypeError:  # symbols don't have round methods
                pass
        return d

    def subs(self, *args, **kwargs):
        """subs.

        Replace any ``sympy`` symbols that are used in the dict with values.
        Please see ``help(sympy.Symbol.subs)`` for more info.

        Parameters
        ----------
        arguments : substitutions.
            Same parameters as are inputted into ``sympy.Symbol.subs``.

        Returns
        -------
        res : DictArithmetic object.
            Same as ``self`` but with all the symbols replaced with values.

        """
        d = self.__class__()
        for k, v in self.items():
            try:
                val = v.subs(*args, **kwargs)
            except AttributeError:
                val = v
            d[k] = val

        return d
