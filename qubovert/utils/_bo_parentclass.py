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

"""_bo_parentclass.py.

Contains the BO parent class. See ``help(qubovert.utils.BO)``. Used as a parent
class for some Binary Optimization classes, such as ``qubovert.QUBO``,
``qubovert.Ising``, etc.

"""

from . import Problem, DictArithmetic


class BO(DictArithmetic):
    """BO.

    Parent class for some Binary Optimization classes, such as
    ``qubovert.QUBO``, ``qubovert.Ising``, etc.

    BO inherits some methods and attributes the ``DictArithmetic`` class.
    See ``help(qubovert.utils.DictArithmetic)``.

    A child class of BO must define at least one of either a ``to_qubo`` or
    ``to_ising`` method.

    BO defines a lot of arithmetic and conventions for Binary Optimization
    models.

    Example usage
    -------------
    >>> d = DictArithmetic()
    >>> d[0] += 2
    >>> print(d)
    {0: 2}
    >>> d -= 3
    >>> print(d)
    {0: 2} - 3
    >>> d * 4
    >>> print(d)
    {0: 8} - 12
    >>> d += {('a', 'b'): 3, 'c': -2, 0: -1}
    >>> print(d)
    {('a', 'b'): 3, 'c': -2, 0: 7} - 12

    """

    def __init__(self, d=None, offset=0):
        """__init__.

        This class deals with Binary Optimization models. See child
        classes for info on inputs.

        Parameters
        ----------
        d : dict (optional, defaults to None).
            Dictionary.
        offset: numeric (optional, defaults to 0).

        """
        self._offset = offset
        self._mapping = {}  # convert labels to ints
        self._reverse_mapping = {}  # convert ints to labels
        self._next_label = 0
        if d is None:
            d = {}

        super().__init__()
        for k, v in d.items():
            self[k] = v

    @property
    def offset(self):
        """offset.

        Return the offset of the BO problem, ie the part of the BO that
        does not depend on any variables.

        Return
        ------
        offset : float.
            The part of the BO that does not depend on any variables.

        """
        return self._offset

    @property
    def mapping(self):
        """mapping.

        Return a copy of the mapping dictionary that maps the provided
        labels to integers from 0 to n-1, where n is the number of variables
        in the problem.

        Return
        ------
        mapping : dict.
            Dictionary that maps provided labels to integer labels.

        """
        return self._mapping.copy()

    @property
    def reverse_mapping(self):
        """reverse_mapping.

        Return a copy of the reverse_mapping dictionary that maps the integer
        labels to the provided labels. Opposite of ``mapping``.

        Return
        ------
        reverse_mapping : dict.
            Dictionary that maps integer labels to provided labels.

        """
        return self._reverse_mapping.copy()

    @property
    def num_binary_variables(self):
        """num_binary_variables.

        Return the number of binary variables in the problem.

        Return
        ------
        n : int.
            Number of binary variables in the problem.

        """
        return self._next_label

    def __repr__(self):
        """__repr__.

        Return the representation of the BO object.

        Return
        ------
        r : str.
            BO dictionary string with an offset if necessary.

        Example
        -------
        >>> d = DictArithmetic({('a', 'b'): -1, ('b', 0): 2, 1: 3})
        >>> repr(d)
        '{('a', 'b'): -1, ('b', 0): 2, 1: 3}'

        >>> d = Arithmetic({('a', 'b'): -1, ('b', 0): 2, 1: 3}, 2)
        >>> repr(d)
        '{('a', 'b'): -1, ('b', 0): 2, 1: 3} + 2'

        """
        s = super().__repr__()
        if self._offset == 0:
            return s
        elif self._offset < 0:
            s += " - "
        else:
            s += " + "
        return s + str(abs(self._offset))

    def __eq__(self, other):
        """__eq__.

        Finds whether or not two BOs are equal.

        Parameters
        ----------
        other : BO object or dict.
            Other BO to compare.

        Returns
        -------
        res : bool.

        """
        if not isinstance(other, BO):
            return self._offset == 0 and super().__eq__(other)
        return super().__eq__(other) and self._offset == other.offset

    def is_solution_valid(self, solution):
        """is_solution_valid.

        Included for consistency with other problem classes. Always returns
        True.

        Parameters
        ----------
        solution : iterable or dict.
            The QUBO or Ising solution output, or the output of
            ``convert_solution``. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or -1 or 1 for Ising), or it can be a dictionary that maps the
            label of the variable to is value.

        Return
        ------
        valid : bool.
            Always returns True.

        """
        return True

    def copy(self, *args, **kwargs):
        """copy.

        Same as dict.copy, but we adjust the method so that it returns a
        BO object.

        Parameters
        ----------
        *args and **kwargs : see dict.copy.

        """
        x = super().copy(*args, **kwargs)
        x += self._offset
        return x

    def update(self, other):
        """update.

        Update the QUBO with other. Same as dict.update, except ``other`` can
        also be a BO type.

        Parameters
        ----------
        other : dict or BO object.

        """
        if isinstance(other, BO):
            self._offset = other.offset
        for k, v in other.items():
            self[k] = v

    def __iadd__(self, other):
        """__iadd__.

        Same as the __add__ method, but done in place.

        Parameters
        ----------
        other : a BO or dict object, or a numeric type.

        Return
        -------
        Q : a BO object, self.

        """
        if not isinstance(other, (BO, dict)):
            self._offset += other
            return self
        elif isinstance(other, BO):
            self._offset += other.offset
        return super().__iadd__(other)

    def __isub__(self, other):
        """__isub__.

        Same as the __sub__ method, but done in place.

        Parameters
        ----------
        other : a BO or dict object, or a numeric type.

        Returns
        -------
        Q : a BO object, self.

        """
        if not isinstance(other, (BO, dict)):
            self._offset -= other
            return self
        elif isinstance(other, BO):
            self._offset -= other.offset
        return super().__isub__(other)

    def __imul__(self, other):
        """__imul__.

        Same as ``__mul__``, but done in place.

        Parameters
        ----------
        other : numeric.

        Return
        -------
        B : a BO object, self.

        """
        self._offset *= other
        return super().__imul__(other)

    def __itruediv__(self, other):
        """__itruediv__.

        Same as ``__truediv__``, but done in place. For parameters and return
        type, see ``__truediv__``.

        """
        self._offset /= other
        return super().__itruediv__(other)

    def __ifloordiv__(self, other):
        """__ifloordiv__.

        Same as ``__floordiv__``, but done in place. For parameters and return
        type, see ``__floordiv__``.

        """
        self._offset //= other
        return super().__ifloordiv__(other)

    # The three following methods just uses code written in the Problem class.
    # I want to avoid duplicate code as much as possible! So we use what we
    # have already written. But we don't make BO a subclass of Problem
    # for various reasons (some inconsistencies in code, convention, and
    # docstrings).

    def to_ising(self, *args, **kwargs):
        """to_ising.

        Create and return upper triangular J representing the coupling of the
        Ising formulation of the problem and the h representing the field.

        Parameters
        ----------
        args and kwargs are the same as the parameters defined in the
        ``to_qubo`` method.

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
        return Problem.to_ising(self, *args, **kwargs)

    def to_qubo(self, *args, **kwargs):
        """to_qubo.

        Create and return upper triangular QUBO representing the problem.
        The labels will be integers from 0 to n-1.

        Parameters
        ----------
        args and kwargs are the same as the parameters defined in the
        ``to_ising`` method.

        Return
        -------
        result : tuple (Q, offset).
            Q : qubovert.utils.super() object.
                The upper triangular QUBO matrix, a super() object.
                For most practical purposes, you can use super() in the
                same way as an ordinary dictionary. For more information,
                see ``help(qubovert.utils.super())``.
            offset : float.
                The sum of the terms in the formulation that don't involve any
                variables.

        """
        return Problem.to_qubo(self, *args, **kwargs)

    def solve_bruteforce(self, *args, **kwargs):
        """solve_bruteforce.

        Solve the problem bruteforce. THIS SHOULD NOT BE USED FOR LARGE
        PROBLEMS! This converts the problem to QUBO with integer labels, solves
        it with ``qubovert.utils.solve_qubo_bruteforce``, and then calls and
        returns ``convert_solution``.

        Parameters
        ----------
        *args and **kwargs : arguments and keyword arguments.
            Contains args and kwargs for the ``to_qubo`` method. Also contains
            a ``all_solutions`` boolean flag, which indicates whether or not
            to return all the solutions, or just the best one found.
            ``all_solutions`` defaults to False.

        Return
        ------
        res : the output or outputs of the ``convert_solution`` method.
            If ``all_solutions`` is False, then ``res`` is just the output
            of the ``convert_solution`` method.
            If ``all_solutions`` is True, then ``res`` is a list of outputs
            of the ``convert_solution`` method, e.g. a converted solution
            for each solution that the bruteforce solver returns.

        """
        return Problem.solve_bruteforce(self, *args, **kwargs)
