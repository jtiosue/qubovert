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

from . import Problem, DictArithmetic, Conversions


__all__ = 'BO',


class BO(DictArithmetic, Conversions):
    """BO.

    Parent class for some Binary Optimization classes, such as
    ``qubovert.QUBO``, ``qubovert.Ising``, etc.

    BO inherits some methods and attributes the ``DictArithmetic`` class.
    See ``help(qubovert.utils.DictArithmetic)``.

    BO inherits some methods and attributes the ``Conversions`` class.
    See ``help(qubovert.utils.Conversions)``.

    A child class of BO must define at least one of either a ``to_qubo`` or
    ``to_ising`` method.

    BO defines a lot of arithmetic and conventions for Binary Optimization
    models.

    Example usage
    -------------
    >>> d = BO()
    >>> d[(0,)] += 2
    >>> print(d)
    {(0,): 2}
    >>> d -= 3
    >>> print(d)
    {(0,): 2, (): -3}
    >>> d * 4
    >>> print(d)
    {(0,): 8, (): -12}
    >>> d += {('a', 'b'): 3, ('c', ): -2, (0,): -1}
    >>> print(d)
    {('a', 'b'): 3, ('c',): -2, (0,): 7, {}: -12}

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        This class deals with Binary Optimization models. See child
        classes for info on inputs.

        Initialize the object. If you supply args and kwargs that
        represent a dictionary, they will be reinitialized to follow the
        conventions set in ``__setitem__``.

        Parameters
        ---------
        *args and **kwargs : see the docstring for dict.

        """
        self._mapping = {}  # convert labels to ints
        self._reverse_mapping = {}  # convert ints to labels
        self._next_label = 0
        d = dict(*args, **kwargs)
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
        return self[()]

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

    # The following method just uses code written in the Problem class.
    # I want to avoid duplicate code as much as possible! So we use what we
    # have already written. But we don't make BO a subclass of Problem
    # for various reasons (some inconsistencies in code, convention, and
    # docstrings).

    def solve_bruteforce(self, *args, **kwargs):
        """solve_bruteforce.

        Solve the problem bruteforce. THIS SHOULD NOT BE USED FOR LARGE
        PROBLEMS! This converts the problem to PUBO with integer labels, solves
        it with ``qubovert.utils.solve_pubo_bruteforce``, and then calls and
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
