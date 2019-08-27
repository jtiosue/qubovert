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

from . import Problem, Conversions


__all__ = 'BO',


class BO(Conversions):
    """BO.

    Parent class for some Binary Optimization classes, such as
    ``qubovert.QUBO``, ``qubovert.Ising``, etc.

    BO inherits some methods and attributes the ``DictArithmetic`` class.
    See ``help(qubovert.utils.DictArithmetic)``.

    BO inherits some methods and attributes the ``Conversions`` class.
    See ``help(qubovert.utils.Conversions)``.

    """

    def __new__(cls, *args, **kwargs):
        """__new__.

        This class deals with Binary Optimization models. See child
        classes for info on inputs.

        Parameters
        ---------
        Defined in child classes.

        Return
        -------
        obj : instance of the child class.

        """
        obj = super().__new__(cls)
        obj._mapping, obj._reverse_mapping, obj._next_label = {}, {}, 0
        return obj

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

    @property
    def max_index(self):
        """max_index.

        Return the maximum label of the integer labeled version of the problem.

        Return
        ------
        m : int.

        """
        return self.num_binary_variables - 1

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

    def __setitem__(self, key, value):
        """__setitem__.

        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        Ising dictionary will ever have zero value.

        Parameters
        ---------
        key : tuple.
            Element of the dictionary that follows convensions of the class.
        value : numeric.
            Value corresponding to the key.

        """
        super().__setitem__(key, value)

        for i in key:
            if i not in self._mapping:
                self._mapping[i] = self._next_label
                self._reverse_mapping[self._next_label] = i
                self._next_label += 1

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
