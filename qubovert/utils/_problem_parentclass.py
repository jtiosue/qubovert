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

"""
This file contains the class Problem, which is the parent class to all
the problem classes.
"""

from qubovert import __str__ as MODULE_NAME
from ._conversions import qubo_to_ising, ising_to_qubo


class Problem:

    """
    This acts a parent class to all the QUBO and Ising conversion problem
    classes. The __new__ method keeps track of the problem args. The repr
    method uses those input args, such that eval(repr(cls)) == cls. Finally, we
    define a __eq__ method to determine if two problems are the same. The rest
    of the methods are to be implemented in child classes.

    Additionally, his class defines the to_qubo and to_ising methods. to_qubo
    calls to_ising and then converts from ising to qubo. to_ising calls
    to_qubo and then converts from qubo to ising. Thus, the child classes
    MUST define either to_qubo or to_ising. In this way, by only defining
    one of those, both are implemented.
    """

    def __new__(cls, *args, **kwargs):
        """
        Creates the object and keeps track of the input arguments and keyword
        arguments. Child classes should not change this. This method will be
        called before every __init__ is called. We use __new__ to keep track
        of input arguments instead of using __init__ so that child class
        implementations don't have to worry about it. Ie child classes
        don't have to call `super().__init__(*args, **kwargs)` in their
        __init__ method.
        """
        # obj = object.__new__(cls)
        obj = super().__new__(cls)
        obj._problem_args, obj._problem_kwargs = args, kwargs.copy()
        return obj

    def __repr__(self):
        """
        Defined such that the following is true (assuming you have imported
        qubovert as qubovert).
            >>> s = Class_derivedfrom_Problem(*args)
            >>> eval(repr(s)) == s
        """
        return MODULE_NAME + "." + str(self)

    def __str__(self):
        """
        Defined such that the following is true (assuming you have imported
        * from qubovert).
            >>> s = Class_derivedfrom_Problem(*args)
            >>> eval(str(s)) == s
        """
        s = self.__class__.__name__ + "("
        for a in self._problem_args:
            val = str(a) if not isinstance(a, str) else "'%s'" % a
            s += val + ", "
        for k, v in self._problem_kwargs.items():
            val = str(v) if not isinstance(v, str) else "'%s'" % v
            s += str(k) + "=" + val + ", "
        return s[:-2] + ")"

    def __eq__(self, other):
        """
        Find if self and other define the same problem.

        other: must be a class derived from Problem.

        returns a boolean.
        """
        return (
            isinstance(other, type(self)) and
            self._problem_args == other._problem_args and
            self._problem_kwargs == other._problem_kwargs
        )

    def to_qubo(self, *args, **kwargs):
        """
        Create and return upper triangular QUBO representing the problem.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls to_ising and
        converts the ising formulation to a QUBO formulation.

        returns the tuple (Q, offset).
            Q is the upper triangular QUBO matrix, a QUBOMatrix object.
                For most practical purposes, you can use QUBOMatrix in the
                same way as an ordinary dictionary. For more information,
                see help(qubovert.utils.QUBOMatrix).
            offset is a float. It is the sum of the terms in the formulation in
                the cited paper that don't involve any variables.
        """
        return ising_to_qubo(*self.to_ising(*args, **kwargs))

    def to_ising(self, *args, **kwargs):
        """
        Create and return upper triangular J representing the coupling of the
        Ising formulation of the problem and the h representing the field.
        Should be implemented in child classes. If this method is not
        implemented in the child class, then it simply calls to_qubo and
        converts the QUBO formulation to an Ising formulation.

        returns the tuple (h, J, offset).
            h represents the field of each spin in the Ising formulation.
                h is a IsingField object. For most practical purposes, you can
                use IsingField in he same way as an ordinary dictionary. For
                more information, see help(QUBOConver.utils.IsingField).
            J is the upper triangular coupling matrix, a IsingCoupling object.
                For most practical purposes, you can use IsingCoupling in the
                same way as an ordinary dictionary. For more information,
                see help(qubovert.utils.IsingCoupling).
            offset is a float. It is the sum of the terms in the formulation in
                the cited paper that don't involve any variables.
        """
        return qubo_to_ising(*self.to_qubo(*args, **kwargs))

    def convert_solution(self, solution, *args, **kwargs):
        """
        Convert the solution to the QUBO to the solution to the problem.
        Should be implemented in child classes. If it is not implemented in the
        child class, then this function will by default return the same
        solution as what inputted.

        solution is the QUBO or Ising solution output. The QUBO solution output
            is either a list/tuple where indices specify the label of the
            binary variable and the element specifies whether it's 0 or 1, or
            it can be a dictionary that maps the label of the binary variable
            to whether it is a 0 or 1. The Ising solution output is the same,
            but with -1 corresponding to the QUBO 0, and 1 corresponding to the
            QUBO 1.
        """
        return solution

    def is_solution_valid(self, solution, *args, **kwargs):
        """
        Returns whether or not the proposed solution is valid. Should be
        implemented in child classes. If it is not implemented in the child
        class, then this function will by default return True.

        solution can either be the output of convert_solution or it
            can be the actual QUBO or Ising solution output. The QUBO solution
            output is either a list where indices specify the label of the
            binary variable and the element specifies whether it's 0 or 1, or
            it can be a dictionary that maps the label of the binary variable
            to whether it is a 0 or 1. The Ising solution output is the same,
            but with -1 corresponding to the QUBO 0, and 1 corresponding to the
            QUBO 1.

        returns a boolean, True if the proposed solution is valid, else False.
        """
        return True

    def num_binary_variables(self, *args, **kwargs):
        """
        Find the number of binary variables that the QUBO uses. Should be
        implemented in the child class.

        returns an integer, the number of variables in the QUBO formulation.
        """
        raise NotImplementedError("Method to be implemented in child classes")
