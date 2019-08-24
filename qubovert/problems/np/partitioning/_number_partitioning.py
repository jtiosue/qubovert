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

"""_number_partitioning.py.

Contains the NumberPartitioning class. See
``help(qubovert.problems.NumberPartitioning)``.

"""

from qubovert.utils import Problem, IsingMatrix


__all__ = 'NumberPartitioning',


class NumberPartitioning(Problem):
    """NumberPartitioning.

    Class to manage converting the Number Partitioning problem to and from its
    QUBO and Ising formluations. Based on the paper hereforth designated as
    [Lucas].

    The goal of the NumberPartitioning problem is as follows (quoted from
    [Lucas]):

    Given a list of N positive numbers S = [n1, . . . , nN], is there a
    partition of this set of numbers into two disjoint subsets R and S − R,
    such that the sum of the elements in both sets is the same?

    Note that if we can't do this partitioning, then the next goal is to
    find a partition that almost does this, ie a partition that minimizes
    the difference in the sum between the two partitions.

    This class inherits some methods and attributes from the Problem class. For
    more info, see ``help(qubovert.utils.Problem)``.

    Example usage
    -------------
    >>> from qubovert import NumberPartitioning
    >>> from any_module import qubo_solver
    >>> # or you can use my bruteforce solver...
    >>> # from qubovert.utils import solve_qubo_bruteforce as qubo_solver
    >>> S = [1, 2, 3, 4]
    >>> problem = NumberPartitioning(S)
    >>> Q = problem.to_qubo()
    >>> obj, sol = qubo_solver(Q)
    >>> solution = problem.convert_solution(sol)

    >>> print(solution)
    ([1, 4], [2, 3])  # or ([2, 3], [1, 4])

    >>> print(problem.is_solution_valid(solution))
    True  # since sum([1, 4]) == sum([2, 3])

    >>> print(obj == 0)
    True  # since the solution is valid.

    References
    ----------
    .. [Lucas] Andrew Lucas. Ising formulations of many np problems. Frontiers
       in Physics, 2:5, 2014.

    """

    def __init__(self, S):
        """__init__.

        The goal of the NumberPartitioning problem is as follows (quoted from
        [Lucas]):

        Given a list of N positive numbers S = [n1, . . . , nN], is there a
        partition of this set of numbers into two disjoint subsets R and S − R,
        such that the sum of the elements in both sets is the same?

        Note that if we can't do this partitioning, then the next goal is to
        find a partition that almost does this, ie a partition that minimizes
        the difference in the sum between the two partitions.

        All naming conventions follow the names in the paper [Lucas].

        Parameters
        ----------
        S: tuple or list.
            tuple or list of N positive numbers that we are attempting to
            partition into two partitions of equal sum.

        Example
        --------
        >>> problem = NumberPartitioning([1, 2, 3, 4])

        """
        self._input_type = type(S)
        self._S = self._input_type(x for x in S)  # copy the input
        self._N = len(S)

    @property
    def S(self):
        """S.

        A copy of the S list. Updating the copy will not update the instance
        list.

        Return
        -------
        S : tuple or list.
            A copy of the list of numbers defining the partitioning problem.

        """
        return self._input_type(x for x in self._S)

    @property
    def num_binary_variables(self):
        """num_binary_variables.

        The number of binary variables that the QUBO and Ising use.

        Return
        -------
        num : int.
            The number of variables in the QUBO/Ising formulation.

        """
        return self._N

    def to_ising(self, A=1):
        r"""to_ising.

        Create and return the number partitioning problem in Ising form
        following section 2.1 of [Lucas]. It is
        the value such that the solution to the Ising formulation is 0
        if a valid number partitioning exists.

        Parameters
        ----------
        A: positive float (optional, defaults to 1).
            Factor in front of objective function. See section 2.1 of [Lucas].

        Return
        -------
        I : qubovert.utils.IsingMatrix object.
            For most practical purposes, you can use IsingMatrix in the
            same way as an ordinary dictionary. For more information, see
            ``help(qubovert.utils.IsingMatrix)``.

        Example
        --------
        >>> problem = NumberPartitioning([1, 2, 3, 4])
        >>> I = problem.to_ising()

        """
        I = IsingMatrix(() = A * sum(pow(x, 2) for x in self._S))

        for i in range(self._N):
            for j in range(i+1, self._N):
                I[(i, j)] += (2 * A * self._S[i] * self._S[j])

        return I

    def convert_solution(self, solution):
        """convert_solution.

        Convert the solution to the QUBO or Ising to the solution to the
        Number Partitioning problem.

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
        res: tuple of iterables (partition1, partition2).
            partition1 : list, tuple, or iterable.
                The first partition. If the inputted S is a tuple, then
                partition1 will as a tuple, otherwise it will be a list.
            partition2 : list, tuple, or iterable.
                The second partition.

        Example
        -------
        >>> problem = NumberPartitioning([1, 2, 3, 4])
        >>> Q = problem.to_qubo()
        >>> value, solution = solve_qubo(Q)
        >>> print(solution)
        [0, 1, 1, 0]
        # ie 1 and 4 are in one partition, and 2 and 3 are in the other

        >>> print(problem.convert_solution(solution))
        ([1, 4], [2, 3])

        """
        if not isinstance(solution, dict):
            solution = dict(enumerate(solution))
        partition1 = self._input_type(
            self._S[i] for i, v in solution.items() if v == 1
        )
        partition2 = self._input_type(
            self._S[i] for i, v in solution.items() if v != 1
        )
        return partition1, partition2

    def is_solution_valid(self, solution):
        """is_solution_valid.

        Returns whether or not the proposed solution partitions S into two sets
        of equal sum.

        Parameters
        ----------
        solution : iterable or dict.
            solution can be the output of NumberPartitioning.convert_solution,
            or the  QUBO or Ising solver output. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or -1 or 1 for Ising), or it can be a dictionary that maps the
            label of the variable to is value.

        Return
        -------
        valid : boolean.
            True if the proposed solution is valid, else False.

        """
        not_converted = (
            not isinstance(solution, tuple) or len(solution) != 2 or
            not isinstance(solution[0], self._input_type) or
            not isinstance(solution[1], self._input_type)
        )

        if not_converted:
            solution = self.convert_solution(solution)

        return sum(solution[0]) == sum(solution[1])
