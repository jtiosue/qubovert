#   Copyright 2020 Joseph T. Iosue
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

from qubovert.utils import QUSOMatrix
from qubovert.problems import Problem


__all__ = 'NumberPartitioning',


class NumberPartitioning(Problem):
    """NumberPartitioning.

    Class to manage converting the Number Partitioning problem to and from its
    QUBO and QUSO formluations. Based on the paper "Ising formulations of many
    NP problems", hereforth designated as [Lucas].

    The goal of the NumberPartitioning problem is as follows (quoted from
    [Lucas]):

    Given a list of N positive numbers S = [n1, . . . , nN], is there a
    partition of this set of numbers into two disjoint subsets R and S − R,
    such that the sum of the elements in both sets is the same?

    Note that if we can't do this partitioning, then the next goal is to
    find a partition that almost does this, ie a partition that minimizes
    the difference in the sum between the two partitions.

    This class inherits some methods and attributes from the Problem class. For
    more info, see ``help(qubovert.problems.Problem)``.

    Example
    -------
    >>> from qubovert.problems import NumberPartitioning
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
            tuple or list of N nonzero numbers that we are attempting to
            partition into two partitions of equal sum.

        Example
        -------
        >>> problem = NumberPartitioning([1, 2, 3, 4])

        """
        if not all(S):
            raise ValueError("Numbers must be nonzero")
        self._input_type = type(S)
        self._S = self._input_type(x for x in S)  # copy the input
        self._N = len(S)

    @property
    def S(self):
        """S.

        A copy of the S list. Updating the copy will not update the instance
        list.

        Return
        ------
        S : tuple or list.
            A copy of the list of numbers defining the partitioning problem.

        """
        return self._input_type(x for x in self._S)

    @property
    def num_binary_variables(self):
        """num_binary_variables.

        The number of binary variables that the QUBO and QUSO use.

        Return
        ------
        num : int.
            The number of variables in the QUBO/QUSO formulation.

        """
        return self._N

    def to_quso(self, A=1):
        r"""to_quso.

        Create and return the number partitioning problem in QUSO form
        following section 2.1 of [Lucas]. It is
        the value such that the solution to the QUSO formulation is 0
        if a valid number partitioning exists.

        Parameters
        ----------
        A: positive float (optional, defaults to 1).
            Factor in front of objective function. See section 2.1 of [Lucas].

        Return
        ------
        L : qubovert.utils.QUSOMatrix object.
            For most practical purposes, you can use QUSOMatrix in the
            same way as an ordinary dictionary. For more information, see
            ``help(qubovert.utils.QUSOMatrix)``.

        Example
        -------
        >>> problem = NumberPartitioning([1, 2, 3, 4])
        >>> L = problem.to_quso()

        """
        L = QUSOMatrix({(i,): self._S[i] for i in range(self._N)})
        return A * L * L

    def convert_solution(self, solution, spin=False):
        """convert_solution.

        Convert the solution to the QUBO or QUSO to the solution to the
        Number Partitioning problem.

        Parameters
        ----------
        solution : iterable or dict.
            The QUBO or QUSO solution output. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or 1 or -1 for QUSO), or it can be a dictionary that maps the
            label of the variable to is value.
        spin : bool (optional, defaults to False).
            `spin` indicates whether ``solution`` is the solution to the
            boolean {0, 1} formulation of the problem or the spin {1, -1}
            formulation of the problem. This parameter usually does not matter,
            and it will be ignored if possible. The only time it is used is if
            ``solution`` contains all 1's. In this case, it is unclear whether
            ``solution`` came from a spin or boolean formulation of the
            problem, and we will figure it out based on the ``spin`` parameter.

        Return
        ------
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

    def is_solution_valid(self, solution, spin=False):
        """is_solution_valid.

        Returns whether or not the proposed solution partitions S into two sets
        of equal sum.

        Parameters
        ----------
        solution : iterable or dict.
            solution can be the output of NumberPartitioning.convert_solution,
            or the  QUBO or QUSO solver output. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or 1 or -1 for QUSO), or it can be a dictionary that maps the
            label of the variable to is value.
        spin : bool (optional, defaults to False).
            `spin` indicates whether ``solution`` is the solution to the
            boolean {0, 1} formulation of the problem or the spin {1, -1}
            formulation of the problem. This parameter usually does not matter,
            and it will be ignored if possible. The only time it is used is if
            ``solution`` contains all 1's. In this case, it is unclear whether
            ``solution`` came from a spin or boolean formulation of the
            problem, and we will figure it out based on the ``spin`` parameter.

        Return
        ------
        valid : boolean.
            True if the proposed solution is valid, else False.

        """
        not_converted = (
            not isinstance(solution, tuple) or len(solution) != 2 or
            not isinstance(solution[0], self._input_type) or
            not isinstance(solution[1], self._input_type)
        )

        if not_converted:
            solution = self.convert_solution(solution, spin)

        return sum(solution[0]) == sum(solution[1])
