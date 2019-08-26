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

"""_alternating_sectors_chain.py.

Contains the AlternatingSectorsChain class. See
``help(qubovert.problems.AlternatingSectorsChain)``.
"""

from qubovert.utils import Problem, IsingMatrix


__all__ = 'AlternatingSectorsChain',


class AlternatingSectorsChain(Problem):
    """AlternatingSectorsChain.

    Class to manage converting Alternating Sectors Chain to and from its QUBO
    and Ising formluations.

    The Alternating Sectors Chain problem has a solution for which
    all the binary variable or spins are equal. It is a trivial problem,
    but useful for benchmarking some solvers or solving techniques.

    AlternatingSectorsChain inherits some methods and attributes from the
    Problem class. See ``help(qubovert.utils.Problem)``.

    Example usage
    -------------
    >>> from qubovert import AlternatingSectorsChain
    >>> from any_module import qubo_solver
    >>> # or you can use my bruteforce solver...
    >>> # from qubovert.utils import solve_qubo_bruteforce as qubo_solver
    >>> problem = AlternatingSectorsChain(10)
    >>> Q  = problem.to_qubo()
    >>> obj, sol = qubo_solver(Q)
    >>> solution = problem.convert_solution(sol)

    >>> print(solution)
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # or (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    >>> print(problem.is_solution_valid(solution))
    True  # since they are all the same.

    """

    def __init__(self, num_binary_variables,
                 chain_length=3, min_strength=1, max_strength=10):
        """__init__.

        The Alternating Sectors Chain problem has a solution for which
        all the binary variable or spins are equal. It is a trivial problem,
        but useful for benchmarking some solvers or solving techniques.

        Parameters
        ----------
        num_binary_variables : int > 0.
            Number of variables in the problem.
        chain_length : int > 1 (optional, defaults to 3).
            The length of a chain of couples.
        min_strength : int > 0 (optional, defaults to 1).
            The strength of the couples for the minimium chain.
        max_strength : int > 0 (optional, defaults to 10).
            The strength of the couples for the maximum chain.

        Examples
        -------
        >>> args = n, l, min_s, max_s = 6, 3, 1, 5
        >>> problem = AlternatingSectorsChain(*args)
        >>> h, J, offset = problem.to_ising(pbc=True)
        >>> h
        {}
        >>> offset
        0
        >>> J
        {(0, 1): 5, (1, 2): 5, (2, 3): 5, (3, 4): 1, (4, 5): 1, (5, 0): 1}

        """
        self._N, self._chain_length = num_binary_variables, chain_length
        self._min_strength, self._max_strength = -min_strength, -max_strength
        if min_strength < 0 or max_strength < 0:
            raise ValueError("Coupling strengths must be positive")
        elif num_binary_variables < 1:
            raise ValueError("Must have at least 1 binary variable")
        elif chain_length < 2:
            raise ValueError("Chain length must be at least 2")

    @property
    def num_binary_variables(self):
        """num_binary_variables.

        The number of binary variables that the QUBO and Ising use.

        Return
        -------
        num : integer.
            The number of variables in the QUBO/Ising formulation.

        """
        return self._N

    def to_ising(self, pbc=False):
        r"""to_ising.

        Create and return the alternating Sectors chain problem in Ising form
        The J coupling matrix for the Ising will be returned as an
        uppertriangular dictionary. Thus, the problem becomes minimizing
        :math:`\sum_{i <= j} z_i z_j J_{ij} + \sum_{i} z_i h_i + offset`.

        Parameters
        ----------
        pbc: bool (optional, defaults to False).
            Whether or not to use periodic boundary conditions.

        Return
        -------
        L : qubovert.utils.IsingMatrix object.
            For most practical purposes, you can use IsingMatrix in the
            same way as an ordinary dictionary. For more information, see
            ``help(qubovert.utils.IsingMatrix)``.

        Example
        -------
        >>> args = n, l, min_s, max_s = 6, 3, 1, 5
        >>> problem = AlternatingSectorsChain(*args)
        >>> L = problem.to_ising(pbc=True)
        >>> L
        {(0, 1): 5, (1, 2): 5, (2, 3): 5, (3, 4): 1, (4, 5): 1, (5, 0): 1}

        >>> L = problem.to_ising(pbc=False)
        >>> L
        {(0, 1): 5, (1, 2): 5, (2, 3): 5, (3, 4): 1, (4, 5): 1}

        """
        L = IsingMatrix()

        for q in range(self._N-1):
            L[(q, q+1)] = (
                self._min_strength
                if (q // self._chain_length) % 2
                else self._max_strength
            )

        if pbc:
            L[(self._N-1, 0)] = (
                self._min_strength
                if ((self._N-1) // self._chain_length) % 2
                else self._max_strength
            )

        return L

    def convert_solution(self, solution):
        """convert_solution.

        Convert the solution to the QUBO or Ising to the solution to the
        Alternating Sectors Chain problem.

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
        res : tuple.
            Value of each spin, -1 or 1.

        Examples
        --------
        >>> problem = AlternatingSectorsChain(5)
        >>> problem.convert_solution([0, 0, 1, 0, 1])
        (-1, -1, 1, -1, 1)
        >>> problem.convert_solution([-1, -1, 1, -1, 1])
        (-1, -1, 1, -1, 1)

        """
        if isinstance(solution, dict):
            solution = tuple(v for k, v in sorted(solution.items()))
        return tuple(x if x else -1 for x in solution)

    def is_solution_valid(self, solution):
        """is_solution_valid.

        Returns whether or not the proposed solution is the correct solution,
        ie that all the variables are equal.

        Parameters
        ----------
        solution : iterable or dict.
            solution can be the output of
            AlternatingSectorsChain.convert_solution,
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
        if isinstance(solution, dict):
            solution = self.convert_solution(solution)

        return all(
            x == 1 for x in solution
        ) or all(
            x != 1 for x in solution
        )
