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

"""_alternating_sectors_chain.py.

Contains the AlternatingSectorsChain class. See
``help(qubovert.problems.AlternatingSectorsChain)``.
"""

from qubovert.utils import QUSOMatrix, boolean_to_spin, is_solution_spin
from qubovert.problems import Problem


__all__ = 'AlternatingSectorsChain',


class AlternatingSectorsChain(Problem):
    """AlternatingSectorsChain.

    Class to manage converting Alternating Sectors Chain to and from its QUBO
    and QUSO formluations.

    The Alternating Sectors Chain problem has a solution for which
    all the boolean variables or spins are equal. It is a trivial problem,
    but useful for benchmarking some solvers or solving techniques.

    AlternatingSectorsChain inherits some methods and attributes from the
    Problem class. See ``help(qubovert.problems.Problem)``.

    Example
    -------
    >>> from qubovert.problems import AlternatingSectorsChain
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
        all the boolean variables or spins are equal. It is a trivial problem,
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
        --------
        >>> args = n, l, min_s, max_s = 6, 3, 1, 5
        >>> problem = AlternatingSectorsChain(*args)
        >>> h, J, offset = problem.to_quso(pbc=True)
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

        The number of binary variables that the QUBO and QUSO use.

        Return
        ------
        num : integer.
            The number of variables in the QUBO/QUSO formulation.

        """
        return self._N

    def to_quso(self, pbc=False):
        r"""to_quso.

        Create and return the alternating Sectors chain problem in QUSO form
        The J coupling matrix for the QUSO will be returned as an
        uppertriangular dictionary. Thus, the problem becomes minimizing
        :math:`\sum_{i <= j} z_i z_j J_{ij} + \sum_{i} z_i h_i + offset`.

        Parameters
        ----------
        pbc: bool (optional, defaults to False).
            Whether or not to use periodic boundary conditions.

        Return
        ------
        L : qubovert.utils.QUSOMatrix object.
            For most practical purposes, you can use QUSOMatrix in the
            same way as an ordinary dictionary. For more information, see
            ``help(qubovert.utils.QUSOMatrix)``.

        Example
        -------
        >>> args = n, l, min_s, max_s = 6, 3, 1, 5
        >>> problem = AlternatingSectorsChain(*args)
        >>> L = problem.to_quso(pbc=True)
        >>> L
        {(0, 1): 5, (1, 2): 5, (2, 3): 5, (3, 4): 1, (4, 5): 1, (5, 0): 1}

        >>> L = problem.to_quso(pbc=False)
        >>> L
        {(0, 1): 5, (1, 2): 5, (2, 3): 5, (3, 4): 1, (4, 5): 1}

        """
        L = QUSOMatrix()

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

    def convert_solution(self, solution, spin=False):
        """convert_solution.

        Convert the solution to the QUBO or QUSO to the solution to the
        Alternating Sectors Chain problem.

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
        res : tuple.
            Value of each spin, -1 or 1.

        Examples
        --------
        >>> problem = AlternatingSectorsChain(5)
        >>> problem.convert_solution([0, 0, 1, 0, 1])
        (1, 1, -1, 1, -1)
        >>> problem.convert_solution([-1, -1, 1, -1, 1])
        (-1, -1, 1, -1, 1)

        """
        if isinstance(solution, dict):
            solution = tuple(v for _, v in sorted(solution.items()))
        if not is_solution_spin(solution, spin):
            return boolean_to_spin(solution)
        return solution

    def is_solution_valid(self, solution, spin=False):
        """is_solution_valid.

        Returns whether or not the proposed solution is the correct solution,
        ie that all the variables are equal.

        Parameters
        ----------
        solution : iterable or dict.
            solution can be the output of
            AlternatingSectorsChain.convert_solution,
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
        if isinstance(solution, dict):
            solution = self.convert_solution(solution, spin)

        return all(
            x == 1 for x in solution
        ) or all(
            x != 1 for x in solution
        )
