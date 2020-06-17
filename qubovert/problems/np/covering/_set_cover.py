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

"""_set_cover.py.

Contains the SetCover class. See ``help(qubovert.problems.SetCover)``.

"""

from numpy import allclose
from math import log2
from qubovert.utils import (
    QUBOMatrix, solve_qubo_bruteforce, is_solution_spin, spin_to_boolean
)
from qubovert.problems import Problem


__all__ = 'SetCover',


class SetCover(Problem):
    """SetCover.

    Class to manage converting (Weighted) Set Cover to and from its QUBO and
    QUSO formluations. Based on the paper "Ising formulations of many
    NP problems", hereforth designated as [Lucas].

    The goal of the SetCover problem is to find the smallest number of subsets
    of U in V such that union over the elements equals U. The goal of the
    Weghted SetCover problem is to find the smallest weight of subsets of U
    in V such that union over the elements equals U, where each element in V
    has an associated weight.

    This class inherits some methods and attributes from the
    ``qubovert.problems.Problem`` class.

    Example
    -------
    >>> from qubovert.problems import SetCover
    >>> from any_module import qubo_solver
    >>> # or you can use my bruteforce solver...
    >>> # from qubovert.utils import solve_qubo_bruteforce as qubo_solver
    >>> U = {"a", "b", "c", "d"}
    >>> V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]
    >>> problem = SetCover(U, V)
    >>> Q = problem.to_qubo()
    >>> obj, sol = qubo_solver(Q)
    >>> solution = problem.convert_solution(sol)

    >>> print(solution)
    {0, 2}
    >>> print(problem.is_solution_valid(solution))
    True  # since V[0] + V[2] covers all of U
    >>> print(obj == len(solution))
    True

    """

    def __init__(self, U, V, weights=None, log_trick=True, M=None):
        """__init__.

        The goal of the SetCover problem is to find the smallest number of
        elements in V such that the union over the elements equals U. Weighted
        Set Cover is the task of finding the smallest weight of elements in V
        such that the union over the elements equals U, where each element in V
        has an associated weight. All naming conventions follow the names in
        the paper [Lucas].

        Parameters
        ----------
        U : set.
            The set of all elements to cover.
        V : iterable of sets.
            Each set is one of the subsets.
        weights : iterable of numbers (optional, defaults to None).
            Important: weights must be normalized such that
            `max(weights) == 1`; similarly, `len(weights) == len(V)`. These are
            the weights for the weighted Set Cover problem. If `weights` is
            left to `None`, then the problem will default to the unweighted Set
            Cover problem (ie all the weights equal to 1).
        log_trick : boolean (optional, defaults to True).
            Indicates whether or not to use the log trick discussed in [Lucas].
        M : int (optional, defaults to None). We recommend not adjusting this.
            The number of duplicated elements to allow for, see [Lucas]. If M
            is None, then it will be set to the value required to ensure
            accuracy of the model. To possibly sacrifice accuracy but decrease
            the number of variables in the model, you can set M to something
            smaller.

        """
        self._log_trick = log_trick
        self._U = U.copy()
        self._V = type(V)(x.copy() for x in V)
        self._N, self._n = len(self.V), len(self.U)

        if weights is None:
            self._weights = type(V)(1 for _ in V)
        else:
            if len(weights) != len(V):
                raise ValueError("`weights` must be the same length as `V`")
            elif not allclose(max(weights), 1):
                raise ValueError("`weights` must be normalized such that "
                                 "`max(weights) == 1`")
            self._weights = type(weights)(weights)  # copy weights

        if M is not None:
            self._M = M
        else:
            self._M = max(
                sum(int(alpha in v) for v in self.V)
                for alpha in self.U
            )
        self._log_M = int(log2(self._M))+1

        # map each alpha in U to a unique integer index. used for the QUBO and
        # quso conversions.
        self._alpha_to_index = {alpha: i for i, alpha in enumerate(self._U)}

    @property
    def U(self):
        """U.

        A copy of the U set. Updating the copy will not update the
        instance U.

        Return
        ------
        U : set.
            A copy of the set of all elements.

        """
        return self._U.copy()

    @property
    def V(self):
        """V.

        A copy of the V list/tuple. Updating the copy will not update the
        instance V.

        Return
        ------
        V : iterable of sets.
            A copy of the subsets.

        """
        return type(self._V)(x.copy() for x in self._V)

    @property
    def weights(self):
        """weights.

        A copy of the weights list/tuple. Updating the copy will not update the
        instance weights.

        Return
        ------
        weights : iterable of numbers, where ``max(weights) == 1``.

        """
        return type(self._weights)(self._weights)

    @property
    def M(self):
        """M.

        A copy of the value for the M variable. See [Lucas].

        Returns
        -------
        M : int.

        """
        return self._M

    @property
    def log_trick(self):
        """log_trick.

        A copy of the value for log_trick, indicating whether or not to use
        the log trick. See [Lucas].

        Return
        ------
        log_trick : bool.

        """
        return self._log_trick

    @property
    def num_binary_variables(self):
        """num_binary_variables.

        The number of binary variables that the QUBO and QUSO use.

        Return
        ------
        num :  int.
            The number of variables in the QUBO/QUSO formulation.

        """
        if self._log_trick:
            return self._N + self._n*(self._log_M+1)
        return self._N + self._n*self._M

    def is_coverable(self):
        """is_coverable.

        Returns whether or not there exists a valid solution. Ie if there
        exists a combination of subsets in V that cover U.

        Return
        ------
        coverable : bool.
            True if it is possible to construct a valid solution from V and U,
            False otherwise.

        Examples
        --------
        >>> U = {0, 1, 2}
        >>> V = [{0}, {0, 1}]
        >>> SetCover(U, V).is_coverable()
        False

        >>> U = {0, 1, 2}
        >>> V = [{0, 2}, {0, 1}]
        >>> SetCover(U, V).is_coverable()
        True

        """
        # assuming every subset is chosen, does this cover U?
        return self.is_solution_valid([1]*self._N)

    def _filtered_range(self, alpha, start=0):
        """_filtered_range.

        Find each set in self.V[start:] that contains alpha.

        Parameters
        ----------
        alpha : element in self.U.
        start : int (optional, defaults to 0)
            Must be less than ``len(self.V)``. Indicates at which subset to
            start looking at.

        Return
        ------
        f : filter object.
            The indices of V corresponding to subsets that contain alpha.

        """
        return filter(lambda k: alpha in self._V[k], range(start, self._N))

    def _x(self, alpha, m):
        r"""_x.

        Return a unique index for each of the ancilla variables.

        Parameters
        ----------
        alpha : element in self.U.
        m : int.
            Index of the ancilla binary variables :math:`x_{\alpha, m}` (see
            the Set Cover section in [Lucas]).

        Return
        ------
        i : int.
            Unique index of the ancilla variable.

        """
        mm = m if self._log_trick else m-1
        return self._N + self._alpha_to_index[alpha] + self._n*mm

    def to_qubo(self, A=2, B=1):
        r"""to_qubo.

        Create and return the set cover problem in QUBO form following section
        5.1 of [Lucas]. The Q matrix for the QUBO
        will be returned as an uppertriangular dictionary. Thus, the problem
        becomes minimizing :math:`\sum_{i \leq j} x_i x_j Q_{ij}`. A and B are
        parameters to enforce constraints.

        If all the constraints are satisfied, then the objective function
        will be equal to the total number of sets in the cover (or for
        the weighted Set Cover problem, it will equal the total weight
        of included sets in the cover).

        Parameters
        ----------
        A: positive float (optional, defaults to 2).
            A enforces the constraints. See section 5.1 of [Lucas].
        B: positive float that is less than A (optional, defaults to 1).
            See section 5.1 of [Lucas].

        Return
        ------
        Q : qubovert.utils.QUBOMatrix object.
            The upper triangular QUBO matrix, a QUBOMatrix object.
            For most practical purposes, you can use QUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUBOMatrix)``.

        """
        # all naming conventions follow the paper listed in the docstring

        Q = QUBOMatrix()

        Q += self._n * A  # constant comes from the constraints

        # encode H_B (equation 46)
        for i in range(self._N):
            Q[(i,)] += self._weights[i] * B

        # encode H_A

        for alpha in self._U:

            if not self._log_trick:  # (Equation 45)

                # first constraint
                for m in range(1, self._M+1):
                    i = self._x(alpha, m)
                    Q[(i, i)] -= A
                    for mp in range(m+1, self._M+1):
                        ip = self._x(alpha, mp)
                        Q[(i, ip)] += 2*A

                # second constraint
                for m in range(1, self._M+1):
                    i = self._x(alpha, m)
                    Q[(i, i)] += A*m*m
                    for mp in range(m+1, self._M+1):
                        ip = self._x(alpha, mp)
                        Q[(i, ip)] += 2*A*m*mp

                    for j in self._filtered_range(alpha):
                        Q[(j, i)] -= 2*A*m

            else:  # using the log_trick

                # no first constraint now, but modify the second constraint.
                for m in range(self._log_M+1):
                    i = self._x(alpha, m)
                    Q[(i, i)] += A * (pow(2, 2*m) + 2 * pow(2, m))
                    for mp in range(m+1, self._log_M+1):
                        ip = self._x(alpha, mp)
                        Q[(i, ip)] += 2*A*pow(2, m+mp)
                    for j in self._filtered_range(alpha):
                        Q[(j, i)] -= 2*A*pow(2, m)

            for i in self._filtered_range(alpha):
                Q[(i,)] += A if not self._log_trick else -A
                for j in self._filtered_range(alpha, i+1):
                    Q[(i, j)] += 2 * A

        return Q

    def convert_solution(self, solution, spin=False):
        """convert_solution.

        Convert the solution to the QUBO or QUSO to the solution to the Set
        Cover problem.

        Parameters
        ----------
        solution : iterable or dict.
            The QUBO or QUSO solution output. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or -1 or 1 for QUSO), or it can be a dictionary that maps the
            label of the variable to is value.
        spin : bool (optional, defaults to False).
            `spin` indicates whether ``solution`` is the solution to the
            boolean {0, 1} formulation of the problem or the spin {1, -1}
            formulation of the problem. This parameter usually does not matter,
            and it will be ignored if possible. The only time it is used is if
            ``solution`` contains all 1's. In this case, it is unclear whether
            ``solution`` came from a spin or boolean formulation of the
            problem, and we will figure it out based on the ``spin`` parameter.

        Returns
        -------
        res : set.
            A set of which sets are included in the set cover. So if this
            function returns ``{0, 2, 3}``, then the set cover is the sets
            ``V[0]``, ``V[2]``, and ``V[3]``.

        """
        if is_solution_spin(solution, spin):
            solution = spin_to_boolean(solution)
        return set(i for i in range(self._N) if solution[i])

    def is_solution_valid(self, solution, spin=False):
        """is_solution_valid.

        Returns whether or not the proposed solution covers all the elements in
        U.

        Parameters
        ----------
        solution : iterable or dict.
            solution can be the output of SetCover.convert_solution,
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
        if not isinstance(solution, set):
            solution = self.convert_solution(solution, spin)

        covered = set(x for i in solution for x in self._V[i])
        return covered == self._U

    def solve_bruteforce(self, all_solutions=False):
        """solve_bruteforce.

        Solves the Set Cover problem exactly with a brute force method. THIS
        SHOULD NOT BE USED FOR LARGE PROBLEMS! The advantage over this method
        as opposed to using a brute force QUBO solver is that the QUBO
        formulation has many slack variables.

        Parameters
        ----------
        all_solutions : boolean (optional, defaults to False).
            If ``all_solutions`` is set to True, all the best solutions to the
            problem will be returned rather than just one of the best. If the
            problem is very big, then it is best if ``all_solutions == False``,
            otherwise this function will use a lot of memory.

        Returns
        -------
        res : set or list of sets.
            A set of which sets are included in the set cover. So if this
            function returns ``{0, 2, 3}``, then the set cover is the sets
            ``V[0]``, ``V[2]``, and ``V[3]``. If ``all_solutions == True``,
            then ``res`` will be a list of sets, where each element of the list
            is one of the optimal solutions.

        Examples
        --------
        >>> from qubovert import SetCover
        >>> U = {"a", "b", "c", "d"}
        >>> V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]
        >>> problem = SetCover(U, V)
        >>> print(problem.solve_bruteforce())
        {0, 2}

        """
        # minimize the objective function
        Q = {(i,): self._weights[i] for i in range(self._N)}
        # subject to the constraint
        valid = self.is_solution_valid

        obj, sol = solve_qubo_bruteforce(Q, all_solutions, valid)

        if obj is None:
            raise ValueError("Problem is not solvable. See "
                             "``SetCover.is_coverable()``")

        if all_solutions:
            return [self.convert_solution(s) for s in sol]
        return self.convert_solution(sol)

#        best = None
#        all_sols = {}
#        for x in range(1 << self._N):
#            sol = decimal_to_boolean(x, self._N)
#            cover = self.convert_solution(sol)
#            if self.is_solution_valid(cover):
#                if not all_solutions and (best is None or
#                                          len(cover) < len(best)):
#                    best = cover
#                elif all_solutions and (best is None or
#                                        len(cover) <= len(best)):
#                    best = cover
#                    all_sols.setdefault(len(cover), []).append(cover)
#
#        if best is None:
#            raise ValueError("Problem is not solvable. See "
#                             "``SetCover.is_coverable()``")
#        if all_solutions:
#            return all_sols[len(best)]
#        return best
