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
Contains the SetCover class. See `help(qubovert.SetCover)`.
"""

from numpy import log2
from qubovert.utils import Problem, QUBOMatrix


class SetCover(Problem):

    """
    Class to manage converting Set Cover to and from its QUBO and
    Ising formluations. Based on the paper hereforth designated [Lucas]:
    [Andrew Lucas. Ising formulations of many np problems. Frontiers in
    Physics, 2:5, 2014.]

    The goal of the SetCover problem is to find the smallest number of subsets
    of U in V such that union over the elements equals U.

    Example usage:

        from qubovert import SetCover
        from any_module import qubo_solver
        # or you can use my bruteforce solver...
        # from qubovert.utils import solve_qubo_bruteforce as qubo_solver

        U = {"a", "b", "c", "d"}
        V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]

        problem = SetCover(U, V)
        Q, offset = problem.to_qubo()

        obj, sol = qubo_solver(Q)
        obj += offset

        solution = problem.convert_solution(sol)

        # will print {0, 2}
        print(solution)
        # will print True, since V[0] + V[2] covers all of U
        print(problem.is_solution_valid(solution))
        # will print True
        print(obj == len(solution))
    """

    def __init__(self, U, V):
        """
        The goal of the SetCover problem is to find the smallest number of
        elements in V such that union over the elements equals U. All naming
        conventions follow the names in the paper [Lucas].

        U: set, the set of all elements to cover.
        V: list or tuple of sets, where each set is one of the subsets.
        """
        self._U = U.copy()
        self._V = type(V)(x.copy() for x in V)
        self._N, self._n = len(self.V), len(self.U)
        self._M = max(
            sum(int(alpha in v) for v in self.V)
            for alpha in self.U
        )
        self._log_M = int(log2(self._M))+1

        # map each alpha in U to a unique integer index. used for the QUBO and
        # ising conversions.
        self._alpha_to_index = {alpha: i for i, alpha in enumerate(self._U)}

    @property
    def U(self):
        """
        A copy of the U set. Updating the copy will not update the
        instance U.
        """
        return self._U.copy()

    @property
    def V(self):
        """
        A copy of the V list/tuple. Updating the copy will not update the
        instance V.
        """
        return type(self._V)(x.copy() for x in self._V)

    def _filtered_range(self, alpha, start=0):
        """
        Find each set in self.V[start:] that contains alpha.

        alpha: element in self.U.
        start: optional int < len(self.V), defaults to 0.

        returns: a filter object,
        """
        return filter(lambda k: alpha in self._V[k], range(start, self._N))

    def _x(self, alpha, m, log_trick):
        """

        alpha: element in self.U.
        m: index of the ancilla binary variables x_{alpha, m} (see the Set
            Cover section in [Lucas]).
        log_trick: bool, whether or not the log trick is being used.

        returns: int, index of the ancilla variable.
        """
        mm = m if log_trick else m-1
        return self._N + self._alpha_to_index[alpha] + self._n*mm

    def to_qubo(self, A=2, B=1, log_trick=True):
        """
        Create and return the set cover problem in QUBO form following section
        5.1 of [Lucas]. The Q matrix for the QUBO
        will be returned as an uppertriangular dictionary. Thus, the problem
        becomes minimizing sum_{i <= j} x[i] x[j] Q[(i, j)]. A and B are
        parameters to enforce constraints.

        A: positive float, defaults to 2. See section 5.1 of [Lucas].
        B: positive float that is less than A, defaults to 1. See section 5.1
            of [Lucas].
        log_trick: boolean, indicates whether or not to use the log trick
            discussed in [Lucas]. Defaults to True.

        returns the tuple (Q, offset).
            Q is the upper triangular QUBO matrix, a QUBOMatrix object.
                For most practical purposes, you can use QUBOMatrix in the
                same way as an ordinary dictionary. For more information,
                see help(qubovert.utils.QUBOMatrix).
            offset is a float. It is the sum of the terms in the formulation in
                the cited paper that don't involve any variables. It is
                formatted such that if all the constraints are satisfied, then
                sum_{i <= j} x[i] x[j] Q[(i, j)] + offset will be equal to the
                total number of sets in the cover.
        """
        # all naming conventions follow the paper listed in the docstring

        Q = QUBOMatrix()

        offset = self._n * A  # comes from the first constraint

        # encode H_B (equation 46)
        for i in range(self._N):
            Q[(i, i)] += B

        # encode H_A

        for alpha in self._U:

            if not log_trick:  # (Equation 45)

                # first constraint
                for m in range(1, self._M+1):
                    i = self._x(alpha, m, log_trick)
                    Q[(i, i)] -= A
                    for mp in range(m+1, self._M+1):
                        ip = self._x(alpha, mp, log_trick)
                        Q[(i, ip)] += 2*A

                # second constraint
                for m in range(1, self._M+1):
                    i = self._x(alpha, m, log_trick)
                    Q[(i, i)] += A*m*m
                    for mp in range(m+1, self._M+1):
                        ip = self._x(alpha, mp, log_trick)
                        Q[(i, ip)] += 2*A*m*mp

                    for j in self._filtered_range(alpha):
                        Q[(j, i)] -= 2*A*m

            else:  # using the log_trick

                # first constraint
                for m in range(self._log_M+1):
                    i = self._x(alpha, m, log_trick)
                    Q[(i, i)] -= A
                    for mp in range(m+1, self._log_M+1):
                        ip = self._x(alpha, mp, log_trick)
                        Q[(i, ip)] += A

                # second constraint
                for m in range(self._log_M+1):
                    i = self._x(alpha, m, log_trick)
                    Q[(i, i)] += A*pow(2, 2*m)
                    for mp in range(m+1, self._log_M+1):
                        ip = self._x(alpha, mp, log_trick)
                        Q[(i, ip)] += 2*A*pow(2, m+mp)
                    for j in self._filtered_range(alpha):
                        Q[(j, i)] -= 2*A*pow(2, m)

            # for both using and not using the log trick
            for i in self._filtered_range(alpha):
                Q[(i, i)] += A
                for j in self._filtered_range(alpha, i+1):
                    Q[(i, j)] += 2*A

        return Q, offset

    def convert_solution(self, solution):
        """
        Convert the solution to the QUBO or Ising to the solution to the Set
        Cover problem.

        solution is the QUBO or Ising solution output. The QUBO solution output
            is either a list where indices specify the label of the binary
            variable and the element specifies whether it's 0 or 1, or it can
            be a dictionary that maps the label of the binary variable to
            whether it is a 0 or 1. The Ising solution output is the same, but
            with -1 corresponding to the QUBO 0, and 1 corresponding to the
            QUBO 1.

        returns a set of which sets are included in the set cover. So if this
        function returns {0, 2, 3}, then the set cover is the sets V[0], V[2],
        and V[3].
        """
        return set(i for i in range(self._N) if solution[i] == 1)

    def is_solution_valid(self, solution):
        """
        Returns whether or not the proposed solution covers all the elements in
        U.

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
        if not isinstance(solution, set):
            solution = self.convert_solution(solution)

        covered = set(x for i in solution for x in self._V[i])
        return covered == self._U

    def num_binary_variables(self, log_trick=True):
        """
        Find the number of binary variables that the QUBO and Ising use.

        log_trick: boolean, indicates whether to use the log trick mentioned
            in [Lucas]. Defaults to True.

        returns an integer, the number of variables in the QUBO/Ising
            formulation.
        """
        if log_trick:
            return self._N + self._n*(self._log_M+1)
        return self._N + self._n*self._M
