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

    This class inherits some methods and attributes from the
    ``qubovert.utils.Problem`` class

    Example
    -------
    >>> from qubovert import SetCover
    >>> from any_module import qubo_solver
    >>> # or you can use my bruteforce solver...
    >>> # from qubovert.utils import solve_qubo_bruteforce as qubo_solver
    >>> U = {"a", "b", "c", "d"}
    >>> V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]
    >>> problem = SetCover(U, V)
    >>> Q, offset = problem.to_qubo()
    >>> obj, sol = qubo_solver(Q)
    >>> obj += offset
    >>> solution = problem.convert_solution(sol)

    >>> print(solution)
    {0, 2}
    >>> print(problem.is_solution_valid(solution))
    True  # since V[0] + V[2] covers all of U
    >>> print(obj == len(solution))
    True
    """

    def __init__(self, U, V, log_trick=True, M=None):
        """
        The goal of the SetCover problem is to find the smallest number of
        elements in V such that union over the elements equals U. All naming
        conventions follow the names in the paper [Lucas].

        Parameters
        ----------
        U : set.
            The set of all elements to cover.
        V : iterable of sets.
            Each set is one of the subsets.
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

        if M is not None:
            self._M = M
        else:
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

        Returns
        -------
        U : set.
            A copy of the set of all elements.
        """
        return self._U.copy()

    @property
    def V(self):
        """
        A copy of the V list/tuple. Updating the copy will not update the
        instance V.

        Returns
        -------
        V : iterable of sets.
            A copy of the subsets.
        """
        return type(self._V)(x.copy() for x in self._V)

    @property
    def M(self):
        """
        A copy of the value for the M variable. See [Lucas].

        Returns
        -------
        M : int.
        """
        return self._M

    @property
    def log_trick(self):
        """
        A copy of the value for log_trick, indicating whether or not to use
        the log trick. See [Lucas].

        Returns
        -------
        log_trick : bool.
        """
        return self._log_trick

    @property
    def num_binary_variables(self):
        """
        The number of binary variables that the QUBO and Ising use.

        Returns
        -------
        num :  int.
            The number of variables in the QUBO/Ising formulation.
        """
        if self._log_trick:
            return self._N + self._n*(self._log_M+1)
        return self._N + self._n*self._M

    def _filtered_range(self, alpha, start=0):
        """
        Find each set in self.V[start:] that contains alpha.

        Parameters
        ----------
        alpha : element in self.U.
        start : int (optional, defaults to 0)
            Must be less than len(self.V). Indicates at which subset to start
            looking at,

        Returns
        -------
        f : filter object.
            The indices of V corresponding to subsets that contain alpha.
        """
        return filter(lambda k: alpha in self._V[k], range(start, self._N))

    def _x(self, alpha, m):
        """
        Return a unique index for each of the ancilla variables.

        Parameters
        ----------
        alpha : element in self.U.
        m : int.
            Index of the ancilla binary variables x_{alpha, m} (see the Set
            Cover section in [Lucas]).

        Returns
        -------
        i : int.
            Unique index of the ancilla variable.
        """
        mm = m if self._log_trick else m-1
        return self._N + self._alpha_to_index[alpha] + self._n*mm

    def to_qubo(self, A=2, B=1):
        """
        Create and return the set cover problem in QUBO form following section
        5.1 of [Lucas]. The Q matrix for the QUBO
        will be returned as an uppertriangular dictionary. Thus, the problem
        becomes minimizing sum_{i <= j} x[i] x[j] Q[(i, j)]. A and B are
        parameters to enforce constraints.

        Parameters
        ----------
        A: positive float (optional, defaults to 2).
            A enforces the constraints. See section 5.1 of [Lucas].
        B: positive float that is less than A (optional, defaults to 1).
            See section 5.1 of [Lucas].

        Returns
        -------
        res : tuple (Q, offset).
            Q : qubovert.utils.QUBOMatrix object.
                The upper triangular QUBO matrix, a QUBOMatrix object.
                For most practical purposes, you can use QUBOMatrix in the
                same way as an ordinary dictionary. For more information,
                see ``help(qubovert.utils.QUBOMatrix)``.
            offset : float.
                The sum of the terms in the formulation that don't involve any
                variables. It is formatted such that if all the constraints are
                satisfied, then sum_{i <= j} x[i] x[j] Q[(i, j)] + offset will
                be equal to the total number of sets in the cover.
        """
        # all naming conventions follow the paper listed in the docstring

        Q = QUBOMatrix()

        offset = self._n * A  # comes from the first constraint

        # encode H_B (equation 46)
        for i in range(self._N):
            Q[(i, i)] += B

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

                # first constraint
                for m in range(self._log_M+1):
                    i = self._x(alpha, m)
                    Q[(i, i)] -= A
                    for mp in range(m+1, self._log_M+1):
                        ip = self._x(alpha, mp)
                        Q[(i, ip)] += A

                # second constraint
                for m in range(self._log_M+1):
                    i = self._x(alpha, m)
                    Q[(i, i)] += A*pow(2, 2*m)
                    for mp in range(m+1, self._log_M+1):
                        ip = self._x(alpha, mp)
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

        Parameters
        ----------
        solution : iterable or dict.
            The QUBO or Ising solution output. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or -1 or 1 for Ising), or it can be a dictionary that maps the
            label of the variable to is value.

        Returns
        -------
        res : set.
            A set of which sets are included in the set cover. So if this
            function returns {0, 2, 3}, then the set cover is the sets
            V[0], V[2], and V[3].
        """
        return set(i for i in range(self._N) if solution[i] == 1)

    def is_solution_valid(self, solution):
        """
        Returns whether or not the proposed solution covers all the elements in
        U.

        Parameters
        ----------
        solution : iterable or dict.
            solution can be the output of NumberPartitioning.convert_solution,
            or the  QUBO or Ising solver output. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or -1 or 1 for Ising), or it can be a dictionary that maps the
            label of the variable to is value.

        Returns
        -------
        valid : boolean.
            True if the proposed solution is valid, else False.
        """
        if not isinstance(solution, set):
            solution = self.convert_solution(solution)

        covered = set(x for i in solution for x in self._V[i])
        return covered == self._U
