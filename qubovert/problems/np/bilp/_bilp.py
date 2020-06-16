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

"""_bilp.py.

Contains the BILP class. See ``help(qubovert.problems.BILP)``.

"""

from qubovert.utils import QUBOMatrix, is_solution_spin, spin_to_boolean
from qubovert.problems import Problem
import numpy as np


__all__ = 'BILP',


class BILP(Problem):
    r"""BILP.

    Class to manage converting Binary Integer Linear Programming problems to
    and from their QUBO and QUSO formluations. Based on the paper "Ising
    formulations of many NP problems", hereforth designated as [Lucas].

    The goal of the BILP problem is to find the minimum value of
    :math:`\mathbf{c} \cdot \mathbf{x}` subject to
    :math:`S \mathbf{x} = \mathbf{b}`. Here :math:`\mathbf{c}`,
    :math:`\mathbf{b}`, and :math:`S` define the problem, and
    :math:`\mathbf{x}`, is a vector of boolean variables.

    BILP inherits some methods and attributes from the Problem class.
    See ``help(qubovert.problems.Problem)``.

    Example
    -------
    >>> from qubovert.problems import BILP
    >>> from any_module import qubo_solver
    >>> # or you can use my bruteforce solver...
    >>> # from qubovert.utils import solve_qubo_bruteforce as qubo_solver
    >>>
    >>> c = [1, 2, -1]
    >>> S = [[1, 0, 0], [0, 1, -1]]
    >>> b = [0, 1]
    >>> problem = BILP(c, S, b)
    >>>
    >>> Q = problem.to_qubo()
    >>> obj, sol = qubo_solver(Q)
    >>> solution = problem.convert_solution(sol)
    >>>
    >>> print(solution)
    [0 1 0]
    >>> print(problem.is_solution_valid(solution))
    True
    >>> print(obj == np.dot(c, solution))
    True

    """

    def __init__(self, c, S, b):
        r"""__init__.

        The goal of the BILP problem is to find the minimum value of
        :math:`\mathbf{c} \cdot \mathbf{x}` subject to
        :math:`S \mathbf{x} = \mathbf{b}`. Here :math:`\mathbf{c}`,
        :math:`\mathbf{b}`, and :math:`S` define the problem, and
        :math:`\mathbf{x}`, is a vector of boolean variables.
        All naming conventions follow the names in the paper [Lucas].

        Parameters
        ----------
        c : list or numpy one-dim array of integers.
            Array representing the :math:`\mathbf{c}` vector. ``c`` should
            have length ``N``.
        S : list of lists or two-dim numpy array of integers.
            Array representing the :math:`S` matrix. ``S`` should be ``m`` by
            ``N``.
        b : list or one-dim numpy array of integers.
            Array representing the :math:`\mathbf{b}` vector. ``b`` should have
            length ``m``.

        Examples
        --------
        >>> c =
        >>> b =
        >>> S =
        >>> problem = BILP(c, S, b)

        """
        self._c, self._S, self._b = np.array(c), np.array(S), np.array(b)
        self._m, self._N = self._S.shape

        if len(self._c.shape) > 1 or len(self._b.shape) > 1:
            raise ValueError("b and c must be one dimensional")
        elif self._c.shape[0] != self._N or self._b.shape[0] != self._m:
            raise ValueError("Incorrect dimensions")

    @property
    def c(self):
        """c.

        A copy of the ``c`` vector. Updating the copy will not
        update the instance set.

        Return
        ------
        c : numpy array.

        """
        return self._c.copy()

    @property
    def S(self):
        """S.

        A copy of the ``S`` matrix. Updating the copy will not
        update the instance set.

        Return
        ------
        S : numpy array.

        """
        return self._S.copy()

    @property
    def b(self):
        """b.

        A copy of the ``b`` vector. Updating the copy will not
        update the instance set.

        Return
        ------
        b : numpy array.

        """
        return self._b.copy()

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

    def to_qubo(self, A=None, B=1):
        r"""to_qubo.

        Create and return the BILP problem in QUBO form following
        section 3 of [Lucas]. The Q matrix for the QUBO
        will be returned as an uppertriangular dictionary. Thus, the problem
        becomes minimizing :math:`\sum_{i \leq j} x_i x_j Q_{ij}`. ``A`` and
        ``B`` are parameters to enforce constraints.

        Parameters
        ----------
        A: positive float (optional, defaults to None).
            A enforces the constraints. If ``A`` is
            None, then a default value will be chosen as given in section 3 of
            [Lucas].
        B: positive float that is less than A (optional, defaults to 1).
            See section 3 of [Lucas].

        Return
        ------
        Q : qubovert.utils.QUBOMatrix object.
            The upper triangular QUBO matrix, a QUBOMatrix object.
            For most practical purposes, you can use QUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see help(qubovert.utils.QUBOMatrix).

        """
        # all naming conventions follow the paper listed in the docstring
        if A is None:
            A = B * self._N

        Q = QUBOMatrix()

        # equation 23
        for i in range(self._N):
            Q[(i,)] += B * self._c[i]

        # equation 22
        for j in range(self._m):
            Qtemp = QUBOMatrix()
            Qtemp += self._b[j]
            for i in range(self._N):
                Qtemp[(i,)] -= self._S[j][i]
            Qtemp *= A * Qtemp
            Q += Qtemp

        return Q

    def convert_solution(self, solution, spin=False):
        r"""convert_solution.

        Convert the solution to the QUBO or QUSO to the solution to the BILP
        problem.

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
        res : np.array.
            An array representing the :math:`\mathbf{x}` vector.

        """
        if is_solution_spin(solution, spin):
            solution = spin_to_boolean(solution)
        return np.array([int(bool(solution[i])) for i in range(self._N)])

    def is_solution_valid(self, solution, spin=False):
        r"""is_solution_valid.

        Returns whether or not the proposed solution satisfies the constraint
        that :math:`S\mathbf{x} = \mathbf{b}`.

        Parameters
        ----------
        solution : iterable or dict.
            solution can be the output of BILP.convert_solution,
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
        if not isinstance(solution, np.ndarray):
            solution = self.convert_solution(solution, spin)

        return np.allclose(
            self._S @ np.array([solution]).T,
            np.array([self._b]).T
        )
