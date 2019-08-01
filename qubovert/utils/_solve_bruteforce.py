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

"""_solve_bruteforce.py.

This file contains bruteforce solvers for QUBO and Ising, as well as QUBO and
Ising objective function evaluators.

"""

from . import binary_to_spin


def qubo_value(x, Q, offset=0):
    r"""qubo_value.

    Find the value of
    :math:`\sum_{i,j} Q_{ij} x_{i} x_{j} + offset.`

    Parameters
    ----------
    x : dict or iterable.
        Maps binary variable indices to their binary values, 0 or 1. Ie
        ``x[i]`` must be the binary value of variable i.
    Q : dict or qubovert.utils.QUBOMatrix object.
        Maps tuples of binary variables indices to the Q value.
    offset : float (optional, defaults to 0).
        The part of the objective function that does not depend on the
        variables.

    Return
    -------
    value : float.
        The value of the QUBO with the given assignment `x`. Ie

        >>> sum(
                Q[(i, j)] * x[i] * x[j]
                for i in range(n) for j in range(n)
            ) + offset

    Example
    -------
    >>> Q = {(0, 0): 1, (0, 1): -1}
    >>> x = {0: 1, 1: 0}
    >>> qubo_value(x, Q)
    1

    """
    return sum(v * x[i] * x[j] for (i, j), v in Q.items()) + offset


def ising_value(z, h, J, offset=0):
    r"""ising_value.

    Find the value of
        :math:`\sum_{i,j} J_{ij} z_{i} z_{j} + \sum_{i} h_{i} z_{i} + offset`.

    Parameters
    ----------
    z: dict or iterable.
        Maps variable labels to their values, -1 or 1. Ie z[i] must be the
        value of variable i.
    J: dict or qubovert.utils.IsingCoupling object.
        Maps pairs of variables labels to the J value.
    h: dict or qubovert.utils.IsingField object.
        Maps variable names to their field value.
    offset: float (optional, defaults to 0).
        The part of the objective function that does not depend on the
        variables.

    Return
    -------
    value : float.
        The value of the Ising with the given assignment `z`. Ie

        >>> sum(
                J[(i, j)] * z[i] * z[j]
                for i in range(n) for j in range(n)
            ) + sum(
                h[i] * z[i] for i in range(n)
            ) + offset

    Example
    -------
    >>> h = {0: 1}
    >>> J = {(0, 1): -1}
    >>> z = {0: -1, 1: 1}
    >>> qubo_value(z, h, J)
    0

    """
    return sum(
        v * z[i] * z[j] for (i, j), v in J.items()
    ) + sum(
        v * z[i] for i, v in h.items()
    ) + offset


def solve_qubo_bruteforce(Q, offset=0, all_solutions=False):
    """solve_qubo_bruteforce.

    Iterate through all the possible solutions to a QUBO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    Q : dict or qubovert.utils.QUBOMatrix object.
        Maps binary variables labels to the Q value.
    offset : float (optional, defaults to 0).
        The part of the objective function that does not depend on the
        variables.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the QUBO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.

    Return
    -------
    res : tuple (objective, solution).

        if all_solutions is False:
            objective : float.
                The best value of the QUBO. Equal to
                sum(Q[(i, j)] * solution[i] * solution[j]) + offset
            solution : dict.
                Maps the binary variable label to its solution value, {0, 1}.

        if all_solutions is True:
            objective : float.
                The best value of the QUBO. Equal to
                ``sum(Q[(i, j)] * solution[x][i] * solution[x][j]) + offset``
                where `solution[x]` is one of the solutions to the QUBO.
            solution : list of dicts.
                Each dictionary maps the label to the value of each binary
                variable. Ie each ``s`` in ``solution`` is a solution that
                gives the best objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = x_0x_1 + x_1x_2 - x_1 - 2x_2`,
    the Q dictionary is

    >>> Q = {(0, 1): 1, (1, 2): 1, (1, 1): -1, (2, 2): -2}

    Then to solve this Q, run

    >>> obj_val, solution = solve_qubo_bruteforce(Q)
    >>> obj_val
    -2
    >>> solution
    {0: 0, 1: 0, 2: 1}

    ``obj_val`` will be the smallest value of ``obj``.
    ``solution`` will be a dictionary that indicates what each of :math:`x_0`,
    :math:`x_1`, and :math:`x_2` are for the solution. In this case,
    ``x = {0: 0, 1: 0, 2: 1}``, indicating that :math:`x_0` is 0, :math:`x_1`
    is 0, :math:`x_2` is 1.

    """
    if not Q:
        return offset, []

    var = set(y for x in Q for y in x)
    N = len(var)

    # map qubit name to 0 through N-1
    mapping = dict(enumerate(var))

    best = None, None
    all_sols = {}

    for n in range(1 << N):
        test_sol = ("{0:0%db}" % N).format(n)
        x = {mapping[i]: int(v) for i, v in enumerate(test_sol)}
        v = qubo_value(x, Q, offset)
        if all_solutions and (best[0] is None or v <= best[0]):
            best = v, x
            all_sols.setdefault(v, []).append(x)
        elif best[0] is None or v < best[0]:
            best = v, x

    if all_solutions:
        best = best[0], all_sols[best[0]]

    return best


def solve_ising_bruteforce(h, J, offset=0, all_solutions=False):
    """solve_ising_bruteforce.

    Iterate through all the possible solutions to an Ising formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    h : dict or qubovert.utils.IsingField object.
        Maps variable labels to their field values.
    J : dict or qubovert.utils.IsingCoupling object.
        Maps pairs of variable labels to their coupling values.
    offset : float (optional, defaults to 0).
        The part of the objective function that does not depend on the
        variables.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the Ising
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.

    Return
    -------
    res : tuple (objective, solution).

        if ``all_solutions`` is False:
            objective : float.
                The best value of the Ising. Equal to
                ``sum(J[(i, j)] * solution[i] * solution[j]) +
                sum(h[i] * solution[i]) +
                offset``.
            solution : dict.
                Maps the variable label to its solution value, {-1, 1}.

        if ``all_solutions`` is True:
            objective : float.
                The best value of the Ising. Equal to
                ``sum(J[(i, j)] * solution[x][i] * solution[x][j]) +
                sum(h[i] * solution[x][i]) +
                offset``
                where solution[x] is one of the solutions to the Ising.
            solution : list of dicts.
                Each dictionary maps the label to the value of each variable.
                Ie each ``s`` in ``solution`` is a solution that gives the best
                objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = z_0z_1 + z_1z_2 - z_1 - 2z_2`,
    we have

    >>> J = {(0, 1): 1, (1, 2): 1}
    >>> h = {1: -1, 2: -2}

    Then to solve this, run

    >>> obj_val, solution = solve_ising_bruteforce(h, J)
    >>> obj_val
    -3
    >>> solution
    {0: 1, 1: -1, 2: 1}

    ``obj_val`` will be the smallest value of ``obj``.
    ``solution`` will be a dictionary that indicates what each of
    :math:`z_0, z_1`, and :math:`z_2` are for the solution. In this case,
    ``z = {0: 1, 1: -1, 2: 1}``, indicating that :math:`z_0` is 1, :math:`z_1`
    is -1, :math:`z_2` is 1.

    """
    if not h and not J:
        return offset, []

    var = set(h.keys())
    var.update(set(y for x in J for y in x))
    N = len(var)

    # map qubit name to 0 through N-1
    mapping = dict(enumerate(var))

    best = None, None
    all_sols = {}

    for n in range(1 << N):
        test_sol = ("{0:0%db}" % N).format(n)
        z = {
            mapping[i]: binary_to_spin(int(v)) for i, v in enumerate(test_sol)
        }
        v = ising_value(z, h, J, offset)
        if all_solutions and (best[0] is None or v <= best[0]):
            best = v, z
            all_sols.setdefault(v, []).append(z)
        elif best[0] is None or v < best[0]:
            best = v, z

    if all_solutions:
        best = best[0], all_sols[best[0]]

    return best
