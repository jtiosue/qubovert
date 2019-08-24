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
Ising objective function evaluators, and for PUBO and HIsings.

"""

from . import decimal_to_binary, decimal_to_spin


__all__ = (
    'pubo_value', 'qubo_value', 'hising_value', 'ising_value',
    'solve_pubo_bruteforce', 'solve_qubo_bruteforce',
    'solve_hising_bruteforce', 'solve_ising_bruteforce'
)


def pubo_value(x, P):
    r"""pubo_value.

    Find the value of
    :math:`\sum_{i,...,j} P_{i...j} x_{i} ... x_{j}`

    Parameters
    ----------
    x : dict or iterable.
        Maps binary variable indices to their binary values, 0 or 1. Ie
        ``x[i]`` must be the binary value of variable i.
    P : dict, qubovert.utils.PUBOMatrix, or qubovert.PUBO object.
        Maps tuples of binary variables indices to the P value.

    Return
    -------
    value : float.
        The value of the PUBO with the given assignment `x`. Ie

    Example
    -------
    >>> P = {(0, 0): 1, (0, 1): -1}
    >>> x = {0: 1, 1: 0}
    >>> pubo_value(x, P)
    1

    """
    return sum(v for k, v in P.items() if all(x[i] for i in k))


def qubo_value(x, Q):
    r"""qubo_value.

    Find the value of
    :math:`\sum_{i,j} Q_{ij} x_{i} x_{j}`

    Parameters
    ----------
    x : dict or iterable.
        Maps binary variable indices to their binary values, 0 or 1. Ie
        ``x[i]`` must be the binary value of variable i.
    Q : dict or qubovert.utils.QUBOMatrix object.
        Maps tuples of binary variables indices to the Q value.

    Return
    -------
    value : float.
        The value of the QUBO with the given assignment `x`. Ie

        >>> sum(
                Q[(i, j)] * x[i] * x[j]
                for i in range(n) for j in range(n)
            )

    Example
    -------
    >>> Q = {(0, 0): 1, (0, 1): -1}
    >>> x = {0: 1, 1: 0}
    >>> qubo_value(x, Q)
    1

    """
    return pubo_value(x, Q)


def hising_value(z, H):
    r"""hising_value.

    Find the value of
        :math:`\sum_{i,...,j} H_{i...j} z_{i} ... z_{j}`.

    Parameters
    ----------
    z: dict or iterable.
        Maps variable labels to their values, -1 or 1. Ie z[i] must be the
        value of variable i.
    H : dict, qubovert.utils.HIsingMatrix, or qubovert.HIsing object.
        Maps spin labels to values.

    Return
    -------
    value : float.
        The value of the HIsing with the given assignment `z`.

    Example
    -------
    >>> I = {(0, 1): -1, (0,): 1}
    >>> z = {0: -1, 1: 1}
    >>> hising_value(z, I)
    0

    """
    return sum(
        v * pow(-1, [z[i] for i in k].count(-1) % 2)
        for k, v in H.items()
    )


def ising_value(z, I):
    r"""ising_value.

    Find the value of
        :math:`\sum_{i,j} J_{ij} z_{i} z_{j} + \sum_{i} h_{i} z_{i}t`.
    The J's are encoded by keys with pairs of labels in I, and the h's are
    encoded by keys with a single label in I.

    Parameters
    ----------
    z: dict or iterable.
        Maps variable labels to their values, -1 or 1. Ie z[i] must be the
        value of variable i.
    I : dict, qubovert.utils.IsingMatrix, or qubovert.Ising object.
        Maps spin labels to values.

    Return
    -------
    value : float.
        The value of the Ising with the given assignment `z`.

    Example
    -------
    >>> I = {(0, 1): -1, (0,): 1}
    >>> z = {0: -1, 1: 1}
    >>> ising_value(z, I)
    0

    """
    return hising_value(z, I)




def _solve_bruteforce(D, all_solutions, spin=False):
    """_solve_bruteforce.

    Helper function for solve_pubo_bruteforce and solve_hising_bruteforce.

    Iterate through all the possible solutions to a BO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    D : dict.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the problem
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    spin : bool (optional, defaults to False).
        Whether we're bruteforce solving a spin model or binary model.

    Return
    -------
    res : tuple (objective, solution).

        if all_solutions is False:
            objective : float.
                The best value of the problem.
            solution : dict.
                Maps the binary variable label to its solution value,
                {0, 1} of not spin else {-1, 1}.

        if all_solutions is True:
            objective : float.
                The best value of the problem
            solution : list of dicts.
                Each dictionary maps the label to the value of each binary
                variable. Ie each ``s`` in ``solution`` is a solution that
                gives the best objective function value.

    """
    if not D:
        return 0, ({} if not all_solutions else [{}])
    elif () in D:
        offset = D.pop(())
        if not D:
            D[()] = offset
            return offset, ({} if not all_solutions else [{}])
        D[()] = offset

    var = set()
    for x in D:
        var.update(set(x))
    N = len(var)

    # map qubit name to 0 through N-1
    mapping = dict(enumerate(var))

    best = None, None
    all_sols = {}

    for n in range(1 << N):
        test_sol = decimal_to_spin(n, N) if spin else decimal_to_binary(n, N)
        x = {mapping[i]: v for i, v in enumerate(test_sol)}
        v = hising_value(x, D) if spin else pubo_value(x, D)
        if all_solutions and (best[0] is None or v <= best[0]):
            best = v, x
            all_sols.setdefault(v, []).append(x)
        elif best[0] is None or v < best[0]:
            best = v, x

    if all_solutions:
        best = best[0], all_sols[best[0]]

    return best


def solve_pubo_bruteforce(P, all_solutions=False):
    """solve_pubo_bruteforce.

    Iterate through all the possible solutions to a PUBO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    P : dict, qubovert.utils.PUBOMatrix, or qubovert.PUBO object.
        Maps binary variables labels to the P value.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the PUBO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.

    Return
    -------
    res : tuple (objective, solution).

        if all_solutions is False:
            objective : float.
                The best value of the PUBO.
            solution : dict.
                Maps the binary variable label to its solution value, {0, 1}.

        if all_solutions is True:
            objective : float.
                The best value of the PUBO.
            solution : list of dicts.
                Each dictionary maps the label to the value of each binary
                variable. Ie each ``s`` in ``solution`` is a solution that
                gives the best objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = x_0x_1 + x_1x_2 - x_1 - 2x_2`,
    the P dictionary is

    >>> P = {(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2}

    Then to solve this P, run

    >>> obj_val, solution = solve_pubo_bruteforce(Q)
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
    return _solve_bruteforce(P, all_solutions, spin=False)


def solve_qubo_bruteforce(Q, all_solutions=False):
    """solve_qubo_bruteforce.

    Iterate through all the possible solutions to a QUBO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    Q : dict, qubovert.utils.QUBOMatrix, or qubovert.QUBO object.
        Maps binary variables labels to the Q value.
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
                The best value of the QUBO.
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
    the P dictionary is

    >>> Q = {(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2}

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
    return solve_pubo_bruteforce(Q, all_solutions)


def solve_hising_bruteforce(H, all_solutions=False):
    """solve_hising_bruteforce.

    Iterate through all the possible solutions to an HIsing formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    H : dict, qubovert.utils.HIsingMatrix, or qubovert.HIsing object.
        Maps spin labels to values.
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
                The best value of the HIsing.
            solution : dict.
                Maps the variable label to its solution value, {-1, 1}.

        if ``all_solutions`` is True:
            objective : float.
                The best value of the HIsing.
            solution : list of dicts.
                Each dictionary maps the label to the value of each variable.
                Ie each ``s`` in ``solution`` is a solution that gives the best
                objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = z_0z_1 + z_1z_2 - z_1 - 2z_2`,
    we have

    >>> H = {(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2}

    Then to solve this, run

    >>> obj_val, solution = solve_hising_bruteforce(I)
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
    return _solve_bruteforce(H, all_solutions, spin=True)


def solve_ising_bruteforce(I, all_solutions=False):
    """solve_ising_bruteforce.

    Iterate through all the possible solutions to an Ising formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    I : dict, qubovert.utils.IsingMatrix, or qubovert.Ising object.
        Maps spin labels to values.
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
                The best value of the Ising.
            solution : dict.
                Maps the variable label to its solution value, {-1, 1}.

        if ``all_solutions`` is True:
            objective : float.
                The best value of the HIsing.
            solution : list of dicts.
                Each dictionary maps the label to the value of each variable.
                Ie each ``s`` in ``solution`` is a solution that gives the best
                objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = z_0z_1 + z_1z_2 - z_1 - 2z_2`,
    we have

    >>> I = {(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2}

    Then to solve this, run

    >>> obj_val, solution = solve_ising_bruteforce(I)
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
    return solve_hising_bruteforce(I, all_solutions)
