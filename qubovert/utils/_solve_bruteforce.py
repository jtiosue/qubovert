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

"""_solve_bruteforce.py.

This file contains bruteforce solvers for QUBO/PUBO and QUSO/PUSO.

"""

import itertools
from . import pubo_value, puso_value, qubo_value, quso_value

__all__ = (
    'solve_pubo_bruteforce', 'solve_qubo_bruteforce',
    'solve_puso_bruteforce', 'solve_quso_bruteforce'
)


def _solve_bruteforce(D, all_solutions, valid, spin, value):
    """_solve_bruteforce.

    Helper function for solve_pubo_bruteforce, solve_puso_bruteforce,
    solve_qubo_bruteforce, and solve_quso_bruteforce.

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
    valid : function.
        ``valid`` takes in a bitstring or spinstring and outputs a boolean
        indicating whether that bitstring or spinstring is a valid solutions.
    spin : bool.
        Whether we're bruteforce solving a spin model or boolean model.
    value : function.
        One of ``qubo_value``, ``quso_value``, ``pubo_value``, or
        ``puso_value``.

    Returns
    -------
    res : tuple (objective, solution).

        if all_solutions is False:
            objective : float.
                The best value of the problem.
            solution : dict.
                Maps the binary variable label to its solution value,
                {0, 1} if not spin else {-1, 1}.

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

    # if D is a Matrix object or QUBO, PUBO, etc, then these are defined
    try:
        N = D.num_binary_variables
        # could do D.reverse_mapping, but that creates a copy. We just need to
        # not mutate it here, then we don't have to waste time copying.
        mapping = D._reverse_mapping
    except AttributeError:
        var = set()
        for x in D:
            var.update(set(x))
        N = len(var)

        # map qubit name to 0 through N-1
        mapping = dict(enumerate(var))

    best = None, {}
    all_sols = {None: [{}]}

    for test_sol in itertools.product((1, -1) if spin else (0, 1), repeat=N):
        x = {mapping[i]: v for i, v in enumerate(test_sol)}
        if not valid(x):
            continue
        v = value(x, D)
        if all_solutions and (best[0] is None or v <= best[0]):
            best = v, x
            all_sols.setdefault(v, []).append(x)
        elif best[0] is None or v < best[0]:
            best = v, x

    if all_solutions:
        best = best[0], all_sols[best[0]]

    return best


def solve_pubo_bruteforce(P, all_solutions=False, valid=lambda x: True):
    """solve_pubo_bruteforce.

    Iterate through all the possible solutions to a PUBO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    P : dict, qubovert.utils.PUBOMatrix, or qubovert.PUBO object.
        Maps boolean variables labels to the P value.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the PUBO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    valid : function (optional, defaults to ``lambda x: True``).
        ``valid`` takes in a bitstring and outputs a boolean
        indicating whether that bitstring is a valid solutions.

    Returns
    -------
    res : tuple (objective, solution).

        if all_solutions is False:
            objective : float.
                The best value of the PUBO.
            solution : dict.
                Maps the boolean variable label to its solution value, {0, 1}.

        if all_solutions is True:
            objective : float.
                The best value of the PUBO.
            solution : list of dicts.
                Each dictionary maps the label to the value of each boolean
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
    return _solve_bruteforce(P, all_solutions, valid, False, pubo_value)


def solve_qubo_bruteforce(Q, all_solutions=False, valid=lambda x: True):
    """solve_qubo_bruteforce.

    Iterate through all the possible solutions to a QUBO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    Q : dict, qubovert.utils.QUBOMatrix, or qubovert.QUBO object.
        Maps boolean variables labels to the Q value.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the QUBO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    valid : function (optional, defaults to ``lambda x: True``).
        ``valid`` takes in a bitstring and outputs a boolean
        indicating whether that bitstring is a valid solutions.

    Returns
    -------
    res : tuple (objective, solution).

        if all_solutions is False:
            objective : float.
                The best value of the QUBO.
            solution : dict.
                Maps the boolean variable label to its solution value, {0, 1}.

        if all_solutions is True:
            objective : float.
                The best value of the QUBO. Equal to
                ``sum(Q[(i, j)] * solution[x][i] * solution[x][j]) + offset``
                where `solution[x]` is one of the solutions to the QUBO.
            solution : list of dicts.
                Each dictionary maps the label to the value of each boolean
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
    # we could just do the same as we did in solve_pubo_bruteforce, but the
    # qubo_value function is much faster than the pubo_value function, so this
    # will be much faster!
    return _solve_bruteforce(Q, all_solutions, valid, False, qubo_value)


def solve_puso_bruteforce(H, all_solutions=False, valid=lambda x: True):
    """solve_puso_bruteforce.

    Iterate through all the possible solutions to an PUSO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    H : dict, qubovert.utils.PUSOMatrix, or qubovert.PUSO object.
        Maps spin labels to values.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the QUSO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    valid : function (optional, defaults to ``lambda x: True``).
        ``valid`` takes in a spinstring and outputs a boolean
        indicating whether that spinstring is a valid solutions.

    Returns
    -------
    res : tuple (objective, solution).

        if ``all_solutions`` is False:
            objective : float.
                The best value of the PUSO.
            solution : dict.
                Maps the spin variable label to its solution value, {-1, 1}.

        if ``all_solutions`` is True:
            objective : float.
                The best value of the PUSO.
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

    >>> obj_val, solution = solve_puso_bruteforce(H)
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
    return _solve_bruteforce(H, all_solutions, valid, True, puso_value)


def solve_quso_bruteforce(L, all_solutions=False, valid=lambda x: True):
    """solve_quso_bruteforce.

    Iterate through all the possible solutions to an QUSO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    L : dict, qubovert.utils.QUSOMatrix, or qubovert.QUSO object.
        Maps spin labels to values.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the QUSO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    valid : function (optional, defaults to ``lambda x: True``).
        ``valid`` takes in a spinstring and outputs a boolean
        indicating whether that spinstring is a valid solutions.

    Returns
    -------
    res : tuple (objective, solution).

        if ``all_solutions`` is False:
            objective : float.
                The best value of the QUSO.
            solution : dict.
                Maps the spin variable label to its solution value, {-1, 1}.

        if ``all_solutions`` is True:
            objective : float.
                The best value of the PUSO.
            solution : list of dicts.
                Each dictionary maps the label to the value of each variable.
                Ie each ``s`` in ``solution`` is a solution that gives the best
                objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = z_0z_1 + z_1z_2 - z_1 - 2z_2`,
    we have

    >>> L = {(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2}

    Then to solve this, run

    >>> obj_val, solution = solve_quso_bruteforce(L)
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
    # we could just do the same as we did in solve_puso_bruteforce, but the
    # quso_value function is much faster than the puso_value function, so this
    # will be much faster!
    return _solve_bruteforce(L, all_solutions, valid, True, quso_value)
