"""
This file contains bruteforce solvers for QUBO and Ising, as well as QUBO and
Ising objective function evaluators.
"""

from ._conversions import binary_to_spin


def qubo_value(x, Q, offset=0):
    """
    Find the value of
        sum_{ij} Q_{ij} x_{i} x_{j} + offset.

    x: dict or list mapping binary variable indices to their binary values,
        0 or 1. Ie x[i] must be the binary value of variable i.
    Q: dictionary mapping binary variables indices to the Q value. Q can
        also be a qubovert.utils.QUBOMatrix object. For example,
            >>> Q = {(0, 1): 1, (1, 1): -1, ...}
    offset: an optional float that defaults to 0, the part of the objective
        function that does not depend on the variables.

    returns: float, sum_{ij}Q[(i, j)] x[i] x[j] + offset
    """
    return sum(v * x[i] * x[j] for (i, j), v in Q.items()) + offset


def ising_value(z, h, J, offset=0):
    """
    Find the value of
        sum_{ij} J_{ij} z_{i} z_{j} + sum_{i} h_{i} z_{i} +  offset.

    z: dict or list mapping variable indices to their values, -1 or 1. Ie z[i]
        must be the value of variable i.
    J: dictionary mapping variables indices to the J value. J can also be a
        qubovert.utils.IsingCoupling object. For example,
            >>> J = {(0, 1): 1, (1, 2): -1, ...}
    h: dictionary mapping variables indices to the h value. h can also be a
        qubovert.utils.IsingField object. For example,
            >>> h = {0: 1, 2: -1, ...}
    offset: an optional float that defaults to 0, the part of the objective
        function that does not depend on the variables.

    returns: float,
        sum_{ij} J[(i, j)] z[i] z[j] + sum_{i} h[i] z[i] +  offset.
    """
    return sum(
        v * z[i] * z[j] for (i, j), v in J.items()
    ) + sum(
        v * z[i] for i, v in h.items()
    ) + offset


def solve_qubo_bruteforce(Q, offset=0, all_solutions=False):
    """
    Iterate through all the possible solutions to a QUBO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    For example, to find the minimum of the problem
        obj = x_0*x_1 + x_1*x_2 - x_1 - 2*x_2,
    the Q dictionary is
        >>> Q = {(0, 1): 1, (1, 2): 1, (1, 1): -1, (2, 2): -2}.
    Then to solve this Q, run
        >>> obj_val, solution = solve_qubo_bruteforce(Q)
    obj_val will be the smallest value of obj, for this example it will be -2.
    solution will be a list that indicated what each of x_0, x_1, and x_2 are
    for the solution. In this case, x = {0: 0, 1: 0, 2: 1}, indicating that
    x_0 is 0, x_1 is 0, x_2 is 1.

    Q: dictionary mapping binary variables indices to the Q value. Q can also
        be a qubovert.utils.QUBOMatrix object. For example,
            >>> Q = {(0, 1): 1, (1, 1): -1, ...}
    offset: an optional float that defaults to 0, the part of the objective
        function that does not depend on the variables.
    all_solutions: an optional boolean that defaults to False. If all_solutions
        is set to True, all the best solutions to the QUBO will be returned
        rather than just one of the best. If the problem is very big, then it
        is best if all_solutions is False, otherwise this function will use
        a lot of memory.

    returns a tuple (objective, solution).
        if all_solutions is False:
            objective is a float equal to
                sum(Q[(i, j)] * solution[i] * solution[j]) + offset
            solution is a dictionary, mapping the label to the value of each
                binary variable.
        if all_solutions is True:
            objective is a float equal to
                sum(Q[(i, j)] * solution[x][i] * solution[x][j]) + offset
                where solution[x] is one of the solutions to the QUBO.
            solution is a list of dictionaries, where each dictionary maps the
                label to the value of each binary variable. Ie each s in
                solution is a solution that gives the best objective function
                value.
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
    """
    Iterate through all the possible solutions to a Ising formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    For example, to find the minimum of the problem
        obj = z_0*z_1 + z_1*z_2 - z_1 - 2*z_2,
    the Ising formulation is
        >>> h = {1: -1, 2: -2}
        >>> J = {(0, 1): 1, (1, 2): 1}
    Then to solve this problem, run
        >>> obj_val, solution = solve_ising_bruteforce(h, J)
    obj_val will be the smallest value of obj, for this example it will be -3.
    solution will be a list that indicated what each of z_0, z_1, and z_2 are
    for the solution. In this case, z = [-1, 1, 1], indicating that z_0 is -1,
    z_1 is 1, and z_2 is 1.

    h: dictionary mapping spins indices to the field value. h can also be a
        qubovert.utils.IsingField object.
    J: dictionary mapping tuples of spin indices to the coupling value. J can
        also be a qubovert.utils.IsingCoupling object.
    offset: an optional float, the part of the objective function that does
        not depend on the variables.
    all_solutions: an optional boolean that defaults to False. If all_solutions
        is set to True, all the best solutions to the Ising will be returned
        rather than just one of the best. If the problem is very big, then it
        is best if all_solutions is False, otherwise this function will use
        a lot of memory.

    returns a tuple (objective, solution).
        if all_solutions is False:
            objective is a float equal to
                sum(J[(i, j)] * solution[i] * solution[j]) +
                sum(h[i] * solution[i]) +
                offset
            solution is a list, the value of each Ising spin (-1 or 1).
        if all_solutions is True:
            objective is a float equal to
                sum(J[(i, j)] * solution[x][i] * solution[x][j]) +
                sum(h[i] * solution[x][i]) +
                offset
                where solution[x] is one of the solutions to the Ising.
            solution is a list of lists, where each list contains the value of
                each Ising spin (-1 or 1). Ie each s in solution is a solution
                that gives the best objective function value.
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
