"""
This file contains bruteforce solvers for QUBO and Ising.
"""


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
    for the solution. In this case, x = [0, 0, 1], indicating that x_0 is 0,
    x_1 is 0, x_2 is 1.

    Q: dictionary mapping binary variables indices to the Q value. Note that
        binary variable indices must be integer labeled starting from 0.
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
            solution is a list, the value of each binary variable.
        if all_solutions is True:
            objective is a float equal to
                sum(Q[(i, j)] * solution[x][i] * solution[x][j]) + offset
                where solution[x] is one of the solutions to the QUBO.
            solution is a list of lists, where each list contains the value of
                each binary variable. Ie each s in solution is a solution that
                gives the best objective function value.
    """
    if not Q: return offset, []

    N = max(y for x in Q for y in x) + 1
    obj = lambda x: sum(
        Q[(i, j)]*int(x[i])*int(x[j]) for i, j in Q
    )

    best = None
    all_sols = {}

    for n in range(1 << N):
        x = ("{0:0%db}" % N).format(n)
        v = obj(x) + offset
        if all_solutions and (best is None or v <= best[0]):
            x_list = [int(i) for i in x]
            best = v, x_list
            all_sols.setdefault(v, []).append(x_list)
        elif best is None or v < best[0]:
            x_list = [int(i) for i in x]
            best = v, x_list

    if not all_solutions: return best
    else: return best[0], all_sols[best[0]]


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

    h: dictionary mapping spins indices to the field value. Note that
        spin variable indices must be integer labeled starting from 0.
    J: dictionary mapping tuples of spin indices to the coupling value. Note
        that spin variable indices must be integer labeled starting from 0.
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
    if h and J: N = max(max(y for x in J for y in x), max(h.keys())) + 1
    elif h: N = max(h.keys()) + 1
    elif J: N = max(y for x in J for y in x) + 1
    else: return offset, []

    to_spin = lambda x: 2 * int(x) - 1

    obj = lambda x: sum(
        J[(i, j)]*to_spin(x[i])*to_spin(x[j]) for i, j in J
    ) + sum(
        h[i]*to_spin(x[i]) for i in h
    )

    best = None
    all_sols = {}

    for n in range(1 << N):
        x = ("{0:0%db}" % N).format(n)
        v = obj(x) + offset
        if all_solutions and (best is None or v <= best[0]):
            x_list = [to_spin(i) for i in x]
            best = v, x_list
            all_sols.setdefault(v, []).append(x_list)
        elif best is None or v < best[0]:
            x_list = [to_spin(i) for i in x]
            best = v, x_list

    if not all_solutions: return best
    else: return best[0], all_sols[best[0]]
