"""
Contains tests for the SetCover class.
"""

from qubovert import SetCover
from qubovert.utils import solve_qubo_bruteforce, solve_ising_bruteforce


U = {"a", "b", "c", "d"}
V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]
problem = SetCover(U, V)


def test_setcover_str():

    assert eval(str(problem)) == problem


# QUBO

def test_setcover_qubo_logtrick_solve():

    Q, offset = problem.to_qubo()
    e, sol = solve_qubo_bruteforce(Q, offset)
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert solution == {0, 2}
    assert e == len(solution)


def test_setcover_qubo_solve():

    Q_notlog, offset = problem.to_qubo(log_trick=False)
    e, sol = solve_qubo_bruteforce(Q_notlog, offset)
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert solution == {0, 2}
    assert e == len(solution)


def test_setcover_qubo_logtrick_numvars():

    Q, offset = problem.to_qubo()
    assert len(set(y for x in Q for y in x)) == problem.num_binary_variables()


def test_setcover_qubo_numvars():

    Q_notlog, offset = problem.to_qubo(log_trick=False)
    assert (
        len(set(y for x in Q_notlog for y in x)) ==
        problem.num_binary_variables(False)
    )


# ising

def test_setcover_ising_logtrick_solve():

    h, J, offset = problem.to_ising()
    e, sol = solve_ising_bruteforce(h, J, offset)
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert solution == {0, 2}
    assert e == len(solution)


def test_setcover_ising_solve():

    h, J, offset = problem.to_ising(log_trick=False)
    e, sol = solve_ising_bruteforce(h, J, offset)
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert solution == {0, 2}
    assert e == len(solution)


def test_setcover_ising_logtrick_numvars():

    h, J, offset = problem.to_ising()
    assert (
        len(set(y for x in J for y in x).union(set(h.keys()))) ==
        problem.num_binary_variables()
    )


def test_setcover_ising_numvars():

    h, J, offset = problem.to_ising(log_trick=False)
    assert (
        len(set(y for x in J for y in x).union(set(h.keys()))) ==
        problem.num_binary_variables(False)
    )
