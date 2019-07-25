"""
Contains tests for the VertexCover class.
"""

from qubovert import VertexCover
from qubovert.utils import solve_qubo_bruteforce, solve_ising_bruteforce


edges = {("a", "b"), ("a", "c"), ("c", "d"), ("a", "d"), ("c", "e")}
problem = VertexCover(edges)


def test_vertexcover_str():

    assert eval(str(problem)) == problem


# QUBO

def test_vertexcover_qubo_solve():

    Q, offset = problem.to_qubo()
    e, sol = solve_qubo_bruteforce(Q, offset)
    solution = problem.convert_solution(sol)

    assert solution == {"a", "c"}
    assert problem.is_solution_valid(solution)
    assert e == 2


def test_vertexcover_qubo_numvars():

    Q, offset = problem.to_qubo()
    assert (
        len(set(y for x in Q for y in x)) ==
        problem.num_binary_variables()
    )


# ising

def test_vertexcover_ising_solve():

    h, J, offset = problem.to_ising()
    e, sol = solve_ising_bruteforce(h, J, offset)
    solution = problem.convert_solution(sol)

    assert solution == {"a", "c"}
    assert problem.is_solution_valid(solution)
    assert e == 2


def test_vertexcover_ising_numvars():

    h, J, _ = problem.to_ising()
    assert (
        len(set(y for x in J for y in x).union(set(h.keys()))) ==
        problem.num_binary_variables()
    )
