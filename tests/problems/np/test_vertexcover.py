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
Contains tests for the VertexCover class.
"""

from qubovert.problems import VertexCover
from qubovert.utils import (
    solve_qubo_bruteforce, solve_quso_bruteforce,
    solve_pubo_bruteforce, solve_puso_bruteforce
)
from numpy import allclose


edges = {("a", "b"), ("a", "c"), ("c", "d"), ("a", "d"), ("c", "e")}
problem = VertexCover(edges)


def test_vertexcover_str():

    assert eval(str(problem)) == problem


def test_properties():

    assert problem.E == edges
    problem.V


def test_vertexcover_bruteforce():

    assert problem.solve_bruteforce() == {"a", "c"}


# QUBO

def test_vertexcover_qubo_solve():

    e, sol = solve_qubo_bruteforce(problem.to_qubo())
    solution = problem.convert_solution(sol)

    assert solution == {"a", "c"}
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, 2)


def test_vertexcover_qubo_numvars():

    Q = problem.to_qubo()
    assert (
        len(set(y for x in Q for y in x)) ==
        problem.num_binary_variables ==
        Q.num_binary_variables
    )


# quso

def test_vertexcover_quso_solve():

    e, sol = solve_quso_bruteforce(problem.to_quso())
    solution = problem.convert_solution(sol)

    assert solution == {"a", "c"}
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, 2)


def test_vertexcover_quso_numvars():

    L = problem.to_quso()
    assert L.num_binary_variables == problem.num_binary_variables


# PUBO

def test_vertexcover_pubo_solve():

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    solution = problem.convert_solution(sol)

    assert solution == {"a", "c"}
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, 2)


# puso

def test_vertexcover_puso_solve():

    e, sol = solve_puso_bruteforce(problem.to_puso())
    solution = problem.convert_solution(sol)

    assert solution == {"a", "c"}
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, 2)
