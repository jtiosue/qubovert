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

"""
Contains tests for the AlternatingSectorsChain class.
"""

from qubovert.problems import AlternatingSectorsChain
from qubovert.utils import (
    solve_qubo_bruteforce, solve_quso_bruteforce,
    solve_pubo_bruteforce, solve_puso_bruteforce
)
from numpy import allclose
from numpy.testing import assert_raises


problem = AlternatingSectorsChain(12)


def test_AlternatingSectorsChain_str():

    assert eval(str(problem)) == problem


def test_errors():

    with assert_raises(ValueError):
        AlternatingSectorsChain(10, min_strength=-1)

    with assert_raises(ValueError):
        AlternatingSectorsChain(0)

    with assert_raises(ValueError):
        AlternatingSectorsChain(3, chain_length=1)


def test_AlternatingSectorsChain_bruteforce():

    assert problem.solve_bruteforce() in ((-1,)*12, (1,)*12)
    assert (
        problem.solve_bruteforce(all_solutions=True) in
        ([(-1,)*12, (1,)*12], [(1,)*12, (-1,)*12])
    )


# QUBO

def test_AlternatingSectorsChain_qubo_solve():

    e, sol = solve_qubo_bruteforce(problem.to_qubo(True))
    solution = problem.convert_solution(sol)
    assert solution == problem.convert_solution(solution, True)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, -66)

    # not pbc

    e, sol = solve_qubo_bruteforce(problem.to_qubo(False))
    solution = problem.convert_solution(sol)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, -65)


def test_AlternatingSectorsChain_qubo_numvars():

    Q = problem.to_qubo()
    assert (
        len(set(y for x in Q for y in x)) ==
        problem.num_binary_variables ==
        Q.num_binary_variables
    )


# quso

def test_AlternatingSectorsChain_quso_solve():

    e, sol = solve_quso_bruteforce(problem.to_quso(True))
    solution = problem.convert_solution(sol, True)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, -66)

    # not pbc

    e, sol = solve_quso_bruteforce(problem.to_quso(False))
    solution = problem.convert_solution(sol, True)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, -65)


def test_AlternatingSectorsChain_quso_numvars():

    L = problem.to_quso()
    assert L.num_binary_variables == problem.num_binary_variables


# PUBO

def test_AlternatingSectorsChain_pubo_solve():

    e, sol = solve_pubo_bruteforce(problem.to_pubo(True))
    solution = problem.convert_solution(sol)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, -66)

    # not pbc

    e, sol = solve_pubo_bruteforce(problem.to_pubo(False))
    solution = problem.convert_solution(sol)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, -65)


# puso

def test_AlternatingSectorsChain_puso_solve():

    e, sol = solve_puso_bruteforce(problem.to_puso(True))
    solution = problem.convert_solution(sol, True)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, -66)

    # not pbc

    e, sol = solve_puso_bruteforce(problem.to_puso(False))
    solution = problem.convert_solution(sol, True)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert allclose(e, -65)
