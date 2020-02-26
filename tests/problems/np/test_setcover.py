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
Contains tests for the SetCover class.
"""

from qubovert.problems import SetCover
from qubovert.utils import (
    solve_qubo_bruteforce, solve_quso_bruteforce,
    solve_pubo_bruteforce, solve_puso_bruteforce
)
from numpy import allclose
from numpy.testing import assert_raises


U = {"a", "b", "c", "d"}
V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]
problem_log = SetCover(U, V)
problem = SetCover(U, V, log_trick=False)
problem_weighted = SetCover(U, V, weights=(.1, .2, 1))


def test_errors():

    with assert_raises(ValueError):
        SetCover(U, V, weights=(.1, 1))

    with assert_raises(ValueError):
        SetCover(U, V, weights=(.1, 1.1, .3))


def test_setcover_str():

    assert eval(str(problem)) == problem
    assert eval(str(problem_log)) == problem_log
    assert eval(str(problem_weighted)) == problem_weighted


def test_properties():

    problem.weights
    problem.M
    problem.log_trick


def test_coverable():

    assert problem.is_coverable()
    assert not SetCover(U, V[:-1]).is_coverable()
    with assert_raises(ValueError):
        SetCover(U, V[:-1]).solve_bruteforce()


def test_setcover_bruteforce():

    assert problem.solve_bruteforce() == {0, 2}
    assert problem.solve_bruteforce(True) == [{0, 2}]


# QUBO

def test_setcover_qubo_logtrick_solve():

    e, sol = solve_qubo_bruteforce(problem_log.to_qubo())
    solution = problem_log.convert_solution(sol)
    assert problem_log.is_solution_valid(solution)
    assert problem_log.is_solution_valid(sol)
    assert solution == {0, 2}
    assert allclose(e, len(solution))

    p = SetCover(U, V, M=1)
    e, sol = solve_qubo_bruteforce(p.to_qubo())
    solution = p.convert_solution(sol)
    assert p.is_solution_valid(solution)
    assert p.is_solution_valid(sol)
    assert solution == {0, 2}
    assert allclose(e, len(solution))


def test_setcover_qubo_solve():

    e, sol = solve_qubo_bruteforce(problem.to_qubo())
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert solution == {0, 2}
    assert allclose(e, len(solution))


def test_setcover_qubo_logtrick_numvars():

    Q = problem_log.to_qubo()
    assert (
        len(set(y for x in Q for y in x)) ==
        problem_log.num_binary_variables ==
        Q.num_binary_variables
    )


def test_setcover_qubo_numvars():

    Q_notlog = problem.to_qubo()
    assert (
        len(set(y for x in Q_notlog for y in x)) ==
        problem.num_binary_variables ==
        Q_notlog.num_binary_variables
    )


def test_weighted_setcover():

    e, sol = solve_qubo_bruteforce(problem_weighted.to_qubo())
    solution = problem_weighted.convert_solution(sol)
    assert problem_weighted.is_solution_valid(solution)
    assert problem_weighted.is_solution_valid(sol)
    assert solution == {0, 2}
    assert allclose(e, 1.1)


# quso

def test_setcover_quso_logtrick_solve():

    e, sol = solve_quso_bruteforce(problem_log.to_quso())
    solution = problem_log.convert_solution(sol)
    assert problem_log.is_solution_valid(solution)
    assert problem_log.is_solution_valid(sol)
    assert solution == {0, 2}
    assert allclose(e, len(solution))


def test_setcover_quso_solve():

    e, sol = solve_quso_bruteforce(problem.to_quso())
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert solution == {0, 2}
    assert allclose(e, len(solution))


def test_setcover_quso_logtrick_numvars():

    L = problem_log.to_quso()
    assert L.num_binary_variables == problem_log.num_binary_variables


def test_setcover_quso_numvars():

    L = problem.to_quso()
    assert L.num_binary_variables == problem.num_binary_variables


# PUBO

def test_setcover_pubo_solve():

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert solution == {0, 2}
    assert allclose(e, len(solution))


# puso

def test_setcover_puso_solve():

    e, sol = solve_puso_bruteforce(problem.to_puso())
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert problem.is_solution_valid(sol)
    assert solution == {0, 2}
    assert allclose(e, len(solution))
