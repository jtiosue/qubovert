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
Contains tests for the SetCover class.
"""

from qubovert import SetCover
from qubovert.utils import solve_qubo_bruteforce, solve_ising_bruteforce
from numpy import allclose


U = {"a", "b", "c", "d"}
V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]
problem_log = SetCover(U, V)
problem = SetCover(U, V, log_trick=False)
problem_weighted = SetCover(U, V, weights=(.1, .2, 1))


def test_setcover_str():

    assert eval(str(problem)) == problem
    assert eval(str(problem_log)) == problem_log
    assert eval(str(problem_weighted)) == problem_weighted


def test_coverable():

    assert problem.is_coverable()
    assert not SetCover(U, V[:-1]).is_coverable()


def test_setcover_bruteforce():

    assert problem.solve_bruteforce() == {0, 2}


# QUBO

def test_setcover_qubo_logtrick_solve():

    Q, offset = problem_log.to_qubo()
    e, sol = solve_qubo_bruteforce(Q, offset)
    solution = problem_log.convert_solution(sol)
    assert problem_log.is_solution_valid(solution)
    assert solution == {0, 2}
    assert allclose(e, len(solution))


def test_setcover_qubo_solve():

    Q_notlog, offset = problem.to_qubo()
    e, sol = solve_qubo_bruteforce(Q_notlog, offset)
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert solution == {0, 2}
    assert allclose(e, len(solution))


def test_setcover_qubo_logtrick_numvars():

    Q, offset = problem_log.to_qubo()
    assert (
        len(set(y for x in Q for y in x))
        ==
        problem_log.num_binary_variables
    )


def test_setcover_qubo_numvars():

    Q_notlog, offset = problem.to_qubo()
    assert (
        len(set(y for x in Q_notlog for y in x))
        ==
        problem.num_binary_variables
    )


def test_weighted_setcover():

    Q, offset = problem_weighted.to_qubo()
    e, sol = solve_qubo_bruteforce(Q, offset)
    solution = problem_weighted.convert_solution(sol)
    assert problem_weighted.is_solution_valid(solution)
    assert solution == {0, 2}
    assert allclose(e, 1.1)


# ising

def test_setcover_ising_logtrick_solve():

    h, J, offset = problem_log.to_ising()
    e, sol = solve_ising_bruteforce(h, J, offset)
    solution = problem_log.convert_solution(sol)
    assert problem_log.is_solution_valid(solution)
    assert solution == {0, 2}
    assert allclose(e, len(solution))


def test_setcover_ising_solve():

    h, J, offset = problem.to_ising()
    e, sol = solve_ising_bruteforce(h, J, offset)
    solution = problem.convert_solution(sol)
    assert problem.is_solution_valid(solution)
    assert solution == {0, 2}
    assert allclose(e, len(solution))


def test_setcover_ising_logtrick_numvars():

    h, J, offset = problem_log.to_ising()
    assert (
        len(set(y for x in J for y in x).union(set(h.keys())))
        ==
        problem_log.num_binary_variables
    )


def test_setcover_ising_numvars():

    h, J, offset = problem.to_ising()
    assert (
        len(set(y for x in J for y in x).union(set(h.keys())))
        ==
        problem.num_binary_variables
    )
