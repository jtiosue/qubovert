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
Contains tests for the AlternatingSectorsChain class.
"""

from qubovert.problems import AlternatingSectorsChain
from qubovert.utils import solve_qubo_bruteforce, solve_ising_bruteforce
from numpy import allclose


problem = AlternatingSectorsChain(12)


def test_AlternatingSectorsChain_str():

    assert eval(str(problem)) == problem


def test_AlternatingSectorsChain_bruteforce():

    assert problem.solve_bruteforce() in ((-1,)*12, (1,)*12)
    assert (
        problem.solve_bruteforce(all_solutions=True) in
        ([(-1,)*12, (1,)*12], [(1,)*12, (-1,)*12])
    )


# QUBO

def test_AlternatingSectorsChain_qubo_solve():

    Q, offset = problem.to_qubo(True)
    e, sol = solve_qubo_bruteforce(Q, offset)
    solution = problem.convert_solution(sol)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert allclose(e, -66)

    # not pbc

    Q, offset = problem.to_qubo(False)
    e, sol = solve_qubo_bruteforce(Q, offset)
    solution = problem.convert_solution(sol)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert allclose(e, -65)


def test_AlternatingSectorsChain_qubo_numvars():

    Q, offset = problem.to_qubo()
    assert (
        len(set(y for x in Q for y in x)) ==
        problem.num_binary_variables
    )


# ising

def test_AlternatingSectorsChain_ising_solve():

    args = problem.to_ising(True)
    e, sol = solve_ising_bruteforce(*args)
    solution = problem.convert_solution(sol)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert allclose(e, -66)

    # not pbc

    args = problem.to_ising(False)
    e, sol = solve_ising_bruteforce(*args)
    solution = problem.convert_solution(sol)

    assert solution == (-1,) * 12 or solution == (1,) * 12
    assert problem.is_solution_valid(solution)
    assert allclose(e, -65)


def test_AlternatingSectorsChain_ising_numvars():

    h, J, _ = problem.to_ising()
    assert (
        len(set(y for x in J for y in x).union(set(h.keys()))) ==
        problem.num_binary_variables
    )
