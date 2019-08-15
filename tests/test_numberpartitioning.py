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
Contains tests for the NumberPartitioning class.
"""

from qubovert import NumberPartitioning
from qubovert.utils import solve_qubo_bruteforce, solve_ising_bruteforce
from numpy import allclose


S_withsoln = 1, 2, 3, 4
S_withoutsoln = 1, 3, 4, 4

problem_withsoln = NumberPartitioning(S_withsoln)
problem_withoutsoln = NumberPartitioning(S_withoutsoln)

solutions_withsoln = ((1, 4), (2, 3)), ((2, 3), (1, 4))
solutions_withoutsoln = ((1, 4), (3, 4)), ((3, 4), (1, 4))


def test_numberpartitioning_str():

    assert eval(str(problem_withsoln)) == problem_withsoln
    assert eval(str(problem_withoutsoln)) == problem_withoutsoln


def test_numberpartitioning_bruteforce():

    assert problem_withsoln.solve_bruteforce() in solutions_withsoln
    assert (
        problem_withsoln.solve_bruteforce(all_solutions=True) in
        (list(solutions_withsoln), list(reversed(solutions_withsoln)))
    )

    assert problem_withoutsoln.solve_bruteforce() in solutions_withoutsoln


# QUBO

def test_numberpartitioning_qubo_solve():

    Q, offset = problem_withsoln.to_qubo()
    e, sol = solve_qubo_bruteforce(Q, offset)
    solution = problem_withsoln.convert_solution(sol)

    assert solution in solutions_withsoln
    assert problem_withsoln.is_solution_valid(solution)
    assert allclose(e, 0)

    Q, offset = problem_withoutsoln.to_qubo()
    e, sol = solve_qubo_bruteforce(Q, offset)
    solution = problem_withoutsoln.convert_solution(sol)

    assert solution in solutions_withoutsoln
    assert not problem_withoutsoln.is_solution_valid(solution)
    assert e != 0


def test_numberpartitioning_qubo_numvars():

    Q, _ = problem_withsoln.to_qubo()
    assert (
        len(set(y for x in Q for y in x)) ==
        problem_withsoln.num_binary_variables
    )

    Q, offset = problem_withoutsoln.to_qubo()
    assert (
        len(set(y for x in Q for y in x)) ==
        problem_withoutsoln.num_binary_variables
    )

# ising


def test_numberpartitioning_ising_solve():

    h, J, offset = problem_withsoln.to_ising()
    e, sol = solve_ising_bruteforce(h, J, offset)
    solution = problem_withsoln.convert_solution(sol)

    assert solution in solutions_withsoln
    assert problem_withsoln.is_solution_valid(solution)
    assert allclose(e, 0)

    h, J, offset = problem_withoutsoln.to_ising()
    e, sol = solve_ising_bruteforce(h, J, offset)
    solution = problem_withoutsoln.convert_solution(sol)

    assert solution in solutions_withoutsoln
    assert not problem_withoutsoln.is_solution_valid(solution)
    assert e != 0


def test_numberpartitioning_ising_numvars():

    h, J, _ = problem_withsoln.to_ising()
    assert (
        len(set(y for x in J for y in x).union(set(h.keys()))) ==
        problem_withsoln.num_binary_variables
    )

    h, J, _ = problem_withoutsoln.to_ising()
    assert (
        len(set(y for x in J for y in x).union(set(h.keys()))) ==
        problem_withoutsoln.num_binary_variables
    )
