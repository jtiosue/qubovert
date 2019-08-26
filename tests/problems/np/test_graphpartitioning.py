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
Contains tests for the GraphPartitioning class.
"""

from qubovert.problems import GraphPartitioning
from qubovert.utils import solve_qubo_bruteforce, solve_ising_bruteforce
from numpy import allclose


edges = {("a", "b"), ("a", "c"), ("c", "d"),
         ("b", "c"), ("e", "f"), ("d", "e")}
problem = GraphPartitioning(edges)
solutions = (
    ({"a", "b", "c"}, {"d", "e", "f"}),
    ({"d", "e", "f"}, {"a", "b", "c"})
)

problem_weighted = GraphPartitioning({(0, 1): 1, (1, 2): 3, (0, 3): 1})
solutions_weighted = (
    ({0, 3}, {1, 2}),
    ({1, 2}, {0, 3})
)


def test_graphpartitioning_str():

    assert eval(str(problem)) == problem


def test_graphpartitioning_bruteforce():

    assert problem.solve_bruteforce() in solutions
    assert (
        problem.solve_bruteforce(all_solutions=True) in
        (list(solutions), list(reversed(solutions)))
    )


# QUBO

def test_graphpartitioning_qubo_solve():

    e, sol = solve_qubo_bruteforce(problem.to_qubo())
    solution = problem.convert_solution(sol)

    assert solution in solutions
    assert problem.is_solution_valid(solution)
    assert allclose(e, 1)

    e, sol = solve_qubo_bruteforce(problem_weighted.to_qubo())
    solution = problem_weighted.convert_solution(sol)

    assert solution in solutions_weighted
    assert problem.is_solution_valid(solution)
    assert allclose(e, 1)


def test_graphpartitioning_qubo_numvars():

    Q = problem.to_qubo()
    assert (
        len(set(y for x in Q for y in x)) ==
        problem.num_binary_variables ==
        Q.num_binary_variables
    )


# ising

def test_graphpartitioning_ising_solve():

    e, sol = solve_ising_bruteforce(problem.to_ising())
    solution = problem.convert_solution(sol)

    assert solution in solutions
    assert problem.is_solution_valid(solution)
    assert allclose(e, 1)

    e, sol = solve_ising_bruteforce(problem_weighted.to_ising())
    solution = problem_weighted.convert_solution(sol)

    assert solution in solutions_weighted
    assert problem.is_solution_valid(solution)
    assert allclose(e, 1)


def test_graphpartitioning_ising_numvars():

    L = problem.to_ising()
    assert L.num_binary_variables == problem.num_binary_variables
