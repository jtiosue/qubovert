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
Contains tests for the PUBO class.
"""

from qubovert import PUBO
from qubovert.utils import (
    solve_qubo_bruteforce, solve_ising_bruteforce,
    solve_pubo_bruteforce, solve_hising_bruteforce,
    pubo_value
)
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises


class Problem:

    def __init__(self, problem, solution, obj):

        self.problem, self.solution, self.obj = problem, solution, obj

    def is_valid(self, e, solution):

        sol = self.problem.convert_solution(solution)
        return all((
            self.problem.is_solution_valid(sol),
            self.problem.is_solution_valid(solution),
            sol == self.solution,
            allclose(e, self.obj)
        ))

    def runtests(self):

        assert self.problem.solve_bruteforce() == self.solution

        e, sol = solve_qubo_bruteforce(self.problem.to_qubo())
        assert self.is_valid(e, sol)

        e, sol = solve_ising_bruteforce(self.problem.to_ising())
        assert self.is_valid(e, sol)

        for deg in (None,) + tuple(range(2, self.problem.degree + 1)):

            e, sol = solve_hising_bruteforce(self.problem.to_hising(deg))
            assert self.is_valid(e, sol)

            e, sol = solve_pubo_bruteforce(self.problem.to_pubo(deg))
            assert self.is_valid(e, sol)

        assert (
            self.problem.value(self.solution) ==
            pubo_value(self.solution, self.problem) ==
            e
        )


def test_pubo_on_qubo():

    problem = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    solution = {'c': 1, 'b': 1, 'a': 1}
    obj = -8

    with assert_raises(ValueError):
        problem.to_pubo(deg=1)

    Problem(problem, solution, obj).runtests()


def test_pubo_on_deg_3_pubo():

    problem = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        (0, 1, 2): 1, (0,): -1, (1,): -2, (2,): 1
    })
    solution = {'c': 1, 'b': 1, 'a': 1, 0: 1, 1: 1, 2: 0}
    obj = -11

    Problem(problem, solution, obj).runtests()


def test_pubo_on_deg_5_pubo():

    problem = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        (0, 1, 2): 1, (0,): -1, (1,): -2, (2,): 1, ('a', 0, 4, 'b', 'c'): -3,
        (4, 2, 3, 'a', 'b'): 2, (4, 2, 3, 'b'): -1, ('c',): 4, (3,): 1,
        (0, 1): -2
    })
    solution = {0: 1, 1: 1, 'c': 1, 2: 0, 4: 1, 3: 0, 'b': 1, 'a': 1}
    obj = -12

    Problem(problem, solution, obj).runtests()


# testing methods

def test_pubo_checkkey():

    with assert_raises(KeyError):
        PUBO({0: -1})


def test_pubo_default_valid():

    d = PUBO()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0,): 1}


def test_pubo_remove_value_when_zero():

    d = PUBO()
    d[(0, 0, 1, 2)] += 1
    d[(0, 1, 2)] -= 1
    assert d == {}


def test_pubo_reinitialize_dictionary():

    d = PUBO({(0, 0): 1, ('1', 0): 2, (2, 0): 0, (0, '1'): 1})
    assert d in ({(0,): 1, ('1', 0): 3}, {(0,): 1, (0, '1'): 3})


def test_pubo_update():

    d = PUBO({('0',): 1, ('0', 1): 2})
    d.update({('0', '0'): 0, (1, '0'): 1, (1, 1): -1})
    assert d in ({(1, '0'): 1, (1,): -1}, {('0', 1): 1, (1,): -1})

    d = PUBO({(0, 0): 1, (0, 1): 2})
    d.update(PUBO({(1, 0): 1, (1, 1): -1}))
    d -= 1
    assert d == PUBO({(0,): 1, (0, 1): 1, (1,): -1, (): -1})

    assert d.offset == -1


def test_pubo_num_binary_variables():

    d = PUBO({(0, 0): 1, (0, 1, 2, 3, 5): 2})
    assert d.num_binary_variables == 5
    assert d.max_index == 4


def test_pubo_degree():

    d = PUBO()
    assert d.degree == 0
    d[(0,)] += 2
    assert d.degree == 1
    d[(1,)] -= 3
    assert d.degree == 1
    d[(1, 2)] -= 2
    assert d.degree == 2
    d[(1, 2, 4)] -= 2
    assert d.degree == 3
    d[(1, 2, 4, 5, 6)] += 2
    assert d.degree == 5


def test_pubo_addition():

    temp = PUBO({('0', '0'): 1, ('0', 1): 2})
    temp1 = {('0',): -1, (1, '0'): 3}
    temp2 = {(1, '0'): 5}, {('0', 1): 5}
    temp3 = {('0',): 2, (1, '0'): -1}, {('0',): 2, ('0', 1): -1}

    # constant
    d = temp.copy()
    d += 5
    assert d in ({('0',): 1, (1, '0'): 2, (): 5},
                 {('0',): 1, ('0', 1): 2, (): 5})

    # __add__
    d = temp.copy()
    g = d + temp1
    assert g in temp2

    # __iadd__
    d = temp.copy()
    d += temp1
    assert d in temp2

    # __radd__
    d = temp.copy()
    g = temp1 + d
    assert g in temp2

    # __sub__
    d = temp.copy()
    g = d - temp1
    assert g in temp3

    # __isub__
    d = temp.copy()
    d -= temp1
    assert d in temp3

    # __rsub__
    d = temp.copy()
    g = temp1 - d
    assert g == PUBO(temp3[0])*-1


def test_pubo_multiplication():

    temp = PUBO({('0', '0'): 1, ('0', 1): 2})
    temp1 = {('0',): 3, (1, '0'): 6}, {('0',): 3, ('0', 1): 6}
    temp2 = {('0',): .5, (1, '0'): 1}, {('0',): .5, ('0', 1): 1}

    # constant
    d = temp.copy()
    d += 3
    d *= -2
    assert d in ({('0',): -2, (1, '0'): -4, (): -6},
                 {('0',): -2, ('0', 1): -4, (): -6})

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g in temp1

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d in temp1

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g in temp1

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g in temp2

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d in temp2

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g in ({(1, '0'): 1}, {('0', 1): 1})

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d in ({(1, '0'): 1}, {('0', 1): 1})

    # __mul__ but with dict
    d = temp.copy()
    d *= {(1,): 2, ('0', '0'): -1}
    assert d in ({('0',): -1, (1, '0'): 4}, {('0',): -1, ('0', 1): 4})

    # __pow__
    d = temp.copy()
    d -= 2
    d **= 2
    assert d == {('0',): -3, (): 4}

    d = temp.copy()
    assert d ** 3 == d * d * d


def test_properties():

    temp = PUBO({('0', '0'): 1, ('0', 1): 2})
    temp.offset
    temp.mapping
    temp.reverse_mapping


def test_round():

    d = PUBO({(0,): 3.456, (1,): -1.53456})

    assert round(d) == {(0,): 3, (1,): -2}
    assert round(d, 1) == {(0,): 3.5, (1,): -1.5}
    assert round(d, 2) == {(0,): 3.46, (1,): -1.53}
    assert round(d, 3) == {(0,): 3.456, (1,): -1.535}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = PUBO()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}
