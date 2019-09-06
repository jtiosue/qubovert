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
Contains tests for the HOIO class.
"""

from qubovert import HOIO
from qubovert.utils import (
    solve_qubo_bruteforce, solve_ising_bruteforce,
    solve_pubo_bruteforce, solve_hising_bruteforce,
    hising_value
)
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises


""" TESTS FOR THE METHODS THAT HOBO INHERITS FROM HISING """


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
            hising_value(self.solution, self.problem) ==
            e
        )


def test_hoio_on_ising():

    problem = HOIO({('a',): -1, ('b',): 2,
                   ('a', 'b'): -3, ('b', 'c'): -4, (): -2})
    solution = {'c': -1, 'b': -1, 'a': -1}
    obj = -10

    Problem(problem, solution, obj).runtests()


def test_hoio_on_deg_3_hising():

    problem = HOIO({('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4,
                   (): -2, (0, 1, 2): 1, (0,): 1, (1,): 1, (2,): 1})
    solution = {'c': -1, 'b': -1, 'a': -1, 0: -1, 1: -1, 2: -1}
    obj = -14

    Problem(problem, solution, obj).runtests()


def test_hoio_on_deg_5_hising():

    problem = HOIO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        (0, 1, 2): 1, (0,): -1, (1,): -2, (2,): 1, ('a', 0, 4, 'b', 'c'): -3,
        (4, 2, 3, 'a', 'b'): 2, (4, 2, 3, 'b'): -1, ('c',): 4, (3,): 1
    })
    solution = {0: 1, 1: 1, 'c': -1, 2: -1, 4: -1, 3: -1, 'b': -1, 'a': -1}
    obj = -26

    Problem(problem, solution, obj).runtests()


# testing methods

def test_hoio_checkkey():

    with assert_raises(KeyError):
        HOIO({0: -1})


def test_ising_default_valid():

    d = HOIO()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(): 1}

    d = HOIO()
    assert d[(0, 1)] == 0
    d[(0, 1)] += 1
    assert d == {(0, 1): 1}


def test_ising_remove_value_when_zero():

    d = HOIO()
    d[(0, 1)] += 1
    d[(0, 1)] -= 1
    assert d == {}


def test_ising_reinitialize_dictionary():

    d = HOIO({(0, 0): 1, ('1', 0): 2, (2, 0): 0, (0, '1'): 1})
    assert d in ({(): 1, ('1', 0): 3}, {(): 1, (0, '1'): 3})


def test_ising_update():

    d = HOIO({('0',): 1, ('0', 1): 2})
    d.update({('0', '0'): 0, (1, '0'): 1, (1, 1): -1})
    assert d in ({('0',): 1, (): -1, (1, '0'): 1},
                 {('0',): 1, (): -1, ('0', 1): 1})

    d = HOIO({(0, 0): 1, (0, 1): 2})
    d.update(HOIO({(1, 0): 1, (1, 1): -1}))
    d -= 1
    assert d == HOIO({(0, 1): 1, (): -2})

    assert d.offset == -2


def test_ising_num_binary_variables():

    d = HOIO({(0,): 1, (0, 3): 2})
    assert d.num_binary_variables == 2
    assert d.max_index == 1


def test_hoio_degree():

    d = HOIO()
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


def test_ising_addition():

    temp = HOIO({('0', '0'): 1, ('0', 1): 2})
    temp1 = {('0',): -1, (1, '0'): 3}
    temp2 = {(1, '0'): 5, (): 1, ('0',): -1}, {('0', 1): 5, (): 1, ('0',): -1}
    temp3 = {(): 1, (1, '0'): -1, ('0',): 1}, {(): 1, ('0', 1): -1, ('0',): 1}

    # constant
    d = temp.copy()
    d += 5
    assert d in ({(1, '0'): 2, (): 6}, {('0', 1): 2, (): 6})

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
    assert g == HOIO(temp3[0])*-1


def test_ising_multiplication():

    temp = HOIO({('0', '0'): 1, ('0', 1): 2})
    temp1 = {(): 3, (1, '0'): 6}, {(): 3, ('0', 1): 6}
    temp2 = {(): .5, (1, '0'): 1}, {(): .5, ('0', 1): 1}
    temp3 = {(1, '0'): 1}, {('0', 1): 1}

    # constant
    d = temp.copy()
    d += 3
    d *= -2
    assert d in ({(1, '0'): -4, (): -8}, {('0', 1): -4, (): -8})

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
    assert g in temp3

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d in temp3

    # __mul__ but with dict
    d = temp.copy()
    d *= {(1,): 2, ('0', '0'): -1}
    assert d in ({(1,): 2, (): -1, ('0',): 4, ('0', 1): -2},
                 {(1,): 2, (): -1, ('0',): 4, (1, '0'): -2})

    # __pow__
    d = temp.copy()
    d -= 2
    d **= 2
    assert d in ({(): 5, ('0', 1): -4}, {(): 5, (1, '0'): -4})

    d = temp.copy()
    assert d ** 2 == d * d
    assert d ** 3 == d * d * d

    d = HOIO({('0', 1): 1, ('1', 2): -1})**2
    assert d ** 4 == d * d * d * d


def test_properties():

    temp = HOIO({('0', '0'): 1, ('0', 1): 2})
    temp.offset
    temp.mapping
    temp.reverse_mapping


def test_round():

    d = HOIO({(0,): 3.456, (1,): -1.53456})

    assert round(d) == {(0,): 3, (1,): -2}
    assert round(d, 1) == {(0,): 3.5, (1,): -1.5}
    assert round(d, 2) == {(0,): 3.46, (1,): -1.53}
    assert round(d, 3) == {(0,): 3.456, (1,): -1.535}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = HOIO()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}

    d.add_constraint_eq_zero({(0,): a, (1,): -b})
    assert d == {(0,): -a, (0, 1): -2*a*b + 2, (1,): b, (): a**2 + b**2}
    assert d.subs(a, 0) == {(0, 1): 2, (1,): b, (): b**2}
    assert d.subs({a: 0, b: 2}) == {(0, 1): 2, (1,): 2, (): 4}


""" TESTS FOR THE CONSTRAINT METHODS """


def test_hobo_eq_constraint():

    problem = HOIO({('a',): -1, ('b',): -2, ('a', 'b'): -10, (): -2})
    problem.add_constraint_eq_zero({('a',): 1, ('b',): 1})
    solution = {'b': 1, 'a': -1}
    obj = 7

    sol = problem.solve_bruteforce()
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = HOIO({('a',): -1, ('b',): -2, ('a', 'b'): -10, (): -2})
    problem.add_constraint_eq_zero({('a',): 1, ('b',): 1}, lam=10)
    solution = {'b': 1, 'a': -1}
    obj = 7

    sol = problem.solve_bruteforce()
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))
