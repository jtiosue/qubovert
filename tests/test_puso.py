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

""" Contains tests for the PUSO class. """

from qubovert import PUSO
from qubovert.utils import (
    solve_qubo_bruteforce, solve_quso_bruteforce,
    solve_pubo_bruteforce, solve_puso_bruteforce,
    puso_value, pubo_to_puso, PUSOMatrix
)
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises


def test_pretty_str():

    def equal(expression, string):
        assert expression.pretty_str() == string

    z = [PUSO() + {(i,): 1} for i in range(3)]
    a, b = Symbol('a'), Symbol('b')

    equal(z[0], "z(0)")
    equal(-z[0], "-z(0)")
    equal(z[0] * 0, "0")
    equal(2*z[0]*z[1] - 3*z[2], "2 z(0) z(1) - 3 z(2)")
    equal(0*z[0] + 1, "1")
    equal(0*z[0] - 1, "-1")
    equal(0*z[0] + a, "(a)")
    equal(0*z[0] + a * b, "(a*b)")
    equal((a+b)*(z[0]*z[1] - z[2]), "(a + b) z(0) z(1) + (-a - b) z(2)")
    equal(2*z[0]*z[1] - z[2], "2 z(0) z(1) - z(2)")
    equal(-z[2] + z[0]*z[1], "-z(2) + z(0) z(1)")
    equal(-2*z[2] + 2*z[0]*z[1], "-2 z(2) + 2 z(0) z(1)")


def test_create_var():

    d = PUSO.create_var(0)
    assert d == {(0,): 1}
    assert d.name == 0
    assert type(d) == PUSO

    d = PUSO.create_var('x')
    assert d == {('x',): 1}
    assert d.name == 'x'
    assert type(d) == PUSO


class Problem:

    def __init__(self, problem, solution, obj):

        self.problem, self.solution, self.obj = problem, solution, obj

    def is_valid(self, e, solution, spin):

        sol = self.problem.convert_solution(solution, spin)
        return all((
            self.problem.is_solution_valid(sol),
            self.problem.is_solution_valid(solution),
            sol == self.solution,
            allclose(e, self.obj)
        ))

    def runtests(self):

        assert self.problem.solve_bruteforce() == self.solution

        e, sol = solve_qubo_bruteforce(self.problem.to_qubo())
        assert self.is_valid(e, sol, False)

        e, sol = solve_quso_bruteforce(self.problem.to_quso())
        assert self.is_valid(e, sol, True)

        for deg in (None,) + tuple(range(2, self.problem.degree + 1)):

            e, sol = solve_puso_bruteforce(self.problem.to_puso(deg))
            assert self.is_valid(e, sol, True)

            e, sol = solve_pubo_bruteforce(self.problem.to_pubo(deg))
            assert self.is_valid(e, sol, False)

        assert (
            self.problem.value(self.solution) ==
            puso_value(self.solution, self.problem) ==
            e
        )


def test_puso_on_quso():

    problem = PUSO({('a',): -1, ('b',): 2,
                   ('a', 'b'): -3, ('b', 'c'): -4, (): -2})
    solution = {'c': -1, 'b': -1, 'a': -1}
    obj = -10

    Problem(problem, solution, obj).runtests()


def test_puso_on_deg_3_puso():

    problem = PUSO({('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4,
                   (): -2, (0, 1, 2): 1, (0,): 1, (1,): 1, (2,): 1})
    solution = {'c': -1, 'b': -1, 'a': -1, 0: -1, 1: -1, 2: -1}
    obj = -14

    Problem(problem, solution, obj).runtests()


def test_puso_on_deg_5_puso():

    problem = PUSO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        (0, 1, 2): 1, (0,): -1, (1,): -2, (2,): 1, ('a', 0, 4, 'b', 'c'): -3,
        (4, 2, 3, 'a', 'b'): 2, (4, 2, 3, 'b'): -1, ('c',): 4, (3,): 1
    })
    solution = {0: 1, 1: 1, 'c': -1, 2: -1, 4: -1, 3: -1, 'b': -1, 'a': -1}
    obj = -26

    Problem(problem, solution, obj).runtests()


# testing methods

def test_puso_checkkey():

    with assert_raises(KeyError):
        PUSO({0: -1})


def test_puso_default_valid():

    d = PUSO()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(): 1}

    d = PUSO()
    assert d[(0, 1)] == 0
    d[(0, 1)] += 1
    assert d == {(0, 1): 1}


def test_puso_remove_value_when_zero():

    d = PUSO()
    d[(0, 1)] += 1
    d[(0, 1)] -= 1
    assert d == {}


def test_puso_reinitialize_dictionary():

    d = PUSO({(0, 0): 1, ('1', 0): 2, (2, 0): 0, (0, '1'): 1})
    assert d in ({(): 1, ('1', 0): 3}, {(): 1, (0, '1'): 3})


def test_puso_update():

    d = PUSO({('0',): 1, ('0', 1): 2})
    d.update({('0', '0'): 0, (1, '0'): 1, (1, 1): -1})
    assert d in ({('0',): 1, (): -1, (1, '0'): 1},
                 {('0',): 1, (): -1, ('0', 1): 1})

    d = PUSO({(0, 0): 1, (0, 1): 2})
    d.update(PUSO({(1, 0): 1, (1, 1): -1}))
    d -= 1
    assert d == PUSO({(0, 1): 1, (): -2})

    assert d.offset == -2


def test_puso_num_binary_variables():

    d = PUSO({(0,): 1, (0, 3): 2})
    assert d.num_binary_variables == 2
    assert d.max_index == 1


def test_num_terms():

    d = PUSO({(0,): 1, (0, 3): 2, (0, 2): -1})
    assert d.num_terms == len(d)


def test_puso_degree():

    d = PUSO()
    assert d.degree == -float("inf")
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


def test_puso_addition():

    temp = PUSO({('0', '0'): 1, ('0', 1): 2})
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
    assert g == PUSO(temp3[0])*-1


def test_puso_multiplication():

    temp = PUSO({('0', '0'): 1, ('0', 1): 2})
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

    d = PUSO({('0', 1): 1, ('1', 2): -1})**2
    assert d ** 4 == d * d * d * d


def test_properties():

    temp = PUSO({('0', '0'): 1, ('0', 1): 2})
    assert temp.offset == 1

    d = PUSO()
    d[(0,)] += 1
    d[(1,)] += 2
    assert d == d.to_quso() == {(0,): 1, (1,): 2}
    assert d.mapping == d.reverse_mapping == {0: 0, 1: 1}

    d.set_mapping({1: 0, 0: 1})
    assert d.to_quso() == {(1,): 1, (0,): 2}
    assert d.mapping == d.reverse_mapping == {0: 1, 1: 0}

    # an old bug
    d = PUSO()
    d.set_mapping({0: 0})
    d[(0,)] += 1
    assert d.num_binary_variables == 1
    assert d.variables == {0}


def test_round():

    d = PUSO({(0,): 3.456, (1,): -1.53456})

    assert round(d) == {(0,): 3, (1,): -2}
    assert round(d, 1) == {(0,): 3.5, (1,): -1.5}
    assert round(d, 2) == {(0,): 3.46, (1,): -1.53}
    assert round(d, 3) == {(0,): 3.456, (1,): -1.535}


def test_normalize():

    temp = {(0,): 4, (1,): -2}
    d = PUSO(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}

    temp = {(0,): -4, (1,): 2}
    d = PUSO(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = PUSO()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}


def test_convert_solution_all_1s():

    d = PUSO({(0,): 1})
    assert d.convert_solution({0: 0}) == {0: 1}
    assert d.convert_solution({0: -1}) == {0: -1}
    assert d.convert_solution({0: 1}) == {0: 1}
    assert d.convert_solution({0: 1}, False) == {0: -1}


def test_set_mapping():

    d = PUSO({('a', 'b'): 1, ('a',): 2})
    d.set_mapping({'a': 0, 'b': 2})
    assert d.to_puso() == {(0, 2): 1, (0,): 2}

    d = PUSO({('a', 'b'): 1, ('a',): 2})
    d.set_reverse_mapping({0: 'a', 2: 'b'})
    assert d.to_puso() == {(0, 2): 1, (0,): 2}


def test_to_enumerated():

    d = PUSO({('a', 'b'): 1, ('a',): 2})
    dt = d.to_enumerated()
    assert type(dt) == PUSOMatrix
    assert dt == d.to_puso()


def test_puso_degree_reduction_pairs():

    puso = pubo_to_puso({
        ('x0', 'x1'): -1, ('x1',): 1, ('x1', 'x2'): -1, ('x2',): 1,
        ('x3', 'x2'): -1, ('x3',): 1, ('x4', 'x3'): -1, ('x4',): 1,
        ('x4', 'x5'): -1, ('x5',): 1, ('x5', 'x6'): -1, ('x6',): 1,
        ('x7', 'x6'): -1, ('x7',): 1, ('x8', 'x7'): -1, ('x8',): 1,
        ('x9', 'x8'): -1, ('x9',): 1
     }) ** 2
    pairs = {
        ('x0', 'x1'), ('x1', 'x2'), ('x2', 'x3'), ('x3', 'x4'), ('x4', 'x5'),
        ('x5', 'x6'), ('x6', 'x7'), ('x7', 'x8'), ('x8', 'x9')
    }
    quso1 = puso.to_quso()
    quso2 = puso.to_quso(pairs=pairs)
    assert quso1.num_binary_variables - puso.num_binary_variables > 9
    assert quso2.num_binary_variables - puso.num_binary_variables == 9
    qubo1 = puso.to_qubo()
    qubo2 = puso.to_qubo(pairs=pairs)
    assert qubo1.num_binary_variables - puso.num_binary_variables > 9
    assert qubo2.num_binary_variables - puso.num_binary_variables == 9


def test_puso_degree_reduction_lam():

    puso = PUSO({
        ('x0', 'x1'): -1, ('x1',): 1, ('x1', 'x2'): -1, ('x2',): 1,
        ('x3', 'x2'): -1, ('x3',): 1, ('x4', 'x3'): -1, ('x4',): 1,
     }) ** 2

    # just make sure it runs
    puso.to_qubo(lam=4)
    puso.to_qubo(lam=lambda v: v)
    puso.to_qubo(lam=Symbol('lam'))
    puso.to_qubo(lam=lambda v: v * Symbol('lam'))
    puso.to_quso(lam=4)
    puso.to_quso(lam=lambda v: v)
    puso.to_quso(lam=Symbol('lam'))
    puso.to_quso(lam=lambda v: v * Symbol('lam'))
