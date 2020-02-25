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
Contains tests for the QUSO class.
"""

from qubovert import QUSO
from qubovert.utils import (
    solve_qubo_bruteforce, solve_quso_bruteforce,
    solve_pubo_bruteforce, solve_puso_bruteforce,
    quso_value
)
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises


problem = QUSO({('a',): -1, ('b',): 2,
               ('a', 'b'): -3, ('b', 'c'): -4, (): -2})
solution = {'c': -1, 'b': -1, 'a': -1}


def test_quso_qubo_solve():

    e, sols = solve_qubo_bruteforce(problem.to_qubo())
    sol = problem.convert_solution(sols, False)
    assert problem.is_solution_valid(sol)
    assert problem.is_solution_valid(sols)
    assert sol == solution
    assert allclose(e, -10)


def test_quso_quso_solve():

    e, sols = solve_quso_bruteforce(problem.to_quso())
    sol = problem.convert_solution(sols)
    assert problem.is_solution_valid(sol)
    assert problem.is_solution_valid(sols)
    assert sol == solution
    assert allclose(e, -10)

    assert (
        problem.value(sol) ==
        quso_value(sol, problem) ==
        e
    )


def test_quso_pubo_solve():

    e, sols = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sols, False)
    assert problem.is_solution_valid(sol)
    assert problem.is_solution_valid(sols)
    assert sol == solution
    assert allclose(e, -10)


def test_quso_puso_solve():

    e, sols = solve_puso_bruteforce(problem.to_puso())
    sol = problem.convert_solution(sols)
    assert problem.is_solution_valid(sol)
    assert problem.is_solution_valid(sols)
    assert sol == solution
    assert allclose(e, -10)

    assert (
        problem.value(sol) ==
        quso_value(sol, problem) ==
        e
    )


def test_quso_bruteforce_solve():

    assert problem.solve_bruteforce() == solution


# testing methods

def test_quso_checkkey():

    with assert_raises(KeyError):
        QUSO({0: -1})

    with assert_raises(KeyError):
        QUSO({(0, 1, 2): -1})


def test_quso_default_valid():

    d = QUSO()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(): 1}

    d = QUSO()
    assert d[(0, 1)] == 0
    d[(0, 1)] += 1
    assert d == {(0, 1): 1}


def test_quso_remove_value_when_zero():

    d = QUSO()
    d[(0, 1)] += 1
    d[(0, 1)] -= 1
    assert d == {}


def test_quso_reinitialize_dictionary():

    d = QUSO({(0, 0): 1, ('1', 0): 2, (2, 0): 0, (0, '1'): 1})
    assert d in ({(): 1, ('1', 0): 3}, {(): 1, (0, '1'): 3})


def test_quso_update():

    d = QUSO({('0',): 1, ('0', 1): 2})
    d.update({('0', '0'): 0, (1, '0'): 1, (1, 1): -1})
    assert d in ({('0',): 1, (): -1, (1, '0'): 1},
                 {('0',): 1, (): -1, ('0', 1): 1})

    d = QUSO({(0, 0): 1, (0, 1): 2})
    d.update(QUSO({(1, 0): 1, (1, 1): -1}))
    d -= 1
    assert d == QUSO({(0, 1): 1, (): -2})

    assert d.offset == -2


def test_quso_num_binary_variables():

    d = QUSO({(0,): 1, (0, 3): 2})
    assert d.num_binary_variables == 2
    assert d.max_index == 1


def test_num_terms():

    d = QUSO({(0,): 1, (0, 3): 2, (0, 2): -1})
    assert d.num_terms == len(d)


def test_quso_degree():

    d = QUSO()
    assert d.degree == 0
    d[(0,)] += 2
    assert d.degree == 1
    d[(1,)] -= 3
    assert d.degree == 1
    d[(1, 2)] -= 2
    assert d.degree == 2


def test_quso_addition():

    temp = QUSO({('0', '0'): 1, ('0', 1): 2})
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
    assert g == QUSO(temp3[0])*-1


def test_quso_multiplication():

    temp = QUSO({('0', '0'): 1, ('0', 1): 2})
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

    # should raise a KeyError since can't fit this into QUSO.
    with assert_raises(KeyError):
        QUSO({('0', 1): 1, ('1', 2): -1})**2


def test_properties():

    temp = QUSO({('0', '0'): 1, ('0', 1): 2})
    assert temp.offset == 1

    d = QUSO()
    d[(0,)] += 1
    d[(1,)] += 2
    assert d == d.to_quso() == {(0,): 1, (1,): 2}
    assert d.mapping == d.reverse_mapping == {0: 0, 1: 1}

    d.set_mapping({1: 0, 0: 1})
    assert d.to_quso() == {(1,): 1, (0,): 2}
    assert d.mapping == d.reverse_mapping == {0: 1, 1: 0}


def test_round():

    d = QUSO({(0,): 3.456, (1,): -1.53456})

    assert round(d) == {(0,): 3, (1,): -2}
    assert round(d, 1) == {(0,): 3.5, (1,): -1.5}
    assert round(d, 2) == {(0,): 3.46, (1,): -1.53}
    assert round(d, 3) == {(0,): 3.456, (1,): -1.535}


def test_normalize():

    temp = {(0,): 4, (1,): -2}
    d = QUSO(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}

    temp = {(0,): -4, (1,): 2}
    d = QUSO(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = QUSO()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}


def test_convert_solution_all_1s():

    d = QUSO({(0,): 1})
    assert d.convert_solution({0: 0}) == {0: 1}
    assert d.convert_solution({0: -1}) == {0: -1}
    assert d.convert_solution({0: 1}) == {0: 1}
    assert d.convert_solution({0: 1}, False) == {0: -1}


def test_set_mapping():

    d = QUSO({('a', 'b'): 1, ('a',): 2})
    d.set_mapping({'a': 0, 'b': 2})
    assert d.to_quso() == {(0, 2): 1, (0,): 2}

    d = QUSO({('a', 'b'): 1, ('a',): 2})
    d.set_reverse_mapping({0: 'a', 2: 'b'})
    assert d.to_quso() == {(0, 2): 1, (0,): 2}
