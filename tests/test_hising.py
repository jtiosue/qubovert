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
Contains tests for the HHIsing class.
"""

from qubovert import HIsing
from qubovert.utils import solve_pubo_bruteforce, solve_hising_bruteforce
from numpy import allclose


problem = HIsing({('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4,
                  (): -2, (0, 1, 2): 1, (0,): 1, (1,): 1, (2,): 1})
solution = {'c': -1, 'b': -1, 'a': -1, 0: -1, 1: -1, 2: -1}
obj = -14


def test_hising_pubo_solve():

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    assert problem.is_solution_valid(sol)
    assert sol == solution
    assert allclose(e, obj)


def test_hising_hising_solve():

    e, sol = solve_hising_bruteforce(problem.to_hising())
    sol = problem.convert_solution(sol)
    assert problem.is_solution_valid(sol)
    assert sol == solution
    assert allclose(e, obj)


def test_hising_bruteforce_solve():

    assert problem.solve_bruteforce() == solution


def test_ising_default_valid():

    d = HIsing()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(): 1}

    d = HIsing()
    assert d[(0, 1)] == 0
    d[(0, 1)] += 1
    assert d == {(0, 1): 1}


def test_ising_remove_value_when_zero():

    d = HIsing()
    d[(0, 1)] += 1
    d[(0, 1)] -= 1
    assert d == {}


def test_ising_reinitialize_dictionary():

    d = HIsing({(0, 0): 1, ('1', 0): 2, (2, 0): 0, (0, '1'): 1})
    assert d in ({(): 1, ('1', 0): 3}, {(): 1, (0, '1'): 3})


def test_ising_update():

    d = HIsing({('0',): 1, ('0', 1): 2})
    d.update({('0', '0'): 0, (1, '0'): 1, (1, 1): -1})
    assert d in ({('0',): 1, (): -1, (1, '0'): 1},
                 {('0',): 1, (): -1, ('0', 1): 1})

    d = HIsing({(0, 0): 1, (0, 1): 2})
    d.update(HIsing({(1, 0): 1, (1, 1): -1}))
    d -= 1
    assert d == HIsing({(0, 1): 1, (): -2})

    assert d.offset == -2


def test_ising_num_binary_variables():

    d = HIsing({(0,): 1, (0, 3): 2})
    assert d.num_binary_variables == 2
    assert d.max_index == 1


def test_ising_addition():

    temp = HIsing({('0', '0'): 1, ('0', 1): 2})
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
    assert g == HIsing(temp3[0])*-1


def test_ising_multiplication():

    temp = HIsing({('0', '0'): 1, ('0', 1): 2})
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

    d = HIsing({('0', 1): 1, ('1', 2): -1})**2
    assert d ** 4 == d * d * d * d


def test_properties():

    temp = HIsing({('0', '0'): 1, ('0', 1): 2})
    temp.offset
    temp.mapping
    temp.reverse_mapping
