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
Contains tests for the QUBO class.
"""

from qubovert import QUBO
from qubovert.utils import solve_qubo_bruteforce, solve_ising_bruteforce
from numpy import allclose


problem = QUBO({('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2})
solution = {'c': 1, 'b': 1, 'a': 1}
obj = -8


def test_qubo_qubo_solve():

    e, sol = solve_qubo_bruteforce(problem.to_qubo())
    sol = problem.convert_solution(sol)
    assert problem.is_solution_valid(sol)
    assert sol == solution
    assert allclose(e, obj)


def test_qubo_ising_solve():

    e, sol = solve_ising_bruteforce(problem.to_ising())
    sol = problem.convert_solution(sol)
    assert problem.is_solution_valid(sol)
    assert sol == solution
    assert allclose(e, obj)


def test_qubo_bruteforce_solve():

    assert problem.solve_bruteforce() == solution


def test_qubo_default_valid():

    d = QUBO()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0,): 1}


def test_qubo_remove_value_when_zero():

    d = QUBO()
    d[(0, 0)] += 1
    d[(0, 0)] -= 1
    assert d == {}


def test_qubo_reinitialize_dictionary():

    d = QUBO({(0, 0): 1, ('1', 0): 2, (2, 0): 0, (0, '1'): 1})
    assert d in ({(0,): 1, ('1', 0): 3}, {(0,): 1, (0, '1'): 3})


def test_qubo_update():

    d = QUBO({('0',): 1, ('0', 1): 2})
    d.update({('0', '0'): 0, (1, '0'): 1, (1, 1): -1})
    assert d in ({(1, '0'): 1, (1,): -1}, {('0', 1): 1, (1,): -1})

    d = QUBO({(0, 0): 1, (0, 1): 2})
    d.update(QUBO({(1, 0): 1, (1, 1): -1}))
    d -= 1
    assert d == QUBO({(0,): 1, (0, 1): 1, (1,): -1, (): -1})

    assert d.offset == -1


def test_qubo_num_binary_variables():

    d = QUBO({(0, 0): 1, (0, 3): 2})
    assert d.num_binary_variables == 2
    assert d.max_index == 1


def test_qubo_addition():

    temp = QUBO({('0', '0'): 1, ('0', 1): 2})
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
    assert g == QUBO(temp3[0])*-1


def test_qubo_multiplication():

    temp = QUBO({('0', '0'): 1, ('0', 1): 2})
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

    # should raise a KeyError since can't fit this into QUBO.
    try:
        QUBO({('0', 1): 1, (1, 2): -1})**2
        assert False
    except KeyError:
        pass


def test_properties():

    temp = QUBO({('0', '0'): 1, ('0', 1): 2})
    temp.offset
    temp.mapping
    temp.reverse_mapping
