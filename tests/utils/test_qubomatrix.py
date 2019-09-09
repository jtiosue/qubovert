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
Contains tests for the QUBOMatrix object.
"""

from qubovert.utils import QUBOMatrix
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises


def test_qubo_checkkey():

    with assert_raises(KeyError):
        QUBOMatrix({('a',): -1})

    with assert_raises(KeyError):
        QUBOMatrix({0: -1})

    with assert_raises(KeyError):
        QUBOMatrix({(0, 1, 2): -1})


def test_properties():

    Q = QUBOMatrix()
    Q[(0,)] -= 1
    Q[(0, 1)] += 1
    Q += 2
    assert Q.offset == 2
    assert Q.Q == {(0, 0): -1, (0, 1): 1}


def test_qubo_default_valid():

    d = QUBOMatrix()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0,): 1}


def test_qubo_remove_value_when_zero():

    d = QUBOMatrix()
    d[(0, 0)] += 1
    d[(0, 0)] -= 1
    assert d == {}

    d.refresh()
    assert d.degree == 0
    assert d.num_binary_variables == 0
    assert d.variables == set()


def test_qubo_reinitialize_dictionary():

    d = QUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0, (0, 1): 1})
    assert d == {(0,): 1, (0, 1): 3}


def test_qubo_update():

    d = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    d.update({(0,): 0, (1, 0): 1, (1, 1): -1})
    assert d == {(0, 1): 1, (1,): -1}


def test_qubo_num_binary_variables():

    d = QUBOMatrix({(0,): 1, (0, 3): 2})
    assert d.num_binary_variables == 2


def test_qubo_max_index():

    d = QUBOMatrix({(0, 0): 1, (0, 3): 2})
    assert d.max_index == 3


def test_qubo_degree():

    d = QUBOMatrix()
    assert d.degree == 0
    d[(0,)] += 2
    assert d.degree == 1
    d[(1,)] -= 3
    assert d.degree == 1
    d[(1, 2)] -= 2
    assert d.degree == 2


def test_qubo_addition():

    temp = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    temp1 = {(0,): -1, (1, 0): 3}
    temp2 = {(0, 1): 5}
    temp3 = {(0,): 2, (0, 1): -1}

    # add constant
    d = temp.copy()
    d += 5
    d[()] -= 2
    d == {(0,): 1, (0, 1): 2, (): 3}

    # __add__
    d = temp.copy()
    g = d + temp1
    assert g == temp2

    # __iadd__
    d = temp.copy()
    d += temp1
    assert d == temp2

    # __radd__
    d = temp.copy()
    g = temp1 + d
    assert g == temp2

    # __sub__
    d = temp.copy()
    g = d - temp1
    assert g == temp3

    # __isub__
    d = temp.copy()
    d -= temp1
    assert d == temp3

    # __rsub__
    d = temp.copy()
    g = temp1 - d
    assert g == QUBOMatrix(temp3)*-1


def test_qubo_multiplication():

    temp = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    temp += 2

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g == {(0,): 3, (0, 1): 6, (): 6}

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d == {(0,): 3, (0, 1): 6, (): 6}

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g == {(0,): 3, (0, 1): 6, (): 6}

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g == {(0,): .5, (0, 1): 1, (): 1}

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d == {(0,): .5, (0, 1): 1, (): 1}

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g == {(0, 1): 1, (): 1}

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d == {(0, 1): 1, (): 1}

    # __mul__ but with dict
    d = temp.copy()
    d *= {(1,): 2, (0,): -1}
    assert d == {(0,): -3, (0, 1): 4, (1,): 4}

    # __pow__
    d = temp.copy()
    d **= 2
    assert d == {(0,): 5, (0, 1): 16, (): 4}

    temp = d.copy()
    assert d ** 3 == d * d * d

    # should raise a KeyError since can't fit this into QUBO.
    with assert_raises(KeyError):
        QUBOMatrix({(0, 1): 1, (1, 2): -1})**2


def test_round():

    d = QUBOMatrix({(0,): 3.456, (1,): -1.53456})

    assert round(d) == {(0,): 3, (1,): -2}
    assert round(d, 1) == {(0,): 3.5, (1,): -1.5}
    assert round(d, 2) == {(0,): 3.46, (1,): -1.53}
    assert round(d, 3) == {(0,): 3.456, (1,): -1.535}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = QUBOMatrix()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}


def test_qubomatrix_solve_bruteforce():

    Q = QUBOMatrix({(0, 1): 1, (1, 2): 1, (1, 1): -1, (2,): -2})
    sol = Q.solve_bruteforce()
    assert sol == {0: 0, 1: 0, 2: 1}
    assert allclose(Q.value(sol), -2)

    Q = QUBOMatrix({(0, 0): 1, (0, 1): -1, (): 1})
    sols = Q.solve_bruteforce(True)
    assert sols == [{0: 0, 1: 0}, {0: 0, 1: 1}, {0: 1, 1: 1}]
    assert all(allclose(Q.value(s), 1) for s in sols)
