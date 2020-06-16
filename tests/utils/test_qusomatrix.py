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
Contains tests for the QUSOMatrix object.
"""

from qubovert.utils import QUSOMatrix
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises


def test_pretty_str():

    def equal(expression, string):
        assert expression.pretty_str() == string
        assert QUSOMatrix.pretty_str(dict(expression)) == string

    z = [QUSOMatrix() + {(i,): 1} for i in range(3)]
    a, b = Symbol('a'), Symbol('b')

    equal(z[0], "z(0)")
    equal(-z[0], "-z(0)")
    equal(z[0] * 0, "")
    equal(2*z[0]*z[1] - 3*z[2], "2 z(0) z(1) - 3 z(2)")
    equal(0*z[0] + 1, "1")
    equal(0*z[0] - 1, "-1")
    equal(0*z[0] + a, "(a)")
    equal(0*z[0] + a * b, "(a*b)")
    equal((a+b)*(z[0]*z[1] - z[2]), "(a + b) z(0) z(1) + (-a - b) z(2)")
    equal(2*z[0]*z[1] - z[2], "2 z(0) z(1) - z(2)")
    equal(-z[2] + z[0]*z[1], "-z(2) + z(0) z(1)")
    equal(-2*z[2] + 2*z[0]*z[1], "-2 z(2) + 2 z(0) z(1)")


def test_checkkey():

    with assert_raises(KeyError):
        QUSOMatrix({('a',): -1})

    with assert_raises(KeyError):
        QUSOMatrix({0: -1})

    with assert_raises(KeyError):
        QUSOMatrix({(0, 1, 2): -1})


def test_properties():

    L = QUSOMatrix()
    L[(0,)] -= 1
    L[(0, 1)] += 1
    L += 2
    assert L.offset == 2
    assert L.h == {0: -1}
    assert L.J == {(0, 1): 1}


def test_quso_default_valid():

    d = QUSOMatrix()
    assert d[(0,)] == 0
    d[(0,)] += 1
    assert d == {(0,): 1}


def test_quso_remove_value_when_zero():

    d = QUSOMatrix()
    d[(0,)] += 1
    d[(0,)] -= 1
    assert d == {}

    d.refresh()
    assert d.degree == 0
    assert d.num_binary_variables == 0
    assert d.variables == set()


def test_quso_reinitialize_dictionary():

    d = QUSOMatrix({(0,): 1, (1, 0): 2, (2, 0): 0, (0, 1): 1})
    assert d == {(0,): 1, (0, 1): 3}


def test_quso_update():

    d = QUSOMatrix({(0,): 1, (0, 1): 2})
    d.update({(0,): 0, (1, 0): 1, (1,): -1})
    assert d == {(0, 1): 1, (1,): -1}


def test_quso_num_binary_variables():

    d = QUSOMatrix({(0,): 1, (0, 3): 2})
    assert d.num_binary_variables == 2


def test_num_terms():

    d = QUSOMatrix({(0,): 1, (0, 3): 2, (0, 2): -1})
    assert d.num_terms == len(d)


def test_quso_max_index():

    d = QUSOMatrix({(0,): 1, (0, 3): 2})
    assert d.max_index == 3


def test_quso_degree():

    d = QUSOMatrix()
    assert d.degree == 0
    d[(0,)] += 2
    assert d.degree == 1
    d[(1,)] -= 3
    assert d.degree == 1
    d[(1, 2)] -= 2
    assert d.degree == 2


def test_quso_addition():

    temp = QUSOMatrix({(0,): 1, (0, 1): 2})
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
    assert g == QUSOMatrix(temp3)*-1


def test_quso_multiplication():

    temp = QUSOMatrix({(0,): 1, (0, 1): 2})
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
    assert d == {(): -1, (0,): 2, (1,): 2, (0, 1): 2}

    # __pow__
    d = temp.copy()
    d **= 2
    assert d == {(): 9, (0,): 4, (1,): 4, (0, 1): 8}

    temp = d.copy()
    assert d ** 3 == d * d * d

    # should raise a KeyError since can't fit this into QUSO.
    with assert_raises(KeyError):
        QUSOMatrix({(0, 1): 1, (2, 3): -1})**2


def test_round():

    d = QUSOMatrix({(0,): 3.456, (1,): -1.53456})

    assert round(d) == {(0,): 3, (1,): -2}
    assert round(d, 1) == {(0,): 3.5, (1,): -1.5}
    assert round(d, 2) == {(0,): 3.46, (1,): -1.53}
    assert round(d, 3) == {(0,): 3.456, (1,): -1.535}


def test_normalize():

    temp = {(0,): 4, (1,): -2}
    d = QUSOMatrix(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}

    temp = {(0,): -4, (1,): 2}
    d = QUSOMatrix(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = QUSOMatrix()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}


def test_qusomatrix_solve_bruteforce():

    L = QUSOMatrix({(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2})
    sol = L.solve_bruteforce()
    assert sol in ({0: -1, 1: 1, 2: 1}, {0: 1, 1: -1, 2: 1})
    assert allclose(L.value(sol), -3)

    L = QUSOMatrix({(0,): 0.25, (1,): -0.25, (0, 1): -0.25, (): 1.25})
    sols = L.solve_bruteforce(True)
    assert sols == [{0: 1, 1: 1}, {0: -1, 1: 1}, {0: -1, 1: -1}]
    assert all(allclose(L.value(s), 1) for s in sols)
