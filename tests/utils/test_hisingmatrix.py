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
Contains tests for the HIsingMatrix class.
"""

from qubovert.utils import HIsingMatrix
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises


def test_qubo_checkkey():

    with assert_raises(KeyError):
        HIsingMatrix({('a',): -1})

    with assert_raises(KeyError):
        HIsingMatrix({0: -1})


def test_hising_default_valid():

    d = HIsingMatrix()
    assert d[(0,)] == 0
    d[(0,)] += 1
    assert d == {(0,): 1}


def test_hising_remove_value_when_zero():

    d = HIsingMatrix()
    d[(0,)] += 1
    d[(0,)] -= 1
    assert d == {}

    d.refresh()
    assert d.degree == 0
    assert d.num_binary_variables == 0
    assert d.variables == set()


def test_hising_reinitialize_dictionary():

    d = HIsingMatrix({(0,): 1, (1, 0): 2, (2, 0): 0, (0, 1): 1, (2, 0, 1): 1})
    assert d == {(0,): 1, (0, 1): 3, (0, 1, 2): 1}


def test_hising_update():

    d = HIsingMatrix({(0,): 1, (0, 1): 2})
    d.update({(0,): 0, (1, 0): 1, (1,): -1, (0, 2, 1): -1})
    assert d == {(0, 1): 1, (1,): -1, (0, 1, 2): -1}


def test_hising_num_binary_variables():

    d = HIsingMatrix({(0,): 1, (0, 3): 2, (0, 4, 3): 3})
    assert d.num_binary_variables == 3


def test_hising_max_index():

    d = HIsingMatrix({(0,): 1, (0, 3): 2, (0, 4, 3): 3})
    assert d.max_index == 4


def test_hising_degree():

    d = HIsingMatrix()
    assert d.degree == 0
    d[(0,)] += 2
    assert d.degree == 1
    d[(1,)] -= 3
    assert d.degree == 1
    d[(1, 2)] -= 2
    assert d.degree == 2
    d[(0, 1, 2)] -= 1
    assert d.degree == 3
    d[(0, 1, 2, 4, 8)] -= 1
    assert d.degree == 5


def test_hising_addition():

    temp = HIsingMatrix({(0,): 1, (0, 1): 2, (2, 1, 0): -1})
    temp1 = {(0,): -1, (1, 0): 3}
    temp2 = {(0, 1): 5, (0, 1, 2): -1}
    temp3 = {(0,): 2, (0, 1): -1, (0, 1, 2): -1}

    # add constant
    d = temp.copy()
    d += 5
    d[()] -= 2
    d == {(0,): 1, (0, 1): 2, (): 3, (0, 1, 2): -1}

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
    assert g == HIsingMatrix(temp3)*-1


def test_hising_multiplication():

    temp = HIsingMatrix({(0,): 1, (0, 1): 2, (0, 2, 3): 4})
    temp += 2

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g == {(0,): 3, (0, 1): 6, (): 6, (0, 2, 3): 12}

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d == {(0,): 3, (0, 1): 6, (): 6, (0, 2, 3): 12}

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g == {(0,): 3, (0, 1): 6, (): 6, (0, 2, 3): 12}

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g == {(0,): .5, (0, 1): 1, (): 1, (0, 2, 3): 2}

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d == {(0,): .5, (0, 1): 1, (): 1, (0, 2, 3): 2}

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g == {(0, 1): 1, (): 1, (0, 2, 3): 2}

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d == {(0, 1): 1, (): 1, (0, 2, 3): 2}

    # __mul__ but with dict
    d = temp.copy()
    d *= {(1,): 2, (0,): -1}
    assert d == {(0, 1): 2, (): -1, (0,): 2, (1,): 2,
                 (0, 1, 2, 3): 8, (2, 3): -4}

    # __pow__
    d = temp.copy()
    d **= 2
    assert d == {(): 25, (1,): 4, (2, 3): 8, (0,): 4,
                 (1, 2, 3): 16, (0, 1): 8, (0, 2, 3): 16}

    temp = d.copy()
    assert d ** 3 == d * d * d

    temp = d.copy()
    assert d ** 4 == d * d * d * d


def test_round():

    d = HIsingMatrix({(0,): 3.456, (1,): -1.53456})

    assert round(d) == {(0,): 3, (1,): -2}
    assert round(d, 1) == {(0,): 3.5, (1,): -1.5}
    assert round(d, 2) == {(0,): 3.46, (1,): -1.53}
    assert round(d, 3) == {(0,): 3.456, (1,): -1.535}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = HIsingMatrix()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}


def test_hisingmatrix_solve_bruteforce():

    H = HIsingMatrix({
        (0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2,
        (3, 4, 5): -1, (3,): -1, (4,): -1, (5,): -1
    })
    sol = H.solve_bruteforce()
    assert sol in (
        {0: -1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        {0: 1, 1: -1, 2: 1, 3: 1, 4: 1, 5: 1},
    )
    assert allclose(H.value(sol), -7)

    H = HIsingMatrix({(0,): 0.25, (1,): -0.25, (0, 1): -0.25, (): 1.25,
                      (3, 4, 5): -1, (3,): -1, (4,): -1, (5,): -1})
    sols = H.solve_bruteforce(True)
    assert (
        sols
        ==
        [{0: -1, 1: -1, 3: 1, 4: 1, 5: 1},
         {0: -1, 1: 1, 3: 1, 4: 1, 5: 1},
         {0: 1, 1: 1, 3: 1, 4: 1, 5: 1}]
    )
    assert all(allclose(H.value(s), -3) for s in sols)
