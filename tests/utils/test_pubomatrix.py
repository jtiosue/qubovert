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
Contains tests for the PUBOMatrix class.
"""

from qubovert.utils import PUBOMatrix
from numpy import allclose


def test_pubo_default_valid():

    d = PUBOMatrix()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0,): 1}


def test_pubo_remove_value_when_zero():

    d = PUBOMatrix()
    d[(0, 0)] += 1
    d[(0, 0)] -= 1
    assert d == {}

    d.refresh()
    assert d.degree == 0
    assert d.num_binary_variables == 0
    assert d.variables == set()


def test_pubo_reinitialize_dictionary():

    d = PUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0, (0, 1): 1, (2, 0, 1): -2})
    assert d == {(0,): 1, (0, 1): 3, (0, 1, 2): -2}


def test_pubo_update():

    d = PUBOMatrix({(0, 0): 1, (0, 1): 2})
    d.update({(0,): 0, (1, 0): 1, (1, 1): -1, (2, 1, 1, 0): -3})
    assert d == {(0, 1): 1, (1,): -1, (0, 1, 2): -3}


def test_pubo_num_binary_variables():

    d = PUBOMatrix({(0,): 1, (0, 3): 2, (0, 3, 4): -1})
    assert d.num_binary_variables == 3


def test_pubo_max_index():

    d = PUBOMatrix({(0,): 1, (0, 3): 2, (0, 3, 4): -1})
    assert d.max_index == 4


def test_pubo_degree():

    d = PUBOMatrix()
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


def test_pubo_addition():

    temp = PUBOMatrix({(0, 0): 1, (0, 1): 2, (1, 0, 3): 2})
    temp1 = {(0,): -1, (1, 0): 3}
    temp2 = {(0, 1): 5, (0, 1, 3): 2}
    temp3 = {(0,): 2, (0, 1): -1, (0, 1, 3): 2}

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
    assert g == PUBOMatrix(temp3)*-1


def test_pubo_multiplication():

    temp = PUBOMatrix({(0, 0): 1, (0, 1): 2})
    temp[(2, 0, 1)] -= 4
    temp += 2

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g == {(0,): 3, (0, 1): 6, (): 6, (0, 1, 2): -12}

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d == {(0,): 3, (0, 1): 6, (): 6, (0, 1, 2): -12}

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g == {(0,): 3, (0, 1): 6, (): 6, (0, 1, 2): -12}

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g == {(0,): .5, (0, 1): 1, (): 1, (0, 1, 2): -2}

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d == {(0,): .5, (0, 1): 1, (): 1, (0, 1, 2): -2}

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g == {(0, 1): 1, (): 1, (0, 1, 2): -2}

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d == {(0, 1): 1, (): 1, (0, 1, 2): -2}

    # todo: add __mul__ test with dicts

    # __pow__
    assert temp ** 2 == temp * temp
    assert temp ** 3 == temp * temp * temp


def test_pubomatrix_solve_bruteforce():

    P = PUBOMatrix({
        (0, 1): 1, (1, 2): 1, (1, 1): -1, (2,): -2,
        (3, 4, 5): 1, (3,): 1, (4,): 1, (5,): 1
    })
    sol = P.solve_bruteforce()
    assert sol == {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0}
    assert allclose(P.value(sol), -2)

    P = PUBOMatrix({
        (0, 0): 1, (0, 1): -1, (): 1,
        (3, 4, 5): 1, (3,): 1, (4,): 1, (5,): 1
    })
    sols = P.solve_bruteforce(True)
    assert (
        sols
        ==
        [{0: 0, 1: 0, 3: 0, 4: 0, 5: 0},
         {0: 0, 1: 1, 3: 0, 4: 0, 5: 0},
         {0: 1, 1: 1, 3: 0, 4: 0, 5: 0}]
    )
    assert all(allclose(P.value(s), 1) for s in sols)
