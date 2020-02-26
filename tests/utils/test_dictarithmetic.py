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
Contains tests for the DictArithmetic class.
"""

from qubovert.utils import DictArithmetic
from sympy import Symbol
from numpy.testing import assert_raises


def test_name():

    d = DictArithmetic({(0,): 1, (1, 2): -1})
    assert d.name is None
    d.name = 'd'
    assert d.name == 'd'
    d.name = 1
    assert d.name == 1


def test_num_terms():

    d = DictArithmetic({(0,): 1, (0, 3): 2, (0, 2): -1})
    assert d.num_terms == len(d)


def test_dictarithmetic_default_valid():

    d = DictArithmetic()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0, 0): 1}


def test_dictarithmetic_remove_value_when_zero():

    d = DictArithmetic()
    d[(0, 0)] += 1
    d[(0, 0)] -= 1
    assert d == {}


def test_dictarithmetic_reinitialize_dictionary():

    d = DictArithmetic({(0, 0): 1, (2, 0): 0, (0, 1): 1})
    assert d == {(0, 0): 1, (0, 1): 1}


def test_dictarithmetic_update():

    d = DictArithmetic({(0, 0): 1, (0, 1): 2})
    d.update({(0, 0): 0, (0, 1): 1, (1, 1): -1})
    assert d == {(0, 1): 1, (1, 1): -1}


def test_dictarithmetic_addition():

    temp = DictArithmetic({(0, 0): 1, (0, 1): 2})
    temp1 = {(0, 0): -1, (0, 1): 3}
    temp2 = {(0, 1): 5}
    temp3 = {(0, 0): 2, (0, 1): -1}

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
    assert g == DictArithmetic(temp3)*-1


def test_dictarithmetic_multiplication():

    temp = DictArithmetic({(0, 0): 1, (0, 1): 2})

    # __pos__
    assert temp == +temp

    # __neg__
    assert -temp == {(0, 0): -1, (0, 1): -2}

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g == {(0, 0): 3, (0, 1): 6}

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d == {(0, 0): 3, (0, 1): 6}

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g == {(0, 0): 3, (0, 1): 6}

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g == {(0, 0): .5, (0, 1): 1}

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d == {(0, 0): .5, (0, 1): 1}

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g == {(0, 1): 1}

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d == {(0, 1): 1}

    # __mul__ with dicts
    d = temp.copy()
    d *= {1: 2, 3: 2, (0, 1): -1}
    assert d == {(0, 0, 1): 2, (0, 0, 3): 2, (0, 0, 0, 1): -1,
                 (0, 1, 1): 4, (0, 1, 3): 4, (0, 1, 0, 1): -2}

    # __pow__
    d = temp.copy()
    d **= 2
    assert d == {(0, 0, 0, 0): 1, (0, 0, 0, 1): 2,
                 (0, 1, 0, 0): 2, (0, 1, 0, 1): 4}

    assert temp ** 1 == temp

    d = temp.copy()
    assert d ** 3 == d * d * d

    # ___pow__to non integer power
    d = temp.copy()
    with assert_raises(ValueError):
        d ** .5


def test_dictarithmetic_round():

    d = DictArithmetic(a=3.456, b=-1.53456)

    assert round(d) == dict(a=3, b=-2)
    assert round(d, 1) == dict(a=3.5, b=-1.5)
    assert round(d, 2) == dict(a=3.46, b=-1.53)
    assert round(d, 3) == dict(a=3.456, b=-1.535)


def test_normalize():

    temp = {(0,): 4, (1,): -2}
    d = DictArithmetic(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}

    temp = {(0,): -4, (1,): 2}
    d = DictArithmetic(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}


def test_subgraph():

    G = DictArithmetic({(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2})

    assert G.subgraph({0, 2}, {1: 5}) == {(0,): -17, (0, 2): -1, (): 10}
    assert G.subgraph({0, 2}) == {(0, 2): -1, (0,): 3}
    assert G.subgraph({0, 1}, {2: -10}) == {(0, 1): -4, (0,): 13, (1,): 2}
    assert G.subgraph({0, 1}) == {(0, 1): -4, (0,): 3, (1,): 2}
    assert type(G.subgraph({0, 1})) == DictArithmetic


def test_subvalue():

    G = DictArithmetic({(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2})
    assert G.subvalue({2: -3}) == {(0, 1): -4, (0,): 6, (1,): 2, (): 2}
    assert type(G.subvalue({})) == DictArithmetic


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = DictArithmetic()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}

    # rounding when symbols are involved.
    round(d)
