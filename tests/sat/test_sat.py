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
Contains tests for the ``qubovert.sat`` library.
"""

from qubovert import PUBO, boolean_var, BOOLEAN_MODELS
from qubovert.sat import ONE, NOT, AND, NAND, OR, NOR, XOR, XNOR
from qubovert.utils import decimal_to_boolean


def test_sat_one():

    f = ONE

    assert f('x') == {('x',): 1}
    assert f({('x', 'y'): 1}) == PUBO({('x', 'y'): 1})

    # testing type
    x = boolean_var(0)
    for model in BOOLEAN_MODELS:
        assert isinstance(f(model(x)), model)


def test_sat_not():

    f = NOT

    assert f('x') == {(): 1, ('x',): -1}
    assert f({('x', 'y'): 1}) == 1 - PUBO({('x', 'y'): 1})

    # testing type
    x = boolean_var(0)
    for model in BOOLEAN_MODELS:
        assert isinstance(f(model(x)), model)


def test_sat_and():

    f = AND

    assert f() == {(): 1}
    assert f('x', 'y') == PUBO({('x', 'y'): 1})
    assert f({('x', 'y'): 1}, 'a') == PUBO({('x', 'y', 'a'): 1})

    for n in range(1, 5):
        P = f(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_boolean(i, n)
            if all(sol):
                assert P.value(sol) == 1
            else:
                assert not P.value(sol)

    # testing type
    x, y = boolean_var(0), boolean_var(1)
    for model in BOOLEAN_MODELS:
        assert isinstance(f(model(x), y), model)


def test_sat_nand():

    f = NAND

    assert f() == {}
    assert f('x', 'y') == PUBO({(): 1, ('x', 'y'): -1})
    assert f({('x', 'y'): 1}, 'a') == PUBO({(): 1, ('x', 'y', 'a'): -1})

    for n in range(1, 5):
        P = f(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_boolean(i, n)
            if all(sol):
                assert not P.value(sol)
            else:
                assert P.value(sol) == 1

    # testing type
    x, y = boolean_var(0), boolean_var(1)
    for model in BOOLEAN_MODELS:
        assert isinstance(f(model(x), y), model)


def test_sat_or():

    f = OR

    assert f() == {(): 1}
    assert f('x', 'y') == PUBO({('x',): 1, ('y',): 1, ('x', 'y'): -1})
    assert (
        f({('x', 'y'): 1}, 'a') ==
        PUBO({('x', 'y'): 1, ('a',): 1, ('x', 'y', 'a'): -1})
    )

    for n in range(1, 5):
        P = f(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_boolean(i, n)
            if any(sol):
                assert P.value(sol) == 1
            else:
                assert not P.value(sol)

    # testing type
    x, y = boolean_var(0), boolean_var(1)
    for model in BOOLEAN_MODELS:
        assert isinstance(f(model(x), y), model)


def test_sat_nor():

    f = NOR

    assert f() == {}
    assert (
        f('x', 'y') == PUBO({(): 1, ('x',): -1, ('y',): -1, ('x', 'y'): 1})
    )
    assert (
        f({('x', 'y'): 1}, 'a') ==
        PUBO({(): 1, ('x', 'y'): -1, ('a',): -1, ('x', 'y', 'a'): 1})
    )

    for n in range(1, 5):
        P = f(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_boolean(i, n)
            if any(sol):
                assert not P.value(sol)
            else:
                assert P.value(sol) == 1

    # testing type
    x, y = boolean_var(0), boolean_var(1)
    for model in BOOLEAN_MODELS:
        assert isinstance(f(model(x), y), model)


def test_sat_xor():

    f = XOR

    assert f() == {(): 1}
    assert f('x', 'y') == PUBO({('x',): 1, ('y',): 1, ('x', 'y'): -2})
    assert (
        f({('x', 'y'): 1}, 'a') ==
        PUBO({('x', 'y'): 1, ('a',): 1, ('x', 'y', 'a'): -2})
    )

    for n in range(1, 5):
        P = f(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_boolean(i, n)
            if sum(sol) % 2 == 1:
                assert P.value(sol) == 1
            else:
                assert not P.value(sol)

    # testing type
    x, y = boolean_var(0), boolean_var(1)
    for model in BOOLEAN_MODELS:
        assert isinstance(f(model(x), y), model)


def test_sat_xnor():

    f = XNOR

    assert f() == {}
    assert (
        f('x', 'y') == PUBO({(): 1, ('x',): -1, ('y',): -1, ('x', 'y'): 2})
    )
    assert (
        f({('x', 'y'): 1}, 'a') ==
        PUBO({(): 1, ('x', 'y'): -1, ('a',): -1, ('x', 'y', 'a'): 2})
    )

    for n in range(1, 5):
        P = f(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_boolean(i, n)
            if sum(sol) % 2 == 1:
                assert not P.value(sol)
            else:
                assert P.value(sol) == 1

    # testing type
    x, y = boolean_var(0), boolean_var(1)
    for model in BOOLEAN_MODELS:
        assert isinstance(f(model(x), y), model)
