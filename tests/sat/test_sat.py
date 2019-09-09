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

from qubovert import PUBO
from qubovert.sat import ONE, NOT, AND, NAND, OR, NOR, XOR, XNOR
from qubovert.utils import decimal_to_binary


def test_sat_one():

    assert ONE('x') == {('x',): 1}
    assert ONE({('x', 'y'): 1}) == PUBO({('x', 'y'): 1})


def test_sat_not():

    assert NOT('x') == {(): 1, ('x',): -1}
    assert NOT({('x', 'y'): 1}) == 1 - PUBO({('x', 'y'): 1})


def test_sat_and():

    assert AND() == {(): 1}
    assert AND('x', 'y') == PUBO({('x', 'y'): 1})
    assert AND({('x', 'y'): 1}, 'a') == PUBO({('x', 'y', 'a'): 1})

    for n in range(1, 5):
        P = AND(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_binary(i, n)
            if all(sol):
                assert P.value(sol) == 1
            else:
                assert not P.value(sol)


def test_sat_nand():

    assert NAND() == {}
    assert NAND('x', 'y') == PUBO({(): 1, ('x', 'y'): -1})
    assert NAND({('x', 'y'): 1}, 'a') == PUBO({(): 1, ('x', 'y', 'a'): -1})

    for n in range(1, 5):
        P = NAND(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_binary(i, n)
            if all(sol):
                assert not P.value(sol)
            else:
                assert P.value(sol) == 1


def test_sat_or():

    assert OR() == {(): 1}
    assert OR('x', 'y') == PUBO({('x',): 1, ('y',): 1, ('x', 'y'): -1})
    assert (
        OR({('x', 'y'): 1}, 'a') ==
        PUBO({('x', 'y'): 1, ('a',): 1, ('x', 'y', 'a'): -1})
    )

    for n in range(1, 5):
        P = OR(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_binary(i, n)
            if any(sol):
                assert P.value(sol) == 1
            else:
                assert not P.value(sol)


def test_sat_nor():

    assert NOR() == {}
    assert (
        NOR('x', 'y') == PUBO({(): 1, ('x',): -1, ('y',): -1, ('x', 'y'): 1})
    )
    assert (
        NOR({('x', 'y'): 1}, 'a') ==
        PUBO({(): 1, ('x', 'y'): -1, ('a',): -1, ('x', 'y', 'a'): 1})
    )

    for n in range(1, 5):
        P = NOR(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_binary(i, n)
            if any(sol):
                assert not P.value(sol)
            else:
                assert P.value(sol) == 1


def test_sat_xor():

    assert XOR() == {(): 1}
    assert XOR('x', 'y') == PUBO({('x',): 1, ('y',): 1, ('x', 'y'): -2})
    assert (
        XOR({('x', 'y'): 1}, 'a') ==
        PUBO({('x', 'y'): 1, ('a',): 1, ('x', 'y', 'a'): -2})
    )

    for n in range(1, 5):
        P = XOR(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_binary(i, n)
            if sum(sol) % 2 == 1:
                assert P.value(sol) == 1
            else:
                assert not P.value(sol)


def test_sat_xnor():

    assert XNOR() == {}
    assert (
        XNOR('x', 'y') == PUBO({(): 1, ('x',): -1, ('y',): -1, ('x', 'y'): 2})
    )
    assert (
        XNOR({('x', 'y'): 1}, 'a') ==
        PUBO({(): 1, ('x', 'y'): -1, ('a',): -1, ('x', 'y', 'a'): 2})
    )

    for n in range(1, 5):
        P = XNOR(*tuple(range(n)))
        for i in range(1 << n):
            sol = decimal_to_binary(i, n)
            if sum(sol) % 2 == 1:
                assert not P.value(sol)
            else:
                assert P.value(sol) == 1
