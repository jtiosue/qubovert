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
Contains tests for the subgraph function.
"""

from qubovert.utils import subgraph, subvalue
from sympy import Symbol
from qubovert.utils import QUBOMatrix, PUBOMatrix, QUSOMatrix, PUSOMatrix
from qubovert import QUBO, PUBO, QUSO, PUSO, PCBO, PCSO
from numpy.testing import assert_raises


def test_subgraph():

    G = {(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2}

    assert subgraph(G, {0, 2}, {1: 5}) == {(0,): -17, (0, 2): -1, (): 10}
    assert subgraph(G, {0, 2}) == {(0, 2): -1, (0,): 3}
    assert subgraph(G, {0, 1}, {2: -10}) == {(0, 1): -4, (0,): 13, (1,): 2}
    assert subgraph(G, {0, 1}) == {(0, 1): -4, (0,): 3, (1,): 2}

    a = Symbol('a')

    for t in (QUBO, PUBO, QUSO, PUSO, PCBO, PCSO,
              QUBOMatrix, PUBOMatrix, QUSOMatrix, PUSOMatrix):
        S = subgraph(t(G), {0, 1}, {2: a})
        assert type(S) == t
        assert S == {(0, 1): -4, (0,): 3-a, (1,): 2}
        assert S.subs(a, -10) == {(0, 1): -4, (0,): 13, (1,): 2}

    with assert_raises(ValueError):
        subgraph({0: 1, (0, 1): 1}, {})


def test_subvalue():

    G = {(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2}
    assert subvalue({2: -3}, G) == {(0, 1): -4, (0,): 6, (1,): 2, (): 2}

    a = Symbol('a')

    for t in (QUBO, PUBO, QUSO, PUSO, PCBO, PCSO,
              QUBOMatrix, PUBOMatrix, QUSOMatrix, PUSOMatrix):
        S = subvalue({2: a}, t(G))
        assert type(S) == t
        assert S == {(0, 1): -4, (0,): 3-a, (1,): 2, (): 2}
        assert S.subs(a, -10) == {(0, 1): -4, (0,): 13, (1,): 2, (): 2}

    with assert_raises(ValueError):
        subvalue({}, {0: 1, (0, 1): 1})

    # edge case
    assert subvalue({}, G) == G
    assert subvalue({0: 0, 1: 0, 2: 0}, G) == {(): 2}
