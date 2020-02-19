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
Contains tests for functions in the _approximate_extrema.py file.
"""

from qubovert.utils import (
    approximate_pubo_extrema, approximate_puso_extrema,
    approximate_qubo_extrema, approximate_quso_extrema
)


def test_pubo_extrema():

    P = {(0,): 1, (1,): 3, (2, 3, 4): -2, (): 9}
    assert approximate_pubo_extrema(P) == (7, 13)

    assert approximate_pubo_extrema({}) == (0, 0)
    assert approximate_pubo_extrema({(): 4}) == (4, 4)


def test_pubo_extrema():

    P = {(0,): 1, (1,): 3, (2, 3, 4): -2, (): 9}
    assert approximate_puso_extrema(P) == (3, 15)

    assert approximate_puso_extrema({}) == (0, 0)
    assert approximate_puso_extrema({(): 4}) == (4, 4)


def test_pubo_extrema():

    P = {(0,): 1, (1,): 3, (2,): -2, (): 9}
    assert approximate_qubo_extrema(P) == (7, 13)

    assert approximate_qubo_extrema({}) == (0, 0)
    assert approximate_qubo_extrema({(): 4}) == (4, 4)


def test_pubo_extrema():

    P = {(0,): 1, (1,): 3, (2,): -2, (): 9}
    assert approximate_quso_extrema(P) == (3, 15)

    assert approximate_quso_extrema({}) == (0, 0)
    assert approximate_quso_extrema({(): 4}) == (4, 4)
