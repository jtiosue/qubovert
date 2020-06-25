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
Contains tests for functions in the _values.py file.
"""

from qubovert.utils import (
    pubo_value, puso_value, qubo_value, quso_value,
    pubo_to_puso, qubo_to_quso, boolean_to_spin
)
from numpy.testing import assert_allclose
import random
import itertools


def test_pubo_qubo_equal():

    random.seed(123)
    qubo = {(i, j): random.random() for i in range(7) for j in range(7)}
    qubo.update({(i,): random.random() for i in range(7)})
    qubo[()] = random.random()
    for sol in itertools.product((0, 1), repeat=7):
        assert_allclose(pubo_value(sol, qubo), qubo_value(sol, qubo))


def test_puso_quso_equal():

    random.seed(321)
    quso = {(i, j): random.random() for i in range(7) for j in range(7)}
    quso.update({(i,): random.random() for i in range(7)})
    quso[()] = random.random()
    for sol in itertools.product((-1, 1), repeat=7):
        assert_allclose(puso_value(sol, quso), quso_value(sol, quso))


def test_pubo_puso_equal():

    random.seed(518)
    pubo = {
        (i, j, k): random.random()
        for i in range(7) for j in range(7) for k in range(7)
    }
    pubo.update({(i, j): random.random() for i in range(7) for j in range(7)})
    pubo.update({(i,): random.random() for i in range(7)})
    pubo[()] = random.random()
    for sol in itertools.product((0, 1), repeat=7):
        assert_allclose(
            pubo_value(sol, pubo),
            puso_value(boolean_to_spin(sol), pubo_to_puso(pubo))
        )


def test_qubo_quso_equal():

    random.seed(815)
    qubo = {(i, j): random.random() for i in range(7) for j in range(7)}
    qubo.update({(i,): random.random() for i in range(7)})
    qubo[()] = random.random()
    for sol in itertools.product((0, 1), repeat=7):
        assert_allclose(
            qubo_value(sol, qubo),
            quso_value(boolean_to_spin(sol), qubo_to_quso(qubo))
        )
