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
Contains tests for the QUBO/PUBO to/from QUSO/PUSO functions.
"""

from qubovert.utils import (
    qubo_to_quso, quso_to_qubo, pubo_to_puso, puso_to_pubo,
    boolean_to_spin, spin_to_boolean, decimal_to_boolean, decimal_to_spin,
    qubo_to_matrix, matrix_to_qubo, boolean_to_decimal, spin_to_decimal,
    QUBOMatrix, QUSOMatrix, PUBOMatrix, PUSOMatrix
)
from qubovert import QUBO, QUSO, PUBO, PUSO
from sympy import Symbol
import numpy as np
from numpy.testing import assert_raises


def test_qubo_to_quso_to_qubo():

    qubo = {(0,): 1, (0, 1): 1, (1,): -1, (1, 2): .2, (): -2, (2,): 1}
    assert qubo == quso_to_qubo(qubo_to_quso(qubo))

    # type asserting
    assert type(qubo_to_quso(qubo)) == QUSO
    assert type(qubo_to_quso(QUBOMatrix(qubo))) == QUSOMatrix
    assert type(qubo_to_quso(QUBO(qubo))) == QUSO

    qubo = {
        ('0',): 1, ('0', 1): 1, (1,): -1, (1, '2'): .2, (): -2, ('2',): 1,
        (0, 0): 1
    }
    # need to reformat qubo so it is sorted with the same hash
    assert QUBO(qubo) == quso_to_qubo(qubo_to_quso(qubo))

    # type asserting
    assert type(qubo_to_quso(qubo)) == QUSO
    assert type(qubo_to_quso(QUBO(qubo))) == QUSO


def test_quso_to_qubo_to_quso():

    quso = {(0, 1): -4, (0, 2): 3, (): -2, (0,): 1, (2,): -2}
    assert quso == qubo_to_quso(quso_to_qubo(quso))

    # type asserting
    assert type(quso_to_qubo(quso)) == QUBO
    assert type(quso_to_qubo(QUSOMatrix(quso))) == QUBOMatrix
    assert type(quso_to_qubo(QUSO(quso))) == QUBO

    quso = {('0', 1): -4, ('0', '2'): 3, (): -2, ('0',): 1, ('2', '2'): -2}
    # need to reformat quso so it is sorted with the same hash and squashed key
    assert QUSO(quso) == qubo_to_quso(quso_to_qubo(quso))

    # type asserting
    assert type(quso_to_qubo(quso)) == QUBO
    assert type(quso_to_qubo(QUSO(quso))) == QUBO


def test_pubo_to_puso_to_pubo():

    pubo = {
        (0,): 1, (0, 1): 1, (1,): -1, (1, 2): .5, (): -2, (2,): 1,
        (0, 2, 3): -3, (0, 1, 2): -2
    }
    assert pubo == puso_to_pubo(pubo_to_puso(pubo))

    # type asserting
    assert type(pubo_to_puso(pubo)) == PUSO
    assert type(pubo_to_puso(PUBOMatrix(pubo))) == PUSOMatrix
    assert type(pubo_to_puso(PUBO(pubo))) == PUSO

    pubo = {
        ('0',): 1, ('0', 1): 1, (1,): -1, (1, '2'): .5, (): -2, ('2',): 1,
        ('0', '2', 3): -3, ('0', 1, '2'): -2, ('0', '0', 1, '0', '2', '2'): -9,
        (4, 2, 4, 0, 2, 0, 0): 3
    }
    # need to reformat pubo so it is sorted with the same hash and squashed key
    assert PUBO(pubo) == puso_to_pubo(pubo_to_puso(pubo))

    # type asserting
    assert type(pubo_to_puso(pubo)) == PUSO
    assert type(pubo_to_puso(PUBO(pubo))) == PUSO


def test_puso_to_pubo_to_puso():

    puso = {
        (0, 1): -4, (0, 2): 3, (): -2, (0,): 1, (2,): -2,
        (0, 1, 2): 3, (0, 2, 3): -1
    }
    assert puso == pubo_to_puso(puso_to_pubo(puso))

    # type asserting
    assert type(puso_to_pubo(puso)) == PUBO
    assert type(puso_to_pubo(PUSOMatrix(puso))) == PUBOMatrix
    assert type(puso_to_pubo(PUSO(puso))) == PUBO

    puso = {
        ('0', 1): -4, ('0', '2'): 3, (): -2, ('0',): 1, ('2',): -2,
        ('0', 1, '2'): 3, ('0', '2', 3): -1,
        ('2', 0, 0, '1', 0): -2, (0, 1, 1, 0, 3, 0, 1, 1, 3, 2, 3): -8
    }
    # need to reformat qubo so it is sorted with the same hash
    assert PUSO(puso) == pubo_to_puso(puso_to_pubo(puso))

    # type asserting
    assert type(puso_to_pubo(puso)) == PUBO
    assert type(puso_to_pubo(PUSO(puso))) == PUBO


def test_qubo_to_quso_eq_pubo_to_puso():

    qubo = {(0,): 1, (0, 1): 1, (1,): -1, (1, 2): .2, (): -2, (2,): 1}
    assert qubo_to_quso(qubo) == pubo_to_puso(qubo)


def test_quso_to_qubo_eq_puso_to_pubo():

    quso = {(0, 1): -4, (0, 2): 3, (): -2, (0,): 1, (2,): -2}
    assert quso_to_qubo(quso) == puso_to_pubo(quso)


def test_decimal_to_boolean():

    assert decimal_to_boolean(10, 7) == (0, 0, 0, 1, 0, 1, 0)
    assert decimal_to_boolean(10) == (1, 0, 1, 0)

    with assert_raises(ValueError):
        decimal_to_boolean(.5)

    with assert_raises(ValueError):
        decimal_to_boolean(1000, 2)


def test_decimal_to_spin():

    assert decimal_to_spin(10, 7) == (1, 1, 1, -1, 1, -1, 1)
    assert decimal_to_spin(10) == (-1, 1, -1, 1)


def test_boolean_to_decimal():

    for i in range(8):
        assert i == boolean_to_decimal(decimal_to_boolean(i))


def test_spin_to_decimal():

    for i in range(8):
        assert i == spin_to_decimal(decimal_to_spin(i))


def test_boolean_to_spin():

    assert boolean_to_spin(0) == 1
    assert boolean_to_spin(1) == -1
    assert boolean_to_spin((0, 1)) == (1, -1)
    assert boolean_to_spin([0, 1]) == [1, -1]
    assert boolean_to_spin({"a": 0, "b": 1}) == {"a": 1, "b": -1}


def test_spin_to_boolean():

    assert spin_to_boolean(-1) == 1
    assert spin_to_boolean(1) == 0
    assert spin_to_boolean((-1, 1)) == (1, 0)
    assert spin_to_boolean([-1, 1]) == [1, 0]
    assert spin_to_boolean({"a": -1, "b": 1}) == {"a": 1, "b": 0}


def test_matrix_to_qubo():

    matrix, qubo = [[-3, 1], [-1, 2]], {(0,): -3, (1,): 2}
    assert matrix_to_qubo(matrix) == qubo

    matrix, qubo = np.array([[-3, 1], [-1, 2]]), {(0,): -3, (1,): 2}
    assert matrix_to_qubo(matrix) == qubo

    matrix, qubo = [[-3, 1], [-1, 2]], QUBOMatrix({(0,): -3, (1,): 2})
    assert matrix_to_qubo(matrix) == qubo

    with assert_raises(ValueError):
        matrix_to_qubo([[1, 2, 3], [1, 0, 1]])


def test_qubo_to_matrix():

    matrix, qubo = [[-3, 1], [0, 2]], {(0, 0): -3, (0, 1): 1, (1, 1): 2}
    assert matrix == qubo_to_matrix(qubo, array=False)
    assert np.all(np.array(matrix) == qubo_to_matrix(qubo))

    matrix = [[-3, .5], [.5, 2]]
    assert matrix == qubo_to_matrix(qubo, array=False, symmetric=True)
    assert np.all(np.array(matrix) == qubo_to_matrix(qubo, symmetric=True))

    matrix = [[-3, 1], [0, 2]]
    qubo = QUBOMatrix({(0, 0): -3, (0, 1): 1, (1, 1): 2})
    assert matrix == qubo_to_matrix(qubo, array=False)
    assert np.all(np.array(matrix) == qubo_to_matrix(qubo))

    matrix = [[-3, .5], [.5, 2]]
    assert matrix == qubo_to_matrix(qubo, array=False, symmetric=True)
    assert np.all(np.array(matrix) == qubo_to_matrix(qubo, symmetric=True))

    with assert_raises(ValueError):
        qubo_to_matrix({})

    with assert_raises(ValueError):
        qubo_to_matrix({(): 1, (0,): -1})


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    quso = {(0,): 1.0*a, (0, 1): 1., (1,): -1.0*a,
            (1, 2): 1., (): -2.*b, (2,): 1.0*a}
    quso1 = qubo_to_quso(quso_to_qubo(quso))
    quso1.simplify()
    quso = QUSO(quso)
    quso.simplify()
    assert quso == quso1

    a, b = Symbol('a'), Symbol('b')
    qubo = {(0,): 1.0*a, (0, 1): 1., (1,): -1.0*a,
            (1, 2): 1., (): -2.0*b, (2,): 1.0*a}
    qubo1 = quso_to_qubo(qubo_to_quso(qubo))
    qubo1.simplify()
    qubo = QUBO(qubo)
    qubo.simplify()
    assert qubo == qubo1
