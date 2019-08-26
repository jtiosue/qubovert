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
Contains tests for the QUBO/PUBO to/from Ising/HIsing functions.
"""

from qubovert.utils import (
    qubo_to_ising, ising_to_qubo, pubo_to_hising, hising_to_pubo,
    binary_to_spin, spin_to_binary, decimal_to_binary, decimal_to_spin,
    qubo_to_matrix, matrix_to_qubo
)
import numpy as np


def test_binary_to_spin():

    assert binary_to_spin(0) == -1
    assert binary_to_spin(1) == 1
    assert binary_to_spin((0, 1)) == (-1, 1)
    assert binary_to_spin([0, 1]) == [-1, 1]
    assert binary_to_spin({"a": 0, "b": 1}) == {"a": -1, "b": 1}


def test_spin_to_binary():

    assert spin_to_binary(-1) == 0
    assert spin_to_binary(1) == 1
    assert spin_to_binary((-1, 1)) == (0, 1)
    assert spin_to_binary([-1, 1]) == [0, 1]
    assert spin_to_binary({"a": -1, "b": 1}) == {"a": 0, "b": 1}


def test_qubo_to_ising_to_qubo():

    qubo = {(0,): 1, (0, 1): 1, (1,): -1, (1, 2): .2, (): -2, (2,): 1}

    assert qubo == ising_to_qubo(qubo_to_ising(qubo))


def test_ising_to_qubo_to_ising():

    ising = {(0, 1): -4, (0, 2): 3, (): -2, (0,): 1, (2,): -2}

    assert ising == qubo_to_ising(ising_to_qubo(ising))


def test_pubo_to_hising_to_pubo():

    pubo = {
        (0,): 1, (0, 1): 1, (1,): -1, (1, 2): .5, (): -2, (2,): 1,
        (0, 2, 3): -3, (0, 1, 2): -2
    }

    assert pubo == hising_to_pubo(pubo_to_hising(pubo))


def test_hising_to_pubo_to_hising():

    hising = {
        (0, 1): -4, (0, 2): 3, (): -2, (0,): 1, (2,): -2,
        (0, 1, 2): 3, (0, 2, 3): -1
    }

    assert hising == pubo_to_hising(hising_to_pubo(hising))


def test_qubo_to_ising_eq_pubo_to_hising():

    qubo = {(0,): 1, (0, 1): 1, (1,): -1, (1, 2): .2, (): -2, (2,): 1}

    assert qubo_to_ising(qubo) == pubo_to_hising(qubo)


def test_ising_to_qubo_eq_hising_to_pubo():

    ising = {(0, 1): -4, (0, 2): 3, (): -2, (0,): 1, (2,): -2}

    assert ising_to_qubo(ising) == hising_to_pubo(ising)


def test_decimal_to_binary():

    assert decimal_to_binary(10, 7) == (0, 0, 0, 1, 0, 1, 0)
    assert decimal_to_binary(10) == (1, 0, 1, 0)


def test_decimal_to_spin():

    assert decimal_to_spin(10, 7) == (-1, -1, -1, 1, -1, 1, -1)
    assert decimal_to_spin(10) == (1, -1, 1, -1)


def test_matrix_to_qubo():

    matrix, qubo = [[-3, 1], [-1, 2]], {(0,): -3, (1,): 2}
    assert matrix_to_qubo(matrix) == qubo


def test_qubo_to_matrix():

    matrix, qubo = [[-3, 1], [0, 2]], {(0, 0): -3, (0, 1): 1, (1, 1): 2}
    assert matrix == qubo_to_matrix(qubo, array=False)
    assert np.all(np.array(matrix) == qubo_to_matrix(qubo))

    matrix = [[-3, .5], [.5, 2]]
    assert matrix == qubo_to_matrix(qubo, array=False, symmetric=True)
    assert np.all(np.array(matrix) == qubo_to_matrix(qubo, symmetric=True))
