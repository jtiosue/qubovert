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
Contains tests for the QUBOMatrix, IsingCoupling, and IsingField classes.
"""

from qubovert.utils import QUBOMatrix, IsingCoupling, IsingField


# QUBO

def test_qubo_default_valid():

    d = QUBOMatrix()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0, 0): 1}


def test_qubo_remove_value_when_zero():

    d = QUBOMatrix()
    d[(0, 0)] += 1
    d[(0, 0)] -= 1
    assert d == {}


def test_qubo_reinitialize_dictionary():

    d = QUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0, (0, 1): 1})
    assert d == {(0, 0): 1, (0, 1): 3}


def test_qubo_update():

    d = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    d.update({(0, 0): 0, (1, 0): 1, (1, 1): -1})
    assert d == {(0, 1): 1, (1, 1): -1}


def test_qubo_num_binary_variables():

    d = QUBOMatrix({(0, 0): 1, (0, 3): 2})
    assert d.num_binary_variables == 2


def test_qubo_max_index():

    d = QUBOMatrix({(0, 0): 1, (0, 3): 2})
    assert d.max_index == 3


def test_qubo_to_matrix():
    d = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    assert d.to_matrix(array=False) == [[1, 2], [0, 0]]


def test_qubo_addition():

    temp = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    temp1 = {(0, 0): -1, (1, 0): 3}
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
    assert g == QUBOMatrix(temp3)*-1


def test_qubo_multiplication():

    temp = QUBOMatrix({(0, 0): 1, (0, 1): 2})

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


# Ising

def test_ising_default_valid():

    J = IsingCoupling()
    assert J[(0, 1)] == 0
    J[(0, 1)] += 1
    assert J == {(0, 1): 1}

    h = IsingField()
    assert h[1] == 0
    h[1] += 1
    assert h == {1: 1}


def test_ising_remove_value_when_zero():

    J = IsingCoupling()
    J[(1, 2)] += 1
    J[(1, 2)] -= 1
    assert J == {}

    h = IsingField()
    h[0] += 1
    h[0] -= 1
    assert h == {}


def test_ising_reinitialize_dictionary():

    J = IsingCoupling({(0, 1): 1, (1, 0): 2, (2, 0): 0})
    assert J == {(0, 1): 3}

    h = IsingField({0: -2, 2: 0})
    assert h == {0: -2}


def test_ising_update():

    d = IsingCoupling({(0, 1): 1, (0, 2): 2})
    d.update({(1, 0): 0, (2, 1): 1})
    assert d == {(1, 2): 1, (0, 2): 2}

    d = IsingField({0: 1, 2: -2})
    d.update({0: 0, 1: 1})
    assert d == {1: 1, 2: -2}


def test_ising_num_binary_variables():

    d = IsingCoupling({(0, 1): 1, (0, 4): 2})
    assert d.num_binary_variables == 3

    d = IsingField({0: 1, 3: -2})
    assert d.num_binary_variables == 2


def test_ising_max_index():

    d = IsingCoupling({(0, 1): 1, (0, 4): 2})
    assert d.max_index == 4

    d = IsingField({0: 1, 3: -2})
    assert d.max_index == 3


def test_ising_to_matrix():

    d = IsingCoupling({(0, 1): 1, (1, 2): 2, (0, 2): -1})
    assert d.to_matrix(array=False) == [[0, 1, -1], [0, 0, 2], [0, 0, 0]]

    d = IsingField({0: -1, 1: 3})
    assert d.to_matrix(array=False) == [[-1], [3]]
    assert d.to_matrix(vector=True, array=False) == [-1, 3]


def test_ising_addition():

    # IsingCoupling
    temp = IsingCoupling({(0, 2): 1, (0, 1): 2})
    temp1 = {(0, 2): -1, (1, 0): 3}
    temp2 = {(0, 1): 5}
    temp3 = {(0, 2): 2, (0, 1): -1}

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
    assert g == IsingCoupling(temp3)*-1

    # IsingField
    temp = IsingField({0: 1, 1: 2})
    temp1 = {0: -1, 1: 3}
    temp2 = {1: 5}
    temp3 = {0: 2, 1: -1}

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
    assert g == IsingField(temp3)*-1


def test_ising_multiplication():

    # IsingCoupling
    temp = IsingCoupling({(0, 2): 1, (0, 1): 2})

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g == {(0, 2): 3, (0, 1): 6}

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d == {(0, 2): 3, (0, 1): 6}

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g == {(0, 2): 3, (0, 1): 6}

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g == {(0, 2): .5, (0, 1): 1}

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d == {(0, 2): .5, (0, 1): 1}

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g == {(0, 1): 1}

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d == {(0, 1): 1}

    # IsingField
    temp = IsingField({0: 1, 1: 2})

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g == {0: 3, 1: 6}

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d == {0: 3, 1: 6}

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g == {0: 3, 1: 6}

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g == {0: .5, 1: 1}

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d == {0: .5, 1: 1}

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g == {1: 1}

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d == {1: 1}
