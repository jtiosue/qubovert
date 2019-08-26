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
Contains tests for the QUBOMatrix, IsingMatrix, PUBOMatrix, and HIsingMatrix
classes.
"""

from qubovert.utils import QUBOMatrix, IsingMatrix, PUBOMatrix, HIsingMatrix


# QUBO

def test_qubo_default_valid():

    d = QUBOMatrix()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0,): 1}


def test_qubo_remove_value_when_zero():

    d = QUBOMatrix()
    d[(0, 0)] += 1
    d[(0, 0)] -= 1
    assert d == {}


def test_qubo_reinitialize_dictionary():

    d = QUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0, (0, 1): 1})
    assert d == {(0,): 1, (0, 1): 3}


def test_qubo_update():

    d = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    d.update({(0,): 0, (1, 0): 1, (1, 1): -1})
    assert d == {(0, 1): 1, (1,): -1}


def test_qubo_num_binary_variables():

    d = QUBOMatrix({(0,): 1, (0, 3): 2})
    assert d.num_binary_variables == 2


def test_qubo_max_index():

    d = QUBOMatrix({(0, 0): 1, (0, 3): 2})
    assert d.max_index == 3


def test_qubo_addition():

    temp = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    temp1 = {(0,): -1, (1, 0): 3}
    temp2 = {(0, 1): 5}
    temp3 = {(0,): 2, (0, 1): -1}

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
    assert g == QUBOMatrix(temp3)*-1


def test_qubo_multiplication():

    temp = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    temp += 2

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g == {(0,): 3, (0, 1): 6, (): 6}

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d == {(0,): 3, (0, 1): 6, (): 6}

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g == {(0,): 3, (0, 1): 6, (): 6}

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g == {(0,): .5, (0, 1): 1, (): 1}

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d == {(0,): .5, (0, 1): 1, (): 1}

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g == {(0, 1): 1, (): 1}

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d == {(0, 1): 1, (): 1}

    # __mul__ but wiiht dict
    d = temp.copy()
    d *= {(1,): 2, (0,): -1}
    assert d == {(0,): -3, (0, 1): 4, (1,): 4}

    # __pow__
    d = temp.copy()
    d **= 2
    assert d == {(0,): 5, (0, 1): 16, (): 4}

    temp = d.copy()
    assert d ** 3 == d * d * d

    # should raise a KeyError since can't fit this into QUBO.
    try:
        QUBOMatrix({(0, 1): 1, (1, 2): -1})**2
        assert False
    except KeyError:
        pass


# Ising

def test_ising_default_valid():

    d = IsingMatrix()
    assert d[(0,)] == 0
    d[(0,)] += 1
    assert d == {(0,): 1}


def test_ising_remove_value_when_zero():

    d = IsingMatrix()
    d[(0,)] += 1
    d[(0,)] -= 1
    assert d == {}


def test_ising_reinitialize_dictionary():

    d = IsingMatrix({(0,): 1, (1, 0): 2, (2, 0): 0, (0, 1): 1})
    assert d == {(0,): 1, (0, 1): 3}


def test_ising_update():

    d = IsingMatrix({(0,): 1, (0, 1): 2})
    d.update({(0,): 0, (1, 0): 1, (1,): -1})
    assert d == {(0, 1): 1, (1,): -1}


def test_ising_num_binary_variables():

    d = IsingMatrix({(0,): 1, (0, 3): 2})
    assert d.num_binary_variables == 2


def test_ising_max_index():

    d = IsingMatrix({(0,): 1, (0, 3): 2})
    assert d.max_index == 3


def test_ising_addition():

    temp = IsingMatrix({(0,): 1, (0, 1): 2})
    temp1 = {(0,): -1, (1, 0): 3}
    temp2 = {(0, 1): 5}
    temp3 = {(0,): 2, (0, 1): -1}

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
    assert g == IsingMatrix(temp3)*-1


def test_ising_multiplication():

    temp = IsingMatrix({(0,): 1, (0, 1): 2})
    temp += 2

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g == {(0,): 3, (0, 1): 6, (): 6}

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d == {(0,): 3, (0, 1): 6, (): 6}

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g == {(0,): 3, (0, 1): 6, (): 6}

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g == {(0,): .5, (0, 1): 1, (): 1}

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d == {(0,): .5, (0, 1): 1, (): 1}

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g == {(0, 1): 1, (): 1}

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d == {(0, 1): 1, (): 1}


# PUBO

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


# HIsing

def test_hising_default_valid():

    d = HIsingMatrix()
    assert d[(0,)] == 0
    d[(0,)] += 1
    assert d == {(0,): 1}


def test_hising_remove_value_when_zero():

    d = HIsingMatrix()
    d[(0,)] += 1
    d[(0,)] -= 1
    assert d == {}


def test_hising_reinitialize_dictionary():

    d = HIsingMatrix({(0,): 1, (1, 0): 2, (2, 0): 0, (0, 1): 1, (2, 0, 1): 1})
    assert d == {(0,): 1, (0, 1): 3, (0, 1, 2): 1}


def test_hising_update():

    d = HIsingMatrix({(0,): 1, (0, 1): 2})
    d.update({(0,): 0, (1, 0): 1, (1,): -1, (0, 2, 1): -1})
    assert d == {(0, 1): 1, (1,): -1, (0, 1, 2): -1}


def test_hising_num_binary_variables():

    d = HIsingMatrix({(0,): 1, (0, 3): 2, (0, 4, 3): 3})
    assert d.num_binary_variables == 3


def test_hising_max_index():

    d = HIsingMatrix({(0,): 1, (0, 3): 2, (0, 4, 3): 3})
    assert d.max_index == 4


def test_hising_addition():

    temp = HIsingMatrix({(0,): 1, (0, 1): 2, (2, 1, 0): -1})
    temp1 = {(0,): -1, (1, 0): 3}
    temp2 = {(0, 1): 5, (0, 1, 2): -1}
    temp3 = {(0,): 2, (0, 1): -1, (0, 1, 2): -1}

    # add constant
    d = temp.copy()
    d += 5
    d[()] -= 2
    d == {(0,): 1, (0, 1): 2, (): 3, (0, 1, 2): -1}

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
    assert g == HIsingMatrix(temp3)*-1


def test_hising_multiplication():

    temp = HIsingMatrix({(0,): 1, (0, 1): 2, (0, 2, 3): 4})
    temp += 2

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g == {(0,): 3, (0, 1): 6, (): 6, (0, 2, 3): 12}

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d == {(0,): 3, (0, 1): 6, (): 6, (0, 2, 3): 12}

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g == {(0,): 3, (0, 1): 6, (): 6, (0, 2, 3): 12}

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g == {(0,): .5, (0, 1): 1, (): 1, (0, 2, 3): 2}

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d == {(0,): .5, (0, 1): 1, (): 1, (0, 2, 3): 2}

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g == {(0, 1): 1, (): 1, (0, 2, 3): 2}

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d == {(0, 1): 1, (): 1, (0, 2, 3): 2}
