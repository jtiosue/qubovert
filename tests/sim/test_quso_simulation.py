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
Contains tests for the QUSOSimulation functionality in the
``qubovert.sim`` library.
"""

from qubovert.sim import QUSOSimulation
from qubovert import spin_var, QUSO
from qubovert.utils import QUSOMatrix
from numpy.testing import assert_raises


def test_qusosimulation_str():

    assert str(QUSOSimulation({})) == "QUSOSimulation"


def test_qusosimulation_set_state():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(9))

    sim = QUSOSimulation(ising)
    assert sim.state == {i: 1 for i in ising.variables}

    sim = QUSOSimulation(ising, {i: -1 for i in ising.variables})
    assert sim.state == {i: -1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        QUSOSimulation(ising, {i: 0 for i in ising.variables})

    sim = QUSOSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([0])

    sim.set_state([-1])
    assert sim.state == {0: -1}

    # test the same but with matrix
    ising = QUSOMatrix(
        sum(-spin_var(i) * spin_var(i+1) for i in range(9))
    )

    sim = QUSOSimulation(ising)
    assert sim.state == {i: 1 for i in ising.variables}

    sim = QUSOSimulation(ising, {i: -1 for i in ising.variables})
    assert sim.state == {i: -1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        QUSOSimulation(ising, {i: 0 for i in ising.variables})

    sim = QUSOSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([0])

    sim.set_state([-1])
    assert sim.state == {0: -1}

    # test the same for QUSO
    ising = QUSO(
        sum(-spin_var(i) * spin_var(i+1) for i in range(9))
    )
    sim = QUSOSimulation(ising)
    assert sim.state == {i: 1 for i in ising.variables}

    sim = QUSOSimulation(ising, {i: -1 for i in ising.variables})
    assert sim.state == {i: -1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        QUSOSimulation(ising, {i: 0 for i in ising.variables})

    sim = QUSOSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([0])

    sim.set_state([-1])
    assert sim.state == {0: -1}


def test_qusosimulation_reset():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    initial_state = {0: 1, 1: 1, 2: -1, 3: 1}
    sim = QUSOSimulation(ising, initial_state)

    sim.schedule_update([(2, 100)])
    sim.update(2, 2000)
    sim.reset()
    assert sim.state == initial_state == sim.initial_state

    # test the same thing but with matrix
    ising = QUSOMatrix(
        sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    )
    initial_state = {0: 1, 1: 1, 2: -1, 3: 1}
    sim = QUSOSimulation(ising, initial_state)

    sim.schedule_update([(2, 100)])
    sim.update(2, 2000)
    sim.reset()
    assert sim.state == initial_state == sim.initial_state

    # test the same thing but with QUSO
    ising = QUSO(
        sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    )
    initial_state = {0: 1, 1: 1, 2: -1, 3: 1}
    sim = QUSOSimulation(ising, initial_state)

    sim.schedule_update([(2, 100)])
    sim.update(2, 2000)
    sim.reset()
    assert sim.state == initial_state == sim.initial_state


def test_qusosimulation_initialstate_variables():

    ising = dict(sum(-spin_var(i) * spin_var(i+1) for i in range(3)))
    initial_state = {0: 1, 1: 1, 2: -1, 3: 1}
    sim = QUSOSimulation(ising, initial_state)
    assert sim._variables == set(initial_state.keys())


def test_qusosimulation_updates():

    sim = QUSOSimulation(spin_var(0))
    state = sim.state
    sim.update(5, 0)
    assert sim.state == state
    sim.schedule_update([(5, 0), (3, 0)])
    assert sim.state == state
    sim.update(4, -1)
    assert sim.state == state


def test_qusosimulation_bigrun():

    # test that it runs on a big problem
    model = QUSOMatrix(
        {(i, j): 1 for i in range(2, 200, 3) for j in range(2, 200, 2)}
    )
    sim = QUSOSimulation(model)
    sim.update(3, 1000)
    sim.update(3, 1000, in_order=True)
