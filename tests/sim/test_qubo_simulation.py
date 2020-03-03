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
Contains tests for the PUSOSimulation functionality in the
``qubovert.sim`` library.
"""

from qubovert.sim import QUSOSimulation, QUBOSimulation
from qubovert import spin_var, QUBO
from qubovert.utils import boolean_to_spin, quso_to_qubo, QUSOMatrix
from numpy.testing import assert_raises


def test_qubosimulation_str():

    assert str(QUBOSimulation({})) == "QUBOSimulation"


def test_qubosimulation_set_state():

    ising = quso_to_qubo(sum(-spin_var(i) * spin_var(i+1) for i in range(9)))

    sim = QUBOSimulation(ising)
    assert sim.state == {i: 0 for i in ising.variables}

    sim = QUBOSimulation(ising, {i: 1 for i in ising.variables})
    assert sim.state == {i: 1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        QUBOSimulation(ising, {i: -1 for i in ising.variables})

    sim = QUBOSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([-1])

    sim.set_state([1])
    assert sim.state == {0: 1}

    # test the same thing but wiht matrix
    ising = quso_to_qubo(QUSOMatrix(
        sum(-spin_var(i) * spin_var(i+1) for i in range(9))
    ))

    sim = QUBOSimulation(ising)
    assert sim.state == {i: 0 for i in ising.variables}

    sim = QUBOSimulation(ising, {i: 1 for i in ising.variables})
    assert sim.state == {i: 1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        QUBOSimulation(ising, {i: -1 for i in ising.variables})

    sim = QUBOSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([-1])

    sim.set_state([1])
    assert sim.state == {0: 1}

    # test the same thing but wiht QUBO
    ising = quso_to_qubo(QUBO(
        sum(-spin_var(i) * spin_var(i+1) for i in range(9))
    ))

    sim = QUBOSimulation(ising)
    assert sim.state == {i: 0 for i in ising.variables}

    sim = QUBOSimulation(ising, {i: 1 for i in ising.variables})
    assert sim.state == {i: 1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        QUBOSimulation(ising, {i: -1 for i in ising.variables})

    sim = QUBOSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([-1])

    sim.set_state([1])
    assert sim.state == {0: 1}


def test_qubosimulation_reset():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    initial_state = {0: 0, 1: 0, 2: 1, 3: 0}
    sim = QUBOSimulation(ising, initial_state)
    sim.schedule_update([(2, 100)])
    sim.update(2, 2000)
    sim.reset()
    assert sim.state == initial_state == sim.initial_state


def test_qubosimulation_vs_qusosimulation():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(15))
    qubo = quso_to_qubo(ising)

    schedule = [(T, 20) for T in range(3, 0, -1)]

    spin = QUSOSimulation(ising)
    boolean = QUBOSimulation(qubo)

    assert spin.initial_state == boolean_to_spin(boolean.initial_state)

    spin.schedule_update(schedule, seed=4)
    boolean.schedule_update(schedule, seed=4)
    assert spin.state == boolean_to_spin(boolean.state)

    initial_state = [0] * 8 + [1] * 8
    spin = QUSOSimulation(ising, boolean_to_spin(initial_state))
    boolean = QUBOSimulation(qubo, initial_state)

    assert spin.initial_state == boolean_to_spin(boolean.initial_state)

    spin.schedule_update(schedule, seed=4)
    boolean.schedule_update(schedule, seed=4)
    assert spin.state == boolean_to_spin(boolean.state)
