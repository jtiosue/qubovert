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
Contains tests for the PUBOSimulation functionality in the
``qubovert.sim`` library.
"""

from qubovert.sim import PUSOSimulation, PUBOSimulation
from qubovert import spin_var
from qubovert.utils import boolean_to_spin, puso_to_pubo
from numpy.testing import assert_raises


def test_pubosimulation_str():

    for memory in range(5):
        s = PUBOSimulation({}, memory=memory)
        assert str(s) == "PUBOSimulation(memory=%d)" % memory


def test_pubosimulation_set_state():

    ising = puso_to_pubo(sum(-spin_var(i) * spin_var(i+1) for i in range(9)))

    sim = PUBOSimulation(ising)
    assert sim.state == {i: 0 for i in ising.variables}

    sim = PUBOSimulation(ising, {i: 1 for i in ising.variables})
    assert sim.state == {i: 1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        PUBOSimulation(ising, {i: -1 for i in ising.variables})

    sim = PUBOSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([-1])

    sim.set_state([1])
    assert sim.state == {0: 1}


def test_pubosimulation_paststates_reset():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    initial_state = {0: 0, 1: 0, 2: 1, 3: 0}
    sim = PUBOSimulation(ising, initial_state, 1000)

    assert sim.memory == 1000
    assert sim.get_past_states() == [sim.state]

    states = [sim.state]
    for _ in range(100):
        sim.update(2)
        states.append(sim.state)
    assert states == sim.get_past_states()
    assert states[-50:] == sim.get_past_states(50)
    assert sim.get_past_states(1) == [sim.state]

    sim.update(2, 2000)
    assert len(sim.get_past_states()) == 1000

    sim.reset()
    assert sim.state == initial_state == sim.initial_state
    assert sim.get_past_states() == [initial_state]


def test_pubosimulation_vs_pusosimulation():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(15))
    qubo = puso_to_pubo(ising)

    schedule = [(T, 20) for T in range(3, 0, -1)]

    spin = PUSOSimulation(ising)
    boolean = PUBOSimulation(qubo)

    assert spin.initial_state == boolean_to_spin(boolean.initial_state)

    spin.schedule_update(schedule, seed=4)
    boolean.schedule_update(schedule, seed=4)
    assert spin.state == boolean_to_spin(boolean.state)

    initial_state = [0] * 8 + [1] * 8
    spin = PUSOSimulation(ising, boolean_to_spin(initial_state))
    boolean = PUBOSimulation(qubo, initial_state)

    assert spin.initial_state == boolean_to_spin(boolean.initial_state)

    spin.schedule_update(schedule, seed=4)
    boolean.schedule_update(schedule, seed=4)
    assert spin.state == boolean_to_spin(boolean.state)
