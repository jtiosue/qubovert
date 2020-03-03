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

from qubovert.sim import PUSOSimulation
from qubovert import spin_var
from numpy.testing import assert_raises


def test_pusosimulation_str():

    for memory in range(5):
        s = PUSOSimulation({}, memory=memory)
        assert str(s) == "PUSOSimulation(memory=%d)" % memory


def test_pusosimulation_set_state():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(9))

    sim = PUSOSimulation(ising)
    assert sim.state == {i: 1 for i in ising.variables}

    sim = PUSOSimulation(ising, {i: -1 for i in ising.variables})
    assert sim.state == {i: -1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        PUSOSimulation(ising, {i: 0 for i in ising.variables})

    sim = PUSOSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([0])

    sim.set_state([-1])
    assert sim.state == {0: -1}


def test_pusosimulation_flip_bit():

    sim = PUSOSimulation({(0,): 1, (1,): -1}, [1, -1])
    sim._flip_bit(0)
    assert sim.state == {0: -1, 1: -1}
    sim._flip_bit(1)
    assert sim.state == {0: -1, 1: 1}


def test_pusosimulation_paststates_reset():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    initial_state = {0: 1, 1: 1, 2: -1, 3: 1}
    sim = PUSOSimulation(ising, initial_state, 1000)

    assert sim.memory == 1000
    assert sim.get_past_states() == [sim.state]

    states = [sim.state]
    for _ in range(100):
        sim.update(2)
        states.append(sim.state)
    assert states == sim.get_past_states()
    assert states[-50:] == sim.get_past_states(50)

    sim.update(2, 2000)
    assert len(sim.get_past_states()) == 1000
    assert sim.get_past_states(1) == [sim.state]

    sim.reset()
    assert sim.state == initial_state == sim.initial_state
    assert sim.get_past_states() == [initial_state]


def test_pusosimulation_initialstate_variables():

    ising = dict(sum(-spin_var(i) * spin_var(i+1) for i in range(3)))
    initial_state = {0: 1, 1: 1, 2: -1, 3: 1}
    sim = PUSOSimulation(ising, initial_state)
    assert sim._variables == list(initial_state.keys())


def test_pusosimulation_update_vs_updateschedule():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    sim = PUSOSimulation(ising)

    sim.update(4, 100, seed=0)
    sim.update(2, 23)
    sim.update(5, 48)
    result1 = sim.state
    sim.reset()
    sim.schedule_update([(4, 100), (2, 23), (5, 48)], seed=0)
    result2 = sim.state
    sim.reset()
    sim.schedule_update([(4, 100), (2, 23), (5, 48)], seed=1)
    result3 = sim.state
    assert result1 == result2
    assert result1 != result3


def test_pusosimulation_updates():

    sim = PUSOSimulation(spin_var(0))
    state = sim.state
    sim.update(5, 0)
    assert sim.state == state
    sim.schedule_update([(5, 0), (3, 0)])
    assert sim.state == state

    with assert_raises(ValueError):
        sim.update(4, -1)
