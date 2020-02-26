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
Contains tests for the boolean and spin simulation functionality in the
``qubovert.sim`` library.
"""

from qubovert.sim import SpinSimulation, BooleanSimulation
from qubovert import spin_var
from qubovert.utils import boolean_to_spin, puso_to_pubo
from numpy.testing import assert_raises


# spin simulation

def test_spinsimulation_str():

    for memory in range(5):
        s = SpinSimulation({}, memory=memory)
        assert str(s) == "SpinSimulation(memory=%d)" % memory


def test_spinsimulation_set_state():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(9))

    sim = SpinSimulation(ising)
    assert sim.state == {i: 1 for i in ising.variables}

    sim = SpinSimulation(ising, {i: -1 for i in ising.variables})
    assert sim.state == {i: -1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        SpinSimulation(ising, {i: 0 for i in ising.variables})

    sim = SpinSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([0])

    sim.set_state([-1])
    assert sim.state == {0: -1}


def test_spinsimulation_flip_bit():

    sim = SpinSimulation({(0,): 1, (1,): -1}, [1, -1])
    sim._flip_bit(0)
    assert sim.state == {0: -1, 1: -1}
    sim._flip_bit(1)
    assert sim.state == {0: -1, 1: 1}


def test_spinsimulation_paststates_reset():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    initial_state = {0: 1, 1: 1, 2: -1, 3: 1}
    sim = SpinSimulation(ising, initial_state, 1000)

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


def test_spinsimulation_initialstate_variables():

    ising = dict(sum(-spin_var(i) * spin_var(i+1) for i in range(3)))
    initial_state = {0: 1, 1: 1, 2: -1, 3: 1}
    sim = SpinSimulation(ising, initial_state)
    assert sim._variables == list(initial_state.keys())


def test_spinsimulation_update_vs_updateschedule():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    sim = SpinSimulation(ising)

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


def test_spinsimulation_updates():

    sim = SpinSimulation(spin_var(0))
    state = sim.state
    sim.update(5, 0)
    assert sim.state == state
    sim.schedule_update([(5, 0), (3, 0)])
    assert sim.state == state

    with assert_raises(ValueError):
        sim.update(4, -1)


# boolean simulation

def test_booleansimulation_str():

    for memory in range(5):
        s = BooleanSimulation({}, memory=memory)
        assert str(s) == "BooleanSimulation(memory=%d)" % memory


def test_booleansimulation_set_state():

    ising = puso_to_pubo(sum(-spin_var(i) * spin_var(i+1) for i in range(9)))

    sim = BooleanSimulation(ising)
    assert sim.state == {i: 0 for i in ising.variables}

    sim = BooleanSimulation(ising, {i: 1 for i in ising.variables})
    assert sim.state == {i: 1 for i in ising.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in ising.variables})

    with assert_raises(ValueError):
        BooleanSimulation(ising, {i: -1 for i in ising.variables})

    sim = BooleanSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([-1])

    sim.set_state([1])
    assert sim.state == {0: 1}


def test_booleansimulation_flip_bit():

    sim = BooleanSimulation({(0,): 1, (1,): -1}, [0, 1])
    sim._flip_bit(0)
    assert sim.state == {0: 1, 1: 1}
    sim._flip_bit(1)
    assert sim.state == {0: 1, 1: 0}


def test_booleansimulation_paststates_reset():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    initial_state = {0: 0, 1: 0, 2: 1, 3: 0}
    sim = BooleanSimulation(ising, initial_state, 1000)

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


def test_booleansimulation_vs_spinsimulation():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(15))
    qubo = puso_to_pubo(ising)

    schedule = [(T, 20) for T in range(3, 0, -1)]

    spin = SpinSimulation(ising)
    boolean = BooleanSimulation(qubo)

    assert spin.initial_state == boolean_to_spin(boolean.initial_state)

    spin.schedule_update(schedule, seed=4)
    boolean.schedule_update(schedule, seed=4)
    assert spin.state == boolean_to_spin(boolean.state)

    initial_state = [0] * 8 + [1] * 8
    spin = SpinSimulation(ising, boolean_to_spin(initial_state))
    boolean = BooleanSimulation(qubo, initial_state)

    assert spin.initial_state == boolean_to_spin(boolean.initial_state)

    spin.schedule_update(schedule, seed=4)
    boolean.schedule_update(schedule, seed=4)
    assert spin.state == boolean_to_spin(boolean.state)
