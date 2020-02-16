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
from qubovert import spin_var, boolean_var
from qubovert.sat import NOT
from numpy.testing import assert_raises


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


def test_booleansimulation_set_state():

    model = sum(-boolean_var(i) * NOT(boolean_var(i+1)) for i in range(9))

    sim = BooleanSimulation(model)
    assert sim.state == {i: 0 for i in model.variables}

    sim = BooleanSimulation(model, {i: 1 for i in model.variables})
    assert sim.state == {i: 1 for i in model.variables}

    with assert_raises(ValueError):
        sim.set_state({i: 3 for i in model.variables})

    with assert_raises(ValueError):
        BooleanSimulation(model, {i: -1 for i in model.variables})

    sim = BooleanSimulation({(0,): 1})
    with assert_raises(ValueError):
        sim.set_state([-1])

    sim.set_state([1])
    assert sim.state == {0: 1}


def test_spinsimulation_flip_bit():

    sim = SpinSimulation({(0,): 1, (1,): -1}, [1, -1])
    sim._flip_bit(0)
    assert sim.state == {0: -1, 1: -1}
    sim._flip_bit(1)
    assert sim.state == {0: -1, 1: 1}


def test_booleansimulation_flip_bit():

    sim = BooleanSimulation({(0,): 1, (1,): -1}, [0, 1])
    sim._flip_bit(0)
    assert sim.state == {0: 1, 1: 1}
    sim._flip_bit(1)
    assert sim.state == {0: 1, 1: 0}


def test_spinsimulation_paststates_reset():

    ising = sum(-spin_var(i) * spin_var(i+1) for i in range(3))
    initial_state = {0: 1, 1: 1, 2: -1, 3: 1}
    sim = SpinSimulation(ising, initial_state)

    assert sim.get_past_states() == [sim.state]

    states = [sim.state]
    for _ in range(100):
        sim.update(2)
        states.append(sim.state)
    assert states == sim.get_past_states()
    assert states[-50:] == sim.get_past_states(50)

    sim.update(2, 2000)
    assert len(sim.get_past_states()) == 1000

    sim.reset()
    assert sim.state == initial_state
    assert sim.get_past_states() == [initial_state]


def test_booleansimulation_paststates_reset():

    model = sum(-boolean_var(i) * NOT(boolean_var(i+1)) for i in range(3))
    initial_state = {0: 0, 1: 0, 2: 1, 3: 0}
    sim = BooleanSimulation(model, initial_state)

    assert sim.get_past_states() == [sim.state]

    states = [sim.state]
    for _ in range(100):
        sim.update(2)
        states.append(sim.state)
    assert states == sim.get_past_states()
    assert states[-50:] == sim.get_past_states(50)

    sim.update(2, 2000)
    assert len(sim.get_past_states()) == 1000

    sim.reset()
    assert sim.state == initial_state
    assert sim.get_past_states() == [initial_state]


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
    sim.schedule_update([(4, 100), (2, 23), (5, 48)], seed=2)
    result3 = sim.state
    assert result1 == result2
    assert result1 != result3


def test_booleansimulation_update_vs_updateschedule():

    model = sum(-boolean_var(i) * NOT(boolean_var(i+1)) for i in range(3))
    sim = BooleanSimulation(model)

    sim.update(4, 100, seed=0)
    sim.update(2, 23)
    sim.update(5, 48)
    result1 = sim.state
    sim.reset()
    sim.schedule_update([(4, 100), (2, 23), (5, 48)], seed=0)
    result2 = sim.state
    sim.reset()
    sim.schedule_update([(4, 100), (2, 23), (5, 48)], seed=2)
    result3 = sim.state
    assert result1 == result2
    assert result1 != result3
