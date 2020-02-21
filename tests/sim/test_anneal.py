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
Contains tests for the functions in the ``qubovert.sim._anneal`` file.
"""

from qubovert.sim import (
    anneal_qubo, anneal_quso, anneal_pubo, anneal_puso,
    anneal_temperature_range, AnnealResults
)
from qubovert.utils import puso_to_pubo, quso_to_qubo, QUBOVertWarning
from numpy.testing import assert_raises, assert_allclose, assert_warns
import numpy as np


def test_temperature_range():

    with assert_raises(ValueError):
        anneal_temperature_range({}, end_flip_prob=-.3)
    with assert_raises(ValueError):
        anneal_temperature_range({}, end_flip_prob=2)
    with assert_raises(ValueError):
        anneal_temperature_range({}, end_flip_prob=1)
    with assert_raises(ValueError):
        anneal_temperature_range({}, start_flip_prob=-.3)
    with assert_raises(ValueError):
        anneal_temperature_range({}, start_flip_prob=2)
    with assert_raises(ValueError):
        anneal_temperature_range({}, start_flip_prob=1)
    with assert_raises(ValueError):
        anneal_temperature_range({}, .3, .9)

    assert anneal_temperature_range({}) == (0, 0)
    assert anneal_temperature_range({(): 3}) == (0, 0)

    H = {(0, 1, 2): 2, (3,): -1, (4, 5): 5, (): -2}
    probs = .1, .25, .57, .7
    for i, end_flip_prob in enumerate(probs):
        for start_flip_prob in probs[i:]:
            # spin model
            T0, Tf = anneal_temperature_range(
                H, start_flip_prob, end_flip_prob, True
            )
            assert_allclose(T0, -10 / np.log(start_flip_prob))
            assert_allclose(Tf, -2 / np.log(end_flip_prob))

            # boolean model
            assert_allclose(
                (T0, Tf),
                anneal_temperature_range(
                    puso_to_pubo(H), start_flip_prob, end_flip_prob, False
                )
            )

    H = {(0, 1): 1, (1, 2,): -2, (1, 2, 3): 6, (): 11}
    probs = 0, .16, .56, .98
    for i, end_flip_prob in enumerate(probs):
        for start_flip_prob in probs[i:]:
            # spin model
            T0, Tf = anneal_temperature_range(
                H, start_flip_prob, end_flip_prob, True
            )
            assert_allclose(
                T0, -18 / np.log(start_flip_prob) if start_flip_prob else 0
            )
            assert_allclose(
                Tf, -2 / np.log(end_flip_prob) if end_flip_prob else 0
            )

            # boolean model
            assert_allclose(
                (T0, Tf),
                anneal_temperature_range(
                    puso_to_pubo(H), start_flip_prob, end_flip_prob, False
                )
            )


def test_anneal_puso():

    H = {(i, i+1, i+2): -1 for i in range(3)}

    with assert_raises(ValueError):
        anneal_puso(H, anneal_duration=-1)

    with assert_raises(ValueError):
        anneal_puso(H, anneal_duration=-2)

    with assert_warns(QUBOVertWarning):
        anneal_puso(H, temperature_range=(1, 2), schedule=[(3, 10), (2, 15)])

    with assert_raises(ValueError):
        anneal_puso(H, temperature_range=(1, 2))

    with assert_raises(ValueError):
        anneal_puso(H, schedule='something')

    empty_result = AnnealResults(True)
    for _ in range(4):
        empty_result.add_state({}, 2)
    assert anneal_puso({(): 2}, num_anneals=4) == empty_result

    assert anneal_puso(H, num_anneals=0) == AnnealResults(True)
    assert anneal_puso(H, num_anneals=-1) == AnnealResults(True)

    # just make sure everything runs
    anneal_puso(H, schedule='linear')
    anneal_puso(H, initial_state=[1]*5)

    # check to see if we find the groundstate of a simple but largeish model.
    H = {(i, i+1): -1 for i in range(30)}
    res = anneal_puso(H, num_anneals=4, seed=0)
    assert res.best.state in (
        dict(enumerate([1]*31)), dict(enumerate([-1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4


def test_anneal_quso():

    L = {(i, i+1): -1 for i in range(3)}

    with assert_raises(ValueError):
        anneal_quso(L, anneal_duration=-1)

    with assert_raises(ValueError):
        anneal_quso(L, anneal_duration=-2)

    with assert_warns(QUBOVertWarning):
        anneal_quso(L, temperature_range=(1, 2), schedule=[(3, 10), (2, 15)])

    with assert_raises(ValueError):
        anneal_quso(L, temperature_range=(1, 2))

    with assert_raises(ValueError):
        anneal_quso(L, schedule='something')

    empty_result = AnnealResults(True)
    for _ in range(4):
        empty_result.add_state({}, 2)
    assert anneal_quso({(): 2}, num_anneals=4) == empty_result

    assert anneal_quso(L, num_anneals=0) == AnnealResults(True)
    assert anneal_quso(L, num_anneals=-1) == AnnealResults(True)

    # just make sure everything runs
    anneal_quso(L, schedule='linear')
    anneal_quso(L, initial_state=[1]*4)

    # check to see if we find the groundstate of a simple but largeish model.
    L = {(i, i+1): -1 for i in range(30)}
    res = anneal_quso(L, num_anneals=4, seed=0)
    assert res.best.state in (
        dict(enumerate([1]*31)), dict(enumerate([-1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4


def test_anneal_pubo():

    P = puso_to_pubo({(i, i+1, i+2): -1 for i in range(3)})

    with assert_raises(ValueError):
        anneal_pubo(P, anneal_duration=-1)

    with assert_raises(ValueError):
        anneal_pubo(P, anneal_duration=-2)

    with assert_warns(QUBOVertWarning):
        anneal_pubo(P, temperature_range=(1, 2), schedule=[(3, 10), (2, 15)])

    with assert_raises(ValueError):
        anneal_pubo(P, temperature_range=(1, 2))

    with assert_raises(ValueError):
        anneal_pubo(P, schedule='something')

    empty_result = AnnealResults(False)
    for _ in range(4):
        empty_result.add_state({}, 2)
    assert anneal_pubo({(): 2}, num_anneals=4) == empty_result

    assert anneal_pubo(P, num_anneals=0) == AnnealResults(False)
    assert anneal_pubo(P, num_anneals=-1) == AnnealResults(False)

    # just make sure everything runs
    anneal_pubo(P, schedule='linear')
    anneal_pubo(P, initial_state=[1]*5)

    # check to see if we find the groundstate of a simple but largeish model.
    P = puso_to_pubo({(i, i+1): -1 for i in range(30)})
    res = anneal_pubo(P, num_anneals=4, seed=0)
    assert res.best.state in (
        dict(enumerate([0]*31)), dict(enumerate([1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4


def test_anneal_qubo():

    Q = quso_to_qubo({(i, i+1): -1 for i in range(3)})

    with assert_raises(ValueError):
        anneal_qubo(Q, anneal_duration=-1)

    with assert_raises(ValueError):
        anneal_qubo(Q, anneal_duration=-2)

    with assert_warns(QUBOVertWarning):
        anneal_qubo(Q, temperature_range=(1, 2), schedule=[(3, 10), (2, 15)])

    with assert_raises(ValueError):
        anneal_qubo(Q, temperature_range=(1, 2))

    with assert_raises(ValueError):
        anneal_qubo(Q, schedule='something')

    empty_result = AnnealResults(False)
    for _ in range(4):
        empty_result.add_state({}, 2)
    assert anneal_qubo({(): 2}, num_anneals=4) == empty_result

    assert anneal_qubo(Q, num_anneals=0) == AnnealResults(False)
    assert anneal_qubo(Q, num_anneals=-1) == AnnealResults(False)

    # just make sure everything runs
    anneal_qubo(Q, schedule='linear')
    anneal_qubo(Q, initial_state=[1]*4)

    # check to see if we find the groundstate of a simple but largeish model.
    Q = quso_to_qubo({(i, i+1): -1 for i in range(30)})
    res = anneal_qubo(Q, num_anneals=4, seed=0)
    assert res.best.state in (
        dict(enumerate([0]*31)), dict(enumerate([1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4
