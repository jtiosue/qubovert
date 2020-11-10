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
    AnnealResults, SCHEDULES
)
from qubovert.utils import (
    puso_to_pubo, quso_to_qubo, QUBOVertWarning,
    QUBOMatrix, QUSOMatrix, PUBOMatrix, PUSOMatrix
)
from qubovert import QUBO, QUSO, PUBO, PUSO
from numpy.testing import assert_raises, assert_warns
import numpy as np


def test_anneal_puso():

    _anneal_puso(dict)
    _anneal_puso(PUSOMatrix)
    _anneal_puso(PUSO)


def _anneal_puso(type_):

    H = type_({(i, i+1, i+2): -1 for i in range(3)})

    with assert_raises(ValueError):
        anneal_puso(H, anneal_duration=-1)

    with assert_raises(ValueError):
        anneal_puso(H, anneal_duration=-2)

    with assert_warns(QUBOVertWarning):
        anneal_puso(H, temperature_range=(1, 2), schedule=[3, 2])

    with assert_warns(QUBOVertWarning):
        # a quadratic model warns that you shouldn't use anneal_puso
        anneal_puso({(0, 1): 1})

    with assert_raises(ValueError):
        anneal_puso(H, temperature_range=(1, 2))

    with assert_raises(ValueError):
        anneal_puso(H, schedule='something')

    empty_result = AnnealResults()
    for _ in range(4):
        empty_result.add_state({}, 2, True)
    # less than quadratic model so will warn
    with assert_warns(QUBOVertWarning):
        assert anneal_puso({(): 2}, num_anneals=4) == empty_result

    assert anneal_puso(H, num_anneals=0) == AnnealResults()
    assert anneal_puso(H, num_anneals=-1) == AnnealResults()

    # just make sure everything runs
    anneal_puso(H, schedule='linear')
    res = anneal_puso(H, initial_state=[1] * 5)
    for x in res:
        assert all(i in (1, -1) for i in x.state.values())

    # check to see if we find the groundstate of a simple but largeish model.
    H = type_({(i, i+1): -1 for i in range(30)})
    # quadratic model so will warn
    with assert_warns(QUBOVertWarning):
        res = anneal_puso(H, num_anneals=4, seed=0)
    assert res.best.state in (
        dict(enumerate([1]*31)), dict(enumerate([-1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4

    # check to see if we find the groundstate of same but out of order
    # quadratic so will warn
    with assert_warns(QUBOVertWarning):
        res = anneal_puso(H, num_anneals=4, in_order=False, seed=0)
    assert res.best.state in (
        dict(enumerate([1]*31)), dict(enumerate([-1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4

    # make sure we run branch where an explicit schedule is given and no
    # temperature range is supplied
    # quadratic so will warn
    with assert_warns(QUBOVertWarning):
        anneal_puso(H, schedule=[3, 2])

    # make sure it works with fields
    res = anneal_puso(
        type_({(0, 1, 2): 1, (1,): -1, (): 2}),
        num_anneals=10
    )
    assert len(res) == 10
    res.sort()
    for i in range(9):
        assert res[i].value <= res[i + 1].value

    # bigish ordering
    res = anneal_puso(
        type_(
            {(i, j, j + 1): 1 for i in range(70) for j in range(i+1, 70)}
        ),
        num_anneals=20
    )
    assert len(res) == 20
    res.sort()
    for i in range(19):
        assert res[i].value <= res[i + 1].value


def test_anneal_quso():

    _anneal_quso(dict)
    _anneal_quso(QUSOMatrix)
    _anneal_quso(PUSOMatrix)
    _anneal_quso(QUSO)
    _anneal_quso(PUSO)


def _anneal_quso(type_):

    L = type_({(i, i+1): -1 for i in range(3)})

    with assert_raises(ValueError):
        anneal_quso(L, anneal_duration=-1)

    with assert_raises(ValueError):
        anneal_quso(L, anneal_duration=-2)

    with assert_warns(QUBOVertWarning):
        anneal_quso(L, temperature_range=(1, 2), schedule=[3, 15])

    with assert_raises(ValueError):
        anneal_quso(L, temperature_range=(1, 2))

    with assert_raises(ValueError):
        anneal_quso(L, schedule='something')

    empty_result = AnnealResults()
    for _ in range(4):
        empty_result.add_state({}, 2, True)
    assert anneal_quso({(): 2}, num_anneals=4) == empty_result

    assert anneal_quso(L, num_anneals=0) == AnnealResults()
    assert anneal_quso(L, num_anneals=-1) == AnnealResults()

    # just make sure everything runs
    anneal_quso(L, schedule='linear')
    res = anneal_quso(L, initial_state=[1] * 5)
    for x in res:
        assert all(i in (1, -1) for i in x.state.values())

    # check to see if we find the groundstate of a simple but largeish model.
    L = type_({(i, i+1): -1 for i in range(30)})
    res = anneal_quso(L, num_anneals=4, seed=0)
    assert res.best.state in (
        dict(enumerate([1]*31)), dict(enumerate([-1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4

    # check to see if we find the groundstate of a sane but out of order
    res = anneal_quso(L, num_anneals=4, in_order=False, seed=0)
    assert res.best.state in (
        dict(enumerate([1]*31)), dict(enumerate([-1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4

    # make sure we run branch where an explicit schedule is given and no
    # temperature range is supplied
    anneal_quso(L, schedule=[3] * 10 + [2] * 15)

    # make sure it works with fields
    res = anneal_quso(type_({(0, 1): 1, (1,): -1, (): 2}), num_anneals=10)
    assert len(res) == 10
    res.sort()
    for i in range(9):
        assert res[i].value <= res[i + 1].value

    # big ordering
    res = anneal_quso(
        type_({(i, j): 1 for i in range(70) for j in range(i+1, 70)}),
        num_anneals=20
    )
    assert len(res) == 20
    res.sort()
    for i in range(19):
        assert res[i].value <= res[i + 1].value


def test_anneal_pubo():

    _anneal_pubo(dict)
    _anneal_pubo(PUBOMatrix)
    _anneal_pubo(PUBO)


def _anneal_pubo(type_):

    P = type_(puso_to_pubo({(i, i+1, i+2): -1 for i in range(3)}))

    with assert_raises(ValueError):
        anneal_pubo(P, anneal_duration=-1)

    with assert_raises(ValueError):
        anneal_pubo(P, anneal_duration=-2)

    with assert_warns(QUBOVertWarning):
        anneal_pubo(P, temperature_range=(1, 2), schedule=[3, 2])

    with assert_warns(QUBOVertWarning):
        # a quadratic model warns that you shouldn't use anneal_pubo
        anneal_pubo({(0, 1): 1})

    with assert_raises(ValueError):
        anneal_pubo(P, temperature_range=(1, 2))

    with assert_raises(ValueError):
        anneal_pubo(P, schedule='something')

    empty_result = AnnealResults()
    for _ in range(4):
        empty_result.add_state({}, 2, False)
    # less than quadratic so will warn
    with assert_warns(QUBOVertWarning):
        assert anneal_pubo({(): 2}, num_anneals=4) == empty_result

    assert anneal_pubo(P, num_anneals=0) == AnnealResults()
    assert anneal_pubo(P, num_anneals=-1) == AnnealResults()

    # just make sure everything runs
    anneal_pubo(P, schedule='linear')
    res = anneal_pubo(P, initial_state=[1] * 5)
    for x in res:
        assert all(i in (0, 1) for i in x.state.values())

    # check to see if we find the groundstate of a simple but largeish model.
    P = type_(puso_to_pubo({(i, i+1): -1 for i in range(30)}))
    # quadratic so will warn
    with assert_warns(QUBOVertWarning):
        res = anneal_pubo(P, num_anneals=4, seed=0)
    assert res.best.state in (
        dict(enumerate([0]*31)), dict(enumerate([1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4

    # check to see if we find the groundstate of same but out of order
    # quadratic so will warn
    with assert_warns(QUBOVertWarning):
        res = anneal_pubo(P, num_anneals=4, in_order=False, seed=0)
    assert res.best.state in (
        dict(enumerate([0]*31)), dict(enumerate([1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4

    # make sure we run branch where an explicit schedule is given and no
    # temperature range is supplied
    # quadratic so will warn
    with assert_warns(QUBOVertWarning):
        anneal_pubo(P, schedule=[3] * 10 + [2] * 15)

    # make sure it works with fields
    res = anneal_pubo(type_({(0, 1, 2): 1, (1,): -1, (): 2}), num_anneals=10)
    assert len(res) == 10
    res.sort()
    for i in range(9):
        assert res[i].value <= res[i + 1].value

    # bigish ordering
    res = anneal_pubo(
        type_(
            {(i, j, j + 1): 1 for i in range(70) for j in range(i+1, 70)}
        ),
        num_anneals=20
    )
    assert len(res) == 20
    res.sort()
    for i in range(19):
        assert res[i].value <= res[i + 1].value


def test_anneal_qubo():

    _anneal_qubo(dict)
    _anneal_qubo(QUBOMatrix)
    _anneal_qubo(PUBOMatrix)
    _anneal_qubo(QUBO)
    _anneal_qubo(PUBO)


def _anneal_qubo(type_):

    Q = type_(quso_to_qubo({(i, i+1): -1 for i in range(3)}))

    with assert_raises(ValueError):
        anneal_qubo(Q, anneal_duration=-1)

    with assert_raises(ValueError):
        anneal_qubo(Q, anneal_duration=-2)

    with assert_warns(QUBOVertWarning):
        anneal_qubo(Q, temperature_range=(1, 2), schedule=[3, 2])

    with assert_raises(ValueError):
        anneal_qubo(Q, temperature_range=(1, 2))

    with assert_raises(ValueError):
        anneal_qubo(Q, schedule='something')

    empty_result = AnnealResults()
    for _ in range(4):
        empty_result.add_state({}, 2, False)
    assert anneal_qubo({(): 2}, num_anneals=4) == empty_result

    assert anneal_qubo(Q, num_anneals=0) == AnnealResults()
    assert anneal_qubo(Q, num_anneals=-1) == AnnealResults()

    # just make sure everything runs
    anneal_qubo(Q, schedule='linear')
    res = anneal_qubo(Q, initial_state=[1] * 5)
    for x in res:
        assert all(i in (0, 1) for i in x.state.values())

    # check to see if we find the groundstate of a simple but largeish model.
    Q = type_(quso_to_qubo({(i, i+1): -1 for i in range(30)}))
    res = anneal_qubo(Q, num_anneals=4, seed=0)
    assert res.best.state in (
        dict(enumerate([0]*31)), dict(enumerate([1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4

    # check to see if we find the groundstate of the same but out of order
    res = anneal_qubo(Q, num_anneals=4, in_order=False, seed=0)
    assert res.best.state in (
        dict(enumerate([0]*31)), dict(enumerate([1]*31))
    )
    assert res.best.value == -30
    assert len([x for x in res]) == 4

    # make sure we run branch where an explicit schedule is given and no
    # temperature range is supplied
    anneal_qubo(Q, schedule=[3] * 10 + [2] * 15)

    # make sure it works with fields
    res = anneal_qubo(type_({(0, 1): 1, (1,): -1, (): 2}), num_anneals=10)
    assert len(res) == 10
    res.sort()
    for i in range(9):
        assert res[i].value <= res[i + 1].value

    # big ordering
    res = anneal_qubo(
        type_({(i, j): 1 for i in range(70) for j in range(i+1, 70)}),
        num_anneals=20
    )
    assert len(res) == 20
    res.sort()
    for i in range(19):
        assert res[i].value <= res[i + 1].value


def test_anneal_quso_vs_anneal_puso():

    L = {(i, j): 1 for i in range(10) for j in range(i+1, 10)}
    L.update({(i,): 1 for i in range(10)})
    kwargs = {}
    for seed in range(10):
        kwargs['seed'] = seed
        for schedule in SCHEDULES:
            kwargs['schedule'] = schedule
            for in_order in range(2):
                kwargs['in_order'] = in_order
                for anneal_duration in (10, 100, 1000):
                    kwargs['anneal_duration'] = anneal_duration
                    for num_anneals in range(1, 7):
                        kwargs['num_anneals'] = num_anneals
                        for T0 in (.1, 1, 10):
                            kwargs['temperature_range'] = T0, T0 / 2
                            # quadratic so anneal_puso will warn
                            with assert_warns(QUBOVertWarning):
                                respuso = anneal_puso(L, **kwargs)
                            assert respuso == anneal_quso(L, **kwargs)


def test_anneal_qubo_vs_anneal_pubo():

    Q = {(i, j): 1 for i in range(10) for j in range(i+1, 10)}
    Q.update({(i,): 1 for i in range(10)})
    kwargs = {}
    for seed in range(10):
        kwargs['seed'] = seed
        for schedule in SCHEDULES:
            kwargs['schedule'] = schedule
            for in_order in range(2):
                kwargs['in_order'] = in_order
                for anneal_duration in (10, 100, 1000):
                    kwargs['anneal_duration'] = anneal_duration
                    for num_anneals in range(1, 7):
                        kwargs['num_anneals'] = num_anneals
                        for T0 in (.1, 1, 10):
                            kwargs['temperature_range'] = T0, T0 / 2
                            # quadratic so anneal_pubo will warn
                            with assert_warns(QUBOVertWarning):
                                respubo = anneal_pubo(Q, **kwargs)
                            assert respubo == anneal_qubo(Q, **kwargs)
