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
Contains tests for the objects in the ``qubovert.sim._anneal_results.py`` file.
"""

from qubovert.sim import AnnealResult, AnnealResults
from qubovert.utils import spin_to_boolean


def test_annealresult():

    res = AnnealResult({1: 0, 2: 1}, 2, False)
    assert not res.spin
    assert res.state == {1: 0, 2: 1}
    assert res.value == 2
    assert res.copy() == res
    assert res.to_boolean() == res
    assert res.to_spin() == AnnealResult({1: 1, 2: -1}, 2, True)
    assert eval(repr(res)) == res
    str(res)

    res = AnnealResult({1: 1, 2: -1}, 2, True)
    assert res.spin
    assert res.state == {1: 1, 2: -1}
    assert res.value == 2
    assert res.copy() == res
    assert res.to_spin() == res
    assert res.to_boolean() == AnnealResult({1: 0, 2: 1}, 2, False)
    assert eval(repr(res)) == res
    str(res)


def test_annealresults():

    states = [
        ({0: -1, 1: 1, 'a': -1}, 1),
        ({0: 1, 1: 1, 'a': -1}, 9),
        ({0: -1, 1: -1, 'a': -1}, -3),
    ]
    sorted_states = sorted(states, key=lambda x: x[1])

    res, boolean_res = AnnealResults(True), AnnealResults(False)
    for s in states:
        res.add_state(*s)
        boolean_res.add_state(spin_to_boolean(s[0]), s[1])

    for s in states:
        assert AnnealResult(*s, True) in res
        assert AnnealResult(s[0], s[1]+1, True) not in res

    assert res.spin
    assert len(res) == 3
    assert len(boolean_res) == 3
    assert res.best.state == states[2][0]
    assert res.best.value == states[2][1]
    assert res.copy() == res
    assert res.to_spin() == res
    assert res.to_boolean() == boolean_res
    assert boolean_res.to_spin() == res
    str(res)
    str(boolean_res)
    repr(res)
    repr(boolean_res)

    count = 0
    for s in res:
        assert s == AnnealResult(*states[count], True)
        count += 1

    res.sort_by_value()
    count = 0
    for s in res:
        assert s == AnnealResult(*sorted_states[count], True)
        count += 1
