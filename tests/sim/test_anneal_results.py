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
    anneal_states = [AnnealResult(*s, True) for s in states]
    anneal_sorted_states = [AnnealResult(*s, True) for s in sorted_states]

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
    assert res == anneal_states
    assert boolean_res == [x.to_boolean() for x in anneal_states]
    str(res)
    str(boolean_res)
    repr(res)
    repr(boolean_res)

    count = 0
    for s in res:
        assert s == AnnealResult(*states[count], True)
        count += 1

    for i in range(3):
        assert res[i] == anneal_states[i]

    assert res[:2] == anneal_states[:2]
    assert isinstance(res[1:3], AnnealResults)

    res.sort_by_value()
    count = 0
    for s in res:
        assert s == AnnealResult(*sorted_states[count], True)
        count += 1

    for i in range(3):
        assert res[i] == anneal_sorted_states[i]

    assert res[:2] == anneal_sorted_states[:2]

    boolean_res.sort_by_value()
    assert res == anneal_sorted_states
    assert boolean_res == [x.to_boolean() for x in anneal_sorted_states]

    assert type(res * 2) == AnnealResults
    assert type(res + res) == AnnealResults
    res *= 2
    assert type(res) == AnnealResults
    res += res
    assert type(res) == AnnealResults
    res += anneal_states
    assert type(res) == AnnealResults
    res.extend(anneal_sorted_states)
    assert type(res) == AnnealResults

    res.clear()
    assert res.best is None
    assert not res


def test_annealresults_filter():

    states = [
        AnnealResult({0: -1, 1: 1, 'a': -1}, 1, True),
        AnnealResult({0: 1, 1: 1, 'a': -1}, 9, True),
        AnnealResult({0: -1, 1: -1, 'a': -1}, -3, True)
    ]
    filtered_states = [
        AnnealResult({0: -1, 1: 1, 'a': -1}, 1, True),
        AnnealResult({0: -1, 1: -1, 'a': -1}, -3, True)
    ]
    res = AnnealResults.from_list(states, True)
    filtered_res = res.filter(lambda x: x.state[0] == -1)
    assert filtered_res == filtered_states


def test_annealresults_filter_states():

    states = [
        AnnealResult({0: -1, 1: 1, 'a': -1}, 1, True),
        AnnealResult({0: 1, 1: 1, 'a': -1}, 9, True),
        AnnealResult({0: -1, 1: -1, 'a': -1}, -3, True)
    ]
    filtered_states = [
        AnnealResult({0: -1, 1: 1, 'a': -1}, 1, True),
        AnnealResult({0: 1, 1: 1, 'a': -1}, 9, True)
    ]
    res = AnnealResults.from_list(states, True)
    filtered_res = res.filter_states(lambda x: x[1] == 1)
    assert filtered_res == filtered_states


def test_annealresults_apply_function():

    states = [
        AnnealResult({0: -1, 1: 1, 'a': -1}, 1, True),
        AnnealResult({0: 1, 1: 1, 'a': -1}, 9, True),
        AnnealResult({0: -1, 1: -1, 'a': -1}, -3, True)
    ]
    new_states = [
        AnnealResult({0: -1, 1: 1, 'a': -1}, 1+2, True),
        AnnealResult({0: 1, 1: 1, 'a': -1}, 9+2, True),
        AnnealResult({0: -1, 1: -1, 'a': -1}, -3+2, True)
    ]
    res = AnnealResults.from_list(states, True)
    new_res = res.apply_function(
        lambda x: AnnealResult(x.state, x.value + 2, x.spin)
    )
    assert new_res == new_states


def test_annealresults_convert_states():

    states = [
        AnnealResult({0: -1, 1: 1, 'a': -1}, 1, True),
        AnnealResult({0: 1, 1: 1, 'a': -1}, 9, True),
        AnnealResult({0: -1, 1: -1, 'a': -1}, -3, True)
    ]
    new_states = [
        AnnealResult({'b': -1, 1: 1, 'a': -1}, 1, True),
        AnnealResult({'b': 1, 1: 1, 'a': -1}, 9, True),
        AnnealResult({'b': -1, 1: -1, 'a': -1}, -3, True)
    ]
    res = AnnealResults.from_list(states, True)
    new_res = res.convert_states(
        lambda x: {k if k else 'b': v for k, v in x.items()}
    )
    assert new_res == new_states
