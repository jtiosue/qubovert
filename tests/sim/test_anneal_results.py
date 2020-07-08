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
from numpy.testing import assert_raises


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


def test_annealresult_comparison():

    res1 = AnnealResult({1: 0, 2: 1}, 2, False)
    res2 = AnnealResult({1: 0, 2: 1}, 1, False)
    res3 = AnnealResult({1: 0, 2: 0}, 2, False)
    res4 = AnnealResult({1: 0, 2: 0}, 1, False)

    assert res1.copy() == res1
    assert res2 < res1
    assert res2 <= res1
    assert res1 > res2
    assert res1 >= res2
    assert res1 != res2
    assert res4 < res1
    assert res4 <= res1
    assert res1 > res4
    assert res1 >= res4
    assert res1 != res4
    assert res4 < res3
    assert res4 <= res3
    assert res3 > res4
    assert res3 >= res4
    assert res3 != res4
    assert res2 <= res4 and res2 >= res4

    res1 = AnnealResult({1: 1, 2: -1}, 2, True)
    res2 = AnnealResult({1: 1, 2: -1}, 1, True)
    res3 = AnnealResult({1: 1, 2: 1}, 2, True)
    res4 = AnnealResult({1: 1, 2: 1}, 1, True)

    assert res1.copy() == res1
    assert res2 < res1
    assert res1 > res2
    assert res1 != res2
    assert res4 < res1
    assert res1 > res4
    assert res1 != res4
    assert res4 < res3
    assert res3 > res4
    assert res3 != res4


def test_annealresults():

    states = [
        ({0: -1, 1: 1, 'a': -1}, 1),
        ({0: 1, 1: 1, 'a': -1}, 9),
        ({0: -1, 1: -1, 'a': -1}, -3),
    ]
    sorted_states = sorted(states, key=lambda x: x[1])
    anneal_states = [AnnealResult(*s, True) for s in states]
    anneal_sorted_states = [AnnealResult(*s, True) for s in sorted_states]

    res, boolean_res = AnnealResults(), AnnealResults()
    for s in states:
        res.add_state(*s, True)
        boolean_res.add_state(spin_to_boolean(s[0]), s[1], False)

    for s in states:
        assert AnnealResult(*s, True) in res
        assert AnnealResult(s[0], s[1]+1, True) not in res

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

    res.sort()
    count = 0
    for s in res:
        assert s == AnnealResult(*sorted_states[count], True)
        count += 1

    for i in range(3):
        assert res[i] == anneal_sorted_states[i]

    assert res[:2] == anneal_sorted_states[:2]

    boolean_res.sort()
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
    res = AnnealResults(states)
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
    res = AnnealResults(states)
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
    res = AnnealResults(states)
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
    res = AnnealResults(states)
    new_res = res.convert_states(
        lambda x: {k if k else 'b': v for k, v in x.items()}
    )
    assert new_res == new_states


def test_annealresults_extend_add():

    res0 = AnnealResults(AnnealResult({}, 2, True) for _ in range(4))
    assert res0.best.value == 2
    res1 = AnnealResults(AnnealResult({}, 1, True) for _ in range(3))
    assert res1.best.value == 1

    assert (res0 + res1).best.value == 1
    assert type(res0 + res1) == AnnealResults

    temp = res0.copy()
    temp += res1
    assert temp.best.value == 1
    assert type(temp) == AnnealResults

    temp = res0.copy()
    temp.extend(res1)
    assert temp.best.value == 1
    assert type(temp) == AnnealResults

    assert (res1 + res0).best.value == 1
    assert type(res1 + res0) == AnnealResults

    temp = res1.copy()
    temp += res0
    assert temp.best.value == 1
    assert type(temp) == AnnealResults

    temp = res1.copy()
    temp.extend(res0)
    assert temp.best.value == 1
    assert type(temp) == AnnealResults


def test_annealresults_insert_remove_pop():

    # make sure the best attribute is updated
    res0 = AnnealResults([
        AnnealResult({}, 2, True), AnnealResult({}, 1, True)
    ])
    a0 = AnnealResult({}, 1, True) 
    a1 = AnnealResult({}, 2, True)
    assert res0.best == a0

    # insert
    temp = res0.copy()
    temp.insert(0, a0)
    assert len(temp) == 3

    r = AnnealResults([a0.copy()])
    assert r.best == a0
    temp = r.copy()
    temp.insert(0, a1)
    assert temp.best == a0
    temp = r.copy()
    temp.insert(1, a1)
    assert temp.best == a0

    r = AnnealResults([a1.copy()])
    assert r.best == a1
    temp = r.copy()
    temp.insert(0, a0)
    assert temp.best == a0
    temp = r.copy()
    temp.insert(1, a0)
    assert temp.best == a0

    # remove
    temp = res0.copy()
    temp.remove(a1)
    assert len(temp) == 1 and temp.best == a0
    temp.remove(a0)
    assert not len(temp) and temp.best is None

    temp = res0.copy()
    temp.remove(a0)
    assert len(temp) == 1 and temp.best == a1
    temp.remove(a1)
    assert temp and temp.best is None
    with assert_raises(ValueError):
        temp.remove(a0)

    # pop
    temp = res0.copy()
    t = temp.pop(1)
    assert len(temp) == 1 and temp.best == a0 and t == a1
    t = temp.pop(0)
    assert not len(temp) and temp.best is None and t == a0

    temp = res0.copy()
    t = temp.pop(0)
    assert len(temp) == 1 and temp.best == a1 and t == a0
    t = temp.pop(0)
    assert not len(temp) and temp.best is None and t == a1
    with assert_raises(IndexError):
        temp.pop(0)
