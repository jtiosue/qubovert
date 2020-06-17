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
Contains tests for the functions in the
``qubovert.sim._anneal_temperature_range`` file.
"""

from qubovert.sim import anneal_temperature_range
from qubovert.utils import puso_to_pubo
from numpy.testing import assert_raises, assert_allclose
from math import log


def test_anneal_temperature_range():

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
            assert_allclose(T0, -10 / log(start_flip_prob))
            assert_allclose(Tf, -2 / log(end_flip_prob))

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
                T0, -18 / log(start_flip_prob) if start_flip_prob else 0
            )
            assert_allclose(
                Tf, -2 / log(end_flip_prob) if end_flip_prob else 0
            )

            # boolean model
            assert_allclose(
                (T0, Tf),
                anneal_temperature_range(
                    puso_to_pubo(H), start_flip_prob, end_flip_prob, False
                )
            )
