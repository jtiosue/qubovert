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
Contains tests for functions in the _binary_helpers.py file.
"""

from qubovert.utils import solution_type, num_bits
from numpy.testing import assert_raises


def test_solution_type():

    assert solution_type((0, 1, 1, 0)) == 'bool'
    assert solution_type((1, -1, -1, 1)) == 'spin'
    assert solution_type((1, 1, 1, 1), 'testing') == 'testing'
    assert solution_type((1, 1, 1, 1)) == 'bool'
    assert solution_type((1, 1, 1, 1), 'spin') == 'spin'

    assert solution_type(dict(enumerate((0, 1, 1, 0)))) == 'bool'
    assert solution_type(dict(enumerate((1, -1, -1, 1)))) == 'spin'
    assert solution_type(dict(enumerate((1, 1, 1, 1))), 'testing') == 'testing'
    assert solution_type(dict(enumerate((1, 1, 1, 1)))) == 'bool'
    assert solution_type(dict(enumerate((1, 1, 1, 1))), 'spin') == 'spin'


def test_num_bits():

    assert num_bits(7) == 3
    assert num_bits(8) == 4
    assert num_bits(7, False) == 7
    assert num_bits(8, False) == 8

    with assert_raises(ValueError):
        num_bits(-1)

    with assert_raises(ValueError):
        num_bits(-1, False)
