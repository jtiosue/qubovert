#   Copyright 2019 Joseph T. Iosue
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
Contains tests for the bruteforce solvers.
"""

from qubovert.utils import solve_qubo_bruteforce, solve_ising_bruteforce


def test_solve_qubo_bruteforce():

    Q = {('0', 1): 1, (1, '2'): 1, (1, 1): -1, ('2', '2'): -2}
    assert solve_qubo_bruteforce(Q) == (-2, {'0': 0, 1: 0, '2': 1})

    Q = {(0, 0): 1, (0, 1): -1}
    assert (
        solve_qubo_bruteforce(Q, 1, True)
        ==
        (1, [{0: 0, 1: 0}, {0: 0, 1: 1}, {0: 1, 1: 1}])
    )


def test_solve_ising_bruteforce():

    h = {'a': -1, 2: -2}
    J = {(0, 'a'): 1, ('a', 2): 1}
    assert solve_ising_bruteforce(h, J) in (
        (-3, {0: -1, 'a': 1, 2: 1}),
        (-3, {0: 1, 'a': -1, 2: 1}),
    )

    h, J, offset = {0: 0.25, 1: -0.25}, {(0, 1): -0.25}, 1.25
    assert (
        solve_ising_bruteforce(h, J, offset, True)
        ==
        (1, [{0: -1, 1: -1}, {0: -1, 1: 1}, {0: 1, 1: 1}])
    )
