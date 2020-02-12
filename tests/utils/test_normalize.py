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
Contains tests for the normalize function.
"""

from qubovert.utils import normalize
from qubovert.utils import (
    DictArithmetic, QUBOMatrix, PUBOMatrix, QUSOMatrix, PUSOMatrix
)
from qubovert import QUBO, PUBO, QUSO, PUSO, PCBO, PCSO


def test_subgraph():

    temp0 = {(0,): 4, (1,): -2}
    assert normalize(temp0) == {k: v / 4 for k, v in temp0.items()}

    temp1 = {(0,): -4, (1,): 2}
    assert normalize(temp1) == {k: v / 4 for k, v in temp1.items()}

    for t in (QUBO, PUBO, QUSO, PUSO, PCBO, PCSO, DictArithmetic,
              QUBOMatrix, PUBOMatrix, QUSOMatrix, PUSOMatrix):
        n = normalize(t(temp0))
        assert type(n) == t
