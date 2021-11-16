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
Contains tests for functions in the utils/_info.py file.
"""

from qubovert.utils import get_info, create_from_info
from qubovert import QUBO, QUSO, PUBO, PUSO, PCBO, PCSO
from qubovert.utils import (
    QUBOMatrix, QUSOMatrix, PUBOMatrix, PUSOMatrix, DictArithmetic,
    qubo_to_quso
)

dictarithmetic = DictArithmetic({(i, i+1): -1 for i in range(3)})
qubomatrix = QUBOMatrix(dictarithmetic)
qusomatrix = qubo_to_quso(qubomatrix)
pubomatrix = PUBOMatrix(qubomatrix)
pusomatrix = PUSOMatrix(qusomatrix)

qubo = QUBO({("x%d" % i, "x%d" % (i+1)): 1 for i in range(3)})
quso = qubo_to_quso(qubo)
pubo = PUBO(qubo)
puso = PUSO(quso)

p = {("x0",): 1, ("x0", "x1"): -1}
pcbo = PCBO(pubo).add_constraint_eq_zero(p).add_constraint_ne_zero(
    p).add_constraint_le_zero(p).add_constraint_ge_zero(
    p).add_constraint_lt_zero(p).add_constraint_gt_zero(p)

p = qubo_to_quso({("x0",): 1, ("x0", "x1"): -1})
pcso = PCSO(puso).add_constraint_eq_zero(p).add_constraint_ne_zero(
    p).add_constraint_le_zero(p).add_constraint_ge_zero(
    p).add_constraint_lt_zero(p).add_constraint_gt_zero(p)

dictarithmetic.name = "dictarithmetic"
qubomatrix.name = "qubomatrix"
qusomatrix.name = "qusomatrix"
pubomatrix.name = "pubomatrix"
pusomatrix.name = "pusomatrix"
qubo.name = "qubo"
quso.name = "quso"
pubo.name = "pubo"
puso.name = "puso"
pcbo.name = "pcbo"
pcso.name = "pcso"

models = (
    dictarithmetic, qubomatrix, qusomatrix, pubomatrix, pusomatrix,
    qubo, quso, pubo, puso, pcbo, pcso
)


def test_from_get():

    for m in models:
        mp = create_from_info(get_info(m))
        assert mp == m
        assert mp.name == m.name
        assert type(mp) == type(m)
        if hasattr(m, "variables"):
            assert mp.variables == m.variables
        if hasattr(m, "mapping"):
            assert mp.mapping == m.mapping
        if hasattr(m, "num_ancillas"):
            assert mp.num_ancillas == m.num_ancillas
        if hasattr(m, "constraints"):
            assert mp.constraints == m.constraints


def test_get_from():

    def f(model):
        return get_info(create_from_info(model))

    dictarithmetic_info = dict(
        type="DictArithmetic", terms=dict(dictarithmetic)
    )
    d = f(dictarithmetic_info)
    dictarithmetic_info["name"] = None
    assert d == dictarithmetic_info
    dictarithmetic_info["name"] = "dictarithmetic_info"
    assert dictarithmetic_info == f(dictarithmetic_info)
    d = dictarithmetic_info.copy()
    d["mapping"] = None
    assert dictarithmetic_info == f(d)
    d["num_ancillas"] = 0
    assert dictarithmetic_info == f(d)
    d["constraints"] = {}
    assert dictarithmetic_info == f(d)

    qubomatrix_info = dict(type="QUBOMatrix", terms=dict(qubomatrix))
    d = f(qubomatrix_info)
    qubomatrix_info["name"] = None
    assert d == qubomatrix_info
    qubomatrix_info["name"] = "qubomatrix_info"
    assert qubomatrix_info == f(qubomatrix_info)
    d = qubomatrix_info.copy()
    d["mapping"] = None
    assert qubomatrix_info == f(d)
    d["num_ancillas"] = 0
    assert qubomatrix_info == f(d)
    d["constraints"] = {}
    assert qubomatrix_info == f(d)

    qusomatrix_info = dict(type="QUSOMatrix", terms=dict(qusomatrix))
    d = f(qusomatrix_info)
    qusomatrix_info["name"] = None
    assert d == qusomatrix_info
    qusomatrix_info["name"] = "qusomatrix_info"
    assert qusomatrix_info == f(qusomatrix_info)
    d = qusomatrix_info.copy()
    d["mapping"] = None
    assert qusomatrix_info == f(d)
    d["num_ancillas"] = 0
    assert qusomatrix_info == f(d)
    d["constraints"] = {}
    assert qusomatrix_info == f(d)

    pubomatrix_info = dict(type="PUBOMatrix", terms=dict(pubomatrix))
    d = f(pubomatrix_info)
    pubomatrix_info["name"] = None
    assert d == pubomatrix_info
    pubomatrix_info["name"] = "pubomatrix_info"
    assert pubomatrix_info == f(pubomatrix_info)
    d = pubomatrix_info.copy()
    d["mapping"] = None
    assert pubomatrix_info == f(d)
    d["num_ancillas"] = 0
    assert pubomatrix_info == f(d)
    d["constraints"] = {}
    assert pubomatrix_info == f(d)

    pusomatrix_info = dict(type="PUSOMatrix", terms=dict(pusomatrix))
    d = f(pusomatrix_info)
    pusomatrix_info["name"] = None
    assert d == pusomatrix_info
    pusomatrix_info["name"] = "pusomatrix_info"
    assert pusomatrix_info == f(pusomatrix_info)
    d = pusomatrix_info.copy()
    d["mapping"] = None
    assert pusomatrix_info == f(d)
    d["num_ancillas"] = 0
    assert pusomatrix_info == f(d)
    d["constraints"] = {}
    assert pusomatrix_info == f(d)

    qubo_info = dict(type="QUBO", terms=dict(qubo), mapping=qubo.mapping)
    d = f(qubo_info)
    qubo_info["name"] = None
    assert d == qubo_info
    qubo_info["name"] = "qubo"
    assert qubo_info == f(qubo_info)
    d = qubo_info.copy()
    d["num_ancillas"] = 0
    assert qubo_info == f(d)
    d["constraints"] = {}
    assert qubo_info == f(d)

    quso_info = dict(type="QUSO", terms=dict(quso), mapping=quso.mapping)
    d = f(quso_info)
    quso_info["name"] = None
    assert d == quso_info
    quso_info["name"] = "quso"
    assert quso_info == f(quso_info)
    d = quso_info.copy()
    d["num_ancillas"] = 0
    assert quso_info == f(d)
    d["constraints"] = {}
    assert quso_info == f(d)

    pubo_info = dict(type="PUBO", terms=dict(pubo), mapping=pubo.mapping)
    d = f(pubo_info)
    pubo_info["name"] = None
    assert d == pubo_info
    pubo_info["name"] = "pubo"
    assert pubo_info == f(pubo_info)
    d = pubo_info.copy()
    d["num_ancillas"] = 0
    assert pubo_info == f(d)
    d["constraints"] = {}
    assert pubo_info == f(d)

    puso_info = dict(type="PUSO", terms=dict(puso), mapping=puso.mapping)
    d = f(puso_info)
    puso_info["name"] = None
    assert d == puso_info
    puso_info["name"] = "puso"
    assert puso_info == f(puso_info)
    d = puso_info.copy()
    d["num_ancillas"] = 0
    assert puso_info == f(d)
    d["constraints"] = {}
    assert puso_info == f(d)

    pcbo_info = dict(
        type="PCBO", terms=dict(pcbo), mapping=pcbo.mapping, name=pcbo.name,
        num_ancillas=pcbo.num_ancillas, constraints=pcbo.constraints
    )
    assert pcbo_info == f(pcbo_info)

    pcso_info = dict(
        type="PCSO", terms=dict(pcso), mapping=pcso.mapping, name=pcso.name,
        num_ancillas=pcso.num_ancillas, constraints=pcso.constraints
    )
    assert pcso_info == f(pcso_info)


def test_create_empty():

    qubo = create_from_info(dict(type="QUBO"))
    assert qubo == {}
    assert qubo.name is None
