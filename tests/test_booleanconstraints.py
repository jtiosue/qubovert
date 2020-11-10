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
Contains tests for the BooleanConstraints class.
"""

from qubovert import BooleanConstraints, PUBO, integer_var, boolean_var
from qubovert.utils import (
    solve_pubo_bruteforce, decimal_to_boolean, QUBOVertWarning
)
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises, assert_warns


# testing methods


def test_constraints():

    d = BooleanConstraints()
    assert d.constraints == {}
    temp = {(0,): 1, (1,): -1}
    d.add_constraint_eq_zero(temp)
    assert d.constraints == {'eq': [temp]}
    str(d)
    repr(d)


def test_round():

    d = BooleanConstraints()
    d.add_constraint_eq_zero({(0,): 3.456, (1,): -1.53456})

    assert round(d).constraints == {"eq": [{(0,): 3, (1,): -2}]}
    assert round(d, 1).constraints == {"eq": [{(0,): 3.5, (1,): -1.5}]}
    assert round(d, 2).constraints == {"eq": [{(0,): 3.46, (1,): -1.53}]}
    assert round(d, 3).constraints == {"eq": [{(0,): 3.456, (1,): -1.535}]}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = BooleanConstraints()
    d.add_constraint_eq_zero({(0,): a, (1,): -b}, bounds=(-1, 1))
    assert d.to_penalty() == {
        (0,): a**2, (0, 1): -2*a*b, (1,): b**2
    }
    assert d.subs(a, 0).to_penalty() == {(1,): b**2}
    assert d.subs({a: 0, b: 2}).to_penalty() == {(1,): 4}


def test_pop_constraint():

    P = BooleanConstraints().add_constraint_eq_zero(
        {(0,): 1, (1,): -2}
    ).add_constraint_eq_zero({(0,): 1})
    temp = P.copy()
    P._pop_constraint('gt')
    assert temp.constraints == P.constraints
    P._pop_constraint('eq')
    assert P.constraints == {'eq': [{(0,): 1, (1,): -2}]}
    P._pop_constraint('eq')
    assert P.constraints == {}


""" TESTS FOR THE CONSTRAINT METHODS """


def test_booleanconstraints_eq_constraint():

    lam = Symbol('lam')

    pubo = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    C = BooleanConstraints().add_constraint_eq_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): -1},
        lam=lam
    )
    P = pubo + C.to_penalty()
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4

    problem = pubo.subs(lam, 1)
    _, sol = solve_pubo_bruteforce(problem, valid=C.is_solution_valid)
    assert all((
        C.is_solution_valid(sol),
        sol == solution
    ))

    problem = P.subs(lam, 1)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = P.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))

    # lam = 0
    pubo = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    C = BooleanConstraints().add_constraint_eq_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): -1},
        lam=0
    )
    P = pubo + C.to_penalty()

    _, sol = solve_pubo_bruteforce(pubo, valid=C.is_solution_valid)
    assert all((
        C.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(P.to_pubo())
    sol = P.convert_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_booleanconstraints_ne_constraint_logtrick():

    for i in range(1 << 4):
        P = integer_var('a', 4) - i
        H = BooleanConstraints().add_constraint_ne_zero(P)
        for sol in solve_pubo_bruteforce(H.to_penalty(),
                                         True, H.is_solution_valid)[1]:
            assert P.value(sol)

    for i in range(1 << 2):
        P = integer_var('a', 2) - i
        H = BooleanConstraints().add_constraint_ne_zero(P).to_penalty()
        for sol in solve_pubo_bruteforce(H, True)[1]:
            assert P.value(sol)


def test_booleanconstraints_ne_constraint():

    for i in range(1 << 2):
        P = integer_var('a', 2) - i
        H = BooleanConstraints().add_constraint_ne_zero(
            P, log_trick=False).to_penalty()
        for sol in solve_pubo_bruteforce(H, True)[1]:
            assert P.value(sol)

    with assert_warns(QUBOVertWarning):  # never satisfied
        BooleanConstraints().add_constraint_ne_zero({})

    with assert_warns(QUBOVertWarning):  # always satisfed
        BooleanConstraints().add_constraint_ne_zero({(): 2, ('a',): -1})

    with assert_warns(QUBOVertWarning):  # always satisfed
        BooleanConstraints().add_constraint_ne_zero({(): -2, ('a',): 1})

    # same as above but with suppress warnings
    BooleanConstraints().add_constraint_ne_zero(
        {}, suppress_warnings=True)
    BooleanConstraints().add_constraint_ne_zero(
        {(): 2, ('a',): -1}, suppress_warnings=True)
    BooleanConstraints().add_constraint_ne_zero(
        {(): -2, ('a',): 1}, suppress_warnings=True)


def test_booleanconstraints_lt_constraint_logtrick():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    C = BooleanConstraints().add_constraint_lt_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3},
        lam=lam
    )
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4
    model = P + C.to_penalty()

    problem = model.subs(lam, 1)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    sol = problem.solve_bruteforce()
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))

    # lam = 0
    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    C = BooleanConstraints().add_constraint_lt_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3},
        lam=0
    )
    model = P + C.to_penalty()

    sol = C.remove_ancilla_from_solution(
        solve_pubo_bruteforce(model, valid=C.is_solution_valid)[1])
    assert all((
        C.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(model.to_pubo())
    sol = model.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_booleanconstraints_lt_constraint():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    C = BooleanConstraints().add_constraint_lt_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3},
        lam=lam, log_trick=False
    )
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4
    model = P + C.to_penalty()

    problem = model.subs(lam, 1)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_booleanconstraints_le_constraint_logtrick():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_le_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3},
        lam=lam
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8
    model = P + C.to_penalty()

    problem = model.subs(lam, .5)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))

    # lam = 0
    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_le_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3},
        lam=0
    )
    model = P + C.to_penalty()
    e, sol = solve_pubo_bruteforce(model.to_pubo())
    sol = model.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_booleanconstraints_le_constraint():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_le_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3},
        lam=lam, log_trick=False
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8
    model = P + C.to_penalty()

    problem = model.subs(lam, .5)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_booleanconstraints_le_constraint_minval_zero():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_le_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1},
        lam=lam
    )
    solution = {'c': 0, 'b': 0, 'a': 0, 'd': 0}
    obj = -2
    model = P + C.to_penalty()

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))

    lam = Symbol("lam")

    P1 = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C1 = BooleanConstraints().add_constraint_le_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1},
        lam=lam, log_trick=False
    )
    assert model == P1 + C1.to_penalty()


def test_booleanconstraints_gt_constraint_logtrick():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    C = BooleanConstraints().add_constraint_gt_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3},
        lam=lam
    )
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4
    model = P + C.to_penalty()

    problem = model.subs(lam, 1)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))

    # lam = 0
    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    C = BooleanConstraints().add_constraint_gt_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3},
        lam=0
    )
    model = P + C.to_penalty()
    e, sol = solve_pubo_bruteforce(model.to_pubo())
    sol = model.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_booleanconstraints_gt_constraint():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    C = BooleanConstraints().add_constraint_gt_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3},
        lam=lam, log_trick=False
    )
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4
    model = P + C.to_penalty()

    problem = model.subs(lam, 1)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_booleanconstraints_ge_constraint_logtrick():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_ge_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3},
        lam=lam
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8
    model = P + C.to_penalty()

    problem = model.subs(lam, .5)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))

    # lam = 0
    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_ge_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3},
        lam=0
    )
    model = P + C.to_penalty()
    e, sol = solve_pubo_bruteforce(model.to_pubo())
    sol = model.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_booleanconstraints_ge_constraint():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_ge_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3},
        lam=lam, log_trick=False
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8
    model = P + C.to_penalty()

    problem = model.subs(lam, .5)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_booleanconstraints_constraints_warnings():

    with assert_warns(QUBOVertWarning):  # qlwayss satisfied
        BooleanConstraints().add_constraint_eq_zero({(): 0})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        BooleanConstraints().add_constraint_eq_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        BooleanConstraints().add_constraint_eq_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        BooleanConstraints().add_constraint_lt_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        BooleanConstraints().add_constraint_lt_zero({(): 1, (0,): -1})

    with assert_warns(QUBOVertWarning):  # always satisfied
        BooleanConstraints().add_constraint_lt_zero({(): -1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        BooleanConstraints().add_constraint_le_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # always satisfied
        BooleanConstraints().add_constraint_le_zero({(): -1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        BooleanConstraints().add_constraint_gt_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        BooleanConstraints().add_constraint_gt_zero({(): -1, (0,): 1})

    with assert_warns(QUBOVertWarning):  # always satisfied
        BooleanConstraints().add_constraint_gt_zero({(): 1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        BooleanConstraints().add_constraint_ge_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # always satisfied
        BooleanConstraints().add_constraint_ge_zero({(): 1, (0,): .5})

    # all the same but with ignore warnings
    sup = dict(suppress_warnings=True)
    BooleanConstraints().add_constraint_eq_zero({(): 0}, **sup)
    BooleanConstraints().add_constraint_eq_zero({(): 1, (0,): -.5}, **sup)
    BooleanConstraints().add_constraint_eq_zero({(): -1, (0,): .5}, **sup)
    BooleanConstraints().add_constraint_lt_zero({(): 1, (0,): -.5}, **sup)
    BooleanConstraints().add_constraint_lt_zero({(): 1, (0,): -1}, **sup)
    BooleanConstraints().add_constraint_lt_zero({(): -1, (0,): -.5}, **sup)
    BooleanConstraints().add_constraint_le_zero({(): 1, (0,): -.5}, **sup)
    BooleanConstraints().add_constraint_le_zero({(): -1, (0,): -.5}, **sup)
    BooleanConstraints().add_constraint_gt_zero({(): -1, (0,): .5}, **sup)
    BooleanConstraints().add_constraint_gt_zero({(): -1, (0,): 1}, **sup)
    BooleanConstraints().add_constraint_gt_zero({(): 1, (0,): .5}, **sup)
    BooleanConstraints().add_constraint_ge_zero({(): -1, (0,): .5}, **sup)
    BooleanConstraints().add_constraint_ge_zero({(): 1, (0,): .5}, **sup)


def test_booleanconstraints_logic():

    with assert_raises(ValueError):
        BooleanConstraints().add_constraint_eq_NOR('a')

    with assert_raises(ValueError):
        BooleanConstraints().add_constraint_eq_NAND('a')

    with assert_raises(ValueError):
        BooleanConstraints().add_constraint_eq_OR('a')

    with assert_raises(ValueError):
        BooleanConstraints().add_constraint_eq_AND('a')

    H = BooleanConstraints().add_constraint_eq_NAND(
        'c', 'a', 'b').add_constraint_AND('a', 'b').to_penalty()
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 1, 'b': 1, 'c': 0}

    H = BooleanConstraints().add_constraint_eq_OR(
        'c', 'a', 'b').add_constraint_NOR('a', 'b').to_penalty()
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 0, 'b': 0, 'c': 0}

    H = BooleanConstraints().add_constraint_eq_XOR(
        'c', 'a', 'b').add_constraint_XNOR(
        'a', 'b').add_constraint_BUFFER('a').to_penalty()
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 1, 'b': 1, 'c': 0}

    H = BooleanConstraints().add_constraint_eq_NOT(
        'a', 'b').add_constraint_BUFFER('a').to_penalty()
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 1, 'b': 0}

    H = BooleanConstraints().add_constraint_NAND('a', 'b').add_constraint_NOT(
        'a').add_constraint_OR(
        'a', 'b').add_constraint_eq_BUFFER('a', 'c').to_penalty()
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 0, 'b': 1, 'c': 0}

    H = BooleanConstraints().add_constraint_XOR(
        'a', 'b').add_constraint_eq_NOR('b', 'a', 'c').add_constraint_BUFFER(
            'c').add_constraint_eq_BUFFER('a', 'c').to_penalty()
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 1, 'b': 0, 'c': 1}

    H = BooleanConstraints().add_constraint_eq_AND(
        'c', 'a', 'b').add_constraint_eq_XNOR(
        'c', 'a', 'b').add_constraint_BUFFER('c').to_penalty()
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'c': 1, 'a': 1, 'b': 1}

    # add_constraint_eq_NOR
    H = BooleanConstraints().add_constraint_eq_NOR(
        0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not any(sol[i] for i in range(1, 6))) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_NOR(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if (not any(sol[i] for i in range(1, 6))) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_eq_NOR(0, 1, 2).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not any(sol[i] for i in range(1, 3))) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_NOR(0, 1, 2)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if (not any(sol[i] for i in range(1, 3))) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # add_constraint_eq_OR
    H = BooleanConstraints().add_constraint_eq_OR(
        0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert any(sol[i] for i in range(1, 6)) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_OR(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if any(sol[i] for i in range(1, 6)) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_eq_OR(0, 1, 2).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert any(sol[i] for i in range(1, 3)) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_OR(0, 1, 2)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if any(sol[i] for i in range(1, 3)) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # add_constraint_eq_NAND
    H = BooleanConstraints().add_constraint_eq_NAND(
        0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not all(sol[i] for i in range(1, 6))) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_NAND(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if (not all(sol[i] for i in range(1, 6))) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_eq_NAND(0, 1, 2).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not all(sol[i] for i in range(1, 3))) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_NAND(0, 1, 2)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if (not all(sol[i] for i in range(1, 3))) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # add_constraint_eq_AND
    H = BooleanConstraints().add_constraint_eq_AND(
        0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert all(sol[i] for i in range(1, 6)) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_AND(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if all(sol[i] for i in range(1, 6)) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_eq_AND(0, 1, 2).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert all(sol[i] for i in range(1, 3)) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_AND(0, 1, 2)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if all(sol[i] for i in range(1, 3)) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # add_constraint_eq_XOR
    H = BooleanConstraints().add_constraint_eq_XOR(
        0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert sum(sol[i] for i in range(1, 6)) % 2 == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_XOR(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if sum(sol[i] for i in range(1, 6)) % 2 == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_eq_XOR(0, 1, 2).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert sum(sol[i] for i in range(1, 3)) % 2 == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_XOR(0, 1, 2)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if sum(sol[i] for i in range(1, 3)) % 2 == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # add_constraint_eq_XNOR
    H = BooleanConstraints().add_constraint_eq_XNOR(
        0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not sum(sol[i] for i in range(1, 6)) % 2) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_XNOR(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if (not sum(sol[i] for i in range(1, 6)) % 2) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_eq_XNOR(0, 1, 2).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not sum(sol[i] for i in range(1, 3)) % 2) == sol[0]
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_eq_XNOR(0, 1, 2)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if (not sum(sol[i] for i in range(1, 3)) % 2) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # NOR
    H = BooleanConstraints().add_constraint_NOR(0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not any(sol[i] for i in range(6))
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_NOR(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if not any(sol[i] for i in range(6)):
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_NOR(0, 1).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not any(sol[i] for i in range(2))
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_NOR(0, 1)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if not any(sol[i] for i in range(2)):
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # OR
    H = BooleanConstraints().add_constraint_OR(0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert any([sol[i] for i in range(6)])  # list so all branches covered
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_OR(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if any(sol[i] for i in range(6)):
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_OR(0, 1).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert any([sol[i] for i in range(2)])  # list so all branches covered
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_OR(0, 1)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if any(sol[i] for i in range(2)):
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # NAND
    H = BooleanConstraints().add_constraint_NAND(0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        # list so all branches covered
        assert not all([sol[i] for i in range(6)])
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_NAND(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if not all(sol[i] for i in range(6)):
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_NAND(0, 1).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        # list so all branches covered
        assert not all([sol[i] for i in range(2)])
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_NAND(0, 1)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if not all(sol[i] for i in range(2)):
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # AND
    H = BooleanConstraints().add_constraint_AND(0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert all(sol[i] for i in range(6))
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_AND(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if all(sol[i] for i in range(6)):
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_AND(0, 1).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert all(sol[i] for i in range(2))
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_AND(0, 1)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if all(sol[i] for i in range(2)):
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # XOR
    H = BooleanConstraints().add_constraint_XOR(0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert sum(sol[i] for i in range(6)) % 2
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_XOR(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if sum(sol[i] for i in range(6)) % 2:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_XOR(0, 1).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert sum(sol[i] for i in range(2)) % 2
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_XOR(0, 1)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if sum(sol[i] for i in range(2)) % 2:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    # XNOR
    H = BooleanConstraints().add_constraint_XNOR(0, 1, 2, 3, 4, 5).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not sum(sol[i] for i in range(6)) % 2
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_XNOR(0, 1, 2, 3, 4, 5)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if not sum(sol[i] for i in range(6)) % 2:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)

    H = BooleanConstraints().add_constraint_XNOR(0, 1).to_penalty()
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not sum(sol[i] for i in range(2)) % 2
        assert not H.value(sol)
    H = BooleanConstraints().add_constraint_XNOR(0, 1)
    C = H.to_penalty()
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if not sum(sol[i] for i in range(2)) % 2:
            assert H.is_solution_valid(sol)
            assert not C.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert C.value(sol)


def test_le_right_bounds():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_le_zero(  # one sided bounds
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3},
        lam=lam, log_trick=False, bounds=(None, 1)
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8
    model = P + C.to_penalty()

    problem = model.subs(lam, .5)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_le_left_bounds():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_le_zero(  # one sided bounds
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3},
        lam=lam, log_trick=False, bounds=(-3, None)
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8
    model = P + C.to_penalty()

    problem = model.subs(lam, .5)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_ge_bounds():

    lam = Symbol("lam")

    P = PUBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    C = BooleanConstraints().add_constraint_ge_zero(  # one sided bounds
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3},
        lam=lam, log_trick=False, bounds=(-1, 3)
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8
    model = P + C.to_penalty()

    problem = model.subs(lam, .5)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_eq_special_bounds():

    P = {('a',): 1, ('a', 'b'): 1, (): -2}
    assert BooleanConstraints().add_constraint_eq_zero(
        P).to_penalty() == -PUBO(P)


def test_booleanconstraints_special_constraint_le():

    # first one

    H = BooleanConstraints().add_constraint_le_zero(
        {(0,): 1, (1,): 1, (2,): 1, (): -1}).to_penalty()
    assert H == {(i, j): 1 for i in range(3) for j in range(i+1, 3)}

    H = BooleanConstraints().add_constraint_le_zero(
        {(0,): 1, (1,): 1, (2, 3): 1, (): -1}).to_penalty()
    assert H == {(0, 1): 1, (0, 2, 3): 1, (1, 2, 3): 1}

    H = BooleanConstraints().add_constraint_lt_zero(
        {(0,): 1, (1,): 1, (2,): 1, (): -2}).to_penalty()
    assert H == {(i, j): 1 for i in range(3) for j in range(i+1, 3)}

    H = BooleanConstraints().add_constraint_lt_zero(
        {(0,): 1, (1,): 1, (2, 3): 1, (): -2}).to_penalty()
    assert H == {(0, 1): 1, (0, 2, 3): 1, (1, 2, 3): 1}

    H = BooleanConstraints().add_constraint_ge_zero(
        {(0,): -1, (1,): -1, (2,): -1, (): 1}).to_penalty()
    assert H == {(i, j): 1 for i in range(3) for j in range(i+1, 3)}

    H = BooleanConstraints().add_constraint_ge_zero(
        {(0,): -1, (1,): -1, (2, 3): -1, (): 1}).to_penalty()
    assert H == {(0, 1): 1, (0, 2, 3): 1, (1, 2, 3): 1}

    H = BooleanConstraints().add_constraint_gt_zero(
        {(0,): -1, (1,): -1, (2,): -1, (): 2}).to_penalty()
    assert H == {(i, j): 1 for i in range(3) for j in range(i+1, 3)}

    H = BooleanConstraints().add_constraint_gt_zero(
        {(0,): -1, (1,): -1, (2, 3): -1, (): 2}).to_penalty()
    assert H == {(0, 1): 1, (0, 2, 3): 1, (1, 2, 3): 1}

    # second one

    desired = PUBO(
        {(0,): 1, (1,): 1, (2,): 1, ('__a0',): -1, ('__a1',): -1}
    ) ** 2
    H1 = BooleanConstraints().add_constraint_le_zero(
        {(0,): 1, (1,): 1, (2,): 1, (): -2},
        log_trick=False
    ).to_penalty()
    H2 = BooleanConstraints().add_constraint_ge_zero(
        {(0,): -1, (1,): -1, (2,): -1, (): 2},
        log_trick=False
    ).to_penalty()
    H3 = BooleanConstraints().add_constraint_lt_zero(
        {(0,): 1, (1,): 1, (2,): 1, (): -3},
        log_trick=False
    ).to_penalty()
    H4 = BooleanConstraints().add_constraint_gt_zero(
        {(0,): -1, (1,): -1, (2,): -1, (): 3},
        log_trick=False
    ).to_penalty()
    assert H1 == H2 == H3 == H4 == desired

    # third one

    H1 = BooleanConstraints().add_constraint_le_zero(
        {(0,): -1, (1,): -1, (): 1}).to_penalty()
    H2 = BooleanConstraints().add_constraint_ge_zero(
        {(0,): 1, (1,): 1, (): -1}).to_penalty()
    H3 = BooleanConstraints().add_constraint_lt_zero(
        {(0,): -1, (1,): -1}).to_penalty()
    H4 = BooleanConstraints().add_constraint_gt_zero(
        {(0,): 1, (1,): 1}).to_penalty()
    desired = BooleanConstraints().add_constraint_OR(0, 1).to_penalty()
    assert H1 == H2 == H3 == H4 == desired

    H1 = BooleanConstraints().add_constraint_le_zero(
        {(0, 1): -1, (2, 3, 4): -1, (): 1}).to_penalty()
    H2 = BooleanConstraints().add_constraint_ge_zero(
        {(0, 1): 1, (2, 3, 4): 1, (): -1}).to_penalty()
    H3 = BooleanConstraints().add_constraint_lt_zero(
        {(0, 1): -1, (2, 3, 4): -1}).to_penalty()
    H4 = BooleanConstraints().add_constraint_gt_zero(
        {(0, 1): 1, (2, 3, 4): 1}).to_penalty()
    desired = BooleanConstraints().add_constraint_OR(
        {(0, 1): 1}, {(2, 3, 4): 1}).to_penalty()
    assert H1 == H2 == H3 == H4 == desired

    # fourth one

    desired = {(0,): 1, (0, 1): -1}
    H1 = BooleanConstraints().add_constraint_le_zero(
        {(0,): 1, (1,): -1}).to_penalty()
    H2 = BooleanConstraints().add_constraint_lt_zero(
        {(0,): 1, (1,): -1, (): -1}).to_penalty()
    H3 = BooleanConstraints().add_constraint_ge_zero(
        {(0,): -1, (1,): 1}).to_penalty()
    H4 = BooleanConstraints().add_constraint_gt_zero(
        {(0,): -1, (1,): 1, (): 1}).to_penalty()
    assert H1 == H2 == H3 == H4 == desired

    desired = {(0, 1): 1, (0, 1, 2, 3, 4): -1}
    H1 = BooleanConstraints().add_constraint_le_zero(
        {(0, 1): 1, (2, 3, 4): -1}).to_penalty()
    H2 = BooleanConstraints().add_constraint_lt_zero(
        {(0, 1): 1, (2, 3, 4): -1, (): -1}).to_penalty()
    H3 = BooleanConstraints().add_constraint_ge_zero(
        {(0, 1): -1, (2, 3, 4): 1}).to_penalty()
    H4 = BooleanConstraints().add_constraint_gt_zero(
        {(0, 1): -1, (2, 3, 4): 1, (): 1}).to_penalty()
    assert H1 == H2 == H3 == H4 == desired


def test_booleanconstraints_special_constraint_eq():

    x, y, z = boolean_var('x'), boolean_var('y'), boolean_var('z')
    H1 = BooleanConstraints().add_constraint_eq_zero(z - x * y).to_penalty()
    H2 = BooleanConstraints().add_constraint_eq_zero(x * y - z).to_penalty()
    H3 = BooleanConstraints().add_constraint_eq_AND(z, x, y).to_penalty()
    assert H1 == H2 == H3
