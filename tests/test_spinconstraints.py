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
Contains tests for the SpinConstraints class.
"""

from qubovert import SpinConstraints, PUSO, integer_var, PUBO
from qubovert.utils import (
    solve_pubo_bruteforce, solve_puso_bruteforce, pubo_to_puso,
    boolean_to_spin, QUBOVertWarning
)
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises, assert_warns


def test_constraints():

    d = SpinConstraints()
    assert d.constraints == {}
    temp = {(0,): 1, (1,): 1}
    d.add_constraint_eq_zero(temp)
    assert d.constraints == {'eq': [temp]}
    str(d)
    repr(d)


def test_round():

    d = SpinConstraints()
    d.add_constraint_eq_zero({(0,): 3.456, (1,): -1.53456})

    assert round(d).constraints == {"eq": [{(0,): 3, (1,): -2}]}
    assert round(d, 1).constraints == {"eq": [{(0,): 3.5, (1,): -1.5}]}
    assert round(d, 2).constraints == {"eq": [{(0,): 3.46, (1,): -1.53}]}
    assert round(d, 3).constraints == {"eq": [{(0,): 3.456, (1,): -1.535}]}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = SpinConstraints()
    d.add_constraint_eq_zero({(0,): a, (1,): -b}, bounds=(-1, 1))
    t = d.to_penalty()
    t.simplify()
    assert t == {(): 1.*a**2 + 1.*b**2, (0, 1): -2.*a*b}
    t = d.subs(a, 0).to_penalty()
    t.simplify()
    assert t == {(): 1.*b**2}
    t = d.subs({a: 0, b: 2}).to_penalty()
    t.simplify()
    assert t == {(): 4.}


""" TESTS FOR THE CONSTRAINT METHODS """


def test_spinconstraints_eq_constraint():

    lam = Symbol('lam')

    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    C = SpinConstraints().add_constraint_eq_zero(
        pubo_to_puso({('a',): 1, ('b',): 1, ('b', 'c'): -1}),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4
    model = H + C.to_penalty()

    problem = model.subs(lam, 1)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, False)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = model.subs(lam, 10)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    assert all((
        C.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))

    # lam = 0
    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    C = SpinConstraints().add_constraint_eq_zero(
        pubo_to_puso({('a',): 1, ('b',): 1, ('b', 'c'): -1}),
        lam=0
    )
    model = H + C.to_penalty()

    e, sol = solve_pubo_bruteforce(model.to_pubo())
    sol = model.convert_solution(sol, False)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_spinconstraints_ne_constraint_logtrick():

    for i in range(1 << 4):
        P = pubo_to_puso(integer_var('a', 4)) - i
        H = SpinConstraints().add_constraint_ne_zero(P).to_penalty()
        for sol in H.solve_bruteforce(True):
            assert P.value(sol)

    for i in range(1 << 2):
        P = pubo_to_puso(integer_var('a', 2)) - i
        H = SpinConstraints().add_constraint_ne_zero(P).to_penalty()
        for sol in solve_puso_bruteforce(H, True)[1]:
            assert P.value(sol)


def test_spinconstraints_ne_constraint():

    for i in range(1 << 2):
        P = pubo_to_puso(integer_var('a', 2)) - i
        H = SpinConstraints().add_constraint_ne_zero(
            P, log_trick=False).to_penalty()
        for sol in H.solve_bruteforce(True):
            assert P.value(sol)

    for i in range(1 << 2):
        P = pubo_to_puso(integer_var('a', 2)) - i
        H = SpinConstraints().add_constraint_ne_zero(
            P, lam=0, log_trick=False)
        assert H.constraints["ne"] == [P]  


def test_spinconstraints_lt_constraint_logtrick():

    lam = Symbol("lam")

    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    C = SpinConstraints().add_constraint_lt_zero(
        pubo_to_puso({('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3}),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4
    model = H + C.to_penalty()

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
    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    C = SpinConstraints().add_constraint_lt_zero(
        pubo_to_puso({('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3}),
        lam=0
    )
    model = H + C.to_penalty()
    e, sol = solve_pubo_bruteforce(model.to_pubo())
    sol = model.convert_solution(sol, spin=False)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_spinconstraints_lt_constraint():

    lam = Symbol("lam")

    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    C = SpinConstraints().add_constraint_lt_zero(
        pubo_to_puso({('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3}),
        lam=lam, log_trick=False
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4
    model = H + C.to_penalty()

    problem = model.subs(lam, 1)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
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


def test_spinconstraints_le_constraint_logtrick():

    lam = Symbol("lam")

    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    C = SpinConstraints().add_constraint_le_zero(
        pubo_to_puso(
            {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3}
        ),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 1, 'd': 0})
    obj = -8
    model = H + C.to_penalty()

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
    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    C = SpinConstraints().add_constraint_le_zero(
        pubo_to_puso(
            {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3}
        ),
        lam=0
    )
    model = H + C.to_penalty()
    e, sol = solve_pubo_bruteforce(model.to_pubo())
    sol = model.convert_solution(sol, spin=False)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_spinconstraints_le_constraint():

    lam = Symbol("lam")

    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    C = SpinConstraints().add_constraint_le_zero(
        pubo_to_puso(
            {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3}
        ),
        lam=lam, log_trick=False
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 1, 'd': 0})
    obj = -8
    model = H + C.to_penalty()

    problem = model.subs(lam, .5)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
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


def test_spinconstraints_gt_constraint_logtrick():

    lam = Symbol("lam")

    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    C = SpinConstraints().add_constraint_gt_zero(
        pubo_to_puso({('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3}),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4
    model = H + C.to_penalty()

    problem = model.subs(lam, 1)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
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
    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    C = SpinConstraints().add_constraint_gt_zero(
        pubo_to_puso({('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3}),
        lam=0
    )
    model = H + C.to_penalty()

    e, sol = solve_pubo_bruteforce(model.to_pubo())
    sol = model.convert_solution(sol, spin=False)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_spinconstraints_gt_constraint():

    lam = Symbol("lam")

    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    C = SpinConstraints().add_constraint_gt_zero(
        pubo_to_puso({('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3}),
        lam=lam, log_trick=False
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4
    model = H + C.to_penalty()

    problem = model.subs(lam, 1)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
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


def test_spinconstraints_ge_constraint_logtrick():

    lam = Symbol("lam")

    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    C = SpinConstraints().add_constraint_ge_zero(
        pubo_to_puso(
            {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3}
        ),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 1, 'd': 0})
    obj = -8
    model = H + C.to_penalty()

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
    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    C = SpinConstraints().add_constraint_ge_zero(
        pubo_to_puso(
            {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3}
        ),
        lam=0
    )
    model = H + C.to_penalty()

    e, sol = solve_pubo_bruteforce(model.to_pubo())
    sol = model.convert_solution(sol, spin=False)
    sol = C.remove_ancilla_from_solution(sol)
    assert all((
        not C.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))


def test_spinconstraints_ge_constraint():

    lam = Symbol("lam")

    H = PUSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    C = SpinConstraints().add_constraint_ge_zero(
        pubo_to_puso(
            {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3}
        ),
        lam=lam, log_trick=False
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 1, 'd': 0})
    obj = -8
    model = H + C.to_penalty()

    problem = model.subs(lam, .5)
    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
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


def test_spinconstraints_constraints_warnings():

    with assert_warns(QUBOVertWarning):  # qlwayss satisfied
        SpinConstraints().add_constraint_eq_zero({(): 0})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        SpinConstraints().add_constraint_eq_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        SpinConstraints().add_constraint_eq_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        SpinConstraints().add_constraint_lt_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        SpinConstraints().add_constraint_lt_zero({(): 1, (0,): -1})

    with assert_warns(QUBOVertWarning):  # always satisfied
        SpinConstraints().add_constraint_lt_zero({(): -1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        SpinConstraints().add_constraint_le_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # always satisfied
        SpinConstraints().add_constraint_le_zero({(): -1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        SpinConstraints().add_constraint_gt_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        SpinConstraints().add_constraint_gt_zero({(): -1, (0,): 1})

    with assert_warns(QUBOVertWarning):  # always satisfied
        SpinConstraints().add_constraint_gt_zero({(): 1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        SpinConstraints().add_constraint_ge_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # always satisfied
        SpinConstraints().add_constraint_ge_zero({(): 1, (0,): .5})
