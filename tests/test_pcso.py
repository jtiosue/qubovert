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
Contains tests for the PCSO class.
"""

from qubovert import PCSO, spin_var, integer_var
from qubovert.utils import (
    solve_qubo_bruteforce, solve_quso_bruteforce,
    solve_pubo_bruteforce, solve_puso_bruteforce,
    puso_value, pubo_to_puso, boolean_to_spin,
    QUBOVertWarning
)
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises, assert_warns


""" TESTS FOR THE METHODS THAT PCSO INHERITS FROM PUSO """


class Problem:

    def __init__(self, problem, solution, obj):

        self.problem, self.solution, self.obj = problem, solution, obj

    def is_valid(self, e, solution, spin):

        sol = self.problem.convert_solution(solution, spin)
        return all((
            self.problem.is_solution_valid(sol),
            self.problem.is_solution_valid(solution),
            sol == self.solution,
            allclose(e, self.obj)
        ))

    def runtests(self):

        assert self.problem.solve_bruteforce() == self.solution

        e, sol = solve_qubo_bruteforce(self.problem.to_qubo())
        assert self.is_valid(e, sol, False)

        e, sol = solve_quso_bruteforce(self.problem.to_quso())
        assert self.is_valid(e, sol, True)

        for deg in (None,) + tuple(range(2, self.problem.degree + 1)):

            e, sol = solve_puso_bruteforce(self.problem.to_puso(deg))
            assert self.is_valid(e, sol, True)

            e, sol = solve_pubo_bruteforce(self.problem.to_pubo(deg))
            assert self.is_valid(e, sol, False)

        assert (
            self.problem.value(self.solution) ==
            puso_value(self.solution, self.problem) ==
            e
        )


def test_pcso_on_quso():

    problem = PCSO({('a',): -1, ('b',): 2,
                   ('a', 'b'): -3, ('b', 'c'): -4, (): -2})
    solution = {'c': -1, 'b': -1, 'a': -1}
    obj = -10

    Problem(problem, solution, obj).runtests()


def test_pcso_on_deg_3_puso():

    problem = PCSO({('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4,
                   (): -2, (0, 1, 2): 1, (0,): 1, (1,): 1, (2,): 1})
    solution = {'c': -1, 'b': -1, 'a': -1, 0: -1, 1: -1, 2: -1}
    obj = -14

    Problem(problem, solution, obj).runtests()


def test_pcso_on_deg_5_puso():

    problem = PCSO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        (0, 1, 2): 1, (0,): -1, (1,): -2, (2,): 1, ('a', 0, 4, 'b', 'c'): -3,
        (4, 2, 3, 'a', 'b'): 2, (4, 2, 3, 'b'): -1, ('c',): 4, (3,): 1
    })
    solution = {0: 1, 1: 1, 'c': -1, 2: -1, 4: -1, 3: -1, 'b': -1, 'a': -1}
    obj = -26

    Problem(problem, solution, obj).runtests()


# testing methods

def test_pcso_checkkey():

    with assert_raises(KeyError):
        PCSO({0: -1})


def test_quso_default_valid():

    d = PCSO()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(): 1}

    d = PCSO()
    assert d[(0, 1)] == 0
    d[(0, 1)] += 1
    assert d == {(0, 1): 1}


def test_quso_remove_value_when_zero():

    d = PCSO()
    d[(0, 1)] += 1
    d[(0, 1)] -= 1
    assert d == {}


def test_quso_reinitialize_dictionary():

    d = PCSO({(0, 0): 1, ('1', 0): 2, (2, 0): 0, (0, '1'): 1})
    assert d in ({(): 1, ('1', 0): 3}, {(): 1, (0, '1'): 3})


def test_quso_update():

    d = PCSO({('0',): 1, ('0', 1): 2})
    d.update({('0', '0'): 0, (1, '0'): 1, (1, 1): -1})
    assert d in ({('0',): 1, (): -1, (1, '0'): 1},
                 {('0',): 1, (): -1, ('0', 1): 1})

    d = PCSO({(0, 0): 1, (0, 1): 2})
    d.update(PCSO({(1, 0): 1, (1, 1): -1}))
    d -= 1
    assert d == PCSO({(0, 1): 1, (): -2})

    assert d.offset == -2


def test_quso_num_binary_variables():

    d = PCSO({(0,): 1, (0, 3): 2})
    assert d.num_binary_variables == 2
    assert d.max_index == 1


def test_num_terms():

    d = PCSO({(0,): 1, (0, 3): 2, (0, 2): -1})
    assert d.num_terms == len(d)


def test_pcso_degree():

    d = PCSO()
    assert d.degree == 0
    d[(0,)] += 2
    assert d.degree == 1
    d[(1,)] -= 3
    assert d.degree == 1
    d[(1, 2)] -= 2
    assert d.degree == 2
    d[(1, 2, 4)] -= 2
    assert d.degree == 3
    d[(1, 2, 4, 5, 6)] += 2
    assert d.degree == 5


def test_quso_addition():

    temp = PCSO({('0', '0'): 1, ('0', 1): 2})
    temp1 = {('0',): -1, (1, '0'): 3}
    temp2 = {(1, '0'): 5, (): 1, ('0',): -1}, {('0', 1): 5, (): 1, ('0',): -1}
    temp3 = {(): 1, (1, '0'): -1, ('0',): 1}, {(): 1, ('0', 1): -1, ('0',): 1}

    # constant
    d = temp.copy()
    d += 5
    assert d in ({(1, '0'): 2, (): 6}, {('0', 1): 2, (): 6})

    # __add__
    d = temp.copy()
    g = d + temp1
    assert g in temp2

    # __iadd__
    d = temp.copy()
    d += temp1
    assert d in temp2

    # __radd__
    d = temp.copy()
    g = temp1 + d
    assert g in temp2

    # __sub__
    d = temp.copy()
    g = d - temp1
    assert g in temp3

    # __isub__
    d = temp.copy()
    d -= temp1
    assert d in temp3

    # __rsub__
    d = temp.copy()
    g = temp1 - d
    assert g == PCSO(temp3[0])*-1


def test_quso_multiplication():

    temp = PCSO({('0', '0'): 1, ('0', 1): 2})
    temp1 = {(): 3, (1, '0'): 6}, {(): 3, ('0', 1): 6}
    temp2 = {(): .5, (1, '0'): 1}, {(): .5, ('0', 1): 1}
    temp3 = {(1, '0'): 1}, {('0', 1): 1}

    # constant
    d = temp.copy()
    d += 3
    d *= -2
    assert d in ({(1, '0'): -4, (): -8}, {('0', 1): -4, (): -8})

    # __mul__
    d = temp.copy()
    g = d * 3
    assert g in temp1

    d = temp.copy()
    g = d * 0
    assert g == {}

    # __imul__
    d = temp.copy()
    d *= 3
    assert d in temp1

    d = temp.copy()
    d *= 0
    assert d == {}

    # __rmul__
    d = temp.copy()
    g = 3 * d
    assert g in temp1

    d = temp.copy()
    g = 0 * d
    assert g == {}

    # __truediv__
    d = temp.copy()
    g = d / 2
    assert g in temp2

    # __itruediv__
    d = temp.copy()
    d /= 2
    assert d in temp2

    # __floordiv__
    d = temp.copy()
    g = d // 2
    assert g in temp3

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d in temp3

    # __mul__ but with dict
    d = temp.copy()
    d *= {(1,): 2, ('0', '0'): -1}
    assert d in ({(1,): 2, (): -1, ('0',): 4, ('0', 1): -2},
                 {(1,): 2, (): -1, ('0',): 4, (1, '0'): -2})

    # __pow__
    d = temp.copy()
    d -= 2
    d **= 2
    assert d in ({(): 5, ('0', 1): -4}, {(): 5, (1, '0'): -4})

    d = temp.copy()
    assert d ** 2 == d * d
    assert d ** 3 == d * d * d

    d = PCSO({('0', 1): 1, ('1', 2): -1})**2
    assert d ** 4 == d * d * d * d


def test_properties():

    temp = PCSO({('0', '0'): 1, ('0', 1): 2})
    assert temp.offset == 1

    d = PCSO()
    d[(0,)] += 1
    d[(1,)] += 2
    assert d == d.to_quso() == {(0,): 1, (1,): 2}
    assert d.mapping == d.reverse_mapping == {0: 0, 1: 1}

    d.set_mapping({1: 0, 0: 1})
    assert d.to_quso() == {(1,): 1, (0,): 2}
    assert d.mapping == d.reverse_mapping == {0: 1, 1: 0}

    assert d.constraints == {}
    temp = d.copy()
    d.add_constraint_eq_zero(temp)
    assert d.constraints == {'eq': [temp]}

    # an old bug
    d = PCSO()
    d.set_mapping({0: 0})
    d[(0,)] += 1
    assert d.num_binary_variables == 1
    assert d.variables == {0}


def test_round():

    d = PCSO({(0,): 3.456, (1,): -1.53456})

    assert round(d) == {(0,): 3, (1,): -2}
    assert round(d, 1) == {(0,): 3.5, (1,): -1.5}
    assert round(d, 2) == {(0,): 3.46, (1,): -1.53}
    assert round(d, 3) == {(0,): 3.456, (1,): -1.535}


def test_normalize():

    temp = {(0,): 4, (1,): -2}
    d = PCSO(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}

    temp = {(0,): -4, (1,): 2}
    d = PCSO(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = PCSO()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}

    d.add_constraint_eq_zero({(0,): a, (1,): -b}, bounds=(-1, 1))
    d.simplify()
    assert d == {(0,): -1.*a, (0, 1): -2.*a*b + 2.,
                 (1,): 1.*b, (): 1.*a**2 + 1.*b**2}
    assert d.subs(a, 0) == {(0, 1): 2, (1,): 1.*b, (): 1.*b**2}
    assert d.subs({a: 0, b: 2}) == {(0, 1): 2, (1,): 2, (): 4}


def test_init_pcso():

    H = PCSO().add_constraint_eq_zero({(0,): 1, (1,): -2}, bounds=(-1, 1))
    d = PCSO(H)
    assert H.constraints == d.constraints


def test_convert_solution_all_1s():

    d = PCSO({(0,): 1})
    assert d.convert_solution({0: 0}) == {0: 1}
    assert d.convert_solution({0: -1}) == {0: -1}
    assert d.convert_solution({0: 1}) == {0: 1}
    assert d.convert_solution({0: 1}, False) == {0: -1}


def test_spin_var():

    z = [spin_var(i) for i in range(5)]
    assert all(z[i] == {(i,): 1} for i in range(5))
    assert z[0] * z[1] * z[2] == {(0, 1, 2): 1}
    assert sum(z) == {(i,): 1 for i in range(5)}
    assert isinstance(z[0], PCSO)
    assert all(z[i].name == i for i in range(5))


""" TESTS FOR THE CONSTRAINT METHODS """


def test_pcso_eq_constraint():

    lam = Symbol('lam')

    H = PCSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    H.add_constraint_eq_zero(
        pubo_to_puso({('a',): 1, ('b',): 1, ('b', 'c'): -1}),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4

    problem = H.subs(lam, 1)
    sol = problem.solve_bruteforce()
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, False)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = H.subs(lam, 10)
    sol = problem.solve_bruteforce()
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_pcso_ne_constraint_logtrick():

    for i in range(1 << 4):
        P = pubo_to_puso(integer_var('a', 4)) - i
        H = PCSO().add_constraint_ne_zero(P)
        for sol in H.solve_bruteforce(True):
            assert P.value(sol)


def test_pcso_ne_constraint():

    for i in range(1 << 3):
        P = pubo_to_puso(integer_var('a', 3)) - i
        H = PCSO().add_constraint_ne_zero(P, log_trick=False)
        for sol in H.solve_bruteforce(True):
            assert P.value(sol)


def test_pcso_lt_constraint_logtrick():

    lam = Symbol("lam")

    H = PCSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    H.add_constraint_lt_zero(
        pubo_to_puso({('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3}),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4

    problem = H.subs(lam, 1)
    sol = problem.remove_ancilla_from_solution(problem.solve_bruteforce())
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = H.subs(lam, 10)
    sol = problem.solve_bruteforce()
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_pcso_lt_constraint():

    lam = Symbol("lam")

    H = PCSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    H.add_constraint_lt_zero(
        pubo_to_puso({('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3}),
        lam=lam, log_trick=False
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4

    problem = H.subs(lam, 1)
    sol = problem.remove_ancilla_from_solution(problem.solve_bruteforce())
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = H.subs(lam, 10)
    sol = problem.solve_bruteforce()
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_pcso_le_constraint_logtrick():

    lam = Symbol("lam")

    H = PCSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    H.add_constraint_le_zero(
        pubo_to_puso(
            {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3}
        ),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 1, 'd': 0})
    obj = -8

    problem = H.subs(lam, .5)
    sol = problem.remove_ancilla_from_solution(problem.solve_bruteforce())
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = H.subs(lam, 10)
    sol = problem.solve_bruteforce()
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_pcso_le_constraint():

    lam = Symbol("lam")

    H = PCSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    H.add_constraint_le_zero(
        pubo_to_puso(
            {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3}
        ),
        lam=lam, log_trick=False
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 1, 'd': 0})
    obj = -8

    problem = H.subs(lam, .5)
    sol = problem.remove_ancilla_from_solution(problem.solve_bruteforce())
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = H.subs(lam, 10)
    sol = problem.solve_bruteforce()
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_pcso_gt_constraint_logtrick():

    lam = Symbol("lam")

    H = PCSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    H.add_constraint_gt_zero(
        pubo_to_puso({('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3}),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4

    problem = H.subs(lam, 1)
    sol = problem.remove_ancilla_from_solution(problem.solve_bruteforce())
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = H.subs(lam, 10)
    sol = problem.solve_bruteforce()
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_pcso_gt_constraint():

    lam = Symbol("lam")

    H = PCSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    }))
    H.add_constraint_gt_zero(
        pubo_to_puso({('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3}),
        lam=lam, log_trick=False
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 0})
    obj = -4

    problem = H.subs(lam, 1)
    sol = problem.remove_ancilla_from_solution(problem.solve_bruteforce())
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = H.subs(lam, 10)
    sol = problem.solve_bruteforce()
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_pcso_ge_constraint_logtrick():

    lam = Symbol("lam")

    H = PCSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    H.add_constraint_ge_zero(
        pubo_to_puso(
            {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3}
        ),
        lam=lam
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 1, 'd': 0})
    obj = -8

    problem = H.subs(lam, .5)
    sol = problem.remove_ancilla_from_solution(problem.solve_bruteforce())
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = H.subs(lam, 10)
    sol = problem.solve_bruteforce()
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_pcso_ge_constraint():

    lam = Symbol("lam")

    H = PCSO(pubo_to_puso({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    }))
    H.add_constraint_ge_zero(
        pubo_to_puso(
            {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3}
        ),
        lam=lam, log_trick=False
    )
    solution = boolean_to_spin({'c': 1, 'b': 1, 'a': 1, 'd': 0})
    obj = -8

    problem = H.subs(lam, .5)
    sol = problem.remove_ancilla_from_solution(problem.solve_bruteforce())
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol, spin=False)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = H.subs(lam, 10)
    sol = problem.solve_bruteforce()
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    sol = problem.remove_ancilla_from_solution(sol)
    assert all((
        problem.is_solution_valid(sol),
        sol == solution,
        allclose(e, obj)
    ))


def test_pcso_constraints_warnings():

    with assert_warns(QUBOVertWarning):  # qlwayss satisfied
        PCSO().add_constraint_eq_zero({(): 0})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCSO().add_constraint_eq_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCSO().add_constraint_eq_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCSO().add_constraint_lt_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCSO().add_constraint_lt_zero({(): 1, (0,): -1})

    with assert_warns(QUBOVertWarning):  # always satisfied
        PCSO().add_constraint_lt_zero({(): -1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCSO().add_constraint_le_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # always satisfied
        PCSO().add_constraint_le_zero({(): -1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCSO().add_constraint_gt_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCSO().add_constraint_gt_zero({(): -1, (0,): 1})

    with assert_warns(QUBOVertWarning):  # always satisfied
        PCSO().add_constraint_gt_zero({(): 1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCSO().add_constraint_ge_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # always satisfied
        PCSO().add_constraint_ge_zero({(): 1, (0,): .5})
