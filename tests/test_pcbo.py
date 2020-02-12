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
Contains tests for the PCBO class.
"""

from qubovert import PCBO, boolean_var, integer_var
from qubovert.utils import (
    solve_qubo_bruteforce, solve_quso_bruteforce,
    solve_pubo_bruteforce, solve_puso_bruteforce,
    pubo_value, decimal_to_boolean, QUBOVertWarning
)
from sympy import Symbol
from numpy import allclose
from numpy.testing import assert_raises, assert_warns


""" TESTS FOR THE METHODS THAT PCBO INHERITS FROM PUBO """


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
            pubo_value(self.solution, self.problem) ==
            e
        )


def test_pcbo_on_qubo():

    problem = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    solution = {'c': 1, 'b': 1, 'a': 1}
    obj = -8

    Problem(problem, solution, obj).runtests()


def test_pcbo_on_deg_3_pubo():

    problem = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        (0, 1, 2): 1, (0,): -1, (1,): -2, (2,): 1
    })
    solution = {'c': 1, 'b': 1, 'a': 1, 0: 1, 1: 1, 2: 0}
    obj = -11

    Problem(problem, solution, obj).runtests()


def test_pcbo_on_deg_5_pubo():

    problem = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        (0, 1, 2): 1, (0,): -1, (1,): -2, (2,): 1, ('a', 0, 4, 'b', 'c'): -3,
        (4, 2, 3, 'a', 'b'): 2, (4, 2, 3, 'b'): -1, ('c',): 4, (3,): 1,
        (0, 1): -2
    })
    solution = {0: 1, 1: 1, 'c': 1, 2: 0, 4: 1, 3: 0, 'b': 1, 'a': 1}
    obj = -12

    Problem(problem, solution, obj).runtests()


# testing methods

def test_pcbo_checkkey():

    with assert_raises(KeyError):
        PCBO({0: -1})


def test_pcbo_default_valid():

    d = PCBO()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0,): 1}


def test_pcbo_remove_value_when_zero():

    d = PCBO()
    d[(0, 0, 1, 2)] += 1
    d[(0, 1, 2)] -= 1
    assert d == {}


def test_pcbo_reinitialize_dictionary():

    d = PCBO({(0, 0): 1, ('1', 0): 2, (2, 0): 0, (0, '1'): 1})
    assert d in ({(0,): 1, ('1', 0): 3}, {(0,): 1, (0, '1'): 3})


def test_pcbo_update():

    d = PCBO({('0',): 1, ('0', 1): 2})
    d.update({('0', '0'): 0, (1, '0'): 1, (1, 1): -1})
    assert d in ({(1, '0'): 1, (1,): -1}, {('0', 1): 1, (1,): -1})

    d = PCBO({(0, 0): 1, (0, 1): 2})
    d.update(PCBO({(1, 0): 1, (1, 1): -1}))
    d -= 1
    assert d == PCBO({(0,): 1, (0, 1): 1, (1,): -1, (): -1})

    assert d.offset == -1


def test_pcbo_num_binary_variables():

    d = PCBO({(0, 0): 1, (0, 1, 2, 3, 5): 2})
    assert d.num_binary_variables == 5
    assert d.max_index == 4


def test_num_terms():

    d = PCBO({(0,): 1, (0, 3): 2, (0, 2): -1})
    assert d.num_terms == len(d)


def test_pcbo_degree():

    d = PCBO()
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


def test_pcbo_addition():

    temp = PCBO({('0', '0'): 1, ('0', 1): 2})
    temp1 = {('0',): -1, (1, '0'): 3}
    temp2 = {(1, '0'): 5}, {('0', 1): 5}
    temp3 = {('0',): 2, (1, '0'): -1}, {('0',): 2, ('0', 1): -1}

    # constant
    d = temp.copy()
    d += 5
    assert d in ({('0',): 1, (1, '0'): 2, (): 5},
                 {('0',): 1, ('0', 1): 2, (): 5})

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
    assert g == PCBO(temp3[0])*-1


def test_pcbo_multiplication():

    temp = PCBO({('0', '0'): 1, ('0', 1): 2})
    temp1 = {('0',): 3, (1, '0'): 6}, {('0',): 3, ('0', 1): 6}
    temp2 = {('0',): .5, (1, '0'): 1}, {('0',): .5, ('0', 1): 1}

    # constant
    d = temp.copy()
    d += 3
    d *= -2
    assert d in ({('0',): -2, (1, '0'): -4, (): -6},
                 {('0',): -2, ('0', 1): -4, (): -6})

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
    assert g in ({(1, '0'): 1}, {('0', 1): 1})

    # __ifloordiv__
    d = temp.copy()
    d //= 2
    assert d in ({(1, '0'): 1}, {('0', 1): 1})

    # __mul__ but with dict
    d = temp.copy()
    d *= {(1,): 2, ('0', '0'): -1}
    assert d in ({('0',): -1, (1, '0'): 4}, {('0',): -1, ('0', 1): 4})

    # __pow__
    d = temp.copy()
    d -= 2
    d **= 2
    assert d == {('0',): -3, (): 4}

    d = temp.copy()
    assert d ** 3 == d * d * d


def test_properties():

    temp = PCBO({('0', '0'): 1, ('0', 1): 2})
    assert not temp.offset

    d = PCBO()
    d[(0,)] += 1
    d[(1,)] += 2
    assert d == d.to_qubo() == {(0,): 1, (1,): 2}
    assert d.mapping == d.reverse_mapping == {0: 0, 1: 1}

    d.set_mapping({1: 0, 0: 1})
    assert d.to_qubo() == {(1,): 1, (0,): 2}
    assert d.mapping == d.reverse_mapping == {0: 1, 1: 0}


def test_round():

    d = PCBO({(0,): 3.456, (1,): -1.53456})

    assert round(d) == {(0,): 3, (1,): -2}
    assert round(d, 1) == {(0,): 3.5, (1,): -1.5}
    assert round(d, 2) == {(0,): 3.46, (1,): -1.53}
    assert round(d, 3) == {(0,): 3.456, (1,): -1.535}


def test_normalize():

    temp = {(0,): 4, (1,): -2}
    d = PCBO(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}

    temp = {(0,): -4, (1,): 2}
    d = PCBO(temp)
    d.normalize()
    assert d == {k: v / 4 for k, v in temp.items()}


def test_symbols():

    a, b = Symbol('a'), Symbol('b')
    d = PCBO()
    d[(0,)] -= a
    d[(0, 1)] += 2
    d[(1,)] += b
    assert d == {(0,): -a, (0, 1): 2, (1,): b}
    assert d.subs(a, 2) == {(0,): -2, (0, 1): 2, (1,): b}
    assert d.subs(b, 1) == {(0,): -a, (0, 1): 2, (1,): 1}
    assert d.subs({a: -3, b: 4}) == {(0,): 3, (0, 1): 2, (1,): 4}

    d.add_constraint_eq_zero({(0,): a, (1,): -b}, bounds=(-1, 1))
    assert d == {(0,): a**2 - a, (0, 1): -2*a*b + 2, (1,): b**2 + b}
    assert d.subs(a, 0) == {(0, 1): 2, (1,): b**2 + b}
    assert d.subs({a: 0, b: 2}) == {(0, 1): 2, (1,): 6}

    d = PCBO()
    d += a
    assert d.subs(a, 0) == {}


def test_convert_solution_all_1s():

    d = PCBO({(0,): 1})
    assert d.convert_solution({0: 0}) == {0: 0}
    assert d.convert_solution({0: -1}) == {0: 1}
    assert d.convert_solution({0: 1}) == {0: 1}
    assert d.convert_solution({0: 1}, True) == {0: 0}


def test_boolean_var():

    x = [boolean_var(i) for i in range(5)]
    assert all(x[i] == {(i,): 1} for i in range(5))
    assert x[0] * x[1] * x[2] == {(0, 1, 2): 1}
    assert sum(x) == {(i,): 1 for i in range(5)}
    assert isinstance(x[0], PCBO)


def test_integer_var():

    var = integer_var('a', 4)
    assert var == {('a0',): 1, ('a1',): 2, ('a2',): 4, ('a3',): 8}

    var = integer_var('a', 4, log_trick=False)
    assert var == {('a0',): 1, ('a1',): 1, ('a2',): 1, ('a3',): 1}


""" TESTS FOR THE CONSTRAINT METHODS """


def test_pcbo_eq_constraint():

    lam = Symbol('lam')

    P = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    P.add_constraint_eq_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): -1},
        lam=lam
    )
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4

    problem = P.subs(lam, 1)
    sol = problem.solve_bruteforce()
    assert all((
        problem.is_solution_valid(sol),
        sol == solution
    ))

    e, sol = solve_pubo_bruteforce(problem.to_pubo())
    sol = problem.convert_solution(sol)
    assert all((
        not problem.is_solution_valid(sol),
        sol != solution,
        not allclose(e, obj)
    ))

    problem = P.subs(lam, 10)
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


def test_pcbo_ne_constraint_logtrick():

    for i in range(1 << 4):
        P = integer_var('a', 4) - i
        H = PCBO().add_constraint_ne_zero(P)
        for sol in H.solve_bruteforce(True):
            assert P.value(sol)


def test_pcbo_ne_constraint():

    for i in range(1 << 3):
        P = integer_var('a', 3) - i
        H = PCBO().add_constraint_ne_zero(P, log_trick=False)
        for sol in H.solve_bruteforce(True):
            assert P.value(sol)


def test_pcbo_lt_constraint_logtrick():

    lam = Symbol("lam")

    P = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    P.add_constraint_lt_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3},
        lam=lam
    )
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4

    problem = P.subs(lam, 1)
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

    problem = P.subs(lam, 10)
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


def test_pcbo_lt_constraint():

    lam = Symbol("lam")

    P = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    P.add_constraint_lt_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, (): -3},
        lam=lam, log_trick=False
    )
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4

    problem = P.subs(lam, 1)
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

    problem = P.subs(lam, 10)
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


def test_pcbo_le_constraint_logtrick():

    lam = Symbol("lam")

    P = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    P.add_constraint_le_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3},
        lam=lam
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8

    problem = P.subs(lam, .5)
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

    problem = P.subs(lam, 10)
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


def test_pcbo_le_constraint():

    lam = Symbol("lam")

    P = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    P.add_constraint_le_zero(
        {('a',): 1, ('b',): 1, ('b', 'c'): 1, ('d',): 1, (): -3},
        lam=lam, log_trick=False
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8

    problem = P.subs(lam, .5)
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

    problem = P.subs(lam, 10)
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


def test_pcbo_gt_constraint_logtrick():

    lam = Symbol("lam")

    P = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    P.add_constraint_gt_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3},
        lam=lam
    )
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4

    problem = P.subs(lam, 1)
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

    problem = P.subs(lam, 10)
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


def test_pcbo_gt_constraint():

    lam = Symbol("lam")

    P = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2
    })
    P.add_constraint_gt_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, (): 3},
        lam=lam, log_trick=False
    )
    solution = {'c': 1, 'b': 1, 'a': 0}
    obj = -4

    problem = P.subs(lam, 1)
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

    problem = P.subs(lam, 10)
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


def test_pcbo_ge_constraint_logtrick():

    lam = Symbol("lam")

    P = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    P.add_constraint_ge_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3},
        lam=lam
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8

    problem = P.subs(lam, .5)
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

    problem = P.subs(lam, 10)
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


def test_pcbo_ge_constraint():

    lam = Symbol("lam")

    P = PCBO({
        ('a',): -1, ('b',): 2, ('a', 'b'): -3, ('b', 'c'): -4, (): -2,
        ('d',): -1
    })
    P.add_constraint_ge_zero(
        {('a',): -1, ('b',): -1, ('b', 'c'): -1, ('d',): -1, (): 3},
        lam=lam, log_trick=False
    )
    solution = {'c': 1, 'b': 1, 'a': 1, 'd': 0}
    obj = -8

    problem = P.subs(lam, .5)
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

    problem = P.subs(lam, 10)
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


def test_pcbo_constraints_warnings():

    with assert_warns(QUBOVertWarning):  # qlwayss satisfied
        PCBO().add_constraint_eq_zero({(): 0})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCBO().add_constraint_eq_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCBO().add_constraint_eq_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCBO().add_constraint_lt_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCBO().add_constraint_lt_zero({(): 1, (0,): -1})

    with assert_warns(QUBOVertWarning):  # always satisfied
        PCBO().add_constraint_lt_zero({(): -1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCBO().add_constraint_le_zero({(): 1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # always satisfied
        PCBO().add_constraint_le_zero({(): -1, (0,): -.5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCBO().add_constraint_gt_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCBO().add_constraint_gt_zero({(): -1, (0,): 1})

    with assert_warns(QUBOVertWarning):  # always satisfied
        PCBO().add_constraint_gt_zero({(): 1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # not satisfiable
        PCBO().add_constraint_ge_zero({(): -1, (0,): .5})

    with assert_warns(QUBOVertWarning):  # always satisfied
        PCBO().add_constraint_ge_zero({(): 1, (0,): .5})


def test_pcbo_logic():

    H = PCBO().add_constraint_eq_NAND(
        'c', 'a', 'b').add_constraint_AND('a', 'b')
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 1, 'b': 1, 'c': 0}

    H = PCBO().add_constraint_eq_OR(
        'c', 'a', 'b').add_constraint_NOR('a', 'b')
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 0, 'b': 0, 'c': 0}

    H = PCBO().add_constraint_eq_XOR(
        'c', 'a', 'b').add_constraint_XNOR(
        'a', 'b').add_constraint_ONE('a')
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 1, 'b': 1, 'c': 0}

    H = PCBO().add_constraint_eq_NOT('a', 'b').add_constraint_ONE('a')
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 1, 'b': 0}

    H = PCBO().add_constraint_NAND('a', 'b').add_constraint_NOT(
        'a').add_constraint_OR(
        'a', 'b').add_constraint_eq_ONE('a', 'c')
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 0, 'b': 1, 'c': 0}

    H = PCBO().add_constraint_XOR('a', 'b').add_constraint_eq_NOR(
        'b', 'a', 'c').add_constraint_ONE('c').add_constraint_eq_ONE('a', 'c')
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'a': 1, 'b': 0, 'c': 1}

    H = PCBO().add_constraint_eq_AND('c', 'a', 'b').add_constraint_eq_XNOR(
        'c', 'a', 'b').add_constraint_ONE('c')
    sols = H.solve_bruteforce(True)
    assert len(sols) == 1 and sols[0] == {'c': 1, 'a': 1, 'b': 1}

    # add_constraint_eq_NOR
    H = PCBO().add_constraint_eq_NOR(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not any(sol[i] for i in range(1, 6))) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_NOR(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if (not any(sol[i] for i in range(1, 6))) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_eq_NOR(0, 1, 2)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not any(sol[i] for i in range(1, 3))) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_NOR(0, 1, 2)
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if (not any(sol[i] for i in range(1, 3))) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # add_constraint_eq_OR
    H = PCBO().add_constraint_eq_OR(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert any(sol[i] for i in range(1, 6)) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_OR(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if any(sol[i] for i in range(1, 6)) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_eq_OR(0, 1, 2)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert any(sol[i] for i in range(1, 3)) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_OR(0, 1, 2)
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if any(sol[i] for i in range(1, 3)) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # add_constraint_eq_NAND
    H = PCBO().add_constraint_eq_NAND(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not all(sol[i] for i in range(1, 6))) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_NAND(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if (not all(sol[i] for i in range(1, 6))) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_eq_NAND(0, 1, 2)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not all(sol[i] for i in range(1, 3))) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_NAND(0, 1, 2)
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if (not all(sol[i] for i in range(1, 3))) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # add_constraint_eq_AND
    H = PCBO().add_constraint_eq_AND(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert all(sol[i] for i in range(1, 6)) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_AND(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if all(sol[i] for i in range(1, 6)) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_eq_AND(0, 1, 2)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert all(sol[i] for i in range(1, 3)) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_AND(0, 1, 2)
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if all(sol[i] for i in range(1, 3)) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # add_constraint_eq_XOR
    H = PCBO().add_constraint_eq_XOR(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert sum(sol[i] for i in range(1, 6)) % 2 == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_XOR(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if sum(sol[i] for i in range(1, 6)) % 2 == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_eq_XOR(0, 1, 2)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert sum(sol[i] for i in range(1, 3)) % 2 == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_XOR(0, 1, 2)
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if sum(sol[i] for i in range(1, 3)) % 2 == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # add_constraint_eq_XNOR
    H = PCBO().add_constraint_eq_XNOR(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not sum(sol[i] for i in range(1, 6)) % 2) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_XNOR(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if (not sum(sol[i] for i in range(1, 6)) % 2) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_eq_XNOR(0, 1, 2)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert (not sum(sol[i] for i in range(1, 3)) % 2) == sol[0]
        assert not H.value(sol)
    H = PCBO().add_constraint_eq_XNOR(0, 1, 2)
    for sol in (decimal_to_boolean(i, 3) for i in range(1 << 3)):
        if (not sum(sol[i] for i in range(1, 3)) % 2) == sol[0]:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # NOR
    H = PCBO().add_constraint_NOR(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not any(sol[i] for i in range(6))
        assert not H.value(sol)
    H = PCBO().add_constraint_NOR(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if not any(sol[i] for i in range(6)):
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_NOR(0, 1)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not any(sol[i] for i in range(2))
        assert not H.value(sol)
    H = PCBO().add_constraint_NOR(0, 1)
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if not any(sol[i] for i in range(2)):
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # OR
    H = PCBO().add_constraint_OR(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert any(sol[i] for i in range(6))
        assert not H.value(sol)
    H = PCBO().add_constraint_OR(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if any(sol[i] for i in range(6)):
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_OR(0, 1)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert any(sol[i] for i in range(2))
        assert not H.value(sol)
    H = PCBO().add_constraint_OR(0, 1)
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if any(sol[i] for i in range(2)):
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # NAND
    H = PCBO().add_constraint_NAND(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not all(sol[i] for i in range(6))
        assert not H.value(sol)
    H = PCBO().add_constraint_NAND(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if not all(sol[i] for i in range(6)):
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_NAND(0, 1)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not all(sol[i] for i in range(2))
        assert not H.value(sol)
    H = PCBO().add_constraint_NAND(0, 1)
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if not all(sol[i] for i in range(2)):
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # AND
    H = PCBO().add_constraint_AND(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert all(sol[i] for i in range(6))
        assert not H.value(sol)
    H = PCBO().add_constraint_AND(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if all(sol[i] for i in range(6)):
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_AND(0, 1)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert all(sol[i] for i in range(2))
        assert not H.value(sol)
    H = PCBO().add_constraint_AND(0, 1)
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if all(sol[i] for i in range(2)):
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # XOR
    H = PCBO().add_constraint_XOR(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert sum(sol[i] for i in range(6)) % 2
        assert not H.value(sol)
    H = PCBO().add_constraint_XOR(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if sum(sol[i] for i in range(6)) % 2:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_XOR(0, 1)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert sum(sol[i] for i in range(2)) % 2
        assert not H.value(sol)
    H = PCBO().add_constraint_XOR(0, 1)
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if sum(sol[i] for i in range(2)) % 2:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    # XNOR
    H = PCBO().add_constraint_XNOR(0, 1, 2, 3, 4, 5)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not sum(sol[i] for i in range(6)) % 2
        assert not H.value(sol)
    H = PCBO().add_constraint_XNOR(0, 1, 2, 3, 4, 5)
    for sol in (decimal_to_boolean(i, 6) for i in range(1 << 6)):
        if not sum(sol[i] for i in range(6)) % 2:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)

    H = PCBO().add_constraint_XNOR(0, 1)
    sols = H.solve_bruteforce(True)
    for sol in sols:
        assert not sum(sol[i] for i in range(2)) % 2
        assert not H.value(sol)
    H = PCBO().add_constraint_XNOR(0, 1)
    for sol in (decimal_to_boolean(i, 2) for i in range(1 << 2)):
        if not sum(sol[i] for i in range(2)) % 2:
            assert H.is_solution_valid(sol)
            assert not H.value(sol)
        else:
            assert not H.is_solution_valid(sol)
            assert H.value(sol)


def test_pcbo_special_constraint_le():

    H = PCBO().add_constraint_le_zero({(0,): 1, (1,): 1, (2,): 1, (): -1})
    assert H == {(i, j): 1 for i in range(3) for j in range(i+1, 3)}

    H = PCBO().add_constraint_le_zero({(0,): 1, (1,): 1, (2, 3): 1, (): -1})
    assert H == {(0, 1): 1, (0, 2, 3): 1, (1, 2, 3): 1}

    H = PCBO().add_constraint_lt_zero({(0,): 1, (1,): 1, (2,): 1, (): -2})
    assert H == {(i, j): 1 for i in range(3) for j in range(i+1, 3)}

    H = PCBO().add_constraint_lt_zero({(0,): 1, (1,): 1, (2, 3): 1, (): -2})
    assert H == {(0, 1): 1, (0, 2, 3): 1, (1, 2, 3): 1}

    H = PCBO().add_constraint_ge_zero({(0,): -1, (1,): -1, (2,): -1, (): 1})
    assert H == {(i, j): 1 for i in range(3) for j in range(i+1, 3)}

    H = PCBO().add_constraint_ge_zero({(0,): -1, (1,): -1, (2, 3): -1, (): 1})
    assert H == {(0, 1): 1, (0, 2, 3): 1, (1, 2, 3): 1}

    H = PCBO().add_constraint_gt_zero({(0,): -1, (1,): -1, (2,): -1, (): 2})
    assert H == {(i, j): 1 for i in range(3) for j in range(i+1, 3)}

    H = PCBO().add_constraint_gt_zero({(0,): -1, (1,): -1, (2, 3): -1, (): 2})
    assert H == {(0, 1): 1, (0, 2, 3): 1, (1, 2, 3): 1}


def test_pcbo_special_constraint_eq():

    x, y, z = boolean_var('x'), boolean_var('y'), boolean_var('z')
    H1 = PCBO().add_constraint_eq_zero(z - x * y)
    H2 = PCBO().add_constraint_eq_zero(x * y - z)
    H3 = PCBO().add_constraint_eq_AND(z, x, y)
    assert H1 == H2 == H3
