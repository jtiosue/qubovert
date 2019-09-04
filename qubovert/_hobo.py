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

"""_hobo.py.

Contains the HOBO class. See ``help(qubovert.HOBO)``.

"""

from . import PUBO
from .utils import pubo_value


__all__ = 'HOBO',


# TODO: add better constraints for constraints that match known forms.
# including
# For the log trick he mentions, we usually need a constraint like
# ``sum(x) >= 1``. In order to enforce this constraint, we add a penalty to the
# QUBO of the form ``1 - sum(x) + sum(x[i] x[j] for i in range(len(x)) for j in
# range(i+1, len(x)))`` (the idea comes from arXiv:1811.11538v5).


def _create_tuple(x):
    """_create_tuple.

    Create a tuple from ``x``.

    Parameters
    ----------
    x : object.

    Return
    ------
    res : tuple.

    """
    try:
        return tuple(x)
    except TypeError:
        return x,


class HOBO(PUBO):
    """HOBO.

    This class deals with Higher Order Binary Optimization problems. HOBO
    inherits some methods and attributes from the ``PUBO`` class. See
    ``help(qubovert.PUBO)``.

    ``HOBO`` has all the same methods as ``PUBO``, but adds some constraint
    methods; namely

    - ``add_constraint_eq_zero(P, lam=1)`` enforces that ``P == 0`` by
      penalizing with ``lam``,
    - ``add_constraint_lt_zero(P, lam=1)`` enforces that ``P < 0`` by
      penalizing with ``lam``,
    - ``add_constraint_le_zero(P, lam=1)`` enforces that ``P <= 0`` by
      penalizing with ``lam``,
    - ``add_constraint_gt_zero(P, lam=1)`` enforces that ``P > 0`` by
      penalizing with ``lam``, and
    - ``add_constraint_ge_zero(P, lam=1)`` enforces that ``P >= 0`` by
      penalizing with ``lam``.

    Each of these takes in a PUBO ``P`` and a lagrange multiplier ``lam`` that
    defaults to 1. See each of their docstrings for important details on their
    implementation.

    We then implement logical operations:
    AND, OR, XOR, ONE, NOT, NAND, NOR, NXOR, add_constraint_AND,
    add_constraint_OR, add_constraint_XOR, add_constraint_ONE,
    add_constraint_NOT, add_constraint_NAND, add_constraint_NOR,
    add_constraint_NXOR.
    See each of their docstrings for important details on their implementation.

    Notes
    -----
    Variables names that begin with ``"_a"`` should not be used since they are
    used internally to deal with some ancilla variables to enforce constraints.

    The ``self.solve_bruteforce`` method will solve the HOBO ensuring that all
    the inputted constraints are satisfied. Whereas
    ``qubovert.utils.solve_pubo_bruteforce(self)`` or
    ``qubovert.utils.solve_pubo_bruteforce(self.to_pubo())`` will solve the
    PUBO created from the HOBO. If the inputted constraints are not enforced
    strong enough (ie too small lagrange multipliers) then these may not give
    the correct result, whereas ``self.solve_bruteforce()`` will always give
    the correct result (ie one that satisfies all the constraints).

    Examples
    --------
    See ``qubovert.PUBO`` for more examples of using HOBO without constraints.

    >>> H = HOBO()
    >>> H.add_constraint_eq_zero({('a', 1): 2, (1, 2): -1, (): -1})
    >>> H
    {('a', 1, 2): -4, (1, 2): 3, (): 1}
    >>> H -= 1
    >>> H
    {('a', 1, 2): -4, (1, 2): 3}

    >>> H = HOBO()
    >>> H.add_constraint_eq_zero(
            {(0, 1): 1}
        ).add_constraint_eq_zero(
            {(1, 2): 1, (): -1}
        )
    >>> H
    {(0, 1): 1, (1, 2): -1, (): 1}

    >>> H = HOBO().add_constraint_AND('a', 'b', 'c')
    >>> H
    {('c',): 3, ('b', 'a'): 1, ('c', 'a'): -2, ('c', 'b'): -2}

    >>> from any_module import qubo_solver
    >>> # or from qubovert.utils import solve_qubo_bruteforce as qubo_solver
    >>> H = HOBO()
    >>>
    >>> # AND variables a and b, and variables b and c
    >>> H.AND('a', 'b').AND('b', 'c')
    >>>
    >>> # OR variables b and c
    >>> H.OR('b', 'c')
    >>>
    >>> # (a AND b) OR (c AND d)
    >>> H.OR(['a', 'b'], ['c', 'd'])
    >>>
    >>> H
    {('b', 'a'): -2, (): 4, ('b',): -1, ('c',): -1,
     ('c', 'd'): -1, ('c', 'd', 'b', 'a'): 1}
    >>> Q = H.to_qubo()
    >>> Q
    {(): 4, (0,): -1, (2,): -1, (2, 3): 1, (4,): 6, (0, 4): -4,
     (1, 4): -4, (5,): 6, (2, 5): -4, (3, 5): -4, (4, 5): 1}
    >>> obj_value, sol = qubo_solver(Q)
    >>> sol
    {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0}
    >>> solution = H.convert_solution(sol)
    >>> solution
    {'b': 1, 'a': 1, 'c': 1, 'd': 0}

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        This class deals with higher order binary optimization problems.
        Note that it is generally more efficient to initialize an empty HOBO
        object and then build the HOBO, rather than initialize a HOBO object
        with an already built dict.

        Parameters
        ----------
        args and kwargs : define a dictionary with ``dict(*args, **kwargs)``.
            The dictionary will be initialized to follow all the convensions of
            the class. Alternatively, ``args[0]`` can be a HOBO object.

        Examples
        -------
        >>> hobo = HOBO()
        >>> hobo[('a',)] += 5
        >>> hobo[(0, 'a')] -= 2
        >>> hobo -= 1.5
        >>> hobo
        {('a',): 5, ('a', 0): -2, (): -1.5}
        >>> hobo.add_constraint_eq_zero({('a',): 1}, lam=5)
        >>> hobo
        {('a',): 10, ('a', 0): -2, (): -1.5}

        >>> hobo = HOBO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> hobo
        {('a',): 5, ('a', 0, 1): -2, (): -1.5}

        """
        super().__init__(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], HOBO):
            self._constraints = args[0].constraints
            self._ancilla = args[0].num_ancillas
        else:
            self._ancilla, self._constraints = 0, {}

    def update(self, *args, **kwargs):
        """update.

        Update the HOBO but following all the conventions of this class.

        Parameters
        ----------
        *args and **kwargs : defines a dictionary or HOBO.
            Ie ``d = dict(*args, **kwargs)``.
            Each element in d will be added in place to this instance following
            all the required convensions.

        """
        super().update(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], HOBO):
            for k, v in args[0]._constraints:
                self._constraints.setdefault(k, []).extend(v)

    @property
    def constraints(self):
        """constraints.

        Return the constraints of the HOBO.

        Return
        ------
        res : dict.
            The keys of ``res`` are some or all of
            ``'eq'``, ``'lt'``, ``'le'``, ``'gt'``, and ``'ge'``.
            The values are lists of ``qubovert.PUBO`` objects. For a
            given key, value pair ``k, v``, the ``v[i]`` element represents
            the PUBO ``v[i]`` being == 0 if ``k == 'eq'``,
            < 0 if ``k == 'lt'``, <= 0 if ``k == 'le'``,
            > 0 if ``k == 'gt'``, >= 0 if ``k == 'ge'``.

        """
        return {k: [x.copy() for x in v] for k, v in self._constraints.items()}

    @property
    def num_ancillas(self):
        """num_ancillas.

        Return the number of ancilla variables introduced to the HOBO in
        order to enforce the inputted constraints.

        Returns
        -------
        num : int.
            Number of ancillas in the HOBO.

        """
        return self._ancilla

    def convert_solution(self, solution):
        """convert_solution.

        Convert the solution to the integer labeled HOBO to the solution to
        the originally labeled HOBO.

        Parameters
        ----------
        solution : iterable or dict.
            The PUBO, HIsing, QUBO, or Ising solution output. The PUBO solution
            output is either a list or tuple where indices specify the label of
            the variable and the element specifies whether it's 0 or 1 for PUBO
            (or -1 or 1 for Ising), or it can be a dictionary that maps the
            label of the variable to is value. The QUBO/Ising solution output
            includes the assignment for the ancilla variables used to reduce
            the degree of the PUBO.

        Return
        -------
        res : dict.
            Maps binary variable labels to their HOBO solutions values {0, 1}.

        """
        sol = super().convert_solution(solution)
        for a in range(self._ancilla):
            sol.pop("_a%d" % a, 0)
        return sol

    def is_solution_valid(self, solution):
        """is_solution_valid.

        Finds whether or not the given solution satisfies the constraints.

        Parameters
        ----------
        solution : dict or iterable.
            Either the output of ``self.convert_solution`` or the input to
            ``self.convert_solution`` (see ``help(self.convert_solution)``).

        Return
        ------
        valid : bool.
            Whether or not the given solution satisfies the constraints.

        """
        if not isinstance(solution, dict) or solution.keys() != self._vars:
            solution = self.convert_solution(solution)

        if any(pubo_value(solution, v) != 0
               for v in self._constraints.get('eq', [])):
            return False

        if any(pubo_value(solution, v) >= 0
               for v in self._constraints.get("lt", [])):
            return False

        if any(pubo_value(solution, v) > 0
               for v in self._constraints.get("le", [])):
            return False

        if any(pubo_value(solution, v) <= 0
               for v in self._constraints.get("gt", [])):
            return False

        if any(pubo_value(solution, v) < 0
               for v in self._constraints.get("ge", [])):
            return False

        return True

    # override
    def __round__(self, ndigits=None):
        """round.

        Round values of the HOBO object.

        Parameters
        ----------
        ndigits : int.
            Number of decimal digits to round to.

        Returns
        -------
        res : HOBO object.
            Copy of self but with each value rounded to ``ndigits`` decimal
            digits. Each value has a type according to the docstring
            specifications of ``round``, see ``help(round)``.

        """
        d = super.__round__(ndigits)
        d._constraints = self.constraints
        return d

    # constraints/logic

    def add_constraint_eq_zero(self, P, lam=1):
        r"""add_constraint_eq_zero.

        Enforce that ``P == 0`` by adding ``lam * P**2`` to the HOBO.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P == 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        The following enforces that :math:`\prod_{i=0}^{3} x_i == 0`.

        >>> H = HOBO()
        >>> H.add_constraint_eq_zero({(0, 1, 2, 3): 1})
        >>> H
        {(0, 1, 2, 3): 1}

        The following enforces that :math:`\sum_{i=1}^{3} i x_i x_{i+1} == 0`.

        >>> H = HOBO()
        >>> H.add_constraint_eq_zero({(1, 2): 1, (2, 3): 2, (3, 4): 3})
        >>> H
        {(1, 2): 1, (1, 2, 3): 4, (1, 2, 3, 4): 6,
         (2, 3): 4, (2, 3, 4): 12, (3, 4): 9}

        Here we show how operations can be strung together.

        >>> H = HOBO()
        >>> H.add_constraint_eq_zero(
                {(0, 1): 1}
            ).add_constraint_eq_zero(
                {(1, 2): 1, (): -1}
            )
        >>> H
        {(0, 1): 1, (1, 2): -1, (): 1}

        """
        P = PUBO(P)
        self += lam * P ** 2
        self._constraints.setdefault("eq", []).append(P)
        return self

    def add_constraint_lt_zero(self, P, lam=1):
        r"""add_constraint_lt_zero.

        Enforce that ``P < 0`` by penalizing invalid solution with ``lam``.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P < 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        FINISH

        Notes
        -----
        FINISH

        """
        P = PUBO(P)
        raise NotImplementedError("Coming soon!")
        self._constraints.setdefault("lt", []).append(P)
        return self

    def add_constraint_le_zero(self, P, lam=1):
        r"""add_constraint_le_zero.

        Enforce that ``P <= 0`` by penalizing invalid solution with ``lam``.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P <= 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        FINISH

        Notes
        -----
        FINISH

        """
        P = PUBO(P)
        raise NotImplementedError("Coming soon!")
        self._constraints.setdefault("le", []).append(P)
        return self

    def add_constraint_gt_zero(self, P, lam=1):
        r"""add_constraint_gt_zero.

        Enforce that ``P > 0`` by penalizing invalid solution with ``lam``.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P > 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        FINISH

        Notes
        -----
        FINISH

        """
        P = PUBO(P)
        raise NotImplementedError("Coming soon!")
        self._constraints.setdefault("gt", []).append(P)
        return self

    def add_constraint_ge_zero(self, P, lam=1):
        r"""add_constraint_ge_zero.

        Enforce that ``P >= 0`` by penalizing invalid solution with ``lam``.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P >= 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        FINISH

        Notes
        -----
        FINISH

        """
        P = PUBO(P)
        raise NotImplementedError("Coming soon!")
        self._constraints.setdefault("ge", []).append(P)
        return self

    def AND(self, a, b, lam=1):
        r"""AND.

        Add a penalty to the HOBO that enforces that :math:`a \land b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.AND('a', 'b')  # enforce a AND b
        >>> H
        {('b', 'a'): -1, (): 1}

        >>> H = HOBO()
        >>> H.AND(['a', 'b'], ['c', 'd'])  # enforce a AND b AND c AND d
        >>> H
        {('c', 'd', 'b', 'a'): -1, (): 1}

        """
        return self.ONE(_create_tuple(a) + _create_tuple(b), lam)

    def OR(self, a, b, lam=1):
        r"""OR.

        Add a penalty to the HOBO that enforces that :math:`a \lor b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.OR('a', 'b')  # enforce a OR b
        >>> H
        {('a',): -1, ('b',): -1, ('b', 'a'): 1, (): 1}

        >>> H = HOBO()
        >>> H.OR(['a', 'b'], ['c', 'd'])  # enforce (a AND b) OR (c AND d)
        >>> H
        {('b', 'a'): -1, ('c', 'd'): -1, ('c', 'd', 'b', 'a'): 1, (): 1}

        """
        a, b = _create_tuple(a), _create_tuple(b)
        self += {a: -lam, b: -lam, a+b: lam, (): lam}
        return self

    def XOR(self, a, b, lam=1):
        r"""XOR.

        Add a penalty to the HOBO that enforces that :math:`a \oplus b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.XOR('a', 'b')  # enforce a XOR b

        >>> H = HOBO()
        >>> H.XOR(['a', 'b'], ['c', 'd'])  # enforce (a AND b) XOR (c AND d)

        """
        a, b = _create_tuple(a), _create_tuple(b)
        self += {a: -lam, b: -lam, a+b: 2*lam, (): lam}
        return self

    def ONE(self, a, lam=1):
        r"""ONE.

        Add a penalty to the HOBO that enforces that :math:`a == 1` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.ONE('a')  # enforce a

        >>> H = HOBO()
        >>> H.ONE(['a', 'b'])  # enforce (a AND b)

        """
        a = _create_tuple(a)
        self[a] -= lam
        self += lam
        return self

    def NAND(self, a, b, lam=1):
        r"""NAND.

        Add a penalty to the HOBO that enforces that :math:`\lnot(a \land b)`
        with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.NAND('a', 'b')  # enforce a NAND b

        >>> H = HOBO()
        >>> H.AND(['a', 'b'], ['c', 'd'])  # enforce (a AND b) NAND (c AND d)

        """
        return self.NOT(_create_tuple(a) + _create_tuple(b), lam)

    def NOR(self, a, b, lam=1):
        r"""NOR.

        Add a penalty to the HOBO that enforces that :math:`\lnot(a \lor b)`
        with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.NOR('a', 'b')  # enforce a OR b

        >>> H = HOBO()
        >>> H.NOR(['a', 'b'], ['c', 'd'])  # enforce (a AND b) OR (c AND d)

        """
        a, b = _create_tuple(a), _create_tuple(b)
        self += {a: lam, b: lam, a+b: -lam}
        return self

    def NXOR(self, a, b, lam=1):
        r"""NXOR.

        Add a penalty to the HOBO that enforces that :math:`\lnot(a \oplus b)`
        with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.NXOR('a', 'b')  # enforce a NXOR b

        >>> H = HOBO()
        >>> H.NXOR(['a', 'b'], ['c', 'd'])  # enforce (a AND b) NXOR (c AND d)

        """
        a, b = _create_tuple(a), _create_tuple(b)
        self += {a: lam, b: lam, a+b: -2*lam}
        return self

    def NOT(self, a, lam=1):
        r"""NOT.

        Add a penalty to the HOBO that enforces that :math:`\lnot a` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.NOT('a')  # enforce not a
        >>> H
        {('a',): 1}

        >>> H = HOBO()
        >>> H.NOT(['a', 'b'])  # enforce NOT (a AND b)
        >>> H
        {(a, b): 1}

        """
        self[_create_tuple(a)] += lam
        return self

    def add_constraint_AND(self, a, b, c, lam=1):
        r"""add_constraint_AND.

        Add a penalty to the HOBO that enforces that :math:`a \land b == c`
        with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        c : any hashable object or an iterable of hashable objects.
            The label for binary variables ``c``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.add_constraint_AND('a', 'b', 'c')  # enforce a AND b == c

        >>> H = HOBO()
        >>> # enforce (a AND b AND c AND d) == 'e'
        >>> H.add_constraint_AND(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b AND c AND d) == 'e' AND 'f'
        >>> H.add_constraint_AND(['a', 'b'], ['c', 'd'], ['e', 'f'])

        References
        ----------
        https://arxiv.org/pdf/1307.8041.pdf equation 6.

        """
        a, b, c = _create_tuple(a), _create_tuple(b), _create_tuple(c)
        P = PUBO({c: 3, a+b: 1, a+c: -2, b+c: -2})
        self += lam * P
        self._constraints.setdefault("eq", []).append(P)
        return self

    def add_constraint_OR(self, a, b, c, lam=1):
        r"""add_constraint_OR.

        Add a penalty to the HOBO that enforces that :math:`a \lor b == c` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        c : any hashable object or an iterable of hashable objects.
            The label for binary variables ``c``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.add_constraint_OR('a', 'b', 'c')  # enforce a OR b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) OR (c AND d) == 'e'
        >>> H.add_constraint_OR(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) OR (c AND d) == ('e' AND 'f')
        >>> H.add_constraint_OR(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        a, b, c = _create_tuple(a), _create_tuple(b), _create_tuple(c)
        P = PUBO({a+b: 1, a+c: -2, b+c: -2, a: 1, b: 1, c: 1})
        self += lam * P
        self._constraints.setdefault("eq", []).append(P)
        return self

    def add_constraint_XOR(self, a, b, c, lam=1):
        r"""add_constraint_XOR.

        Add a penalty to the HOBO that enforces that :math:`a \oplus b == c`
        with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        c : any hashable object or an iterable of hashable objects.
            The label for binary variables ``c``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.add_constraint_XOR('a', 'b', 'c')  # enforce a XOR b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) XOR (c AND d) == 'e'
        >>> H.add_constraint_XOR(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) XOR (c AND d) == ('e' AND 'f')
        >>> H.add_constraint_XOR(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        c = {_create_tuple(c): 1}
        return self.add_constraint_eq_zero(HOBO().NXOR(a, b) - c, lam)

    def add_constraint_ONE(self, a, b, lam=1):
        r"""add_constraint_ONE.

        Add a penalty to the HOBO that enforces that :math:`a == b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.add_constraint_ONE('a', 'b')  # enforce a == b

        >>> H = HOBO()
        >>> H.add_constraint_ONE(['a', 'b'], 'c')  # enforce (a AND b) == c

        >>> H = HOBO()
        >>> # enforce (a AND b) == (c AND d)
        >>> H.add_constraint_ONE(['a', 'b'], ['c', 'd'])

        """
        a, b = PUBO({_create_tuple(a): 1}), PUBO({_create_tuple(b): 1})
        return self.add_constraint_eq_zero(a - b, lam)

    def add_constraint_NAND(self, a, b, c, lam=1):
        r"""add_constraint_NAND.

        Add a penalty to the HOBO that enforces that
        :math:`\lnot (a \land b) == c` with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        c : any hashable object or an iterable of hashable objects.
            The label for binary variables ``c``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.add_constraint_NAND('a', 'b', 'c')  # enforce a NAND b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) NAND (c AND d) == 'e'
        >>> H.add_constraint_NAND(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) NAND (c AND d) == 'e' AND 'f'
        >>> H.add_constraint_NAND(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        # TODO: figure out if I can do this without third order connection
        # like with HOBO.add_constraint_AND.
        c = {_create_tuple(c): 1}
        return self.add_constraint_eq_zero(HOBO().AND(a, b) - c, lam)

    def add_constraint_NOR(self, a, b, c, lam=1):
        r"""add_constraint_NOR.

        Add a penalty to the HOBO that enforces that
        :math:`\lnot (a \lor b) == c` with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        c : any hashable object or an iterable of hashable objects.
            The label for binary variables ``c``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.add_constraint_NOR('a', 'b', 'c')  # enforce a NOR b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) NOR (c AND d) == 'e'
        >>> H.add_constraint_NOR(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) NOR (c AND d) == ('e' AND 'f')
        >>> H.add_constraint_NOR(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        # TODO: figure out if I can do this without third order connection
        # like with HOBO.add_constraint_OR.
        c = {_create_tuple(c): 1}
        return self.add_constraint_eq_zero(HOBO().OR(a, b) - c, lam)

    def add_constraint_NXOR(self, a, b, c, lam=1):
        r"""add_constraint_NXOR.

        Add a penalty to the HOBO that enforces that
        :math:`\lnot(a \oplus b) == c` with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        c : any hashable object or an iterable of hashable objects.
            The label for binary variables ``c``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.add_constraint_NXOR('a', 'b', 'c')  # enforce a NXOR b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) NXOR (c AND d) == 'e'
        >>> H.add_constraint_NXOR(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) NXOR (c AND d) == ('e' AND 'f')
        >>> H.add_constraint_NXOR(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        c = {_create_tuple(c): 1}
        return self.add_constraint_eq_zero(HOBO().XOR(a, b) - c, lam)

    def add_constraint_NOT(self, a, b, lam=1):
        r"""NOT.

        Add a penalty to the HOBO that enforces that :math:`\lnot a == b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or an iterable of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or an iterable of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.add_constraint_NOT('a', 'b')  # enforce NOT(a) == b

        >>> H = HOBO()
        >>> H.add_constraint_NOT(['a', 'b'], 'c')  # enforce NOT(a AND b) == c
        >>> H = HOBO()
        >>> # enforce NOT(a AND b) == (c AND d)
        >>> H.add_constraint_NOT(['a', 'b'], ['c', 'd'])

        """
        b = {_create_tuple(b): 1}
        return self.add_constraint_eq_zero(HOBO().ONE(a) - b, lam)
