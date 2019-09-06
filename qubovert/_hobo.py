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
from .utils import QUBOVertWarning
from numpy import log2, ceil


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
    x : object or list.

    Return
    ------
    res : tuple.

    """
    return tuple(x) if isinstance(x, list) else (x,)


def _pubo_value_extrema(P):
    """_pubo_value_extrema.

    Find the approximate minimum and maximum possible values that a PUBO can
    take.

    Parameters
    ----------
    P : PUBO object.

    Return
    ------
    res : tuple (min, max).

    """
    offset = P.offset
    P -= offset
    min_, max_ = offset, offset
    for v in P.values():
        if v < 0:
            min_ += v
        elif v > 0:
            max_ += v
    P += offset
    return min_, max_


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

    - ``AND``, ``NAND``, ``AND_eq``, ``NAND_eq``,
    - ``OR``, ``NOR```, ``OR_eq``, ``NOR_eq``,
    - ``XOR``, ``NXOR``, ``XOR_eq``, ``NXOR_eq``,
    - ``ONE``, ``NOT``, ``ONE_eq``, ``NOT_eq``.

    See each of their docstrings for important details on their implementation.

    Notes
    -----
    - Variables names that begin with ``"_a"`` should not be used since they
      are used internally to deal with some ancilla variables to enforce
      constraints.
    - The ``self.solve_bruteforce`` method will solve the HOBO ensuring that
      all the inputted constraints are satisfied. Whereas
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

    >>> H = HOBO().AND_eq('a', 'b', 'c')
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
        arguments : define a dictionary with ``dict(*args, **kwargs)``.
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
        # use self.__class__ here because HOIO uses this code as well.
        super(self.__class__, self).__init__(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], self.__class__):
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
        # use self.__class__ here because HOIO uses this code as well.
        super(self.__class__, self).update(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], self.__class__):
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

    @property
    def _next_ancilla(self):
        """_next_ancilla.

        Get the next available ancilla bit and increment.

        Return
        ------
        a : int.
            Ancilla bit label.

        """
        self._ancilla += 1
        return "_a%d" % (self._ancilla - 1)

    @classmethod
    def remove_ancilla_from_solution(cls, solution):
        """remove_ancilla_from_solution.

        Take a solution to the HOBO and remove all the ancilla variables, (
        represented by `_a` prefixes).

        Parameters
        ----------
        solution : dict.
            Must be the solution in terms of the original variables. Thus if
            ``solution`` is the solution to the ``self.to_pubo``,
            ``self.to_qubo``, ``self.to_hising``, or ``self.to_ising``
            formulations, then you should first call ``self.convert_solution``.
            See ``help(self.convert_solution)``.

        Return
        ------
        res : dict.
            The same as ``solution`` but with all the ancilla bits removed.

        """
        return {k: v for k, v in solution.items() if str(k)[:2] != "_a"}

    def is_solution_valid(self, solution):
        """is_solution_valid.

        Finds whether or not the given solution satisfies the constraints.

        Parameters
        ----------
        solution : dict.
            Must be the solution in terms of the original variables. Thus if
            ``solution`` is the solution to the ``self.to_pubo``,
            ``self.to_qubo``, ``self.to_hising``, or ``self.to_ising``
            formulations, then you should first call ``self.convert_solution``.
            See ``help(self.convert_solution)``.

        Return
        ------
        valid : bool.
            Whether or not the given solution satisfies the constraints.

        """
        if any(v.value(solution) != 0
               for v in self._constraints.get('eq', [])):
            return False

        if any(v.value(solution) >= 0
               for v in self._constraints.get("lt", [])):
            return False

        if any(v.value(solution) > 0
               for v in self._constraints.get("le", [])):
            return False

        if any(v.value(solution) <= 0
               for v in self._constraints.get("gt", [])):
            return False

        if any(v.value(solution) < 0
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
        # use self.__class__ here because HOIO uses this code as well.
        d = super(self.__class__, self).__round__(ndigits)
        d._constraints = self.constraints
        return d

    # override
    def subs(self, *args, **kwargs):
        """subs.

        Replace any ``sympy`` symbols that are used in the dict with values.
        Please see ``help(sympy.Symbol.subs)`` for more info.

        Parameters
        ----------
        arguments : substitutions.
            Same parameters as are inputted into ``sympy.Symbol.subs``.

        Returns
        -------
        res : HOBO object.
            Same as ``self`` but with all the symbols replaced with values.

        """
        # use self.__class__ here because HOIO uses this code as well.
        d = super(self.__class__, self).subs(*args, **kwargs)
        d._constraints = {
            k: [P.subs(*args, **kwargs) for P in v]
            for k, v in self._constraints.items()
        }
        return d

    # constraints/logic

    def add_constraint_eq_zero(self,
                               P, lam=1,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_eq_zero.

        Enforce that ``P == 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P == 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
            Please note that if ``P`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        bounds : two element tuple (optional, defaluts to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they may be
            calculated (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

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
        self._constraints.setdefault("eq", []).append(P)

        min_val, max_val = _pubo_value_extrema(P) if bounds is None else bounds

        if min_val == max_val == 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        elif min_val > 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self += lam * P
        elif max_val < 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self -= lam * P
        else:
            self += lam * P ** 2

        return self

    def add_constraint_lt_zero(self,
                               P, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_lt_zero.

        Enforce that ``P < 0`` by penalizing invalid solutions with ``lam``.
        See Notes below for more details.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P < 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
            Please note that if ``P`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        log_trick : bool (optional, defaults to True).
            Whether or not to use the log trick to enforce the inequality
            constraint. See Notes below for more details.
        bounds : two element tuple (optional, defaluts to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they will be
            calculated (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`x_0 + 2x_1 + ... < 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = HOBO()
          >>> H.add_constraint_lt_zero({(0,): 1, (1,): 2, (2,): -.5, (): .4})
          >>> test_sol = {0: 0, 1: 0, 2: 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(sol)
          0.01

          {0: 0, 1: 0, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\min_x \text{P.value(x)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\min_x \text{P.value(x)}|` ancilla
          bits will be used.

        Examples
        --------
        Enforce that :math:`-x_a x_b x_c + x_a -4x_a x_b + 3x_c < 2`.

        >>> H = HOBO().add_constraint_lt_zero(
                {('a', 'b', 'c'): -1, ('a',): 1,
                 ('a', 'b'): -4, ('c',): 3, (): -2}
            )
        >>> H
        {('b', 'c', 'a'): -19, ('b', '_a0', 'c', 'a'): -2,
         ('b', 'c', 'a', '_a1'): -4, ('a',): -3, ('b', 'a'): 24, ('c', 'a'): 6,
         ('_a0', 'a'): 2, ('a', '_a1'): 4, ('b', '_a0', 'a'): -8,
         ('b', 'a', '_a1'): -16, ('c',): -3, ('_a0', 'c'): 6, ('c', '_a1'): 12,
         (): 4, ('_a0',): -3, ('_a1',): -4, ('_a0', '_a1'): 4}
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._constraints.setdefault("lt", []).append(P)

        min_val, max_val = _pubo_value_extrema(P) if bounds is None else bounds

        if min_val >= 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self += lam * P
        elif max_val < 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        else:
            if int(min_val) == min_val:
                # copy P, don't do +=
                P = P + 1
                min_val += 1
                max_val += 1
            self += HOBO().add_constraint_le_zero(
                P, lam=lam, log_trick=log_trick,
                bounds=(min_val, max_val), suppress_warnings=True
            )

        return self

    def add_constraint_le_zero(self,
                               P, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_le_zero.

        Enforce that ``P <= 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P <= 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
            Please note that if ``P`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        log_trick : bool (optional, defaults to True).
            Whether or not to use the log trick to enforce the inequality
            constraint. See Notes below for more details.
        bounds : two element tuple (optional, defaluts to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they will be
            calculated (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`x_0 + 2x_1 + ... \leq 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = HOBO()
          >>> H.add_constraint_le_zero({(0,): 1, (1,): 2, (2,): -1.5, (): .4})
          >>> H
          {(0,): 1.7999999999999998, (0, 1): 4, (0, 2): -3.0, (0, '_a0'): 2,
           (1,): 5.6, (1, 2): -6.0, (1, '_a0'): 4, (2,): 1.0499999999999998,
           (2, '_a0'): -3.0, (): 0.16000000000000003, ('_a0',): 1.8}
          >>> test_sol = {0: 0, 1: 0, 2: 1, '_a0': 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(sol)
          0.01

          {0: 0, 1: 0, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\min_x \text{P.value(x)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\min_x \text{P.value(x)}|` ancilla
          bits will be used.

        Examples
        --------
        Enforce that :math:`-x_a x_b x_c + x_a -4x_a x_b + 3x_c \leq 2`.

        >>> H = HOBO().add_constraint_le_zero(
                {('a', 'b', 'c'): -1, ('a',): 1,
                 ('a', 'b'): -4, ('c',): 3, (): -2}
            )
        >>> H
        {('b', 'c', 'a'): -19, ('b', 'c', 'a', '_a0'): -2,
         ('_a1', 'b', 'c', 'a'): -4, ('_a2', 'b', 'c', 'a'): -8, ('a',): -3,
         ('b', 'a'): 24, ('c', 'a'): 6, ('a', '_a0'): 2, ('_a1', 'a'): 4,
         ('_a2', 'a'): 8, ('b', 'a', '_a0'): -8, ('_a1', 'b', 'a'): -16,
         ('_a2', 'b', 'a'): -32, ('c',): -3, ('c', '_a0'): 6, ('_a1', 'c'): 12,
         ('_a2', 'c'): 24, (): 4, ('_a0',): -3, ('_a1',): -4,
         ('_a1', '_a0'): 4, ('_a2', '_a0'): 8, ('_a2', '_a1'): 16}
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._constraints.setdefault("le", []).append(P)

        min_val, max_val = _pubo_value_extrema(P) if bounds is None else bounds

        if min_val > 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self += lam * P
        elif max_val <= 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        else:
            # don't mutate the P that we put in self._constraints
            P = P.copy()
            if log_trick and min_val:
                for i in range(int(ceil(log2(-min_val)))):
                    P[(self._next_ancilla,)] += pow(2, i)
                    max_val += pow(2, i)
            elif min_val:
                for _ in range(int(ceil(-min_val))):
                    P[(self._next_ancilla,)] += 1
                    max_val += 1

            # self += lam * P ** 2

            self += HOBO().add_constraint_eq_zero(
                P, lam=lam,
                bounds=(min_val, max_val), suppress_warnings=True
            )

        return self

    def add_constraint_gt_zero(self,
                               P, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_gt_zero.

        Enforce that ``P > 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P > 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
            Please note that if ``P`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        log_trick : bool (optional, defaults to True).
            Whether or not to use the log trick to enforce the inequality
            constraint. See Notes below for more details.
        bounds : two element tuple (optional, defaluts to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they will be
            calculated (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`x_0 + 2x_1 + ... > 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = HOBO()
          >>> H.add_constraint_gt_zero({(0,): -1, (1,): -2, (2,): .5, (): -.4})
          >>> test_sol = {0: 0, 1: 0, 2: 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(sol)
          0.01

          {0: 0, 1: 0, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\min_x \text{P.value(x)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\min_x \text{P.value(x)}|` ancilla
          bits will be used.

        Examples
        --------
        Enforce that :math:`x_a x_b x_c - x_a + 4x_a x_b - 3x_c > -2`.

        >>> H = HOBO().add_constraint_gt_zero(
                {('a', 'b', 'c'): 1, ('a',): -1,
                 ('a', 'b'): 4, ('c',): -3, (): 2}
            )
        >>> H
        {('b', 'c', 'a'): -19, ('b', '_a0', 'c', 'a'): -2,
         ('b', 'c', 'a', '_a1'): -4, ('a',): -3, ('b', 'a'): 24, ('c', 'a'): 6,
         ('_a0', 'a'): 2, ('a', '_a1'): 4, ('b', '_a0', 'a'): -8,
         ('b', 'a', '_a1'): -16, ('c',): -3, ('_a0', 'c'): 6, ('c', '_a1'): 12,
         (): 4, ('_a0',): -3, ('_a1',): -4, ('_a0', '_a1'): 4}
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._constraints.setdefault("gt", []).append(P)
        if bounds is not None:
            bounds = -bounds[0], -bounds[1]
        self += HOBO().add_constraint_lt_zero(
            -P, lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        return self

    def add_constraint_ge_zero(self,
                               P, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_ge_zero.

        Enforce that ``P >= 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P >= 0. Note that ``P`` will be
            converted to a ``qubovert.PUBO`` object if it is not already, thus
            it must follow the conventions, see ``help(qubovert.PUBO)``.
            Please note that if ``P`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        log_trick : bool (optional, defaults to True).
            Whether or not to use the log trick to enforce the inequality
            constraint. See Notes below for more details.
        bounds : two element tuple (optional, defaluts to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they will be
            calculated (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`x_0 + 2x_1 + ... \geq 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = HOBO()
          >>> H.add_constraint_ge_zero({(0,): -1, (1,): -2, (2,):1.5, (): -.4})
          >>> H
          {(0,): 1.7999999999999998, (0, 1): 4, (0, 2): -3.0, (0, '_a0'): 2,
           (1,): 5.6, (1, 2): -6.0, (1, '_a0'): 4, (2,): 1.0499999999999998,
           (2, '_a0'): -3.0, (): 0.16000000000000003, ('_a0',): 1.8}
          >>> test_sol = {0: 0, 1: 0, 2: 1, '_a0': 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(sol)
          0.01

          {0: 0, 1: 0, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\min_x \text{P.value(x)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\min_x \text{P.value(x)}|` ancilla
          bits will be used.

        Examples
        --------
        Enforce that :math:`x_a x_b x_c - x_a + 4x_a x_b - 3x_c \geq -2`.

        >>> H = HOBO().add_constraint_ge_zero(
                {('a', 'b', 'c'): 1, ('a',): -1,
                 ('a', 'b'): 4, ('c',): -3, (): 2}
            )
        >>> H
        {('b', 'c', 'a'): -19, ('b', 'c', 'a', '_a0'): -2,
         ('_a1', 'b', 'c', 'a'): -4, ('_a2', 'b', 'c', 'a'): -8, ('a',): -3,
         ('b', 'a'): 24, ('c', 'a'): 6, ('a', '_a0'): 2, ('_a1', 'a'): 4,
         ('_a2', 'a'): 8, ('b', 'a', '_a0'): -8, ('_a1', 'b', 'a'): -16,
         ('_a2', 'b', 'a'): -32, ('c',): -3, ('c', '_a0'): 6, ('_a1', 'c'): 12,
         ('_a2', 'c'): 24, (): 4, ('_a0',): -3, ('_a1',): -4,
         ('_a1', '_a0'): 4, ('_a2', '_a0'): 8, ('_a2', '_a1'): 16}
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._constraints.setdefault("ge", []).append(P)
        if bounds is not None:
            bounds = -bounds[0], -bounds[1]
        self += HOBO().add_constraint_le_zero(
            -P, lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        return self

    def AND(self, *variables, lam=1, constraint=False):
        r"""AND.

        Add a penalty to the HOBO that is only zero when
        :math:`a \land b \land c \land ...` is True, with a penalty factor
        ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a list of
            hashable objects. They are the label of the binary variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

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

        >>> H = HOBO()
        >>> H.AND('a', 'b', 'c', 'd')  # enforce a AND b AND c AND d

        """
        t = ()
        for x in variables:
            t += _create_tuple(x)
        return self.ONE(list(t), lam=lam, constraint=constraint)

    def OR(self, *variables, lam=1, constraint=False):
        r"""OR.

        Add a penalty to the HOBO that is only nonzero when
        :math:`a \lor b \lor c \lor d \lor ...` is True, with a penalty factor
        ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a list of
            hashable objects. They are the label of the binary variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

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

        >>> H = HOBO()
        >>> H.OR('a', 'b', 'c', 'd')  # enforce a OR b OR c OR d

        >>> H = HOBO()
        >>> # enforce (a AND b) OR (c AND d) OR (e AND f AND g)
        >>> H.OR(['a', 'b'], ['c', 'd'], ['e', 'f', 'g'])

        """
        def or_(vars_):
            if len(vars_) == 1:
                return vars_[0]
            x = or_(vars_[:-1])
            return x + vars_[-1] - x * vars_[-1]

        variables = tuple(PUBO({_create_tuple(x): 1}) for x in variables)
        P = 1 - or_(variables)

        if constraint:
            self._constraints.setdefault("eq", []).append(P)
        self += lam * P
        return self

    def XOR(self, a, b, lam=1, constraint=False):
        r"""XOR.

        Add a penalty to the HOBO that is only nonzero when :math:`a \oplus b`
        is True, with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a list of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or a list of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

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
        P = PUBO({a: -1, b: -1, a+b: 2, (): 1})
        if constraint:
            self._constraints.setdefault("eq", []).append(P)
        self += lam * P
        return self

    def ONE(self, a, lam=1, constraint=False):
        r"""ONE.

        Add a penalty to the HOBO that is only nonzero when :math:`a == 1` is
        True, with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a list of hashable objects.
            The label for binary variables ``a``.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

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
        P = PUBO({(): 1, _create_tuple(a): -1})
        if constraint:
            self._constraints.setdefault("eq", []).append(P)
        self += lam * P
        return self

    def NAND(self, *variables, lam=1, constraint=True):
        r"""NAND.

        Add a penalty to the HOBO that is only zero when
        :math:`\lnot (a \land b \land c \land ...)` is True, with a penalty
        factor ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a list of
            hashable objects. They are the label of the binary variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

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
        >>> H.NAND(['a', 'b'], ['c', 'd'])  # enforce (a AND b) NAND (c AND d)

        >>> H = HOBO()
        >>> H.NAND('a', 'b', 'c', 'd')  # enforce a NAND b NAND c NAND d

        >>> H = HOBO()
        >>> # enforce (a AND b) NAND (c AND d AND e) NAND f
        >>> H.NAND(['a', 'b'], ['c', 'd', 'e'], 'f')

        """
        t = ()
        for x in variables:
            t += _create_tuple(x)
        return self.NOT(list(t), lam=lam, constraint=constraint)

    def NOR(self, *variables, lam=1, constraint=False):
        r"""NOR.

        Add a penalty to the HOBO that is only nonzero when
        :math:`\lnot(a \lor b \lor c \lor d \lor ...)` is True, with a penalty
        factor ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a list of
            hashable objects. They are the label of the binary variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.NOR('a', 'b')  # enforce a NOR b

        >>> H = HOBO()
        >>> H.NOR(['a', 'b'], ['c', 'd'])  # enforce (a AND b) NOR (c AND d)

        >>> H = HOBO()
        >>> H.NOR('a', 'b', 'c', 'd')  # enforce a NOR b NOR c NOR d

        >>> H = HOBO()
        >>> # enforce (a AND b) NOR (c AND d NAD e) NOR f
        >>> H.NOR(['a', 'b'], ['c', 'd', 'e'], 'f')

        """
        P = 1 - HOBO().OR(*variables)

        if constraint:
            self._constraints.setdefault("eq", []).append(P)
        self += lam * P
        return self

    def NXOR(self, a, b, lam=1, constraint=False):
        r"""NXOR.

        Add a penalty to the HOBO that is only nonzero when
        :math:`\lnot(a \oplus b)` is True, with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a list of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or a list of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

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
        P = PUBO({a: 1, b: 1, a+b: -2})
        if constraint:
            self._constraints.setdefault("eq", []).append(P)
        self += lam * P
        return self

    def NOT(self, a, lam=1, constraint=False):
        r"""NOT.

        Add a penalty to the HOBO that is only nonzero when
        :math:`\lnot a` is True, with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a list of hashable objects.
            The label for binary variables ``a``.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

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
        a = _create_tuple(a)
        self[a] += lam
        if constraint:
            self._constraints.setdefault("eq", []).append(PUBO({a: 1}))
        return self

    def AND_eq(self, *variables, lam=1, constraint=False):
        r"""AND_eq.

        Add a penalty to the HOBO that enforces that
        :math:`v_0 \land v_1 \land v_2 \land ... == v_n`,
        with a penalty factor
        ``lam``, where ``v_1 = variables[0]``, ``v_2 = variables[1]``, ...,
        ``v_n = variables[-1]``.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a list of
            hashable objects. They are the label of the binary variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.AND_eq('a', 'b', 'c')  # enforce a AND b == c

        >>> H = HOBO()
        >>> # enforce (a AND b AND c AND d) == 'e'
        >>> H.AND_eq(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b AND c AND d) == 'e'
        >>> H.AND_eq('a', 'b', 'c', 'd', 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b AND c AND d) == 'e' AND 'f'
        >>> H.AND_eq(['a', 'b'], ['c', 'd'], ['e', 'f'])

        References
        ----------
        https://arxiv.org/pdf/1307.8041.pdf equation 6.

        """
        n = len(variables)
        if n < 3:
            raise ValueError("Must supply at least three variables. "
                             "See ``ONE_eq`` for less.")
        c = _create_tuple(variables[-1])
        a, b = (), ()
        for v in variables[:n // 2]:
            a += _create_tuple(v)
        for v in variables[n // 2:-1]:
            b += _create_tuple(v)

        P = PUBO({c: 3, a+b: 1, a+c: -2, b+c: -2})
        self += lam * P
        if constraint:
            self._constraints.setdefault("eq", []).append(P)
        return self

    def OR_eq(self, *variables, lam=1, constraint=False):
        r"""OR_eq.

        Add a penalty to the HOBO that enforces that
        :math:`v_0 \lor v_1 \lor v_2 \lor ... == v_n`,
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ..., ``v_n = variables[-1]``.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a list of
            hashable objects. They are the label of the binary variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.OR_eq('a', 'b', 'c')  # enforce a OR b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) OR (c AND d) == 'e'
        >>> H.OR_eq(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) OR (c AND d) OR e == f
        >>> H.OR_eq(['a', 'b'], ['c', 'd'], 'e', 'f')

        >>> H = HOBO()
        >>> # enforce (a AND b) OR (c AND d) == ('e' AND 'f')
        >>> H.OR_eq(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        n = len(variables)
        if n < 3:
            raise ValueError("Must supply at least three variables.")

        if n == 3:
            a, b, c = tuple(_create_tuple(x) for x in variables)
            P = PUBO({a: 1, b: 1, c: 1, a+b: 1, a+c: -2, b+c: -2})
            self += lam * P
            if constraint:
                self._constraints.setdefault("eq", []).append(P)
        else:
            P = HOBO().NOR(*variables[:-1]) - {_create_tuple(variables[-1]): 1}
            self += lam * P**2
            if constraint:
                self._constraints.setdefault("eq", []).append(P)

        return self

    def XOR_eq(self, a, b, c, lam=1, constraint=False):
        r"""XOR_eq.

        Add a penalty to the HOBO that enforces that :math:`a \oplus b == c`
        with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a list of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or a list of hashable objects.
            The label for binary variables ``b``.
        c : any hashable object or a list of hashable objects.
            The label for binary variables ``c``.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.XOR_eq('a', 'b', 'c')  # enforce a XOR b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) XOR (c AND d) == 'e'
        >>> H.XOR_eq(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) XOR (c AND d) == ('e' AND 'f')
        >>> H.XOR_eq(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        P = HOBO().NXOR(a, b) - {_create_tuple(c): 1}
        if constraint:
            return self.add_constraint_eq_zero(P, lam=lam)
        self += lam * P**2
        return self

    def ONE_eq(self, a, b, lam=1, constraint=False):
        r"""ONE_eq.

        Add a penalty to the HOBO that enforces that :math:`a == b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a list of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or a list of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.ONE_eq('a', 'b')  # enforce a == b

        >>> H = HOBO()
        >>> H.ONE_eq(['a', 'b'], 'c')  # enforce (a AND b) == c

        >>> H = HOBO()
        >>> # enforce (a AND b) == (c AND d)
        >>> H.ONE_eq(['a', 'b'], ['c', 'd'])

        """
        P = {_create_tuple(a): 1, _create_tuple(b): -1}
        if constraint:
            return self.add_constraint_eq_zero(P, lam=lam)
        self += lam * P**2
        return self

    def NAND_eq(self, *variables, lam=1, constraint=False):
        r"""NAND_eq.

        Add a penalty to the HOBO that enforces that
        :math:`\lnot (v_0 \land v_1 \land v_2 \land ...) == v_n`, with a
        penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ..., ``v_n = variables[-1]``.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a list of
            hashable objects. They are the label of the binary variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.NAND_eq('a', 'b', 'c')  # enforce a NAND b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) NAND (c AND d) == 'e'
        >>> H.NAND_eq(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) NAND (c AND d) NAND e == f
        >>> H.NAND_eq(['a', 'b'], ['c', 'd'], 'e', 'f')

        >>> H = HOBO()
        >>> # enforce (a AND b) NAND (c AND d) == 'e' AND 'f'
        >>> H.NAND_eq(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        n = len(variables)
        if n < 3:
            raise ValueError("Must supply at least three variables. "
                             "See ``NOT_eq`` for less.")
        c = _create_tuple(variables[-1])
        a, b = (), ()
        for v in variables[:n // 2]:
            a += _create_tuple(v)
        for v in variables[n // 2:-1]:
            b += _create_tuple(v)

        P = PUBO({(): 3, a: -2, b: -2, c: -3, a+b: 1, a+c: 2, b+c: 2})
        self += lam * P
        if constraint:
            self._constraints.setdefault("eq", []).append(P)
        return self

    def NOR_eq(self, *variables, lam=1, constraint=False):
        r"""NOR_eq.

        Add a penalty to the HOBO that enforces that
        :math:`\lnot(v_0 \lor v_1 \lor v_2 \lor ... == v_n)`,
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ..., ``v_n = variables[-1]``.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a list of
            hashable objects. They are the label of the binary variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.NOR_eq('a', 'b', 'c')  # enforce a NOR b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) NOR (c AND d) == 'e'
        >>> H.NOR_eq(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) NOR (c AND d) NOR e == f
        >>> H.NOR_eq(['a', 'b'], ['c', 'd'], 'e', 'f')

        >>> H = HOBO()
        >>> # enforce (a AND b) NOR (c AND d) == ('e' AND 'f')
        >>> H.NOR_eq(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        n = len(variables)
        if n < 3:
            raise ValueError("Must supply at least three variables.")

        if n == 3:
            a, b, c = tuple(_create_tuple(x) for x in variables)
            P = PUBO({a: -1, b: -1, c: -1, (): 1, a+b: 1, a+c: 2, b+c: 2})
            self += lam * P
            if constraint:
                self._constraints.setdefault("eq", []).append(P)
        else:
            P = HOBO().OR(*variables[:-1]) - {_create_tuple(variables[-1]): 1}
            self += lam * P**2
            if constraint:
                self._constraints.setdefault("eq", []).append(P)

        return self

    def NXOR_eq(self, a, b, c, lam=1, constraint=False):
        r"""NXOR_eq.

        Add a penalty to the HOBO that enforces that
        :math:`\lnot(a \oplus b) == c` with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a list of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or a list of hashable objects.
            The label for binary variables ``b``.
        c : any hashable object or a list of hashable objects.
            The label for binary variables ``c``.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.NXOR_eq('a', 'b', 'c')  # enforce a NXOR b == c

        >>> H = HOBO()
        >>> # enforce (a AND b) NXOR (c AND d) == 'e'
        >>> H.NXOR_eq(['a', 'b'], ['c', 'd'], 'e')

        >>> H = HOBO()
        >>> # enforce (a AND b) NXOR (c AND d) == ('e' AND 'f')
        >>> H.NXOR_eq(['a', 'b'], ['c', 'd'], ['e', 'f'])

        """
        P = HOBO().XOR(a, b) - {_create_tuple(c): 1}
        if constraint:
            return self.add_constraint_eq_zero(P, lam=lam)
        self += lam * P**2
        return self

    def NOT_eq(self, a, b, lam=1, constraint=False):
        r"""NOT.

        Add a penalty to the HOBO that enforces that :math:`\lnot a == b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a list of hashable objects.
            The label for binary variables ``a``.
        b : any hashable object or a list of hashable objects.
            The label for binary variables ``b``.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.
        constraint: bool (optional, defaults to False).
            Whether or not this should expression should be added to the set
            of constraints (see ``self.constraints``). This mostly only effects
            the method ``self.solve_bruteforce``. If ``constraint`` is True,
            then the ``solve_bruteforce`` method will guarentee that this
            AND clause holds, otherwise it will just minimize the penalty
            coming from the AND addition to the HOBO.

        Return
        ------
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = HOBO()
        >>> H.NOT_eq('a', 'b')  # enforce NOT(a) == b

        >>> H = HOBO()
        >>> H.NOT_eq(['a', 'b'], 'c')  # enforce NOT(a AND b) == c
        >>> H = HOBO()
        >>> # enforce NOT(a AND b) == (c AND d)
        >>> H.NOT_eq(['a', 'b'], ['c', 'd'])

        """
        P = HOBO().ONE(a) - {_create_tuple(b): 1}
        if constraint:
            return self.add_constraint_eq_zero(P, lam=lam)
        self += lam * P**2
        return self
