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

"""_booleanconstraints.py.

Contains the BooleanConstraints class. See
``help(qubovert.BooleanConstraints)``.

"""

from . import PUBO
from .utils import QUBOVertWarning, num_bits, approximate_pubo_extrema
from .sat import OR, XOR, BUFFER, NOT, AND


__all__ = 'BooleanConstraints',


# special constraint forms

def _special_constraints_eq_zero(boolean_constraints, P, lam):
    """_special_constraints_eq_zero.

    See if the constraint that ``P == 0`` matches any special forms.

    Parameters
    ----------
    boolean_constraints : BooleanConstraints object.
        The BooleanConstraints to add the constraint to.
    P : PUBO object.
        The PUBO constraint such that ``P == 0``.
        are present.
    lam : float > 0 or sympy.Symbol.
        Langrange multiplier to penalize violations of the constraint.

    Return
    ------
    success : bool.
        True if a special constraint was found and added to ``pcbo``, else
        False.

    """
    keys, values = tuple(P.keys()), tuple(P.values())

    # if P is of the form z == x * y. ie z == AND(x, y)
    if (
            not P.offset and P.num_binary_variables == 3 and
            P.num_terms == 2 and values[0] == -values[1] and
            {len(keys[0]), len(keys[1])} == {1, 2}
    ):
        a, = keys[0] if len(keys[0]) == 1 else keys[1]
        b, c = keys[0] if len(keys[0]) == 2 else keys[1]
        boolean_constraints._penalty += BooleanConstraints(
            ).add_constraint_eq_AND(a, b, c, lam=lam).to_penalty()
        return True

    # if P is of the form z == 1 - x * y. ie z == NAND(x, y)

    # if P is of the form z == x * y. ie z == OR(x, y)

    # if P is of the form z == x * y. ie z == NOR(x, y)

    return False


def _special_constraints_le_zero(boolean_constraints, P,
                                 lam, log_trick, bounds):
    """_special_constraints_le_zero.

    See if the constraint that ``P <= 0`` matches any special forms.

    Parameters
    ----------
    boolean_constraints : BooleanConstraints object.
        The BooleanConstraints to add the constraint to.
    P : PUBO object.
        The PUBO constraint such that ``P <= 0``.
        are present.
    lam : float > 0 or sympy.Symbol.
        Langrange multiplier to penalize violations of the constraint.
    log_trick : bool.
        Whether or not to use the log trick to enforce the inequality
        constraint.
    bounds : two element tuple.
        A tuple ``(min, max)``, the minimum and maximum values that the
        PUBO ``P`` can take.

    Return
    ------
    success : bool.
        True if a special constraint was found and added to ``pcbo``, else
        False.

    """
    min_val, max_val = bounds

    # P without offset defined for convenience
    P_wo_offset = P - P.offset

    # if P is of the form sum(x_i) <= 1.
    if P.offset == -1 and all(x == 1 for x in P_wo_offset.values()):
        boolean_constraints._penalty += lam * P * P_wo_offset / 2
        return True

    # if P is of the form P_wo_offset <= -P.offset and P_wo_offset is always
    # >= 0, and (important!) log_trick is False!!
    elif (
            not log_trick and not min_val - P.offset and
            P.offset <= 0 and min_val
    ):
        # We have that P_wo_offset <= P.offset and min(P_wo_offset) = 0. So we
        # can do a penalty lam(P_wo_offset - sum(ancillas))**2

        # create ancillas
        ancillas = PUBO()
        for i in range(num_bits(-P.offset, log_trick)):
            ancillas[(boolean_constraints._next_ancilla,)] += 1

        diff = P_wo_offset - ancillas
        boolean_constraints._penalty += lam * diff * diff

        return True

    # if P is of the form 1 <= x + y, then it is the same as OR(x, y)
    elif (P.offset == 1 and len(P_wo_offset) == 2 and
          set(P_wo_offset.values()) == {-1}):
        variables = tuple(P_wo_offset.keys())
        x, y = AND(*variables[0]), AND(*variables[1])
        boolean_constraints._penalty += BooleanConstraints(
            ).add_constraint_OR(x, y, lam=lam).to_penalty()
        return True

    # if P is of the form x <= y
    elif not P.offset and len(P) == 2 and set(P.values()) == {1, -1}:
        coef = {v: k for k, v in P.items()}
        x, y = AND(*coef[1]), AND(*coef[-1])
        boolean_constraints._penalty += lam * x * (1 - y)
        return True

    return False


# helpers

def _get_bounds(P, bounds):
    """_get_bounds.

    Compute the missing bounds of ``P``.

    Parameters
    ----------
    P : PUBO object.
    bounds : two element tuple or None.
        A tuple ``(min, max)``, the minimum and maximum values that the
        PUBO ``P`` can take. If ``bounds`` is None, then they will be
        calculated (approximately), or if either of the elements of
        ``bounds`` is None, then that element will be calculated
        (approximately).

    Return
    ------
    res : two element tuple.
        The bounds of ``P``.

    """
    if bounds is None or bounds == (None, None):
        bounds = approximate_pubo_extrema(P)
    elif bounds[0] is None:
        bounds = approximate_pubo_extrema(P)[0], bounds[1]
    elif bounds[1] is None:
        bounds = bounds[0], approximate_pubo_extrema(P)[1]

    return bounds


# main class

class BooleanConstraints:
    """BooleanConstraints.

    Keep track of Boolean constraints.

    - ``add_constraint_eq_zero(P, lam=1, ...)`` enforces that ``P == 0`` by
      penalizing with ``lam``,
    - ``add_constraint_ne_zero(P, lam=1, ...)`` enforces that ``P != 0`` by
      penalizing with ``lam``,
    - ``add_constraint_lt_zero(P, lam=1, ...)`` enforces that ``P < 0`` by
      penalizing with ``lam``,
    - ``add_constraint_le_zero(P, lam=1, ...)`` enforces that ``P <= 0`` by
      penalizing with ``lam``,
    - ``add_constraint_gt_zero(P, lam=1, ...)`` enforces that ``P > 0`` by
      penalizing with ``lam``, and
    - ``add_constraint_ge_zero(P, lam=1, ...)`` enforces that ``P >= 0`` by
      penalizing with ``lam``.

    Each of these takes in a PUBO ``P`` and a lagrange multiplier ``lam`` that
    defaults to 1. See each of their docstrings for important details on their
    implementation.

    We then implement logical operations:

    - ``add_constraint_AND``, ``add_constraint_NAND``,
    - ``add_constraint_eq_AND``, ``add_constraint_eq_NAND``,
    - ``add_constraint_OR``, ``add_constraint_NOR```,
    - ``add_constraint_eq_OR``, ``add_constraint_eq_NOR``,
    - ``add_constraint_XOR``, ``add_constraint_XNOR``,
    - ``add_constraint_eq_XOR``, ``add_constraint_eq_XNOR``,
    - ``add_constraint_BUFFER``, ``add_constraint_NOT``,
    - ``add_constraint_eq_BUFFER``, ``add_constraint_eq_NOT``.

    See each of their docstrings for important details on their implementation.

    Notes
    -----
    - Variables names that begin with ``"__a"`` should not be used since they
      are used internally to deal with some ancilla variables to enforce
      constraints.

    """

    def __init__(self):
        """__init__.

        Initialize a BooleanConstraints object.

        """
        self._constraints, self._penalty, self._ancilla = {}, PUBO(), 0

    def __repr__(self):
        """__repr__.

        Return a representation of ``self``.

        Returns
        -------
        res : str.

        """
        return str(self._constraints)

    def __str__(self):
        """__str__.

        Return a string showing the object.

        Returns
        -------
        s : str.

        """
        s = "CONSTRAINTS\n"
        for con, d in self:
            s += con + " :  " + str(d) + "\n"
        return s[:-1]

    def __iter__(self):
        """__iter__.

        Iterate through ``self``.

        Yields
        ------
        constraints : tuple (str, dict).

        Examples
        --------
        >>> b = BooleanConstraints()
        >>> b.add_constraint_eq_zero({(0,): 1, (1,): 1})
        >>> b.add_constraint_le_zero({(0,): 1, (1,): -1})
        >>> for con, pubo in b:
        >>>     print(con, pubo)
        "eq" {(0,): 1, (1,): 1}
        "le" {(0,): 1, (1,): -1}

        >>> s = SpinConstraints()
        >>> s.add_constraint_eq_zero({(0,): 1, (1,): 1})
        >>> s.add_constraint_le_zero({(0,): 1, (1,): -1})
        >>> for con, puso in b:
        >>>     print(con, puso)
        "eq" {(0,): 1, (1,): 1}
        "le" {(0,): 1, (1,): -1}

        """
        for k, vals in self._constraints.items():
            for v in vals:
                yield k, v

    @property
    def num_ancillas(self):
        """num_ancillas.

        Return the number of ancilla variables introduced in
        order to enforce the inputted constraints.

        Returns
        -------
        num : int.
            Number of ancillas.

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
        return "__a%d" % (self._ancilla - 1)

    @classmethod
    def remove_ancilla_from_solution(cls, solution):
        """remove_ancilla_from_solution.

        Take a solution and remove all the ancilla variables, (
        represented by `_a` prefixes).

        Parameters
        ----------
        solution : dict.
            Must be the solution in terms of the original variables. Thus if
            ``solution`` is the solution to the ``self.to_pubo``,
            ``self.to_qubo``, ``self.to_puso``, or ``self.to_quso``
            formulations, then you should first call ``self.convert_solution``.
            See ``help(self.convert_solution)``.

        Return
        ------
        res : dict.
            The same as ``solution`` but with all the ancilla bits removed.

        """
        return {k: v for k, v in solution.items() if str(k)[:3] != "__a"}

    @property
    def constraints(self):
        """constraints.

        Return a copy of the constraints.

        Return
        ------
        res : dict.
            The keys of ``res`` are some or all of
            ``'eq'``, ``'ne'``, ``'lt'``, ``'le'``, ``'gt'``, and ``'ge'``.
            The values are lists of ``qubovert.PUBO`` objects. For a
            given key, value pair ``k, v``, the ``v[i]`` element represents
            the PUBO ``v[i]`` being == 0 if ``k == 'eq'``,
            != 0 if ``k == 'ne'``,
            < 0 if ``k == 'lt'``, <= 0 if ``k == 'le'``,
            > 0 if ``k == 'gt'``, >= 0 if ``k == 'ge'``.

        """
        return {k: [x.copy() for x in v] for k, v in self._constraints.items()}

    def _append_constraint(self, key, constraint):
        """_append_constraint.

        Internal method to add a constraint to the constraints dictionary.

        Parameters
        ----------
        key : str.
            One of ``'eq'``, ``'ne'``, ``'lt'``, ``'le'``, ``'gt'``, or
            ``'ge'``.
        constraint : qubovert.PUBO object.

        """
        self._constraints.setdefault(key, []).append(constraint)

    def _pop_constraint(self, key):
        """_pop_constraint.

        Internal method to remove the most recently added constraint.

        Parameters
        ----------
        key : str.
            One of ``'eq'``, ``'ne'``, ``'lt'``, ``'le'``, ``'gt'``, or
            ``'ge'``.

        """
        if self._constraints.get(key, []):
            self._constraints[key].pop()
            if not self._constraints[key]:
                self._constraints.pop(key)

    def is_solution_valid(self, solution):
        """is_solution_valid.

        Finds whether or not the given solution satisfies the constraints.

        Parameters
        ----------
        solution : dict.
            Must be the solution in terms of the original variables. Thus if
            ``solution`` is the solution to the ``self.to_pubo``,
            ``self.to_qubo``, ``self.to_puso``, or ``self.to_quso``
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

        if any(v.value(solution) == 0
               for v in self._constraints.get('ne', [])):
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

    def copy(self):
        """copy.

        Return a copy.

        Returns
        -------
        d : same type as self.

        """
        d = self.__class__()
        d._constraints = self.constraints  # copies
        d._penalty = self.to_penalty()  # copies
        d._ancilla = self._ancilla
        return d

    def __round__(self, ndigits=None):
        """round.

        Round all constraints. See ``help(round)``.

        Parameters
        ----------
        ndigits : int (optional, defaults to None).
            Number of digits to round to.

        Returns
        -------
        res : same type as self.

        """
        d = self.__class__()
        d._ancilla = self._ancilla
        d._penalty = round(self._penalty, ndigits)
        for k, v in self._constraints.items():
            d._constraints[k] = [round(x, ndigits) for x in v]
        return d

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
        res : same type as self.
            Same as ``self`` but with all the symbols replaced with values.

        """
        d = self.__class__()
        d._ancilla = self._ancilla
        d._penalty = self._penalty.subs(*args, **kwargs)
        for k, v in self._constraints.items():
            d._constraints[k] = [x.subs(*args, **kwargs) for x in v]
        return d

    def to_penalty(self):
        """to_penalty.

        Return the penalty function that enforces all of the constraints.

        Returns
        -------
        P : qv.PUBO or qv.PUSO object.

        """
        return self._penalty.copy()

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
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they may be
            calculated (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        The following enforces that :math:`\prod_{i=0}^{3} x_i == 0`.

        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_zero({(0, 1, 2, 3): 1})

        The following enforces that :math:`\sum_{i=1}^{3} i x_i x_{i+1} == 0`.

        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_zero({(1, 2): 1, (2, 3): 2, (3, 4): 3})

        Here we show how operations can be strung together.

        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_zero(
                {(0, 1): 1}
            ).add_constraint_eq_zero(
                {(1, 2): 1, (): -1}
            )

        """
        P = PUBO(P)
        self._append_constraint("eq", P)
        if not lam:
            return self

        if _special_constraints_eq_zero(self, P, lam):
            return self

        min_val, max_val = _get_bounds(P, bounds)

        if min_val == max_val == 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        elif min_val > 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self._penalty += lam * P
        elif max_val < 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self._penalty -= lam * P
        elif min_val == 0:
            self._penalty += lam * P
        elif max_val == 0:
            self._penalty -= lam * P
        else:
            self._penalty += lam * P * P

        return self

    def add_constraint_ne_zero(self,
                               P, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_ne_zero.

        Enforce that ``P != 0`` by penalizing invalid solutions with ``lam``.
        See Notes below for more details.

        Parameters
        ----------
        P : dict representing a PUBO.
            The PUBO constraint such that P != 0. Note that ``P`` will be
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
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`x_0 + 2x_1 + ... != 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function.
        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\min_x \text{P.value(x)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\min_x \text{P.value(x)}|` ancilla
          bits will be used.

        Examples
        --------
        Enforce that :math:`-x_a x_b x_c + x_a -4x_a x_b + 3x_c != 2`.

        >>> H = BooleanConstraints().add_constraint_ne_zero(
                {('a', 'b', 'c'): -1, ('a',): 2,
                ('a', 'b'): -4, ('c',): 3, (): -2}
            )

        """
        P = PUBO(P)
        self._append_constraint("ne", P)
        if not lam:
            return self

        min_val, max_val = _get_bounds(P, bounds)

        if min_val == max_val == 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self._penalty += lam
        elif min_val > 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        elif max_val < 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        elif min_val == 0:
            self.add_constraint_gt_zero(
                P, lam=lam,
                bounds=(min_val, max_val), suppress_warnings=suppress_warnings
            )
            self._pop_constraint('gt')
        elif max_val == 0:
            self.add_constraint_lt_zero(
                P, lam=lam,
                bounds=(min_val, max_val), suppress_warnings=suppress_warnings
            )
            self._pop_constraint('lt')

        else:
            # don't mutate the P that we put in self._constraints
            P = P.copy()
            sign = 2 * PUBO.create_var(self._next_ancilla) - 1
            P += sign
            max_val += 1
            min_val -= 1
            for i in range(num_bits(max_val - min_val - 1, log_trick)):
                v = pow(2, i) if log_trick else 1
                P += sign * v * PUBO.create_var(self._next_ancilla)
                max_val += v
                min_val -= v

            self.add_constraint_eq_zero(
                P, lam=lam,
                bounds=(min_val, max_val), suppress_warnings=True
            )
            self._pop_constraint("eq")

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
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`x_0 + 2x_1 + ... < 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = BooleanConstraints()
          >>> H.add_constraint_lt_zero({(0,): 1, (1,): 2, (2,): -.5, (): .4})
          >>> P = H.to_penalty()
          >>> test_sol = {0: 0, 1: 0, 2: 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> P.value(test_sol)
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

        >>> H = BooleanConstraints().add_constraint_lt_zero(
                {('a', 'b', 'c'): -1, ('a',): 1,
                 ('a', 'b'): -4, ('c',): 3, (): -2}
            )
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._append_constraint("lt", P)
        if not lam:
            return self

        min_val, max_val = _get_bounds(P, bounds)

        if min_val >= 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self._penalty += lam * P
        elif max_val < 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        else:
            # copy P, don't do +=
            P = P + 1
            min_val += 1
            max_val += 1
            self.add_constraint_le_zero(
                P, lam=lam, log_trick=log_trick,
                bounds=(min_val, max_val), suppress_warnings=True
            )
            self._pop_constraint("le")

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
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`x_0 + 2x_1 + ... \leq 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = BooleanConstraints()
          >>> H.add_constraint_le_zero({(0,): 1, (1,): 2, (2,): -1.5, (): .4})
          >>> P = H.to_penalty()
          >>> P
          {(0,): 1.7999999999999998, (0, 1): 4, (0, 2): -3.0, (0, '__a0'): 2,
           (1,): 5.6, (1, 2): -6.0, (1, '__a0'): 4, (2,): 1.0499999999999998,
           (2, '__a0'): -3.0, (): 0.16000000000000003, ('__a0',): 1.8}
          >>> test_sol = {0: 0, 1: 0, 2: 1, '__a0': 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> P.value(test_sol)
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

        >>> H = BooleanConstraints().add_constraint_le_zero(
                {('a', 'b', 'c'): -1, ('a',): 1,
                 ('a', 'b'): -4, ('c',): 3, (): -2}
            )
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._append_constraint("le", P)
        if not lam:
            return self

        bounds = min_val, max_val = _get_bounds(P, bounds)
        if _special_constraints_le_zero(self, P, lam, log_trick, bounds):
            return self

        if min_val > 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self._penalty += lam * P
        elif max_val <= 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        else:
            # don't mutate the P that we put in self._constraints
            P = P.copy()
            if min_val:
                for i in range(num_bits(-min_val, log_trick)):
                    v = pow(2, i) if log_trick else 1
                    P[(self._next_ancilla,)] += v
                    max_val += v

            self.add_constraint_eq_zero(
                P, lam=lam,
                bounds=(min_val, max_val), suppress_warnings=True
            )
            self._pop_constraint("eq")

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
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`x_0 + 2x_1 + ... > 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = BooleanConstraints()
          >>> H.add_constraint_gt_zero({(0,): -1, (1,): -2, (2,): .5, (): -.4})
          >>> P = H.to_penalty()
          >>> test_sol = {0: 0, 1: 0, 2: 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> P.value(test_sol)
          0.01

          {0: 0, 1: 0, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\max_x \text{P.value(x)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\max_x \text{P.value(x)}|` ancilla
          bits will be used.

        Examples
        --------
        Enforce that :math:`x_a x_b x_c - x_a + 4x_a x_b - 3x_c > -2`.

        >>> H = BooleanConstraints().add_constraint_gt_zero(
                {('a', 'b', 'c'): 1, ('a',): -1,
                 ('a', 'b'): 4, ('c',): -3, (): 2}
            )
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._append_constraint("gt", P)
        if not lam:
            return self
        min_val, max_val = _get_bounds(P, bounds)
        bounds = -max_val, -min_val
        self.add_constraint_lt_zero(
            -P, lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._pop_constraint("lt")
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
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUBO ``P`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`x_0 + 2x_1 + ... \geq 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = BooleanConstraints()
          >>> H.add_constraint_ge_zero({(0,): -1, (1,): -2, (2,):1.5, (): -.4})
          >>> P = H.to_penalty()
          >>> P
          {(0,): 1.7999999999999998, (0, 1): 4, (0, 2): -3.0, (0, '__a0'): 2,
           (1,): 5.6, (1, 2): -6.0, (1, '__a0'): 4, (2,): 1.0499999999999998,
           (2, '__a0'): -3.0, (): 0.16000000000000003, ('__a0',): 1.8}
          >>> test_sol = {0: 0, 1: 0, 2: 1, '__a0': 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> P.value(test_sol)
          0.01

          {0: 0, 1: 0, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\max_x \text{P.value(x)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\max_x \text{P.value(x)}|` ancilla
          bits will be used.

        Examples
        --------
        Enforce that :math:`x_a x_b x_c - x_a + 4x_a x_b - 3x_c \geq -2`.

        >>> H = BooleanConstraints().add_constraint_ge_zero(
                {('a', 'b', 'c'): 1, ('a',): -1,
                 ('a', 'b'): 4, ('c',): -3, (): 2}
            )
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._append_constraint("ge", P)
        if not lam:
            return self
        min_val, max_val = _get_bounds(P, bounds)
        bounds = -max_val, -min_val
        self.add_constraint_le_zero(
            -P, lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._pop_constraint("le")
        return self

    def add_constraint_eq_AND(self, a, *variables, lam=1):
        r"""add_constraint_eq_AND.

        Enforces that
        :math:`a = v_0 \land v_1 \land v_2 \land ...`,
        with a penalty factor
        ``lam``, where ``v_1 = variables[0]``, ``v_2 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_AND('a', 'b', 'c')  # enforce a == b AND c

        >>> H = BooleanConstraints()
        >>> # enforce (a AND b AND c AND d) == 'e'
        >>> H.add_constraint_eq_AND('e', 'a', 'b', 'c', 'd')

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b, c = boolean_var('a'), boolean_var('b'), boolean_var('c')
        >>> H = BooleanConstraints()
        >>> # enforce that a == b AND c
        >>> H.add_constraint_eq_AND(a, b, c)

        References
        ----------
        https://arxiv.org/pdf/1307.8041.pdf equation 6.

        """
        n = len(variables)
        if n < 2:
            raise ValueError("Must supply at least two variables to AND. "
                             "See ``add_constraint_eq_BUFFER`` for less.")

        a = BUFFER(a)
        b, c = 1, 1
        for v in variables[:n // 2]:
            b *= BUFFER(v)
        for v in variables[n // 2:]:
            c *= BUFFER(v)

        P = 3 * a + b * c - 2 * a * (b + c)

        return self.add_constraint_eq_zero(P, lam, bounds=(0, 3))

    def add_constraint_eq_OR(self, a, *variables, lam=1):
        r"""add_constraint_eq_OR.

        Enforce that
        :math:`v_0 \lor v_1 \lor v_2 \lor ... == a`,
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_OR('a', 'b', 'c')  # enforce a == b OR c

        >>> H = BooleanConstraints()
        >>> # enforce a == b OR c OR d
        >>> H.add_constraint_eq_OR('a', 'b', 'c', 'd')

        """
        n = len(variables)
        if n < 2:
            raise ValueError("Must supply at least two variables to OR.")

        a = BUFFER(a)
        if n == 2:
            b, c = BUFFER(variables[0]), BUFFER(variables[1])
            P = a + b + c + b * c - 2 * a * (b + c)
            bounds = 0, 3
        else:
            P = BooleanConstraints(
                ).add_constraint_NOR(*variables).to_penalty() - a
            bounds = -1, 1

        return self.add_constraint_eq_zero(P, lam=lam, bounds=bounds)

    def add_constraint_eq_XOR(self, a, *variables, lam=1):
        r"""add_constraint_eq_XOR.

        Enforce that
        :math:`v_0 \oplus v_1 \oplus ... == a`
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_XOR('a', 'b', 'c')  # enforce a == b XOR c

        >>> H = BooleanConstraints()
        >>> # enforce a == b XOR c XOR d
        >>> H.add_constraint_eq_XOR('a', 'b', 'c', 'd')

        """
        P = BooleanConstraints(
            ).add_constraint_XNOR(*variables).to_penalty() - BUFFER(a)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(-1, 1))

    def add_constraint_eq_BUFFER(self, a, b, lam=1):
        r"""add_constraint_eq_BUFFER.

        Enforce that :math:`a == b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variables ``a``, or its PUBO representation.
        b : any hashable object or a dict.
            The label for boolean variables ``b``, or its PUBO representation.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_BUFFER('a', 'b')  # enforce a == b

        >>> from qubovert import BooleanConstraints, boolean_var
        >>> a, b = boolean_var('a'), boolean_var('b')
        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_BUFFER(a, b)  # enforce a == b

        """
        P = BUFFER(a) - BUFFER(b)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(-1, 1))

    def add_constraint_eq_NAND(self, a, *variables, lam=1):
        r"""add_constraint_eq_NAND.

        Enforce that
        :math:`\lnot (v_0 \land v_1 \land v_2 \land ...) == a`, with a
        penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variables ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_NAND('a', 'b', 'c')  # enforce a == b NAND c

        >>> H = BooleanConstraints()
        >>> # enforce a == b NAND c NAND d
        >>> H.add_constraint_eq_NAND('a', 'b', 'c', 'd')

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b, c = boolean_var('a'), boolean_var('b'), boolean_var('c')
        >>> # enforce a == b NAND c
        >>> H = BooleanConstraints().add_constraint_eq_NAND(a, b, c)

        """
        n = len(variables)
        if n < 2:
            raise ValueError("Must supply at least two variables to NAND. "
                             "See ``add_constraint_eq_NOT`` for less.")
        b, c = 1, 1
        for v in variables[:n // 2]:
            b *= BUFFER(v)
        for v in variables[n // 2:]:
            c *= BUFFER(v)

        P = NOT(a) * (3 - 2 * (b + c)) + b * c
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 3))

    def add_constraint_eq_NOR(self, a, *variables, lam=1):
        r"""add_constraint_eq_NOR.

        Enforce that
        :math:`\lnot(v_0 \lor v_1 \lor v_2 \lor ...) == a`,
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_NOR('a', 'b', 'c')  # enforce a == b NOR c

        >>> H = BooleanConstraints()
        >>> # enforce a == b NOR c NOR d
        >>> H.add_constraint_eq_NOR('a', 'b', 'c', 'd')

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> # enforce a == b NOR c
        >>> a, b, c = boolean_var('a'), boolean_var('b'), boolean_var('c')
        >>> H.add_constraint_eq_NOR(a, b, c)

        """
        n = len(variables)
        if n < 2:
            raise ValueError("Must supply at least two variables to NOR.")

        a = BUFFER(a)
        if n == 2:
            b, c = BUFFER(variables[0]), BUFFER(variables[1])
            P = 1 - a - b - c + b * c + 2 * a * (b + c)
            bounds = 0, 3
        else:
            P = BooleanConstraints(
                ).add_constraint_OR(*variables).to_penalty() - a
            bounds = -1, 1

        return self.add_constraint_eq_zero(P, lam=lam, bounds=bounds)

    def add_constraint_eq_XNOR(self, a, *variables, lam=1):
        r"""add_constraint_eq_XNOR.

        Enforce that
        :math:\lnot(`v_0 \oplus v_1 \oplus ...) == a`
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns
            ``self`` so that operations
            can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_XNOR('a', 'b', 'c')  # enforce a == b XNOR c

        >>> H = BooleanConstraints()
        >>> # enforce a == b XNOR c XNOR d
        >>> H.add_constraint_eq_XNOR('a', 'b', 'c', 'd')

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b, c = boolean_var('a'), boolean_var('b'), boolean_var('c')
        >>> # enforce a == b XNOR c
        >>> H = BooleanConstraints().add_constraint_eq_XNOR(a, b, c)

        """
        P = BooleanConstraints(
            ).add_constraint_XOR(*variables).to_penalty() - BUFFER(a)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(-1, 1))

    def add_constraint_eq_NOT(self, a, b, lam=1):
        r"""NOT.

        Enforce that :math:`\lnot a == b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variable ``a``, or its PUBO representation.
        b : any hashable object or a dict.
            The label for boolean variable ``b``, or its PUBO representation.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_eq_NOT('a', 'b')  # enforce NOT(a) == b

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b = boolean_var('a'), boolean_var('b')
        >>> H.add_constraint_eq_NOT(a, b)  # enforce NOT(a) == b

        """
        P = BooleanConstraints(
            ).add_constraint_BUFFER(a).to_penalty() - BUFFER(b)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(-1, 1))

    def add_constraint_AND(self, *variables, lam=1):
        r"""add_constraint_AND.

        Enforce that
        :math:`a \land b \land c \land ...` is True, with a penalty factor
        ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_AND('a', 'b')  # enforce a AND b
        >>> H
        {('b', 'a'): -1, (): 1}

        >>> H = BooleanConstraints()
        >>> H.add_constraint_AND('a', 'b', 'c', 'd')
        >>> # enforce a AND b AND c AND d

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b = boolean_var('a'), boolean_var('b')
        >>> # enforce a AND b
        >>> H = BooleanConstraints().add_constraint_AND(a, b)

        """
        return self.add_constraint_BUFFER(AND(*variables), lam=lam)

    def add_constraint_OR(self, *variables, lam=1):
        r"""add_constraint_OR.

        Enforce that
        :math:`a \lor b \lor c \lor d \lor ...` is True, with a penalty factor
        ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_OR('a', 'b')  # enforce a OR b
        >>> H
        {('a',): -1, ('b',): -1, ('b', 'a'): 1, (): 1}

        >>> H = BooleanConstraints()
        >>> H.add_constraint_OR('a', 'b', 'c', 'd')  # enforce a OR b OR c OR d

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b = boolean_var('a'), boolean_var('b')
        >>> # enforce a OR b
        >>> H = BooleanConstraints().add_constraint_OR(a, b)

        """
        P = 1 - OR(*variables)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 1))

    def add_constraint_XOR(self, *variables, lam=1):
        r"""add_constraint_XOR.

        Enforce that
        :math:`v_0 \oplus v_1 \oplus ... \oplus v_n` is True, with a penalty
        factor ``lam``, where ``v_0 = variables[0]``, ``v_1 = variables[1]``,
        ..., ``v_n = variables[-1]``. See ``qubovert.sat.XOR`` for the
        XOR convention that qubovert uses for more than two inputs.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_XOR('a', 'b')  # enforce a XOR b

        >>> H = BooleanConstraints()
        >>> H.add_constraint_XOR('a', 'b', 'c')  # enforce a XOR b XOR c

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b = boolean_var('a'), boolean_var('b')
        >>> # enforce a XOR b
        >>> H = BooleanConstraints().add_constraint_XOR(a, b)

        """
        P = 1 - XOR(*variables)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 1))

    def add_constraint_BUFFER(self, a, lam=1):
        r"""add_constraint_BUFFER.

        Enforce that :math:`a == 1` is
        True, with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variable ``a``, or its PUBO representation.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns
            ``self`` so that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_BUFFER('a')  # enforce a

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a = boolean_var('a')
        >>> # enforce a
        >>> H = BooleanConstraints().add_constraint_BUFFER(a)

        """
        return self.add_constraint_eq_zero(NOT(a), lam=lam, bounds=(0, 1))

    def add_constraint_NAND(self, *variables, lam=1):
        r"""add_constraint_NAND.

        Enforce that
        :math:`\lnot (a \land b \land c \land ...)` is True, with a penalty
        factor ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns
            ``self`` so that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_NAND('a', 'b')  # enforce a NAND b

        >>> H = BooleanConstraints()
        >>> H.add_constraint_NAND('a', 'b', 'c', 'd')
        >>> # enforce a NAND b NAND c NAND d

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b = boolean_var('a'), boolean_var('b')
        >>> # enforce a NAND b
        >>> H = BooleanConstraints().add_constraint_NAND(a, b)

        """
        return self.add_constraint_NOT(AND(*variables), lam=lam)

    def add_constraint_NOR(self, *variables, lam=1):
        r"""add_constraint_NOR.

        Enforce that
        :math:`\lnot(a \lor b \lor c \lor d \lor ...)` is True, with a penalty
        factor ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_NOR('a', 'b')  # enforce a NOR b

        >>> H = BooleanConstraints()
        >>> H.add_constraint_NOR('a', 'b', 'c', 'd')
        >>> # enforce a NOR b NOR c NOR d

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b = boolean_var('a'), boolean_var('b')
        >>> # enforce a NOR b
        >>> H = BooleanConstraints().add_constraint_NOR(a, b)

        """
        P = 1 - BooleanConstraints(
            ).add_constraint_OR(*variables).to_penalty()
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 1))

    def add_constraint_XNOR(self, *variables, lam=1):
        r"""add_constraint_XNOR.

        Enforce that
        :math:`\lnot(v_0 \oplus v_1 \oplus ... \oplus v_n)` is True, with a
        penalty  factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ..., ``v_n = variables[-1]``. See
        ``qubovert.sat.XNOR`` for the XNOR convention that qubovert uses for
        more than two inputs.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the boolean
            variables.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_XNOR('a', 'b')  # enforce a XNOR b

        >>> H = BooleanConstraints()
        >>> H.add_constraint_XNOR('a', 'b', 'c')  # enforce a XNOR b XNOR c

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a, b = boolean_var('a'), boolean_var('b')
        >>> # enforce a XNOR b
        >>> H = BooleanConstraints().add_constraint_XNOR(a, b)

        """
        P = 1 - BooleanConstraints().add_constraint_XOR(
            *variables).to_penalty()
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 1))

    def add_constraint_NOT(self, a, lam=1):
        r"""add_constraint_NOT.

        Enforce that
        :math:`\lnot a` is True, with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for boolean variables ``a``, or its PUBO representation.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the clause.

        Return
        ------
        self : BooleanConstraints.
            Updates the BooleanConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        >>> H = BooleanConstraints()
        >>> H.add_constraint_NOT('a')  # enforce not a
        >>> H
        {('a',): 1}

        >>> from qubovert import boolean_var, BooleanConstraints
        >>> a = boolean_var('a')
        >>> # enforce not a
        >>> H = BooleanConstraints().add_constraint_NOT(a)

        """
        return self.add_constraint_eq_zero(BUFFER(a), lam=lam, bounds=(0, 1))
