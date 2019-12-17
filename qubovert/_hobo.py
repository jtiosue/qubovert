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
from .sat import OR, XOR, ONE, NOT, AND
from numpy import log2, ceil


__all__ = 'HOBO', 'binary_var'


# special constraint forms

def _special_constraints_le_zero(hobo, P, lam):
    """_special_constraints_le_zero.

    See if the constraint that ``P <= 0`` matches any special forms.

    Parameters
    ----------
    hobo : HOBO object.
        The HOBO to add the constraint to.
    P : PUBO object.
        The PUBO constraint such that ``P <= 0``.
        are present.
    lam : float > 0 or sympy.Symbol.
        Langrange multiplier to penalize violations of the constraint.

    Return
    ------
    success : bool.
        True if a special constraint was found and added to ``hobo``, else
        False.

    """
    # if P is of the form sum(x_i) <= 1.
    if P.offset == -1 and all(x == 1 for x in (P + 1).values()):
        hobo += lam * P * (P + 1) / 2
        return True

    return False


# helpers

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
        bounds = _pubo_value_extrema(P)
    elif bounds[0] is None:
        bounds = _pubo_value_extrema(P)[0], bounds[1]
    elif bounds[1] is None:
        bounds = bounds[0], _pubo_value_extrema(P)[1]

    return bounds


def binary_var(name):
    """binary_var.

    Create a HOBO (see ``qubovert.HOBO``) from a single binary variable.

    Parameters
    ----------
    name : any hashable object.
        Name of the binary variable.

    Return
    ------
    hobo : qubovert.HOBO object.
        The model representing the binary variable.

    Examples
    --------
    >>> from qubovert import binary_var, HOBO
    >>>
    >>> x0 = binary_var("x0")
    >>> print(x0)
    {('x0',): 1}
    >>> print(isinstance(x0, HOBO))
    True

    >>> x = [binary_var('x{}'.format(i)) for i in range(5)]
    >>> hobo = sum(x)
    >>> print(hobo)
    {('x0',): 1, ('x1',): 1, ('x2',): 1, ('x3',): 1, ('x4',): 1}
    >>> hobo **= 2
    >>> print(hobo)
    {('x0',): 1, ('x0', 'x1'): 2, ('x2', 'x0'): 2, ('x3', 'x0'): 2,
     ('x4', 'x0'): 2, ('x1',): 1, ('x2', 'x1'): 2, ('x3', 'x1'): 2,
     ('x4', 'x1'): 2, ('x2',): 1, ('x2', 'x3'): 2, ('x2', 'x4'): 2, ('x3',): 1,
     ('x4', 'x3'): 2, ('x4',): 1}
    >>> hobo *= -1
    >>> print(hobo.solve_bruteforce())
    {'x0': 1, 'x1': 1, 'x2': 1, 'x3': 1, 'x4': 1}
    >>> hobo.add_constraint_eq_zero(x[0] + x[1])
    >>> print(hobo.solve_bruteforce())
    {'x0': 0, 'x1': 0, 'x2': 1, 'x3': 1, 'x4': 1}

    """
    return HOBO({(name,): 1})


# main class

class HOBO(PUBO):
    """HOBO.

    This class deals with Higher Order Binary Optimization problems. HOBO
    inherits some methods and attributes from the ``PUBO`` class. See
    ``help(qubovert.PUBO)``.

    ``HOBO`` has all the same methods as ``PUBO``, but adds some constraint
    methods; namely

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
    - ``add_constraint_ONE``, ``add_constraint_NOT``,
    - ``add_constraint_eq_ONE``, ``add_constraint_eq_NOT``.

    See each of their docstrings for important details on their implementation.

    Notes
    -----
    - Variables names that begin with ``"__a"`` should not be used since they
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
    >>>
    >>> # minimize -x_0 - x_1 - x_2
    >>> for i in (0, 1, 2):
    >>>     H[(i,)] -= 1
    >>>
    >>> # subject to constraints
    >>> H.add_constraint_eq_zero(  # enforce that x_0 x_1 - x_2 == 0
    >>>     {(0, 1): 1, (2,): -1}
    >>> ).add_constraint_lt_zero(  # enforce that x_1 x_2 + x_0 < 1
    >>>     {(1, 2): 1, (0,): 1, (): -1}
    >>> )
    >>> print(H)
    >>> # {(1,): -2, (2,): -1, (0, 1): 2, (1, 2): 2, (0, 1, 2): 2}
    >>>
    >>> print(H.solve_bruteforce(all_solutions=True))
    >>> # [{0: 0, 1: 1, 2: 0}]
    >>>
    >>> Q = H.to_qubo()
    >>> solutions = [H.convert_solution(sol)
    >>>              for sol in Q.solve_bruteforce(all_solutions=True)]
    >>> print(solutions)
    >>> # [{0: 0, 1: 1, 2: 0}]  # matches the HOBO solution!
    >>>
    >>> L = H.to_ising()
    >>> solutions = [H.convert_solution(sol)
    >>>              for sol in L.solve_bruteforce(all_solutions=True)]
    >>> print(solutions)
    >>> # [{0: 0, 1: 1, 2: 0}]  # matches the HOBO solution!

    Enforce that c == a b

    >>> H = HOBO().add_constraint_eq_AND('c', 'a', 'b')
    >>> H
    {('c',): 3, ('b', 'a'): 1, ('c', 'a'): -2, ('c', 'b'): -2}

    >>> from any_module import qubo_solver
    >>> # or from qubovert.utils import solve_qubo_bruteforce as qubo_solver
    >>> H = HOBO()
    >>>
    >>> # make it favorable to AND variables a and b, and variables b and c
    >>> H.add_constraint_AND('a', 'b').add_constraint_AND('b', 'c')
    >>>
    >>> # make it favorable to OR variables b and c
    >>> H.add_constraint_OR('b', 'c')
    >>>
    >>> # make it favorable to (a AND b) OR (c AND d)
    >>> H.add_constraint_OR(['a', 'b'], ['c', 'd'])
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
        arguments : defines a dictionary or HOBO.
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
        """_append_constraint.

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
        return "__a%d" % (self._ancilla - 1)

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
        return {k: v for k, v in solution.items() if str(k)[:3] != "__a"}

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

    @staticmethod
    def integer_var(prefix, num_bits, log_trick=True):
        """integer_var.

        Return a HOBO object representing an integer variable with `num_bits`
        bits.

        Parameters
        ----------
        prefix : str.
            The prefix for the binary variable names.
        num_bits : int.
            Number of bits to represent the integer variable with.
        log_trick : bool (optional, defaults to True).
            Whether or not to use a log encoding for the integer.

        Return
        ------
        i : qubovert.HOBO object.

        Example
        -------
        >>> from qubovert import HOBO
        >>> var = HOBO.integer_var('a', 4)
        >>> print(var)
        {('a0',): 1, ('a1',): 2, ('a2',): 4, ('a3',): 8}

        >>> from qubovert import HOBO
        >>> var = HOBO.integer_var('a', 4, log_trick=False)
        >>> print(var)
        {('a0',): 1, ('a1',): 1, ('a2',): 1, ('a3',): 1}

        """
        var = HOBO()
        for i in range(num_bits):
            var[(str(prefix) + str(i),)] = pow(2, i) if log_trick else 1
        return var

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
        self._append_constraint("eq", P)

        min_val, max_val = _get_bounds(P, bounds)

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
        elif min_val == 0:
            self += lam * P
        elif max_val == 0:
            self -= lam * P
        else:
            self += lam * P * P

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
        self : HOBO.
            Updates the HOBO in place, but returns ``self`` so that operations
            can be strung together.

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

        >>> H = HOBO().add_constraint_ne_zero(
                {('a', 'b', 'c'): -1, ('a',): 2,
                ('a', 'b'): -4, ('c',): 3, (): -2}
            )
        >>> print(H)
        {('c', 'a', 'b'): -19, ('__a0', 'c', 'a', 'b'): -4,
         ('__a0', 'c', '__a1', 'a', 'b'): -4, ('c', '__a1', 'a', 'b'): 2,
         ('__a0', 'c', '__a2', 'a', 'b'): -8, ('c', '__a2', 'a', 'b'): 4,
         ('__a0', '__a3', 'c', 'a', 'b'): -16, ('__a3', 'c', 'a', 'b'): 8,
         ('__a0', 'c', '__a4', 'a', 'b'): -32, ('c', '__a4', 'a', 'b'): 16,
         ('a',): -8, ('c', 'a'): 12, ('__a0', 'a'): 8,
         ('__a0', '__a1', 'a'): 8, ('__a1', 'a'): -4,
         ('__a0', '__a2', 'a'): 16, ('__a2', 'a'): -8,
         ('__a0', '__a3', 'a'): 32, ('__a3', 'a'): -16,
         ('__a0', '__a4', 'a'): 64, ('__a4', 'a'): -32, ('a', 'b'): 24,
         ('__a0', 'a', 'b'): -16, ('__a0', '__a1', 'a', 'b'): -16,
         ('__a1', 'a', 'b'): 8, ('__a0', '__a2', 'a', 'b'): -32,
         ('__a2', 'a', 'b'): 16, ('__a0', '__a3', 'a', 'b'): -64,
         ('__a3', 'a', 'b'): 32, ('__a0', '__a4', 'a', 'b'): -128,
         ('__a4', 'a', 'b'): 64, ('__a0', 'c'): 12, ('__a0', 'c', '__a1'): 12,
         ('c', '__a1'): -6, ('__a0', 'c', '__a2'): 24, ('c', '__a2'): -12,
         ('__a0', '__a3', 'c'): 48, ('__a3', 'c'): -24,
         ('__a0', 'c', '__a4'): 96, ('c', '__a4'): -48, ('c',): -9, (): 9,
         ('__a0',): -8, ('__a0', '__a1'): -8, ('__a1',): 7,
         ('__a0', '__a2'): -16, ('__a2',): 16, ('__a3',): 40,
         ('__a0', '__a4'): -64, ('__a4',): 112, ('__a2', '__a1'): 4,
         ('__a3', '__a1'): 8, ('__a4', '__a1'): 16, ('__a3', '__a2'): 16,
         ('__a4', '__a2'): 32, ('__a0', '__a3'): -32, ('__a3', '__a4'): 64}
        >>> print(H.is_solution_valid({'b': 0, 'c': 0, 'a': 1}))
        False
        >>> print(H.is_solution_valid({'b': 0, 'c': 1, 'a': 1}))
        True

        """
        P = PUBO(P)
        self._append_constraint("ne", P)

        min_val, max_val = _get_bounds(P, bounds)

        if min_val == max_val == 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint cannot be satisfied")
            self += lam
        elif min_val > 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        elif max_val < 0:
            if not suppress_warnings:
                QUBOVertWarning.warn("Constraint is always satisfied")
        elif min_val == 0:
            return self.add_constraint_gt_zero(
                P, lam=lam,
                bounds=(min_val, max_val), suppress_warnings=suppress_warnings
            )
            self._pop_constraint('gt')
        elif max_val == 0:
            return self.add_constraint_lt_zero(
                P, lam=lam,
                bounds=(min_val, max_val), suppress_warnings=suppress_warnings
            )
            self._pop_constraint('lt')

        else:
            # don't mutate the P that we put in self._constraints
            P = P.copy()
            sign = 2 * binary_var(self._next_ancilla) - 1
            P += sign
            max_val += 1
            min_val -= 1
            if log_trick:
                for i in range(int(ceil(log2(max_val-min_val)))):
                    P += sign * pow(2, i) * binary_var(self._next_ancilla)
                    max_val += pow(2, i)
                    min_val -= pow(2, i)
            else:
                for _ in range(int(ceil(max_val-min_val-1))):
                    P += sign * binary_var(self._next_ancilla)
                    max_val += 1
                    min_val -= 1

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
          >>> H.value(test_sol)
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
        {('b', 'c', 'a'): -19, ('b', '__a0', 'c', 'a'): -2,
         ('b', 'c', 'a', '__a1'): -4, ('a',): -3, ('b', 'a'): 24,
         ('c', 'a'): 6, ('__a0', 'a'): 2, ('a', '__a1'): 4,
         ('b', '__a0', 'a'): -8, ('b', 'a', '__a1'): -16, ('c',): -3,
         ('__a0', 'c'): 6, ('c', '__a1'): 12, (): 4, ('__a0',): -3,
         ('__a1',): -4, ('__a0', '__a1'): 4}
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._append_constraint("lt", P)

        min_val, max_val = _get_bounds(P, bounds)

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
          {(0,): 1.7999999999999998, (0, 1): 4, (0, 2): -3.0, (0, '__a0'): 2,
           (1,): 5.6, (1, 2): -6.0, (1, '__a0'): 4, (2,): 1.0499999999999998,
           (2, '__a0'): -3.0, (): 0.16000000000000003, ('__a0',): 1.8}
          >>> test_sol = {0: 0, 1: 0, 2: 1, '__a0': 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(test_sol)
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
        {('b', 'c', 'a'): -19, ('b', 'c', 'a', '__a0'): -2,
         ('__a1', 'b', 'c', 'a'): -4, ('__a2', 'b', 'c', 'a'): -8, ('a',): -3,
         ('b', 'a'): 24, ('c', 'a'): 6, ('a', '__a0'): 2, ('__a1', 'a'): 4,
         ('__a2', 'a'): 8, ('b', 'a', '__a0'): -8, ('__a1', 'b', 'a'): -16,
         ('__a2', 'b', 'a'): -32, ('c',): -3, ('c', '__a0'): 6,
         ('__a1', 'c'): 12, ('__a2', 'c'): 24, (): 4, ('__a0',): -3,
         ('__a1',): -4, ('__a1', '__a0'): 4, ('__a2', '__a0'): 8,
         ('__a2', '__a1'): 16}
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._append_constraint("le", P)

        if _special_constraints_le_zero(self, P, lam):
            return self

        min_val, max_val = _get_bounds(P, bounds)

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
                for i in range(int(ceil(log2(-min_val + 1)))):
                    P[(self._next_ancilla,)] += pow(2, i)
                    max_val += pow(2, i)
            elif min_val:
                for _ in range(int(ceil(-min_val))):
                    P[(self._next_ancilla,)] += 1
                    max_val += 1

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
          >>> H.value(test_sol)
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

        >>> H = HOBO().add_constraint_gt_zero(
                {('a', 'b', 'c'): 1, ('a',): -1,
                 ('a', 'b'): 4, ('c',): -3, (): 2}
            )
        >>> H
        {('b', 'c', 'a'): -19, ('b', '__a0', 'c', 'a'): -2,
         ('b', 'c', 'a', '__a1'): -4, ('a',): -3, ('b', 'a'): 24,
         ('c', 'a'): 6, ('__a0', 'a'): 2, ('a', '__a1'): 4,
         ('b', '__a0', 'a'): -8, ('b', 'a', '__a1'): -16, ('c',): -3,
         ('__a0', 'c'): 6, ('c', '__a1'): 12, (): 4, ('__a0',): -3,
         ('__a1',): -4, ('__a0', '__a1'): 4}
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._append_constraint("gt", P)
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
          {(0,): 1.7999999999999998, (0, 1): 4, (0, 2): -3.0, (0, '__a0'): 2,
           (1,): 5.6, (1, 2): -6.0, (1, '__a0'): 4, (2,): 1.0499999999999998,
           (2, '__a0'): -3.0, (): 0.16000000000000003, ('__a0',): 1.8}
          >>> test_sol = {0: 0, 1: 0, 2: 1, '__a0': 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(test_sol)
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

        >>> H = HOBO().add_constraint_ge_zero(
                {('a', 'b', 'c'): 1, ('a',): -1,
                 ('a', 'b'): 4, ('c',): -3, (): 2}
            )
        >>> H
        {('b', 'c', 'a'): -19, ('b', 'c', 'a', '__a0'): -2,
         ('__a1', 'b', 'c', 'a'): -4, ('__a2', 'b', 'c', 'a'): -8, ('a',): -3,
         ('b', 'a'): 24, ('c', 'a'): 6, ('a', '__a0'): 2, ('__a1', 'a'): 4,
         ('__a2', 'a'): 8, ('b', 'a', '__a0'): -8, ('__a1', 'b', 'a'): -16,
         ('__a2', 'b', 'a'): -32, ('c',): -3, ('c', '__a0'): 6,
         ('__a1', 'c'): 12, ('__a2', 'c'): 24, (): 4, ('__a0',): -3,
         ('__a1',): -4, ('__a1', '__a0'): 4, ('__a2', '__a0'): 8,
         ('__a2', '__a1'): 16}
        >>> H.is_solution_valid({'b': 0, 'c': 0, 'a': 1})
        True
        >>> H.is_solution_valid({'b': 0, 'c': 1, 'a': 1})
        False

        """
        P = PUBO(P)
        self._append_constraint("ge", P)
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

        Add a penalty to the HOBO that enforces that
        :math:`a = v_0 \land v_1 \land v_2 \land ...`,
        with a penalty factor
        ``lam``, where ``v_1 = variables[0]``, ``v_2 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_eq_AND('a', 'b', 'c')  # enforce a == b AND c

        >>> H = HOBO()
        >>> # enforce (a AND b AND c AND d) == 'e'
        >>> H.add_constraint_eq_AND('e', 'a', 'b', 'c', 'd')

        >>> from qubovert import binary_var, HOBO
        >>> a, b, c = binary_var('a'), binary_var('b'), binary_var('c')
        >>> H = HOBO()
        >>> # enforce that a == b AND c
        >>> H.add_constraint_eq_AND(a, b, c)

        References
        ----------
        https://arxiv.org/pdf/1307.8041.pdf equation 6.

        """
        n = len(variables)
        if n < 2:
            raise ValueError("Must supply at least two variables to AND. "
                             "See ``add_constraint_eq_ONE`` for less.")

        a = ONE(a)
        b, c = 1, 1
        for v in variables[:n // 2]:
            b *= ONE(v)
        for v in variables[n // 2:]:
            c *= ONE(v)

        P = 3 * a + b * c - 2 * a * (b + c)

        return self.add_constraint_eq_zero(P, lam, bounds=(0, 3))

    def add_constraint_eq_OR(self, a, *variables, lam=1):
        r"""add_constraint_eq_OR.

        Add a penalty to the HOBO that enforces that
        :math:`v_0 \lor v_1 \lor v_2 \lor ... == a`,
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_eq_OR('a', 'b', 'c')  # enforce a == b OR c

        >>> H = HOBO()
        >>> # enforce a == b OR c OR d
        >>> H.add_constraint_eq_OR('a', 'b', 'c', 'd')

        """
        n = len(variables)
        if n < 2:
            raise ValueError("Must supply at least two variables to OR.")

        a = ONE(a)
        if n == 2:
            b, c = ONE(variables[0]), ONE(variables[1])
            P = a + b + c + b * c - 2 * a * (b + c)
            bounds = 0, 3
        else:
            P = HOBO().add_constraint_NOR(*variables) - a
            bounds = -1, 1

        return self.add_constraint_eq_zero(P, lam=lam, bounds=bounds)

    def add_constraint_eq_XOR(self, a, *variables, lam=1):
        r"""add_constraint_eq_XOR.

        Add a penalty to the HOBO that enforces that
        :math:`v_0 \oplus v_1 \oplus ... == a`
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_eq_XOR('a', 'b', 'c')  # enforce a == b XOR c

        >>> H = HOBO()
        >>> # enforce a == b XOR c XOR d
        >>> H.add_constraint_eq_XOR('a', 'b', 'c', 'd')

        """
        P = HOBO().add_constraint_XNOR(*variables) - ONE(a)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(-1, 1))

    def add_constraint_eq_ONE(self, a, b, lam=1):
        r"""add_constraint_eq_ONE.

        Add a penalty to the HOBO that enforces that :math:`a == b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variables ``a``, or its PUBO representation.
        b : any hashable object or a dict.
            The label for binary variables ``b``, or its PUBO representation.
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
        >>> H.add_constraint_eq_ONE('a', 'b')  # enforce a == b

        >>> from qubovert import HOBO, binary_var
        >>> a, b = binary_var('a'), binary_var('b')
        >>> H = HOBO()
        >>> H.add_constraint_eq_ONE(a, b)  # enforce a == b

        """
        P = ONE(a) - ONE(b)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(-1, 1))

    def add_constraint_eq_NAND(self, a, *variables, lam=1):
        r"""add_constraint_eq_NAND.

        Add a penalty to the HOBO that enforces that
        :math:`\lnot (v_0 \land v_1 \land v_2 \land ...) == a`, with a
        penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variables ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_eq_NAND('a', 'b', 'c')  # enforce a == b NAND c

        >>> H = HOBO()
        >>> # enforce a == b NAND c NAND d
        >>> H.add_constraint_eq_NAND('a', 'b', 'c', 'd')

        >>> from qubovert import binary_var, HOBO
        >>> a, b, c = binary_var('a'), binary_var('b'), binary_var('c')
        >>> # enforce a == b NAND c
        >>> H = HOBO().add_constraint_eq_NAND(a, b, c)

        """
        n = len(variables)
        if n < 2:
            raise ValueError("Must supply at least two variables to NAND. "
                             "See ``add_constraint_eq_NOT`` for less.")
        b, c = 1, 1
        for v in variables[:n // 2]:
            b *= ONE(v)
        for v in variables[n // 2:]:
            c *= ONE(v)

        P = NOT(a) * (3 - 2 * (b + c)) + b * c
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 3))

    def add_constraint_eq_NOR(self, a, *variables, lam=1):
        r"""add_constraint_eq_NOR.

        Add a penalty to the HOBO that enforces that
        :math:`\lnot(v_0 \lor v_1 \lor v_2 \lor ...) == a`,
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_eq_NOR('a', 'b', 'c')  # enforce a == b NOR c

        >>> H = HOBO()
        >>> # enforce a == b NOR c NOR d
        >>> H.add_constraint_eq_NOR('a', 'b', 'c', 'd')

        >>> from qubovert import binary_var, HOBO
        >>> # enforce a == b NOR c
        >>> a, b, c = binary_var('a'), binary_var('b'), binary_var('c')
        >>> H.add_constraint_eq_NOR(a, b, c)

        """
        n = len(variables)
        if n < 2:
            raise ValueError("Must supply at least two variables to NOR.")

        a = ONE(a)
        if n == 2:
            b, c = ONE(variables[0]), ONE(variables[1])
            P = 1 - a - b - c + b * c + 2 * a * (b + c)
            bounds = 0, 3
        else:
            P = HOBO().add_constraint_OR(*variables) - a
            bounds = -1, 1

        return self.add_constraint_eq_zero(P, lam=lam, bounds=bounds)

    def add_constraint_eq_XNOR(self, a, *variables, lam=1):
        r"""add_constraint_eq_XNOR.

        Add a penalty to the HOBO that enforces that
        :math:\lnot(`v_0 \oplus v_1 \oplus ...) == a`
        with a penalty factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ...

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variable ``a``, or its PUBO representation.
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_eq_XNOR('a', 'b', 'c')  # enforce a == b XNOR c

        >>> H = HOBO()
        >>> # enforce a == b XNOR c XNOR d
        >>> H.add_constraint_eq_XNOR('a', 'b', 'c', 'd')

        >>> from qubovert import binary_var, HOBO
        >>> a, b, c = binary_var('a'), binary_var('b'), binary_var('c')
        >>> # enforce a == b XNOR c
        >>> H = HOBO().add_constraint_eq_XNOR(a, b, c)

        """
        P = HOBO().add_constraint_XOR(*variables) - ONE(a)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(-1, 1))

    def add_constraint_eq_NOT(self, a, b, lam=1):
        r"""NOT.

        Add a penalty to the HOBO that enforces that :math:`\lnot a == b` with
        a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variable ``a``, or its PUBO representation.
        b : any hashable object or a dict.
            The label for binary variable ``b``, or its PUBO representation.
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
        >>> H.add_constraint_eq_NOT('a', 'b')  # enforce NOT(a) == b

        >>> from qubovert import binary_var, HOBO
        >>> a, b = binary_var('a'), binary_var('b')
        >>> H.add_constraint_eq_NOT(a, b)  # enforce NOT(a) == b

        """
        P = HOBO().add_constraint_ONE(a) - ONE(b)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(-1, 1))

    def add_constraint_AND(self, *variables, lam=1):
        r"""add_constraint_AND.

        Add a penalty to the HOBO that is only zero when
        :math:`a \land b \land c \land ...` is True, with a penalty factor
        ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_AND('a', 'b')  # enforce a AND b
        >>> H
        {('b', 'a'): -1, (): 1}

        >>> H = HOBO()
        >>> H.add_constraint_AND('a', 'b', 'c', 'd')
        >>> # enforce a AND b AND c AND d

        >>> from qubovert import binary_var, HOBO
        >>> a, b = binary_var('a'), binary_var('b')
        >>> # enforce a AND b
        >>> H = HOBO().add_constraint_AND(a, b)

        """
        return self.add_constraint_ONE(AND(*variables), lam=lam)

    def add_constraint_OR(self, *variables, lam=1):
        r"""add_constraint_OR.

        Add a penalty to the HOBO that is only nonzero when
        :math:`a \lor b \lor c \lor d \lor ...` is True, with a penalty factor
        ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_OR('a', 'b')  # enforce a OR b
        >>> H
        {('a',): -1, ('b',): -1, ('b', 'a'): 1, (): 1}

        >>> H = HOBO()
        >>> H.add_constraint_OR('a', 'b', 'c', 'd')  # enforce a OR b OR c OR d

        >>> from qubovert import binary_var, HOBO
        >>> a, b = binary_var('a'), binary_var('b')
        >>> # enforce a OR b
        >>> H = HOBO().add_constraint_OR(a, b)

        """
        P = 1 - OR(*variables)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 1))

    def add_constraint_XOR(self, *variables, lam=1):
        r"""add_constraint_XOR.

        Add a penalty to the HOBO that is only nonzero when
        :math:`v_0 \oplus v_1 \oplus ... \oplus v_n` is True, with a penalty
        factor ``lam``, where ``v_0 = variables[0]``, ``v_1 = variables[1]``,
        ..., ``v_n = variables[-1]``. See ``qubovert.sat.XOR`` for the
        XOR convention that qubovert uses for more than two inputs.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_XOR('a', 'b')  # enforce a XOR b

        >>> H = HOBO()
        >>> H.add_constraint_XOR('a', 'b', 'c')  # enforce a XOR b XOR c

        >>> from qubovert import binary_var, HOBO
        >>> a, b = binary_var('a'), binary_var('b')
        >>> # enforce a XOR b
        >>> H = HOBO().add_constraint_XOR(a, b)

        """
        P = 1 - XOR(*variables)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 1))

    def add_constraint_ONE(self, a, lam=1):
        r"""add_constraint_ONE.

        Add a penalty to the HOBO that is only nonzero when :math:`a == 1` is
        True, with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variable ``a``, or its PUBO representation.
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
        >>> H.add_constraint_ONE('a')  # enforce a

        >>> from qubovert import binary_var, HOBO
        >>> a = binary_var('a')
        >>> # enforce a
        >>> H = HOBO().add_constraint_ONE(a)

        """
        return self.add_constraint_eq_zero(NOT(a), lam=lam, bounds=(0, 1))

    def add_constraint_NAND(self, *variables, lam=1):
        r"""add_constraint_NAND.

        Add a penalty to the HOBO that is only zero when
        :math:`\lnot (a \land b \land c \land ...)` is True, with a penalty
        factor ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_NAND('a', 'b')  # enforce a NAND b

        >>> H = HOBO()
        >>> H.add_constraint_NAND('a', 'b', 'c', 'd')
        >>> # enforce a NAND b NAND c NAND d

        >>> from qubovert import binary_var, HOBO
        >>> a, b = binary_var('a'), binary_var('b')
        >>> # enforce a NAND b
        >>> H = HOBO().add_constraint_NAND(a, b)

        """
        return self.add_constraint_NOT(AND(*variables), lam=lam)

    def add_constraint_NOR(self, *variables, lam=1):
        r"""add_constraint_NOR.

        Add a penalty to the HOBO that is only nonzero when
        :math:`\lnot(a \lor b \lor c \lor d \lor ...)` is True, with a penalty
        factor ``lam``, where ``a = variables[0]``, ``b = variables[1]``, etc.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_NOR('a', 'b')  # enforce a NOR b

        >>> H = HOBO()
        >>> H.add_constraint_NOR('a', 'b', 'c', 'd')
        >>> # enforce a NOR b NOR c NOR d

        >>> from qubovert import binary_var, HOBO
        >>> a, b = binary_var('a'), binary_var('b')
        >>> # enforce a NOR b
        >>> H = HOBO().add_constraint_NOR(a, b)

        """
        P = 1 - HOBO().add_constraint_OR(*variables)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 1))

    def add_constraint_XNOR(self, *variables, lam=1):
        r"""add_constraint_XNOR.

        Add a penalty to the HOBO that is only nonzero when
        :math:`\lnot(v_0 \oplus v_1 \oplus ... \oplus v_n)` is True, with a
        penalty  factor ``lam``, where ``v_0 = variables[0]``,
        ``v_1 = variables[1]``, ..., ``v_n = variables[-1]``. See
        ``qubovert.sat.XNOR`` for the XNOR convention that qubovert uses for
        more than two inputs.

        Parameters
        ----------
        *variables : arguments.
            Each element of variables is a hashable object or a dict
            (its PUBO representation). They are the label of the binary
            variables.
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
        >>> H.add_constraint_XNOR('a', 'b')  # enforce a XNOR b

        >>> H = HOBO()
        >>> H.add_constraint_XNOR('a', 'b', 'c')  # enforce a XNOR b XNOR c

        >>> from qubovert import binary_var, HOBO
        >>> a, b = binary_var('a'), binary_var('b')
        >>> # enforce a XNOR b
        >>> H = HOBO().add_constraint_XNOR(a, b)

        """
        P = 1 - HOBO().add_constraint_XOR(*variables)
        return self.add_constraint_eq_zero(P, lam=lam, bounds=(0, 1))

    def add_constraint_NOT(self, a, lam=1):
        r"""add_constraint_NOT.

        Add a penalty to the HOBO that is only nonzero when
        :math:`\lnot a` is True, with a penalty factor ``lam``.

        Parameters
        ----------
        a : any hashable object or a dict.
            The label for binary variables ``a``, or its PUBO representation.
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
        >>> H.add_constraint_NOT('a')  # enforce not a
        >>> H
        {('a',): 1}

        >>> from qubovert import binary_var, HOBO
        >>> a = binary_var('a')
        >>> # enforce not a
        >>> H = HOBO().add_constraint_NOT(a)

        """
        return self.add_constraint_eq_zero(ONE(a), lam=lam, bounds=(0, 1))
