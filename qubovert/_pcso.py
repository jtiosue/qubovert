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

"""_pcso.py.

Contains the PCSO class. See ``help(qubovert.PCSO)``.

"""

from . import PUSO, PCBO
from .utils import puso_to_pubo, pubo_to_puso


__all__ = 'PCSO', 'spin_var'


def spin_var(name):
    """spin_var.

    Create a PCSO (see ``qubovert.PCSO``) from a single spin variable.

    Parameters
    ----------
    name : any hashable object.
        Name of the spin variable.

    Return
    ------
    pcso : qubovert.PCSO object.
        The model representing the spin variable.

    Examples
    --------
    >>> from qubovert import spin_var, PCSO
    >>>
    >>> z0 = spin_var("z0")
    >>> print(z0)
    {('z0',): 1}
    >>> print(isinstance(z0, PCSO))
    True
    >>> print(z0.name)
    z0

    >>> z = [spin_var('z{}'.format(i)) for i in range(5)]
    >>> pcso = sum(z)
    >>> print(pcso)
    {('z0',): 1, ('z1',): 1, ('z2',): 1, ('z3',): 1, ('z4',): 1}
    >>> pcso **= 2
    >>> print(pcso)
    {(): 5, ('z0', 'z1'): 2, ('z2', 'z0'): 2, ('z3', 'z0'): 2, ('z0', 'z4'): 2,
     ('z2', 'z1'): 2, ('z3', 'z1'): 2, ('z4', 'z1'): 2, ('z3', 'z2'): 2,
     ('z2', 'z4'): 2, ('z3', 'z4'): 2}
    >>> pcso *= -1
    >>> print(pcso.solve_bruteforce(all_solutions=True))
    [{'z0': -1, 'z1': -1, 'z2': -1, 'z3': -1, 'z4': -1},
     {'z0': 1, 'z1': 1, 'z2': 1, 'z3': 1, 'z4': 1}]
    >>> pcso.add_constraint_eq_zero(z[0] + z[1])
    >>> print(pcso.solve_bruteforce(all_solutions=True))
    [{'z0': -1, 'z1': 1, 'z2': -1, 'z3': -1, 'z4': -1},
     {'z0': -1, 'z1': 1, 'z2': 1, 'z3': 1, 'z4': 1},
     {'z0': 1, 'z1': -1, 'z2': -1, 'z3': -1, 'z4': -1},
     {'z0': 1, 'z1': -1, 'z2': 1, 'z3': 1, 'z4': 1}]

    Notes
    -----
    ``qubovert.spin_var(name)`` is equivalent to
    ``qubovert.PCSO.create_var(name)``.

    """
    return PCSO.create_var(name)


def _empty_pcbo(pcso):
    """_empty_pcbo.

    Create an empty PCBO whose ancilla variables begin at
    ``pcso.num_ancillas``.

    Parameters
    ----------
    pcso : PCSO object.

    Return
    ------
    pcbo : PCBO object.

    """
    h = PCBO()
    h._ancilla = pcso._ancilla
    return h


class PCSO(PUSO):
    """PCBO.

    This class deals with Polynomial Constrained Spin Optimization. PCSO
    inherits some methods and attributes from the ``PUSO`` class. See
    ``help(qubovert.PUSO)``.

    ``PCSO`` has all the same methods as ``PUSO``, but adds some constraint
    methods; namely

    - ``add_constraint_eq_zero(H, lam=1, ...)`` enforces that ``H == 0`` by
      penalizing with ``lam``,
    - ``add_constraint_ne_zero(H, lam=1, ...)`` enforces that ``H != 0`` by
      penalizing with ``lam``,
    - ``add_constraint_lt_zero(H, lam=1, ...)`` enforces that ``H < 0`` by
      penalizing with ``lam``,
    - ``add_constraint_le_zero(H, lam=1, ...)`` enforces that ``H <= 0`` by
      penalizing with ``lam``,
    - ``add_constraint_gt_zero(H, lam=1, ...)`` enforces that ``H > 0`` by
      penalizing with ``lam``, and
    - ``add_constraint_ge_zero(H, lam=1, ...)`` enforces that ``H >= 0`` by
      penalizing with ``lam``.

    Each of these takes in a PUSO ``H`` and a lagrange multiplier ``lam``
    that defaults to 1. See each of their docstrings for important details on
    their implementation.

    Notes
    -----
    - Variables names that begin with ``"__a"`` should not be used since they
      are used internally to deal with some ancilla variables to enforce
      constraints.
    - The ``self.solve_bruteforce`` method will solve the PCSO ensuring that
      all the inputted constraints are satisfied. Whereas
      ``qubovert.utils.solve_puso_bruteforce(self)`` or
      ``qubovert.utils.solve_puso_bruteforce(self.to_pubo())`` will solve the
      PUSO created from the PCSO. If the inputted constraints are not
      enforced strong enough (ie too small lagrange multipliers) then these may
      not give the correct result, whereas ``self.solve_bruteforce()`` will
      always give the correct result (ie one that satisfies all the
      constraints).

    Examples
    --------
    See ``qubovert.PUSO`` for more examples of using PCSO without
    constraints. See ``qubovert.PCBO`` for many constraint examples in PUBO
    form. ``PCSO`` is the same but converting to PUSO.

    >>> H = PCSO()
    >>> H.add_constraint_eq_zero({('a', 1): 2, (1, 2): -1, (): -1})
    >>> H
    {(): 6, ('a', 2): -4, ('a', 1): -4, (1, 2): 2}

    >>> H = PCSO()
    >>> H.add_constraint_eq_zero(
            {(0, 1): 1}
        ).add_constraint_lt_zero(
            {(1, 2): 1, (): -1}
        )

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        This class deals with polynomial constrained spin optimization.
        Note that it is generally more efficient to initialize an empty PCSO
        object and then build the PCSO, rather than initialize a PCSO object
        with an already built dict.

        Parameters
        ----------
        arguments : define a dictionary with ``dict(*args, **kwargs)``.
            The dictionary will be initialized to follow all the convensions of
            the class. Alternatively, ``args[0]`` can be a PCSO.

        Examples
        --------
        >>> pcso = PCSO()
        >>> pcso[('a',)] += 5
        >>> pcso[(0, 'a')] -= 2
        >>> pcso -= 1.5
        >>> pcso
        {('a',): 5, ('a', 0): -2, (): -1.5}
        >>> pcso.add_constraint_eq_zero({('a',): 1, ('b',): 1}, lam=5)
        >>> pcso
        {('a',): 5, ('a', 0): -2, (): 8.5, ('a', 'b'): 10}

        >>> pcso = PCSO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> pcso
        {('a',): 5, ('a', 0, 1): -2, (): -1.5}

        """
        PCBO.__init__(self, *args, **kwargs)

    def update(self, *args, **kwargs):
        """update.

        Update the PCSO but following all the conventions of this class.

        Parameters
        ----------
        ``*args/**kwargs`` : defines a dictionary or PCSO.
            Ie ``d = dict(*args, **kwargs)``.
            Each element in d will be added in place to this instance following
            all the required convensions.

        """
        PCBO.update(self, *args, **kwargs)

    @property
    def constraints(self):
        """constraints.

        Return the constraints of the PCSO.

        Return
        ------
        res : dict.
            The keys of ``res`` are ``'eq'``, ``'ne'``, ``'lt'``, ``'le'``,
            ``'gt'``, and ``'ge'``. The values are lists of ``qubovert.PUSO``
            objects. For a given key, value pair ``k, v``, the ``v[i]`` element
            represents the PUSO ``v[i]`` being
            == 0 if ``k == 'eq'``, != 0 if ``k == 'ne'``,
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
            One of ``'eq'``, ``'lt'``, ``'le'``, ``'gt'``, or ``'ge'``.
        constraint : qubovert.PUSO object.

        """
        PCBO._append_constraint(self, key, constraint)

    @property
    def num_ancillas(self):
        """num_ancillas.

        Return the number of ancilla variables introduced to the PCSO in
        order to enforce the inputted constraints.

        Returns
        -------
        num : int.
            Number of ancillas in the PCSO.

        """
        return self._ancilla

    @classmethod
    def remove_ancilla_from_solution(cls, solution):
        """remove_ancilla_from_solution.

        Take a solution to the PCSO and remove all the ancilla variables, (
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
        return PCBO.remove_ancilla_from_solution(solution)

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
        return PCBO.is_solution_valid(self, solution)

    # override
    def __round__(self, ndigits=None):
        """round.

        Round values of the PCSO object.

        Parameters
        ----------
        ndigits : int.
            Number of decimal digits to round to.

        Returns
        -------
        res : PCSO object.
            Copy of self but with each value rounded to ``ndigits`` decimal
            digits. Each value has a type according to the docstring
            specifications of ``round``, see ``help(round)``.

        """
        return PCBO.__round__(self, ndigits)

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
        res : a PCSO object.
            Same as ``self`` but with all the symbols replaced with values.

        """
        return PCBO.subs(self, *args, **kwargs)

    def add_constraint_eq_zero(self,
                               H, lam=1,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_eq_zero.

        Enforce that ``H == 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        H : dict representing a PUSO.
            The PUSO constraint such that H == 0. Note that ``H`` will be
            converted to a ``qubovert.PUSO`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.PUSO)``.
            Please note that if ``H`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUSO ``H`` can take. If ``bounds`` is None, then they may be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : PCSO.
            Updates the PCSO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        The following enforces that :math:`\sum_{i=1}^{3} i z_i z_{i+1} == 0`.

        >>> H = PCSO()
        >>> H.add_constraint_eq_zero({(1, 2): 1, (2, 3): 2, (3, 4): 3})

        Here we show how operations can be strung together.

        >>> H = PCSO()
        >>> H.add_constraint_lt_zero(
                {(0, 1): 1}
            ).add_constraint_eq_zero(
                {(1, 2): 1, (): -1}
            )

        """
        H = PUSO(H)
        self._append_constraint("eq", H)
        if not lam:
            return self

        h = _empty_pcbo(self).add_constraint_eq_zero(
            puso_to_pubo(H), lam=lam,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self += pubo_to_puso(h)
        return self

    def add_constraint_ne_zero(self,
                               H, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_ne_zero.

        Enforce that ``H != 0`` by penalizing invalid solutions with ``lam``.
        See Notes below for more details.

        Parameters
        ----------
        H : dict representing a PUSO.
            The PUSO constraint such that H != 0. Note that ``H`` will be
            converted to a ``qubovert.PUSO`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.PUSO)``.
            Please note that if ``H`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        log_trick : bool (optional, defaults to True).
            Whether or not to use the log trick to enforce the inequality
            constraint. See Notes below for more details.
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUSO ``H`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : PCSO.
            Updates the PCSO in place, but returns ``self`` so that operations
            can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints. However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function.
        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\min_x \text{P.value(x)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\min_x \text{P.value(x)}|` ancilla
          bits will be used.

        """
        H = PUSO(H)
        self._append_constraint("ne", H)
        if not lam:
            return self
        h = _empty_pcbo(self).add_constraint_ne_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self += pubo_to_puso(h)
        return self

    def add_constraint_lt_zero(self,
                               H, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_lt_zero.

        Enforce that ``H < 0`` by penalizing invalid solutions with ``lam``.
        See Notes below for more details.

        Parameters
        ----------
        H : dict representing a PUSO.
            The PUSO constraint such that H < 0. Note that ``H`` will be
            converted to a ``qubovert.PUSO`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.PUSO)``.
            Please note that if ``H`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        log_trick : bool (optional, defaults to True).
            Whether or not to use the log trick to enforce the inequality
            constraint. See Notes below for more details.
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUSO ``H`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : PCSO.
            Updates the PCSO in place, but returns ``self`` so that operations
            can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`z_0 + 2z_1 + ... < 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = PCSO()
          >>> H.add_constraint_lt_zero(
          >>>     {(0,): 0.5, (): 1.65, (1,): 1.0, (2,): -0.25})
          >>> test_sol = {0: -1, 1: -1, 2: 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(test_sol)
          0.01

          {0: -1, 1: -1, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\min_z \text{H.value(z)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\min_z \text{H.value(z)}|` ancilla
          bits will be used.

        """
        H = PUSO(H)
        self._append_constraint("lt", H)
        if not lam:
            return self
        h = _empty_pcbo(self).add_constraint_lt_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self += pubo_to_puso(h)
        return self

    def add_constraint_le_zero(self,
                               H, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_le_zero.

        Enforce that ``H <= 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        H : dict representing a PUSO.
            The PUSO constraint such that H <= 0. Note that ``H`` will be
            converted to a ``qubovert.PUSO`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.PUSO)``.
            Please note that if ``H`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        log_trick : bool (optional, defaults to True).
            Whether or not to use the log trick to enforce the inequality
            constraint. See Notes below for more details.
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUSO ``H`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : PCSO.
            Updates the PCSO in place, but returns ``self`` so that operations
            can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`z_0 + 2z_1 + ... \leq 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = PCSO()
          >>> H.add_constraint_le_zero(
                  {(0,): 0.5, (): 1.15, (1,): 1.0, (2,): -0.75})
          >> H
          {(0,): 1.65, (): 4.785, (0, 1): 1.0, (1,): 3.3, (0, 2): -0.75,
           (2,): -2.4749999999999996, ('__a0', 0): 0.5, ('__a0',): 1.65,
           (1, 2): -1.5, ('__a0', 1): 1.0, ('__a0', 2): -0.75}
          >>> test_sol = {0: -1, 1: -1, 2: 1, '__a0': 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(test_sol)
          0.01

          {0: -1, 1: -1, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\min_z \text{H.value(z)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\min_z \text{H.value(z)}|` ancilla
          bits will be used.

        """
        H = PUSO(H)
        self._append_constraint("le", H)
        if not lam:
            return self
        h = _empty_pcbo(self).add_constraint_le_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self += pubo_to_puso(h)
        return self

    def add_constraint_gt_zero(self,
                               H, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_gt_zero.

        Enforce that ``H > 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        H : dict representing a PUSO.
            The PUSO constraint such that H > 0. Note that ``H`` will be
            converted to a ``qubovert.PUSO`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.PUSO)``.
            Please note that if ``H`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        log_trick : bool (optional, defaults to True).
            Whether or not to use the log trick to enforce the inequality
            constraint. See Notes below for more details.
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUSO ``H`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : PCSO.
            Updates the PCSO in place, but returns ``self`` so that operations
            can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`z_0 + 2z_1 + ... > 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = PCSO()
          >>> H.add_constraint_gt_zero(
                  {(0,): -0.5, (): -1.65, (1,): -1.0, (2,): 0.25})
          >>> test_sol = {0: -1, 1: -1, 2: 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(sol)
          0.01

          {0: -1, 1: -1, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\max_z \text{H.value(z)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\max_z \text{H.value(z)}|` ancilla
          bits will be used.

        """
        H = PUSO(H)
        self._append_constraint("gt", H)
        if not lam:
            return self
        h = _empty_pcbo(self).add_constraint_gt_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self += pubo_to_puso(h)
        return self

    def add_constraint_ge_zero(self,
                               H, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_ge_zero.

        Enforce that ``H >= 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        H : dict representing a PUSO.
            The PUSO constraint such that H >= 0. Note that ``H`` will be
            converted to a ``qubovert.PUSO`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.PUSO)``.
            Please note that if ``H`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        log_trick : bool (optional, defaults to True).
            Whether or not to use the log trick to enforce the inequality
            constraint. See Notes below for more details.
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            PUSO ``H`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : PCSO.
            Updates the PCSO in place, but returns ``self`` so that operations
            can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`z_0 + 2z_1 + ... \geq 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = PCSO()
          >>> H.add_constraint_ge_zero(
                  {(0,): -0.5, (): -1.15, (1,): -1.0, (2,): 0.75})
          >>> H
          {(0,): 1.65, (): 4.785, (0, 1): 1.0, (1,): 3.3, (0, 2): -0.75,
           (2,): -2.4749999999999996, ('__a0', 0): 0.5, ('__a0',): 1.65,
           (1, 2): -1.5, ('__a0', 1): 1.0, ('__a0', 2): -0.75}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.value(test_sol)
          0.01

          {0: -1, 1: -1, 2: 1} is a valid solution to ``H``, but it will still
          cause a nonzero penalty to be added to the objective function.

        - To enforce the inequality constraint, ancilla bits will be
          introduced (labels with `_a`). If ``log_trick`` is ``True``, then
          approximately :math:`\log_2 |\max_z \text{H.value(z)}|`
          ancilla bits will be used. If ``log_trick`` is ``False``, then
          approximately :math:`|\max_z \text{H.value(z)}|` ancilla
          bits will be used.

        """
        H = PUSO(H)
        self._append_constraint("ge", H)
        if not lam:
            return self
        h = _empty_pcbo(self).add_constraint_ge_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self += pubo_to_puso(h)
        return self
