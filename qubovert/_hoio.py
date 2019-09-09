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

"""_hoio.py.

Contains the HOIO class. See ``help(qubovert.HOIO)``.

"""

from . import HIsing, HOBO
from .utils import hising_to_pubo, pubo_to_hising


__all__ = 'HOIO',


class HOIO(HIsing):
    """HOBO.

    This class deals with Higher Order Ising Optimization problems. HOIO
    inherits some methods and attributes from the ``HIsing`` class. See
    ``help(qubovert.HIsing)``.

    ``HOIO`` has all the same methods as ``HIsing``, but adds some constraint
    methods; namely

    - ``add_constraint_eq_zero(H, lam=1, ...)`` enforces that ``H == 0`` by
      penalizing with ``lam``,
    - ``add_constraint_lt_zero(H, lam=1, ...)`` enforces that ``H < 0`` by
      penalizing with ``lam``,
    - ``add_constraint_le_zero(H, lam=1, ...)`` enforces that ``H <= 0`` by
      penalizing with ``lam``,
    - ``add_constraint_gt_zero(H, lam=1, ...)`` enforces that ``H > 0`` by
      penalizing with ``lam``, and
    - ``add_constraint_ge_zero(H, lam=1, ...)`` enforces that ``H >= 0`` by
      penalizing with ``lam``.

    Each of these takes in a HIsing ``H`` and a lagrange multiplier ``lam``
    that defaults to 1. See each of their docstrings for important details on
    their implementation.

    Notes
    -----
    - Variables names that begin with ``"__a"`` should not be used since they
      are used internally to deal with some ancilla variables to enforce
      constraints.
    - The ``self.solve_bruteforce`` method will solve the HOIO ensuring that
      all the inputted constraints are satisfied. Whereas
      ``qubovert.utils.solve_hising_bruteforce(self)`` or
      ``qubovert.utils.solve_hising_bruteforce(self.to_pubo())`` will solve the
      HIsing created from the HOIO. If the inputted constraints are not
      enforced strong enough (ie too small lagrange multipliers) then these may
      not give the correct result, whereas ``self.solve_bruteforce()`` will
      always give the correct result (ie one that satisfies all the
      constraints).

    Examples
    --------
    See ``qubovert.HIsing`` for more examples of using HOIO without
    constraints. See ``qubovert.HOBO`` for many constraint examples in PUBO
    form. ``HOIO`` is the same but converting to HIsing.

    >>> H = HOIO()
    >>> H.add_constraint_eq_zero({('a', 1): 2, (1, 2): -1, (): -1})
    >>> H
    {(): 6, ('a', 2): -4, ('a', 1): -4, (1, 2): 2}

    >>> H = HOIO()
    >>> H.add_constraint_eq_zero(
            {(0, 1): 1}
        ).add_constraint_lt_zero(
            {(1, 2): 1, (): -1}
        )

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        This class deals with higher order ising optimization problems.
        Note that it is generally more efficient to initialize an empty HOIO
        object and then build the HOIO, rather than initialize a HOIO object
        with an already built dict.

        Parameters
        ----------
        arguments : define a dictionary with ``dict(*args, **kwargs)``.
            The dictionary will be initialized to follow all the convensions of
            the class. Alternatively, ``args[0]`` can be a HOIO.

        Examples
        -------
        >>> hoio = HOIO()
        >>> hoio[('a',)] += 5
        >>> hoio[(0, 'a')] -= 2
        >>> hoio -= 1.5
        >>> hoio
        {('a',): 5, ('a', 0): -2, (): -1.5}
        >>> hoio.add_constraint_eq_zero({('a',): 1, ('b',): 1}, lam=5)
        >>> hoio
        {('a',): 5, ('a', 0): -2, (): 8.5, ('a', 'b'): 10}

        >>> hoio = HOIO({('a',): 5, (0, 'a', 1): -2, (): -1.5})
        >>> hoio
        {('a',): 5, ('a', 0, 1): -2, (): -1.5}

        """
        HOBO.__init__(self, *args, **kwargs)

    def update(self, *args, **kwargs):
        """update.

        Update the HOIO but following all the conventions of this class.

        Parameters
        ----------
        *args and **kwargs : defines a dictionary or HOIO.
            Ie ``d = dict(*args, **kwargs)``.
            Each element in d will be added in place to this instance following
            all the required convensions.

        """
        HOBO.update(self, *args, **kwargs)

    @property
    def constraints(self):
        """constraints.

        Return the constraints of the HOIO.

        Return
        ------
        res : dict.
            The keys of ``res`` are ``'eq'``, ``'lt'``, ``'le'``, ``'gt'``, and
            ``'ge'``. The values are lists of ``qubovert.HIsing`` objects. For
            a given key, value pair ``k, v``, the ``v[i]`` element represents
            the HIsing ``v[i]`` being == 0 if ``k == 'eq'``,
            < 0 if ``k == 'lt'``, <= 0 if ``k == 'le'``,
            > 0 if ``k == 'gt'``, >= 0 if ``k == 'ge'``.

        """
        return {k: [x.copy() for x in v] for k, v in self._constraints.items()}

    @property
    def num_ancillas(self):
        """num_ancillas.

        Return the number of ancilla variables introduced to the HOIO in
        order to enforce the inputted constraints.

        Returns
        -------
        num : int.
            Number of ancillas in the HOIO.

        """
        return self._ancilla

    @classmethod
    def remove_ancilla_from_solution(cls, solution):
        """remove_ancilla_from_solution.

        Take a solution to the HOIO and remove all the ancilla variables, (
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
        return HOBO.remove_ancilla_from_solution(solution)

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
        return HOBO.is_solution_valid(self, solution)

    # override
    def __round__(self, ndigits=None):
        """round.

        Round values of the HOIO object.

        Parameters
        ----------
        ndigits : int.
            Number of decimal digits to round to.

        Returns
        -------
        res : HOIO object.
            Copy of self but with each value rounded to ``ndigits`` decimal
            digits. Each value has a type according to the docstring
            specifications of ``round``, see ``help(round)``.

        """
        return HOBO.__round__(self, ndigits)

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
        res : a HOIO object.
            Same as ``self`` but with all the symbols replaced with values.

        """
        return HOBO.subs(self, *args, **kwargs)

    def add_constraint_eq_zero(self,
                               H, lam=1,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_eq_zero.

        Enforce that ``H == 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        H : dict representing a HIsing.
            The HIsing constraint such that H == 0. Note that ``H`` will be
            converted to a ``qubovert.HIsing`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.HIsing)``.
            Please note that if ``H`` contains any symbols, then ``bounds``
            must be supplied, since they cannot be determined when symbols
            are present.
        lam : float > 0 or sympy.Symbol (optional, defaults to 1).
            Langrange multiplier to penalize violations of the constraint.
        bounds : two element tuple (optional, defaults to None).
            A tuple ``(min, max)``, the minimum and maximum values that the
            HIsing ``H`` can take. If ``bounds`` is None, then they may be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : HOIO.
            Updates the HOIO in place, but returns ``self`` so that operations
            can be strung together.

        Examples
        --------
        The following enforces that :math:`\sum_{i=1}^{3} i z_i z_{i+1} == 0`.

        >>> H = HOIO()
        >>> H.add_constraint_eq_zero({(1, 2): 1, (2, 3): 2, (3, 4): 3})

        Here we show how operations can be strung together.

        >>> H = HOIO()
        >>> H.add_constraint_lt_zero(
                {(0, 1): 1}
            ).add_constraint_eq_zero(
                {(1, 2): 1, (): -1}
            )

        """
        H = HIsing(H)
        self._constraints.setdefault("eq", []).append(H)
        self += pubo_to_hising(HOBO().add_constraint_eq_zero(
            hising_to_pubo(H), lam=lam,
            bounds=bounds, suppress_warnings=suppress_warnings
        ))
        return self

    def add_constraint_lt_zero(self,
                               H, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_lt_zero.

        Enforce that ``H < 0`` by penalizing invalid solutions with ``lam``.
        See Notes below for more details.

        Parameters
        ----------
        H : dict representing a HIsing.
            The HIsing constraint such that H < 0. Note that ``H`` will be
            converted to a ``qubovert.HIsing`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.HIsing)``.
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
            HIsing ``H`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : HOIO.
            Updates the HOIO in place, but returns ``self`` so that operations
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

          >>> H = HOIO()
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
        H = HIsing(H)
        self._constraints.setdefault("lt", []).append(H)
        self += pubo_to_hising(HOBO().add_constraint_lt_zero(
            hising_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        ))
        return self

    def add_constraint_le_zero(self,
                               H, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_le_zero.

        Enforce that ``H <= 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        H : dict representing a HIsing.
            The HIsing constraint such that H <= 0. Note that ``H`` will be
            converted to a ``qubovert.HIsing`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.HIsing)``.
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
            HIsing ``H`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : HOIO.
            Updates the HOIO in place, but returns ``self`` so that operations
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

          >>> H = HOIO()
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
        H = HIsing(H)
        self._constraints.setdefault("le", []).append(H)
        self += pubo_to_hising(HOBO().add_constraint_le_zero(
            hising_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        ))
        return self

    def add_constraint_gt_zero(self,
                               H, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_gt_zero.

        Enforce that ``H > 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        H : dict representing a HIsing.
            The HIsing constraint such that H > 0. Note that ``H`` will be
            converted to a ``qubovert.HIsing`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.HIsing)``.
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
            HIsing ``H`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : HOIO.
            Updates the HOIO in place, but returns ``self`` so that operations
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

          >>> H = HOIO()
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
        H = HIsing(H)
        self._constraints.setdefault("gt", []).append(H)
        self += pubo_to_hising(HOBO().add_constraint_gt_zero(
            hising_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        ))
        return self

    def add_constraint_ge_zero(self,
                               H, lam=1, log_trick=True,
                               bounds=None, suppress_warnings=False):
        r"""add_constraint_ge_zero.

        Enforce that ``H >= 0`` by penalizing invalid solutions with ``lam``.

        Parameters
        ----------
        H : dict representing a HIsing.
            The HIsing constraint such that H >= 0. Note that ``H`` will be
            converted to a ``qubovert.HIsing`` object if it is not already,
            thus it must follow the conventions, see ``help(qubovert.HIsing)``.
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
            HIsing ``H`` can take. If ``bounds`` is None, then they will be
            calculated (approximately), or if either of the elements of
            ``bounds`` is None, then that element will be calculated
            (approximately).
        suppress_warnings : bool (optional, defaults to False).
            Whether or not to surpress warnings.

        Return
        ------
        self : HOIO.
            Updates the HOIO in place, but returns ``self`` so that operations
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

          >>> H = HOIO()
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
        H = HIsing(H)
        self._constraints.setdefault("ge", []).append(H)
        self += pubo_to_hising(HOBO().add_constraint_ge_zero(
            hising_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        ))
        return self
