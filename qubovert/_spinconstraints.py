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

"""_spinconstraints.py.

Contains the SpinConstraints class. See ``help(qubovert.SpinConstraints)``.

"""

from . import PUSO, BooleanConstraints
from .utils import puso_to_pubo, pubo_to_puso


__all__ = 'SpinConstraints',


def _empty_booleanconstraints(spinconstraints):
    """_empty_booleanconstraints.

    Create an empty BooleanConstraints object whose ancilla variables begin at
    ``spinconstraints.num_ancillas``.

    Parameters
    ----------
    spinconstraints : SpinConstraints object.

    Return
    ------
    booleanconstraints : BooleanConstraints object.

    """
    h = BooleanConstraints()
    h._ancilla = spinconstraints._ancilla
    return h


class SpinConstraints:
    """SpinConstraints.

    This class deals spin constraints.

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

    Examples
    --------
    >>> H = SpinConstraints()
    >>> H.add_constraint_eq_zero({('a', 1): 2, (1, 2): -1, (): -1})
    >>> H.to_penalty()
    {(): 6, ('a', 2): -4, ('a', 1): -4, (1, 2): 2}

    >>> H = SpinConstraints()
    >>> H.add_constraint_eq_zero(
            {(0, 1): 1}
        ).add_constraint_lt_zero(
            {(1, 2): 1, (): -1}
        )

    """

    def __init__(self):
        """__init__.

        Initialize a SpinConstraints object.

        """
        self._constraints, self._penalty, self._ancilla = {}, PUSO(), 0

    __repr__ = BooleanConstraints.__repr__

    __str__ = BooleanConstraints.__str__

    __iter__ = BooleanConstraints.__iter__

    num_ancillas = BooleanConstraints.num_ancillas

    _next_ancilla = BooleanConstraints._next_ancilla

    remove_ancilla_from_solution = (
        BooleanConstraints.remove_ancilla_from_solution
    )

    constraints = BooleanConstraints.constraints

    _append_constraint = BooleanConstraints._append_constraint

    _pop_constraint = BooleanConstraints._pop_constraint

    is_solution_valid = BooleanConstraints.is_solution_valid

    copy = BooleanConstraints.copy

    __round__ = BooleanConstraints.__round__

    subs = BooleanConstraints.subs

    to_penalty = BooleanConstraints.to_penalty

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
        self : SpinConstraints.
            Updates the SpinConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Examples
        --------
        The following enforces that :math:`\sum_{i=1}^{3} i z_i z_{i+1} == 0`.

        >>> H = SpinConstraints()
        >>> H.add_constraint_eq_zero({(1, 2): 1, (2, 3): 2, (3, 4): 3})

        Here we show how operations can be strung together.

        >>> H = SpinConstraints()
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

        h = _empty_booleanconstraints(self).add_constraint_eq_zero(
            puso_to_pubo(H), lam=lam,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self._penalty += pubo_to_puso(h._penalty)
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
        self : SpinConstraints.
            Updates the SpinConstraints in place, but returns ``self`` so
            that operations can be strung together.

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
        h = _empty_booleanconstraints(self).add_constraint_ne_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self._penalty += pubo_to_puso(h._penalty)
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
        self : SpinConstraints.
            Updates the SpinConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`z_0 + 2z_1 + ... < 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = SpinConstraints()
          >>> H.add_constraint_lt_zero(
          >>>     {(0,): 0.5, (): 1.65, (1,): 1.0, (2,): -0.25})
          >>> test_sol = {0: -1, 1: -1, 2: 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.to_penalty().value(test_sol)
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
        h = _empty_booleanconstraints(self).add_constraint_lt_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self._penalty += pubo_to_puso(h._penalty)
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
        self : SpinConstraints.
            Updates the SpinConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`z_0 + 2z_1 + ... \leq 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = SpinConstraints()
          >>> H.add_constraint_le_zero(
                  {(0,): 0.5, (): 1.15, (1,): 1.0, (2,): -0.75})
          >>> test_sol = {0: -1, 1: -1, 2: 1, '__a0': 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.to_penalty().value(test_sol)
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
        h = _empty_booleanconstraints(self).add_constraint_le_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self._penalty += pubo_to_puso(h._penalty)
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
        self : SpinConstraints.
            Updates the SpinConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`z_0 + 2z_1 + ... > 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = SpinConstraints()
          >>> H.add_constraint_gt_zero(
                  {(0,): -0.5, (): -1.65, (1,): -1.0, (2,): 0.25})
          >>> test_sol = {0: -1, 1: -1, 2: 1}
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.to_penalty().value(sol)
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
        h = _empty_booleanconstraints(self).add_constraint_gt_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self._penalty += pubo_to_puso(h._penalty)
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
        self : SpinConstraints.
            Updates the SpinConstraints in place, but returns ``self`` so
            that operations can be strung together.

        Notes
        -----
        - There is no general way to enforce non integer inequality
          constraints. Thus this function is only guarenteed to work for
          integer inequality constraints (ie constraints of the form
          :math:`z_0 + 2z_1 + ... \geq 0`). However, it can be used for non
          integer inequality constraints, but it is recommended that the value
          of ``lam`` be set small, since valid solutions may still recieve a
          penalty to the objective function. For example,

          >>> H = SpinConstraints()
          >>> H.add_constraint_ge_zero(
                  {(0,): -0.5, (): -1.15, (1,): -1.0, (2,): 0.75})
          >>> H.is_solution_valid(test_sol)
          True
          >>> H.to_penalty().value(test_sol)
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
        h = _empty_booleanconstraints(self).add_constraint_ge_zero(
            puso_to_pubo(H), lam=lam, log_trick=log_trick,
            bounds=bounds, suppress_warnings=suppress_warnings
        )
        self._ancilla = h._ancilla
        self._penalty += pubo_to_puso(h._penalty)
        return self
