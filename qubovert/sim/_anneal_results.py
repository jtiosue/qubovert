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

"""_annea_resultsl.py.

This file contains the objects to deal with anneal results.

"""

from qubovert.utils import spin_to_boolean, boolean_to_spin


__all__ = 'AnnealResult', 'AnnealResults'


class AnnealResult:
    """AnnealResult.

    This class deals with an individual result from an anneal. See the below
    example.

    Example
    -------
    In this example, ``res`` represents a state the gives value ``-2``,
    and is part of a spin model.

    >>> from qubovert.sim import AnnealResult
    >>>
    >>> res = AnnealResult({0: 1, 1: -1, 2: -1}, -2, True)
    >>> print(res.state)
    {0: 1, 1: -1, 2: -1}
    >>> print(res.value)
    -2
    >>> print(res.spin)
    True

    We can convert it to a boolean state with

    >>> boolean_res = res.to_boolean()
    >>> print(boolean_res.state)
    {0: 0, 1: 1, 2: 1}
    >>> print(boolean_res.value)
    -2
    >>> print(boolean_res.spin)
    False

    """

    def __init__(self, state, value, spin):
        """__init__.

        Parameters
        ----------
        state : dict.
            Maps binary labels to their values.
        value : number.
            The value of the model with ``state``.
        spin : bool.
            Indicates whether ``state`` came from a boolean or spin model.

        """
        self.state, self.value, self.spin = state, value, spin

    def __eq__(self, other):
        """__eq__.

        Determine if two AnnealResult objects are equivalent.

        Parameters
        ----------
        other : AnnealResult object.

        Returns
        -------
        res : bool.

        """
        return all((
            self.state == other.state,
            self.value == other.value,
            self.spin == other.spin
        ))

    def copy(self):
        """copy.

        Return a copy of ``self``.

        Returns
        -------
        res : AnnealResult.

        """
        return AnnealResult(self.state.copy(), self.value, self.spin)

    def to_boolean(self):
        """to_boolean.

        Convert the result to a boolean result.

        Returns
        -------
        res : AnnealResult object.
            A boolean version of ``self``. If ``self.spin == False``, then
            ``res`` will be the same as ``self``.

        """
        if not self.spin:
            return self.copy()
        return AnnealResult(spin_to_boolean(self.state), self.value, False)

    def to_spin(self):
        """to_spin.

        Convert the result to a spin result.

        Returns
        -------
        res : AnnealResult object.
            A spin version of ``self``. If ``self.spin == True``, then
            ``res`` will be the same as ``self``.

        """
        if self.spin:
            return self.copy()
        return AnnealResult(boolean_to_spin(self.state), self.value, True)

    def __repr__(self):
        """__repr__.

        Return the string representation of ``self``.

        Return
        ------
        s : str.

        """
        return "  state: %s\n  value: %g" % (self.state, self.value)


class AnnealResults:
    """AnnealResults.

    An object to manage accessing the results of the simulated annealing
    functions.

    Let ``res`` be the output of one of the simulated annealing functions. Then
    a user can access the best result with ``res.best``. The best state will
    be ``res.best.state`` and its corresponding value ``res.best.value``.

    The user can iterate through all of the results with

    >>> for r in res:
    >>>     print(r.state, r.value)

    The user can sort the results from best to worst with
    ``res.sort_by_value()``. Then iterating through ``res`` will be in that
    order.

    The user can also convert each result state to/from boolean and spin
    states.

    """

    def __init__(self, spin):
        """__init__.

        Parameters
        ----------
        spin : bool.
            Indicates whether the results are coming from a boolean or spin
            model.

        """
        self._spin, self._best, self._results = spin, None, []

    def __eq__(self, other):
        """__eq__.

        Determine if ``self`` and ``other`` are equivalent.

        Parameters
        ----------
        other : AnnealResults object.

        Returns
        -------
        res : bool.

        """
        return self._results == other._results

    @property
    def spin(self):
        """spin.

        Return
        ------
        spin : bool.
            Whether ``self`` contains spin or boolean results.

        """
        return self._spin

    @property
    def best(self):
        """best.

        Return the best result.

        Return
        ------
        best : AnealResult object.
            Access the best ``state``  with ``self.best.state``, and its value
            with ``self.best.value``.

        """
        return self._best

    def copy(self):
        """copy.

        Return
        ------
        res : AnnealResults object.
            A deep copy of ``self``.

        """
        res = AnnealResults(self._spin)
        for r in self._results:
            res.add_result(r.copy())
        return res

    def add_state(self, state, value):
        """add_state.

        Add the state to the record.

        Parameters
        ----------
        state : dict.
            Maps variable labels to their values.
        value : number.
            The value of the model with ``state``.

        Returns
        -------
        None.

        """
        self.add_result(AnnealResult(state, value, self._spin))

    def add_result(self, result):
        """add_result.

        Add the result to the record.

        Parameters
        ----------
        result : AnnealResult object.
            See ``help(qubovert.sim.AnnealResult)``.

        Returns
        -------
        None.

        """
        if self._best is None or result.value < self.best.value:
            self._best = result
        self._results.append(result)

    def to_boolean(self):
        """to_boolean.

        Convert each result to a boolean result. Note that if ``self.spin`` is
        False, then this function will just return a copy of ``self``.

        Returns
        -------
        res : AnnealResults object.

        """
        res = AnnealResults(False)
        for r in self._results:
            res.add_result(r.to_boolean())
        return res

    def to_spin(self):
        """to_spin.

        Convert each result to a spin result. Note that if ``self.spin`` is
        True, then this function will just return a copy of ``self``.

        Returns
        -------
        res : AnnealResults object.

        """
        res = AnnealResults(True)
        for r in self._results:
            res.add_result(r.to_spin())
        return res

    def sort_by_value(self):
        """sort_by_value.

        Sort the results in ``self`` in increasing order of their values.

        """
        self._results.sort(key=lambda x: x.value)

    def __iter__(self):
        """__iter__.

        Iterate through the results in whatever order they are in. To change
        the order, see the function ``self.sort_by_value``.

        """
        yield from self._results

    def __str__(self):
        """__str__.

        Return the string representation of ``self``.

        Returns
        -------
        s : str.

        """
        s = "AnnealResults\n"
        for r in self:
            s += str(r) + "\n"
        return s[:-1]
