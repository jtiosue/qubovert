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

"""_anneal_results.py.

This file contains the objects to deal with anneal results.

"""

from qubovert.utils import spin_to_boolean, boolean_to_spin


__all__ = 'AnnealResult', 'AnnealResults'


# helper for AnnealResults

def _recompute_best(results):
    """_recompute_best.

    Internal helper function for the AnnealResults class. Computes the best
    AnnealResult in an AnnealResults object.

    Parameters
    ----------
    results : AnnealResults object.

    Returns
    -------
    res : AnnealResult object.
        The AnnealResult from ``results`` with the lowest value.

    """
    best = None
    for r in results:
        if best is None or r.value < best.value:
            best = r
    return best


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

    def __lt__(self, other):
        """__lt__.

        ``self < other`` iff ``self.value < other.value``.

        Parameters
        ----------
        other : AnnealResult object.

        Returns
        -------
        res : bool.

        """
        return self.value < other.value

    def __le__(self, other):
        """__le__.

        ``self <= other`` iff ``self.value <= other.value``.

        Parameters
        ----------
        other : AnnealResult object.

        Returns
        -------
        res : bool.

        """
        return self.value <= other.value

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

    def __str__(self):
        """__str__.

        Return the string representation of ``self``.

        Return
        ------
        s : str.

        """
        return "  state: %s\n  value: %g\n   spin: %s" % (
            self.state, self.value, self.spin
        )

    def __repr__(self):
        """__repr__.

        Create the representation of ``self``.

        Return
        ------
        r : str.

        """
        return "AnnealResult(state=%s, value=%g, spin=%s)" % (
            self.state, self.value, self.spin
        )


class AnnealResults(list):
    """AnnealResults.

    An object to manage accessing the results of the simulated annealing
    functions. ``AnnealResults`` is a subclass of ``list``.

    Let ``res`` be the output of one of the simulated annealing functions. Then
    a user can access the best result with ``res.best``. The best state will
    be ``res.best.state`` and its corresponding value ``res.best.value``.

    The user can iterate through all of the results with

    >>> for r in res:
    >>>     print(r.state, r.value)

    The user can sort the results from best to worst with
    ``res.sort()``. Then iterating through ``res`` will be in that
    order.

    The user can also convert each result state to/from boolean and spin
    states.

    """

    def __init__(self, iterable=()):
        """__init__.

        Parameters
        ----------
        iterable : any iterable (optional, defaults to ``()``).
            The iterable should contain only ``qubovert.sim.AnnealResult``
            objects.

        """
        super().__init__()
        self.best = None
        for x in iterable:
            self.append(x)

    def copy(self):
        """copy.

        Return
        ------
        res : AnnealResults object.
            A deep copy of ``self``.

        """
        return AnnealResults(self)

    def add_state(self, state, value, spin):
        """add_state.

        Add the state to the record.

        Parameters
        ----------
        state : dict.
            Maps variable labels to their values.
        value : number.
            The value of the model with ``state``.
        spin : bool.
            Whether it is from a spin or boolean model.

        Returns
        -------
        None.

        """
        self.append(AnnealResult(state, value, spin))

    def append(self, result):
        """append.

        Add the result to the record at the end and update the
        ``best`` attribute if needed.

        Parameters
        ----------
        result : AnnealResult object.
            See ``help(qubovert.sim.AnnealResult)``.

        Returns
        -------
        None.

        """
        if self.best is None or result.value < self.best.value:
            self.best = result
        super().append(result)

    def insert(self, index, result):
        """insert.

        Add the result to the record at index ``index`` and update the
        ``best`` attribute if needed.

        Parameters
        ----------
        result : AnnealResult object.
            See ``help(qubovert.sim.AnnealResult)``.

        Returns
        -------
        None.

        """
        if self.best is None or result.value < self.best.value:
            self.best = result
        super().insert(index, result)

    def remove(self, result):
        """remove.

        Override ``list.remove`` so we also update the ``best`` attribute.
        Remove the first occurrence of ``result``, or raise ValueError.

        Parameters
        ----------
        result : AnnealResult object.
            See ``help(qubovert.sim.AnnealResult)``.

        Returns
        -------
        None

        """
        super().remove(result)
        if result == self.best:
            self.best = _recompute_best(self)

    def pop(self, index):
        """pop.

        Override ``list.pop`` so we also update the ``best`` attribute.
        Remove and return the element in the ``index`` index, or raise
        an IndexError.

        Parameters
        ----------
        index : int.

        Returns
        -------
        res : AnnealResult object.
            The element at ``index``.

        """
        res = super().pop(index)
        if res == self.best:
            self.best = _recompute_best(self)
        return res

    def to_boolean(self):
        """to_boolean.

        Convert each result to a boolean result. Note that if ``self.spin`` is
        False, then this function will just return a copy of ``self``.

        Returns
        -------
        res : AnnealResults object.

        """
        return AnnealResults(r.to_boolean() for r in self)

    def to_spin(self):
        """to_spin.

        Convert each result to a spin result. Note that if ``self.spin`` is
        True, then this function will just return a copy of ``self``.

        Returns
        -------
        res : AnnealResults object.

        """
        return AnnealResults(r.to_spin() for r in self)

    def __str__(self):
        """__str__.

        Return the string representation of ``self``.

        Returns
        -------
        s : str.

        """
        return "AnnealResults\n\n" + "\n\n".join(str(x) for x in self)

    def __getitem__(self, index):
        """__getitem__.

        Override ``list.__getitem__`` so that when slicing we return a
        ``qubovert.sim.AnnealResults`` item instead of a list.

        Parameters
        ----------
        index : int or slice object.

        Returns
        -------
        res : qubovert.sim.AnnealResults object.

        """
        res = super().__getitem__(index)
        if isinstance(index, slice):
            res = AnnealResults(res)
        return res

    def clear(self):
        """clear.

        Override ``list.clear`` so that it also removes ``self.best``.

        """
        self.best = None
        super().clear()

    def filter(self, func):
        """filter.

        Return a new AnnealResults object whose elements are filtered by the
        function ``func``. ``func`` takes in a ``qubovert.sim.AnnealResult``
        object and returns a boolean indicating whether it should remain
        in the filtered results.

        Parameters
        ----------
        func : function.
            ``func`` takes in a ``qubovert.sim.AnnealResult`` object and
            returns a boolean indicating whether it should remain in the
            filtered results.

        Returns
        -------
        res : qubovert.sim.AnnealResults object.

        Example
        -------
        >>> import qubovert as qv
        >>>
        >>> model = qv.boolean_var(0) * qv.boolean_var(1)
        >>> anneal_res = qv.sim.anneal_qubo(model, num_anneals=3)
        >>>
        >>> anneal_res
        [AnnealResult(state={0: 0, 1: 1}, value=0, spin=False),
         AnnealResult(state={0: 1, 1: 0}, value=0, spin=False),
         AnnealResult(state={0: 0, 1: 0}, value=0, spin=False)]
        >>> filtered_anneal_res = anneal_res.filter(
        >>>     lambda x: x.state[0] == 0
        >>> )
        >>> filtered_anneal_res
        [AnnealResult(state={0: 0, 1: 1}, value=0, spin=False),
         AnnealResult(state={0: 0, 1: 0}, value=0, spin=False)]

        """
        return AnnealResults(filter(func, self))

    def filter_states(self, func):
        """filter_states.

        Return a new AnnealResults object whose states are filtered by the
        function ``func``. ``func`` takes in a ``dict`` representing a state
        and returns a boolean indicating whether it should remain in the
        filtered results.

        Parameters
        ----------
        func : function.
            ``func`` takes in a ``dict`` representing a state and
            returns a boolean indicating whether it should remain in the
            filtered results.

        Returns
        -------
        res : qubovert.sim.AnnealResults object.

        Example
        -------
        >>> import qubovert as qv
        >>>
        >>> model = qv.boolean_var(0) * qv.boolean_var(1)
        >>> anneal_res = qv.sim.anneal_qubo(model, num_anneals=3)
        >>>
        >>> anneal_res
        [AnnealResult(state={0: 0, 1: 1}, value=0, spin=False),
         AnnealResult(state={0: 1, 1: 0}, value=0, spin=False),
         AnnealResult(state={0: 0, 1: 0}, value=0, spin=False)]
        >>> filtered_anneal_res = anneal_res.filter_states(
        >>>     lambda x: x[0] == 0
        >>> )
        >>> filtered_anneal_res
        [AnnealResult(state={0: 0, 1: 1}, value=0, spin=False),
         AnnealResult(state={0: 0, 1: 0}, value=0, spin=False)]

        """
        return AnnealResults(filter(lambda x: func(x.state), self))

    def apply_function(self, func):
        """apply_function.

        Apply the function ``func`` to each element in ``self`` to create a
        new version of ``self``.

        Parameters
        ----------
        func : function.
            ``func`` takes in a ``qubovert.sim.AnnealResult`` object and
            returns a ``qubovert.sim.AnnealResult`` object.

        Returns
        -------
        res : qubovert.sim.AnnealResults object.

        Example
        -------
        >>> import qubovert as qv
        >>>
        >>> model = qv.boolean_var('a') * qv.boolean_var('b')
        >>> qubo = model.to_qubo()
        >>> anneal_res = qv.sim.anneal_qubo(qubo, num_anneals=3)
        >>> anneal_res
        [AnnealResult(state={0: 0, 1: 1}, value=0, spin=False),
         AnnealResult(state={0: 0, 1: 0}, value=0, spin=False),
         AnnealResult(state={0: 0, 1: 0}, value=0, spin=False)]
        >>> new_res = anneal_res.apply_function(
                lambda x: qv.sim.AnnealResult(
                    model.convert_solution(x.state), x.value, x.spin
                )
            )
        >>> new_res
        [AnnealResult(state={'a': 0, 'b': 1}, value=0, spin=False),
         AnnealResult(state={'a': 0, 'b': 0}, value=0, spin=False),
         AnnealResult(state={'a': 0, 'b': 0}, value=0, spin=False)]

        """
        return AnnealResults(func(x) for x in self)

    def convert_states(self, func):
        """convert_states.

        Apply the function ``func`` to each state in ``self`` to create a
        new version of ``self``.

        Parameters
        ----------
        func : function.
            ``func`` takes in a dict that maps variable names to their values,
            and returns a dict.

        Returns
        -------
        res : qubovert.sim.AnnealResults object.

        Example
        -------
        >>> import qubovert as qv
        >>>
        >>> model = qv.boolean_var('a') * qv.boolean_var('b')
        >>> qubo = model.to_qubo()
        >>> anneal_res = qv.sim.anneal_qubo(qubo, num_anneals=3)
        >>> anneal_res
        [AnnealResult(state={0: 0, 1: 1}, value=0, spin=False),
         AnnealResult(state={0: 0, 1: 0}, value=0, spin=False),
         AnnealResult(state={0: 0, 1: 0}, value=0, spin=False)]
        >>> new_res = anneal_res.convert_states(model.convert_solution)
        >>> new_res
        [AnnealResult(state={'a': 0, 'b': 1}, value=0, spin=False),
         AnnealResult(state={'a': 0, 'b': 0}, value=0, spin=False),
         AnnealResult(state={'a': 0, 'b': 0}, value=0, spin=False)]

        """
        return self.apply_function(
            lambda x: AnnealResult(func(x.state), x.value, x.spin)
        )

    def __add__(self, other):
        """__add__.

        Override ``list.__add__`` to return a ``AnnealResults`` object.

        Parameters
        ----------
        other : qubovert.sim.AnnealResults object.

        Returns
        -------
        res : qubovert.sim.AnnealResults object.

        """
        return AnnealResults(super().__add__(other))

    def __iadd__(self, other):
        """__iadd__.

        Override ``list.__iadd__`` to keep track of ``best`` attribute.

        Parameters
        ----------
        other : qubovert.sim.AnnealResults object.

        Returns
        -------
        self : qubovert.sim.AnnealResults object.

        """
        if isinstance(other, AnnealResults):
            if other.best < self.best:
                self.best = other.best
            return super().__iadd__(other)

        self.extend(other)
        return self

    def extend(self, other):
        """extend.

        Override ``list.extend`` to keep track of ``best`` attribute.
        Updates in place.

        Parameters
        ----------
        other : qubovert.sim.AnnealResults object or iterable.

        """
        if isinstance(other, AnnealResults):
            if other.best < self.best:
                self.best = other.best
            super().extend(other)
        else:
            for x in other:
                self.append(x)

    def __mul__(self, other):
        """__mul__.

        Override ``list.__mul__`` to return a ``AnnealResults`` object.

        Parameters
        ----------
        other : qubovert.sim.AnnealResults object.

        Returns
        -------
        res : qubovert.sim.AnnealResults object.

        """
        return AnnealResults(super().__mul__(other))
