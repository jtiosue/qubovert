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

"""_simulations.py.

This file contains the ``SpinSimulation`` and ``BooleanSimulation`` objects
which deal with using a Metropolis algorithm to simulate spin and boolean
models. Their parent class in ``BinarySimulation``, which is also in this file.

"""

import qubovert as qv
import random
from numpy import exp


__all__ = 'BinarySimulation', 'SpinSimulation', 'BooleanSimulation'


class BinarySimulation:
    """BinarySimulation.

    The parent class for ``SpinSimulation`` and ``BooleanSimulation``. See
    their docstrings for more information on usage. Child classes MUST
    implement the ``_flip_bit`` method.

    """

    def __init__(self, model, initial_state):
        """__init__.

        Parameters
        ----------
        model : A type in qubovert.BOOLEAN_MODELS or qubovert.SPIN_MODELS.
            The model that we are simulating.
        initial_state : dict.
            The initial state of the system.

        """
        # keep track of the most recent states
        self._past_states = []

        # create a dictionary mapping each bit to the graph that it
        # affects.
        self._subgraphs = {}
        variables = model.variables
        for v in variables:
            self._subgraphs[v] = type(model)(
                {k: c for k, c in model.items() if v in k}
            )

        # make it a list so we can use random.choice later.
        self._variables = list(variables)

        self._initial_state = initial_state
        self.set_state(initial_state)

    @property
    def state(self):
        """state.

        The current state of the system.

        Returns
        -------
        state : dict.
            Dictionary that maps binary labels to their values.

        """
        return self._state.copy()

    def set_state(self, state):
        """set_state.

        Set the state of the system to ``state``.

        Parameters
        ----------
        state : dict or iterable.
            ``state`` maps the binary variable labels to their corresponding
            values. In other words ``state[v]`` is the value of variable ``v``.
            A value must be either 0 or 1 for a boolean system and 1 or -1 for
            a spin system.

        """
        self._state = {v: state[v] for v in self._variables}

    def reset(self):
        """reset.

        Reset the simulation back to its original state.

        """
        self._state, self._past_states = self._initial_state.copy(), []

    def get_past_states(self, num_states=1000):
        """get_past_states.

        Return the previous ``num_states`` states of the system (if that many
        exist; ``self`` only stores up the previous 1000 states).

        Parameters
        ----------
        num_states : int (optional, defaults to 1000).
            The number of previous update steps to include.

        Returns
        -------
        states : list of dicts.
            Each dict maps binary labels to their values.

        """
        return [
            s.copy() for s in self._past_states[-num_states+1:]
        ] + [self.state]

    def _add_past_state(self, state):
        """_add_past_state.

        Add ``state`` to the ``past_states`` memory.

        Parameters
        ----------
        state : dict.
            Maps binary labels to their values.

        """
        self._past_states.append(state)
        if len(self._past_states) > 1000:
            self._past_states.pop(0)

    def _flip_bit(self, bit):
        """_flip_bit.

        Flip the bit in the internal state. This should be implemented by the
        child classes, since it depends on whether or not it is a spin or
        boolean model.

        Parameters
        ----------
        bit : hashable object.
            The label of the bit to flip.

        """
        raise NotImplementedError("To be implemented in the child classes")

    def update(self, T, num_updates=1, seed=None):
        """update.

        Update the simulation at temperature ``T``. Updates the internal state.

        Parameters
        ----------
        T : number >= 0.
            Temperature.
        num_updates : int >= 1 (optional, defaults to 1).
            The number of times to update the simulation at the temperature.
        seed : number (optional, defaults to None).
            The number to seed ``random`` with. If ``seed is None``, then
            ``random.seed`` will not be called.

        """
        if seed is not None:
            random.seed(seed)

        if num_updates < 0:
            raise ValueError("Cannot update a negative number of times")
        elif num_updates > 1:
            for _ in range(num_updates):
                self.update(T)
        elif num_updates == 1:
            self._add_past_state(self.state)
            for _ in range(len(self._variables)):
                i = random.choice(self._variables)
                E = self._subgraphs[i].value(self._state)
                self._flip_bit(i)
                E_flip = self._subgraphs[i].value(self._state)

                dE = E_flip - E
                if not (dE < 0 or (T and random.random() < exp(-dE / T))):
                    # flip the bit back to where it was
                    self._flip_bit(i)

    def schedule_update(self, schedule, seed=None):
        """schedule_update.

        Update the simulation with a schedule.

        Parameters
        ----------
        schedule : iteranle of tuples.
            Each element in ``schedule`` is a pair ``(T, n)`` which designates
            a temperature and a number of updates. See `Notes` below.
        seed : number (optional, defaults to None).
            The number to seed ``random`` with. If ``seed is None``, then
            ``random.seed`` will not be called.

        Notes
        -----
        The following two code blocks perform exactly the same thing.

        >>> sim = BooleanSimulation(10)
        >>> for T in (3, 2):
        >>>     sim.update(T, 100)
        >>> sim.update(1, 50)

        >>> sim = BooleanSimulation(10)
        >>> schedule = (3, 100), (2, 100), (1, 50)
        >>> sim.schedule_update(schedule)

        """
        if seed is not None:
            random.seed(seed)
        for T, n in schedule:
            self.update(T, n)


class SpinSimulation(BinarySimulation):
    """SpinSimulation.

    ``SpinSimulation`` uses a Metropolis algorithm to simulate a spin system.
    The spin system can be "updated" at a temperature ``T``. Thus, we can get
    an idea of the time evolution of a spin system by creating a temperature
    schedule and seeing how the system evolves.

    ``SpinSimulation`` inherits from ``BinarySimulation``. See
    ``help(qubovert.sim.BinarySimulation)`` for more details.

    Examples
    --------
    Consider the example of the ferromagnetic chain.

    >>> import qubovert as qv
    >>>
    >>> length = 50
    >>> spin_system = sum(
    >>>     -qv.spin_var(i) * qv.spin_var(i+1) for i in range(length)
    >>> )
    >>>
    >>> # initial state is all spin down
    >>> initial_state = {i: -1 for i in range(length)}
    >>> sim = qv.sim.SpinSimulation(spin_system, initial_state)
    >>>
    >>> # define a schedule. here we simulate at temperature 4 for 25 time
    >>> # steps, then temperature 2 for 25 time steps, then temperature 1 for
    >>> # 10 time steps.
    >>> schedule = (4, 25), (2, 25), (1, 10)
    >>> sim.schedule_update(schedule)
    >>>
    >>> print("final state", sim.state)
    >>> print("last 30 states", sim.get_past_states(30))

    See Also
    --------
    - ``qubovert.sim.BooleanSimulation``.
    - ``qubovert.sim.BinarySimulation``.

    """

    def __init__(self, model, initial_state=None):
        """__init__.

        Parameters
        ----------
        model : dict or type in ``qubovert.SPIN_MODELS``.
            The model the simulate.
        initial_state : dict (optional, defaults to None).
            The initial state to start the simulation in. ``initial_state``
            should map spin label names to their initial values, where each
            value is either 1 or -1. If ``initial_state`` is None, then it
            will be initialized to all 1s.

        """
        if not isinstance(model, qv.SPIN_MODELS):
            model = qv.PUSO(model)

        variables = model.variables

        if initial_state is None:
            initial_state = {v: 1 for v in variables}

        super().__init__(model, initial_state)

    def _flip_bit(self, bit):
        """_flip_bit.

        Flip the spin labeled by ``bit`` in the internal state.

        Parameters
        ----------
        bit : hashable object.
            The label of the spin to flip.

        """
        self._state[bit] *= -1

    def set_state(self, state):
        """set_state.

        Set the state of the spin system to ``state``.

        Parameters
        ----------
        state : dict or iterable.
            ``state`` maps the spin variable labels to their corresponding
            values. In other words ``state[v]`` is the value of variable ``v``.
            A value must be either 1 or -1.

        """
        super().set_state(state)
        if any(v not in {1, -1} for v in self._state.values()):
            raise ValueError("State must contain only 1's and -1's")


class BooleanSimulation(BinarySimulation):
    """BooleanSimulation.

    ``BooleanSimulation`` uses a Metropolis algorithm to simulate a boolean
    systems. The boolean system can be "updated" at a temperature ``T``. Thus,
    we can get an idea of the time evolution of a boolean system by creating a
    temperature schedule and seeing how the system evolves.

    ``BooleanSimulation`` inherits from ``BinarySimulation``. See
    ``help(qubovert.sim.BinarySimulation)`` for more details.

    Examples
    --------
    Consider the following example where we minimize an objective function.

    >>> import qubovert as qv
    >>>
    >>> # create the objective function.
    >>> x = [qv.boolean_var(i) for i in range(10)]
    >>> model = sum(x)
    >>> model.add_constraint_le_zero(x[0] + x[2] - 3 * x[5] - 1, lam=3)
    >>>
    >>> # initial state is all variables equal to 1
    >>> initial_state = {i: 11 for i in range(length)}
    >>> sim = qv.sim.BooleanSimulation(model, initial_state)
    >>>
    >>> # define a schedule. here we simulate at temperature 4 for 25 time
    >>> # steps, then temperature 2 for 25 time steps, then temperature 1 for
    >>> # 10 time steps.
    >>> schedule = (4, 25), (2, 25), (1, 10)
    >>> sim.schedule_update(schedule)
    >>>
    >>> print("final state", sim.state)
    >>> print("last 30 states", sim.get_past_states(30))

    See Also
    --------
    - ``qubovert.sim.SpinSimulation``.
    - ``qubovert.sim.BinarySimulation``.

    """

    def __init__(self, model, initial_state=None):
        """__init__.

        Parameters
        ----------
        model : dict or type in ``qubovert.BOOLEAN_MODELS``.
            The model the simulate.
        initial_state : dict (optional, defaults to None).
            The initial state to start the simulation in. ``initial_state``
            should map boolean label names to their initial values, where each
            value is either 0 or 1. If ``initial_state`` is None, then it
            will be initialized to all 0s.

        """
        if not isinstance(model, qv.BOOLEAN_MODELS):
            model = qv.PUBO(model)

        variables = model.variables

        if initial_state is None:
            initial_state = {v: 0 for v in variables}

        super().__init__(model, initial_state)

    def _flip_bit(self, bit):
        """_flip_bit.

        Flip the boolean labeled by ``bit`` in the internal state.

        Parameters
        ----------
        bit : hashable object.
            The label of the boolean to flip.

        """
        self._state[bit] = 1 - self._state[bit]

    def set_state(self, state):
        """set_state.

        Set the state of the boolean system to ``state``.

        Parameters
        ----------
        state : dict or iterable.
            ``state`` maps the boolean variable labels to their corresponding
            values. In other words ``state[v]`` is the value of variable ``v``.
            A value must be either 0 or 1.

        """
        super().set_state(state)
        if any(v not in {0, 1} for v in self._state.values()):
            raise ValueError("State must contain only 0's and 1's")
