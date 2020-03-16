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

"""_quso_simulation.py.

This file contains the ``QUSOSimulation`` object, which deals with using a
Metropolis algorithm to simulate QUSOs. The ``QUSOSimulation`` object deals
with interfacing with the C code for simulating QUSOs.

"""

from ._simulate_quso import c_simulate_quso as simulate_quso
from itertools import chain
from qubovert import QUSO
from qubovert.utils import QUSOMatrix


__all__ = 'QUSOSimulation',


class QUSOSimulation:
    """QUSOSimulation.

    ``QUSOSimulation`` uses a Metropolis algorithm implemnted in C to simulate
    a QUSO. The QUSO can be "updated" at a temperature ``T``. Thus, we can get
    an idea of the time evolution of a QUSO by creating a temperature
    schedule and seeing how the system evolves.

    Examples
    --------
    Consider the example of the ferromagnetic chain.

    >>> import qubovert as qv
    >>>
    >>> length = 50
    >>> spin_system = sum(
    >>>     -qv.spin_var(i) * qv.spin_var(i+1) * qv.spin_var(i+2)
    >>>     for i in range(length-2)
    >>> )
    >>>
    >>> # initial state is all spin down
    >>> initial_state = {i: -1 for i in range(length)}
    >>> sim = qv.sim.QUSOSimulation(spin_system, initial_state)
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
    ``qv.sim.PUBOSimulation``, ``qv.sim.PUSOSimulation``,
    ``qv.sim.QUBOSimulation``.

    """

    def __init__(self, L, initial_state=None):
        """__init__.

        Parameters
        ----------
        L : dict, ``qubovert.utils.QUSOMatrix``, or ``qubovert.QUSO`` object.
            The QUSO to simulate. This should map tuples of spin variable
            labels to their respective coefficient in the Hamiltonian.
            For more information, see the docstrings for
            ``qubovert.utils.QUSOMatrix`` and ``qubovert.QUSO``.
        initial_state : dict (optional, defaults to None).
            The initial state to start the simulation in. ``initial_state``
            should map spin label names to their initial values, where each
            value is either 1 or -1. If ``initial_state`` is None, then it
            will be initialized to all 1s.

        """
        # must use type since we don't want errors from inheritance
        if type(L) == QUSOMatrix:
            N = L.max_index + 1
            model = L
            self._mapping = dict(enumerate(range(N)))
            self._reverse_mapping = self._mapping
            self._variables = set(self._mapping.keys())
        elif type(L) != QUSO:
            L = QUSO(L)

        if type(L) == QUSO:
            N = L.num_binary_variables
            model = L.to_quso()
            self._mapping = L.mapping
            self._reverse_mapping = L.reverse_mapping
            self._variables = L.variables

        self._initial_state = (
            initial_state.copy() if initial_state is not None else
            {v: 1 for v in self._variables}
        )
        self._state = [1] * N
        self.set_state(self._initial_state)

        # C arguments
        # create model arrays
        h, num_neighbors = [0.] * N, [0] * N
        neighbors, J = [[] for _ in range(N)], [[] for _ in range(N)]

        for k, v in model.items():
            val = float(v)
            if len(k) == 1:
                h[k[0]] = val
            elif len(k) == 2:
                i, j = k
                neighbors[i].append(j)
                neighbors[j].append(i)
                num_neighbors[i] += 1
                num_neighbors[j] += 1
                J[i].append(val)
                J[j].append(val)
            # ignore offset.
        # flatten the arrays.
        J, neighbors = list(chain(*J)), list(chain(*neighbors))

        self._c_args = h, num_neighbors, neighbors, J

    def __str__(self):
        """__str__.

        Return
        ------
        s : str.

        """
        return self.__class__.__name__

    @property
    def state(self):
        """state.

        A copy of the current state of the system.

        Returns
        -------
        state : dict.
            Dictionary that maps spin labels to their values in {1, -1}.

        """
        return {self._reverse_mapping[k]: v for k, v in enumerate(self._state)}

    @property
    def initial_state(self):
        """initial_state.

        A copy of the initial state of the system.

        Returns
        -------
        initial_state : dict.
            Dictionary that maps binary labels to their values.

        """
        return self._initial_state.copy()

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
        for v in self._variables:
            self._state[self._mapping[v]] = state[v]
        if any(v not in {1, -1} for v in self._state):
            raise ValueError("State must contain only 1's and -1's")

    def reset(self):
        """reset.

        Reset the simulation back to its original state.

        """
        self.set_state(self._initial_state)

    def update(self, T, num_updates=1, in_order=False, seed=None):
        """update.

        Update the simulation at temperature ``T``. Updates the internal state.

        Parameters
        ----------
        T : number >= 0.
            Temperature.
        num_updates : int >= 1 (optional, defaults to 1).
            The number of times to update the simulation at the temperature.
        in_order : bool (optional, defaults to False).
            Whether to iterate through the variables in order or randomly
            during an update step. When ``in_order`` is False, the simulation
            is more physically realistic, but when using the Simulation for
            annealing, often it is better to have ``in_order = True``.
        seed : number (optional, defaults to None).
            The number to seed ``random`` with.

        """
        self.schedule_update([(T, num_updates)], in_order, seed)

    def schedule_update(self, schedule, in_order=False, seed=None):
        """schedule_update.

        Update the simulation with a schedule. This function wraps the C
        implementation of the simulation.

        Parameters
        ----------
        schedule : iterable of tuples.
            Each element in ``schedule`` is a pair ``(T, n)`` which designates
            a temperature and a number of updates. See `Notes` below.
        in_order : bool (optional, defaults to False).
            Whether to iterate through the variables in order or randomly
            during an update step. When ``in_order`` is False, the simulation
            is more physically realistic, but when using the Simulation for
            annealing, often it is better to have ``in_order = True``.
        seed : number (optional, defaults to None).
            The number to seed ``random`` with.

        Notes
        -----
        The following two code blocks perform exactly the same thing, although
        the second code block will be faster.

        >>> sim = QUSOSimulation(10)
        >>> for T in (3, 2):
        >>>     sim.update(T, 100)
        >>> sim.update(1, 50)

        >>> sim = QUSOSimulation(10)
        >>> schedule = (3, 100), (2, 100), (1, 50)
        >>> sim.schedule_update(schedule)

        """
        # call the C function
        self._state = simulate_quso(
            self._state, *self._c_args,
            *zip(*schedule), int(in_order),
            seed if seed is not None else -1
        )
