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

"""_qubo_simulation.py.

This file contains the ``QUBOSimulation`` object, which deals with using a
Metropolis algorithm to simulate QUBOs.

"""

from qubovert.utils import (
    qubo_to_quso, spin_to_boolean, boolean_to_spin, QUSOMatrix
)
from . import QUSOSimulation


__all__ = 'QUBOSimulation',


class QUBOSimulation(QUSOSimulation):
    """QUBOSimulation.

    ``QUBOSimulation`` uses a Metropolis algorithm to simulate a QUBO.
    The QUBO can be "updated" at a temperature ``T``. Thus,
    we can get an idea of the time evolution of a QUBO by creating a
    temperature schedule and seeing how the system evolves.

    ``QUBOSimulation`` inherits from ``QUSOSimulation``. In fact,
    ``QUBOSimulation`` just deals internally with converting to and from
    a spin system; all the simulation is done with ``QUSOSimulation``. See
    ``help(qubovert.sim.QUSOSimulation)`` for more details.

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
    >>> sim = qv.sim.QUBOSimulation(model, initial_state)
    >>>
    >>> # define a schedule. here we simulate at temperature 4 for 25 time
    >>> # steps, then temperature 2 for 25 time steps, then temperature 1 for
    >>> # 10 time steps.
    >>> schedule = (4, 25), (2, 25), (1, 10)
    >>> sim.schedule_update(schedule)
    >>>
    >>> print("final state", sim.state)

    See Also
    --------
    ``qv.sim.PUSOSimulation``, ``qv.sim.QUSOSimulation``,
    ``qv.sim.PUBOSimulation``.

    """

    def __init__(self, Q, initial_state=None):
        """__init__.

        Parameters
        ----------
        Q : dict, ``qubovert.utils.QUBOMatrix``, or ``qubovert.QUBO`` object.
            The QUBO to simulate. This should map tuples of boolean variable
            labels to their respective coefficient in the objective function.
            For more information, see the docstrings for
            ``qubovert.utils.QUBOMatrix`` and ``qubovert.QUBO``.
        initial_state : dict (optional, defaults to None).
            The initial state to start the simulation in. ``initial_state``
            should map boolean label names to their initial values, where each
            value is either 0 or 1. If ``initial_state`` is None, then it
            will be initialized to all 0s.

        """
        model = qubo_to_quso(Q)
        if initial_state is None:
            var = (
                range(model.max_index + 1)
                if type(model) == QUSOMatrix
                else model._variables
            )
            initial_state = {v: 0 for v in var}
        super().__init__(model, initial_state)

    @property
    def state(self):
        """state.

        A copy of the current state of the system.

        Returns
        -------
        state : dict.
            Dictionary that maps boolean labels to their values in {0, 1}.

        """
        return spin_to_boolean(super().state)

    def set_state(self, state):
        """set_state.

        Set the state of the spin system to ``state``.

        Parameters
        ----------
        state : dict or iterable.
            ``state`` maps the spin variable labels to their corresponding
            values. In other words ``state[v]`` is the value of variable ``v``.
            A value must be either 0 or 1.

        """
        # if we call super then we get the wrong errors
        state = {v: state[v] for v in self._variables}
        if any(v not in {0, 1} for v in state.values()):
            raise ValueError("State must contain only 0's and 1's")
        super().set_state(boolean_to_spin(state))
