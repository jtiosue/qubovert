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

"""_anneal.py.

This file uses the simulation functionality to implement a simulated annealing
algorithm for both boolean and spin models.

"""

from qubovert.utils import (
    pubo_to_puso, qubo_to_quso, QUBOVertWarning, boolean_to_spin,
    QUSOMatrix, PUSOMatrix
)
from qubovert import QUSO, PUSO, PCSO
from . import anneal_temperature_range, AnnealResults, AnnealResult
import numpy as np
from itertools import chain
from ._canneal import c_anneal_quso, c_anneal_puso


__all__ = (
    'anneal_qubo', 'anneal_quso', 'anneal_pubo', 'anneal_puso',
    'SCHEDULES'
)

SCHEDULES = 'linear', 'geometric'


# helpers

def _create_spin_schedule(spin_model, anneal_duration,
                          temperature_range, schedule):
    """_create_spin_schedule.

    Internal function to create the temperature schedule from the input
    parameters.

    Parameters
    ----------
    spin_model : dict or any type in ``qubovert.SPIN_MODELS``.
        Maps spin labels to their values in the objective function.
    anneal_duration : int >= 1 (optional, defaults to 1000).
        The total number of updates to the simulation during the anneal.
        This is related to the amount of time we spend in the cooling schedule.
    temperature_range : tuple (optional, defaults to None).
        The temperature to start and end the anneal at.
        ``temperature = (T0, Tf)``. ``T0`` must be >= ``Tf``. To see more
        details on picking a temperature range, please see the function
        ``qubovert.sim.anneal_temperature_range``. If ``temperature_range`` is
        None, then it will by default be set to
        ``T0, Tf = qubovert.sim.anneal_temperature_range(spin_model)``.
    schedule : str, or list of floats (optional, defaults to ``'geometric'``).
        What type of cooling schedule to use. If ``schedule == 'linear'``,
        then the cooling schedule will be a linear interpolation between the
        values in ``temperature_range``. If ``schedule == 'geometric'``, then
        the cooling schedule will be a geometric interpolation between the
        values in ``temperature_range``. Otherwise, ``schedule`` must be an
        iterable of floats being the explicit temperature schedule for the
        anneal to follow.

    Returns
    -------
    Ts : list of floats.
        The explicit schedule of temperatures to update at each time step.

    Raises
    ------
    ValueError
        If the ``schedule`` argument provided is formatted incorrectly. See the
        Parameters section.
    ValueError
        If the initial temperature is less than the final temperature.

    Warns
    -----
    qubovert.utils.QUBOVertWarning
        If an explicit ``schedule`` is provided (ie ``schedule`` is an
        iterable of floats) and a ``temperature_range`` is provided. The
        ``temperature_range`` will be ignored.

    """
    if not isinstance(schedule, str):
        if temperature_range is not None:
            QUBOVertWarning.warn(
                "Both a temperature range and an explicit schedule was "
                "provided. The temperature range will be ignored and the "
                "schedule used instead."
            )
        return list(schedule)
    elif schedule not in SCHEDULES:
        raise ValueError(
            "Invalid schedule. Must be one of %s. "
            "See the docstring for more info." % str(SCHEDULES)
        )

    T0, Tf = (
        temperature_range or anneal_temperature_range(spin_model, spin=True)
    )
    if T0 < Tf:
        raise ValueError("The final temperature must be less than the "
                         "initial temperature")

    # in the case that model is empty or just an offset and the user didn't
    # supply a temperature range, then T0 and Tf will be 0.
    if temperature_range is None and T0 == Tf == 0:
        T0 = Tf = 1
    return list(
        np.linspace(T0, Tf, anneal_duration) if schedule == 'linear' else
        np.geomspace(T0, Tf, anneal_duration)
    )


def _package_spin_results(states, values, offset, reverse_mapping):
    """_package_spin_results.

    Package the results of the C functions into the desired result form
    of the Python functions.

    Parameters
    ----------
    states : list of lists.
        ``states`` has dimension ``state[num_anneals][len_state]``.
    values : list of floats.
        The value of the objective function that each state gives,
        minus the offset.
    offset : float.
        The part of the objective function that does not depend on any
        variables.
    reverse_mapping : dict.
        Maps the integer spin labels to the original model variables.

    Returns
    -------
    res : qubovert.sim.AnnealResults object.

    """
    res = AnnealResults()
    for i in range(len(states)):
        state = {reverse_mapping[k]: v for k, v in enumerate(states[i])}
        res.add_state(state, values[i] + offset, True)  # spin is True
    return res


# spin annealing functions

def anneal_puso(H, num_anneals=1, anneal_duration=1000, initial_state=None,
                temperature_range=None, schedule='geometric',
                in_order=True, seed=None):
    """anneal_puso.

    Run a simulated annealing algorithm to try to find the minimum of the PUSO
    given by ``H``. Please see all of the parameters for details.

    **Please note** that the ``qv.sim.anneal_quso`` function performs
    faster than the ``qv.sim.anneal_puso`` function. If your system has
    degree 2 or less, then you should use the ``qv.sim.anneal_quso``
    function.

    Parameters
    ----------
    H : dict, or any type in ``qubovert.SPIN_MODELS``.
        Maps spin labels to their values in the Hamiltonian.
        Please see the docstrings of any of the objects in
        ``qubovert.SPIN_MODELS`` to see how ``H`` should be formatted.
    num_anneals : int >= 1 (optional, defaults to 1).
        The number of times to run the simulated annealing algorithm.
    anneal_duration : int >= 1 (optional, defaults to 1000).
        The total number of updates to the simulation during the anneal.
        This is related to the amount of time we spend in the cooling schedule.
        If an explicit schedule is provided, then ``anneal_duration`` will be
        ignored.
    initial_state : dict (optional, defaults to None).
        The initial state to start the anneal in. ``initial_state`` must map
        the spin label names to their values in {1, -1}. If ``initial_state``
        is None, then a random state will be chosen to start each anneal.
        Otherwise, ``initial_state`` will be the starting state for all of the
        anneals.
    temperature_range : tuple (optional, defaults to None).
        The temperature to start and end the anneal at.
        ``temperature = (T0, Tf)``. ``T0`` must be >= ``Tf``. To see more
        details on picking a temperature range, please see the function
        ``qubovert.sim.anneal_temperature_range``. If ``temperature_range`` is
        None, then it will by default be set to
        ``T0, Tf = qubovert.sim.anneal_temperature_range(H, spin=True)``.
        Note that a temperature can only be zero if ``schedule`` is explicitly
        given or if ``schedule`` is linear.
    schedule : str, or list of floats (optional, defaults to ``'geometric'``).
        What type of cooling schedule to use. If ``schedule == 'linear'``,
        then the cooling schedule will be a linear interpolation between the
        values in ``temperature_range``. If ``schedule == 'geometric'``, then
        the cooling schedule will be a geometric interpolation between the
        values in ``temperature_range``. Otherwise, ``schedule`` must be an
        iterable of floats being the explicit temperature schedule for the
        anneal to follow.
    in_order : bool (optional, defaults to True).
        Whether to iterate through the variables in order or randomly
        during an update step. When ``in_order`` is False, the simulation
        is more physically realistic, but when using the simulation for
        annealing, often it is better to have ``in_order = True``.
    seed : number (optional, defaults to None).
        The number to seed Python's builtin ``random`` module with. If
        ``seed is None``, then ``random.seed`` will not be called.

    Returns
    -------
    res : qubovert.sim.AnnealResults object.
        ``res`` contains information on the final states of the simulations.
        See Examples below for an example of how to read from ``res``.
        See ``help(qubovert.sim.AnnealResults)`` for more info.

    Raises
    ------
    ValueError
        If the ``schedule`` argument provided is formatted incorrectly. See the
        Parameters section.
    ValueError
        If the initial temperature is less than the final temperature.

    Warns
    -----
    qubovert.utils.QUBOVertWarning
        If both the ``temperature_range`` and explicit ``schedule`` arguments
        are provided.
    qubovert.utils.QUBOVertWarning
        If the degree of the model is 2 or less then a warning is issued that
        says you should use the ``anneal_qubo`` or ``anneal_quso`` functions.

    Example
    -------
    Consider the example of finding the ground state of the 1D
    antiferromagnetic Ising chain of length 5.

    >>> import qubovert as qv
    >>>
    >>> H = sum(qv.spin_var(i) * qv.spin_var(i+1) for i in range(4))
    >>> anneal_res = qv.sim.anneal_puso(H, num_anneals=3)
    >>>
    >>> print(anneal_res.best.value)
    -4
    >>> print(anneal_res.best.state)
    {0: 1, 1: -1, 2: 1, 3: -1, 4: 1}
    >>> # now sort the results
    >>> anneal_res.sort()
    >>>
    >>> # now iterate through all of the results in the sorted order
    >>> for res in anneal_res:
    >>>     print(res.value, res.state)
    -4, {0: 1, 1: -1, 2: 1, 3: -1, 4: 1}
    -4, {0: -1, 1: 1, 2: -1, 3: 1, 4: -1}
    -4, {0: 1, 1: -1, 2: 1, 3: -1, 4: 1}

    """
    if num_anneals <= 0:
        return AnnealResults()

    Ts = _create_spin_schedule(
        H, anneal_duration, temperature_range, schedule
    )

    # must use type since we don't want errors from inheritance
    if type(H) in (QUSOMatrix, PUSOMatrix):
        N = H.max_index + 1
        model = H
        reverse_mapping = dict(enumerate(range(N)))
    elif type(H) not in (QUSO, PUSO, PCSO):
        H = PUSO(H)

    if type(H) in (QUSO, PUSO, PCSO):
        N = H.num_binary_variables
        model = H.to_puso()
        reverse_mapping = H.reverse_mapping

    if model.degree <= 2:
        QUBOVertWarning.warn(
            "The input problem has degree <= 2; consider using the "
            "``qubovert.sim.anneal_qubo`` or ``qubovert.sim.anneal_quso`` "
            "functions, which are significantly faster than this function "
            "because they take advantage of the low degree."
        )

    # solve `model`, convert solutions back to `H`

    if not N:
        return AnnealResults(
            AnnealResult({}, model.offset, True) for _ in range(num_anneals)
        )

    if initial_state is not None:
        init_state = [1] * N
        for k, v in reverse_mapping.items():
            init_state[k] = initial_state[v]
    else:
        init_state = []

    # create arguments for the C function
    # create terms and couplings
    terms, couplings, num_couplings = [], [], []
    for term, coupling in model.items():
        if term:
            couplings.append(float(coupling))
            terms.extend(term)
            num_couplings.append(len(term))

    states, values = c_anneal_puso(
        N, num_couplings, terms, couplings,  # describe the problem
        Ts, num_anneals, int(in_order), init_state,  # describe the algorithm
        seed if seed is not None else -1
    )
    return _package_spin_results(
        states, values, model.offset, reverse_mapping
    )


def anneal_quso(L, num_anneals=1, anneal_duration=1000, initial_state=None,
                temperature_range=None, schedule='geometric',
                in_order=True, seed=None):
    """anneal_quso.

    Run a simulated annealing algorithm to try to find the minimum of the QUSO
    given by ``L``. Please see all of the parameters for details.

    Parameters
    ----------
    L : dict, ``qubovert.utils.QUSOMatrix`` or ``qubovert.QUSO``.
        Maps spin labels to their values in the objective function.
        Please see the docstring of ``qubovert.QUSO`` for more info on how to
        format ``L``.
    num_anneals : int >= 1 (optional, defaults to 1).
        The number of times to run the simulated annealing algorithm.
    anneal_duration : int >= 1 (optional, defaults to 1000).
        The total number of updates to the simulation during the anneal.
        This is related to the amount of time we spend in the cooling schedule.
        If an explicit schedule is provided, then ``anneal_duration`` will be
        ignored.
    initial_state : dict (optional, defaults to None).
        The initial state to start the anneal in. ``initial_state`` must map
        the spin label names to their values in {1, -1}. If ``initial_state``
        is None, then a random state will be chosen to start each anneal.
        Otherwise, ``initial_state`` will be the starting state for all of the
        anneals.
    temperature_range : tuple (optional, defaults to None).
        The temperature to start and end the anneal at.
        ``temperature = (T0, Tf)``. ``T0`` must be >= ``Tf``. To see more
        details on picking a temperature range, please see the function
        ``qubovert.sim.anneal_temperature_range``. If ``temperature_range`` is
        None, then it will by default be set to
        ``T0, Tf = qubovert.sim.anneal_temperature_range(L, spin=True)``.
    schedule : str, or list of floats (optional, defaults to ``'geometric'``).
        What type of cooling schedule to use. If ``schedule == 'linear'``,
        then the cooling schedule will be a linear interpolation between the
        values in ``temperature_range``. If ``schedule == 'geometric'``, then
        the cooling schedule will be a geometric interpolation between the
        values in ``temperature_range``. Otherwise, ``schedule`` must be an
        iterable of floats being the explicit temperature schedule for the
        anneal to follow.
    in_order : bool (optional, defaults to True).
        Whether to iterate through the variables in order or randomly
        during an update step. When ``in_order`` is False, the simulation
        is more physically realistic, but when using the simulation for
        annealing, often it is better to have ``in_order = True``.
    seed : number (optional, defaults to None).
        The number to seed Python's builtin ``random`` module with. If
        ``seed is None``, then ``random.seed`` will not be called.

    Returns
    -------
    res : qubovert.sim.AnnealResults object.
        ``res`` contains information on the final states of the simulations.
        See Examples below for an example of how to read from ``res``.
        See ``help(qubovert.sim.AnnealResults)`` for more info.

    Raises
    ------
    ValueError
        If the ``schedule`` argument provided is formatted incorrectly. See the
        Parameters section.
    ValueError
        If the initial temperature is less than the final temperature.
    ValueError
        If ``L`` is not degree 2 or less.

    Warns
    -----
    qubovert.utils.QUBOVertWarning
        If both the ``temperature_range`` and explicit ``schedule`` arguments
        are provided.

    Example
    -------
    Consider the example of finding the ground state of the 1D
    antiferromagnetic Ising chain of length 5.

    >>> import qubovert as qv
    >>>
    >>> H = sum(qv.spin_var(i) * qv.spin_var(i+1) for i in range(4))
    >>> anneal_res = qv.sim.anneal_quso(H, num_anneals=3)
    >>>
    >>> print(anneal_res.best.value)
    -4
    >>> print(anneal_res.best.state)
    {0: 1, 1: -1, 2: 1, 3: -1, 4: 1}
    >>> # now sort the results
    >>> anneal_res.sort()
    >>>
    >>> # now iterate through all of the results in the sorted order
    >>> for res in anneal_res:
    >>>     print(res.value, res.state)
    -4, {0: 1, 1: -1, 2: 1, 3: -1, 4: 1}
    -4, {0: -1, 1: 1, 2: -1, 3: 1, 4: -1}
    -4, {0: 1, 1: -1, 2: 1, 3: -1, 4: 1}

    """
    if num_anneals <= 0:
        return AnnealResults()

    Ts = _create_spin_schedule(
        L, anneal_duration, temperature_range, schedule
    )

    # must use type since we don't want errors from inheritance
    if type(L) == QUSOMatrix:
        N = L.max_index + 1
        model = L
        reverse_mapping = dict(enumerate(range(N)))
        # mapping = reverse_mapping
    elif type(L) != QUSO:
        L = QUSO(L)

    if type(L) == QUSO:
        N = L.num_binary_variables
        model = L.to_quso()
        # mapping = L.mapping
        reverse_mapping = L.reverse_mapping

    # solve `model`, convert solutions back to `L`

    if not N:
        return AnnealResults(
            AnnealResult({}, model.offset, True) for _ in range(num_anneals)
        )

    if initial_state is not None:
        init_state = [1] * N
        for k, v in reverse_mapping.items():
            init_state[k] = initial_state[v]
    else:
        init_state = []

    # create arguments for the C function
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

    # flatten the arrays.
    J, neighbors = list(chain(*J)), list(chain(*neighbors))

    states, values = c_anneal_quso(
        h, num_neighbors, neighbors, J,  # describe the problem
        Ts, num_anneals, int(in_order), init_state,  # describe the algorithm
        seed if seed is not None else -1
    )
    return _package_spin_results(
        states, values, model.offset, reverse_mapping
    )


# boolean annealing functions

def anneal_pubo(P, num_anneals=1, anneal_duration=1000, initial_state=None,
                temperature_range=None, schedule='geometric',
                in_order=True, seed=None):
    """anneal_pubo.

    Run a simulated annealing algorithm to try to find the minimum of the PUBO
    given by ``P``. ``anneal_pubo`` converts ``P`` to a PUSO and then uses
    ``qubovert.sim.anneal_quso``. Please see all the parameters for details.

    **Please note** that the ``qv.sim.anneal_qubo`` function performs
    faster than the ``qv.sim.anneal_pubo`` function. If your system has
    degree 2 or less, then you should use the ``qv.sim.anneal_qubo`` function.

    Parameters
    ----------
    P : dict, or any type in ``qubovert.BOOLEAN_MODELS``.
        Maps boolean labels to their values in the objective function.
        Please see the docstrings of any of the objects in
        ``qubovert.BOOLEAN_MODELS`` to see how ``P`` should be formatted.
    num_anneals : int >= 1 (optional, defaults to 1).
        The number of times to run the simulated annealing algorithm.
    anneal_duration : int >= 1 (optional, defaults to 1000).
        The total number of updates to the simulation during the anneal.
        This is related to the amount of time we spend in the cooling schedule.
        If an explicit schedule is provided, then ``anneal_duration`` will be
        ignored.
    initial_state : dict (optional, defaults to None).
        The initial state to start the anneal in. ``initial_state`` must map
        the spin label names to their values in {0, 1}. If ``initial_state``
        is None, then a random state will be chosen to start each anneal.
        Otherwise, ``initial_state`` will be the starting state for all of the
        anneals.
    temperature_range : tuple (optional, defaults to None).
        The temperature to start and end the anneal at.
        ``temperature = (T0, Tf)``. ``T0`` must be >= ``Tf``. To see more
        details on picking a temperature range, please see the function
        ``qubovert.sim.anneal_temperature_range``. If ``temperature_range`` is
        None, then it will by default be set to
        ``T0, Tf = qubovert.sim.anneal_temperature_range(P, spin=False)``.
    schedule : str or iterable of tuple (optional, defaults to ``'geometric'``)
        What type of cooling schedule to use. If ``schedule == 'linear'``, then
        the cooling schedule will be a linear interpolation between the values
        in ``temperature_range``. If ``schedule == 'geometric'``, then the
        cooling schedule will be a geometric interpolation between the values
        in ``temperature_range``. Otherwise, you can supply an explicit
        schedule. In this case, ``schedule`` should be an iterable of tuples,
        where each tuple is a ``(T, n)`` pair, where ``T`` denotes the
        temperature to update the simulation, and ``n`` denote the number of
        times to update the simulation at that temperature. This schedule
        will be sent directly into the
        ``qubovert.sim.PUBOSimulation.schedule_update`` method.
    in_order : bool (optional, defaults to True).
        Whether to iterate through the variables in order or randomly
        during an update step. When ``in_order`` is False, the simulation
        is more physically realistic, but when using the simulation for
        annealing, often it is better to have ``in_order = True``.
    seed : number (optional, defaults to None).
        The number to seed Python's builtin ``random`` module with. If
        ``seed is None``, then ``random.seed`` will not be called.

    Returns
    -------
    res : qubovert.sim.AnnealResults object.
        ``res`` contains information on the final states of the simulations.
        See Examples below for an example of how to read from ``res``.
        See ``help(qubovert.sim.AnnealResults)`` for more info.

    Raises
    ------
    ValueError
        If the ``schedule`` argument provided is formatted incorrectly. See the
        Parameters section.
    ValueError
        If the initial temperature is less than the final temperature.

    Warns
    -----
    qubovert.utils.QUBOVertWarning
        If both the ``temperature_range`` and explicit ``schedule`` arguments
        are provided.
    qubovert.utils.QUBOVertWarning
        If the degree of the model is 2 or less then a warning is issued that
        says you should use the ``anneal_qubo`` or ``anneal_quso`` functions.

    Example
    -------
    Consider the example of finding the ground state of the 1D
    antiferromagnetic Ising chain of length 5 in boolean form.

    >>> import qubovert as qv
    >>>
    >>> H = sum(qv.spin_var(i) * qv.spin_var(i+1) for i in range(4))
    >>> P = H.to_pubo()
    >>> anneal_res = qv.sim.anneal_pubo(P, num_anneals=3)
    >>>
    >>> print(anneal_res.best.value)
    -4
    >>> print(anneal_res.best.state)
    {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
    >>> # now sort the results
    >>> anneal_res.sort()
    >>>
    >>> # now iterate through all of the results in the sorted order
    >>> for res in anneal_res:
    >>>     print(res.value, res.state)
    -4, {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
    -4, {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}
    -4, {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}

    """
    return anneal_puso(
        pubo_to_puso(P), num_anneals, anneal_duration,
        boolean_to_spin(initial_state) if initial_state is not None else None,
        temperature_range, schedule, in_order, seed
    ).to_boolean()


def anneal_qubo(Q, num_anneals=1, anneal_duration=1000, initial_state=None,
                temperature_range=None, schedule='geometric',
                in_order=True, seed=None):
    """anneal_qubo.

    Run a simulated annealing algorithm to try to find the minimum of the QUBO
    given by ``Q``. ``anneal_qubo`` converts ``Q`` to a QUSO and then uses
    ``qubovert.sim.anneal_quso``.
    Please see all of the parameters for details.

    Parameters
    ----------
    Q : dict, ``qubovert.utils.QUBOMatrix`` or ``qubovert.QUBO``.
        Maps boolean labels to their values in the objective function.
        Please see the docstring of ``qubovert.QUBO`` for more info on how to
        format ``Q``.
    num_anneals : int >= 1 (optional, defaults to 1).
        The number of times to run the simulated annealing algorithm.
    anneal_duration : int >= 1 (optional, defaults to 1000).
        The total number of updates to the simulation during the anneal.
        This is related to the amount of time we spend in the cooling schedule.
        If an explicit schedule is provided, then ``anneal_duration`` will be
        ignored.
    initial_state : dict (optional, defaults to None).
        The initial state to start the anneal in. ``initial_state`` must map
        the spin label names to their values in {0, 1}. If ``initial_state``
        is None, then a random state will be chosen to start each anneal.
        Otherwise, ``initial_state`` will be the starting state for all of the
        anneals.
    temperature_range : tuple (optional, defaults to None).
        The temperature to start and end the anneal at.
        ``temperature = (T0, Tf)``. ``T0`` must be >= ``Tf``. To see more
        details on picking a temperature range, please see the function
        ``qubovert.sim.anneal_temperature_range``. If ``temperature_range`` is
        None, then it will by default be set to
        ``T0, Tf = qubovert.sim.anneal_temperature_range(Q, spin=False)``.
    schedule : str or iterable of tuple (optional, defaults to ``'geometric'``)
        What type of cooling schedule to use. If ``schedule == 'linear'``, then
        the cooling schedule will be a linear interpolation between the values
        in ``temperature_range``. If ``schedule == 'geometric'``, then the
        cooling schedule will be a geometric interpolation between the values
        in ``temperature_range``. Otherwise, you can supply an explicit
        schedule. In this case, ``schedule`` should be an iterable of tuples,
        where each tuple is a ``(T, n)`` pair, where ``T`` denotes the
        temperature to update the simulation, and ``n`` denote the number of
        times to update the simulation at that temperature. This schedule
        will be sent directly into the
        ``qubovert.sim.PUBOSimulation.schedule_update`` method.
    in_order : bool (optional, defaults to True).
        Whether to iterate through the variables in order or randomly
        during an update step. When ``in_order`` is False, the simulation
        is more physically realistic, but when using the simulation for
        annealing, often it is better to have ``in_order = True``.
    seed : number (optional, defaults to None).
        The number to seed Python's builtin ``random`` module with. If
        ``seed is None``, then ``random.seed`` will not be called.

    Returns
    -------
    res : qubovert.sim.AnnealResults object.
        ``res`` contains information on the final states of the simulations.
        See Examples below for an example of how to read from ``res``.
        See ``help(qubovert.sim.AnnealResults)`` for more info.

    Raises
    ------
    ValueError
        If the ``schedule`` argument provided is formatted incorrectly. See the
        Parameters section.
    ValueError
        If the initial temperature is less than the final temperature.

    Warns
    -----
    qubovert.utils.QUBOVertWarning
        If both the ``temperature_range`` and explicit ``schedule`` arguments
        are provided.

    Example
    -------
    Consider the example of finding the ground state of the 1D
    antiferromagnetic Ising chain of length 5 in boolean form.

    >>> import qubovert as qv
    >>>
    >>> H = sum(qv.spin_var(i) * qv.spin_var(i+1) for i in range(4))
    >>> Q = H.to_qubo()
    >>> anneal_res = qv.sim.anneal_qubo(Q, num_anneals=3)
    >>>
    >>> print(anneal_res.best.value)
    -4
    >>> print(anneal_res.best.state)
    {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
    >>> # now sort the results
    >>> anneal_res.sort()
    >>>
    >>> # now iterate through all of the results in the sorted order
    >>> for res in anneal_res:
    >>>     print(res.value, res.state)
    -4, {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
    -4, {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}
    -4, {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}

    """
    return anneal_quso(
        qubo_to_quso(Q), num_anneals, anneal_duration,
        boolean_to_spin(initial_state) if initial_state is not None else None,
        temperature_range, schedule, in_order, seed
    ).to_boolean()
