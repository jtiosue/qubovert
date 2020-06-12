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
    puso_value, pubo_to_puso, qubo_to_quso, QUBOVertWarning, boolean_to_spin
)
from . import PUSOSimulation, QUSOSimulation, AnnealResults
import random
import numpy as np
from math import log

__all__ = (
    'anneal_qubo', 'anneal_quso', 'anneal_pubo', 'anneal_puso',
    'anneal_temperature_range'
)


# anneal temperature range function

def anneal_temperature_range(model, start_flip_prob=0.5,
                             end_flip_prob=0.01, spin=False):
    """anneal_temperature_range.

    Calculate the temperature to start and end an anneal of ``model``, such
    that at the start of the anneal there is a ``start_flip_prob`` probability
    that a bit is flipped despite it being energetically unfavorable, and at
    the end of the anneal there is a ``end_flip_prob`` probability that a bit
    is flipped despite it being energetically unfavorable.

    Parameters
    ----------
    model : dict, or any type in ``qubovert.SPIN_MODELS`` or ``BOOLEAN_MODELS``
        Dictionary mapping tuples of binary labels to their values. See any of
        the docstrings of a type in ``qubovert.SPIN_MODELS`` or
        ``BOOLEAN_MODELS`` for more info.
    start_flip_prob : float in [0, 1) (optional, defaults to 0.5).
        The desired probability that a bit flips despite it being energetically
        unfavorable at the start of the anneal. ``start_flip_prob`` must be
        greater than ``end_flip_prob``.
    end_flip_prob : float in [0, 1) (optional, defaults to 0.01).
        The desired probability that a bit flips despite it being energetically
        unfavorable at the end of the anneal. ``end_flip_prob`` must be
        less than ``start_flip_prob``.
    spin : bool (optional, default to False).
        ``spin`` should be True if ``model`` is a spin model (ie
        ``isinstance(model, qubovert.SPIN_MODELS)``) and should be False if
        ``model`` is a boolean model (ie
        ``isinstance(model, qubovert.BOOLEAN_MODELS)``).

    Returns
    -------
    temp_range : tuple (hot, cold).
        The ``hot`` temperature is the temperature to start the anneal at, and
        the ``cold`` temperature is the temperature to end the anneal at.
        Note that ``hot >= cold``.

    """
    # slight modification of _default_ising_beta_range in
    # https://github.com/dwavesystems/dwave-neal/blob/master/neal/sampler.py

    # raise exception if invalid probabilities
    if any((
        start_flip_prob < 0, start_flip_prob >= 1,
        end_flip_prob < 0, end_flip_prob >= 1,
    )):
        raise ValueError("Flip probabilities must be in [0, 1)")
    elif end_flip_prob > start_flip_prob:
        raise ValueError("The starting flip probability must be greater than "
                         "the ending flip probability.")

    if not spin:
        model = pubo_to_puso(model)

    # if D is a Matrix object or QUBO, PUBO, etc, then variables are defined
    try:
        # don't waste time copying (model.variables), since we never mutate it.
        variables = model._variables
    except AttributeError:
        variables = set(v for k in model for v in k)

    # if the model is empty or just an offset
    if not variables:
        return 0, 0

    factor = 2  # should be this
    # factor = 1  # D-Wave neal does this.

    # calculate the approximate minimum possible change in energy by flipping
    # a single bit.
    min_del_energy = factor * min(abs(c) for k, c in model.items() if k)
    # calculate the approximate maximum possible change in energy by flipping
    # a single bit.
    max_del_energy = factor * max(
        sum(abs(c) for k, c in model.items() if v in k)
        for v in variables
    )

    # now ensure that the bolzmann weight satisfy the desired probabilities.
    # ie exp(-del_energy / T) = prob
    T0 = -max_del_energy / log(start_flip_prob) if start_flip_prob else 0
    Tf = -min_del_energy / log(end_flip_prob) if end_flip_prob else 0
    return float(T0), float(Tf)


# main spin annealing function

def _anneal_spin(model, spin_simulation, num_anneals=1,
                 anneal_duration=1000, initial_state=None,
                 temperature_range=None, schedule='geometric',
                 in_order=True, seed=None):
    """_anneal_spin.

    Run a simulated annealing algorithm to try to find the minimum of the spin
    model given by ``model``. ``_anneal_spin`` uses a cooling schedule with the
    ``spin_simulation`` object. Please see all of the parameters for details.

    Both ``qv.sim.anneal_puso`` and ``qv.sim.anneal_quso`` run through this
    function. Since ``qv.sim.QUSOSimulation`` is faster than
    ``qv.sim.PUSOSimulation``, we send in different simulation objects for
    ``anneal_quso`` and ``anneal_puso``.

    Parameters
    ----------
    model : dict, or any type in ``qubovert.SPIN_MODELS``.
        Maps spin labels to their values in the Hamiltonian.
        Please see the docstrings of any of the objects in
        ``qubovert.SPIN_MODELS`` to see how ``H`` should be formatted.
    spin_simulation : qv.sim.PUSOSimulation or qv.sim.QUSOSimulation object.
        Should be a ``qv.sim.QUSOSimulation`` object if this function is called
        from ``qv.sim.anneal_quso``, or a ``qv.sim.PUSOSimulation`` object if
        this function is called from ``qv.sim.anneal_puso``.
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
        ``qubovert.sim.PUSOSimulation.schedule_update`` method.
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

    """
    if seed is not None:
        random.seed(seed)

    if schedule in ('linear', 'geometric'):
        T0, Tf = temperature_range or anneal_temperature_range(model,
                                                               spin=True)
        if T0 < Tf:
            raise ValueError("The final temperature must be less than the "
                             "initial temperature")

        # in the case that H is empty or just an offset and the user didn't
        # supply a temperature range, then T0 and Tf will be 0.
        if temperature_range is None and T0 == Tf == 0:
            T0 = Tf = 1
        Ts = (
            np.linspace(T0, Tf, anneal_duration) if schedule == 'linear' else
            np.geomspace(T0, Tf, anneal_duration)
        )
        schedule = tuple((T, 1) for T in Ts)
    elif isinstance(schedule, str):
        raise ValueError(
            "Invalid schedule. Must be either 'linear', 'geometric', or an "
            "explicit temperature schedule. See the docstring for more info."
        )
    elif temperature_range:
        QUBOVertWarning.warn(
            "Both a temperature range and an explicit schedule was provided. "
            "The temperature range will be ignored and the schedule used "
            "instead."
        )

    sim = spin_simulation(model, initial_state)

    result = AnnealResults(True)
    for _ in range(num_anneals):
        if initial_state is None:
            sim.set_state({v: random.choice((-1, 1)) for v in sim._variables})
        sim.schedule_update(
            schedule, in_order=in_order,
            seed=random.randint(0, 1 << 16) if seed is not None else None
        )
        state = sim.state
        result.add_state(state, puso_value(state, model))
        sim.reset()

    return result


# annealing functions

def anneal_puso(H, num_anneals=1, anneal_duration=1000, initial_state=None,
                temperature_range=None, schedule='geometric',
                in_order=True, seed=None):
    """anneal_puso.

    Run a simulated annealing algorithm to try to find the minimum of the PUSO
    given by ``H``. ``anneal_puso`` uses a cooling schedule with the
    ``qubovert.sim.PUSOSimulation`` object. Please see all of the parameters
    for details.

    **Please note** that the ``qv.sim.anneal_quso`` function performs much
    faster than the ``qv.sim.anneal_puso`` function since the former is written
    in C and wrapped in Python. If your system has degree 2 or less, then you
    should use the ``qv.sim.anneal_quso`` function!

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
        ``qubovert.sim.PUSOSimulation.schedule_update`` method.
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
    return _anneal_spin(
        H, PUSOSimulation, num_anneals, anneal_duration, initial_state,
        temperature_range, schedule, in_order, seed
    )


def anneal_pubo(P, num_anneals=1, anneal_duration=1000, initial_state=None,
                temperature_range=None, schedule='geometric',
                in_order=True, seed=None):
    """anneal_pubo.

    Run a simulated annealing algorithm to try to find the minimum of the PUBO
    given by ``P``. ``anneal_pubo`` converts ``P`` to a PUSO and then uses a
    cooling schedule with the ``qubovert.sim.PUSOSimulation`` object. Please
    see all of the parameters for details.

    **Please note** that the ``qv.sim.anneal_qubo`` function performs much
    faster than the ``qv.sim.anneal_pubo`` function since the former is written
    in C and wrapped in Python. If your system has degree 2 or less, then you
    should use the ``qv.sim.anneal_qubo`` function!

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
        the boolean label names to their values in {0, 1}. If ``initial_state``
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
        None if initial_state is None else boolean_to_spin(initial_state),
        temperature_range, schedule, in_order, seed
    ).to_boolean()


def anneal_quso(L, num_anneals=1, anneal_duration=1000, initial_state=None,
                temperature_range=None, schedule='geometric',
                in_order=True, seed=None):
    """anneal_quso.

    Run a simulated annealing algorithm to try to find the minimum of the QUSO
    given by ``L``. ``anneal_quso`` uses a cooling schedule with the
    ``qubovert.sim.QUSOSimulation`` object. Please see all of the parameters
    for details.

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
        ``qubovert.sim.PUSOSimulation.schedule_update`` method.
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
    return _anneal_spin(
        L, QUSOSimulation, num_anneals, anneal_duration, initial_state,
        temperature_range, schedule, in_order, seed
    )


def anneal_qubo(Q, num_anneals=1, anneal_duration=1000, initial_state=None,
                temperature_range=None, schedule='geometric',
                in_order=True, seed=None):
    """anneal_qubo.

    Run a simulated annealing algorithm to try to find the minimum of the QUBO
    given by ``Q``. ``anneal_qubo`` converts ``Q`` to a QUSO and then uses a
    cooling schedule with the ``qubovert.sim.QUSOSimulation`` object. Please
    see all of the parameters for details.

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
        the boolean label names to their values in {0, 1}. If ``initial_state``
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
        None if initial_state is None else boolean_to_spin(initial_state),
        temperature_range, schedule, in_order, seed
    ).to_boolean()
