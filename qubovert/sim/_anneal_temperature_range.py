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

"""_anneal_temperature_range.py.

This file contains the function to determine the temperature range for
annealing a spin model.

"""

from qubovert.utils import pubo_to_puso
from math import log

__all__ = "anneal_temperature_range",


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

    factor = 2  # should be this (I think)
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
    T0 = -max_del_energy / log(start_flip_prob) if start_flip_prob else 0.
    Tf = -min_del_energy / log(end_flip_prob) if end_flip_prob else 0.
    return T0, Tf
