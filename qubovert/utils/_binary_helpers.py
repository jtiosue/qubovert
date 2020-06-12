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

"""_binary_helpers.py.

This file contains the multiple generic helper functions.

"""

from math import ceil

__all__ = 'solution_type', 'num_bits'


def solution_type(solution, default='bool'):
    """solution_type.

    Figure out if the ``solution`` is a solution to a boolean or a spin model.
    If it cannot be determined (ie if ``solution`` is all 1s), then return
    ``default``.

    Parameters
    ----------
    solution : dict or iterable.
        Either a dictionary that maps variable names to values, or an iterable
        with values. If ``solution`` is a bin type, then all the values should
        be either a 0 or 1. If ``solution`` is a spin type, then all the values
        should be either a 1 or -1.
    default : str (optional, defaults to ``'bool'``).
        The default answer to return if the solution type cannot be determined.

    Returns
    -------
    res : str.
        If it is determined that ``solution`` is the solution to a spin model,
        then ``res`` will be ``'spin'``. If it is determined that ``solution``
        is the solution to a boolean model, then `res`` will be ``'bool'``.
        Otherwise, ``res`` will be ``default``.

    Examples
    --------
    >>> solution_type((0, 1, 1, 0))
    'bool'

    >>> solution_type((1, -1, -1, 1))
    'spin'

    In these cases, the default is invoked.

    >>> solution_type((1, 1, 1, 1))
    'bool'
    >>> solution_type((1, 1, 1, 1), default='bool')
    'bool'
    >>> solution_type((1, 1, 1, 1), default=='spin')
    'spin'

    """
    sol = solution.values() if isinstance(solution, dict) else solution
    for v in sol:
        if v == 0:
            return 'bool'
        elif v == -1:
            return 'spin'
    return default


def num_bits(val, log_trick=True):
    """num_bits.

    Find the number of bits needed to represent the value ``val``.

    Parameters
    ----------
    val : numeric.
    log_trick : bool (optional, defaults to True).
        Whether or not to use a log encoding.

    Return
    ------
    num : int.
        The number of bits needed to encode the number ``val``.

    Examples
    --------
    >>> num_bits(7)
    3
    >>> num_bits(8)
    4
    >>> num_bits(7, log_trick=False)
    7
    >>> num_bits(8, log_trick=False)
    8

    """
    if val < 0:
        raise ValueError("``val`` must be >= 0")
    val = int(ceil(val))
    return int.bit_length(val) if log_trick else val
