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

__all__ = 'is_solution_spin', 'num_bits', 'sum'


def is_solution_spin(solution, default=False):
    """is_solution_spin.

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
    default : bool (optional, defaults to ``False``).
        The default answer to return if the solution type cannot be determined.

    Returns
    -------
    res : bool.
        If it is determined that ``solution`` is the solution to a spin model,
        then ``res`` will be ``True``. If it is determined that ``solution``
        is the solution to a boolean model, then `res`` will be ``False``.
        Otherwise, ``res`` will be ``default``.

    Examples
    --------
    >>> is_solution_spin((0, 1, 1, 0))
    False

    >>> is_solution_spin((1, -1, -1, 1))
    True

    >>> is_solution_spin(dict(enumerate((0, 1, 1, 0))))
    False

    >>> is_solution_spin(dict(enumerate((1, -1, -1, 1))))
    True

    In these cases, the default is invoked.

    >>> is_solution_spin((1, 1, 1, 1))
    False
    >>> is_solution_spin((1, 1, 1, 1), default=False)
    False
    >>> is_solution_spin((1, 1, 1, 1), default=True)
    True

    """
    sol = solution.values() if isinstance(solution, dict) else solution
    for v in sol:
        if v == 0:
            return False
        elif v == -1:
            return True
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


def sum(iterable, start=0):
    """sum.

    A utility for summing qubovert types. This will perform way faster than
    Python's built-in ``sum`` function when you use it with qubovert types.

    Parameters
    ----------
    iterable : any iterable.
    start : numeric or qubovert type (optional, defaults to 0).

    Returns
    -------
    res : same type as ``sum(iterable, start)`` with Python's builtin ``sum``.

    Examples
    --------
    >>> import time
    >>> import qubovert as qv
    >>>
    >>> xs = [qv.boolean_var(i) for i in range(1000)]

    >>> t0 = time.time()
    >>> sum(xs)
    >>> print(time.time() - t0)
    3.345559597015381

    >>> t0 = time.time()
    >>> qv.utils.sum(xs)
    >>> print(time.time() - t0)
    0.011152505874633789

    """
    for i in iterable:
        start += i
    return start
