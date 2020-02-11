#   Copyright 2019 Joseph T. Iosue
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

"""_solution_type.py.

This file contains the function ``solution_type`` which takes in a solution and
determines if it is known whether the solution came from a boolean or spin
formultion.

"""


__all__ = 'solution_type',


def solution_type(solution):
    """solution_type.

    Figure out if the ``solution`` is a solution to a boolean or a spin model.
    If it cannot be determined (ie if ``solution`` is all 1s), then return
    ``None``.

    Parameters
    ----------
    solution : dict or iterable.
        Either a dictionary that maps variable names to values, or an iterable
        with values. If ``solution`` is a bin type, then all the values should
        be either a 0 or 1. If ``solution`` is a spin type, then all the values
        should be either a 1 or -1.

    Returns
    -------
    res : str.
        If it is determined that ``solution`` is the solution to a spin model,
        then ``res`` will be ``'spin'``. If it is determined that ``solution``
        is the solution to a boolean model, then `res`` will be ``'bool'``.
        Otherwise, ``res`` will be ``None``.

    Examples
    --------
    >>> solution_type((0, 1, 1, 0))
    'bin'

    >>> solution_type((1, -1, -1, 1))
    'spin'

    >>> solution_type((1, 1, 1, 1))
    None

    """
    sol = solution.values() if isinstance(solution, dict) else solution
    for v in sol:
        if v == 0:
            return 'bool'
        elif v == -1:
            return 'spin'
