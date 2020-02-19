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

"""_normalize.py.

This file contains the function to normalize QUBOs, PUBOs, QUSOs,
PUSOs, etc.

"""


__all__ = 'normalize',


def normalize(D, value=1):
    """normalize.

    Normalize the coefficients to a maximum magnitude.

    Parameters
    ----------
    D : dict or subclass of dict.
    value : float (optional, defaults to 1).
        Every coefficient value will be normalized such that the
        coefficient with the maximum magnitude will be +/- 1.

    Return
    ------
    res : same as type(D).
        ``D`` but with coefficients that are normalized to be within +/- value.

    Examples
    --------
    >>> from qubovert.utils import DictArithmetic, normalize
    >>> d = {(0, 1): 1, (1, 2, 'x'): 4}
    >>> print(normalize(d))
    {(0, 1): 0.25, (1, 2, 'x'): 1}

    >>> from qubovert.utils import DictArithmetic, normalize
    >>> d = {(0, 1): 1, (1, 2, 'x'): -4}
    >>> print(normalize(d))
    {(0, 1): 0.25, (1, 2, 'x'): -1}

    >>> from qubovert import PUBO
    >>> d = PUBO({(0, 1): 1, (1, 2, 'x'): 4})
    >>> print(normalize(d))
    {(0, 1): 0.25, (1, 2, 'x'): 1}

    >>> from qubovert.utils import PUBO
    >>> d = PUBO({(0, 1): 1, (1, 2, 'x'): -4})
    >>> print(normalize(d))
    {(0, 1): 0.25, (1, 2, 'x'): -1}


    """
    res = type(D)()
    mult = value / max(abs(v) for v in D.values())
    for k, v in D.items():
        res[k] = mult * v
    return res
