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

"""_ordering_key.py.

This file contains the ordering key function that we use to sort keys in
qubovert objects. It is used in PUBOMatrix, PUSOMatrix, QUBOMatrix,
QUSOMatrix, and all of their subclasses.

"""

__all__ = 'ordering_key',


def ordering_key(x):
    """ordering_key.

    Return a key to sort keys of qubovert types with. We sort by
    type first, and then by object.

    Parameters
    ----------
    x : hashable object.

    Returns
    -------
    res : tuple.

    Example
    -------
    >>> sorted([2, 0, 'a'], key=ordering_key)
    [0, 2, 'a']

    """
    return str(type(x)), x
