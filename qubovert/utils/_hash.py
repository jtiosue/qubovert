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

"""_hash.py.

This file contains the hash function that we use to sort keys in the matrices,
used in PUBOMatrix, PUSOMatrix, QUBOMatrix, QUSOMatrix, and all of their
subclasses.

"""

__all__ = 'hash_function',


def hash_function(x):
    """hash_function.

    Function to return (usually) unique hashes for ``x`` such
    that multiple ``x`` s can be ordered. Note that the hash is not consistent
    across Python sessions (except for when ``x`` is an integer)!

    Parameters
    ----------
    x : hashable object.

    Returns
    -------
    res : int.
        If ``x`` is an integer, then ``res == x``. Otherwise,
        ``res == hash(x)``.

    """
    return x if isinstance(x, int) else hash(x)
