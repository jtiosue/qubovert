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

"""_subgraph.py.

This file contains the function to create subgraphs and subvalues of QUBOs,
PUBOs, QUSOs, PUSOs, etc.

"""

import numpy as np


__all__ = 'subgraph', 'subvalue'


def subgraph(G, nodes, connections=None):
    """subgraph.

    Create the subgraph of ``G`` that only includes vertices in ``nodes``, and
    external nodes are given the values in ``connections``.

    Parameters
    ----------
    G : dict or any subclass of dict.
        ``G`` must contain tuple keys.
    nodes : set.
        Nodes of ``G`` to include in the subgraph.
    connections : dict (optional, defaults to {}).
        For each node in ``G`` that is not in ``nodes``, we assign a value
        given by ``connections.get(node, 0)``.

    Return
    ------
    D : same as type(G).
        The subgraph of ``G`` with nodes in ``nodes`` and the values of the
        nodes not included given by ``connections``.

    Notes
    -----
    Any offset value included in ``G`` (ie {(): 1}) will be ignored, however
    there may be an offset in the output ``D``.

    Examples
    --------
    >>> G = {(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2}
    >>> D = subgraph(G, {0, 2}, {1: 5})
    >>> D
    {(0,): -17, (0, 2): -1, (): 10}

    >>> G = {(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2}
    >>> D = subgraph(G, {0, 2})
    >>> D
    {(0, 2): -1, (0,): 3}

    >>> G = {(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2}
    >>> D = subgraph(G, {0, 1}, {2: -10})
    >>> D
    {(0, 1): -4, (0,): 13, (1,): 2}

    >>> G = {(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2}
    >>> D = subgraph(G, {0, 1})
    >>> D
    {(0, 1): -4, (0,): 3, (1,): 2}

    """
    if connections is None:
        connections = {}

    D = type(G)()
    for k, v in G.items():
        if not isinstance(k, tuple):
            raise ValueError("Keys must be tuples")
        if not k:
            continue
        key = tuple(filter(lambda x: x in nodes, k))
        not_key = filter(lambda x: x not in nodes, k)
        value = v * np.prod([connections.get(i, 0) for i in not_key])
        value += D.get(key, 0)
        if value:
            D[key] = value
        else:
            D.pop(key, 0)

    return D


def subvalue(values, G):
    """subvalue.

    Replace each element in ``G`` with a value in ``values`` if it exists.

    Parameters
    ----------
    values : dict.
        For each node ``v`` in ``G`` that is in ``values``, we replace the
        node with ``values[v]``.
    G : dict or any subclass of dict.
        ``G`` must contain tuple keys.

    Return
    ------
    D : same as type(G).

    Examples
    --------
    >>> G = {(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2}
    >>> D = subvalue({0: 2}, G)
    >>> D
    {(1,): -6, (2,): -2, (): 8}

    >>> G = {(0, 1): -4, (0, 2): -1, (0,): 3, (1,): 2, (): 2}
    >>> D = subvalue({2: -3}, G)
    >>> D
    {(0, 1): -4, (0,): 6, (1,): 2, (): 2}

    """
    D = type(G)()
    for k, v in G.items():
        if not isinstance(k, tuple):
            raise ValueError("Keys must be tuples")
        key = tuple(filter(lambda x: x not in values, k))
        not_key = filter(lambda x: x in values, k)
        value = v * np.prod([values[i] for i in not_key])
        value += D.get(key, 0)
        if value:
            D[key] = value
        else:
            D.pop(key, 0)

    return D
