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

"""_info.py.

This file contains functions that aid in representing ``qubovert`` objects
in a basic way so they can be shared easily.

"""

import qubovert as qv

__all__ = 'get_info', 'create_from_info'


def get_info(model):
    """get_info.

    Create a basic representation of the qubovert object encoded by ``model``.
    This basic representation is easily shareable. It works so that
    ``model == qv.utils.create_from_info(qv.utils.get_info(model))`` is True.

    Parameters
    ----------
    model : type in ``qubovert.BOOLEAN_MODELS`` or ``qubovert.SPIN_MODELS``.

    Returns
    -------
    res : dict.
        ``res`` will have many fields. The first is ``type`` which will
        indicate which qubovert type ``model`` is. Then there will be a
        ``terms`` field which indicates the terms in the model. Then there may
        be a ``name`` field which will be set to ``model.name``. Depending on
        the input type, there may be ``mapping``, ``num_ancillas``, and
        ``constraints`` fields.

    Example
    -------
    >>> import qubovert as qv
    >>>
    >>> pcbo = qv.PCBO({(0,): 1}).add_constraint_eq_zero({(0,): 1}, lam=0)
    >>> pcbo
    {(0,): 1}
    >>> pcbo.constraints
    {"eq": [{(0,): 1}]}
    >>> print(pcbo.name)
    None
    >>> info = qv.utils.get_info(pcbo)
    >>> info
    {"type"="PCBO", "name"=None, "terms"={(0,): 1}, "mapping"={0: 0},
     num_ancillas=0, "constraints"={"eq": [{(0,): 1}]}}
    >>> qv.utils.create_from_info(info) == pcbo
    True

    See Also
    --------
    qubovert.utils.create_from_info : opposite function.

    """
    res = dict(
        type=model.__class__.__name__,
        terms=dict(model),
        name=model.name
    )
    for attr in ("mapping", "num_ancillas", "constraints"):
        if hasattr(model, attr):
            res[attr] = getattr(model, attr)

    return res


def create_from_info(info):
    """create_from_info.

    Create a qubovert object encoded by the information stored in ``info``.
    It works so that
    ``info == qv.utils.get_info(qv.utils.create_from_info(info))`` is True.

    Parameters
    ----------
    info : dict.
        Same as the output of ``qubovert.utils.get_info``.
        ``info`` can have many fields. The first is ``type`` which will
        indicate which qubovert type ``model`` is. Then there will be a
        ``terms`` field which indicates the terms in the model. Then there may
        be a ``name`` field which will be set to ``model.name``. Depending on
        the input type, there may be ``mapping``, ``num_ancillas``, and
        ``constraints`` fields.

    Returns
    -------
    res : a type in ``qubovert.BOOLEAN_MODELS`` or ``qubovert.SPIN_MODELS``.
        The qubovert object that ``info`` is describing.

    Example
    -------
    >>> import qubovert as qv
    >>>
    >>> pcbo = qv.PCBO({(0,): 1}).add_constraint_eq_zero({(0,): 1}, lam=0)
    >>> pcbo
    {(0,): 1}
    >>> pcbo.constraints
    {"eq": [{(0,): 1}]}
    >>> print(pcbo.name)
    None
    >>> info = qv.utils.get_info(pcbo)
    >>> info
    {"type"="PCBO", "name"=None, "terms"={(0,): 1}, "mapping"={0: 0},
     num_ancillas=0, "constraints"={"eq": [{(0,): 1}]}}
    >>> qv.utils.create_from_info(info) == pcbo
    True

    See Also
    --------
    qubovert.utils.get_info : opposite function.

    """
    t = info["type"]
    model = getattr(qv.utils, t) if hasattr(qv.utils, t) else getattr(qv, t)
    model = model(info.get("terms", {}))
    model.name = info.get("name", None)
    if "mapping" in info and info["mapping"] is not None:
        model.set_mapping(info["mapping"])
    if "num_ancillas" in info and info["num_ancillas"]:
        model._ancilla = info["num_ancillas"]

    for k, v in info.get("constraints", {}).items():
        method = getattr(model, "add_constraint_%s_zero" % k)
        for x in v:
            method(x, lam=0)

    return model
