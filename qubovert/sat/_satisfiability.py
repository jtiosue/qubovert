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

"""_satisfiability.py.

This file contains functions for converting SAT problems to PUBOs.

"""

# for BOOLEAN_MODELS, we can't just `from qubovert import BOOLEAN_MODELS`
# because it causes circular imports, so instead just import qubovert.
import qubovert as qv


__all__ = "BUFFER", "NOT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR"


def BUFFER(x):
    """BUFFER.

    Return the boolean expression for the buffer of ``x``.

    Parameters
    ----------
    x : hashable object or dict (or subclass of dict).
        The expression to buffer.

    Return
    ------
    P : ``qubovert.PUBO`` object or same type as ``type(x)``.
        If ``x`` is a ``qubovert.QUBO``, ``qubovert.PCBO``,
        ``qubovert.utils.QUBOMatrix``, or ``qubovert.utils.PUBOMatrix`` object,
        then ``type(P) == type(x)``. Otherwise, ``type(P) == type(x)``.

    Example
    -------
    >>> from qubovert.sat import BUFFER
    >>> P = BUFFER('x')
    >>> P
    {('x',): 1}
    >>> P.value({'x': 1})
    1
    >>> P.value({'x': 0})
    0
    >>> type(P)
    qubovert._pubo.PUBO

    >>> P = BUFFER({(0, 1): 1})
    >>> P
    {(0, 1): 1}
    >>> P.value({0: 0, 1: 0})
    0
    >>> P.value({0: 0, 1: 1})
    0
    >>> P.value({0: 1, 1: 0})
    0
    >>> P.value({0: 1, 1: 1})
    1
    >>> type(P)
    qubovert._pubo.PUBO

    >>> from qubovert import boolean_var
    >>> x = boolean_var('x')
    >>> P = BUFFER(x)
    >>> P.value({'x': 1})
    1
    >>> P.value({'x': 0})
    0
    >>> type(P)
    qubovert.PCBO

    """
    if isinstance(x, qv.BOOLEAN_MODELS):
        return x.copy()
    return qv.PUBO(x) if isinstance(x, dict) else qv.PUBO({(x,): 1})


def NOT(x):
    """NOT.

    Return the boolean expression for the NOT of ``x``.

    Parameters
    ----------
    x : hashable object or dict (or subclass of dict).
        The expression to buffer.

    Return
    ------
    P : ``qubovert.PUBO`` object or same type as ``type(x)``.
        The boolean expression for the logic operation.
        If ``x`` is a ``qubovert.QUBO``, ``qubovert.PCBO``,
        ``qubovert.utils.QUBOMatrix``, or ``qubovert.utils.PUBOMatrix`` object,
        then ``type(P) == type(x)``. Otherwise, ``type(P) == type(x)``.

    Example
    -------
    >>> from qubovert.sat import NOT
    >>> P = NOT('x')
    >>> P
    {(): 1, ('x',): -1}
    >>> P.value({'x': 1})
    0
    >>> P.value({'x': 0})
    1
    >>> type(P)
    qubovert._pubo.PUBO

    >>> P = NOT({(0, 1): 1})
    >>> P
    {(): 1, (0, 1): -1}
    >>> P.value({0: 0, 1: 0})
    1
    >>> P.value({0: 0, 1: 1})
    1
    >>> P.value({0: 1, 1: 0})
    1
    >>> P.value({0: 1, 1: 1})
    0

    >>> from qubovert import boolean_var
    >>> x = boolean_var('x')
    >>> P = NOT(x)
    >>> P.value({'x': 1})
    0
    >>> P.value({'x': 0})
    1
    >>> type(P)
    qubovert.PCBO

    """
    return 1 - BUFFER(x)


def AND(*variables):
    """AND.

    Return the boolean expression for the AND of the variables.

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the boolean variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object or same type as ``type(variables[0])``.
        The boolean expression for the logic operation.
        If ``variables[0]`` is a ``qubovert.QUBO``, ``qubovert.PCBO``,
        ``qubovert.utils.QUBOMatrix``, or ``qubovert.utils.PUBOMatrix`` object,
        then ``type(P) == type(variables[0])``. Otherwise,
        ``type(P) == type(variables[0])``.

    Example
    -------
    >>> from qubovert.sat import AND
    >>> P = AND(0, 1)
    >>> P
    {(0, 1): 1}
    >>> P.value({0: 0, 1: 0})
    0
    >>> P.value({0: 0, 1: 1})
    0
    >>> P.value({0: 1, 1: 0})
    0
    >>> P.value({0: 1, 1: 1})
    1
    >>> type(P)
    qubovert._pubo.PUBO

    >>> P = AND({(0, 1): 1}, 'x')  # and of 0, 1, and 'x'.
    >>> P
    {(0, 1, 'x'): 1}
    >>> type(P)
    qubovert._pubo.PUBO

    >>> from qubovert import boolean_var
    >>> x, y = boolean_var('x'), boolean_var('y')
    >>> P = AND(x, y)
    >>> type(P)
    qubovert.PCBO

    """
    if not variables:
        P = qv.PUBO() + 1
    else:
        P = 1
        for v in variables:
            P *= BUFFER(v)
    return P


def NAND(*variables):
    """NAND.

    Return the boolean expression for the NAND of the variables. Equivalent
    to ``NOT(AND(*variables))``

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the boolean variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object or same type as ``type(variables[0])``.
        The boolean expression for the logic operation.
        If ``variables[0]`` is a ``qubovert.QUBO``, ``qubovert.PCBO``,
        ``qubovert.utils.QUBOMatrix``, or ``qubovert.utils.PUBOMatrix`` object,
        then ``type(P) == type(variables[0])``. Otherwise,
        ``type(P) == type(variables[0])``.

    Example
    -------
    >>> from qubovert.sat import NAND
    >>> P = NAND(0, 1)
    >>> P
    {(): 1, (0, 1): -1}
    >>> P.value({0: 0, 1: 0})
    1
    >>> P.value({0: 0, 1: 1})
    1
    >>> P.value({0: 1, 1: 0})
    1
    >>> P.value({0: 1, 1: 1})
    0
    >>> type(P)
    qubovert._pubo.PUBO

    >>> P = NAND({(0, 1): 1}, 'x')  # nand of 0, 1, and 'x'.
    >>> P
    {(): 1, (0, 1, 'x'): -1}
    >>> type(P)
    qubovert._pubo.PUBO

    >>> from qubovert import boolean_var
    >>> x, y = boolean_var('x'), boolean_var('y')
    >>> P = NAND(x, y)
    >>> type(P)
    qubovert.PCBO

    """
    return NOT(AND(*variables))


def OR(*variables):
    """OR.

    Return the boolean expression for the OR of the variables.

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the boolean variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object or same type as ``type(variables[0])``.
        The boolean expression for the logic operation.
        If ``variables[0]`` is a ``qubovert.QUBO``, ``qubovert.PCBO``,
        ``qubovert.utils.QUBOMatrix``, or ``qubovert.utils.PUBOMatrix`` object,
        then ``type(P) == type(variables[0])``. Otherwise,
        ``type(P) == type(variables[0])``.

    Example
    -------
    >>> from qubovert.sat import OR
    >>> P = OR(0, 1)
    >>> P
    {(0,): 1, (0, 1): -1, (1,): 1}
    >>> P.value({0: 0, 1: 0})
    0
    >>> P.value({0: 0, 1: 1})
    1
    >>> P.value({0: 1, 1: 0})
    1
    >>> P.value({0: 1, 1: 1})
    1
    >>> type(P)
    qubovert._pubo.PUBO

    >>> P = OR({(0, 1): 1}, 'x')  # or of 0, 1, and 'x'.
    >>> P
    {(0, 1): 1, (0, 1, 'x'): -1, ('x',): 1}
    >>> type(P)
    qubovert._pubo.PUBO

    >>> from qubovert import boolean_var
    >>> x, y = boolean_var('x'), boolean_var('y')
    >>> P = OR(x, y)
    >>> type(P)
    qubovert.PCBO

    """
    if not variables:
        return qv.PUBO() + 1
    elif len(variables) == 1:
        return BUFFER(variables[0])
    x, v = OR(*variables[:-1]), BUFFER(variables[-1])
    return x + v * (1 - x)


def NOR(*variables):
    """NOR.

    Return the boolean expression for the OR of the variables. Equivalent to
    ``NOT(OR(*variables))``.

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the boolean variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object or same type as ``type(variables[0])``.
        The boolean expression for the logic operation.
        If ``variables[0]`` is a ``qubovert.QUBO``, ``qubovert.PCBO``,
        ``qubovert.utils.QUBOMatrix``, or ``qubovert.utils.PUBOMatrix`` object,
        then ``type(P) == type(variables[0])``. Otherwise,
        ``type(P) == type(variables[0])``.

    Example
    -------
    >>> from qubovert.sat import NOR
    >>> P = NOR(0, 1)
    >>> P
    {(0,): -1, (0, 1): 1, (1,): -1, (): 1}
    >>> P.value({0: 0, 1: 0})
    1
    >>> P.value({0: 0, 1: 1})
    0
    >>> P.value({0: 1, 1: 0})
    0
    >>> P.value({0: 1, 1: 1})
    0
    >>> type(P)
    qubovert._pubo.PUBO

    >>> P = NOR({(0, 1): 1}, 'x')  # nor of 0, 1, and 'x'.
    >>> P
    {(0, 1): -1, (0, 1, 'x'): 1, ('x',): -1, (): 1}
    >>> type(P)
    qubovert._pubo.PUBO

    >>> from qubovert import boolean_var
    >>> x, y = boolean_var('x'), boolean_var('y')
    >>> P = NOR(x, y)
    >>> type(P)
    qubovert.PCBO

    """
    return NOT(OR(*variables))


def XOR(*variables):
    """XOR.

    Return the boolean expression for the XOR of the variables. XOR(a, b) is 1
    if ``a == b``, otherwise it is 0. ``qubovert`` uses the convention that
    an XOR on > 2 bits is a parity gate. Ie
    ``XOR(0, 1, 2, 3) == XOR(XOR(XOR(0, 1), 2), 3)``.

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the boolean variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object or same type as ``type(variables[0])``.
        The boolean expression for the logic operation.
        If ``variables[0]`` is a ``qubovert.QUBO``, ``qubovert.PCBO``,
        ``qubovert.utils.QUBOMatrix``, or ``qubovert.utils.PUBOMatrix`` object,
        then ``type(P) == type(variables[0])``. Otherwise,
        ``type(P) == type(variables[0])``.

    Example
    -------
    >>> from qubovert.sat import XOR
    >>> P = XOR(0, 1)
    >>> P
    {(0,): 1, (0, 1): -2, (1,): 1}
    >>> P.value({0: 0, 1: 0})
    0
    >>> P.value({0: 0, 1: 1})
    1
    >>> P.value({0: 1, 1: 0})
    1
    >>> P.value({0: 1, 1: 1})
    0
    >>> type(P)
    qubovert._pubo.PUBO

    >>> P = XOR({(0, 1): 1}, 'x')  # xor of 0, 1, and 'x'.
    >>> P
    {(0, 1): 1, (0, 1, 'x'): -2, ('x',): 1}
    >>> type(P)
    qubovert._pubo.PUBO

    >>> from qubovert import boolean_var
    >>> x, y = boolean_var('x'), boolean_var('y')
    >>> P = XOR(x, y)
    >>> type(P)
    qubovert.PCBO

    The following test will pass.

    >>> for n in range(1, 5):
    >>>     P = XOR(*tuple(range(n)))
    >>>     for i in range(1 << n):
    >>>         sol = decimal_to_boolean(i, n)
    >>>         if sum(sol) % 2 == 1:
    >>>             assert P.value(sol) == 1
    >>>         else:
    >>>             assert not P.value(sol)

    """
    if not variables:
        return qv.PUBO() + 1
    elif len(variables) == 1:
        return BUFFER(variables[0])
    x, v = XOR(*variables[:-1]), BUFFER(variables[-1])
    return (x - v) ** 2


def XNOR(*variables):
    """XNOR.

    Return the boolean expression for the XNOR of the variables. Equivalent to
    ``NOT(XOR(*variables))``.

    Note the following convention.

    XOR(a, b) is 1 if ``a == b``, otherwise it is 0. ``qubovert`` uses the
    convention that an XOR on > 2 bits is a parity gate. Ie
    ``XOR(0, 1, 2, 3) == XOR(XOR(XOR(0, 1), 2), 3)``.

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the boolean variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object or same type as ``type(variables[0])``.
        The boolean expression for the logic operation.
        If ``variables[0]`` is a ``qubovert.QUBO``, ``qubovert.PCBO``,
        ``qubovert.utils.QUBOMatrix``, or ``qubovert.utils.PUBOMatrix`` object,
        then ``type(P) == type(variables[0])``. Otherwise,
        ``type(P) == type(variables[0])``.

    Example
    -------
    >>> from qubovert.sat import XNOR
    >>> P = XNOR(0, 1)
    >>> P
    {(): 1, (0,): -1, (0, 1): 2, (1,): -1}
    >>> P.value({0: 0, 1: 0})
    1
    >>> P.value({0: 0, 1: 1})
    0
    >>> P.value({0: 1, 1: 0})
    0
    >>> P.value({0: 1, 1: 1})
    1
    >>> type(P)
    qubovert._pubo.PUBO

    >>> P = XNOR({(0, 1): 1}, 'x')  # xnor of 0, 1, and 'x'.
    >>> P
    {(): 1, (0, 1): -1, (0, 1, 'x'): 2, ('x',): -1}
    >>> type(P)
    qubovert._pubo.PUBO

    >>> from qubovert import boolean_var
    >>> x, y = boolean_var('x'), boolean_var('y')
    >>> P = XNOR(x, y)
    >>> type(P)
    qubovert.PCBO

    The following test will pass.

    >>> for n in range(1, 5):
    >>>     P = XNOR(*tuple(range(n)))
    >>>     for i in range(1 << n):
    >>>         sol = decimal_to_boolean(i, n)
    >>>         if sum(sol) % 2 == 1:
    >>>             assert not P.value(sol)
    >>>         else:
    >>>             assert P.value(sol) == 1

    """
    return NOT(XOR(*variables))
