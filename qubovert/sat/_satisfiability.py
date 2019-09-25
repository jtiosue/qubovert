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

"""_satisfiability.py.

This file contains functions for converting SAT problems to PUBOs.

"""

from qubovert import PUBO


__all__ = "ONE", "NOT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR"


def ONE(x):
    """ONE.

    Return the boolean expression for the buffer of ``x``.

    Parameters
    ----------
    x : hashable object or dict (or subclass of dict).
        The expression to buffer.

    Return
    ------
    P : ``qubovert.PUBO`` object.
        The binary expression for the logic operation.

    Example
    -------
    >>> from qubovert.sat import ONE
    >>> P = ONE('x')
    >>> P
    {('x',): 1}
    >>> P.value({'x': 1})
    1
    >>> P.value({'x': 0})
    0

    >>> P = ONE({(0, 1): 1})
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


    """
    return PUBO(x) if isinstance(x, dict) else PUBO({(x,): 1})


def NOT(x):
    """NOT.

    Return the boolean expression for the NOT of ``x``.

    Parameters
    ----------
    x : hashable object or dict (or subclass of dict).
        The expression to buffer.

    Return
    ------
    P : ``qubovert.PUBO`` object.
        The binary expression for the logic operation.

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

    """
    return 1 - ONE(x)


def AND(*variables):
    """AND.

    Return the boolean expression for the AND of the variables.

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the binary variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object.
        The binary expression for the logic operation.

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

    >>> P = AND({(0, 1): 1}, 'x')  # and of 0, 1, and 'x'.
    >>> P
    {(0, 1, 'x'): 1}

    """
    P = PUBO() + 1
    for v in variables:
        P *= ONE(v)
    return P


def NAND(*variables):
    """NAND.

    Return the boolean expression for the NAND of the variables. Equivalent
    to ``NOT(AND(*variables))``

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the binary variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object.
        The binary expression for the logic operation.

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

    >>> P = NAND({(0, 1): 1}, 'x')  # nand of 0, 1, and 'x'.
    >>> P
    {(): 1, (0, 1, 'x'): -1}

    """
    return NOT(AND(*variables))


def OR(*variables):
    """OR.

    Return the boolean expression for the OR of the variables.

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the binary variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object.
        The binary expression for the logic operation.

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

    >>> P = OR({(0, 1): 1}, 'x')  # or of 0, 1, and 'x'.
    >>> P
    {(0, 1): 1, (0, 1, 'x'): -1, ('x',): 1}

    """
    if not variables:
        return PUBO() + 1
    elif len(variables) == 1:
        return ONE(variables[0])
    x, v = OR(*variables[:-1]), ONE(variables[-1])
    return x + v * (1 - x)


def NOR(*variables):
    """NOR.

    Return the boolean expression for the OR of the variables. Equivalent to
    ``NOT(OR(*variables))``.

    Parameters
    ----------
    *variables : arguments.
        ``variables`` can be of arbitrary length. Each variable can be a
        hashable object, which is the label of the binary variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object.
        The binary expression for the logic operation.

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

    >>> P = NOR({(0, 1): 1}, 'x')  # nor of 0, 1, and 'x'.
    >>> P
    {(0, 1): -1, (0, 1, 'x'): 1, ('x',): -1, (): 1}

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
        hashable object, which is the label of the binary variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object.
        The binary expression for the logic operation.

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

    >>> P = XOR({(0, 1): 1}, 'x')  # xor of 0, 1, and 'x'.
    >>> P
    {(0, 1): 1, (0, 1, 'x'): -2, ('x',): 1}

    The following test will pass.

    >>> for n in range(1, 5):
    >>>     P = XOR(*tuple(range(n)))
    >>>     for i in range(1 << n):
    >>>         sol = decimal_to_binary(i, n)
    >>>         if sum(sol) % 2 == 1:
    >>>             assert P.value(sol) == 1
    >>>         else:
    >>>             assert not P.value(sol)

    """
    if not variables:
        return PUBO() + 1
    elif len(variables) == 1:
        return ONE(variables[0])
    x, v = XOR(*variables[:-1]), ONE(variables[-1])
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
        hashable object, which is the label of the binary variable, or a dict
        (or subclass of dict) representing a boolean expression.

    Return
    ------
    P : ``qubovert.PUBO`` object.
        The binary expression for the logic operation.

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

    >>> P = XNOR({(0, 1): 1}, 'x')  # xnor of 0, 1, and 'x'.
    >>> P
    {(): 1, (0, 1): -1, (0, 1, 'x'): 2, ('x',): -1}

    The following test will pass.

    >>> for n in range(1, 5):
    >>>     P = XNOR(*tuple(range(n)))
    >>>     for i in range(1 << n):
    >>>         sol = decimal_to_binary(i, n)
    >>>         if sum(sol) % 2 == 1:
    >>>             assert not P.value(sol)
    >>>         else:
    >>>             assert P.value(sol) == 1

    """
    return NOT(XOR(*variables))
