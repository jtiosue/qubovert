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

"""_solve_bruteforce.py.

This file contains bruteforce solvers for QUBO/PUBO and QUSO/PUSO, as well
as QUBO/PUBO and PUSO objective function evaluators.

"""

__all__ = (
    'boolean_to_spin', 'spin_to_boolean',
    'decimal_to_spin', 'spin_to_decimal',
    'decimal_to_boolean', 'boolean_to_decimal',
    'pubo_value', 'qubo_value', 'puso_value', 'quso_value',
    'solve_pubo_bruteforce', 'solve_qubo_bruteforce',
    'solve_puso_bruteforce', 'solve_quso_bruteforce'
)


def boolean_to_spin(x):
    """boolean_to_spin.

    Convert a boolean number in {0, 1} to a spin in {1, -1}, in that order.

    Parameters
    ----------
    x : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 0 or 1.

    Returns
    -------
    z : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 1 or -1.

    Example
    -------
    >>> boolean_to_spin(0)  # will print 1
    >>> boolean_to_spin(1)  # will print -1
    >>> boolean_to_spin([0, 1, 1])  # will print [1, -1, -1]
    >>> boolean_to_spin({"a": 0, "b": 1})  # will print {"a": 1, "b": -1}

    """
    convert = {0: 1, 1: -1}
    if isinstance(x, (int, float)) and x in convert:
        return convert[x]
    elif isinstance(x, dict):
        return {k: convert[v] for k, v in x.items()}
    return type(x)(convert[i] for i in x)


def spin_to_boolean(z):
    """spin_to_boolean.

    Convert a spin in {1, -1} to a boolean variable in {0, 1}, in that order.

    Parameters
    ----------
    z : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 1 or -1.

    Returns
    -------
    x : int, iterable of ints, or dict mapping labels to ints.
        Each integer is either 0 or 1.

    Example
    -------
    >>> spin_to_boolean(-1)  # will print 1
    >>> spin_to_boolean(1)  # will print 0
    >>> spin_to_boolean([-1, 1, 1])  # will print [1, 0, 0]
    >>> spin_to_boolean({"a": -1, "b": 1})  # will print {"a": 1, "b": 0}

    """
    convert = {-1: 1, 1: 0}
    if isinstance(z, (int, float)) and z in convert:
        return convert[z]
    elif isinstance(z, dict):
        return {k: convert[v] for k, v in z.items()}
    return type(z)(convert[i] for i in z)


def decimal_to_boolean(d, num_bits=None):
    """decimal_to_boolean.

    Convert the integer ``d`` to its boolean representation.

    Parameters
    ----------
    d : int >= 0.
        Number to convert to binary.
    num_bits : int >= 0 (optional, defaults to None).
        Number of bits in the representation. If ``num_bits is None``, then
        the minimum number of bits required will be used.

    Returns
    -------
    b : tuple of length ``num_bits``.
        Each element of ``b`` is a 0 or 1.

    Example
    -------
    >>> decimal_to_boolean(10, 7)
    (0, 0, 0, 1, 0, 1, 0)

    >>> decimal_to_boolean(10)
    (1, 0, 1, 0)

    """
    if int(d) != d or d < 0:
        raise ValueError("``d`` must be an integer >- 0.")
    b = bin(d)[2:]
    lb = len(b)
    if num_bits is None:
        num_bits = lb
    elif num_bits < lb:
        raise ValueError("Not enough bits to represent the number.")
    return (0,) * (num_bits - lb) + tuple(int(x) for x in b)


def boolean_to_decimal(b):
    """boolean_to_decimal.

    Convert a bit string to its decimal form.

    Parameters
    ----------
    b : tuple or list of 0s and 1s.
        The binary bit string.

    Returns
    -------
    d : int.

    Examples
    --------
    >>> boolean_to_decimal((1, 1, 0))
    6

    """
    return int("".join(str(x) for x in b), base=2) if b else 0


def decimal_to_spin(d, num_spins=None):
    """decimal_to_spin.

    Convert the integer ``d`` to its spin representation (ie its binary
    representation, but with 1 and -1 instead of 0 and 1).

    Parameters
    ----------
    d : int >= 0.
        Number to convert to binary.
    num_spins : int >= 0 (optional, defaults to None).
        Number of bits in the representation. If ``num_spins is None``, then
        the minimum number of bits required will be used.

    Returns
    -------
    b : tuple of length ``num_spins``.
        Each element of ``b`` is a 0 or 1.

    Example
    -------
    >>> decimal_to_spin(10, 7)
    (1, 1, 1, -1, 1, -1, 1)

    >>> decimal_to_spin(10)
    (-1, 1, -1, 1)

    """
    return boolean_to_spin(decimal_to_boolean(d, num_spins))


def spin_to_decimal(b):
    """spin_to_decimal.

    Convert a spin string to its decimal form.

    Parameters
    ----------
    b : tuple or list of 1s and -1s.
        The spin bit string.

    Returns
    -------
    d : int.

    Examples
    --------
    >>> spin_to_decimal((-1, -1, 1))
    6

    """
    return boolean_to_decimal(spin_to_boolean(b))


def pubo_value(x, P):
    r"""pubo_value.

    Find the value of
    :math:`\sum_{i,...,j} P_{i...j} x_{i} ... x_{j}`

    Parameters
    ----------
    x : dict or iterable.
        Maps boolean variable indices to their boolean values, 0 or 1. Ie
        ``x[i]`` must be the boolean value of variable i.
    P : dict, qubovert.utils.PUBOMatrix, or qubovert.PUBO object.
        Maps tuples of boolean variables indices to the P value.

    Returns
    -------
    value : float.
        The value of the PUBO with the given assignment `x`. Ie

    Example
    -------
    >>> P = {(0, 0): 1, (0, 1): -1}
    >>> x = {0: 1, 1: 0}
    >>> pubo_value(x, P)
    1

    """
    return sum(v for k, v in P.items() if all(x[i] for i in k))


def qubo_value(x, Q):
    r"""qubo_value.

    Find the value of
    :math:`\sum_{i,j} Q_{ij} x_{i} x_{j}`

    Parameters
    ----------
    x : dict or iterable.
        Maps boolean variable indices to their boolean values, 0 or 1. Ie
        ``x[i]`` must be the boolean value of variable i.
    Q : dict or qubovert.utils.QUBOMatrix object.
        Maps tuples of boolean variables indices to the Q value.

    Returns
    -------
    value : float.
        The value of the QUBO with the given assignment `x`. Ie

        >>> sum(
                Q[(i, j)] * x[i] * x[j]
                for i in range(n) for j in range(n)
            )

    Example
    -------
    >>> Q = {(0, 0): 1, (0, 1): -1}
    >>> x = {0: 1, 1: 0}
    >>> qubo_value(x, Q)
    1

    """
    return pubo_value(x, Q)


def puso_value(z, H):
    r"""puso_value.

    Find the value of
        :math:`\sum_{i,...,j} H_{i...j} z_{i} ... z_{j}`.

    Parameters
    ----------
    z: dict or iterable.
        Maps variable labels to their values, 1 or -1. Ie z[i] must be the
        value of variable i.
    H : dict, qubovert.utils.PUSOMatrix, or qubovert.PUSO object.
        Maps spin labels to values.

    Returns
    -------
    value : float.
        The value of the PUSO with the given assignment `z`.

    Example
    -------
    >>> H = {(0, 1): -1, (0,): 1}
    >>> z = {0: -1, 1: 1}
    >>> puso_value(z, H)
    0

    """
    return sum(
        v * pow(-1, [z[i] for i in k].count(-1) % 2)
        for k, v in H.items()
    )


def quso_value(z, L):
    r"""quso_value.

    Find the value of
        :math:`\sum_{i,j} J_{ij} z_{i} z_{j} + \sum_{i} h_{i} z_{i}`.

    The J's are encoded by keys with pairs of labels in L, and the h's are
    encoded by keys with a single label in L.

    Parameters
    ----------
    z: dict or iterable.
        Maps variable labels to their values, 1 or -1. Ie z[i] must be the
        value of variable i.
    L : dict, qubovert.utils.QUSOMatrix, or qubovert.QUSO object.
        Maps spin labels to values.

    Returns
    -------
    value : float.
        The value of the QUSO with the given assignment `z`.

    Example
    -------
    >>> L = {(0, 1): -1, (0,): 1}
    >>> z = {0: -1, 1: 1}
    >>> quso_value(z, L)
    0

    """
    return puso_value(z, L)


def _solve_bruteforce(D, all_solutions, valid, spin):
    """_solve_bruteforce.

    Helper function for solve_pubo_bruteforce and solve_puso_bruteforce.

    Iterate through all the possible solutions to a BO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    D : dict.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the problem
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    valid : function.
        ``valid`` takes in a bitstring or spinstring and outputs a boolean
        indicating whether that bitstring or spinstring is a valid solutions.
    spin : bool.
        Whether we're bruteforce solving a spin model or boolean model.

    Returns
    -------
    res : tuple (objective, solution).

        if all_solutions is False:
            objective : float.
                The best value of the problem.
            solution : dict.
                Maps the binary variable label to its solution value,
                {0, 1} if not spin else {-1, 1}.

        if all_solutions is True:
            objective : float.
                The best value of the problem
            solution : list of dicts.
                Each dictionary maps the label to the value of each binary
                variable. Ie each ``s`` in ``solution`` is a solution that
                gives the best objective function value.

    """
    if not D:
        return 0, ({} if not all_solutions else [{}])
    elif () in D:
        offset = D.pop(())
        if not D:
            D[()] = offset
            return offset, ({} if not all_solutions else [{}])
        D[()] = offset

    # if D is a Matrix object or QUBO, PUBO, etc, then these are defined
    try:
        N = D.num_binary_variables
        # could do D.reverse_mapping, but that creates a copy. We just need to
        # not mutate it here, then we don't have to waste time copying.
        mapping = D._reverse_mapping
    except AttributeError:
        var = set()
        for x in D:
            var.update(set(x))
        N = len(var)

        # map qubit name to 0 through N-1
        mapping = dict(enumerate(var))

    best = None, {}
    all_sols = {None: [{}]}
    convert = decimal_to_spin if spin else decimal_to_boolean

    for n in range(1 << N):
        test_sol = convert(n, N)
        x = {mapping[i]: v for i, v in enumerate(test_sol)}
        if not valid(x):
            continue
        v = puso_value(x, D) if spin else pubo_value(x, D)
        if all_solutions and (best[0] is None or v <= best[0]):
            best = v, x
            all_sols.setdefault(v, []).append(x)
        elif best[0] is None or v < best[0]:
            best = v, x

    if all_solutions:
        best = best[0], all_sols[best[0]]

    return best


def solve_pubo_bruteforce(P, all_solutions=False, valid=lambda x: True):
    """solve_pubo_bruteforce.

    Iterate through all the possible solutions to a PUBO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    P : dict, qubovert.utils.PUBOMatrix, or qubovert.PUBO object.
        Maps boolean variables labels to the P value.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the PUBO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    valid : function (optional, defaults to ``lambda x: True``).
        ``valid`` takes in a bitstring and outputs a boolean
        indicating whether that bitstring is a valid solutions.

    Returns
    -------
    res : tuple (objective, solution).

        if all_solutions is False:
            objective : float.
                The best value of the PUBO.
            solution : dict.
                Maps the boolean variable label to its solution value, {0, 1}.

        if all_solutions is True:
            objective : float.
                The best value of the PUBO.
            solution : list of dicts.
                Each dictionary maps the label to the value of each boolean
                variable. Ie each ``s`` in ``solution`` is a solution that
                gives the best objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = x_0x_1 + x_1x_2 - x_1 - 2x_2`,
    the P dictionary is

    >>> P = {(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2}

    Then to solve this P, run

    >>> obj_val, solution = solve_pubo_bruteforce(Q)
    >>> obj_val
    -2
    >>> solution
    {0: 0, 1: 0, 2: 1}

    ``obj_val`` will be the smallest value of ``obj``.
    ``solution`` will be a dictionary that indicates what each of :math:`x_0`,
    :math:`x_1`, and :math:`x_2` are for the solution. In this case,
    ``x = {0: 0, 1: 0, 2: 1}``, indicating that :math:`x_0` is 0, :math:`x_1`
    is 0, :math:`x_2` is 1.

    """
    return _solve_bruteforce(P, all_solutions, valid, False)


def solve_qubo_bruteforce(Q, all_solutions=False, valid=lambda x: True):
    """solve_qubo_bruteforce.

    Iterate through all the possible solutions to a QUBO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    Q : dict, qubovert.utils.QUBOMatrix, or qubovert.QUBO object.
        Maps boolean variables labels to the Q value.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the QUBO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    valid : function (optional, defaults to ``lambda x: True``).
        ``valid`` takes in a bitstring and outputs a boolean
        indicating whether that bitstring is a valid solutions.

    Returns
    -------
    res : tuple (objective, solution).

        if all_solutions is False:
            objective : float.
                The best value of the QUBO.
            solution : dict.
                Maps the boolean variable label to its solution value, {0, 1}.

        if all_solutions is True:
            objective : float.
                The best value of the QUBO. Equal to
                ``sum(Q[(i, j)] * solution[x][i] * solution[x][j]) + offset``
                where `solution[x]` is one of the solutions to the QUBO.
            solution : list of dicts.
                Each dictionary maps the label to the value of each boolean
                variable. Ie each ``s`` in ``solution`` is a solution that
                gives the best objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = x_0x_1 + x_1x_2 - x_1 - 2x_2`,
    the P dictionary is

    >>> Q = {(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2}

    Then to solve this Q, run

    >>> obj_val, solution = solve_qubo_bruteforce(Q)
    >>> obj_val
    -2
    >>> solution
    {0: 0, 1: 0, 2: 1}

    ``obj_val`` will be the smallest value of ``obj``.
    ``solution`` will be a dictionary that indicates what each of :math:`x_0`,
    :math:`x_1`, and :math:`x_2` are for the solution. In this case,
    ``x = {0: 0, 1: 0, 2: 1}``, indicating that :math:`x_0` is 0, :math:`x_1`
    is 0, :math:`x_2` is 1.

    """
    return solve_pubo_bruteforce(Q, all_solutions, valid)


def solve_puso_bruteforce(H, all_solutions=False, valid=lambda x: True):
    """solve_puso_bruteforce.

    Iterate through all the possible solutions to an PUSO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    H : dict, qubovert.utils.PUSOMatrix, or qubovert.PUSO object.
        Maps spin labels to values.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the QUSO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    valid : function (optional, defaults to ``lambda x: True``).
        ``valid`` takes in a spinstring and outputs a boolean
        indicating whether that spinstring is a valid solutions.

    Returns
    -------
    res : tuple (objective, solution).

        if ``all_solutions`` is False:
            objective : float.
                The best value of the PUSO.
            solution : dict.
                Maps the spin variable label to its solution value, {-1, 1}.

        if ``all_solutions`` is True:
            objective : float.
                The best value of the PUSO.
            solution : list of dicts.
                Each dictionary maps the label to the value of each variable.
                Ie each ``s`` in ``solution`` is a solution that gives the best
                objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = z_0z_1 + z_1z_2 - z_1 - 2z_2`,
    we have

    >>> H = {(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2}

    Then to solve this, run

    >>> obj_val, solution = solve_puso_bruteforce(H)
    >>> obj_val
    -3
    >>> solution
    {0: 1, 1: -1, 2: 1}

    ``obj_val`` will be the smallest value of ``obj``.
    ``solution`` will be a dictionary that indicates what each of
    :math:`z_0, z_1`, and :math:`z_2` are for the solution. In this case,
    ``z = {0: 1, 1: -1, 2: 1}``, indicating that :math:`z_0` is 1, :math:`z_1`
    is -1, :math:`z_2` is 1.

    """
    return _solve_bruteforce(H, all_solutions, valid, True)


def solve_quso_bruteforce(L, all_solutions=False, valid=lambda x: True):
    """solve_quso_bruteforce.

    Iterate through all the possible solutions to an QUSO formulated problem
    and find the best one (the one that gives the minimum objective value). Do
    not use for large problem sizes! This is meant only for testing very small
    problems.

    Parameters
    ----------
    L : dict, qubovert.utils.QUSOMatrix, or qubovert.QUSO object.
        Maps spin labels to values.
    all_solutions : boolean (optional, defaults to False).
        If all_solutions is set to True, all the best solutions to the QUSO
        will be returned rather than just one of the best. If the problem is
        very big, then it is best if ``all_solutions`` is False, otherwise this
        function will use a lot of memory.
    valid : function (optional, defaults to ``lambda x: True``).
        ``valid`` takes in a spinstring and outputs a boolean
        indicating whether that spinstring is a valid solutions.

    Returns
    -------
    res : tuple (objective, solution).

        if ``all_solutions`` is False:
            objective : float.
                The best value of the QUSO.
            solution : dict.
                Maps the spin variable label to its solution value, {-1, 1}.

        if ``all_solutions`` is True:
            objective : float.
                The best value of the PUSO.
            solution : list of dicts.
                Each dictionary maps the label to the value of each variable.
                Ie each ``s`` in ``solution`` is a solution that gives the best
                objective function value.

    Example
    -------
    To find the minimum of the problem
    :math:`obj = z_0z_1 + z_1z_2 - z_1 - 2z_2`,
    we have

    >>> L = {(0, 1): 1, (1, 2): 1, (1,): -1, (2,): -2}

    Then to solve this, run

    >>> obj_val, solution = solve_quso_bruteforce(L)
    >>> obj_val
    -3
    >>> solution
    {0: 1, 1: -1, 2: 1}

    ``obj_val`` will be the smallest value of ``obj``.
    ``solution`` will be a dictionary that indicates what each of
    :math:`z_0, z_1`, and :math:`z_2` are for the solution. In this case,
    ``z = {0: 1, 1: -1, 2: 1}``, indicating that :math:`z_0` is 1, :math:`z_1`
    is -1, :math:`z_2` is 1.

    """
    return solve_puso_bruteforce(L, all_solutions, valid)
