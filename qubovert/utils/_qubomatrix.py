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

"""_qubomatrix.py.

This file contains the QUBOMatrix object.

"""

import numpy as np
from . import PUBOMatrix, qubo_value, solve_qubo_bruteforce


__all__ = 'QUBOMatrix', 'matrix_to_qubo', 'qubo_to_matrix'


class QUBOMatrix(PUBOMatrix):
    """QUBOMatrix.

    ``QUBOMatrix`` inherits some methods from ``PUBOMatrix``, see
    ``help(qubovert.utils.PUBOMattrix)``.

    A class to handle QUBO matrices. It is the same thing as a dictionary
    with some methods modified. Note that each key must be a tuple of two
    integers >= 0.

    One method is that values will always default to 0. Consider the following
    example:

    >>> d = QUBOMatrix()
    >>> print(d[(0,)]) # will print 0
    >>> d[(0,)] += 1
    >>> print(d) # will print {(0,): 1}

    Compared to an ordinary dictionary.

    >>> g = dict()
    >>> print(g[(0,)]) # will raise KeyError
    >>> g[(0,)] += 1 # will raise KeyError, since (0,) was never set

    One method of QUBOMatrix is that it will always keep the QUBO
    upper triangular! Consider the following example:

    >>> d = QUBOMatrix()
    >>> d[(1, 0)] += 2
    >>> print(d)
    >>> # will print {(0, 1): 2}

    One method is that if we set an item to 0, it will be removed. Consider
    the following example:

    >>> d = QUBOMatrix()
    >>> d[(0,)] += 1
    >>> d[(0,)] -= 1
    >>> print(d) # will print {}

    One method is that if we initialize QUBOMatrix with a previous dictionary
    it will be reinitialized to ensure that the QUBOMatrix is upper
    triangular and contains no zero values. Consider the following example:

    >>> d = QUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0})
    >>> print(d) # will print {(0, 0): 1, (0, 1): 2}

    We also change the update method so that it follows all the conventions.

    >>> d = QUBOMatrix({(0, 0): 1, (0, 1): 2})
    >>> d.update({(0, 0): 0, (1, 0): 1, (1, 1): -1})
    >>> print(d)  # will print {(0, 1): 1, (1,): -1}

    We also include arithmetic, addition, subtraction, scalar division,
    multiplication, and all those in place. For example,

    >>> d = QUBOMatrix((0, 0)=1, (0, 1)=-2)
    >>> g = d + {(0, 0): -1}
    >>> print(g) # will print {(0, 1): -2}
    >>> g *= 4
    >>> print(g) # will print {(0, 1): -8}
    >>> g -= {(0, 1): -8}
    >>> print(g) # will print {}

    >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
    >>> g = {(0,): -1, (1,): 1}
    >>> d *= g
    >>> print(d)
    {(0,): -1, (0, 1): 1}

    >>> d = QUBOMatrix({(0, 0): 1, (0, 1): -1})
    >>> print(d ** 2 == d * d)
    True

    Adding or subtracting constants will update the () element of the
    dict.

    >>> d = QUBOMatrix()
    >>> d += 5
    >>> print(d)
    {(): 5}

    Finally, if you try to access a key out of order, it will sort and squash
    the key. Be careful with this, it can cause unexpected behavior if you
    don't know it. For example,

    >>> d = QUBOMatrix()
    >>> d[(0, 1)] += 2
    >>> print(d[(1, 0)])  # will print 2

    >>> d = QUBOMatrix()
    >>> d[(0, 0)] += 2
    >>> print(d[(0,)])  # will print 2
    >>> print(d[(0, 0)])  # will print 2
    >>> print(d)  # will print {(0,): 2}

    """

    @staticmethod
    def _check_key_valid(key):
        """_check_key_valid.

        Internal method to check if an input key to the dictionary is valid.
        Checks to see if ``key`` is a tuple of non negative integers with <=
        2 unique integers.

        Parameters
        ----------
        key : anything, but must be a tuple to be valid.

        Returns
        -------
        k : the squashed key.

        Raises
        ------
        KeyError if the key is invalid.

        """
        k = PUBOMatrix.squash_key(key)
        if len(k) > 2:
            raise KeyError(
                "Key formatted incorrectly, must be tuple of <= 2 integers "
                "See PUBOMatrix instead.")
        return k

    def value(self, x):
        r"""value.

        Find the value of the QUBO. Calling
        ``self.value(x)`` is the same as calling
        ``qubovert.utils.qubo_value(x, self)``.

        Parameters
        ----------
        x : dict or iterable.
            Maps boolean variable indices to their boolean values, 0 or 1. Ie
            ``x[i]`` must be the boolean value of variable i.

        Return
        ------
        value : float.
            The value of the QUBO with the given assignment `x`. Ie

        Example
        -------
        >>> from qubovert.utils import QUBOMatrix, PUBOMatrix
        >>> from qubovert import QUBO, PUBO

        >>> P = PUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> x = {0: 1, 1: 0}
        >>> P.value(x)
        1

        >>> Q = QUBOMatrix({(0, 0): 1, (0, 1): -1})
        >>> x = {0: 1, 1: 0}
        >>> Q.value(x)
        1

        >>> P = PUBO({(0, 0): 1, (0, 1): -1})
        >>> x = {0: 1, 1: 0}
        >>> P.value(x)
        1

        >>> Q = QUBO({(0, 0): 1, (0, 1): -1})
        >>> x = {0: 1, 1: 0}
        >>> Q.value(x)
        1

        """
        return qubo_value(x, self)

    def solve_bruteforce(self, all_solutions=False):
        """solve_bruteforce.

        Solve the problem bruteforce. THIS SHOULD NOT BE USED FOR LARGE
        PROBLEMS! This is the exact same as calling
        ``qubovert.utils.solve_qubo_bruteforce(
            self, all_solutions, self.is_solution_valid)[1]``.

        Parameters
        ----------
        all_solutions : bool.
            See the description of the ``all_solutions`` parameter in
            ``qubovert.utils.solve_qubo_bruteforce``.

        Return
        ------
        res : the second element of the two element tuple that is returned from
            ``qubovert.utils.solve_qubo_bruteforce``.

        """
        return solve_qubo_bruteforce(self,
                                     all_solutions, self.is_solution_valid)[1]

    @property
    def Q(self):
        """Q.

        Return a plain dictionary representing the QUBO. Each key is a tuple
        of two integers, ie (1, 1) corresponds to (1,). Note that the offset
        in the QUBOMatrix is ignored (ie the value corresponding to the key
        ()). See the ``offset`` property to access it.

        Returns
        -------
        Q : dict.
            Plain dictionary representing the QUBO in standard form.

        """
        return {k * (3 - len(k)): v for k, v in self.items() if k}


def matrix_to_qubo(matrix):
    r"""matrix_to_qubo.

    Convert a matrix to a QUBO dictionary.

    Parameters
    ----------
    matrix : list of lists or 2-dimensional numpy array.
        ``matrix[i][j]`` is equal to :math:`Q_{ij}`.

    Return
    ------
    Q : qubovert.utils.QUBOMatrix object.
       The upper triangular QUBO dictionary. See
       ``help(qubovert.utils.QUBOMatrix)``.

    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square and two-dimensional")

    Q = QUBOMatrix()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            Q[(i, j)] += matrix[i][j]

    return Q


def qubo_to_matrix(Q, symmetric=False, array=True):
    r"""qubo_to_matrix.

    Convert a QUBO dictionary to its matrix form. The indices of the ``Q``
    dictionary should be integers from 0 to ``n-1``, where there are ``n``
    binary variables in the QUBO problem.

    Parameters
    ----------
    Q : dict or qubovert.utils.QUBOMatrix object.
        Input QUBO dictionary, where ``Q[(i, j)]`` corresponds to
        :math:`Q_{ij}`.
    symmetric : bool (optional, defaults to False).
        Whether the returned matrix should be symmetric or upper-triangular.
        If ``symmetric`` is True, then the matrix will be symmetric, ie
        ``matrix[i][j] == matrix[j][i]``. Otherwise, it will be
        upper-triangular, ie ``marix[i][j] == 0`` if ``i > j``.
    array : bool (optional, defaults to True).
        Whether the returned matrix should be a numpy array or list of lists.
        If ``array`` is True, then it will be a numpy array, otherwise, it
        will be a list of lists.

    Return
    ------
    matrix : numpy array or list of lists.
        The matrix representing the QUBO. See the arguments ``symmetric`` and
        ``array`` for info on the return type of ``matrix``.

    """
    if not Q:
        raise ValueError("QUBO dictionary is empty")
    elif not isinstance(Q, QUBOMatrix):
        Q = QUBOMatrix(Q)

    if Q[()] != 0:
        raise ValueError("QUBO cannot have a constant when converting "
                         "to a matrix")

    matrix = np.zeros((Q.max_index+1,)*2)
    for k, v in Q.items():
        if len(k) == 1:
            matrix[k[0]][k[0]] = v
        elif symmetric:
            i, j = k
            matrix[i][j] = v / 2
            matrix[j][i] = v / 2
        else:
            i, j = k
            matrix[i][j] = v

    if not array:
        return matrix.tolist()
    return matrix
