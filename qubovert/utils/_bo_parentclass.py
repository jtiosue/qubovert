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

"""_bo_parentclass.py.

Contains the BO parent class. See ``help(qubovert.utils.BO)``. Used as a parent
class for some Binary Optimization classes, such as ``qubovert.QUBO``,
``qubovert.QUSO``, etc.

"""

from . import Conversions
from ._dict_arithmetic import _generate_key_value_pairs


__all__ = 'BO',


class BO(Conversions):
    """BO.

    Parent class for some Binary Optimization classes, such as
    ``qubovert.QUBO``, ``qubovert.QUSO``, etc.

    BO inherits some methods and attributes the ``DictArithmetic`` class.
    See ``help(qubovert.utils.DictArithmetic)``.

    BO inherits some methods and attributes the ``Conversions`` class.
    See ``help(qubovert.utils.Conversions)``.

    """

    def __init__(self, *args, **kwargs):
        """__init__.

        This class deals with Binary Optimization models. See child
        classes for info on inputs.

        Parameters
        ----------
        arguments : Defined in child classes.

        """
        self._mapping, self._reverse_mapping, self._next_label = {}, {}, 0

    @property
    def mapping(self):
        """mapping.

        Return a copy of the mapping dictionary that maps the provided
        labels to integers from 0 to n-1, where n is the number of variables
        in the problem.

        Return
        ------
        mapping : dict.
            Dictionary that maps provided labels to integer labels.

        """
        return self._mapping.copy()

    @property
    def reverse_mapping(self):
        """reverse_mapping.

        Return a copy of the reverse_mapping dictionary that maps the integer
        labels to the provided labels. Opposite of ``mapping``.

        Return
        ------
        reverse_mapping : dict.
            Dictionary that maps integer labels to provided labels.

        """
        return self._reverse_mapping.copy()

    @property
    def max_index(self):
        """max_index.

        Return the maximum label of the integer labeled version of the problem.

        Return
        ------
        m : int.

        """
        return self.num_binary_variables - 1

    def set_mapping(self, *args, **kwargs):
        """set_mapping.

        ``BO`` sublcasses automatically create a mapping from variable names to
        integers as they are being built. However, the mapping is based on the
        order in which elements are entered and therefore may not be as
        desired. Of course, the ``convert_solution`` method keeps track of the
        mapping and can/should always be used. But if you want a consistent
        mapping, then ``set_mapping`` can be used.

        Consider the following examples (we use the ``qubovert.QUBO`` class for
        the examples, which is a subclass of ``BO``).

        Example 1:

        >>> from qubovert import QUBO
        >>> Q = QUBO()
        >>> Q[(0,)] += 1
        >>> Q[(1,)] += 2
        >>> Q.mapping
        {0: 0, 1: 1}
        >>> Q.to_qubo()
        {(0,): 1, (1,): 2}

        Example 2:

        >>> from qubovert import QUBO
        >>> Q = QUBO()
        >>> Q[(1,)] += 2
        >>> Q[(0,)] += 1
        >>> Q.mapping
        {0: 1, 1: 0}
        >>> Q.to_qubo()
        {(0,): 2, (1,): 1}

        To ensure consistency in mappings, you can provide your own mapping
        with ``set_mapping``. See the following modified examples.

        Modified example 1:

        >>> from qubovert import QUBO
        >>> Q = QUBO()
        >>> Q[(0,)] += 1
        >>> Q[(1,)] += 2
        >>> Q.set_mapping({0: 0, 1: 1})
        >>> Q.mapping
        {0: 0, 1: 1}
        >>> Q.to_qubo()
        {(0,): 1, (1,): 2}

        Modified example 2:

        >>> from qubovert import QUBO
        >>> Q = QUBO()
        >>> Q[(1,)] += 2
        >>> Q[(0,)] += 1
        >>> Q.set_mapping({0: 0, 1: 1})
        >>> Q.mapping
        {0: 0, 1: 1}
        >>> Q.to_qubo()
        {(0,): 1, (1,): 2}

        Parameters
        ----------
        arguments : defines a dictionary with ``d = dict(*args, **kwargs)``.
            ``d`` will become the mapping. See ``help(self.mapping)``

        Notes
        -----
        Using ``set_mapping`` to set the mapping will also automatically
        set the ``reverse_mapping``, so there is no need to call both
        ``set_mapping`` and ``set_reverse_mapping``.

        """
        self._mapping, self._reverse_mapping = {}, {}
        for k, v in _generate_key_value_pairs(*args, **kwargs):
            self._mapping[k] = v
            self._reverse_mapping[v] = k

    def set_reverse_mapping(self, *args, **kwargs):
        """set_reverse_mapping.

        Same as ``set_mapping`` but reversed. See
        ``help(self.reverse_mapping)`` and ``help(self.set_mapping)``.

        Parameters
        ----------
        arguments : defines a dictionary with ``d = dict(*args, **kwargs)``.
            ``d`` will become the reverse mapping. See
            ``help(self.reverse_mapping)``.

        Notes
        -----
        Using ``set_reverse_mapping`` to set the mapping will also
        automatically set the ``mapping``, so there is no need to call both
        ``set_mapping`` and ``set_reverse_mapping``.

        """
        self._mapping, self._reverse_mapping = {}, {}
        for k, v in _generate_key_value_pairs(*args, **kwargs):
            self._mapping[v] = k
            self._reverse_mapping[k] = v

    def __setitem__(self, key, value):
        """__setitem__.

        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        QUSO dictionary will ever have zero value.

        Parameters
        ----------
        key : tuple.
            Element of the dictionary that follows convensions of the class.
        value : numeric.
            Value corresponding to the key.

        """
        super().__setitem__(key, value)

        for i in key:
            if i not in self._mapping:
                self._mapping[i] = self._next_label
                self._reverse_mapping[self._next_label] = i
                self._next_label += 1

    def to_enumerated(self):
        """to_enumerated.

        Return the default enumerated Matrix object.

        If ``self`` is a QUBO,
        ``self.to_enumerated()`` is equivalent to ``self.to_qubo()``.

        If ``self`` is a QUSO,
        ``self.to_enumerated()`` is equivalent to ``self.to_quso()``.

        If ``self`` is a PUBO or PCBO,
        ``self.to_enumerated()`` is equivalent to ``self.to_pubo()``.

        If ``self`` is a PUSO or PCSO,
        ``self.to_enumerated()`` is equivalent to ``self.to_puso()``.

        Returns
        -------
        res : QUBOMatrix, QUSOMatrix, PUBOMatrix, or PUSOMatrix object.
            If ``self`` is a QUBO type, then this method returns the
            corresponding QUBOMatrix type. If ``self`` is a QUSO type,
            then this method returns the corresponding QUSOMatrix type.
            If ``self`` is a PUBO or PCBO type, then this method returns the
            corresponding PUBOMatrix type. If ``self`` is a PUSO or PCSO type,
            then this method returns the corresponding PUSOMatrix type.

        """
        # we replace c with u so that pcbo and pcso go to pubo and puso.
        return getattr(
            self,
            "to_" + self.__class__.__name__.lower().replace('c', 'u')
        )()
