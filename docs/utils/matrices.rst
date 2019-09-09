Matrix Objects
==============

The matrix objects are for dealing with PUBOs, HIsings, QUBOs, and Isings that have integer labels. All the ``to_`` methods return matrix objects. For example, ``HOBO.to_qubo`` returns a ``QUBOMatrix`` object.

Note that the ``utils`` module will not be imported with ``from qubovert import *``. You must import ``qubovert.utils`` explicitly.

Accessed with ``qubovert.utils.matrix_name``.


PUBOMatrix
----------

.. autoclass:: qubovert.utils.PUBOMatrix
    :members:


HIsingMatrix
------------

.. autoclass:: qubovert.utils.HIsingMatrix
    :members:


QUBOMatrix
----------

.. autoclass:: qubovert.utils.QUBOMatrix
    :members:


IsingMatrix
-----------

.. autoclass:: qubovert.utils.IsingMatrix
    :members:
