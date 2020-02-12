Matrix Objects
==============

The matrix objects are for dealing with PUBOs, PUSOs, QUBOs, and QUSOs that have integer labels. All the ``to_`` methods return matrix objects. For example, ``PCBO.to_qubo`` returns a ``QUBOMatrix`` object.

Note that the ``utils`` module will not be imported with ``from qubovert import *``. You must import ``qubovert.utils`` explicitly.

Accessed with ``qubovert.utils.matrix_name``.


PUBOMatrix
----------

.. autoclass:: qubovert.utils.PUBOMatrix
    :members:


PUSOMatrix
------------

.. autoclass:: qubovert.utils.PUSOMatrix
    :members:


QUBOMatrix
----------

.. autoclass:: qubovert.utils.QUBOMatrix
    :members:


QUSOMatrix
-----------

.. autoclass:: qubovert.utils.QUSOMatrix
    :members:
