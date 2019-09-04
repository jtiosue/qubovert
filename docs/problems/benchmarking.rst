Benchmarking Problems
=====================


Note that the ``problems`` or ``benchmarking`` modules will not be imported with ``from qubovert import *``. You must import ``qubovert.problems`` explicitly.

For example, you can use the ``AlternatingSectorsChain`` class with any of the following.

.. code:: python

    import qubovert
    qubovert.problems.AlternatingSectorsChain(...)
    qubovert.problems.benchmarking.AlternatingSectorsChain(...)


.. automodule:: qubovert.problems.benchmarking
    :members:
