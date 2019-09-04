NP Problems
===========


Note that the ``problems`` or ``np`` modules will not be imported with ``from qubovert import *``. You must import ``qubovert.problems`` explicitly.

For example, you can use the ``SetCover`` class with any of the following.

.. code:: python

    import qubovert
    qubovert.problems.SetCover(...)
    qubovert.problems.np.SetCover(...)
    qubovert.problems.np.covering.SetCover(...)


.. automodule:: qubovert.problems.np
    :members:
