Annealing
=========

Note that the ``sim`` module will not be imported with ``from qubovert import *``. You must import ``qubovert.sim`` explicitly. Here we show some functions to use the boolean and spin simulation to run simulated annealing on the models.

**Please note** that the ``qv.sim.anneal_qubo`` and ``qv.sim.anneal_quso`` functions perform much faster than the ``qv.sim.anneal_pubo`` and ``qv.sim.anneal_puso`` functions since the former are written in C and wrapped in Python. If your system has degree 2 or less, then you should use the QUBO or QUSO anneal functions!



Anneal PUBO
-----------

.. autofunction:: qubovert.sim.anneal_pubo


Anneal PUSO
-----------

.. autofunction:: qubovert.sim.anneal_puso


Anneal QUBO
-----------

.. autofunction:: qubovert.sim.anneal_qubo


Anneal QUSO
-----------

.. autofunction:: qubovert.sim.anneal_quso


Anneal Results
--------------

These objects are defined to deal with the output of the annealing functions.


.. autoclass:: qubovert.sim.AnnealResults
   :members:


.. autoclass:: qubovert.sim.AnnealResult
   :members:


Anneal temperature range
------------------------

The following function is used to determine the default annealing temperatures to start and stop at for the above anneal functions if the user does not supply a range themselves.

.. autofunction:: qubovert.sim.anneal_temperature_range
