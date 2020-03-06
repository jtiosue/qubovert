Simulation
==========

Note that the ``sim`` module will not be imported with ``from qubovert import *``. You must import ``qubovert.sim`` explicitly.

**Please note** that the ``qv.sim.QUBOSimulation`` and ``qv.sim.QUSOSimulation`` objects perform simulation much faster than the ``qv.sim.PUBOSimulation`` and ``qv.sim.PUSOSimulation`` objects since the former are written in C and wrapped in Python. If your system has degree 2 or less, then you should use the QUBO or QUSO simulations!


PUBO Simulation
---------------

.. autoclass:: qubovert.sim.PUBOSimulation
   :members:
   :inherited-members:


PUSO Simulation
---------------

.. autoclass:: qubovert.sim.PUSOSimulation
   :members:
   :inherited-members:


QUBO Simulation
---------------

.. autoclass:: qubovert.sim.QUBOSimulation
   :members:
   :inherited-members:


QUSO Simulation
---------------

.. autoclass:: qubovert.sim.QUSOSimulation
   :members:
   :inherited-members:
