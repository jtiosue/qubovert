Higher Order Binary Optimization (HOBO)
=======================================

Accessed with ``qubovert.HOBO``. Note that it is important to use the ``HOBO.convert_solution`` function to convert solutions of the PUBO, QUBO, Hising or Ising formulations of the HOBO back to a solution to the HOBO formulation.

We also discuss the ``qubovert.binary_var`` function here, which just creates a ``HOBO``.


.. autoclass:: qubovert.HOBO
    :members:


.. autofunction:: qubovert.binary_var
