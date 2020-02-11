Higher Order Boolean Optimization (HOBO)
=======================================

Accessed with ``qubovert.HOBO``. Note that it is important to use the ``HOBO.convert_solution`` function to convert solutions of the PUBO, QUBO, Hising or Ising formulations of the HOBO back to a solution to the HOBO formulation.

We also discuss the ``qubovert.boolean_var`` and ``qubovert.integer_var`` functions here, which just create ``HOBO``s.


.. autofunction:: qubovert.boolean_var


.. autofunction:: qubovert.integer_var


.. autoclass:: qubovert.HOBO
    :members:
