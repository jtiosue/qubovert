Polynomial Constrained Boolean Optimization (PCBO)
==================================================

Accessed with ``qubovert.PCBO``. Note that it is important to use the ``PCBO.convert_solution`` function to convert solutions of the PUBO, QUBO, PUSO or QUSO formulations of the PCBO back to a solution to the PCBO formulation.

We also discuss the ``qubovert.boolean_var`` and ``qubovert.integer_var`` functions here, which just create ``PCBO`` objects.


.. autofunction:: qubovert.boolean_var


.. autofunction:: qubovert.integer_var


.. autoclass:: qubovert.PCBO
    :members:
