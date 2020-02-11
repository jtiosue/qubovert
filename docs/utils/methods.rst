Utility Methods
================

Note that the ``utils`` module will not be imported with ``from qubovert import *``. You must import ``qubovert.utils`` explicitly.

Accessed with ``qubovert.utils.function_name``.

Conversions
-----------

.. autofunction:: qubovert.utils.pubo_to_hising

.. autofunction:: qubovert.utils.hising_to_pubo

.. autofunction:: qubovert.utils.qubo_to_ising

.. autofunction:: qubovert.utils.ising_to_qubo

.. autofunction:: qubovert.utils.matrix_to_qubo

.. autofunction:: qubovert.utils.qubo_to_matrix

.. autofunction:: qubovert.utils.boolean_to_spin

.. autofunction:: qubovert.utils.spin_to_boolean

.. autofunction:: qubovert.utils.decimal_to_boolean

.. autofunction:: qubovert.utils.decimal_to_spin

.. autofunction:: qubovert.utils.boolean_to_decimal

.. autofunction:: qubovert.utils.spin_to_decimal


Values
------

.. autofunction:: qubovert.utils.pubo_value

.. autofunction:: qubovert.utils.hising_value

.. autofunction:: qubovert.utils.qubo_value

.. autofunction:: qubovert.utils.ising_value


Bruteforce Solvers
------------------

.. autofunction:: qubovert.utils.solve_pubo_bruteforce

.. autofunction:: qubovert.utils.solve_hising_bruteforce

.. autofunction:: qubovert.utils.solve_qubo_bruteforce

.. autofunction:: qubovert.utils.solve_ising_bruteforce


Hash
----

.. autofunction:: qubovert.utils.hash_function


Useful functions
--------

.. autofunction:: qubovert.utils.subgraph

.. autofunction:: qubovert.utils.normalize
