Utility Methods
================

Note that the ``utils`` module will not be imported with ``from qubovert import *``. You must import ``qubovert.utils`` explicitly.

Accessed with ``qubovert.utils.function_name``.

Conversions
-----------

.. autofunction:: qubovert.utils.pubo_to_puso

.. autofunction:: qubovert.utils.puso_to_pubo

.. autofunction:: qubovert.utils.qubo_to_quso

.. autofunction:: qubovert.utils.quso_to_qubo

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

.. autofunction:: qubovert.utils.puso_value

.. autofunction:: qubovert.utils.qubo_value

.. autofunction:: qubovert.utils.quso_value


Bruteforce Solvers
------------------

.. autofunction:: qubovert.utils.solve_pubo_bruteforce

.. autofunction:: qubovert.utils.solve_puso_bruteforce

.. autofunction:: qubovert.utils.solve_qubo_bruteforce

.. autofunction:: qubovert.utils.solve_quso_bruteforce


Hash
----

.. autofunction:: qubovert.utils.hash_function


Useful functions
----------------

.. autofunction:: qubovert.utils.subgraph

.. autofunction:: qubovert.utils.normalize

.. autofunction:: qubovert.utils.num_bits
