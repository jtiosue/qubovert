qubovert
========

The one-stop package for formulating, simulating, and solving problems in boolean and spin form.


*master branch*

.. image:: https://github.com/jtiosue/qubovert/workflows/build/badge.svg?branch=master
    :target: https://github.com/jtiosue/qubovert/actions?query=workflow%3Abuild+branch%3Amaster
    :alt: GitHub Actions CI
.. image:: https://readthedocs.org/projects/qubovert/badge/?version=latest
    :target: https://qubovert.readthedocs.io/en/latest/
    :alt: Documentation Status
.. image:: https://codecov.io/gh/jtiosue/qubovert/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jtiosue/qubovert/branch/master
    :alt: Code Coverage
.. image:: https://img.shields.io/lgtm/grade/python/g/jtiosue/qubovert.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/jtiosue/qubovert/context:python
    :alt: Code Quality

*dev branch*

.. image:: https://github.com/jtiosue/qubovert/workflows/build/badge.svg?branch=dev
    :target: https://github.com/jtiosue/qubovert/actions?query=workflow%3Abuild+branch%3Adev
    :alt: GitHub Actions CI
.. image:: https://readthedocs.org/projects/qubovert/badge/?version=dev
    :target: https://qubovert.readthedocs.io/en/dev/
    :alt: Documentation Status
.. image:: https://codecov.io/gh/jtiosue/qubovert/branch/dev/graph/badge.svg
    :target: https://codecov.io/gh/jtiosue/qubovert/branch/dev
    :alt: Code Coverage

*pypi distribution*

.. image:: https://badge.fury.io/py/qubovert.svg
    :target: https://badge.fury.io/py/qubovert
    :alt: pypi dist
.. image:: https://pepy.tech/badge/qubovert
    :target: https://pepy.tech/project/qubovert
    :alt: pypi dist downloads


Please see the `Repository <https://github.com/jtiosue/qubovert>`_ and `Docs <https://qubovert.readthedocs.io>`_. For examples/tutorials, see the `notebooks <https://github.com/jtiosue/qubovert/tree/master/notebook_examples>`_.


.. contents::
    :local:
    :backlinks: top


Installation
------------

For the stable release (same version as the *master* branch):

.. code:: shell

  pip install qubovert


Or to install from source:

.. code:: shell

  git clone https://github.com/jtiosue/qubovert.git
  cd qubovert
  pip install -e .


Then you can use it in Python **versions 3.6 and above** with

.. code:: python

    import qubovert as qv


Note that to install from source on Windows you will need `Microsoft Visual C++ Build Tools 14 <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ installed.


Example of the typical workflow
-------------------------------

Here we show an example of formulating a pseudo-boolean objective function. We can also make spin objective functions (Hamiltonians) in a very similar manner. See the `notebooks <https://github.com/jtiosue/qubovert/tree/master/notebook_examples>`_ for examples.


Create the boolean objective function to minimize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from qubovert import boolean_var

    N = 10

    # create the variables
    x = {i: boolean_var('x(%d)' % i) for i in range(N)}

    # minimize \sum_{i=0}^{N-1} (1-2x_{i}) x_{i+1}
    model = 0
    for i in range(N-1):
        model += (1 - 2 * x[i]) * x[i+1]

    # subject to the constraint that x_1 equals the XOR of x_3 and x_5
    # enforce with a penalty factor of 3
    model.add_constraint_eq_XOR(x[1], x[3], x[5], lam=3)

    # subject to the constraints that the sum of all variables is less than 4
    # enforce with a penalty factor of 5
    model.add_constraint_lt_zero(sum(x.values()) - 4, lam=5)


Next we will show multiple ways to solve the model.


Solving the model with bruteforce
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before using the bruteforce solver, always check that ``model.num_binary_variables`` is relatively small!


.. code:: python

    model_solution = model.solve_bruteforce()
    print("Variable assignment:", model_solution)
    print("Model value:", model.value(model_solution))
    print("Constraints satisfied?", model.is_solution_valid(model_solution))


Solving the model with *qubovert*'s simulated annealing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see the definition of PUBO in the next section. We will anneal the PUBO.

.. code:: python

    from qubovert.sim import anneal_pubo

    res = anneal_pubo(model, num_anneals=10)
    model_solution = res.best.state

    print("Variable assignment:", model_solution)
    print("Model value:", res.best.value)
    print("Constraints satisfied?", model.is_solution_valid(model_solution))


Solving the model with D-Wave's simulated annealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`D-Wave's simulated annealer <https://github.com/dwavesystems/dwave-neal>`_ cannot anneal PUBOs as we did above. Instead the model must be reduced to a QUBO. See the next section for definitions of QUBO and PUBO.

.. code:: python

    from neal import SimulatedAnnealingSampler

    # Get the QUBO form of the model
    qubo = model.to_qubo()

    # D-Wave accept QUBOs in a different format than qubovert's format
    # to get the qubo in this form, use the .Q property
    dwave_qubo = qubo.Q

    # solve with D-Wave
    res = SimulatedAnnealingSampler().sample_qubo(dwave_qubo, num_reads=10)
    qubo_solution = res.first.sample

    # convert the qubo solution back to the solution to the model
    model_solution = model.convert_solution(qubo_solution)

    print("Variable assignment:", model_solution)
    print("Model value:", model.value(model_solution))
    print("Constraints satisfied?", model.is_solution_valid(model_solution))


Managing QUBO, QUSO, PUBO, PUSO, PCBO, and PCSO formulations
------------------------------------------------------------

*qubovert* defines, among many others, the following objects.

- QUBO: Quadratic Unconstrained Boolean Optimization (``qubovert.QUBO``)
- QUSO: Quadratic Unconstrained Spin Optimization (``qubovert.QUSO``)
- PUBO: Polynomial Unconstrained Boolean Optimization (``qubovert.PUBO``)
- PUSO: Polynomial Unconstrained Spin Optimization (``qubovert.PUSO``)
- PCBO: Polynomial Constrained Boolean Optimization (``qubovert.PCBO``)
- PCSO: Polynomial Constrained Spin Optimization (``qubovert.PCSO``)

Each of the objects has many methods and arbitary arithmetic defined; see the docstrings of each object and the `notebooks <https://github.com/jtiosue/qubovert/tree/master/notebook_examples>`_ for more info. A boolean optimization model is one whose variables can be assigned to be either 0 or 1, while a spin optimization model is one whose variables can be assigned to be either 1 or -1. The ``qubovert.boolean_var(name)`` function will create a PCBO representing the boolean variable with name ``name``. Similarly, the ``qubovert.spin_var(name)`` function will create a PCSO representing the spin variable with name ``name``.


There are many utilities in the *utils* library that can be helpful. Some examples of utility functions are listed here.

- ``qubovert.utils.solve_pubo_bruteforce``, solve a PUBO by iterating through all possible solutions.
- ``qubovert.utils.solve_puso_bruteforce``, solve a PUSO by iterating through all possible solutions.
- ``qubovert.utils.pubo_to_puso``, convert a PUBO to a PUSO.
- ``qubovert.utils.puso_to_pubo``, convert a PUSO to a PUBO.
- ``qubovert.utils.pubo_value``, determine the value that a PUBO takes with a particular solution mapping.
- ``qubovert.utils.puso_value``, determine the value that a PUSO takes with a particular solution mapping.
- ``qubovert.utils.approximate_pubo_extrema``, approximate the minimum and maximum values that a PUBO can take; the true extrema will lie within these bounds.
- ``qubovert.utils.approximate_puso_extrema``, approximate the minimum and maximum values that a PUSO can take; the true extrema will lie within these bounds.
- ``qubovert.utils.subgraph``, create the subgraph of a model that only contains certain given variables.
- ``qubovert.utils.subvalue``, create the submodel of a model with certain values of the model replaced with values.
- ``qubovert.utils.normalize``, normalize a model such that its coefficients have a maximum absolute magnitude.

See ``qubovert.utils.__all__`` for more. Please note that all conversions between boolean and spin map {0, 1} to/from {1, -1} in that order! This is the convention that *qubovert* uses everywhere.


The PCBO and PCSO objects have constraint methods; for example, the ``.add_constraint_le_zero`` method will enforce that an expression is less than or equal to zero by adding a penalty to the model whenever it does not. The PCBO object also has constraint methods for satisfiability expressions; for example, the ``.add_constraint_OR`` will enforce that the OR of the given boolean expression evaluates to True by adding a penalty to the model whenever it does not. See the docstrings and `notebooks <https://github.com/jtiosue/qubovert/tree/master/notebook_examples>`_ for more info.


For more utilities on satisfiability expressions, *qubovert* also has a *sat* library; see ``qubovert.sat.__all__``. Consider the following 3-SAT example. We have variables ``x0, x1, x2, x3``, labeled by ``0, 1, 2, 3``. We can create an expression ``C`` that evaluates to 1 whenever the 3-SAT conditions are satisfied.

.. code:: python

    from qubovert.sat import AND, NOT, OR

    C = AND(OR(0, 1, 2), OR(NOT(0), 2, NOT(3)), OR(NOT(1), NOT(2), 3))

    # C = 1 for a satisfying assignment, C = 0 otherwise
    # So minimizing -C will solve it.
    P = -C
    solution = P.solve_bruteforce()



Basic examples of common functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See the `notebooks <https://github.com/jtiosue/qubovert/tree/master/notebook_examples>`_ for many fully worked out examples. Here we will just show some basic and brief examples.


The basic building block of a binary optimization model is a Python dictionary. The keys of the dictionary are tuples of variable names, and the values are their corresponding coefficients. For example, in the below code block, ``model1``, ``model2``, and ``model3`` are equivalent.

.. code:: python

    from qubovert import boolean_var, PUBO

    x0, x1, x2 = boolean_var('x0'), boolean_var('x1'), boolean_var('x2')

    model1 = -1 + x0 + 2 * x0 * x1 - 3 * x0 * x2 + x0 * x1 * x2
    model2 = {(): -1, ('x0',): 1, ('x0', 'x1'): 2, ('x0', 'x2'): -3, ('x0', 'x1', 'x2'): 1}
    model3 = PUBO(model2)


Similarly, in the below code block, ``model1``, ``model2``, and ``model3`` are equivalent.

.. code:: python

    from qubovert import spin_var, PUSO

    z0, z1, z2 = spin_var('z0'), spin_var('z1'), spin_var('z2')

    model1 = -1 + z0 + 2 * z0 * z1 - 3 * z0 * z2 + z0 * z1 * z2
    model2 = {(): -1, ('z0',): 1, ('z0', 'z1'): 2, ('z0', 'z2'): -3, ('z0', 'z1', 'z2'): 1}
    model3 = PUSO(model2)



Let's take the same model from above (ie define :code:`model = model1.copy()`). Suppose we want to find the ground state of the model subject to the constraints that the sum of the variables is negative and that the product of ``z0`` and ``z1`` is 1. We have to enforce these constraints with a penalty called ``lam``. For now, let's set it as a Symbol that we can adjust later.

.. code:: python

    from sympy import Symbol

    lam = Symbol('lam')
    model.add_constraint_lt_zero(z0 + z1 + z2, lam=lam)
    model.add_constraint_eq_zero(z0 * z1 - 1, lam=lam)


Note that constraint methods can also be strung together if you want. So we could have written this as

.. code:: python

    model.add_constraint_lt_zero(
        z0 + z1 + z2, lam=lam
    ).add_constraint_eq_zero(
        z0 * z1 - 1, lam=lam
    )


The first thing you notice if you :code:`print(model.variables)` is that there are now new variables in the model called ``'__a0'`` and ``'__a1'``. These are auxillary or *ancilla* variables that are needed to enforce the constraints. The next thing to notice if you :code:`print(model.degree)` is that the model is a polynomial of degree 3. Many solvers (for example D-Wave's solvers) only solve degree 2 models. To get a QUBO or QUSO (which are degree two modes) from ``model``, simply call the ``.to_qubo`` or ``.to_quso`` methods, which will reduce the degree to 2 by introducing more variables.

.. code:: python

    qubo = model.to_qubo()
    quso = model.to_quso()


Next let's solve the QUBO and/or QUSO formulations. First we have to substitute a value in for our placeholder symbol ``lam`` that is used to enforce the constraints. We'll just use ``lam=3`` for now.

.. code:: python

    qubo = qubo.subs({lam: 3})
    quso = quso.subs({lam: 3})


Here we will use `D-Wave's simulated annealer <https://github.com/dwavesystems/dwave-neal>`_.

.. code:: python

    from neal import SimulatedAnnealingSampler

    # D-Wave represents QUBOs a little differently than qubovert does.
    # to get D-Wave's form, use the .Q property
    dwave_qubo = qubo.Q

    # D-Wave represents QUSOs a little differently than qubovert does.
    # to get D-Wave's form, use the .h property the linear terms and the
    # .J property for the quadratic terms
    dwave_linear, dwave_quadratic = quso.h, quso.J

    # call dwave
    qubo_res = SimulatedAnnealingSampler().sample_qubo(dwave_qubo)
    quso_res = SimulatedAnnealingSampler().sample_ising(dwave_linear, dwave_quadratic)

    qubo_solution = qubo_res.first.sample
    quso_solution = quso_res.first.sample


Now we have to convert the solution in terms of the QUBO/QUSO variables back to a solution in terms of the original variables. We can then check if the proposed solution satisfies all of the constraints!

.. code:: python

    converted_qubo_solution = model.convert_solution(qubo_solution)
    print(model.is_solution_valid(converted_qubo_solution))

    converted_quso_solution = model.convert_solution(quso_solution)
    print(model.is_solution_valid(converted_quso_solution))


Convert common problems to quadratic form (the *problems* library)
------------------------------------------------------------------

One of the goals of *qubovert* is to become a large collection of problems mapped to QUBO and QUSO forms in order to aid the recent increase in study of these problems due to quantum optimization algorithms. Use Python's ``help`` function! I have very descriptive doc strings on all the functions and classes. Please see the `notebooks <https://github.com/jtiosue/qubovert/tree/master/notebook_examples>`_ for a few more examples as well.


See the following Set Cover example.

.. code:: python

    from qubovert.problems import SetCover
    from any_module import qubo_solver
    # or you can use my bruteforce solver...
    # from qubovert.utils import solve_qubo_bruteforce as qubo_solver

    U = {"a", "b", "c", "d"}
    V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]

    problem = SetCover(U, V)
    Q = problem.to_qubo()

    obj, sol = qubo_solver(Q)

    solution = problem.convert_solution(sol)

    print(solution)
    # {0, 2}
    print(problem.is_solution_valid(solution))
    # will print True, since V[0] + V[2] covers all of U
    print(obj == len(solution))
    # will print True

To use the QUSO formulation instead:

.. code:: python

    from qubovert.problems import SetCover
    from any_module import quso_solver
    # or you can use my bruteforce solver...
    # from qubovert.utils import solve_quso_bruteforce as quso_solver

    U = {"a", "b", "c", "d"}
    V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]

    problem = SetCover(U, V)
    L = problem.to_quso()

    obj, sol = quso_solver(L)

    solution = problem.convert_solution(sol)

    print(solution)
    # {0, 2}
    print(problem.is_solution_valid(solution))
    # will print True, since V[0] + V[2] covers all of U
    print(obj == len(solution))
    # will print True


To see problem specifics, run

.. code:: python

    help(qubovert.problems.SetCover)
    help(qubovert.problems.VertexCover)
    # etc


====

.. image:: https://raw.githubusercontent.com/jtiosue/qubovert/master/assets/qvfire.png
