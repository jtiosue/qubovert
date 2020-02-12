========
qubovert
========
*master branch*

.. image:: https://travis-ci.com/jiosue/qubovert.svg?branch=master
    :target: https://travis-ci.com/jiosue/qubovert
    :alt: Travis-CI
.. image:: https://readthedocs.org/projects/qubovert/badge/?version=latest
    :target: https://qubovert.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://codecov.io/gh/jiosue/qubovert/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jiosue/qubovert
    :alt: Code Coverage
.. image:: https://img.shields.io/lgtm/grade/python/g/jiosue/qubovert.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/jiosue/qubovert/context:python
    :alt: Code Quality

*dev branch*

.. image:: https://travis-ci.com/jiosue/qubovert.svg?branch=dev
    :target: https://travis-ci.com/jiosue/qubovert
    :alt: Travis-CI
.. image:: https://readthedocs.org/projects/qubovert/badge/?version=dev
    :target: https://qubovert.readthedocs.io/en/latest/?badge=dev
    :alt: Documentation Status
.. image:: https://codecov.io/gh/jiosue/qubovert/branch/dev/graph/badge.svg
    :target: https://codecov.io/gh/jiosue/qubovert
    :alt: Code Coverage

*pypi distribution*

.. image:: https://badge.fury.io/py/qubovert.svg
    :target: https://badge.fury.io/py/qubovert
    :alt: pypi dist
.. image:: https://pepy.tech/badge/qubovert
    :target: https://pepy.tech/project/qubovert
    :alt: pypi dist downloads


Please see the `Repository <https://github.com/jiosue/qubovert>`_ and `Docs <https://qubovert.readthedocs.io>`_. For examples/tutorials, see the `notebooks <https://github.com/jiosue/qubovert/tree/master/notebook_examples>`_.


Installation
------------
`For the stable release`.

.. code:: shell

  pip install qubovert


To install from source:

.. code:: shell

  git clone https://github.com/jiosue/qubovert.git
  cd qubovert
  pip install -e .


Then you can use it in Python **versions 3.6 and above** with

.. code:: python

    import qubovert

    # get info
    help(qubovert)

    # see the main functionality
    print(qubovert.__all__)

    # see the utilities defined
    help(qubovert.utils)
    print(qubovert.utils.__all__)

    # see the satisfiability library
    help(qubovert.sat)
    print(qubovert.sat.__all__)

    # see all the probles defined
    print(qubovert.problems.__all__)

    # to see specifically the np problems:
    help(qubovert.problems.np)
    print(qubovert.problems.np.__all__)

    # to see specifically the benchmarking problems:
    help(qubovert.problems.benchmarking)
    print(qubovert.problems.benchmarking.__all__)

    # etc ...


Example of the typical workflow
-------------------------------

Create the objective function to minimize.

.. code:: python

    from qubovert import boolean_var
    from qubovert.sat import NOT

    N = 10

    x = {i: boolean_var('x(%d)' % i) for i in range(N)}

    model = sum(NOT(x[i-1]) * x[i] for i in range(N-1))

    # enforce that x_1 equals the XOR of x_3 and x_5 with a penalty factor of 3
    model.add_constraint_eq_XOR(x[1], x[3], x[5], lam=3)

    # enforce that the sum of all variables is less than 4 with a penalty factor of 5.
    model.add_constraint_lt_zero(sum(x.values()) - 4, lam=5)


Then, if you have a QUBO solver (or just use ``qubovert.utils.solve_qubo_bruteforce``):

.. code:: python

    from anywhere import qubo_solver

    qubo = model.to_qubo()
    qubo_energy, qubo_solution = qubo_solver(qubo)
    model_solution = model.convert_solution(qubo_solution)
    print(model_solution)


Otherwise, if you have a QUSO solver (or just use ``qubovert.utils.solve_quso_bruteforce``):

.. code:: python

    from anywhere import quso_solver

    quso = model.to_quso()
    quso_energy, quso_solution = quso_solver(quso)
    model_solution = model.convert_solution(quso_solution)
    print(model_solution)


Each ``model_solution`` should be the same! You can test that it is the correct solution by comparing it to ``model.solve_bruteforce()``. You can also check if all of the constraints are satisfied by running ``model.is_solution_valid(model_solution)``.


Managing QUBO, QUSO, PUBO, PUSO, PCBO, and PCSO formulations
---------------------------------------------------------------

See ``qubovert.__all__``.

- QUBO: Quadratic Unconstrained Boolean Optimization
- QUSO: Quadratic Unconstrained Spin Optimization
- PUBO: Polynomial Unconstrained Boolean Optimization
- PUSO: Polynomial Unconstrained Spin Optimization
- PCBO: Polynomial Constrained Boolean Optimization
- PCSO: Polynomial Constrained Spin Optimization

Boolean variables are in {0, 1}, and spin variables are in {1, -1}. See the docstrings for ``qubovert.PCBO``, ``qubovert.PCSO``, ``qubovert.QUBO``, ``qubovert.QUSO``, ``qubovert.PUBO``, and ``qubovert.PUSO``.

See the following PCBO examples (much of the same functionality can be used with PCSO problems).

.. code:: python

    from qubovert import PCBO

    H = PCBO()
    H.add_constraint_eq_zero({('a', 1): 2, (1, 2): -1, (): -1})
    print(H)
    # {('a', 1, 2): -4, (1, 2): 3, (): 1}
    H -= 1
    print(H)
    # {('a', 1, 2): -4, (1, 2): 3}


.. code:: python

    from qubovert import boolean_var

    x0, x1, x2 = boolean_var("x0"), boolean_var("x1"), boolean_var("x2")
    H = x0 + 2 * x1 * x2 - 3 + x2
    print(H)
    # {('x0',): 1, ('x1', 'x2'): 2, (): -3, ('x2',): 1}


Note that for large problems, it is slower to use the `boolean_var` functionality. For example, consider the following where creating `H0` is much faster than creating `H1`!

.. code:: python

    from qubovert import boolean_var, PCBO

    H0 = PCBO()
    for i in range(1000):
        H0[(i,)] += 1

    xs = [boolean_var(i) for i in range(1000)]
    H1 = sum(xs)


Here we show how to solve problems with the bruteforce solver, and how to convert problems to QUBO and QUSO form. You can use any QUBO/QUSO solver you'd like to solve!

.. code:: python

    H = PCBO()

    # minimize -x_0 - x_1 - x_2
    for i in (0, 1, 2):
        H[(i,)] -= 1

    # subject to constraints
    H.add_constraint_eq_zero(  # enforce that x_0 x_1 - x_2 == 0
        {(0, 1): 1, (2,): -1}
    ).add_constraint_lt_zero(  # enforce that x_1 x_2 + x_0 < 1
        {(1, 2): 1, (0,): 1, (): -1}
    )
    print(H)
    # {(1,): -2, (2,): -1, (0, 1): 2, (1, 2): 2, (0, 1, 2): 2}

    print(H.solve_bruteforce(all_solutions=True))
    # [{0: 0, 1: 1, 2: 0}]

    Q = H.to_qubo()
    solutions = [H.convert_solution(sol)
                 for sol in Q.solve_bruteforce(all_solutions=True)]
    print(solutions)
    # [{0: 0, 1: 1, 2: 0}]  # matches the PCBO solution!

    L = H.to_quso()
    solutions = [H.convert_solution(sol)
                 for sol in L.solve_bruteforce(all_solutions=True)]
    print(solutions)
    # [{0: 0, 1: 1, 2: 0}]  # matches the PCBO solution!


Here we show how to add various boolean constraints to models.

.. code:: python

    H = PCBO()
    # make it favorable to AND variables a and b, and variables b and c
    H.add_constraint_AND('a', 'b').add_constraint_AND('b', 'c')

    # make it favorable to OR variables b and c
    H.add_constraint_OR('b', 'c')

    # make it favorable to (a AND b) OR (c AND d) OR e
    H.add_constraint_OR(['a', 'b'], ['c', 'd'], 'e')

    # enforce that 'b' = NOR('a', 'c', 'd')
    H.add_constraint_eq_NOR('b', 'a', 'c', 'd')

    print(H)
    # {(): 5, ('c',): -2, ('c', 'a', 'b', 'd'): 1, ('a', 'e', 'b'): 1,
    #  ('c', 'e', 'd'): 1, ('e',): -1, ('a',): -1, ('c', 'a'): 1,
    #  ('a', 'd'): 1, ('c', 'b'): 2, ('d',): -1, ('b', 'd'): 2}
    Q = H.to_qubo()
    print(Q)
    # {(): 5, (2,): -2, (5,): 12, (0, 1): 4, (0, 5): -8, (1, 5): -8,
    #  (6,): 12, (2, 3): 4, (2, 6): -8, (3, 6): -8, (5, 6): 1, (4, 5): 1,
    #  (4, 6): 1, (4,): -1, (0,): -1, (0, 2): 1, (0, 3): 1, (1, 2): 2,
    #  (3,): -1, (1, 3): 2}
    obj_value, sol = qubo_solver(Q)
    print(sol)
    # {0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0}
    solution = H.convert_solution(sol)
    print(solution)
    # {'a': 0, 'b': 0, 'c': 1, 'd': 0, 'e': 1}


See the following PUBO example.

.. code:: python

    from qubovert import PUBO
    from any_module import qubo_solver
    # or you can use my bruteforce solver...
    # from qubovert.utils import solve_qubo_bruteforce as qubo_solver

    pubo = PUBO()
    pubo[('a', 'b', 'c', 'd')] -= 3
    pubo[('a', 'b', 'c')] += 1
    pubo[('c', 'd')] -= 2
    pubo[('a',)] += 1
    pubo -= 3  # equivalent to pubo[()] -= 3
    pubo **= 4
    pubo *= 2

    Q = pubo.to_qubo()
    obj, sol = qubo_solver(Q)
    solution = pubo.convert_solution(sol)
    print((obj, solution))
    # (2, {'a': 1, 'b': 1, 'c': 1, 'd': 0})


Symbols can also be used, for example:

.. code:: python

    from qubovert import PCSO
    from sympy import Symbol

    a, b = Symbol('a'), Symbol('b')

    # enforce that z_0 + z_1 == 0 with penalty a
    H = PCSO().add_constraint_eq_zero({(0,): 1, (1,): 1}, lam=a)
    print(H)
    # {(): 2*a, (0, 1): 2*a}
    H[(0, 1)] += b
    print(H)
    # {(): 2*a, (0, 1): 2*a + b}
    H_subs = H.subs({a: 2})
    print(H_subs)
    # {(): 4, (0, 1): 4 + b}

    H_subs = H.subs({a: 2, b: 3})
    print(H_subs)
    # {(): 4, (0, 1): 7}

Please note that ``H.mapping`` is not necessarily equal to ``H.subs(...).mapping``. Thus, when using the ``PCBO.convert_solution`` function, make sure that you use the correct ``PCBO`` instance!

The convension used is that ``()`` elements of every dictionary corresponds to offsets. Note that some QUBO solvers accept QUBOs where each key is a two element tuple (since for a QUBO ``{(0, 0): 1}`` is the same as ``{(0,): 1}``). To get this standard form from our ``QUBOMatrix`` object, just access the property ``Q``. Similar for the ``QUSOMatrix``. For example:

.. code:: python

    from qubovert.utils import QUBOMatrix
    Q = QUBOMatrix()
    Q += 3
    Q[(0,)] -= 1
    Q[(0, 1)] += 2
    Q[(1, 1)] -= 3
    print(Q)
    # {(): 3, (0,): -1, (0, 1): 2, (1,): -3}
    print(Q.Q)
    # {(0, 0): -1, (0, 1): 2, (1, 1): -3}
    print(Q.offset)
    # 3

.. code:: python

    from qubovert.utils import QUSOMatrix
    L = QUSOMatrix()
    L += 3
    L[(0, 1, 1)] -= 1
    L[(0, 1)] += 2
    L[(1, 1)] -= 3
    print(L)
    # {(0,): -1, (0, 1): 2}
    print(L.h)
    # {0: -1}
    print(L.J)
    # {(0, 1): 2}
    print(L.offset)
    # 0


Common binary optimization utilities (the ``utils`` library)
------------------------------------------------------------

See ``qubovert.utils.__all__``.

We implement various utility functions, including

- ``solve_pubo_bruteforce``,
- ``solve_puso_bruteforce``,
- ``pubo_value``,
- ``puso_value``,
- ``pubo_to_puso``,
- ``puso_to_pubo``,
- ``subgraph``,
- ``normalize``,

and more. Please note that all conversions between boolean and spin map {0, 1} to/from {1, -1} in that order! This is the convention that qubovert uses everywhere.


Converting SAT problems (the ``sat`` library)
---------------------------------------------

See ``qubovert.sat.__all__``.

Consider the following 3-SAT example.

.. code:: python

    from qubovert.sat import AND, NOT, OR
    from anywhere import qubo_solver

    C = AND(OR(0, 1, 2), OR(NOT(0), 2, NOT(3)), OR(NOT(1), NOT(2), 3))

    # C is 1 for a satisfying assignment, else 0
    # So minimizing P will solve it.
    P = -C

    # P is a PUBO
    Q = P.to_qubo()
    solution = qubo_solver(Q)

    print(solution)  # {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0}
    converted_sol = P.convert_solution(solution)
    print(converted_sol) # {0: 0, 3: 0, 1: 0, 2: 1}

    print(C.value(converted_sol))  # will print 1 because it satisfies C


Convert common problems to QUBO form (the ``problems`` library)
---------------------------------------------------------------

See ``qubovert.problems.__all__``.

So far we have just implemented some of the formulations from [Lucas]_. The goal of QUBOVert is to become a large collection of problems mapped to QUBO and QUSO forms in order to aid the recent increase in study of these problems due to quantum optimization algorithms. Use Python's ``help`` function! I have very descriptive doc strings on all the functions and classes.


See the following Set Cover example. All other problems can be used in a similar way.

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

I have very descriptive doc strings that should explain everything you need to know to use each problem class.


References
----------

.. [Lucas] Andrew Lucas. Ising formulations of many np problems. Frontiers in Physics, 2:5, 2014.
