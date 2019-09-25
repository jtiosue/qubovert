========
QUBOVert
========
*master branch*

.. image:: https://travis-ci.com/jiosue/QUBOVert.svg?branch=master
    :target: https://travis-ci.com/jiosue/QUBOVert
    :alt: Travis-CI
.. image:: https://readthedocs.org/projects/qubovert/badge/?version=latest
    :target: https://qubovert.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://codecov.io/gh/jiosue/QUBOVert/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jiosue/QUBOVert
    :alt: Code Coverage
.. image:: https://img.shields.io/lgtm/grade/python/g/jiosue/QUBOVert.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/jiosue/QUBOVert/context:python
    :alt: Code Quality

*dev branch*

.. image:: https://travis-ci.com/jiosue/QUBOVert.svg?branch=dev
    :target: https://travis-ci.com/jiosue/QUBOVert
    :alt: Travis-CI
.. image:: https://readthedocs.org/projects/qubovert/badge/?version=dev
    :target: https://qubovert.readthedocs.io/en/latest/?badge=dev
    :alt: Documentation Status
.. image:: https://codecov.io/gh/jiosue/QUBOVert/branch/dev/graph/badge.svg
    :target: https://codecov.io/gh/jiosue/QUBOVert
    :alt: Code Coverage

*pypi distribution*

.. image:: https://badge.fury.io/py/qubovert.svg
    :target: https://badge.fury.io/py/qubovert
    :alt: pypi dist
.. image:: https://pepy.tech/badge/qubovert
    :target: https://pepy.tech/project/qubovert
    :alt: pypi dist downloads


Please see the `Repository <https://github.com/jiosue/QUBOVert>`_ and `Docs <https://qubovert.readthedocs.io>`_. For examples/tutorials, see the `notebooks <https://github.com/jiosue/QUBOVert/tree/master/notebook_examples>`_.


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


Managing QUBO, Ising, PUBO, HIsing, HOBO, and HOIO formulations
---------------------------------------------------------------

See ``qubovert.__all__``.

- QUBO: Quadratic Unconstrained Binary Optimization
- Ising: quadratic unconstrained spin-1/2 Hamiltonian
- PUBO: Polynomial Unconstrained Binary Optimization
- HIsing: Higher order unconstrained spin-1/2 Hamiltonian
- HOBO: Higher Order Binary Optimization
- HOIO: Higher Order Ising Optimization

See the docstrings for ``qubovert.HOBO``, ``qubovert.HOIO``, ``qubovert.QUBO``, ``qubovert.Ising``, ``qubovert.PUBO``, and ``qubovert.HIsing``.

See the following HOBO examples (much of the same functionality can be used with HOIO problems).

.. code:: python

    from qubovert import HOBO
    from any_module import qubo_solver
    # or from qubovert.utils import solve_qubo_bruteforce as qubo_solver

    H = HOBO()
    H.add_constraint_eq_zero({('a', 1): 2, (1, 2): -1, (): -1})
    print(H)
    # {('a', 1, 2): -4, (1, 2): 3, (): 1}
    H -= 1
    print(H)
    # {('a', 1, 2): -4, (1, 2): 3}


.. code:: python

    from qubovert import binary_var

    x0, x1, x2 = binary_var("x0"), binary_var("x1"), binary_var("x2")
    H = x0 + 2 * x1 * x2 - 3 + x2
    print(H)
    # {('x0',): 1, ('x1', 'x2'): 2, (): -3, ('x2',): 1}


.. code:: python

    H = HOBO()

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
    # [{0: 0, 1: 1, 2: 0}]  # matches the HOBO solution!

    L = H.to_ising()
    solutions = [H.convert_solution(sol)
                 for sol in L.solve_bruteforce(all_solutions=True)]
    print(solutions)
    # [{0: 0, 1: 1, 2: 0}]  # matches the HOBO solution!

.. code:: python

    # enforce that c == a AND b
    H = HOBO().add_constraint_eq_AND('c', 'a', 'b')
    print(H)
    # {('c',): 3, ('b', 'a'): 1, ('c', 'a'): -2, ('c', 'b'): -2}

.. code:: python

    H = HOBO()
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

    from qubovert import HOIO
    from sympy import Symbol

    a, b = Symbol('a'), Symbol('b')

    # enforce that z_0 + z_1 == 0 with penalty a
    H = HOIO().add_constraint_eq_zero({(0,): 1, (1,): 1}, lam=a)
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

Please note that ``H.mapping`` is not necessarily equal to ``H.subs(...).mapping``. Thus, when using the ``HOBO.convert_solution`` function, make sure that you use the correct ``HOBO`` instance!

The convension used is that ``()`` elements of every dictionary corresponds to offsets. Note that some QUBO solvers accept QUBOs where each key is a two element tuple (since for a QUBO ``{(0, 0): 1}`` is the same as ``{(0,): 1}``). To get this standard form from our ``QUBOMatrix`` object, just access the property ``Q``. Similar for the ``IsingMatrix``. For example:

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

    from qubovert.utils import IsingMatrix
    L = IsingMatrix()
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
- ``solve_hising_bruteforce``,
- ``pubo_value``,
- ``hising_value``,
- ``pubo_to_hising``,
- ``hising_to_pubo``,
- ``subgraph``,

and more.


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

So far we have just implemented some of the formulations from [Lucas]_. The goal of QUBOVert is to become a large collection of problems mapped to QUBO and Ising forms in order to aid the recent increase in study of these problems due to quantum optimization algorithms. Use Python's ``help`` function! I have very descriptive doc strings on all the functions and classes.


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

To use the Ising formulation instead:

.. code:: python

    from qubovert.problems import SetCover
    from any_module import ising_solver
    # or you can use my bruteforce solver...
    # from qubovert.utils import solve_ising_bruteforce as ising_solver

    U = {"a", "b", "c", "d"}
    V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]

    problem = SetCover(U, V)
    L = problem.to_ising()

    obj, sol = ising_solver(L)

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
