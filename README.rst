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

Please see the `Repository <https://github.com/jiosue/QUBOVert>`_ and `Docs <https://qubovert.readthedocs.io>`_. For examples/tutorials, see `notebooks <https://github.com/jiosue/QUBOVert/tree/master/notebook_examples>`_


Installation
------------
`For the old, stable release`.

.. code:: shell

  pip install qubovert


To install from source:

.. code:: shell

  git clone https://github.com/jiosue/qubovert.git
  cd qubovert
  pip install -e .


Then you can use it in Python versions 3.6 and above with

.. code:: python

    import qubovert

    # get info
    help(qubovert)

    # see the main functionality
    print(qubovert.__all__)

    # see all the probles defined
    print(qubovert.problems.__all__)

    # see the utilities defined
    help(qubovert.utils)
    print(qubovert.utils.__all__)

    # to see specifically the np problems:
    help(qubovert.problems.np)
    print(qubovert.problems.np.__all__)

    # to see specifically the benchmarking problems:
    help(qubovert.problems.benchmarking)
    print(qubovert.problems.benchmarking.__all__)

    # etc ...


Managing QUBO, Ising, PUBO, HIsing, HOBO, and HOIO formulations
---------------------------------------------------------------
See the docstrings for ``qubovert.HOBO``, ``qubovert.HOIO``, ``qubovert.QUBO``, ``qubovert.Ising``, ``qubovert.PUBO``, and ``qubovert.HIsing``.

See the following HOBO example.

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

    H = HOBO()
    H.add_constraint_eq_zero(
            {(0, 1): 1}
        ).add_constraint_eq_zero(
            {(1, 2): 1, (): -1}
        )
    print(H)
    # {(0, 1): 1, (1, 2): -1, (): 1}

    H = HOBO().add_constraint_AND('a', 'b', 'c')
    print(H)
    # {('c',): 3, ('b', 'a'): 1, ('c', 'a'): -2, ('c', 'b'): -2}

    H = HOBO()
    # AND variables a and b, and variables b and c
    H.AND('a', 'b').AND('b', 'c')

    # OR variables b and c
    H.OR('b', 'c')

    # (a AND b) OR (c AND d)
    H.OR(['a', 'b'], ['c', 'd'])

    print(H)
    # {('b', 'a'): -2, (): 4, ('b',): -1, ('c',): -1, ('c', 'd'): -1,
    #  ('c', 'd', 'b', 'a'): 1}
    Q = H.to_qubo()
    print(Q)
    # {(): 4, (0,): -1, (2,): -1, (2, 3): 1, (4,): 6, (0, 4): -4,
    #  (1, 4): -4, (5,): 6, (2, 5): -4, (3, 5): -4, (4, 5): 1}
    obj_value, sol = qubo_solver(Q)
    print(sol)
    # {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0}
    solution = H.convert_solution(sol)
    print(solution)
    # {'b': 1, 'a': 1, 'c': 1, 'd': 0}


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

    # enforce that z_0 + b z_1 == 0 with penalty a
    H = HOIO().add_constraint_eq_zero({(0,): 1, (1,): b}, lam=a)
    print(H)
    # {(): a*(b**2 + 1), (0, 1): 2*a*b}
    H_subs = H.subs({b: 1})
    print(H_subs)
    # {(): 2*a, (0, 1): 2*a}
    H_subs_p = H.subs({a: 2, b: 1})
    print(H_subs_p)
    # {(): 4, (0, 1): 4}


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


Convert common problems to QUBO form.
-------------------------------------

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
