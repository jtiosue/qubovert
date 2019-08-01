.. image:: https://travis-ci.com/jiosue/QUBOVert.svg?branch=master
    :target: https://travis-ci.com/jiosue/QUBOVert

========
QUBOVert
========

Convert common problems to QUBO form.
-------------------------------------

So far we have just implemented some of the formulations from [Lucas]. The goal of QUBOVert is to become a large collection of problems mapped to QUBO and Ising forms in order to aid the recent increase in study of these problems due to quantum optimization algorithms. I am hoping to have a lot of participation so that we can compile all these problems!

To participate, fork the repository, add your contributions, and submit a pull request. Add tests for any functionality that you add. Make sure you run ``python -m pytest``, ``python -m pytest --codestyle`` before committing anything (yes, even the `tests` need to pass codestyle checks), and ``python -m pydocstyle convention=numpy qubovert`` to ensure that the build passes. When you push changes to the master branch, Travis-CI will automatically check to see if all the tests pass. Note that all problems should be derived from the ``qubovert.utils.Problem`` class! Make sure all your docstrings follow the Numpydoc standard format.


Use Python's ``help`` function! I have very descriptive doc strings on all the functions and classes. To install from source:

.. code:: shell

  git clone https://github.com/jiosue/qubovert.git
  cd QUBOVert
  pip install -e .


Then you can use it in Python with

.. code:: python

    import qubovert

    # get info
    help(qubovert)

    # see all the problems specified
    print(qubovert.__all__)
    # or equivalently
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


See the following Set Cover example. All other problems can be used in a similar way.

.. code:: python

    from qubovert import SetCover
    from any_module import qubo_solver
    # or you can use my bruteforce solver...
    # from qubovert.utils import solve_qubo_bruteforce as qubo_solver

    U = {"a", "b", "c", "d"}
    V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]

    problem = SetCover(U, V)
    Q, offset = problem.to_qubo()

    obj, sol = qubo_solver(Q)
    obj += offset

    solution = problem.convert_solution(sol)

    print(solution) # will print {0, 2}
    print(problem.is_solution_valid(solution)) # will print True, since V[0] + V[2] covers all of U
    print(obj == len(solution)) # will print True

To use the Ising formulation instead:

.. code:: python

    from qubovert import SetCover
    from any_module import ising_solver
    # or you can use my bruteforce solver...
    # from qubovert.utils import solve_ising_bruteforce as ising_solver

    U = {"a", "b", "c", "d"}
    V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]

    problem = SetCover(U, V)
    h, J, offset = problem.to_ising()

    obj, sol = ising_solver(h, J)
    obj += offset

    solution = problem.convert_solution(sol)

    print(solution) # will print {0, 2}
    print(problem.is_solution_valid(solution)) # will print True, since V[0] + V[2] covers all of U
    print(obj == len(solution)) # will print True


To see problem specifics, run

.. code:: python

    help(qubovert.SetCover)
    help(qubovert.VertexCover)
    # etc

I have very descriptive doc strings that should explain everything you need to know to use each problem class.


Technical details on the conversions
------------------------------------
For the log trick he mentions, we usually need a constraint like

.. math::

    \sum_{i} x_i \geq 1.

In order to enforce this constraint, we add a penalty to the QUBO of the form

.. math::

    1 - \sum_i x_i + \sum_{i < j} x_i x_j

(the idea comes from [Glover et al]).



References
----------

.. [Lucas] Andrew Lucas. Ising formulations of many np problems. Frontiers in Physics, 2:5, 2014.

.. [Glover et al]  Fred Glover, Gary Kochenberger, and Yu Du. A tutorial on formulating and using qubo models. arXiv:1811.11538v5, 2019.