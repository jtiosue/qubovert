[![Build Status](https://travis-ci.com/jiosue/QUBOVert.svg?branch=master)](https://travis-ci.com/jiosue/QUBOVert)

# QUBOVert

## Convert common problems to QUBO form.

So far we have just implemented some of the formulations from [Lucas]. The goal of QUBOVert is to become a large collection of problems mapped to QUBO and Ising forms in order to aid the recent increase in study of these problems due to quantum optimization algorithms. I am hoping to have a lot of participation so that we can compile all these problems!

To participate, fork the repository, add your contributions, and submit a pull request. Add tests for any functionality that you add. Make sure you run `python -m pytest` before committing anything to ensure that the build passes. When you push changes to the master branch, Travis-CI will automatically check to see if all the tests pass. 


Use Python's `help` function! I have very descriptive doc strings on all the functions and classes. To install from source:

```shell
git clone https://github.com/jiosue/qubovert.git
cd QUBOVert
pip install -r requirements.txt
pip install -e .
```

Then you can use it in Python with

```python
import qubovert

# get info
help(qubovert)
help(qubovert.utils)
```


See the following Set Cover example.

```python
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
```

To use the Ising formulation instead:

```python
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
```


# Technical details on the conversions

For the log trick he mentions, we usually need a constraint like $$\sum_{i} x_i \geq 1$$. In order to enforce this constraint, we add a penalty to the QUBO of the form $$1 - \sum_i x_i + \sum_{i < j} x_i x_j$$ (the idea comes from [Glover et al]).



# References

[Lucas] Andrew Lucas. Ising formulations of many np problems. Frontiers in Physics, 2:5, 2014.

[Glover et al]  Fred Glover, Gary Kochenberger, and Yu Du. A tutorial on formulating and using qubo models. arXiv:1811.11538v5, 2019.
