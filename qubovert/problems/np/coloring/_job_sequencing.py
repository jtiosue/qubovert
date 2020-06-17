#   Copyright 2020 Joseph T. Iosue
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""_job_sequencing.py.

Contains the JobSequencing class.
See ``help(qubovert.problems.JobSequencing)``.

"""

from math import log2
from qubovert.utils import (
    QUBOMatrix, decimal_to_boolean, is_solution_spin, spin_to_boolean
)
from qubovert.problems import Problem


__all__ = 'JobSequencing',


class JobSequencing(Problem):
    """JobSequencing.

    Class to manage converting Job Sequencing to and from its QUBO and
    QUSO formluations. Based on the paper "Ising formulations of many
    NP problems", hereforth designated as [Lucas].

    The goal of the JobSequencing problem is as follows. Given workers and
    jobs, where each job has a designated length, assign each of the jobs to
    one of the workers such that largest total length assigned to a worker is
    minimized.

    This class inherits some methods and attributes from the
    ``qubovert.problems.Problem`` class.

    Example
    -------
    >>> from qubovert.problems import JobSequencing
    >>> from any_module import qubo_solver
    >>> # or you can use my bruteforce solver...
    >>> # from qubovert.utils import solve_qubo_bruteforce as qubo_solver

    >>> job_lengths = {"job1": 2, "job2": 3, "job3": 1}
    >>> num_workers = 2
    >>> problem = JobSequencing(job_lengths, num_workers)
    >>> Q = problem.to_qubo()
    >>> obj, sol = qubo_solver(Q)
    >>> solution = problem.convert_solution(sol)

    >>> print(solution)
    ({'job1', 'job3'}, {'job2'})  # or ({'job2'}, {'job1', 'job3'})
    >>> print(problem.is_solution_valid(solution))
    True  # since each job is covered exactly once
    >>> print(
            obj ==
            max(sum(job_lengths[i] for i in x) for x in solution) ==
            3
        )
    True

    """

    def __init__(self, job_lengths, num_workers, log_trick=True, M=None):
        """__init__.

        The goal of the JobSequencing problem is as follows. Given workers and
        jobs, where each job has a designated length, assign each of the jobs
        to  one of the workers such that largest total length assigned to a
        worker is minimized.

        Parameters
        ----------
        job_lengths : tuple or list of integers, or a dict.
            The length of each job. If ``job_lengths`` is a ``dict`` then
            each key is a job that maps to its length. Otherwise,
            the index of ``job_lengths`` is the job and it is mapped to its
            length. Note that the lengths must be integers!
        num_workers : int.
            The number of workers that can be assigned jobs.
        log_trick : boolean (optional, defaults to True).
            Indicates whether or not to use the log trick discussed in [Lucas].
        M : int (optional, defaults to None). We recommend not adjusting this.
            The maximum expected numbers of jobs assigned to a single worker,
            see [Lucas]. If M is None, then it will be set to the value
            required to ensure accuracy of the model. To possibly sacrifice
            accuracy but decrease the number of variables in the model, you can
            set M to something smaller.

        """
        self._log_trick = log_trick
        self._m = int(num_workers)

        if isinstance(job_lengths, dict):
            self._lengths = job_lengths.copy()
            self._input_type = dict
        else:
            self._lengths = dict(enumerate(job_lengths))
            self._input_type = type(job_lengths)

        self._max_L = max(self._lengths.values())
        self._N = len(self._lengths)
        self._M = self._N * self._max_L if M is None else M
        self._log_M = int(log2(self._M)) + 1

        # map job names to integers
        self._job_to_int = {v: k for k, v in enumerate(self._lengths.keys())}

    @property
    def job_lengths(self):
        """job_lengths.

        A copy of the job lengths.

        Return
        ------
        job_lengths : tuple or list of integers, or dict.
            Will be whichever type was inputted in the initialization of the
            class. The length of each job. If ``job_lengths`` is a ``dict``
            then each key is a job that maps to its length. Otherwise,
            the index of ``job_lengths`` is the job and it is mapped to its
            length.

        """
        if self._input_type == dict:
            return self._lengths.copy()
        else:
            return self._input_type(
                self._lengths[x] for x in sorted(self._lengths.keys())
            )

    @property
    def num_workers(self):
        """num_workers.

        A copy of the number of workers.

        Return
        ------
        num_workers : int.
            The number of workers allowed for the problem.

        """
        return self._m

    @property
    def log_trick(self):
        """log_trick.

        A copy of the value for log_trick, indicating whether or not to use
        the log trick. See [Lucas].

        Return
        ------
        log_trick : bool.

        """
        return self._log_trick

    @property
    def M(self):
        """M.

        A copy of the ``M`` constant.

        Return
        ------
        M : int.
            If ``M`` was supplied in the initialization of the class, then this
            will return the same. Otherwise, it will return the value that the
            class calculated for its default.

        """
        return self._M

    @property
    def num_binary_variables(self):
        """num_binary_variables.

        The number of binary variables that the QUBO and QUSO use.

        Return
        ------
        num :  int.
            The number of variables in the QUBO/QUSO formulation.

        """
        if self._log_trick:
            return self._m * self._N + (self._m - 1) * self._log_M
        return self._m * self._N + (self._m - 1) * self._M

    def _x(self, job, worker):
        """_x.

        Return the integer index for the job ``job`` and the worker ``worker``.
        This is equivalent to the ``x`` binary variable in [Lucas].

        Parameters
        ----------
        job : key of self._lengths.
            Job to be covered.
        worker : int.
            Worker to cover the job. ``0 <= worker < self._m``

        Return
        ------
        x : int >= 0.
            Index mapping of the binary variable.

        """
        # same regardless of log trick
        return self._job_to_int[job] * self._m + worker

    def _y(self, i, worker):
        """_y.

        Return the integer index for the slack variables ``y`` in [Lucas].
        The ``y``'s are defined to be >= 0.

        Parameters
        ----------
        i : int.
            Index of the slack variable.
        worker : int.
            Worker to cover the job. ``0 <= worker < self._m``

        Return
        ------
        y : int >= max(self._x(job, worker)).
            Index mapping of the slack variable.

        """
        # same regardless of log trick
        # all the minuses are because we don't need a y(i, 0)!
        return self._N * self._m + i * (self._m - 1) + worker - 1

    def to_qubo(self, A=None, B=1):
        r"""to_qubo.

        Create and return the job sequencing problem in QUBO form following
        section 6.3 of [Lucas]. The Q matrix for the QUBO
        will be returned as an uppertriangular dictionary. Thus, the problem
        becomes minimizing :math:`\sum_{i \leq j} x_i x_j Q_{ij}`. A and B are
        parameters to enforce constraints.

        If all the constraints are satisfied, then the objective function will
        be equal to the total length of the scheduling.

        Parameters
        ----------
        A: positive float (optional, defaults to None).
            ``A`` enforces the constraints. If ``A is None``, then ``A`` will
            be chosen such that the minimum of the QUBO is `guaranteed` to
            satisfy the constraints (``A = B*max(L)``). This may not be the
            best value for any particular QUBO solver, since it may cause a non
            smooth landscape that is hard to minimize.
        B: positive float (optional, defaults to 1).
            ``B`` is the constant in front of the portion of the QUBO to
            minimize.

        Return
        ------
        Q : qubovert.utils.QUBOMatrix object.
            The upper triangular QUBO matrix, a QUBOMatrix object.
            For most practical purposes, you can use QUBOMatrix in the
            same way as an ordinary dictionary. For more information,
            see ``help(qubovert.utils.QUBOMatrix)``.

        """
        # all naming conventions follow the paper listed in the docstring

        if A is None:
            A = B * self._max_L

        Q = QUBOMatrix()

        Q += self._N * A  # offset comes from the first constraint

        # encode H_B (equation 55)
        # minimize worker 0's length
        for job, length in self._lengths.items():
            ind = self._x(job, 0)  # worker zero
            Q[(ind,)] += B * length

        # encode H_A (equation 54)

        # enforce that each job is covered exactly once.
        for job in self._lengths:
            for worker in range(self._m):
                ind = self._x(job, worker)
                Q[(ind,)] -= 2 * A
                for workerp in range(self._m):
                    indp = self._x(job, workerp)
                    Q[(ind, indp)] += A

        # enforce worker 0's length is larger than all the other workers'
        # lengths
        max_M = self._log_M if self._log_trick else self._M
        for worker in range(1, self._m):  # exclude worker 0

            for n in range(max_M):
                ind = self._y(n, worker)
                for np in range(max_M):
                    indp = self._y(np, worker)
                    if self._log_trick:
                        Q[(ind, indp)] += A * pow(2, n + np)
                    else:
                        Q[(ind, indp)] += A * (n + 1) * (np + 1)

                for job, length in self._lengths.items():
                    ind1, ind2 = self._x(job, worker), self._x(job, 0)
                    val = 2 * A * length * (
                        pow(2, n) if self._log_trick else n + 1
                    )
                    Q[(ind, ind1)] += val
                    Q[(ind, ind2)] -= val

            for job, length in self._lengths.items():
                ind1, ind2 = self._x(job, worker), self._x(job, 0)
                for jobp, lengthp in self._lengths.items():
                    ind1p, ind2p = self._x(jobp, worker), self._x(jobp, 0)
                    val = A * length * lengthp
                    Q[(ind1, ind1p)] += val
                    Q[(ind2, ind2p)] += val
                    Q[(ind2, ind1p)] -= val
                    Q[(ind1, ind2p)] -= val

        return Q

    def convert_solution(self, solution, spin=False):
        """convert_solution.

        Convert the solution to the QUBO or QUSO to the solution to the Job
        Sequencing problem.

        Parameters
        ----------
        solution : iterable or dict.
            The QUBO or QUSO solution output. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or 1 or -1 for QUSO), or it can be a dictionary that maps the
            label of the variable to is value.
        spin : bool (optional, defaults to False).
            `spin` indicates whether ``solution`` is the solution to the
            boolean {0, 1} formulation of the problem or the spin {1, -1}
            formulation of the problem. This parameter usually does not matter,
            and it will be ignored if possible. The only time it is used is if
            ``solution`` contains all 1's. In this case, it is unclear whether
            ``solution`` came from a spin or boolean formulation of the
            problem, and we will figure it out based on the ``spin`` parameter.

        Returns
        -------
        res : tuple of sets.
            Each element of the tuple corresponds to a worker. Each element
            of the tuple is a set of jobs that are assigned to that worker.

        """
        if is_solution_spin(solution, spin):
            solution = spin_to_boolean(solution)
        res = tuple(set() for _ in range(self._m))
        for worker in range(self._m):
            for job in self._lengths:
                if solution[self._x(job, worker)] == 1:
                    res[worker].add(job)
        return res

    def is_solution_valid(self, solution, spin=False):
        """is_solution_valid.

        Returns whether or not the proposed solution completes all the jobs
        exactly once.

        Parameters
        ----------
        solution : iterable or dict.
            solution can be the output of JobSequencing.convert_solution,
            or the  QUBO or QUSO solver output. The QUBO solution output
            is either a list or tuple where indices specify the label of the
            variable and the element specifies whether it's 0 or 1 for QUBO
            (or 1 or -1 for QUSO), or it can be a dictionary that maps the
            label of the variable to is value.
        spin : bool (optional, defaults to False).
            `spin` indicates whether ``solution`` is the solution to the
            boolean {0, 1} formulation of the problem or the spin {1, -1}
            formulation of the problem. This parameter usually does not matter,
            and it will be ignored if possible. The only time it is used is if
            ``solution`` contains all 1's. In this case, it is unclear whether
            ``solution`` came from a spin or boolean formulation of the
            problem, and we will figure it out based on the ``spin`` parameter.

        Return
        ------
        valid : boolean.
            True if the proposed solution is valid, else False.

        """
        converted = isinstance(solution, tuple) and all(
            isinstance(x, set) for x in solution
        )
        if not converted:
            solution = self.convert_solution(solution, spin)

        all_jobs, completed_jobs = set(self._lengths.keys()), set()

        for worker in solution:
            for job in worker:
                if job in completed_jobs:  # job covered more than once.
                    return False
                completed_jobs.add(job)

        return completed_jobs == all_jobs  # all jobs must be covered.

    def solve_bruteforce(self, all_solutions=False):
        """solve_bruteforce.

        Solves the JobSequence problem exactly with a brute force method. THIS
        SHOULD NOT BE USED FOR LARGE PROBLEMS! The advantage over this method
        as opposed to using a brute force QUBO solver is that the QUBO
        formulation has many slack variables.

        Parameters
        ----------
        all_solutions : boolean (optional, defaults to False).
            If ``all_solutions`` is set to True, all the best solutions to the
            problem will be returned rather than just one of the best. If the
            problem is very big, then it is best if ``all_solutions == False``,
            otherwise this function will use a lot of memory.

        Returns
        -------
        res : tuple of sets or list of tuple of sets.
            Each element of the tuple corresponds to a worker. Each element
            of the tuple is a set of jobs that are assigned to that worker. If
            ``all_solutions == True`` then ``res`` will be a list, where each
            element of the list will be one of the optimal solutions.

        Example
        -------
        >>> job_lengths = {"job1": 2, "job2": 3, "job3": 1}
        >>> num_workers = 2
        >>> problem = JobSequencing(job_lengths, num_workers)
        >>> print(problem.solve_bruteforce())
        ({'job1', 'job3'}, {'job2'})  # or ({'job2'}, {'job1', 'job3'})

        >>> print(problem.solve_bruteforce(True))
        [({'job1', 'job3'}, {'job2'}), ({'job2'}, {'job1', 'job3'})]

        """
        best = None, None
        all_sols, n = {}, self._m * self._N
        for i in range(1 << n):
            sol = self.convert_solution(decimal_to_boolean(i, n))
            if self.is_solution_valid(sol):
                obj = max(
                    sum(self._lengths[job] for job in cluster)
                    for cluster in sol
                )
                if not all_solutions and (best[0] is None or obj < best[0]):
                    best = obj, sol
                elif all_solutions and (best[0] is None or obj <= best[0]):
                    best = obj, sol
                    all_sols.setdefault(obj, []).append(sol)

        if all_solutions:
            return all_sols[best[0]]
        return best[1]
