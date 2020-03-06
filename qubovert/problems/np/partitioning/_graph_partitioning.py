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

"""_graph_partitioning.py.

Contains the GraphPartitioning class.
See ``help(qubovert.problems.GraphPartitioning)``.

"""

from qubovert.utils import QUSOMatrix
from qubovert import PCSO
from qubovert.problems import Problem


__all__ = 'GraphPartitioning',


class GraphPartitioning(Problem):
    """GraphPartitioning.

    Class to manage converting (Weighted) Graph Partitioning to and from its
    QUBO and QUSO formluations. Based on the paper "Ising formulations of many
    NP problems", hereforth designated [Lucas].

    The goal of the Graph Partitioning problem is to partition the verticies
    of a graph into two equal subsets such that the number of edges (or the
    total weight of the edges) connecting the two subsets is minimized.

    GraphPartitioning inherits some methods and attributes from the Problem
    class. See ``help(qubovert.problems.Problem)``.

    Example
    -------
    >>> from qubovert.problems import GraphPartitioning
    >>> from any_module import qubo_solver
    >>> # or you can use my bruteforce solver...
    >>> # from qubovert.utils import solve_qubo_bruteforce as qubo_solver
    >>> edges = {("a", "b"), ("a", "c"), ("c", "d"),
                 ("b", "c"), ("e", "f"), ("d", "e")}
    >>> problem = GraphPartitioning(edges)
    >>> Q = problem.to_qubo()
    >>> obj, sol = qubo_solver(Q)
    >>> solution = problem.convert_solution(sol)

    >>> print(solution)
    ({'a', 'b', 'c'}, {'d', 'e', 'f'})

    >>> print(problem.is_solution_valid(solution))
    True

    This is True since the number of vertices in the first partition is equal
    to the number of vertices in the second partition.

    >>> print(obj)
    1

    This is 1 because there is 1 edge connecting the partitions.

    """

    def __init__(self, edges):
        """__init__.

        The goal of the (Weighted) Graph Partitioning problem is to partition
        the vertices of a graph into two equal subsets such that the number of
        edges (or the total weight of the edges) connecting the two subsets is
        minimized. All naming conventions follow the names in the paper
        [Lucas].

        Parameters
        ----------
        edges : set or dict.
            If edges is a set, then it must be a set of two element tuples
            describing the edges of the graph. Ie each tuple is a connection
            between two vertices. If a tuple has a repeated label (for example,
            (2, 2)), it will be ignored.

            If edges is a dict then the keys must be
            two element tuples and the values are the weights associated with
            that edge. If a key has a repeated label (for example, (2, 2)), it
            will be ignored.

        Examples
        --------
        >>> edges = {("a", "b"), ("a", "c")}
        >>> problem = GraphPartitioning(edges)

        >>> edges = {(0, 1), (0, 2)}
        >>> problem = GraphPartitioning(edges)

        >>> edges = {(0, 1): 2, (1, 2): -1}
        >>> problem = GraphPartitioning(edges)

        """
        if isinstance(edges, set):
            self._edges = {k: 1 for k in edges if k[0] != k[1]}
        else:
            self._edges = {k: v for k, v in edges.items() if k[0] != k[1]}

        self._vertices = {y for x in edges for y in x}
        self._vertex_to_index = {x: i for i, x in enumerate(self._vertices)}
        self._index_to_vertex = {i: x for x, i in
                                 self._vertex_to_index.items()}
        self._N = len(self._vertices)

        all_degs = {}
        for e in edges:
            for q in e:
                all_degs[q] = all_degs.setdefault(q, 0) + 1
        self._degree = max(all_degs.values()) if all_degs else 0

    @property
    def E(self):
        """E.

        A copy of the set of edges of the graph. Updating the copy will not
        update the instance set.

        Return
        ------
        E : set of two element tuples.
            A copy of the edge set defining the Graph Partitioning problem.

        """
        return set(self._edges.keys())

    @property
    def V(self):
        """V.

        A copy of the vertex set of the graph. Updating the copy will not
        update the instance set.

        Returns
        -------
        V : set.
            A copy of the set of vertices corresponding to the edge set for the
            Graph Partitioning problem.

        """
        return self._vertices.copy()

    @property
    def weights(self):
        """weights.

        Returns a dictionary mapping the edges of the graph to their associated
        weights.

        Return
        ------
        weights : dict.
            Keys are two element tuples, values are numbers.

        """
        return self._edges.copy()

    @property
    def degree(self):
        """degree.

        The maximum degree of the graph.

        Returns
        -------
        deg : int.
            A copy of the variable of the maximal degree of the graph.

        """
        return self._degree

    @property
    def num_binary_variables(self):
        """num_binary_variables.

        The number of binary variables that the QUBO and QUSO use.

        Return
        ------
        num : integer.
            The number of variables in the QUBO/QUSO formulation.

        """
        return self._N

    def to_quso(self, A=None, B=1):
        r"""to_quso.

        Create and return the graph partitioning problem in QUSO form
        following section 2.2 of [Lucas]. A and B are parameters to enforce
        constraints.

        It is formatted such that the solution to the QUSO formulation is
        equal to the the total number of edges connecting the two
        partitions (or the total weight if we are solving weighted
        partitioning).

        Parameters
        ----------
        A: positive float (optional, defaults to None).
            A enforces the constraints. If it is None, then A will be chosen
            to enforce hard constraints (equation 10 in [Lucas]). Note that
            this may not be optimal for a solver, often hard constraints result
            in unsmooth energy landscapes that are difficult to minimize. Thus
            it may be useful to play around with the magnitude of this value.
        B: positive float (optional, defaults to 1).
            Constant in front of the objective function to minimize. See
            section 2.2 of [Lucas].

        Return
        ------
        L : qubovert.utils.QUSOMatrix object.
            For most practical purposes, you can use QUSOMatrix in the
            same way as an ordinary dictionary. For more information, see
            ``help(qubovert.utils.QUSOMatrix)``.

        Example
        -------
        >>> problem = GraphPartitioning({(0, 1), (1, 2), (0, 3)})
        >>> L = problem.to_quso()

        """
        # all naming conventions follow the paper listed in the docstring
        if A is None:
            A = min(2*self._degree, self._N) * B / 8

        L = QUSOMatrix()

        # encode H_A (equation 8)
        L += PCSO().add_constraint_eq_zero(
            {(i,): 1 for i in range(self._N)}, lam=A)

        # encode H_B (equation 9)
        L += B * sum(self._edges.values()) / 2
        for (u, v), w in self._edges.items():
            L[(self._vertex_to_index[u],
              self._vertex_to_index[v])] -= w * B / 2

        return L

        # slower because we convert to PCSO and then to QUSOMatrix
        # H = PCSO()
        # H.set_mapping(self._vertex_to_index)

        # # encode H_A (equation 8)
        # H.add_constraint_eq_zero({(i,): 1 for i in self._vertices}, lam=A)

        # # encode H_B (equation 9)
        # H += B * sum(self._edges.values()) / 2
        # for e, w in self._edges.items():
        #     H[e] -= w * B / 2

        # return H.to_quso()

    def convert_solution(self, solution, spin=False):
        """convert_solution.

        Convert the solution to the QUBO or QUSO to the solution to the
        Graph Partitioning problem.

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

        Return
        ------
        res: tuple of sets (partition1, partition2).
            partition1 : set.
                The first partition of verticies.
            partition2 : set.
                The second partition.

        Example
        -------
        >>> edges = {("a", "b"), ("a", "c"), ("c", "d"),
                     ("b", "c"), ("e", "f"), ("d", "e")}
        >>> problem = GraphPartitioning(edges)
        >>> Q = problem.to_qubo()
        >>> obj, sol = solve_qubo(Q)
        >>> print(problem.convert_solution(sol))
        ({'a', 'b', 'c'}, {'d', 'e', 'f'})

        """
        if not isinstance(solution, dict):
            solution = dict(enumerate(solution))

        partition1 = set(
            self._index_to_vertex[i] for i, v in solution.items() if v == 1
        )
        partition2 = set(
            self._index_to_vertex[i] for i, v in solution.items() if v != 1
        )
        return partition1, partition2

    def is_solution_valid(self, solution, spin=False):
        """is_solution_valid.

        Returns whether or not the proposed solution has an equal number of
        vertices in each partition. NOTE: this is impossible if the number of
        edges is odd!

        Parameters
        ----------
        solution : iterable or dict.
            solution can be the output of GraphPartitioning.convert_solution,
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
        not_converted = (
            not isinstance(solution, tuple) or len(solution) != 2 or
            not isinstance(solution[0], set) or
            not isinstance(solution[1], set)
        )

        if not_converted:
            solution = self.convert_solution(solution, spin)

        return len(solution[0]) == len(solution[1])
