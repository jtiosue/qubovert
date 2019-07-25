"""
Contains the VertexCover class. See `help(qubovert.VertexCover)`.
"""

from qubovert.utils import Problem, QUBOMatrix


class VertexCover(Problem):

    """
    Class to manage converting Vertex Cover to and from its QUBO and
    Ising formluations. Based on the paper hereforth designated [Lucas]:
    [Andrew Lucas. Ising formulations of many np problems. Frontiers in
    Physics, 2:5, 2014.]

    The goal of the VertexCover problem is to find the smallest number of
    verticies that can be coloredsuch that every edge of the graph is
    incident to a colored vertex.

    Example usage:

        from qubovert import VertexCover
        from any_module import qubo_solver
        # or you can use my bruteforce solver...
        # from qubovert.utils import solve_qubo_bruteforce as qubo_solver

        edges = {("a", "b"), ("a", "c"), ("c", "d"), ("a", "d")}

        problem = VertexCover(edges)
        Q, offset = problem.to_qubo()

        obj, sol = qubo_solver(Q)
        obj += offset

        solution = problem.convert_solution(sol)

        # will print {"a", "c"}
        print(solution)
        # will print True, since each edge is adjacent to either "a" or "c".
        print(problem.is_solution_valid(solution))
        # will print True
        print(obj == len(solution))
    """

    def __init__(self, edges):
        """
        The goal of the VertexCover problem is to find the smallest number of
        verticies that can be coloredsuch that every edge of the graph is
        incident to a colored vertex.  All naming conventions follow the names
        in the paper [Lucas].

        edges: set or tuples describing edges of the graph. For example, any of
            the following work.
                >>> edges = {("a", "b"), ("a", "c")}
                >>> edges = {(0, 1), (0, 2)}
        """
        self._edges = edges.copy()
        self._vertices = {y for x in edges for y in x}
        self._vertex_to_index = {x: i for i, x in enumerate(self._vertices)}
        self._index_to_vertex = {i: x for x, i in
                                 self._vertex_to_index.items()}
        self._N, self._n = len(self._vertices), len(self._edges)

    @property
    def E(self):
        """
        A copy of the set of edges of the graph. Updating the copy will not
        update the instance set.
        """
        return self._edges.copy()

    @property
    def V(self):
        """
        A copy of the vertex set of the graph. Updating the copy will not
        update the instance set.
        """
        return self._vertices.copy()

    def to_qubo(self, A=2, B=1):
        """
        Create and return the vertex cover problem in QUBO form following
        section 4.3 of [Lucas]. The Q matrix for the QUBO
        will be returned as an uppertriangular dictionary. Thus, the problem
        becomes minimizing sum_{i <= j} x[i] x[j] Q[(i, j)]. A and B are
        parameters to enforce constraints.

        A: positive float, enforces the constraints, defaults to 2.
            See section 4.3 of [Lucas].
        B: positive float that is less than A, defaults to 1. See section 5.1
            of [Lucas].

        returns the tuple (Q, offset).
            Q is the upper triangular QUBO matrix, a QUBOMatrix object.
                For most practical purposes, you can use QUBOMatrix in the
                same way as an ordinary dictionary. For more information,
                see help(qubovert.utils.QUBOMatrix).
            offset is a float. It is the sum of the terms in the formulation in
                the cited paper that don't involve any variables. It is
                formatted such that if all the constraints are satisfied, then
                sum_{i <= j} x[i] x[j] Q[(i, j)] + offset will be equal to the
                total number of colored verticies.
        """
        # all naming conventions follow the paper listed in the docstring

        Q = QUBOMatrix()

        offset = self._n * A  # comes from the constraint

        # encode H_B (equation 34)
        for i in range(self._N):
            Q[(i, i)] += B

        # encode H_A

        # Q keeps itself upper triangular, so we don't need to worry about it.
        for u, v in self._edges:
            iu, iv = self._vertex_to_index[u], self._vertex_to_index[v]
            Q[(iu, iv)] += A
            Q[(iu, iu)] -= A
            Q[(iv, iv)] -= A

        return Q, offset

    def convert_solution(self, solution):
        """
        Convert the solution to the QUBO or Ising to the solution to the Vertex
        Cover problem.

        solution is the QUBO or Ising solution output. The QUBO solution output
            is either a list where indices specify the label of the binary
            variable and the element specifies whether it's 0 or 1, or it can
            be a dictionary that maps the label of the binary variable to
            whether it is a 0 or 1. The Ising solution output is the same, but
            with -1 corresponding to the QUBO 0, and 1 corresponding to the
            QUBO 1.

        returns a set of which verticies need to be colored. Thus, if this
            function returns {0, 2}, then this means that vertex 0 and 2
            should be colored.
        """
        if not isinstance(solution, dict):
            solution = dict(enumerate(solution))
        return set(
            self._index_to_vertex[i] for i, x in solution.items() if x == 1
        )

    def is_solution_valid(self, solution):
        """
        Returns whether or not the proposed solution satisfies the constraint
        that every edge has at least one colored vertex.

        solution can either be the output of convert_solution or it
            can be the actual QUBO or Ising solution output. The QUBO solution
            output is either a list where indices specify the label of the
            binary variable and the element specifies whether it's 0 or 1, or
            it can be a dictionary that maps the label of the binary variable
            to whether it is a 0 or 1. The Ising solution output is the same,
            but with -1 corresponding to the QUBO 0, and 1 corresponding to the
            QUBO 1.

        returns a boolean, True if the proposed solution is valid, else False.

        """
        if not isinstance(solution, set):
            solution = self.convert_solution(solution)

        return all(i in solution or j in solution for i, j in self._edges)

    def num_binary_variables(self):
        """
        Find the number of binary variables that the QUBO and Ising use.

        returns an integer, the number of variables in the QUBO/Ising
            formulation.
        """
        return self._N
