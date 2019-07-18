from qubovert.utils import qubo_conversion, IsingCoupling, IsingField


class NumberPartitioning(qubo_conversion):
    
    """
    Class to manage converting the Number Partitioning problem to and from its
    QUBO and Ising formluations. Based on the paper hereforth designated as
    [Lucas]: [Andrew Lucas. Ising formulations of many np problems. Frontiers 
    in Physics, 2:5, 2014.]
    
    Example usage:
        
        from qubovert import NumberPartitioning
        from any_module import qubo_solver
        # or you can use my bruteforce solver...
        # from qubovert.utils import solve_qubo_bruteforce as qubo_solver
        
        S = [1, 2, 3, 4]
        
        problem = NumberPartitioning(S)
        Q, offset = problem.to_qubo()
        
        obj, sol = qubo_solver(Q)
        obj += offset
        
        solution = problem.convert_solution(sol)
        
        # will print ([1, 4], [2, 3]) or ([2, 3], [1, 4])
        print(solution)
        # will print True, since sum([1, 4]) == sum([2, 3])
        print(problem.is_solution_valid(solution))
        # will print True since the solution is valid
        print(obj == 0)
    """
    
    def __init__(self, S):
        """
        The goal of the NumberPartitioning problem is as follows (quoted from
        [Lucas]): 
            
        Given a list of N positive numbers S = [n1, . . . , nN], is there a
        partition of this set of numbers into two disjoint subsets R and S − R,
        such that the sum of the elements in both sets is the same?
        
        Note that if we can't do this partitioning, then the next goal is to
        find a partition that almost does this, ie a partition that minimizes
        the difference in the sum between the two partitions.
            
        All naming conventions follow the names in the paper [Lucas].
        
        S: tuple or list of N positive numbers that we are attempting to 
            partition into two partitions of equal sum.
        """
        self._input_type = type(S)
        self._S = self._input_type(x for x in S) # copy the input
        self._N = len(S)        
        super().__init__(self.S)
        
    @property
    def S(self):
        """
        A copy of the S list. Updating the copy will not update the instance
        list.
        """
        return self._input_type(x for x in self._S)
        
    def to_ising(self, A=1):
        """
        Create and return the number partitioning problem in Ising form 
        following section 2.1 of [Lucas]. The J coupling matrix for the Ising 
        will be returned as an uppertriangular dictionary. Thus, the problem 
        becomes minimizing 
            sum_{i <= j} z[i] z[j] J[(i, j)] + sum_{i} s[i] h[i].
        
        A: positive float, defaults to 1. See section 2.1 of [Lucas].
            
        returns the tuple (h, J, offset).
            h is the Ising field, a IsingField object. For most practical
                purposes, you can use IsingField in the same way as an ordinary
                dictionary. For more information, see 
                help(qubovert.utils.IsingField).
            J is the upper triangular Ising coupling matrix, a 
                IsingCoupling object. For most practical purposes, you can use
                IsingCoupling in the same way as an ordinary dictionary. For
                more information, see help(qubovert.utils.IsingCoupling).
            offset is a float. It is the value such that the solution to the
                Ising formulation is 0 if a valid number partitioning exists.
        """
        h, J = IsingField(), IsingCoupling()
        offset = A * sum(pow(x, 2) for x in self._S)
        
        for i in range(self._N):
            for j in range(i+1, self._N):
                J[(i, j)] += (2 * A * self._S[i] * self._S[j])
        
        return h, J, offset
    
    def convert_solution(self, solution):
        """
        Convert the solution to the QUBO or Ising to the solution to the
        Number Partitioning problem. 
        
        solution is the QUBO or Ising solution output. The QUBO solution output 
            is either a list where indices specify the label of the binary 
            variable and the element specifies whether it's 0 or 1, or it can 
            be a dictionary that maps the label of the binary variable to 
            whether it is a 0 or 1. The Ising solution output is the same, but
            with -1 corresponding to the QUBO 0, and 1 corresponding to the
            QUBO 1.
        
        returns a tuple of two lists/tuples, where each list/tuple is a 
            partition (if the inputted S is a tuple, then the partitions will
            be formatted as tuples, otherwise they will be lists). For
            example, if S is [1, 2, 3, 4], then the solution to the Ising
            problem could be [-1, 1, 1, -1]. The solution [-1, 1, 1, -1] is 
            interpreted as 1 and 4 being one partition, and 2 and 3 being 
            another partition. Then the output of this function would be 
            ([1, 4], [2, 3]) or some ordering variation of that.
        """
        partition1 = self._input_type(
            self._S[i] for i, v in enumerate(solution) if v == 1
        )
        partition2 = self._input_type(
            self._S[i] for i, v in enumerate(solution) if v != 1
        )
        return partition1, partition2
    
    def is_solution_valid(self, solution):
        """
        Returns whether or not the proposed solution partitions S into two sets
        of equal sum.
        
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
        if (not isinstance(solution, tuple) or len(solution) != 2 or 
            not isinstance(solution[0], self._input_type) or 
            not isinstance(solution[1], self._input_type)):
            
            solution = self.convert_solution(solution)
            
        return sum(solution[0]) == sum(solution[1])
    
    def num_binary_variables(self):
        """
        Find the number of binary variables that the QUBO and Ising use.
        
        returns an integer, the number of variables in the QUBO/Ising 
            formulation.
        """
        return self._N
