from numpy import log2
from .utils import qubo_conversion, QUBOMatrix


class SetCover(qubo_conversion):
    
    """
    Class to manage converting Set Cover to and from its QUBO and
    Ising formluations.
    """
    
    def __init__(self, U, V):
        """
        The goal of the SetCover problem is to find the smallest number of 
        elements in V such that union over the elements equals U. All naming
        conventions follow the names in the paper 
        https://arxiv.org/pdf/1302.5843.pdf.
        
        U: set, the set of all elements to cover.
        V: dictionary that maps a name to a subset of U.
        """
        self._U = U.copy()
        self._V = type(V)(x.copy() for x in V)
        self._N, self._n = len(self.V), len(self.U)
        self._M = max(
            sum(int(alpha in v) for v in self.V)
            for alpha in self.U
        )
        self._log_M = int(log2(self._M))+1
        super().__init__(self.U, self.V)
        
    @property
    def U(self):
        """
        A copy of the U set. Updating the copy will not update the 
        instance U.
        """
        return self._U.copy()
    
    @property
    def V(self):
        """
        A copy of the V list/tuple. Updating the copy will not update the 
        instance V.
        """
        return type(self._V)(x.copy() for x in self._V)
        
    def to_qubo(self, A=2, B=1, log_trick=True):
        """
        Create and return the set cover problem in QUBO form following section 
        5.1 of https://arxiv.org/pdf/1302.5843.pdf. The Q matrix for the QUBO 
        will be returned as an uppertriangular dictionary. Thus, the problem 
        becomes minimizing sum_{i <= j} x[i] x[j] Q[(i, j)]. A and B are
        parameters to enforce constraints.
        
        A: positive float, defaults to 2. See section 5.1 of 
            https://arxiv.org/pdf/1302.5843.pdf
        B: positive float that is less than A, defaults to 1. See section 5.1 of 
            https://arxiv.org/pdf/1302.5843.pdf
        log_trick: boolean, indicates whether or not to use the log trick
            discussed in the paper. Defaults to True.
            
        returns the tuple (Q, offset).
            Q is the upper triangular QUBO matrix, a QUBOMatrix object.
                For most practical purposes, you can use QUBOMatrix in the 
                same way as an ordinary dictionary. For more information,
                see help(QUBOConvert.utils.QUBOMatrix).
            offset is a float. It is the sum of the terms in the formulation in
                the cited paper that don't involve any variables.
        """
        # all naming conventions follow the paper listed in the docstring
        
        alpha_2_index = {alpha: i for i, alpha in enumerate(self._U)}
        filtered_range = lambda start=0: filter(
            lambda k: alpha in self._V[k], range(start, self._N)
        )
        
        Q = QUBOMatrix()
        
        offset = self._n * A  # comes from the first constraint
        
        # encode H_B (equation 46)
        for i in range(self._N): Q[(i, i)] += B
        
        
        ## encode H_A
            
        x = lambda alpha, m: (
            self._N + alpha_2_index[alpha] + self._n*(m if log_trick else m-1)
        )
            
            
        for alpha in self._U:
        
            if not log_trick: # (Equation 45)
                
                # first constraint
                for m in range(1, self._M+1):
                    i = x(alpha, m)
                    Q[(i, i)] -= A
                    for mp in range(m+1, self._M+1):
                        ip = x(alpha, mp)
                        Q[(i, ip)] += 2*A
                        
                # second constraint
                for m in range(1, self._M+1):
                    i = x(alpha, m)
                    Q[(i, i)] += A*m*m
                    for mp in range(m+1, self._M+1):
                        ip = x(alpha, mp)
                        Q[(i, ip)] += 2*A*m*mp
                        
                    for j in filtered_range():
                        Q[(j, i)] -= 2*A*m
                    
            else: # using the log_trick
                
                # first constraint
                for m in range(self._log_M+1):
                    i = x(alpha, m)
                    Q[(i, i)] -= A
                    for mp in range(m+1, self._log_M+1):
                        ip = x(alpha, mp)
                        Q[(i, ip)] += A
                
                # second constraint
                for m in range(self._log_M+1):
                    i = x(alpha, m)
                    Q[(i, i)] += A*pow(2, 2*m)
                    for mp in range(m+1, self._log_M+1):
                        ip = x(alpha, mp)
                        Q[(i, ip)] += 2*A*pow(2, m+mp)
                    for j in filtered_range():
                        Q[(j, i)] -= 2*A*pow(2, m)
                
            # for both using and not using the log trick
            for i in filtered_range():
                Q[(i, i)] += A
                for j in filtered_range(i+1): Q[(i, j)] += 2*A
            
        
        return Q, offset
    
    def convert_solution(self, solution):
        """
        Convert the solution to the QUBO or Ising to the solution to the Set 
        Cover problem. 
        
        solution is the QUBO or Ising solution output. The QUBO solution output 
            is either a list where indices specify the label of the binary 
            variable and the element specifies whether it's 0 or 1, or it can 
            be a dictionary that maps the label of the binary variable to 
            whether it is a 0 or 1. The Ising solution output is the same, but
            with -1 corresponding to the QUBO 0, and 1 corresponding to the
            QUBO 1.
        
        returns a set of which sets are included in the set cover. So if this
        function returns {0, 2, 3}, then the set cover is the sets V[0], V[2],
        and V[3].
        """
        return set(i for i in range(self._N) if solution[i] == 1)
    
    def is_solution_valid(self, solution):
        """
        Returns whether or not the proposed solution covers all the elements in
        U.
        
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
            
        covered = set(x for i in solution for x in self._V[i])
        return covered == self._U
    
    def num_binary_variables(self, log_trick=True):
        """
        Find the number of binary variables that the QUBO and Ising use.
        
        log_trick: boolean, indicates whether to use the log trick mentioned
            in the paper. Defaults to True.
        
        returns an integer, the number of variables in the QUBO/Ising 
            formulation.
        """
        if log_trick: return self._N + self._n*(self._log_M+1)
        else: return self._N + self._n*self._M
