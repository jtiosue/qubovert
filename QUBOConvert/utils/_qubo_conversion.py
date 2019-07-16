class qubo_conversion:
    
    """
    This acts a parent class to all the QUBO conversion problem classes.
    The init method keeps track of the problem args. The repr method uses those
    input args, such that eval(repr(cls)) == cls. Finally, we define a __eq__
    method to determine if two problems are the same. The rest of the methods
    are to be implemented in child classes.
    """
    
    def __init__(self, *args):
        """
        This method just keeps track of the input args.
        """
        self._problem_args = args
        
    def __repr__(self):
        """
        Defined such that the following is true (assuming you have imported
        QUBOConvert as QUBOConvert).
            >>> s = Class_derivedfrom_qubo_conversion(*args)
            >>> eval(repr(s)) == s
        """
        return "QUBOConvert." + str(self)
        
    def __str__(self):
        """
        Defined such that the following is true (assuming you have imported
        * from QUBOConvert).
            >>> s = Class_derivedfrom_qubo_conversion(*args)
            >>> eval(repr(s)) == s
        """
        s = self.__class__.__name__ + "("
        for a in self._problem_args: s += str(a) + ", "
        return s[:-2] + ")"
        
    def __eq__(self, other):
        """
        Find if self and other define the same problem. 
        
        other: must be a class derived from qubo_conversion.
        
        returns a boolean.
        """
        return (
            type(self) == type(other) and 
            self._problem_args == other._problem_args
        )
    
    def to_qubo(self, *args, **kwargs):
        """
        Create and return upper triangular QUBO representing the problem. 
        Should be implemented in child classes.
            
        returns the tuple (Q, offset).
            Q is the upper triangular QUBO matrix, a QUBOMatrix object.
                For most practical purposes, you can use QUBOMatrix in the 
                same way as an ordinary dictionary. For more information,
                see help(QUBOConvert.utils.QUBOMatrix).
            offset is a float. It is the sum of the terms in the formulation in
                the cited paper that don't involve any variables.
        """
        raise NotImplementedError("Method to be implemented in child classes")
    
    
    def convert_solution(self, solution):
        """
        Convert the solution to the QUBO to the solution to the problem. 
        Should be implemented in child classes.
        
        solution is the QUBO solution output. The QUBO solution output is
            either a list/tuple where indices specify the label of the binary 
            variable and the element specifies whether it's 0 or 1, or it can 
            be a dictionary that maps the label of the binary variable to 
            whether it is a 0 or 1.
        """
        raise NotImplementedError("Method to be implemented in child classes")
    
    def is_solution_valid(self, solution):
        """
        Returns whether or not the proposed solution is valid. Should be
        implemented in child classes.
        
        solution can either be the output of convert_solution or it
            can be the actual QUBO solution output. The QUBO solution output is
            either a list where indices specify the label of the binary 
            variable and the element specifies whether it's 0 or 1, or it can 
            be a dictionary that maps the label of the binary variable to 
            whether it is a 0 or 1.
            
        returns a boolean, True if the proposed solution is valid, else False.
        """
        raise NotImplementedError("Method to be implemented in child classes")
    
    def num_binary_variables(self, *args, **kwargs):
        """
        Find the number of binary variables that the QUBO uses.
        
        returns an integer, the number of variables in the QUBO formulation.
        """
        raise NotImplementedError("Method to be implemented in child classes")