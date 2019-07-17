def solve_qubo_bruteforce(Q, offset=0):
    """
    Iterate through all the possible solutions to a QUBO formulated problem
    and find the best one (the one that gives the minimum objective vale). Do 
    not use for large problem sizes! This is meant only for testing very small 
    problems.
    
    For example, to find the minimum of the problem
        obj = x_0*x_1 + x_1*x_2 - x_1 - 2*x_2,
    the Q dictionary is
        >>> Q = {(0, 1): 1, (1, 2): 1, (1, 1): -1, (2, 2): -2}.
    Then to solve this Q, run
        >>> obj_val, solution = solve_qubo_bruteforce(Q)
    obj_val will be the smallest value of obj, for this example it will be -2. 
    solution will be a list that indicated what each of x_0, x_1, and x_2 are 
    for the solution. In this case, x = [0, 0, 1], indicating that x_0 is 0,
    x_1 is 0, x_2 is 1.
    
    Q: dictionary mapping binary variables indices to the Q value. Note that
        binary variable indices must be integer labeled starting from 0.
    offset: an optional float, the part of the objective function that does 
        not depend on the variables.
    
    returns a tuple (objective, solution).
        objective is a float equal to
            sum(Q[(i, j)] * solution[i] * solution[j]) + offset
        solution is a list, the value of each binary variable.
    """
    if not Q: return offset, []
    
    N = max(y for x in Q for y in x) + 1
    obj = lambda x: sum(
        Q[(i, j)]*int(x[i])*int(x[j]) for i, j in Q
    )
    
    best = None, None
    
    for n in range(1 << N):
        x = ("{0:0%db}" % N).format(n)
        v = obj(x)
        if best[0] is None or v < best[0]:
            best = v, x
            
    return best[0] + offset, [int(i) for i in best[1]]


def solve_ising_bruteforce(h, J, offset=0):
    """
    Iterate through all the possible solutions to a Ising formulated problem
    and find the best one (the one that gives the minimum objective value). Do 
    not use for large problem sizes! This is meant only for testing very small 
    problems.
    
    For example, to find the minimum of the problem
        obj = z_0*z_1 + z_1*z_2 - z_1 - 2*z_2,
    the Ising formulation is
        >>> h = {1: -1, 2: -2}
        >>> J = {(0, 1): 1, (1, 2): 1}
    Then to solve this problem, run
        >>> obj_val, solution = solve_ising_bruteforce(h, J)
    obj_val will be the smallest value of obj, for this example it will be -3. 
    solution will be a list that indicated what each of z_0, z_1, and z_2 are 
    for the solution. In this case, z = [-1, 1, 1], indicating that z_0 is -1,
    z_1 is 1, and z_2 is 1.
    
    h: dictionary mapping spins indices to the field value. Note that
        spin variable indices must be integer labeled starting from 0.
    J: dictionary mapping tuples of spin indices to the coupling value. Note 
        that spin variable indices must be integer labeled starting from 0.
    offset: an optional float, the part of the objective function that does 
        not depend on the variables.
    
    returns a tuple (objective, solution).
        objective is a float equal to
            sum(J[(i, j)] * solution[i] * solution[j]) + 
            sum(h[i] * solution[i]) + 
            offset
        solution is a list, the value of each binary variable.
    """
    if h and J: N = max(max(y for x in J for y in x), max(h.keys())) + 1
    elif h: N = max(h.keys()) + 1
    elif J: N = max(y for x in J for y in x) + 1
    else: return offset, []
    
    to_spin = lambda x: 2 * int(x) - 1
    
    obj = lambda x: sum(
        J[(i, j)]*to_spin(x[i])*to_spin(x[j]) for i, j in J
    ) + sum(
        h[i]*to_spin(x[i]) for i in h
    )
    
    best = None, None
    
    for n in range(1 << N):
        x = ("{0:0%db}" % N).format(n)
        v = obj(x)
        if best[0] is None or v < best[0]:
            best = v, x
            
    return best[0] + offset, [to_spin(i) for i in best[1]]
