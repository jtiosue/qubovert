def solve_qubo_bruteforce(Q):
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
    
    returns a tuple (objective, solution).
        objective is a float equal to
            sum(Q[(i, j)] * solution[i] * solution[j])
        solution is a list, the value of each binary variable.
    """
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
            
    return best[0], [int(i) for i in best[1]]
