from QUBOConvert.utils import solve_qubo_bruteforce, solve_ising_bruteforce


def test_solve_qubo_bruteforce():
    
    Q = {(0, 1): 1, (1, 2): 1, (1, 1): -1, (2, 2): -2}
    assert solve_qubo_bruteforce(Q) == (-2, [0, 0, 1])
    
    
def test_solve_ising_bruteforce():
    
    h = {1: -1, 2: -2}
    J = {(0, 1): 1, (1, 2): 1}
    assert solve_ising_bruteforce(h, J) == (-3, [-1, 1, 1])
