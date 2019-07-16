from QUBOConvert.utils import solve_qubo_bruteforce

def test_solve_qubo_bruteforce():
    
    Q = {(0, 1): 1, (1, 2): 1, (1, 1): -1, (2, 2): -2}
    assert solve_qubo_bruteforce(Q) == (-2, [0, 0, 1])
