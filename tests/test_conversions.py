from QUBOConvert.utils import qubo_to_ising, ising_to_qubo


def test_qubo_to_ising_to_qubo():
    
    qubo_args = Q, offset = {(0, 0): 1, (0, 1): 1, (1, 1): -1, (1, 2): .2}, 3
    
    assert (Q, offset) == ising_to_qubo(*qubo_to_ising(*qubo_args))
    
    
def test_ising_to_qubo_to_ising():
    
    ising_args = h, J, offset = {0: 1, 2: -2}, {(0, 1): -4, (0, 2): 3}, -2
    
    assert (h, J, offset) == qubo_to_ising(*ising_to_qubo(*ising_args))
