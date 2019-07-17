from qubovert.utils import QUBOMatrix, IsingCoupling, IsingField


## QUBO 


def test_qubo_default_valid():
    
    d = QUBOMatrix()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0, 0): 1}
        
    
def test_qubo_remove_value_when_zero():
    
    d = QUBOMatrix()
    d[(0, 0)] += 1
    d[(0, 0)] -= 1
    assert d == {}
        

def test_qubo_reinitialize_dictionary():
    
    d = QUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0, (0, 1): 1})
    assert d == {(0, 0): 1, (0, 1): 3}


## Ising


def test_ising_default_valid():
    
    J = IsingCoupling()
    assert J[(0, 1)] == 0
    J[(0, 1)] += 1
    assert J == {(0, 1): 1}
    
    h = IsingField()
    assert h[1] == 0
    h[1] += 1
    assert h == {1: 1}
        
    
def test_ising_remove_value_when_zero():
    
    J = IsingCoupling()
    J[(1, 2)] += 1
    J[(1, 2)] -= 1
    assert J == {}
    
    h = IsingField()
    h[0] += 1
    h[0] -= 1
    assert h == {}
        

def test_ising_reinitialize_dictionary():
    
    J = IsingCoupling({(0, 1): 1, (1, 0): 2, (2, 0): 0})
    assert J == {(0, 1): 3}
    
    h = IsingField({0: -2, 2: 0})
    assert h == {0: -2}
