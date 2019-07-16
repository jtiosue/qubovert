from QUBOConvert.utils import QUBOMatrix


def test_default_valid():
    
    d = QUBOMatrix()
    assert d[(0, 0)] == 0
    d[(0, 0)] += 1
    assert d == {(0, 0): 1}
        
    
def test_remove_value_when_zero():
    
    d = QUBOMatrix()
    d[(0, 0)] += 1
    d[(0, 0)] -= 1
    assert d == {}
        

def test_reinitialize_dictionary():
    
    d = QUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0})
    assert d == {(0, 0): 1, (0, 1): 2}
