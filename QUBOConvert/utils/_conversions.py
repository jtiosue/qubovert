from ._qubo_matrix import QUBOMatrix, IsingCoupling, IsingField


def qubo_to_ising(Q, offset=0):
    """
    Convert the specified QUBO problem into an Ising problem.
    
    Q: dictionary mapping binary variables indices to the Q value. Note that
        binary variable indices must be integer labeled starting from 0. Q can
        also be a QUBOConvert.utils.QUBOMatrix object.
    offset: an optional float, the part of the objective function that does 
        not depend on the variables.    
        
    returns the tuple (h, J, offset).
        h represents the field of each spin in the Ising formulation.
            h is a IsingField object. For most practical purposes, you can
            use IsingField in he same way as an ordinary dictionary. For
            more information, see help(QUBOConver.utils.IsingField).
        J is the upper triangular coupling matrix, a IsingCoupling object.
            For most practical purposes, you can use IsingCoupling in the 
            same way as an ordinary dictionary. For more information,
            see help(QUBOConvert.utils.IsingCoupling).
        offset is a float. It is the sum of the terms in the formulation in
            the cited paper that don't involve any variables.
    """
    # IsingCoupling deals with keeping J upper triangular, so we don't have to
    # worry about it!
    h, J = IsingField(), IsingCoupling()
    
    for (i, j), v in Q.items():
        if i != j:
            J[(i, j)] += v / 4
            h[i] += v / 4
            h[j] += v / 4
            offset += v / 4
        else:
            h[i] += v / 2
            offset += v / 2
            
    return h, J, offset
    
    
def ising_to_qubo(h, J, offset=0):
    """
    Convert the specified Ising problem into an upper triangular QUBO problem.
    
    h: dictionary mapping spins indices to the field value. Note that
        spin variable indices must be integer labeled starting from 0. h can
        also be a QUBOConvert.utils.IsingField object.
    J: dictionary mapping tuples of spin indices to the coupling value. Note 
        that spin variable indices must be integer labeled starting from 0.
        Also note that J cannot have a key that has a repeated index, ie
        (1, 1) is an invalid key. J can also be a 
        QUBOConvert.utils.IsingCoupling object.
    offset: an optional float, the part of the objective function that does 
        not depend on the variables.
        
    returns the tuple (Q, offset).
        Q is the upper triangular QUBO matrix, a QUBOMatrix object.
            For most practical purposes, you can use QUBOMatrix in the 
            same way as an ordinary dictionary. For more information,
            see help(QUBOConvert.utils.QUBOMatrix).
        offset is a float. It is the sum of the terms in the formulation in
            the cited paper that don't involve any variables.
    """
    # QUBOMarix deals with keeping ! upper triangular, so we don't have to
    # worry about it!
    Q = QUBOMatrix()
    
    for (i, j), v in J.items():
        if i == j:
            raise KeyError("J formatted incorrectly, key cannot "
                           "have repeated indices")
        Q[(i, j)] += 4 * v
        Q[(i, i)] -= 2 * v
        Q[(j, j)] -= 2 * v
        offset += v
    
    for i, v in h.items():
        Q[(i, i)] += 2 * v
        offset -= v
        
    return Q, offset
