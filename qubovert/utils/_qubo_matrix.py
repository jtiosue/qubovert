class QUBOMatrix(dict):
    """
    A class to handle QUBO matrices. It is the same thing as a dictionary
    with some methods modified. Note that each key must be a tuple of two
    integers >= 0.
    
    One method is that values will always default to 0. Consider the following
    example:
        >>> d = QUBOMatrix()
        >>> print(d[(0, 0)]) # will print 0
        >>> d[(0, 0)] += 1
        >>> print(d) # will print {(0, 0): 1}
        
        >>> g = dict()
        >>> print(g[(0, 0)]) # will raise KeyError
        >>> g[(0, 0)] += 1 # will raise KeyError, since (0, 0) was never set
    
    One method of QUBOMatrix is that it will always keep the QUBO 
    upper triangular! Consider the following example:
        >>> d = QUBOMatrix()
        >>> d[(1, 0)] += 2
        >>> print(d)
        >>> # will print {(0, 1): 2}
        
    One method is that if we set an item to 0, it will be removed. Consider
    the following example:
        >>> d = QUBOMatrix()
        >>> d[(0, 0)] += 1
        >>> d[(0, 0)] -= 1
        >>> print(d) # will print {}
        
    One method is that if we initialize QUBOMatrix with a previous dictionary
    it will be reinitialized to ensure that the QUBOMatrix is upper
    triangular and contains no zero values. Consider the following example:
        >>> d = QUBOMatrix({(0, 0): 1, (1, 0): 2, (2, 0): 0})
        >>> print(d) # will print {(0, 0): 1, (0, 1): 2}
        
    Finally, if you try to access a key out of order, it will sort the key.
    For example,
        >>> d = QUBOMatrix()
        >>> d[(0, 1)] += 2
        >>> print(d[(1, 0)])  # will print 2
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a QUBOMatrix object. If you supply args and kwargs that 
        represent a dictionary, they will be reinitialized to ensure that
        the QUBOMatrix is upper triangular and contains no zero values.
        
        *args and **kwargs, see the docstring for dict.
        """
        super().__init__(*args, **kwargs)
        
        # reset to make sure everything is in the proper form
        items = tuple(self.items())
        self.clear()
        for key, value in items: self[key] += value
    
    def __getitem__(self, key):
        """
        Overrides the dict.__getitem__ command so that a KeyError is not
        thrown if `key` is not in the dictionary. Also sorts the key. So 
        if we you try to access the key (1, 0), it will return the value for
        the key (0, 1).
        
        key: tuple of two integers, element of the dictionary.
        
        returns: the value corresponding to the key if the key is in the 
            dictionary, otherwise returns 0.
        """
        try: k = tuple(sorted(key))
        except TypeError: k = key
        
        return self.get(k, 0)
    
    def __setitem__(self, key, value):
        """
        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        QUBOMatrix dictionary will ever have zero value. Additionally, this
        method will keep the QUBO upper triangular, so if key[0] > key[1],
        then we will call __setitem__((key[1], key[0]), value).
        
        key: tuple of two integers, element of the dictionary.
        value: int or float, value corresponding to the key.
        
        returns: None.
        """
        if (not isinstance(key, tuple) or not len(key) == 2 or
            not isinstance(key[0], int) or not isinstance(key[1], int) or
            key[0] < 0 or key[1] < 0):
            
            raise KeyError(
                "Key formatted incorrectly, must be tuple of two integers")
            
        k = tuple(sorted(key))
        if value: super().__setitem__(k, value)
        else: self.pop(k, 0)


class IsingCoupling(QUBOMatrix):
    """
    A class to handle the J coupling Ising matrices. It is the same thing as a 
    dictionary with some methods modified. Note that each key must be a tuple 
    of two integers >= 0. Note that this is almost exactly the same as 
    QUBOMatrix, except that the keys cannot be tuples of the same index. For
    example, QUBOMatrix({(0, 0): 1}) is valid but IsingCoupling({(0, 0): 1})
    is invalid.
    
    One method is that values will always default to 0. Consider the following
    example:
        >>> d = IsingCoupling()
        >>> print(d[(0, 1)]) # will print 0
        >>> d[(0, 1)] += 1
        >>> print(d) # will print {(0, 1): 1}
        
        # compared to an ordinary dict
        >>> g = dict()
        >>> print(g[(0, 1)]) # will raise KeyError
        >>> g[(0, 1)] += 1 # will raise KeyError, since (0, 0) was never set
    
    One method of IsingCoupling is that it will always keep the coupling 
    upper triangular! Consider the following example:
        >>> d = IsingCoupling()
        >>> d[(1, 0)] += 2
        >>> print(d)
        >>> # will print {(0, 1): 2}
        
    One method is that if we set an item to 0, it will be removed. Consider
    the following example:
        >>> d = IsingCoupling()
        >>> d[(0, 1)] += 1
        >>> d[(0, 1)] -= 1
        >>> print(d) # will print {}
        
    One method is that if we initialize IsingCoupling with a previous 
    dictionary, it will be reinitialized to ensure that the IsingCoupling is 
    upper triangular and contains no zero values. Consider the following 
    example:
        >>> d = IsingCoupling({(0, 1): 1, (1, 0): 2, (2, 0): 0})
        >>> print(d) # will print {(0, 1): 3}
        
    Finally, if you try to access a key out of order, it will sort the key.
    For example,
        >>> d = IsingCoupling()
        >>> d[(0, 1)] += 2
        >>> print(d[(1, 0)])  # will print 2
    """
    
    def __setitem__(self, key, value):
        """
        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        IsingCoupling dictionary will ever have zero value. Additionally, this
        method will keep the coupling upper triangular, so if key[0] > key[1],
        then we will call __setitem__((key[1], key[0]), value). Finally, 
        key[0] cannot equal key[1], if so a KeyError will be raised.
        
        key: tuple of two different integers, element of the dictionary.
        value: int or float, value corresponding to the key.
        
        returns: None.
        """
        if not isinstance(key, tuple) or key[0] == key[1]:
            raise KeyError(
                "Key formatted incorrectly, "
                "must be tuple of two different integers")
            
        super().__setitem__(key, value)
    
    
class IsingField(QUBOMatrix):
    """
    A class to handle the h field Ising matrices. It is the same thing as a 
    dictionary with some methods modified. Note that each key must be an 
    an integer >= 0. Note that this is almost exactly the same as 
    QUBOMatrix, except that the keys are integers instead of tuples.
    
    One method is that values will always default to 0. Consider the following
    example:
        >>> d = IsingField()
        >>> print(d[0]) # will print 0
        >>> d[0] += 1
        >>> print(d) # will print {0: 1}
        
        # compared to an ordinary dict
        >>> g = dict()
        >>> print(g[0]) # will raise KeyError
        >>> g[0] += 1 # will raise KeyError, since (0, 0) was never set
        
    One method is that if we set an item to 0, it will be removed. Consider
    the following example:
        >>> d = IsingField()
        >>> d[1] += 1
        >>> d[1] -= 1
        >>> print(d) # will print {}
        
    One method is that if we initialize IsingField with a previous 
    dictionary, it will be reinitialized to ensure that the IsingField is 
    contains no zero values and is all valid. Consider the following 
    example:
        >>> d = IsingField({0: 1, 1: 2, 2: 0})
        >>> print(d) # will print {0: 1, 1: 2}
    """
    
    def __setitem__(self, key, value):
        """
        Overrides the dict.__setitem__ command. If `value` is equal to 0, then
        the key will be removed from the dictionary. Thus no elements in the
        IsingCoupling dictionary will ever have zero value. Additionally, this
        method will keep the coupling upper triangular, so if key[0] > key[1],
        then we will call __setitem__((key[1], key[0]), value). Finally, 
        key[0] cannot equal key[1], if so a KeyError will be raised.
        
        key: tuple of two different integers, element of the dictionary.
        value: int or float, value corresponding to the key.
        
        returns: None.
        """
        if not isinstance(key, int) or key < 0:
            raise KeyError(
                "Key formatted incorrectly, must be a positive integer")
            
        if value: dict.__setitem__(self, key, value)
        else: self.pop(key, 0)
