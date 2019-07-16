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
        for key, value in items: self[key] = value
    
    def __getitem__(self, key):
        """
        Overrides the dict.__getitem__ command so that a KeyError is not
        thrown if `key` is not in the dictionary.
        
        key: tuple of two integers, element of the dictionary.
        
        returns: the value corresponding to the key if the key is in the 
            dictionary, otherwise returns 0.
        """
        return self.get(key, 0)
    
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
        if not isinstance(key, tuple) or not len(key) == 2:
            raise KeyError(
                "Key formatted incorrectly, must be tuple of two integers")
            
        k = tuple(sorted(key))
        if value: super().__setitem__(k, value)
        else: self.pop(k, 0)
