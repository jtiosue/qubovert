"""
QUBOConvert is a module for converting common problems into QUBO  and Ising 
form, where QUBO stands for Quadratic Unconstrained Binary Optimization. QUBO 
problems have a one-to-one mapping to classical Ising problems, and most 
optimization problems are formatted in QUBO form when the solver is a quantum 
computer. See QUBOConvert.__all__ for all the problems defined, and 
QUBOConvert.utils.__all__ for some utilities used.
"""


name = "QUBOConvert"

from ._set_cover import SetCover

# if "from QUBOConvert import *" is called, don't import utils
__all__ = (
    "SetCover",
)

from . import utils
