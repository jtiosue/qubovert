"""
Here we have the QUBO/Ising conversions for common covering problems. 
The conversions are based on
[Andrew Lucas. Ising formulations of many np problems. Frontiers in Physics, 
2:5, 2014.]
"""

name = __str__ = "covering"

from ._set_cover import SetCover

__all__ = (
    "SetCover",
)
