"""
Here we have the QUBO/Ising conversions for common NP-Complete problems,
including Karp's 21 NP-Complete problems. The conversions are based on
[Andrew Lucas. Ising formulations of many np problems. Frontiers in Physics, 
2:5, 2014.]
"""

name = "karp"

from ._set_cover import SetCover
from ._number_partitioning import NumberPartitioning

__all__ = (
    "SetCover",
    "NumberPartitioning",
)
