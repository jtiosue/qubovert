"""
Here we have the QUBO/Ising conversions for common partitioning probems,
The conversions are based on
[Andrew Lucas. Ising formulations of many np problems. Frontiers in Physics,
2:5, 2014.]
"""

from ._number_partitioning import NumberPartitioning

__all__ = (
    "NumberPartitioning",
)

__str__ = "partitioning"
