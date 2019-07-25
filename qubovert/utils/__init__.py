"""
Helper module. Contains necessary or convenient methods.
"""

from ._problem_parentclass import Problem
from ._qubo_matrix import QUBOMatrix, IsingCoupling, IsingField
from ._conversions import (
    qubo_to_ising, ising_to_qubo, binary_to_spin, spin_to_binary
)
from ._solve_bruteforce import (
    solve_qubo_bruteforce, solve_ising_bruteforce, qubo_value, ising_value
)


__all__ = (
    "Problem",
    "QUBOMatrix",
    "IsingCoupling",
    "IsingField",
    "qubo_to_ising",
    "ising_to_qubo",
    "binary_to_spin",
    "spin_to_binary",
    "solve_qubo_bruteforce",
    "solve_ising_bruteforce",
    "qubo_value",
    "ising_value",
)


__str__ = "utils"
