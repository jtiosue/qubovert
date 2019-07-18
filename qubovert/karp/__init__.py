"""
Here we have the QUBO/Ising conversions for Karp's 21 NP-Complete problems.
"""

name = "karp"

from ._set_cover import SetCover

__all__ = (
    "SetCover",
)
