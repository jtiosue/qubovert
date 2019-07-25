"""
This module contains many NP problems. We organize all the problems into
categories. But we import them globally for user use. See the `__all__` value
for all the problems imported.
"""

from .bilp import *
from .coloring import *
from .covering import *
from .cycles import *
from .packing import *
from .partitioning import *
from .tree import *

from .bilp import __all__ as __all_bilp__
from .coloring import __all__ as __all_coloring__
from .covering import __all__ as __all_covering__
from .cycles import __all__ as __all_cycles__
from .packing import __all__ as __all_packing__
from .partitioning import __all__ as __all_partitioning__
from .tree import __all__ as __all_tree__

__all__ = (
    __all_bilp__ + __all_coloring__ + __all_covering__ + __all_cycles__ +
    __all_packing__ + __all_partitioning__ + __all_tree__
)

__str__ = "np"
