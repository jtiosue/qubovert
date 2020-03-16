#   Copyright 2020 Joseph T. Iosue
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""``np`` contains many NP problems.

This module contains many NP problems. We organize all the problems into
categories. But we import them globally for user use. See the ``__all__`` value
for all the problems imported. Most conversions to QUBO/QUSO form are based
on [Lucas]_.

References
----------
.. [Lucas] Andrew Lucas. Ising formulations of many np problems. Frontiers in
   Physics, 2:5, 2014.

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

del __all_bilp__, __all_coloring__, __all_covering__, __all_cycles__
del __all_packing__, __all_partitioning__, __all_tree__

name = "np"
