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

"""``covering`` contains many NP covering problems.

Here we have the QUBO/QUSO conversions for common covering problems.
The conversions are based on [Lucas].

References
----------
.. [Lucas] Andrew Lucas. Ising formulations of many np problems. Frontiers in
    Physics, 2:5, 2014.

"""

from ._set_cover import *
from ._vertex_cover import *

from ._set_cover import __all__ as __all_sc__
from ._vertex_cover import __all__ as __all_vc__

__all__ = (
    __all_sc__ +
    __all_vc__
)

del __all_sc__, __all_vc__

name = "covering"
