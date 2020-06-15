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

"""``sim`` contains ``qubovert``'s simulation and annealing functionality.

See ``__all__`` for a list of uses.

"""

# import order here is important
from ._anneal_temperature_range import *
from ._anneal_results import *
from ._anneal import *

from ._anneal_temperature_range import __all__ as __all_tr__
from ._anneal_results import __all__ as __all_results__
from ._anneal import __all__ as __all_anneal__


__all__ = __all_tr__ + __all_results__ + __all_anneal__

del __all_tr__, __all_results__, __all_anneal__


name = "sim"
